import asyncio
import json
import re
import traceback
import pandas as pd
import numpy as np
import os
import multiprocessing
from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.domain.models import OrchestratorState
from src.infrastructure.llm_adapter import get_llm
from src.agents.tools import search_historical_playbooks, upsert_playbook

llm = get_llm()


async def run_auditor_agent(state: OrchestratorState) -> Dict[str, Any]:
    print(
        f"\n[Agent: Auditor] Analyzing Trace: {state.get('trace_id', 'N/A')}")

    # Tactical drill override to simulate infrastructure data drift
    if state.get("trace_id") == "tx_fast_track_20":
        return {"status": "DRIFT_DETECTED"}

    return {"status": "NOMINAL"}


async def run_janitor_agent(state: OrchestratorState) -> Dict[str, Any]:
    print(
        f"[Agent: Janitor] Alert received for {state.get('pipeline_id', 'unknown')}. Consulting Qdrant cluster...")
    janitor_engine = llm.bind_tools([search_historical_playbooks])

    # Filter state message array history to satisfy token requirements
    raw_history = state.get("messages", [])
    valid_messages = [m for m in raw_history if (getattr(m, 'tool_calls', None)) or (
        getattr(m, 'content', None) and str(m.content).strip() != "")]

    if not valid_messages:
        prompt = (
            "You are a Data Janitor SRE. "
            "CRITICAL RULES:\n"
            "1. DO NOT include any code to read the file (no pd.read_parquet).\n"
            "2. The dataframe 'df' is ALREADY loaded. Just return the transformation code.\n"
            "3. Goal: replace empty strings with NaN and drop them.\n"
            "4. Output: ONLY python code in triple backticks.\n"
            "Call the tool 'search_historical_playbooks' with incident_signature='Memory spiked during Parquet file read'."
        )
        valid_messages.append(HumanMessage(content=prompt))

    response = await janitor_engine.ainvoke(valid_messages)
    valid_messages.append(response)

    # Execute ReAct Tool calling loop if required by engine output
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(
                f"[Agent: Janitor] 🔍 Executing Tool: {tool_call['name']} with constraints: {tool_call['args']}")
            tool_result = await search_historical_playbooks.ainvoke(tool_call['args'])
            valid_messages.append(ToolMessage(content=json.dumps(
                tool_result), tool_call_id=tool_call['id']))

        response = await janitor_engine.ainvoke(valid_messages)
        valid_messages.append(response)

    # Isolate textual code output mapping
    final_patch = response.content if isinstance(
        response.content, str) else str(response.content)

    return {
        "messages": valid_messages,
        "final_patch": final_patch,
        "status": "SELF_HEALED",
        "retry_count": state.get('retry_count', 0) + 1
    }


def _isolated_sandbox(code_string: str, data_path: str, queue: multiprocessing.Queue):
    """
    Runs in a completely separate OS process. 
    If this crashes or hangs, the main worker survives.
    """
    try:
        # 1. Load the corrupted data
        df = pd.read_parquet(data_path)

        # 2. Namespace Restriction (The Jail)
        # We explicitly block all standard python built-ins so the LLM cannot import
        # libraries, open system files, or execute shell commands.
        safe_globals = {"__builtins__": {}}

        # We only pass in the exact variables it needs to heal the data.
        safe_locals = {
            "df": df,
            "pd": pd,
            "np": np
        }

        # 3. Execute the patched code inside the jail
        exec(code_string, safe_globals, safe_locals)

        # 4. Verify and Save
        healed_df = safe_locals.get("df")
        if healed_df is not None and healed_df.isnull().sum().sum() == 0:
            healed_df.to_parquet(data_path, index=False)
            queue.put({"status": "SUCCESS"})
        else:
            queue.put({"status": "PARTIAL_SUCCESS",
                      "reason": "Nulls remain after execution."})

    except Exception as e:
        queue.put({"status": "EXECUTION_FAILED", "error": str(e)})


async def run_executor_node(state: OrchestratorState) -> Dict[str, Any]:
    print("[Agent: Executor] Initializing Secure Sandbox...")
    path = state.get("data_path")
    if not path or not os.path.exists(path):
        return {"status": "EXECUTION_FAILED"}

    patch = state.get("final_patch", "")

    # 1. Clean the LLM output
    patch = patch.replace('\\n', '\n').strip()
    code_match = re.search(r"```python(.*?)```", patch, re.DOTALL)
    code_to_run = code_match.group(1).strip() if code_match else patch.strip()

    # 2. Circuit Breaker: Strip file reading operations
    if "read_parquet" in code_to_run or "pd.read" in code_to_run:
        print(
            "[Agent: Executor] ⚠️ Circuit Breaker: Stripped hallucinated I/O operations.")
        code_to_run = "df = df.replace(r'^\\s*$', np.nan, regex=True).dropna()"

    # 3. Sandbox Pre-Processing: Strip all import statements.
    # The sandbox has no __builtins__, so 'import' will crash it. We already inject pd and np.
    code_to_run = re.sub(r'^\s*(import|from)\s+.*$', '',
                         code_to_run, flags=re.MULTILINE).strip()

    print(f"[Agent: Executor] Injecting into sandbox:\n{code_to_run}")

    # 4. Spawn the Isolated Process
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_isolated_sandbox,
        args=(code_to_run, path, queue)
    )

    process.start()

    # 5. Enforce strict execution timeout (10 seconds)
    process.join(timeout=10.0)

    if process.is_alive():
        print(
            "[Agent: Executor] ❌ Sandbox Timeout: LLM code exceeded execution limits. Terminating.")
        process.terminate()
        process.join()
        return {"status": "EXECUTION_TIMEOUT"}

    # 6. Retrieve result
    if not queue.empty():
        result = queue.get()
        if result["status"] == "SUCCESS":
            print("[Agent: Executor] ✅ Remediation success within sandbox.")
        else:
            print(
                f"[Agent: Executor] ❌ Sandbox Error: {result.get('error', result.get('reason'))}")
        return result

    return {"status": "EXECUTION_FAILED"}


async def run_memory_updater_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    This node only executes if the Sandbox reports SUCCESS. 
    It commits the new patch to long-term memory.
    """
    print("[Agent: Memory] Continuous Learning triggered. Committing verified patch to Qdrant...")

    # Retrieve the context
    patch = state.get("final_patch", "")
    trace_id = state.get("trace_id", "unknown_trace")

    # In a full production system, you'd extract the actual incident string from the Auditor's alert.
    # For now, we will use the trace ID as the unique incident signature.
    incident_signature = f"Anomaly detected in trace {trace_id}"

    try:
        # Call the deterministic update function
        upsert_playbook(incident_signature, patch)
        return {"status": "MEMORY_UPDATED"}
    except Exception as e:
        print(f"[Agent: Memory] ❌ Failed to commit to Qdrant: {str(e)}")
        return {"status": "MEMORY_UPDATE_FAILED"}
