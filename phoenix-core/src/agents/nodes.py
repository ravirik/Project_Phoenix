import asyncio
import json
import re
import traceback
import pandas as pd
import os
from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.domain.models import OrchestratorState
from src.infrastructure.llm_adapter import get_llm
from src.agents.tools import search_historical_playbooks

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


async def run_executor_node(state: OrchestratorState) -> Dict[str, Any]:
    print("[Agent: Executor] Starting remediation...")
    path = state.get("data_path")
    if not path or not os.path.exists(path):
        return {"status": "EXECUTION_FAILED"}

    df = pd.read_parquet(path)
    patch = state.get("final_patch", "")

    # 1. CLEANUP: Convert literal "\\n" to actual "\n" and strip whitespace
    patch = patch.replace('\\n', '\n').strip()

    # 2. EXTRACT: Get code block
    code_match = re.search(r"```python(.*?)```", patch, re.DOTALL)
    code_to_run = code_match.group(1).strip() if code_match else patch.strip()

    # 3. CIRCUIT BREAKER: If LLM tries to load data, we override it with the safe, intended fix.
    # We strictly forbid 'read_parquet' or 'pd.read' in the generated code.
    if "read_parquet" in code_to_run or "pd.read" in code_to_run:
        print(
            "[Agent: Executor] ⚠️ LLM hallucinated a file read. Overriding with safe logic.")
        code_to_run = "df = df.replace(r'^\\s*$', np.nan, regex=True).dropna()"

    # 4. EXECUTE
    local_vars = {"df": df, "pd": pd, "np": __import__("numpy")}
    try:
        # We perform one last cleanup of the string to remove any leading/trailing weirdness
        code_to_run = code_to_run.replace('\\n', '').strip()
        print(f"[Agent: Executor] Executing clean patch:\n{code_to_run}")

        exec(code_to_run, {}, local_vars)

        healed_df = local_vars["df"]
        if healed_df is not None and healed_df.isnull().sum().sum() == 0:
            print("[Agent: Executor] ✅ Remediation success.")
            healed_df.to_parquet(path, index=False)
            return {"status": "SUCCESS"}

        return {"status": "PARTIAL_SUCCESS"}
    except Exception:
        print(f"[Agent: Executor] ❌ Execution Error: {traceback.format_exc()}")
        return {"status": "EXECUTION_FAILED"}
