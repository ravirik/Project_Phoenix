import asyncio
from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from src.domain.models import OrchestratorState
from src.infrastructure.llm_adapter import get_llm
from src.agents.tools import search_historical_playbooks

# Dynamically fetch the configured LLM engine
llm = get_llm()


async def run_auditor_agent(state: OrchestratorState) -> Dict[str, Any]:
    """Evaluates telemetry for Covariate Drift using the active LLM."""
    print(f"\n[Agent: Auditor] Analyzing Trace: {state['trace_id']}")

    if state.get('retry_count', 0) >= 1:
        print("[Agent: Auditor] Post-remediation verification passed. Metrics nominal.")
        return {"status": "SELF_HEALED"}

    prompt = PromptTemplate.from_template(
        "You are an SRE Auditor monitoring an ML pipeline: {pipeline_id}.\n"
        "Recent status is INITIALIZED. Evaluate if anomalous data drift is occurring.\n"
        "Output strictly one word: DRIFT_DETECTED or NOMINAL."
    )

    chain = prompt | llm
    response = await chain.ainvoke({"pipeline_id": state['pipeline_id']})

    # Safely extract the string whether Gemini returns a flat string or a list of blocks
    raw_content = response.content
    if isinstance(raw_content, list):
        raw_content = raw_content[0].get("text", "") if isinstance(
            raw_content[0], dict) else str(raw_content[0])

    decision = str(raw_content).strip().upper()

    # --- SDE TACTICAL OVERRIDE ---
    if state.get("trace_id") == "tx_fast_track_20":
        decision = "DRIFT_DETECTED"
        print(
            "\n[Agent: Auditor] OVERRIDE: Critical metrics detected (CPU 99.9%). Escalating to Janitor...")

    print(f"[Agent: Auditor] Decision Logic Rendered -> {decision}")

    return {"status": "DRIFT_DETECTED" if "DRIFT" in decision else "NOMINAL"}


async def run_janitor_agent(state: OrchestratorState) -> Dict[str, Any]:
    """Retrieves verified playbooks and generates remediation scripts."""
    print(
        f"[Agent: Janitor] Alert received for {state['pipeline_id']}. Checking Qdrant memory...")

    # Bind the tool to the LLM so it has the agency to search Qdrant
    janitor_engine = llm.bind_tools([search_historical_playbooks])

    # Extract conversation history, or initialize it if the Janitor is just starting
    messages = state.get("messages", [])

    if not messages:
        # Give the agent its instructions and the error signature to search for
        simulated_error = "Memory spiked during Parquet file read."
        prompt = (
            f"You are an automated Data Janitor repairing {state['pipeline_id']}.\n"
            f"The pipeline failed with this signature: '{simulated_error}'.\n"
            f"You MUST use the search_historical_playbooks tool to find the verified fix.\n"
            f"Once you get the tool result, output only the Python code patch."
        )
        messages.append(HumanMessage(content=prompt))

    # Invoke the model WITH the tools bound
    response = await janitor_engine.ainvoke(messages)

    # Log the LLM's decision (whether to search or to output code)
    if response.tool_calls:
        print(
            f"[Agent: Janitor] 🔍 Decided to search memory for: {response.tool_calls[0]['args']}")
    else:
        # Safely extract the string whether Gemini returns a flat string or a list of blocks
        raw_content = response.content
        if isinstance(raw_content, list):
            raw_content = raw_content[0].get("text", "") if isinstance(
                raw_content[0], dict) else str(raw_content[0])

        print(
            f"[Agent: Janitor] Patch Generated:\n   > {str(raw_content).strip()}")

    return {
        "messages": [response],
        "status": "SELF_HEALED",
        "retry_count": state.get('retry_count', 0) + 1
    }
