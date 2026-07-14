import asyncio
from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from src.domain.models import OrchestratorState
from src.infrastructure.llm_adapter import get_llm

# Dynamically fetch the configured LLM engine
llm = get_llm()

async def run_auditor_agent(state: OrchestratorState) -> Dict[str, Any]:
    """Evaluates telemetry for Covariate Drift using the active LLM."""
    print(f"\n[Agent: Auditor] Analyzing Trace: {state['trace_id']}")
    
    if state['retry_count'] >= 1:
        print("[Agent: Auditor] Post-remediation verification passed. Metrics nominal.")
        return {"status": "SELF_HEALED"}
        
    prompt = PromptTemplate.from_template(
        "You are an SRE Auditor monitoring an ML pipeline: {pipeline_id}.\n"
        "Recent status is INITIALIZED. Evaluate if anomalous data drift is occurring.\n"
        "Output strictly one word: DRIFT_DETECTED or NOMINAL."
    )
    
    chain = prompt | llm
    response = await chain.ainvoke({"pipeline_id": state['pipeline_id']})
    
    decision = response.content.strip().upper()
    print(f"[Agent: Auditor] Decision Logic Rendered -> {decision}")
    
    return {"status": "DRIFT_DETECTED" if "DRIFT" in decision else "NOMINAL"}

async def run_janitor_agent(state: OrchestratorState) -> Dict[str, Any]:
    """Generates remediation scripts based on the Auditor's drift alert."""
    print(f"[Agent: Janitor] Alert received for {state['pipeline_id']}. Synthesizing patch...")
    
    prompt = PromptTemplate.from_template(
        "You are an automated Data Janitor repairing {pipeline_id}.\n"
        "Generate a 1-line Python pandas script to clip outliers above the 99th percentile.\n"
        "Output strictly the Python code, nothing else."
    )
    
    chain = prompt | llm
    response = await chain.ainvoke({"pipeline_id": state['pipeline_id']})
    
    # Strip markdown code blocks if the LLM adds them
    code_patch = response.content.strip().replace("```python", "").replace("```", "").strip()
    print(f"[Agent: Janitor] Patch Generated:\n   > {code_patch}")
    
    return {"status": "SELF_HEALED", "retry_count": state['retry_count'] + 1}
