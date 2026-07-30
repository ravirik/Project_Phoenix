from langgraph.graph import StateGraph, END
import logfire
from src.domain.models import OrchestratorState
from src.agents.nodes import (
    run_de_crew_node, run_executor_node,
    run_memory_updater_node, run_mle_crew_node,
    run_ml_executor_node, run_auditor_node 
)


def route_de_execution(state: OrchestratorState):
    status = state.get("status")
    retries = state.get("de_retry_count", 0)

    if status == "SUCCESS":
        return "memory_updater"
    elif status == "DE_FAILED" and retries < 3:
        logfire.warn(f"🔁 DE Sandbox failed. Triggering Retry {retries}/3.")
        return "de_crew"
    else:
        logfire.error(f"❌ DE Pipeline permanently failed after {retries} retries. Routing to Auditor for DLQ.")
        return "auditor_node"


def route_mle_execution(state: OrchestratorState):
    status = state.get("status")
    retries = state.get("mle_retry_count", 0)
    escalations = state.get("de_escalation_count", 0)

    if status == "ML_EXECUTION_COMPLETED":
        logfire.info("🏆 MLOps Training Complete. Forwarding to Governance Auditor...")
        return "auditor_node" 
    elif status == "MLE_FAILED":
        if retries < 3:
            logfire.warn(f"🔁 MLE Sandbox failed. Triggering Retry {retries}/3.")
            return "mle_crew"
        elif escalations <= 1:  # 🛡️ THE FIX: Allow 1 escalation back to DE Crew
            logfire.warn("⚠️ MLE retries exhausted. Escalating back to Janitor Crew for data re-patching...")
            return "de_crew"
        else:
            logfire.error("❌ MLE Pipeline permanently failed after retries and escalation. Routing to Auditor for DLQ.")
            return "auditor_node" 
    return "auditor_node"


def build_phoenix_runtime(checkpointer=None):
    workflow = StateGraph(OrchestratorState)

    workflow.add_node("de_crew", run_de_crew_node)
    workflow.add_node("executor", run_executor_node)
    workflow.add_node("memory_updater", run_memory_updater_node)
    workflow.add_node("mle_crew", run_mle_crew_node)
    workflow.add_node("ml_executor", run_ml_executor_node)
    workflow.add_node("auditor_node", run_auditor_node)

    workflow.set_entry_point("de_crew")

    workflow.add_edge("de_crew", "executor")
    workflow.add_edge("memory_updater", "mle_crew")
    workflow.add_edge("mle_crew", "ml_executor")
    workflow.add_edge("auditor_node", END)

    workflow.add_conditional_edges("executor", route_de_execution)
    workflow.add_conditional_edges("ml_executor", route_mle_execution)

    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()