from langgraph.graph import StateGraph, END
import logfire
from src.auditor_watcher import DRIFT_THRESHOLD_PSI
from src.domain.models import OrchestratorState
from src.agents.nodes import (
    run_de_crew_node, run_executor_node,
    run_memory_updater_node, run_mle_crew_node,
    run_ml_executor_node, run_auditor_node 
)


def route_entry(state: OrchestratorState):
    """Ends the run before any model call when telemetry reports PSI under the drift threshold."""
    psi = (state.get("metrics") or {}).get("psi_score")
    if psi is not None and float(psi) <= DRIFT_THRESHOLD_PSI:
        logfire.info(f"🟢 PSI {psi} is under {DRIFT_THRESHOLD_PSI}. No drift, nothing to heal.")
        return "nominal_exit"
    return "de_crew"


async def run_nominal_exit(state: OrchestratorState):
    return {"status": "NOMINAL"}


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
    workflow.add_node("nominal_exit", run_nominal_exit)

    # Free-tier budget: a healthy window costs zero model requests.
    workflow.set_conditional_entry_point(route_entry, ["de_crew", "nominal_exit"])
    workflow.add_edge("nominal_exit", END)

    workflow.add_edge("de_crew", "executor")
    workflow.add_edge("memory_updater", "mle_crew")
    workflow.add_edge("mle_crew", "ml_executor")
    workflow.add_edge("auditor_node", END)

    workflow.add_conditional_edges("executor", route_de_execution)
    workflow.add_conditional_edges("ml_executor", route_mle_execution)

    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()