from langgraph.graph import StateGraph, END
from src.domain.models import OrchestratorState
from src.agents.nodes import (
    run_de_crew_node,
    run_executor_node,
    run_memory_updater_node,
    run_mle_crew_node,
    run_ml_executor_node,
)


def evaluate_executor_edge(state: OrchestratorState) -> str:
    """Routes execution based on Node 2 executor sandbox outcome."""
    if state.get("status") == "SUCCESS":
        return "memory_updater"
    print(
        f"[Graph Routing] Sandbox state was '{state.get('status')}'. Halting flow.", flush=True)
    return END


def build_phoenix_runtime(checkpointer=None):
    workflow = StateGraph(OrchestratorState)

    # Add all 5 nodes
    workflow.add_node("de_crew", run_de_crew_node)
    workflow.add_node("executor", run_executor_node)
    workflow.add_node("memory_updater", run_memory_updater_node)
    workflow.add_node("mle_crew", run_mle_crew_node)
    workflow.add_node("ml_executor", run_ml_executor_node)

    # Set Entry Point
    workflow.set_entry_point("de_crew")

    # Sequential edges
    workflow.add_edge("de_crew", "executor")

    workflow.add_conditional_edges(
        "executor",
        evaluate_executor_edge,
        {
            "memory_updater": "memory_updater",
            END: END
        }
    )

    workflow.add_edge("memory_updater", "mle_crew")
    workflow.add_edge("mle_crew", "ml_executor")
    workflow.add_edge("ml_executor", END)

    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()
