from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from src.domain.models import OrchestratorState
from src.agents.nodes import run_auditor_agent, run_janitor_agent, run_executor_node, run_memory_updater_node
from src.agents.tools import search_historical_playbooks


def evaluate_auditor_edge(state: OrchestratorState):
    return "to_janitor" if state["status"] == "DRIFT_DETECTED" else "to_end"


def evaluate_executor_edge(state: OrchestratorState):
    """Routes to Memory Update only if the patch successfully healed the data."""
    if state["status"] == "SUCCESS":
        return "to_memory"
    return "to_end"


def build_phoenix_runtime(checkpointer):
    workflow = StateGraph(OrchestratorState)
    workflow.add_node("auditor", run_auditor_agent)
    workflow.add_node("janitor", run_janitor_agent)
    workflow.add_node("executor", run_executor_node)
    workflow.add_node("memory_updater", run_memory_updater_node)
    workflow.add_node("qdrant_tools", ToolNode([search_historical_playbooks]))

    workflow.add_edge(START, "auditor")

    # Auditor routing
    workflow.add_conditional_edges(
        "auditor",
        evaluate_auditor_edge,
        {"to_janitor": "janitor", "to_end": END}
    )

    # Janitor ReAct routing
    workflow.add_conditional_edges(
        "janitor",
        tools_condition,
        {"tools": "qdrant_tools", END: "executor"}
    )
    workflow.add_edge("qdrant_tools", "janitor")

    # NEW: Executor deterministic routing
    workflow.add_conditional_edges(
        "executor",
        evaluate_executor_edge,
        {"to_memory": "memory_updater", "to_end": END}
    )

    workflow.add_edge("memory_updater", END)

    return workflow.compile(checkpointer=checkpointer)
