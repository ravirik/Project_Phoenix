from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from src.domain.models import OrchestratorState
from src.agents.nodes import run_auditor_agent, run_janitor_agent, run_executor_node
from src.agents.tools import search_historical_playbooks


def evaluate_conditional_edge(state: OrchestratorState):
    return "to_janitor" if state["status"] == "DRIFT_DETECTED" else "to_end"


def build_phoenix_runtime(checkpointer):
    workflow = StateGraph(OrchestratorState)
    workflow.add_node("auditor", run_auditor_agent)
    workflow.add_node("janitor", run_janitor_agent)
    workflow.add_node("executor", run_executor_node)
    workflow.add_node("qdrant_tools", ToolNode([search_historical_playbooks]))

    workflow.add_edge(START, "auditor")
    workflow.add_conditional_edges("auditor", evaluate_conditional_edge, {
                                   "to_janitor": "janitor", "to_end": END})
    workflow.add_conditional_edges("janitor", tools_condition, {
                                   "tools": "qdrant_tools", END: "executor"})
    workflow.add_edge("qdrant_tools", "janitor")
    workflow.add_edge("executor", END)

    return workflow.compile(checkpointer=checkpointer)
