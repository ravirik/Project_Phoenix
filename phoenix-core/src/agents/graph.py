from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.prebuilt import ToolNode, tools_condition

from src.domain.models import OrchestratorState
from src.agents.nodes import run_auditor_agent, run_janitor_agent
from src.agents.tools import search_historical_playbooks


def evaluate_conditional_edge(state: OrchestratorState) -> Literal["to_janitor", "to_end"]:
    if state["status"] == "DRIFT_DETECTED":
        return "to_janitor"
    return "to_end"


def build_phoenix_runtime(checkpointer: AsyncPostgresSaver) -> StateGraph:
    """Constructs the LangGraph orchestration engine."""
    workflow = StateGraph(OrchestratorState)

    # Add Nodes
    workflow.add_node("auditor_surveillance", run_auditor_agent)
    workflow.add_node("janitor_remediation", run_janitor_agent)

    # Add the ToolNode to execute the Python functions
    workflow.add_node("qdrant_tools", ToolNode([search_historical_playbooks]))

    # Main Execution Flow
    workflow.add_edge(START, "auditor_surveillance")
    workflow.add_conditional_edges(
        "auditor_surveillance",
        evaluate_conditional_edge,
        {"to_janitor": "janitor_remediation", "to_end": END}
    )

    # Tool Routing for the Janitor
    workflow.add_conditional_edges(
        "janitor_remediation",
        tools_condition,
        # REMOVE the quotes around END so it references the LangGraph constant
        {"tools": "qdrant_tools", END: "auditor_surveillance"}
    )

    # After the tool runs, bounce back to the Janitor to read the result
    workflow.add_edge("qdrant_tools", "janitor_remediation")

    return workflow.compile(checkpointer=checkpointer)
