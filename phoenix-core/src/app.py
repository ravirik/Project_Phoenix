import os
from typing import Literal
from psycopg_pool import AsyncConnectionPool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.domain.models import OrchestratorState
from src.agents.nodes import run_auditor_agent, run_janitor_agent

def evaluate_conditional_edge(state: OrchestratorState) -> Literal["to_janitor", "to_end"]:
    if state["status"] == "DRIFT_DETECTED":
        return "to_janitor"
    return "to_end"

def build_phoenix_runtime(checkpointer: AsyncPostgresSaver) -> StateGraph:
    workflow = StateGraph(OrchestratorState)
    workflow.add_node("auditor_surveillance", run_auditor_agent)
    workflow.add_node("janitor_remediation", run_janitor_agent)
    
    workflow.add_edge(START, "auditor_surveillance")
    workflow.add_conditional_edges(
        "auditor_surveillance",
        evaluate_conditional_edge,
        {"to_janitor": "janitor_remediation", "to_end": END}
    )
    workflow.add_edge("janitor_remediation", "auditor_surveillance")
    return workflow.compile(checkpointer=checkpointer)

async def trigger_agentic_pipeline(payload: dict):
    """Entry point called by the NATS consumer when new telemetry arrives."""
    DB_URI = os.getenv("DATABASE_URL", "postgresql://phoenix_admin:secret_vault_pass_2026@localhost:5432/phoenix_telemetry_state")
    
    # FIX: Added kwargs={"autocommit": True} to allow concurrent index creation during setup
    async with AsyncConnectionPool(conninfo=DB_URI, min_size=1, max_size=5, kwargs={"autocommit": True}) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup() 
        
        phoenix_engine = build_phoenix_runtime(checkpointer)
        session_config = {"configurable": {"thread_id": payload['trace_id']}}
        
        initial_state = OrchestratorState(
            trace_id=payload['trace_id'],
            pipeline_id=payload['pipeline_id'],
            status="INITIALIZED",
            retry_count=0
        )
        
        # Execute the AI state machine
        async for event in phoenix_engine.astream(initial_state, config=session_config):
            pass 
