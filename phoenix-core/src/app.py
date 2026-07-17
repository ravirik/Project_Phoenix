import os
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.domain.models import OrchestratorState
from src.agents.graph import build_phoenix_runtime


async def trigger_agentic_pipeline(payload: dict):
    """Entry point called by the NATS consumer when new telemetry arrives."""
    DB_URI = os.getenv(
        "DATABASE_URL", "postgresql://phoenix_admin:secret_vault_pass_2026@localhost:5432/phoenix_telemetry_state")

    # Added kwargs={"autocommit": True} to allow concurrent index creation during setup
    async with AsyncConnectionPool(conninfo=DB_URI, min_size=1, max_size=5, kwargs={"autocommit": True}) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()

        # Build the engine dynamically from our orchestrator factory
        phoenix_engine = build_phoenix_runtime(checkpointer)
        session_config = {"configurable": {"thread_id": payload['trace_id']}}

        # Initialize state, ensuring the messages array exists for the tool calling
        initial_state = OrchestratorState(
            trace_id=payload['trace_id'],
            pipeline_id=payload['pipeline_id'],
            status="INITIALIZED",
            retry_count=0,
            messages=[]
        )

        # Execute the AI state machine
        async for event in phoenix_engine.astream(initial_state, config=session_config):
            pass
