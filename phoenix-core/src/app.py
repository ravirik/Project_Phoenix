import os
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from src.domain.models import OrchestratorState
from src.agents.graph import build_phoenix_runtime


async def trigger_agentic_pipeline(payload: dict):
    DB_URI = os.getenv(
        "DATABASE_URL", "postgresql://phoenix_admin:secret_vault_pass_2026@localhost:5432/phoenix_telemetry_state")

    async with AsyncConnectionPool(conninfo=DB_URI, min_size=1, max_size=5, kwargs={"autocommit": True}) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        phoenix_engine = build_phoenix_runtime(checkpointer)

        initial_state = OrchestratorState(
            trace_id=payload['trace_id'],
            pipeline_id=payload['pipeline_id'],
            status="INITIALIZED",
            retry_count=0,
            messages=[],
            final_patch=None,
            data_path="data/default_ingestion.parquet"
        )

        async for event in phoenix_engine.astream(initial_state, config={"configurable": {"thread_id": payload['trace_id']}}):
            pass
