import os
import logfire
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from src.domain.models import OrchestratorState
from src.agents.graph import build_phoenix_runtime

# Global runtime singletons
_runtime_engine = None


async def initialize_app_runtime():
    """Initializes connection pool, checkpointer, and graph runtime ONCE on startup."""
    global _runtime_engine

    db_url = os.getenv("POSTGRES_URL") or os.getenv(
        "DATABASE_URL",
        "postgresql://phoenix_admin:secret_vault_pass_2026@postgres:5432/phoenix_telemetry_state"
    )

    pool = AsyncConnectionPool(
        conninfo=db_url, min_size=1, max_size=5, open=False, kwargs={"autocommit": True})
    await pool.open()

    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()

    _runtime_engine = build_phoenix_runtime(checkpointer)
    print("[Application] ✅ LangGraph state machine runtime initialized with Postgres checkpointer.", flush=True)


async def trigger_agentic_pipeline(payload: dict):
    """Executes the agentic pipeline reusing the pre-warmed graph runtime engine."""
    global _runtime_engine

    if _runtime_engine is None:
        await initialize_app_runtime()

    initial_state = OrchestratorState(
        trace_id=payload['trace_id'],
        pipeline_id=payload['pipeline_id'],
        status="INITIALIZED",
        retry_count=0,
        messages=[],
        final_patch=None,
        data_path=payload.get("data_path", "data/default_ingestion.parquet")
    )

    with logfire.span("phoenix.pipeline_execution", trace_id=payload['trace_id']):
        async for event in _runtime_engine.astream(
            initial_state,
            config={"configurable": {"thread_id": payload['trace_id']}}
        ):
            pass
