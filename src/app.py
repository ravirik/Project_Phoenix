import os
import logfire
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from src.domain.models import OrchestratorState
from src.agents.graph import build_phoenix_runtime
from src.observability_bus import start_run, finish_run

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


def build_initial_state(payload: dict) -> OrchestratorState:
    """Initial state for one delivery of a telemetry message.

    The Postgres checkpointer resumes an existing thread when a trace_id is seen again
    (a NATS redelivery or a replayed trace), and any key left out here keeps its value
    from the previous run. So every per-run field is reset explicitly; otherwise a replay
    starts near the retry limit and is dead-lettered on its first failure.
    """
    payload_metrics = payload.get("metrics") or {}

    return OrchestratorState(
        trace_id=payload['trace_id'],
        pipeline_id=payload['pipeline_id'],
        status="INITIALIZED",
        retry_count=0,
        messages=[],
        final_patch=None,
        data_path=payload.get("data_path", "data/default_ingestion.parquet"),
        # 🐛 FIX: auditor_watcher.py computes and publishes this in
        # payload["metrics"]["psi_score"], but it was never threaded into
        # OrchestratorState, so run_auditor_node could never see it.
        psi_score=payload_metrics.get("psi_score", 0.0),
        metrics=payload_metrics,
        de_retry_count=0,
        mle_retry_count=0,
        de_escalation_count=0,
        de_error_feedback=None,
        de_previous_code=None,
        mle_error_feedback=None,
        mle_previous_code=None,
        mle_output=None,
        candidate_metrics=None,
        candidate_task_type=None,
        gate_failures=None,
        primary_score=None,
        candidate_f1=None,
        promotion_status=None,
    )


async def trigger_agentic_pipeline(payload: dict):
    """Executes the agentic pipeline reusing the pre-warmed graph runtime engine."""
    global _runtime_engine

    if _runtime_engine is None:
        await initialize_app_runtime()

    initial_state = build_initial_state(payload)

    await start_run(payload["trace_id"], payload.get("pipeline_id", ""))

    final_status = "UNKNOWN"
    with logfire.span("phoenix.pipeline_execution", trace_id=payload['trace_id']):
        async for event in _runtime_engine.astream(
            initial_state,
            config={"configurable": {"thread_id": payload['trace_id']}}
        ):
            for node_output in event.values():
                if isinstance(node_output, dict) and "status" in node_output:
                    final_status = node_output["status"]

    await finish_run(payload["trace_id"], final_status)
