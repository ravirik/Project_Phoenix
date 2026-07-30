from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from typing import Any, Optional

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Shared between the pipeline (nodes.py) and the dashboard, so both sides
# always agree on node order and display names for the live workflow view.
PIPELINE_ORDER = [
    "de_crew",
    "executor",
    "memory_updater",
    "mle_crew",
    "ml_executor",
    "auditor_node"  # Added the final governance gate
]

STAGE_LABELS = {
    "de_crew": {"agent": "Janitor Crew", "icon": "🧹", "desc": "Data Cleaning & Schema Patching"},
    "executor": {"agent": "DE Sandbox", "icon": "📦", "desc": "OS Sandbox Validation"},
    "memory_updater": {"agent": "Memory Updater", "icon": "🧠", "desc": "Qdrant Vector Storage"},
    "mle_crew": {"agent": "Scientist Crew", "icon": "🧬", "desc": "Model Training & Feature Scaling"},
    "ml_executor": {"agent": "MLE Sandbox", "icon": "📦", "desc": "Latency & Fairness Gate"},
    "auditor_node": {"agent": "Auditor Board", "icon": "⚖️", "desc": "AI Compliance & Hard Gate"},
}

_STAGE_TTL_S = 15 * 60
_ACTIVE_TRACE_TTL_S = 10 * 60
_HISTORY_MAX = 50


# --------------------------------------------------------------------------- #
# WRITE SIDE (async - used by nodes.py / app.py)
# --------------------------------------------------------------------------- #
_async_client = None


def _get_async_client():
    global _async_client
    if _async_client is None:
        import redis.asyncio as aioredis

        # Explicit timeouts are load-bearing here, not cosmetic: without
        # them, an unreachable Redis (packet black-holed, not just
        # connection-refused) would make every node's publish_stage()
        # call hang for the OS-level TCP connect timeout before the
        # try/except's exception ever fires - silently adding latency to
        # every single pipeline stage, which defeats the entire point of
        # "best-effort, never affects the pipeline."
        _async_client = aioredis.from_url(
            REDIS_URL, decode_responses=True, socket_connect_timeout=1.5, socket_timeout=1.5
        )
    return _async_client


async def start_run(trace_id: str, pipeline_id: str = "") -> None:
    """Call once, right before entering the graph for a new telemetry
    event. Best-effort: never raises."""
    try:
        client = _get_async_client()
        pipe = client.pipeline()
        pipe.incr("phoenix:stats:total")
        pipe.set("phoenix:active_trace", trace_id, ex=_ACTIVE_TRACE_TTL_S)
        pipe.hset(f"phoenix:meta:{trace_id}", mapping={
                  "pipeline_id": pipeline_id, "started_at": time.time()})
        pipe.expire(f"phoenix:meta:{trace_id}", _STAGE_TTL_S)
        await pipe.execute()
    except Exception:
        pass


async def publish_stage(trace_id: str, node: str, status: str, detail: str = "") -> None:
    """Call at the start (status='RUNNING') and end (status='SUCCESS' /
    'FAILED') of every graph node. Best-effort: never raises, so a Redis
    outage can never affect the actual self-healing pipeline."""
    try:
        client = _get_async_client()
        payload = json.dumps(
            {"status": status, "detail": detail[:300], "ts": time.time()})
        key = f"phoenix:stage:{trace_id}"
        await client.hset(key, node, payload)
        await client.expire(key, _STAGE_TTL_S)
    except Exception:
        pass


async def finish_run(trace_id: str, final_status: str) -> None:
    """Call once, after the graph run completes (success or failure)."""
    try:
        client = _get_async_client()
        pipe = client.pipeline()
        # 🐛 FIX: auditor_node is now the graph's real terminal node (see
        # graph.py), so final_status is one of ITS status strings, not
        # "ML_EXECUTION_COMPLETED" - that was only ever the second-to-last
        # node's status. Without this, every genuinely successful run got
        # silently counted as a failure here.
        if final_status in {
            "SUCCESS", "ML_EXECUTION_COMPLETED", "NOMINAL",
            "AUDIT_PASSED_PROMOTED", "AUDIT_PASSED_CHALLENGER",
            "AUDIT_PASSED_DATA", "AUDITOR_BYPASSED",
        }:
            pipe.incr("phoenix:stats:success")
        else:
            pipe.incr("phoenix:stats:failed")
        summary = json.dumps(
            {"trace_id": trace_id, "final_status": final_status,
                "finished_at": time.time()}
        )
        pipe.lpush("phoenix:history", summary)
        pipe.ltrim("phoenix:history", 0, _HISTORY_MAX - 1)
        await pipe.execute()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# READ SIDE (sync - used by the Streamlit dashboard)
# --------------------------------------------------------------------------- #
@lru_cache
def _get_sync_client():
    import redis

    return redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)


def get_stats() -> dict[str, int]:
    """Returns {'total', 'success', 'failed'}. Zeroed out (not an
    exception) if Redis is unreachable, so the dashboard degrades
    gracefully instead of crashing."""
    try:
        client = _get_sync_client()
        total = int(client.get("phoenix:stats:total") or 0)
        success = int(client.get("phoenix:stats:success") or 0)
        failed = int(client.get("phoenix:stats:failed") or 0)
        return {"total": total, "success": success, "failed": failed}
    except Exception:
        return {"total": 0, "success": 0, "failed": 0}


def get_active_trace() -> Optional[str]:
    try:
        return _get_sync_client().get("phoenix:active_trace")
    except Exception:
        return None


def get_stage_state(trace_id: str) -> dict[str, dict[str, Any]]:
    """Returns {node_name: {"status": ..., "detail": ..., "ts": ...}} for
    every node that has reported in for this trace so far."""
    try:
        raw = _get_sync_client().hgetall(f"phoenix:stage:{trace_id}")
        return {node: json.loads(blob) for node, blob in raw.items()}
    except Exception:
        return {}


def get_recent_history(n: int = 20) -> list[dict[str, Any]]:
    try:
        raw = _get_sync_client().lrange("phoenix:history", 0, n - 1)
        return [json.loads(r) for r in raw]
    except Exception:
        return []


def is_reachable() -> bool:
    try:
        return bool(_get_sync_client().ping())
    except Exception:
        return False
