from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# --- Make `src` importable regardless of where `streamlit run` is invoked from ---
PHOENIX_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PHOENIX_ROOT))

from src import observability_bus as bus  # noqa: E402

try:
    from streamlit_autorefresh import st_autorefresh
    _HAS_AUTOREFRESH = True
except ImportError:
    _HAS_AUTOREFRESH = False

try:
    from qdrant_client import QdrantClient
    _HAS_QDRANT = True
except ImportError:
    _HAS_QDRANT = False


# =============================================================================
# 1. PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Project Phoenix | Mission Control",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 2. STYLING (kept from the original design, extended with live-state classes)
# =============================================================================
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .stApp {
        background-color: #0d0f14;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        text-rendering: optimizeLegibility;
        -webkit-font-smoothing: antialiased;
    }

    [data-testid="stSidebar"] {
        background-color: #131720;
        border-right: 1px solid rgba(255, 255, 255, 0.04);
        padding-top: 1rem;
    }

    .bento-card {
        background-color: #171b26;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        transition: border-color 0.2s ease;
    }
    .bento-card:hover { border-color: rgba(59, 130, 246, 0.3); }

    .sidebar-promo-card {
        background: linear-gradient(135deg, #7c3aed 0%, #db2777 100%);
        border-radius: 14px;
        padding: 18px;
        margin-top: 1.5rem;
        color: white;
        text-align: left;
        box-shadow: 0 8px 24px rgba(124, 58, 237, 0.3);
    }

    .card-title {
        font-size: 0.78rem;
        color: #8a94a6;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
        font-weight: 600;
    }
    .card-value { font-size: 1.65rem; font-weight: 700; color: #ffffff; letter-spacing: -0.02em; }
    .trend-green { color: #34d399; font-size: 0.82rem; font-weight: 600; }
    .trend-blue { color: #60a5fa; font-size: 0.82rem; font-weight: 600; }
    .trend-red { color: #f87171; font-size: 0.82rem; font-weight: 600; }
    .trend-gray { color: #64748b; font-size: 0.82rem; font-weight: 600; }

    /* Real, clickable sidebar nav buttons styled to look like the original
       decorative <p> items - this is the fix for "buttons are not
       clickable": those were plain markdown/HTML with no event wiring. */
    [data-testid="stSidebar"] .stButton button {
        background: transparent;
        border: none;
        color: #94a3b8;
        text-align: left;
        font-size: 0.88rem;
        font-weight: 500;
        padding: 9px 12px;
        width: 100%;
        justify-content: flex-start;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: rgba(255,255,255,0.04);
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] .stButton button:focus:not(:active) {
        color: #94a3b8;
        border: none;
    }
    .nav-active button {
        background: rgba(59, 130, 246, 0.12) !important;
        color: #60a5fa !important;
        font-weight: 600 !important;
    }

    /* Live workflow DAG boxes */
    .stage-box {
        border-radius: 14px;
        padding: 16px 14px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.06);
        min-height: 128px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 4px;
    }
    .stage-idle    { background: #171b26; opacity: 0.55; }
    .stage-running { background: #1e3a5f; border-color: #3b82f6; animation: pulse 1.4s ease-in-out infinite; }
    .stage-success { background: #16302a; border-color: #34d399; }
    .stage-failed  { background: #3a1a1f; border-color: #f87171; }
    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(59,130,246,0.45); }
        70%  { box-shadow: 0 0 0 12px rgba(59,130,246,0); }
        100% { box-shadow: 0 0 0 0 rgba(59,130,246,0); }
    }
    .stage-icon { font-size: 1.5rem; }
    .stage-name { font-size: 0.82rem; font-weight: 700; color: #ffffff; }
    .stage-desc { font-size: 0.72rem; color: #94a3b8; line-height: 1.3; }
    .stage-status { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.05em; margin-top: 2px; }
    .arrow { text-align: center; font-size: 1.4rem; color: #3b4354; padding-top: 45px; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 3. DATA HELPERS (all live, all degrade gracefully)
# =============================================================================
def load_model_registry() -> dict:
    path = PHOENIX_ROOT / "models" / "model_registry.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


@st.cache_resource(ttl=30)
def get_qdrant_client():
    if not _HAS_QDRANT:
        return None
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    try:
        return QdrantClient(url=url, timeout=2)
    except Exception:
        return None


def get_qdrant_point_count(collection: str = "remediation_playbooks") -> int | None:
    client = get_qdrant_client()
    if client is None:
        return None
    try:
        if not client.collection_exists(collection_name=collection):
            return 0
        return client.get_collection(collection).points_count
    except Exception:
        return None


def get_qdrant_sample(collection: str = "remediation_playbooks", limit: int = 15):
    client = get_qdrant_client()
    if client is None:
        return []
    try:
        if not client.collection_exists(collection_name=collection):
            return []
        points, _ = client.scroll(
            collection_name=collection, limit=limit, with_payload=True)
        return [p.payload for p in points]
    except Exception:
        return []


def mask_secret(value: str | None, keep: int = 4) -> str:
    if not value:
        return "— not set —"
    if len(value) <= keep:
        return "•" * len(value)
    return value[:keep] + "•" * max(len(value) - keep, 4)


def mask_url_password(url: str | None) -> str:
    if not url or "@" not in url or "://" not in url:
        return url or "— not set —"
    try:
        scheme, rest = url.split("://", 1)
        creds, host = rest.split("@", 1)
        user = creds.split(":")[0]
        return f"{scheme}://{user}:••••••@{host}"
    except Exception:
        return url


# =============================================================================
# 4. SIDEBAR - branding, REAL clickable navigation, live-status promo card
# =============================================================================
PAGES = ["Overview", "Live Execution", "Agent Fleet",
         "Vector Memory", "ONNX Artifacts", "System Config"]
PAGE_ICONS = {
    "Overview": "⚡", "Live Execution": "📊", "Agent Fleet": "🤖",
    "Vector Memory": "🧠", "ONNX Artifacts": "📦", "System Config": "⚙️",
}

if "page" not in st.session_state:
    st.session_state.page = "Live Execution"

with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 1.6rem; padding-left: 8px;">
            <div style="background: linear-gradient(135deg, #f59e0b, #ef4444); width: 28px; height: 28px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 14px;">🔥</div>
            <span style="font-size: 1.15rem; font-weight: 700; color: #ffffff; letter-spacing: -0.02em;">Project Phoenix</span>
        </div>
    """, unsafe_allow_html=True)

    for page_name in PAGES:
        is_active = st.session_state.page == page_name
        wrapper_class = "nav-active" if is_active else ""
        st.markdown(f'<div class="{wrapper_class}">', unsafe_allow_html=True)
        if st.button(f"{PAGE_ICONS[page_name]}  {page_name}", key=f"nav_{page_name}", use_container_width=True):
            st.session_state.page = page_name
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    redis_ok = bus.is_reachable()
    active_trace = bus.get_active_trace() if redis_ok else None
    status_line = f"Tracking <code>{active_trace}</code>" if active_trace else "No trace in flight"
    st.markdown(f"""
        <div class="sidebar-promo-card">
            <div style="font-weight: 700; font-size: 0.92rem; margin-bottom: 4px;">
                {"🟢 Bus Connected" if redis_ok else "🔴 Bus Unreachable"}
            </div>
            <div style="font-size: 0.78rem; opacity: 0.9; line-height: 1.4;">{status_line}</div>
        </div>
    """, unsafe_allow_html=True)
    if st.button("Inspect Live Trace", key="inspect_trace_btn", use_container_width=True):
        st.session_state.page = "Live Execution"
        st.rerun()


# =============================================================================
# 5. PAGE: OVERVIEW - real counters, real charts
# =============================================================================
def render_overview():
    stats = bus.get_stats()
    registry = load_model_registry()
    champion = registry.get("active_champion", {})
    champion_metrics = champion.get("metrics", {})
    history = registry.get("history", [])
    qdrant_count = get_qdrant_point_count()

    success_rate = (stats["success"] / stats["total"]
                    * 100) if stats["total"] else 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class="bento-card">
                <div class="card-title">Pipelines Executed</div>
                <div class="card-value">{stats['total']:,}</div>
                <div class="trend-blue" style="margin-top: 6px;">↗ Since worker start (Redis counter)</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        trend_cls = "trend-green" if success_rate >= 80 else "trend-red"
        st.markdown(f"""
            <div class="bento-card">
                <div class="card-title">Self-Healing Success</div>
                <div class="card-value">{success_rate:.1f}%</div>
                <div class="{trend_cls}" style="margin-top: 6px;">{stats['success']} success / {stats['failed']} failed</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        qc_display = f"{qdrant_count:,}" if qdrant_count is not None else "—"
        qc_trend = "↗ Playbooks Cached" if qdrant_count is not None else "Qdrant unreachable"
        st.markdown(f"""
            <div class="bento-card">
                <div class="card-title">Qdrant Vector Points</div>
                <div class="card-value">{qc_display}</div>
                <div class="trend-blue" style="margin-top: 6px;">{qc_trend}</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        lat = champion_metrics.get("latency_ms")
        lat_display = f"{lat:.1f} ms" if isinstance(lat, (int, float)) else "—"
        st.markdown(f"""
            <div class="bento-card">
                <div class="card-title">Champion Inference Latency</div>
                <div class="card-value">{lat_display}</div>
                <div class="trend-green" style="margin-top: 6px;">v{champion.get('version', '—')} | {champion.get('status', 'UNKNOWN')}</div>
            </div>
        """, unsafe_allow_html=True)

    row2_left, row2_right = st.columns([1.2, 1])

    with row2_left:
        primary_metric_name = "f1_score" if "f1_score" in champion_metrics else next(
            (k for k in ("rmse", "silhouette_score", "roc_auc")
             if k in champion_metrics), None
        )
        metric_label = (primary_metric_name or "score").replace(
            "_", " ").upper()
        metric_value = champion_metrics.get(
            primary_metric_name) if primary_metric_name else None

        st.markdown(f"""
            <div class="bento-card">
                <div class="card-title">Champion vs. Challenger History ({metric_label})</div>
                <div style="font-size: 1.7rem; font-weight: 700; color: #ffffff;">
                    {f'{metric_value:.4f}' if isinstance(metric_value, (int, float)) else '—'}
                </div>
        """, unsafe_allow_html=True)

        if history and primary_metric_name:
            values = [
                h.get("metrics", {}).get(primary_metric_name)
                for h in history if h.get("metrics", {}).get(primary_metric_name) is not None
            ]
            if values:
                st.line_chart(pd.DataFrame(
                    {metric_label: values}), color="#f43f5e", height=185, use_container_width=True)
            else:
                st.caption("No comparable challenger metrics recorded yet.")
        else:
            st.caption(
                "No challenger history in models/model_registry.json yet - run the MLE loop to populate it.")
        st.markdown("</div>", unsafe_allow_html=True)

    with row2_right:
        st.markdown("""
            <div class="bento-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                    <span style="font-weight: 600; font-size: 0.9rem; color: #ffffff;">Recent Pipeline Runs</span>
                </div>
        """, unsafe_allow_html=True)
        recent = bus.get_recent_history(10)
        if recent:
            df = pd.DataFrame(recent)
            df["finished_at"] = pd.to_datetime(
                df["finished_at"], unit="s").dt.strftime("%H:%M:%S")
            df = df.rename(columns={
                           "trace_id": "Trace ID", "final_status": "Status", "finished_at": "Finished"})
            st.dataframe(df[["Trace ID", "Status", "Finished"]],
                         use_container_width=True, hide_index=True)
        else:
            st.caption(
                "No completed runs recorded yet. Trigger a telemetry event to populate this.")
        st.markdown("</div>", unsafe_allow_html=True)

    if history:
        row3_left, row3_right = st.columns(2)
        with row3_left:
            st.markdown("""
                <div class="bento-card">
                    <div class="card-title">Challenger Attempts (Chronological)</div>
            """, unsafe_allow_html=True)
            versions = [h.get("version", "?")[:14] for h in history[-10:]]
            st.bar_chart(pd.DataFrame({"attempts": range(1, len(versions) + 1)}, index=versions),
                         color="#d97706", height=140, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with row3_right:
            st.markdown("""
                <div class="bento-card">
                    <div class="card-title">Promotion Status Breakdown</div>
            """, unsafe_allow_html=True)
            statuses = pd.Series([h.get("status", "UNKNOWN")
                                 for h in history]).value_counts()
            st.bar_chart(statuses, color="#8b5cf6",
                         height=140, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# 6. PAGE: LIVE EXECUTION - the moving-workflow view (Updated for 6 nodes)
# =============================================================================
def render_live_execution():
    st.markdown("### 📊 Live Execution")
    st.caption(
        "Mirrors what the ai-worker terminal shows per stage, as a live-updating workflow. "
        "Auto-refreshes while this page is open."
    )

    if _HAS_AUTOREFRESH:
        st_autorefresh(interval=1500, key="live_execution_autorefresh")
    else:
        st.info("Install `streamlit-autorefresh` for automatic updates; showing a manual snapshot for now.", icon="ℹ️")
        if st.button("🔄 Refresh now"):
            st.rerun()

    if not bus.is_reachable():
        st.error(
            "Cannot reach Redis at the configured REDIS_URL. The live view needs the same "
            "Redis instance the ai-worker publishes to - check `docker compose ps` / your .env.",
            icon="🔴",
        )
        return

    active_trace = bus.get_active_trace()
    if not active_trace:
        st.warning(
            "No pipeline is currently in flight. Send a telemetry event to the gateway to see it move through the stages live.", icon="⏳")
        return

    st.markdown(f"**Tracking trace:** `{active_trace}`")

    stage_state = bus.get_stage_state(active_trace)

    # 🌟 6 nodes require 11 columns (6 stage boxes + 5 arrows)
    cols = st.columns([3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3])
    stage_columns = [cols[0], cols[2], cols[4], cols[6], cols[8], cols[10]]
    arrow_columns = [cols[1], cols[3], cols[5], cols[7], cols[9]]

    for i, node in enumerate(bus.PIPELINE_ORDER):
        meta = bus.STAGE_LABELS.get(
            node, {"agent": node, "icon": "❓", "desc": ""})
        state = stage_state.get(node)
        status = (state or {}).get("status", "IDLE")
        css_class = {
            "RUNNING": "stage-running",
            "SUCCESS": "stage-success",
            "FAILED": "stage-failed",
        }.get(status, "stage-idle")
        status_color = {"RUNNING": "#60a5fa", "SUCCESS": "#34d399",
                        "FAILED": "#f87171"}.get(status, "#64748b")

        with stage_columns[i]:
            st.markdown(f"""
                <div class="stage-box {css_class}">
                    <div class="stage-icon">{meta['icon']}</div>
                    <div class="stage-name">{meta['agent']}</div>
                    <div class="stage-desc">{meta['desc']}</div>
                    <div class="stage-status" style="color:{status_color};">{status}</div>
                </div>
            """, unsafe_allow_html=True)

    for arrow_col in arrow_columns:
        with arrow_col:
            st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)

    st.markdown("#### Stage Detail Log & Artifacts")
    for node in bus.PIPELINE_ORDER:
        state = stage_state.get(node)
        if not state:
            continue
        meta = bus.STAGE_LABELS.get(node, {"agent": node, "icon": "❓"})
        ts = time.strftime("%H:%M:%S", time.localtime(
            state.get("ts", time.time())))

        detail_text = state.get("detail", "")

        # If the detail contains markdown code blocks or release notes, render them natively!
        with st.expander(f"[{ts}] {meta['icon']} {meta['agent']} — {state.get('status')}"):
            st.markdown(detail_text)


# =============================================================================
# 7. PAGE: AGENT FLEET - static roster + live status if a trace is active
# =============================================================================
def render_agent_fleet():
    st.markdown("### 🤖 Agent Fleet")
    active_trace = bus.get_active_trace() if bus.is_reachable() else None
    stage_state = bus.get_stage_state(active_trace) if active_trace else {}

    rows = []
    for node in bus.PIPELINE_ORDER:
        meta = bus.STAGE_LABELS[node]
        state = stage_state.get(node, {})
        rows.append({
            "Node": node,
            "Agent": meta["agent"],
            "Role": meta["desc"],
            "Current Status": state.get("status", "IDLE"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# =============================================================================
# 8. PAGE: VECTOR MEMORY - real Qdrant sample
# =============================================================================
def render_vector_memory():
    st.markdown("### 🧠 Vector Memory (Qdrant)")
    count = get_qdrant_point_count()
    if count is None:
        st.error("Qdrant is unreachable at the configured QDRANT_URL.", icon="🔴")
        return
    st.metric("Playbooks stored", count)

    sample = get_qdrant_sample()
    if not sample:
        st.info("No playbooks indexed yet.")
        return
    for p in sample:
        with st.expander(p.get("incident", "Unknown incident")[:100]):
            st.code(p.get("code", ""), language="python")
            st.caption(p.get("explanation", ""))


# =============================================================================
# 9. PAGE: ONNX ARTIFACTS - registry + files on disk
# =============================================================================
def render_onnx_artifacts():
    st.markdown("### 📦 ONNX Artifacts & Model Registry")
    registry = load_model_registry()
    if not registry:
        st.info(
            "No models/model_registry.json found yet - run the MLE loop once to create it.")
        return

    st.markdown("#### Active Champion")
    st.json(registry.get("active_champion", {}))

    st.markdown("#### Challenger History")
    history = registry.get("history", [])
    if history:
        st.dataframe(pd.DataFrame(history),
                     use_container_width=True, hide_index=True)
    else:
        st.caption("No challengers recorded yet.")

    models_dir = PHOENIX_ROOT / "models"
    if models_dir.exists():
        st.markdown("#### Files in `models/`")
        files = sorted(models_dir.glob("*"))
        st.dataframe(
            pd.DataFrame([{"file": f.name, "size_kb": round(
                f.stat().st_size / 1024, 1)} for f in files if f.is_file()]),
            use_container_width=True, hide_index=True,
        )


# =============================================================================
# 10. PAGE: SYSTEM CONFIG - sanitized live env
# =============================================================================
def render_system_config():
    st.markdown("### ⚙️ System Config")
    st.caption("Read directly from the running environment. Secrets are masked.")

    rows = [
        {"Setting": "LLM_PROVIDER", "Value": os.getenv(
            "LLM_PROVIDER", "gemini")},
        {"Setting": "GEMINI_MODEL", "Value": os.getenv(
            "GEMINI_MODEL", "gemini-2.5-flash")},
        {"Setting": "GCP_PROJECT_ID", "Value": os.getenv(
            "GCP_PROJECT_ID") or "— using API key fallback —"},
        {"Setting": "GCP_LOCATION", "Value": os.getenv(
            "GCP_LOCATION", "us-central1")},
        {"Setting": "GEMINI_API_KEY", "Value": mask_secret(
            os.getenv("GEMINI_API_KEY"))},
        {"Setting": "QDRANT_URL", "Value": os.getenv(
            "QDRANT_URL", "— not set —")},
        {"Setting": "NATS_URL", "Value": os.getenv("NATS_URL", "— not set —")},
        {"Setting": "REDIS_URL", "Value": mask_url_password(
            os.getenv("REDIS_URL"))},
        {"Setting": "POSTGRES_URL", "Value": mask_url_password(
            os.getenv("POSTGRES_URL") or os.getenv("DATABASE_URL"))},
        {"Setting": "LOGFIRE_TOKEN", "Value": mask_secret(
            os.getenv("LOGFIRE_TOKEN"))},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### Connectivity")
    checks = {
        "Redis (live telemetry bus)": bus.is_reachable(),
        "Qdrant (vector memory)": get_qdrant_point_count() is not None,
    }
    for name, ok in checks.items():
        st.markdown(f"{'🟢' if ok else '🔴'} {name}")


# =============================================================================
# 11. ROUTER
# =============================================================================
PAGE_RENDERERS = {
    "Overview": render_overview,
    "Live Execution": render_live_execution,
    "Agent Fleet": render_agent_fleet,
    "Vector Memory": render_vector_memory,
    "ONNX Artifacts": render_onnx_artifacts,
    "System Config": render_system_config,
}

PAGE_RENDERERS[st.session_state.page]()
