import os
import time
import streamlit as st
import pandas as pd

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Project Phoenix | Mission Control",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. HIGH-DENSITY APPLE-STYLE MINIMALIST CSS (Header kept visible for sidebar toggle functionality)
st.markdown("""
<style>
    /* Hide only navigation menus and footers, keeping the native top header controls visible */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Global Dark Canvas with High Pixel Density Smoothing */
    .stApp {
        background-color: #0d0f14;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        text-rendering: optimizeLegibility;
        -webkit-font-smoothing: antialiased;
    }

    /* Custom Refined Sidebar */
    [data-testid="stSidebar"] {
        background-color: #131720;
        border-right: 1px solid rgba(255, 255, 255, 0.04);
        padding-top: 1rem;
    }

    /* Bento Grid Card Style */
    .bento-card {
        background-color: #171b26;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        transition: border-color 0.2s ease;
    }
    .bento-card:hover {
        border-color: rgba(59, 130, 246, 0.3);
    }

    /* Sidebar Gradient Promo Card */
    .sidebar-promo-card {
        background: linear-gradient(135deg, #7c3aed 0%, #db2777 100%);
        border-radius: 14px;
        padding: 18px;
        margin-top: 3.5rem;
        color: white;
        text-align: left;
        box-shadow: 0 8px 24px rgba(124, 58, 237, 0.3);
    }
    
    .sidebar-promo-btn {
        background-color: #ffffff;
        color: #12151c;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        font-size: 0.82rem;
        width: 100%;
        margin-top: 12px;
        cursor: pointer;
        transition: transform 0.1s ease;
    }
    .sidebar-promo-btn:hover {
        transform: scale(1.02);
    }

    /* Typography & Metrics */
    .card-title {
        font-size: 0.78rem;
        color: #8a94a6;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
        font-weight: 600;
    }
    .card-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.02em;
    }
    .trend-green { color: #34d399; font-size: 0.82rem; font-weight: 600; }
    .trend-blue { color: #60a5fa; font-size: 0.82rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# 3. CUSTOM SIDEBAR BRANDED FOR PROJECT PHOENIX
with st.sidebar:
    # Phoenix Logo Header
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 2.2rem; padding-left: 8px;">
            <div style="background: linear-gradient(135deg, #f59e0b, #ef4444); width: 28px; height: 28px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 14px;">🔥</div>
            <span style="font-size: 1.15rem; font-weight: 700; color: #ffffff; letter-spacing: -0.02em;">Project Phoenix</span>
        </div>
    """, unsafe_allow_html=True)

    # Navigation Links
    st.markdown("""
        <div style="display: flex; flex-direction: column; gap: 6px; font-size: 0.88rem; color: #94a3b8; padding-left: 4px;">
            <p style="padding: 9px 12px; border-radius: 8px; cursor: pointer; transition: background 0.2s;">⚡ &nbsp; Overview</p>
            <p style="padding: 9px 12px; border-radius: 8px; cursor: pointer; transition: background 0.2s;">🤖 &nbsp; Agent Fleet</p>
            <p style="padding: 9px 12px; border-radius: 8px; cursor: pointer; transition: background 0.2s;">🧠 &nbsp; Vector Memory</p>
            <p style="padding: 9px 12px; border-radius: 8px; background: rgba(59, 130, 246, 0.12); color: #60a5fa; font-weight: 600;">📊 &nbsp; Live Execution</p>
            <p style="padding: 9px 12px; border-radius: 8px; cursor: pointer; transition: background 0.2s;">📦 &nbsp; ONNX Artifacts</p>
            <p style="padding: 9px 12px; border-radius: 8px; cursor: pointer; transition: background 0.2s;">⚙️ &nbsp; System Config</p>
        </div>
    """, unsafe_allow_html=True)

    # Bottom Gradient Active Agent Card
    st.markdown("""
        <div class="sidebar-promo-card">
            <div style="font-weight: 700; font-size: 0.92rem; margin-bottom: 4px;">Apex_SRE_Agent</div>
            <div style="font-size: 0.78rem; opacity: 0.9; line-height: 1.4;">Active loop monitoring schema drift and self-healing payloads.</div>
            <button class="sidebar-promo-btn">Inspect Trace</button>
        </div>
    """, unsafe_allow_html=True)


# 4. MAIN BENTO GRID DASHBOARD (PROJECT PHOENIX METRICS)

# ROW 1: Four Top System Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
        <div class="bento-card">
            <div class="card-title">Pipelines Executed</div>
            <div class="card-value">5,576</div>
            <div class="trend-green" style="margin-top: 6px;">↗ +3.31% today</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="bento-card">
            <div class="card-title">Self-Healing Success</div>
            <div class="card-value">98.1%</div>
            <div class="trend-green" style="margin-top: 6px;">↗ Optimal Rate</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="bento-card">
            <div class="card-title">Qdrant Vector Points</div>
            <div class="card-value">1,483</div>
            <div class="trend-blue" style="margin-top: 6px;">↗ Playbooks Cached</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
        <div class="bento-card">
            <div class="card-title">ONNX Inference Latency</div>
            <div class="card-value">12 ms</div>
            <div class="trend-green" style="margin-top: 6px;">↗ Real-Time Ready</div>
        </div>
    """, unsafe_allow_html=True)


# ROW 2: Main Wave Chart (Model ROC-AUC Execution) & Active Agent Diagnostics Table
row2_left, row2_right = st.columns([1.2, 1])

with row2_left:
    st.markdown("""
        <div class="bento-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
                <div>
                    <div class="card-title">Live Model ROC-AUC Score</div>
                    <div style="font-size: 1.7rem; font-weight: 700; color: #ffffff;">0.9420</div>
                    <div style="font-size: 0.8rem; color: #34d399; margin-top: 2px;">↗ LightGBM Ensemble Winner</div>
                </div>
            </div>
    """, unsafe_allow_html=True)

    chart_data = pd.DataFrame({
        'MON': [0.85, 0.89, 0.82, 0.91, 0.94, 0.88, 0.942]
    })
    st.line_chart(chart_data, color="#f43f5e",
                  height=185, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with row2_right:
    st.markdown("""
        <div class="bento-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                <span style="font-weight: 600; font-size: 0.9rem; color: #ffffff;">Active Agent Diagnostician</span>
                <span style="font-size: 0.78rem; color: #60a5fa; cursor: pointer;">View All Logs</span>
            </div>
    """, unsafe_allow_html=True)

    table_data = pd.DataFrame({
        "Agent Node": ["Apex_SRE", "PyEngine_DE", "Matrix_Prof", "Optuna_MLOps"],
        "Target": ["Nulls", "Type Cast", "Cardinality", "Ensemble"],
        "Status": ["Resolved", "Patched", "Validated", "Compiled"]
    })
    st.dataframe(table_data, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ROW 3: Bottom Summary Cards with Bar Charts (Estimator & Feature Stats)
row3_left, row3_right = st.columns(2)

with row3_left:
    st.markdown("""
        <div class="bento-card" style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div class="card-title">Estimator Accuracy Leaderboard</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #ffffff; margin: 4px 0;">92.4% Max</div>
                <div class="trend-green">↗ LightGBM Top Performer</div>
            </div>
    """, unsafe_allow_html=True)
    bar_data_1 = pd.DataFrame({"Bar": [84, 89, 91, 92.4]})
    st.bar_chart(bar_data_1, color="#d97706",
                 height=90, use_container_width=False)
    st.markdown("</div>", unsafe_allow_html=True)

with row3_right:
    st.markdown("""
        <div class="bento-card" style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div class="card-title">Top Feature Importance Weight</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #ffffff; margin: 4px 0;">0.42 (Tenure)</div>
                <div class="trend-blue">↗ GCS Stream Ingested</div>
            </div>
    """, unsafe_allow_html=True)
    bar_data_2 = pd.DataFrame({"Bar": [18, 28, 42]})
    st.bar_chart(bar_data_2, color="#8b5cf6",
                 height=90, use_container_width=False)
    st.markdown("</div>", unsafe_allow_html=True)
