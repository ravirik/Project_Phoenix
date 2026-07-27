import asyncio
import json
import time
import uuid
import numpy as np
import pandas as pd
from scipy.stats import entropy
from nats.aio.client import Client as NATS

# --- CONFIGURATION ---
NATS_URL = "nats://127.0.0.1:4222"
TOPIC = "telemetry.raw.ecommerce"
DATA_PATH = "data/ecommerce_sales.parquet"
DRIFT_THRESHOLD_PSI = 0.20  # Standard enterprise threshold for significant drift

# --- MATHEMATICAL SURVEILLANCE FUNCTIONS ---


def calculate_psi(expected, actual, buckets=10):
    """Calculates the Population Stability Index (PSI) between two arrays."""
    def scale_range(input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
    breakpoints = scale_range(breakpoints, np.min(expected), np.max(expected))

    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

    # Avoid division by zero
    expected_percents = np.where(
        expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

    psi_value = np.sum((actual_percents - expected_percents)
                       * np.log(actual_percents / expected_percents))
    return psi_value


def calculate_kl_divergence(expected, actual, buckets=10):
    """Calculates the Kullback-Leibler Divergence."""
    hist_expected, bin_edges = np.histogram(
        expected, bins=buckets, density=True)
    hist_actual, _ = np.histogram(actual, bins=bin_edges, density=True)

    # Add epsilon to avoid log(0)
    epsilon = 1e-10
    hist_expected = hist_expected + epsilon
    hist_actual = hist_actual + epsilon

    return entropy(hist_actual, hist_expected)

# --- NATS PUBLISHER ---


async def trigger_swarm(psi_score, kl_div):
    """Fires a high-priority alert to the NATS Message Bus to wake up LangGraph."""
    nc = NATS()
    try:
        await nc.connect(NATS_URL)
        trace_id = f"tx_auto_drift_{str(uuid.uuid4())[:8]}"

        payload = {
            "trace_id": trace_id,
            "pipeline_id": "phoenix_ecom_pipeline",
            "payload_uri": DATA_PATH,
            "data_path": DATA_PATH,
            "metrics": {
                "cpu_utilization": 55.4,
                "memory_spill": False,
                "psi_score": round(psi_score, 4),
                "kl_divergence": round(kl_div, 4)
            }
        }

        print(
            f"\n[Auditor Watcher] 🚀 TRIGGERING SWARM! Broadcasting Trace: {trace_id}")
        await nc.publish(TOPIC, json.dumps(payload).encode())
        await nc.flush()

    except Exception as e:
        print(f"[Auditor Watcher] ❌ NATS Connection Error: {e}")
    finally:
        if nc.is_connected:
            await nc.close()

# --- CONTINUOUS MONITORING LOOP ---


async def run_surveillance():
    print("🛡️ [Auditor Watcher] Initializing Continuous Statistical Surveillance...")

    try:
        # 1. Load the baseline "Champion" data distribution
        baseline_df = pd.read_parquet(DATA_PATH)
        baseline_feature = baseline_df['transaction_amount'].values
        print(
            f"📊 Baseline distribution locked. ({len(baseline_feature)} samples)")
    except Exception as e:
        print(f"❌ Error loading baseline data: {e}")
        return

    iteration = 1
    while True:
        print(f"\n--- 📡 Surveillance Cycle {iteration} ---")

        # 2. Simulate an incoming data stream (Live Data)
        # Every 3rd cycle, we intentionally inject a massive data drift to test the system
        if iteration % 3 == 0:
            print(
                "⚠️ [Simulation] Injecting synthetic Covariate Shift (Macro-Economic Event)...")
            live_data = np.random.normal(loc=np.mean(baseline_feature) * 2.5,
                                         scale=np.std(baseline_feature),
                                         size=500)
        else:
            print("✅ [Simulation] Normal traffic flow...")
            live_data = np.random.normal(loc=np.mean(baseline_feature),
                                         scale=np.std(baseline_feature),
                                         size=500)

        # 3. Calculate O4 Drift Metrics
        psi = calculate_psi(baseline_feature, live_data)
        kl_div = calculate_kl_divergence(baseline_feature, live_data)

        print(f"📈 Population Stability Index (PSI): {psi:.4f}")
        print(f"📉 KL-Divergence: {kl_div:.4f}")

        # 4. Evaluate the Gate
        if psi > DRIFT_THRESHOLD_PSI:
            print(
                f"🚨 ALERT! PSI ({psi:.4f}) exceeds threshold ({DRIFT_THRESHOLD_PSI}). Data Drift Detected!")
            await trigger_swarm(psi, kl_div)
            print(
                "⏳ Swarm deployed. Pausing surveillance for 30 seconds to allow remediation...")
            # Wait for the LangGraph orchestrator to finish its job
            await asyncio.sleep(30)
        else:
            print("🛡️ Data stream stable. No action required.")
            await asyncio.sleep(5)  # Normal polling interval

        iteration += 1

if __name__ == "__main__":
    try:
        asyncio.run(run_surveillance())
    except KeyboardInterrupt:
        print("\n🛑 Surveillance terminated by user.")
