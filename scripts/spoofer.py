import pandas as pd
import numpy as np
import time
import shutil
import os

# --- CONFIGURATION ---
BASELINE_PATH = "data/ecommerce_sales.parquet"
LIVE_PATH = "data/live_stream.parquet"

# ==========================================
# 🧪 CHOOSE YOUR CHAOS MODE
# ==========================================
# Options:
# "HEALTHY"               -> Normal traffic. System stays peacefully asleep.
# "DATA_DRIFT"            -> Covariate shift. Wakes the Janitor & Scientist.
# "CONCEPT_DRIFT"         -> Prior probability shift. Wakes the Scientist directly.
# "STRUCTURAL_CORRUPTION" -> Schema breaks. Forces Janitor to write Python patches!
# "FULL_CHAOS"            -> Everything breaks at once. The ultimate stress test.
# ==========================================
CHAOS_MODE = "HEALTHY"  # Change this to test different scenarios


def main():
    if not os.path.exists(BASELINE_PATH):
        print(f"Error: Baseline dataset not found at {BASELINE_PATH}.")
        return

    print(f"Loading baseline dataset from {BASELINE_PATH}...")
    df_baseline = pd.read_parquet(BASELINE_PATH)

    os.makedirs("data", exist_ok=True)
    shutil.copy(BASELINE_PATH, LIVE_PATH)
    print(f"Initialized live stream. Mode: [{CHAOS_MODE}]")

    drift_intensity = 0.0
    loop_counter = 0

    try:
        while True:
            # 1. STATELESS WINDOW: Sample 1000 rows. Memory remains flat.
            df_live = df_baseline.sample(n=1000, replace=True).copy()
            loop_counter += 1

            # Gradually increase intensity up to 100%
            drift_intensity = min(drift_intensity + 0.10, 1.0)

            # -------------------------------------------------------------
            # 🟢 HEALTHY MODE
            # -------------------------------------------------------------
            if CHAOS_MODE == "HEALTHY":
                # Do nothing! Let the clean data pass through.
                pass

            # -------------------------------------------------------------
            # 🚨 STRUCTURAL CORRUPTION (Forces Code Generation)
            # -------------------------------------------------------------
            if CHAOS_MODE in ["STRUCTURAL_CORRUPTION", "FULL_CHAOS"]:
                # Rename a critical column so the schema validator fails
                if 'transaction_amount' in df_live.columns:
                    df_live = df_live.rename(
                        columns={'transaction_amount': 'amt_corrupted_typo'})

                # Inject invalid string types into a float/int column
                if 'customer_age' in df_live.columns:
                    df_live.loc[df_live.sample(
                        frac=0.2).index, 'customer_age'] = "UNKNOWN"

                # Drop random values to create NaNs
                if 'category' in df_live.columns:
                    df_live.loc[df_live.sample(
                        frac=0.3).index, 'category'] = None

            # -------------------------------------------------------------
            # 📈 DATA DRIFT (Statistical Shift)
            # -------------------------------------------------------------
            if CHAOS_MODE in ["DATA_DRIFT", "FULL_CHAOS"]:
                # Skew the transaction amounts
                amount_col = 'transaction_amount' if 'transaction_amount' in df_live.columns else 'amt_corrupted_typo'

                if amount_col in df_live.columns:
                    noise_mean = 300 * drift_intensity
                    df_live[amount_col] = pd.to_numeric(
                        df_live[amount_col], errors='coerce')
                    df_live[amount_col] += np.random.normal(
                        loc=noise_mean, scale=50.0, size=len(df_live))
                    df_live[amount_col] = df_live[amount_col].clip(lower=1.0)

                # Skew the categories safely
                if 'category' in df_live.columns:
                    scramble_mask = np.random.rand(
                        len(df_live)) < (0.5 * drift_intensity)

                    # 🛠️ THE FIX: Check if the column is a number. If so, inject a numeric outlier!
                    if df_live['category'].dtype in ['float64', 'int64', 'float32', 'int32']:
                        # Massive unseen numeric ID
                        df_live.loc[scramble_mask, 'category'] = 9999.0
                    else:
                        df_live.loc[scramble_mask,
                                    'category'] = 'High_Risk_Electronics'

            # -------------------------------------------------------------
            # 🧠 CONCEPT DRIFT (Target Rule Changes)
            # -------------------------------------------------------------
            if CHAOS_MODE in ["CONCEPT_DRIFT", "FULL_CHAOS"]:
                target_col = 'is_fraud' if 'is_fraud' in df_live.columns else 'status'
                amount_col = 'transaction_amount' if 'transaction_amount' in df_live.columns else 'amt_corrupted_typo'

                if target_col in df_live.columns and amount_col in df_live.columns:
                    # Convert to numeric safely in case Structural Corruption messed it up
                    temp_amt = pd.to_numeric(
                        df_live[amount_col], errors='coerce')

                    # New reality: Transactions between $20 and $60 are suddenly highly fraudulent
                    concept_mask = (temp_amt > 20) & (temp_amt < 60)
                    flip_probability = np.random.rand(
                        len(df_live)) < (0.85 * drift_intensity)
                    final_mask = concept_mask & flip_probability

                    df_live.loc[final_mask, target_col] = 1

            # -------------------------------------------------------------
            # 💾 FLUSH TO DISK
            # -------------------------------------------------------------
            df_live.to_parquet(LIVE_PATH, index=False)

            if CHAOS_MODE == "HEALTHY":
                print(
                    f"[{CHAOS_MODE}] Pushed clean live window. Loop: {loop_counter}. Waiting 10s...")
            else:
                print(
                    f"[{CHAOS_MODE}] Pushed live window. Intensity: {drift_intensity:.2f}. Waiting 10s...")

            time.sleep(10)

    except KeyboardInterrupt:
        print("\nSpoofer stopped by user. Cleaning up...")


if __name__ == "__main__":
    main()
