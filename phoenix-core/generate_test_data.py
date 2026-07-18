import pandas as pd
import numpy as np
import os

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Create a dataset
data = {
    "trace_id": ["tx_fast_track_20", "tx_fast_track_21", "tx_fast_track_22", "tx_fast_track_23"],
    "cpu_usage": [99.9, 45.0, 99.9, 12.5],
    "sensor_value": [10.5, np.nan, 30.2, ""],  # Contains NaN and empty string
    "status": ["OK", "OK", "CRITICAL", "OK"]
}

df = pd.DataFrame(data)

# FIX: Convert to the explicit Pandas 'string' type (nullable).
# This prevents PyArrow from trying to force-convert the column to 'double'.
df['sensor_value'] = df['sensor_value'].astype("string")

# Save to Parquet
output_path = "data/default_ingestion.parquet"
df.to_parquet(output_path, index=False)

print(f"Successfully created mock Parquet file at: {output_path}")
