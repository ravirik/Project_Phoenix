import pandas as pd
import json

# Create an absolute mess of a dataset
data = {
    "account_id": ["1001", "1002", "NULL", "1004", "1005", "UNKNOWN"],
    "transaction_value": ["$150.00", "200.50", "NaN", "error", "-50.25", "99.99"],
    # Force the integers to be strings so Parquet accepts the mixed "yes" value
    "is_fraud": ["0", "1", "0", "yes", "1", "0"],
    # Convert dicts to JSON strings so Parquet can store them
    "weird_json_column": [json.dumps({"a": 1}), json.dumps({"b": 2}), "None", "{invalid}", "{}", "None"]
}

df = pd.DataFrame(data)

# Save the chaos dataset into the data folder
# Note: Ensure the path is correct relative to where you run the script!
data_path = "data/chaos_ingestion.parquet"
df.to_parquet(data_path, index=False)

print(f"✅ Chaos data successfully created at {data_path}!")