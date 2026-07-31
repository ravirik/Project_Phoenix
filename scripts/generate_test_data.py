import pandas as pd
import numpy as np
import os

os.makedirs('data', exist_ok=True)

# 1. Standard Ingestion Data
df_default = pd.DataFrame({
    'trace_id': [f'tx_fast_{i}' for i in range(1000)],
    'cpu_usage': np.random.uniform(10.0, 99.9, 1000),
    'sensor_value': np.random.uniform(5.0, 50.0, 1000),
    'status': np.random.choice(['OK', 'CRITICAL'], 1000)
})
df_default.loc[5:10, 'sensor_value'] = np.nan
df_default.to_parquet('data/default_ingestion.parquet', index=False)

# 2. Chaos Data (Heavy Corruption)
df_chaos = pd.DataFrame({
    'trace_id': [f'tx_chaos_{i}' for i in range(1000)],
    'cpu_usage': np.random.choice(['99.9', 'ERR', '', None], 1000),
    'sensor_value': np.random.choice(['10.5', 'NaN', 'CORRUPT', None], 1000),
    'status': np.random.choice(['OK', 'FAIL', '1', '0'], 1000)
})
df_chaos['cpu_usage'] = df_chaos['cpu_usage'].astype('string')
df_chaos['sensor_value'] = df_chaos['sensor_value'].astype('string')
df_chaos['status'] = df_chaos['status'].astype('string')
df_chaos.to_parquet('data/chaos_ingestion.parquet', index=False)

# 3. E-Commerce Sales Data
df_ecom = pd.DataFrame({
    'account_id': [f'acc_{i}' for i in range(1000)],
    'transaction_amount': np.random.uniform(10.0, 500.0, 1000),
    'category': np.random.choice(['electronics', 'clothing', 'home', 'beauty'], 1000),
    'is_fraud': np.random.choice([0, 1], 1000, p=[0.9, 0.1])
})
df_ecom.to_parquet('data/ecommerce_sales.parquet', index=False)

print('✅ All 3 test datasets generated in data/')
