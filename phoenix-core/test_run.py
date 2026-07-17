import asyncio
from src.app import trigger_agentic_pipeline


async def main():
    print("🚀 Firing mock NATS payload into Phoenix Core...")

    # We use "tx_fast_track_20" to trigger the Auditor's SDE tactical override
    mock_payload = {
        "trace_id": "tx_fast_track_20",
        "pipeline_id": "parquet_ingestion_cluster_01"
    }

    await trigger_agentic_pipeline(mock_payload)

if __name__ == "__main__":
    asyncio.run(main())
