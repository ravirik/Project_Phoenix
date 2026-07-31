import asyncio
import json
import uuid
from nats.aio.client import Client as NATS


async def main():
    nc = NATS()
    # Connect to the cloud cluster via our port-forward tunnel
    await nc.connect("nats://localhost:4223")

    # The synthetic telemetry payload (NOW WITH PIPELINE ID)
    payload = {
        "event_id": "sim_84920",
        "trace_id": str(uuid.uuid4()),
        "pipeline_id": "credit_scoring_prod_v1",  # <-- The new missing key!
        "event_type": "data_drift_detected",
        "timestamp": "2026-07-31T12:00:00Z",
        "severity": "CRITICAL",
        "details": {
            "feature": "transaction_amount",
            "wasserstein_distance": 0.85,  # Massive drift!
            "threshold_limit": 0.30
        }
    }

    # Publish to the JetStream subject your worker is listening to
    print(
        f"Firing critical drift event to NATS: {payload['details']['feature']}")
    await nc.publish("telemetry.raw.drift", json.dumps(payload).encode())

    # Ensure the message is flushed to the network before closing
    await nc.flush()
    await nc.close()
    print("🚀 Event injected successfully!")

if __name__ == '__main__':
    asyncio.run(main())
