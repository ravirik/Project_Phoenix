import asyncio
import json
import os
import nats
from nats.errors import TimeoutError

NATS_URL = os.getenv("NATS_URL", "nats://localhost:4222")

async def run_consumer():
    print("[Worker] Initializing Pull-Consumer backpressure framework...")
    nc = await nats.connect(NATS_URL)
    js = nc.jetstream()

    # 1. Bind the persistent JetStream storage bucket
    await js.add_stream(name="PHOENIX_STREAM", subjects=["telemetry.raw.*"])
    
    # 2. Attach a durable pull subscriber
    sub = await js.pull_subscribe(
        subject="telemetry.raw.*",
        durable="janitor_processor_pool",
        stream="PHOENIX_STREAM"
    )

    print("[Worker] Ingestion pipe active. Listening under pull backpressure...")
    
    try:
        while True:
            try:
                # Request 1 message, waiting up to 2 seconds
                messages = await sub.fetch(batch=1, timeout=2)
                for msg in messages:
                    raw_data = json.loads(msg.data.decode("utf-8"))
                    print(f"\n[Worker Captured] Subject: {msg.subject} | Seq: {msg.metadata.sequence.stream}")
                    print(f"   ↳ Trace ID: {raw_data['trace_id']}")
                    
                    # Simulate Phase 1 No-Op processing latency
                    await asyncio.sleep(0.02) 
                    
                    # ACK the message to clear it from the NATS disk buffer
                    await msg.ack()
                    print("   ↳ Transaction finalized cleanly. [ACK Sent]")
                    
            except TimeoutError:
                # Expected behavior when the queue is empty. Back off and poll again.
                await asyncio.sleep(0.1) 
    except KeyboardInterrupt:
        print("\n[Worker] Stopping ingestion consumer thread gracefully...")
    finally:
        await nc.close()

if __name__ == "__main__":
    asyncio.run(run_consumer())
