import asyncio
import json
import os
import nats
from nats.errors import TimeoutError

# Import your LangGraph orchestrator
from src.app import trigger_agentic_pipeline

NATS_URL = os.getenv("NATS_URL", "nats://localhost:4222")

async def run_consumer():
    print("[Worker] Initializing AI Pull-Consumer framework...")
    nc = await nats.connect(NATS_URL)
    js = nc.jetstream()
    await js.add_stream(name="PHOENIX_STREAM", subjects=["telemetry.raw.*"])
    
    sub = await js.pull_subscribe(
        subject="telemetry.raw.*",
        durable="janitor_processor_pool",
        stream="PHOENIX_STREAM"
    )

    print("[Worker] Agentic Pipeline active. Listening for telemetry...")
    
    try:
        while True:
            try:
                messages = await sub.fetch(batch=1, timeout=2)
                for msg in messages:
                    raw_data = json.loads(msg.data.decode("utf-8"))
                    print(f"\n--- INCOMING TELEMETRY DETECTED ---")
                    print(f"Subject: {msg.subject} | Trace ID: {raw_data['trace_id']}")
                    
                    # 🚀 TRIGGER THE LANGGRAPH AI AGENTS
                    await trigger_agentic_pipeline(raw_data)
                    
                    await msg.ack()
                    print(f"--- TRANSACTION {raw_data['trace_id']} FINALIZED ---")
                    
            except TimeoutError:
                await asyncio.sleep(0.1) 
    except KeyboardInterrupt:
        print("\n[Worker] Stopping ingestion consumer thread gracefully...")
    finally:
        await nc.close()

if __name__ == "__main__":
    asyncio.run(run_consumer())
