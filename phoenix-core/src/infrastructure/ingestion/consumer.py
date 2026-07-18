import asyncio
import json
import os
import nats
from nats.errors import TimeoutError
from src.app import trigger_agentic_pipeline

# Configuration
NATS_URL = os.getenv("NATS_URL", "nats://localhost:4222")
STREAM_NAME = "PHOENIX_STREAM"
SUBJECT = "telemetry.raw.*"
DURABLE_NAME = "janitor_processor_pool"

async def run_consumer():
    """Main consumer loop with robust error handling and ACK management."""
    print("[Worker] Initializing AI Pull-Consumer framework...")
    
    # 1. Connect to NATS
    nc = await nats.connect(NATS_URL)
    js = nc.jetstream()

    # 2. Setup Stream (Idempotent)
    await js.add_stream(name=STREAM_NAME, subjects=[SUBJECT])
    
    # 3. Pull Subscription
    sub = await js.pull_subscribe(
        subject=SUBJECT,
        durable=DURABLE_NAME,
        stream=STREAM_NAME
    )

    print(f"[Worker] Agentic Pipeline active. Listening for telemetry on {SUBJECT}...")
    
    try:
        while True:
            try:
                # Fetch message with timeout
                messages = await sub.fetch(batch=1, timeout=5)
                
                for msg in messages:
                    raw_data = None
                    try:
                        raw_data = json.loads(msg.data.decode("utf-8"))
                        trace_id = raw_data.get('trace_id', 'unknown')
                        
                        print(f"\n--- INCOMING TELEMETRY DETECTED ---")
                        print(f"Subject: {msg.subject} | Trace ID: {trace_id}")
                        
                        # 🚀 TRIGGER THE LANGGRAPH AI AGENTS
                        await trigger_agentic_pipeline(raw_data)
                        
                        print(f"[Worker] SUCCESS: Transaction {trace_id} finalized.")
                        
                    except json.JSONDecodeError:
                        print("[Worker] ERROR: Malformed JSON. Discarding message.")
                    except Exception as e:
                        # Catching AI/Agent errors so they don't crash the consumer
                        print(f"[Worker] ERROR: Pipeline execution failed: {e}")
                    finally:
                        # THE GHOST-BUSTER:
                        # ALWAYS ACK, even if the pipeline failed.
                        # This removes the message from the NATS queue, 
                        # preventing it from being re-delivered infinitely.
                        await msg.ack()
                        print(f"[Worker] Message acknowledged (Queue purged).")
                        
            except TimeoutError:
                # Keep loop alive during idle periods
                await asyncio.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\n[Worker] Stopping ingestion consumer thread gracefully...")
    finally:
        await nc.close()

if __name__ == "__main__":
    asyncio.run(run_consumer())
