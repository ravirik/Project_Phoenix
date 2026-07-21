import asyncio
import json
import os
import logging
import logfire
import nats
from nats.errors import TimeoutError
from src.app import trigger_agentic_pipeline, initialize_app_runtime

logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_system_metrics()
logging.basicConfig(handlers=[logfire.LogfireLoggingHandler()])
logger = logging.getLogger(__name__)

NATS_URL = os.getenv("NATS_URL", "nats://phoenix-nats-bus:4222")
STREAM_NAME = "PHOENIX_STREAM"
SUBJECT = "telemetry.raw.*"
DURABLE_NAME = "janitor_processor_pool"


async def run_consumer():
    """Main consumer loop with pre-warmed runtime engine."""
    print("[Worker] Initializing AI Pull-Consumer framework...", flush=True)

    # Pre-warm connection pools and compile graph ONCE
    await initialize_app_runtime()

    nc = await nats.connect(NATS_URL)
    js = nc.jetstream()

    try:
        await js.add_stream(name=STREAM_NAME, subjects=[SUBJECT])
    except Exception:
        pass

    sub = await js.pull_subscribe(
        subject=SUBJECT,
        durable=DURABLE_NAME,
        stream=STREAM_NAME
    )

    print(
        f"[Worker] Agentic Pipeline active. Listening on {SUBJECT}...\n", flush=True)

    while True:
        try:
            messages = await sub.fetch(batch=1, timeout=5)
            for msg in messages:
                try:
                    raw_data = json.loads(msg.data.decode("utf-8"))
                    trace_id = raw_data.get('trace_id', 'unknown')
                    print(
                        f"\n--- INCOMING TELEMETRY DETECTED --- | Trace ID: {trace_id}", flush=True)

                    await trigger_agentic_pipeline(raw_data)
                    print(
                        f"[Worker] SUCCESS: Transaction {trace_id} finalized.", flush=True)

                except Exception as e:
                    print(
                        f"[Worker] ERROR: Pipeline execution failed: {e}", flush=True)

                finally:
                    await msg.ack()
                    print(
                        "[Worker] Message acknowledged (Queue purged).", flush=True)

        except TimeoutError:
            await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            break

    await nc.close()


if __name__ == "__main__":
    asyncio.run(run_consumer())
