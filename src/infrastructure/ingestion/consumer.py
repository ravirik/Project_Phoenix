import asyncio
import json
import os
import logging
import logfire
import nats
from nats.js import api
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
    """Main consumer loop with pre-warmed runtime engine and extended ack timeouts."""
    print("[Worker] Initializing AI Pull-Consumer framework...", flush=True)

    # Pre-warm connection pools and compile graph ONCE
    await initialize_app_runtime()

    nc = await nats.connect(NATS_URL)
    js = nc.jetstream()

    try:
        await js.add_stream(name=STREAM_NAME, subjects=[SUBJECT])
    except Exception:
        pass

    # 🛡️ HARDENED CONFIG: Give the AI agents up to 15 minutes (900s) to think & self-heal
    sub = await js.pull_subscribe(
        subject=SUBJECT,
        durable=DURABLE_NAME,
        stream=STREAM_NAME,
        config=api.ConsumerConfig(
            ack_wait=900,  # 15 minutes timeout prevents NATS redelivery during LLM calls
            max_deliver=3  # Max retries at the message broker level before dead-lettering
        )
    )

    print(
        f"[Worker] Agentic Pipeline active. Listening on {SUBJECT}...\n", flush=True)

    while True:
        try:
            messages = await sub.fetch(batch=1, timeout=5)
            for msg in messages:
                heartbeat_task = None
                try:
                    raw_data = json.loads(msg.data.decode("utf-8"))
                    trace_id = raw_data.get('trace_id', 'unknown')
                    print(
                        f"\n--- INCOMING TELEMETRY DETECTED --- | Trace ID: {trace_id}", flush=True)

                    # 🛡️ HEARTBEAT: Send a ping to NATS every 20 seconds while LLM/Sandbox runs
                    async def send_heartbeat():
                        while True:
                            await asyncio.sleep(20)
                            try:
                                await msg.in_progress()
                                logger.debug(
                                    f"Sent NATS heartbeat for trace {trace_id}")
                            except Exception:
                                break

                    heartbeat_task = asyncio.create_task(send_heartbeat())

                    # Execute the multi-agent self-healing graph
                    await trigger_agentic_pipeline(raw_data)

                    print(
                        f"[Worker] SUCCESS: Transaction {trace_id} finalized.", flush=True)

                except Exception as e:
                    print(
                        f"[Worker] ERROR: Pipeline execution failed: {e}", flush=True)
                    logfire.error(
                        "Pipeline execution failed: {error}", error=str(e))

                finally:
                    # Cancel the background heartbeat task
                    if heartbeat_task:
                        heartbeat_task.cancel()

                    try:
                        await msg.ack()
                        print(
                            "[Worker] Message acknowledged (Queue purged).", flush=True)
                    except Exception as ack_err:
                        print(
                            f"[Worker] Warning: Failed to acknowledge message: {ack_err}", flush=True)

        except TimeoutError:
            await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            break

    await nc.close()


if __name__ == "__main__":
    asyncio.run(run_consumer())
