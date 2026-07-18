import json
import nats
from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager
from src.domain.models import TelemetryLogPacket

# Global state to hold the NATS connection


class AppState:
    js = None
    nc = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Connect to NATS on startup
    nc = await nats.connect("nats://localhost:4222")
    js = nc.jetstream()

    app.state.js = js
    app.state.nc = nc
    yield
    # Close connection on shutdown
    await nc.close()

app = FastAPI(title="Phoenix Ingestion Gateway", lifespan=lifespan)


@app.post("/api/v1/telemetry")
async def ingest_telemetry(packet: TelemetryLogPacket, request: Request):
    """
    Acts as an event producer. Validates payload and pushes to NATS.
    """
    try:
        # 1. Prepare data
        payload = packet.model_dump()
        payload_bytes = json.dumps(payload).encode("utf-8")

        # 2. Publish to NATS stream
        # This matches the 'telemetry.raw.*' subscription in your consumer
        await request.app.state.js.publish("telemetry.raw.ingestion", payload_bytes)

        return {"status": "queued", "trace_id": packet.trace_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Broker Error: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "online"}
