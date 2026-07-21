import json
import os
import nats
from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager
from src.domain.models import TelemetryLogPacket


@asynccontextmanager
async def lifespan(app: FastAPI):
    nats_url = os.getenv("NATS_URL", "nats://phoenix-nats-bus:4222")
    nc = await nats.connect(nats_url)
    js = nc.jetstream()
    app.state.js = js
    app.state.nc = nc
    yield
    await nc.close()


app = FastAPI(title="Phoenix Ingestion Gateway", lifespan=lifespan)


@app.post("/api/v1/telemetry")
async def ingest_telemetry(packet: TelemetryLogPacket, request: Request):
    try:
        payload = packet.model_dump()
        payload_bytes = json.dumps(payload).encode("utf-8")
        await request.app.state.js.publish("telemetry.raw.ingestion", payload_bytes)
        return {"status": "queued", "trace_id": packet.trace_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Broker Error: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "online"}
