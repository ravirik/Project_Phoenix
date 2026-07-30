import json
import os
import nats
from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager
from src.infrastructure.path_safety import is_safe_data_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 🛡️ Default to localhost for terminal runs. Docker Compose will override this!
    nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
    try:
        # Prevent endless loop of gaierrors if NATS isn't reachable
        nc = await nats.connect(nats_url, max_reconnect_attempts=3)
        app.state.js = nc.jetstream()
        app.state.nc = nc
        print(f"✅ Gateway connected to NATS at {nats_url}")
    except Exception as e:
        print(f"⚠️ NATS Connection Error: {e}")
    yield
    if hasattr(app.state, "nc"):
        await app.state.nc.close()

app = FastAPI(title="Phoenix Ingestion Gateway", lifespan=lifespan)


@app.post("/api/v1/telemetry")
async def ingest_telemetry(request: Request):
    try:
        # 🛡️ Raw JSON (not a strict Pydantic model) so optional fields like
        # data_path survive - but that means WE are the validation layer now.
        payload = await request.json()

        if not payload.get("trace_id") or not payload.get("pipeline_id"):
            raise HTTPException(
                status_code=400, detail="trace_id and pipeline_id are required.")

        # 🛡️ Path-traversal / arbitrary-file guard: any caller-supplied
        # data_path or payload_uri MUST resolve inside data/, or we reject
        # the request outright.
        for field in ("data_path", "payload_uri"):
            value = payload.get(field)
            if value is not None and not is_safe_data_path(value):
                raise HTTPException(
                    status_code=400,
                    detail=f"'{field}' must resolve inside the data/ directory."
                )

        payload_bytes = json.dumps(payload).encode("utf-8")

        if hasattr(request.app.state, "js"):
            await request.app.state.js.publish("telemetry.raw.ingestion", payload_bytes)
            return {"status": "queued", "trace_id": payload.get("trace_id", "unknown")}
        else:
            raise Exception("NATS JetStream broker is not initialized.")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Broker Error: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "online"}
