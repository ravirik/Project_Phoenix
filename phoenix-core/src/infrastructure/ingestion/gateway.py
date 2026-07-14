import json
import os
from fastapi import FastAPI, HTTPException, status
import nats
from src.domain.models import TelemetryLogPacket

app = FastAPI(title="Project Phoenix Ingestion Gateway", version="1.0.0")
NATS_URL = os.getenv("NATS_URL", "nats://localhost:4222")

@app.post("/api/v1/telemetry", status_code=status.HTTP_202_ACCEPTED)
async def ingest_telemetry(packet: TelemetryLogPacket):
    """
    Ingress point catching high-velocity streaming log matrices.
    Validates schemas instantly via Pydantic and forwards them into the NATS bus.
    """
    try:
        nc = await nats.connect(NATS_URL)
        js = nc.jetstream()
        
        # Serialize the verified domain model package
        payload_bytes = json.dumps(packet.model_dump(), default=str).encode("utf-8")
        
        # Publish into the dedicated telemetry subject channel
        subject = f"telemetry.raw.{packet.pipeline_id}"
        ack = await js.publish(subject, payload_bytes)
        
        await nc.close()
        return {"status": "QUEUED", "stream": ack.stream, "sequence": ack.seq}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion Engine Transport Fault: {str(e)}"
        )
