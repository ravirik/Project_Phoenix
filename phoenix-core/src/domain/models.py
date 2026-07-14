from datetime import datetime
from typing import Dict, Any
from pydantic import BaseModel, Field, ConfigDict

class TelemetryLogPacket(BaseModel):
    """Immutable data contract validating incoming high-velocity system metric logs."""
    model_config = ConfigDict(frozen=True)

    trace_id: str = Field(..., description="Unique transaction trace signature tracking the packet lifecycle.")
    pipeline_id: str = Field(..., description="Target server cluster partition identifier.")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="System ingestion timestamp.")
    payload_uri: str = Field(..., description="Secure datalake reference index vector link.")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Volatile performance telemetry metrics.")

from typing_extensions import TypedDict

class OrchestratorState(TypedDict):
    """LangGraph shared memory state tracking the agentic remediation lifecycle."""
    trace_id: str
    pipeline_id: str
    status: str 
    retry_count: int
