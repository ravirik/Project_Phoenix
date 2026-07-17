import operator
from datetime import datetime
from typing import Dict, Any, Annotated
from pydantic import BaseModel, Field, ConfigDict
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage


class TelemetryLogPacket(BaseModel):
    """Immutable data contract validating incoming high-velocity system metric logs."""
    model_config = ConfigDict(frozen=True)

    trace_id: str = Field(
        ..., description="Unique transaction trace signature tracking the packet lifecycle.")
    pipeline_id: str = Field(...,
                             description="Target server cluster partition identifier.")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="System ingestion timestamp.")
    payload_uri: str = Field(...,
                             description="Secure datalake reference index vector link.")
    metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Volatile performance telemetry metrics.")


class OrchestratorState(TypedDict):
    """LangGraph shared memory state tracking the agentic remediation lifecycle."""
    trace_id: str
    pipeline_id: str
    status: str
    retry_count: int

    # REQUIRED FOR RAG: This tells LangGraph to append LLM/Tool messages
    # to a running list, rather than overwriting the state every node hop.
    messages: Annotated[list[AnyMessage], operator.add]
