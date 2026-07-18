from typing import TypedDict, List, Any, Optional
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

# 1. Sub-model for the new metrics object
class TelemetryMetrics(BaseModel):
    cpu_utilization: float
    memory_spill: bool

# 2. Updated API Gateway Payload
class TelemetryLogPacket(BaseModel):
    trace_id: str
    pipeline_id: str
    payload_uri: str 
    metrics: TelemetryMetrics

# 3. LangGraph State
class OrchestratorState(TypedDict):
    trace_id: str
    pipeline_id: str
    status: str
    messages: List[BaseMessage]
    final_patch: Optional[str]
    retry_count: int
    data_path: Optional[str]
