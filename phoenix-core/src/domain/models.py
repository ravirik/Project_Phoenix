from typing import TypedDict, List, Optional, Any
from langchain_core.messages import BaseMessage
from pydantic import BaseModel


class TelemetryMetrics(BaseModel):
    cpu_utilization: float
    memory_spill: bool


class TelemetryLogPacket(BaseModel):
    trace_id: str
    pipeline_id: str
    payload_uri: str
    metrics: TelemetryMetrics


class OrchestratorState(TypedDict, total=False):
    trace_id: str
    pipeline_id: str
    data_path: str
    final_patch: Optional[str]
    status: str
    mle_output: Optional[str]

    # Model Registry & Evaluation Metrics
    candidate_f1: Optional[float]
    candidate_roc_auc: Optional[float]
    candidate_latency_ms: Optional[float]
    champion_f1: Optional[float]
    promotion_status: Optional[str]  # "PRODUCTION" (Champion) or "CHALLENGER"
