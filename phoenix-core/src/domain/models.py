from typing import TypedDict, Optional
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
    status: str

    # Execution Tracking
    final_patch: Optional[str]
    mle_output: Optional[str]

    # Model Registry Metrics
    candidate_f1: Optional[float]
    candidate_roc_auc: Optional[float]
    candidate_latency_ms: Optional[float]
    champion_f1: Optional[float]
    promotion_status: Optional[str]

    # --- Revalidation Loop Guardrails ---
    de_error_feedback: Optional[str]
    de_previous_code: Optional[str]
    de_retry_count: int

    mle_error_feedback: Optional[str]
    mle_previous_code: Optional[str]
    mle_retry_count: int
