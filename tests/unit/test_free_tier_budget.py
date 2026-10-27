"""Model-request budget: each incident should fit Gemini's free tier (20 requests per model per day)."""
import asyncio

import pytest

import src.agents.crews as crews
import src.agents.nodes as nodes
from src.agents.graph import route_entry


class TestRouteEntry:
    def test_psi_under_threshold_skips_the_pipeline(self):
        assert route_entry({"metrics": {"psi_score": 0.0076}}) == "nominal_exit"

    def test_psi_over_threshold_starts_the_janitor(self):
        assert route_entry({"metrics": {"psi_score": 3.12}}) == "de_crew"

    def test_structural_sentinel_starts_the_janitor(self):
        assert route_entry({"metrics": {"psi_score": 9.99}}) == "de_crew"

    def test_missing_psi_starts_the_janitor(self):
        assert route_entry({"metrics": {}}) == "de_crew"
        assert route_entry({}) == "de_crew"


class TestQuotaRetries:
    DAILY = ("429 RESOURCE_EXHAUSTED. Quota exceeded for metric: generate_content_free_tier_requests, "
             "limit: 20 ... 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'")

    def test_daily_quota_is_not_retried(self):
        assert crews.is_daily_quota_exhausted(Exception(self.DAILY))
        assert not crews._worth_retrying(Exception(self.DAILY))

    @pytest.mark.parametrize("message", [
        "503 UNAVAILABLE. This model is currently experiencing high demand.",
        "429 RESOURCE_EXHAUSTED 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier'",
    ])
    def test_transient_errors_are_retried(self, message):
        assert crews._worth_retrying(Exception(message))


class TestCrewShapes:
    def test_janitor_is_one_agent_without_tools(self):
        crew = crews.create_de_crew()
        assert len(crew.agents) == 1 and len(crew.tasks) == 1
        assert not crew.agents[0].tools

    def test_auditor_is_one_gemini_agent(self):
        crew = crews.get_auditor_crew("tx", 3.1, {"pr_auc": 0.4}, ["category"])
        assert len(crew.agents) == 1 and len(crew.tasks) == 1
        assert "gemini" in type(crew.agents[0].llm).__name__.lower()


def _audit(state, monkeypatch, tmp_path, crew_factory):
    async def no_publish(*args, **kwargs):
        return None

    monkeypatch.setattr(nodes, "REGISTRY_PATH", str(tmp_path / "model_registry.json"))
    monkeypatch.setattr(nodes, "CANDIDATE_ONNX_PATH", str(tmp_path / "candidate_model.onnx"))
    monkeypatch.setattr(nodes, "CHAMPION_ONNX_PATH", str(tmp_path / "pipeline_model.onnx"))
    (tmp_path / "candidate_model.onnx").write_bytes(b"candidate")
    monkeypatch.setattr(nodes, "publish_stage", no_publish)
    monkeypatch.setattr(crews, "get_auditor_crew", crew_factory)
    return asyncio.run(nodes.run_auditor_node(state))


PASSING = {"task_type": "CLASSIFICATION", "pr_auc": 0.4, "fairness_score": 0.9,
           "inference_latency_ms": 0.1, "onnx_exported": True}
AUDITED = {"trace_id": "tx", "status": "ML_EXECUTION_COMPLETED", "de_retry_count": 1,
           "mle_retry_count": 1, "metrics": {"psi_score": 3.1}, "candidate_metrics": PASSING,
           "primary_score": 0.4, "candidate_task_type": "CLASSIFICATION", "gate_failures": []}


class TestAuditorBudget:
    def test_challenger_makes_no_model_request(self, monkeypatch, tmp_path):
        def must_not_run(*args, **kwargs):
            raise AssertionError("auditor crew built for a rejected candidate")

        failing = {**AUDITED, "gate_failures": ["fairness not measured"]}
        result = _audit(failing, monkeypatch, tmp_path, must_not_run)
        assert result["status"] == "AUDIT_PASSED_CHALLENGER"

    def test_provider_error_does_not_crash_a_promotion(self, monkeypatch, tmp_path):
        class QuotaSpent:
            def __init__(self, *args, **kwargs):
                pass

            async def kickoff_async(self):
                raise RuntimeError(TestQuotaRetries.DAILY)

        result = _audit(AUDITED, monkeypatch, tmp_path, QuotaSpent)
        assert result["status"] == "AUDIT_PASSED_PROMOTED"
