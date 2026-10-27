"""Promotion safety: only an audited, gate-passing candidate can become the champion."""
import asyncio
import json
import multiprocessing
import os

import numpy as np
import pandas as pd
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

import src.agents.crews as crews
import src.agents.nodes as nodes
from src.app import build_initial_state
from src.domain.models import OrchestratorState

PASSING = {"task_type": "CLASSIFICATION", "pr_auc": 0.6, "f1_score": 0.9, "roc_auc": 0.8,
           "fairness_score": 0.9, "inference_latency_ms": 0.1, "onnx_exported": True}


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    """Points the registry and ONNX paths at a temp dir and stubs the dashboard publisher."""
    monkeypatch.setattr(nodes, "REGISTRY_PATH", str(tmp_path / "model_registry.json"))
    monkeypatch.setattr(nodes, "CANDIDATE_ONNX_PATH", str(tmp_path / "candidate_model.onnx"))
    monkeypatch.setattr(nodes, "CHAMPION_ONNX_PATH", str(tmp_path / "pipeline_model.onnx"))

    async def no_publish(*args, **kwargs):
        return None

    monkeypatch.setattr(nodes, "publish_stage", no_publish)
    return tmp_path


class FakeAuditorCrew:
    built = 0

    def __init__(self, *args, **kwargs):
        FakeAuditorCrew.built += 1

    async def kickoff_async(self):
        class Result:
            raw = "release note"
        return Result()


def audit(state, monkeypatch):
    FakeAuditorCrew.built = 0
    monkeypatch.setattr(crews, "get_auditor_crew", FakeAuditorCrew)
    return asyncio.run(nodes.run_auditor_node(state))


def trained(metrics=PASSING, **extra):
    return {"trace_id": "tx", "status": "ML_EXECUTION_COMPLETED", "de_retry_count": 1,
            "mle_retry_count": 1, "candidate_metrics": dict(metrics), "metrics": {"psi_score": 3.1},
            "primary_score": metrics["pr_auc"], "candidate_task_type": "CLASSIFICATION",
            "gate_failures": nodes.evaluate_gates("CLASSIFICATION", metrics), **extra}


def registry(model_dir):
    return json.load(open(model_dir / "model_registry.json"))


class TestAuditorOwnsPromotion:
    def test_passing_candidate_is_promoted_by_the_auditor(self, model_dir, monkeypatch):
        (model_dir / "candidate_model.onnx").write_bytes(b"candidate")
        result = audit(trained(), monkeypatch)

        assert result["status"] == "AUDIT_PASSED_PROMOTED"
        assert (model_dir / "pipeline_model.onnx").read_bytes() == b"candidate"
        assert registry(model_dir)["active_champion"]["metrics"]["primary_score"] == 0.6
        assert FakeAuditorCrew.built == 1

    def test_success_on_the_last_attempt_is_audited_not_dead_lettered(self, model_dir, monkeypatch):
        (model_dir / "candidate_model.onnx").write_bytes(b"candidate")
        result = audit(trained(mle_retry_count=3, de_retry_count=3), monkeypatch)
        assert result["status"] == "AUDIT_PASSED_PROMOTED"

    def test_failing_gates_keep_the_champion(self, model_dir, monkeypatch):
        (model_dir / "candidate_model.onnx").write_bytes(b"candidate")
        (model_dir / "pipeline_model.onnx").write_bytes(b"champion")
        result = audit(trained({**PASSING, "fairness_score": None}), monkeypatch)

        assert result["status"] == "AUDIT_PASSED_CHALLENGER"
        assert (model_dir / "pipeline_model.onnx").read_bytes() == b"champion"
        assert registry(model_dir)["active_champion"] is None
        assert FakeAuditorCrew.built == 0

    def test_failed_training_is_dead_lettered_without_a_registry_write(self, model_dir, monkeypatch):
        result = audit({"trace_id": "tx", "status": "MLE_FAILED", "mle_retry_count": 3}, monkeypatch)
        assert result["status"] == "DLQ_ROUTED"
        assert not (model_dir / "model_registry.json").exists()


class TestGatesFailClosed:
    def test_passing_metrics_pass(self):
        assert nodes.evaluate_gates("CLASSIFICATION", PASSING) == []

    @pytest.mark.parametrize("missing", ["inference_latency_ms", "fairness_score", "onnx_exported"])
    def test_a_missing_measurement_fails(self, missing):
        metrics = {k: v for k, v in PASSING.items() if k != missing}
        assert nodes.evaluate_gates("CLASSIFICATION", metrics)

    def test_fairness_is_not_required_for_regression(self):
        metrics = {"rmse": 1.0, "inference_latency_ms": 0.1, "onnx_exported": True}
        assert nodes.evaluate_gates("REGRESSION", metrics) == []


class TestComputeFairness:
    def frame(self, groups):
        return pd.DataFrame({"category": groups})

    def test_worst_slice_wins(self):
        X = self.frame(["a"] * 40 + ["b"] * 40)
        y = np.array([1, 0] * 40)
        y_pred = np.concatenate([y[:40], 1 - y[40:]])  # perfect on a, all wrong on b
        assert nodes.compute_fairness(X, y, y_pred) == 0.0

    def test_one_row_slices_cannot_be_measured(self):
        X = pd.DataFrame({"category": [f"acc_{i}" for i in range(80)]})
        assert nodes.compute_fairness(X, np.zeros(80), np.zeros(80)) is None

    def test_a_single_slice_cannot_be_measured(self):
        assert nodes.compute_fairness(self.frame(["a"] * 80), np.zeros(80), np.zeros(80)) is None

    def test_missing_attribute_or_predictions_cannot_be_measured(self):
        X = pd.DataFrame({"other": ["a"] * 80})
        assert nodes.compute_fairness(X, np.zeros(80), np.zeros(80)) is None
        assert nodes.compute_fairness(self.frame(["a", "b"] * 40), np.zeros(80), None) is None


SCRIPT = """
import time
import numpy as np
import pandas as pd
X_val = pd.DataFrame({"category": ["a"] * 40 + ["b"] * 40})
y_val = np.array([0, 1] * 40)
y_pred = y_val.copy()
candidate_f1 = 1.0
candidate_pr_auc = 1.0
avg_inference_latency_ms = 0.1
"""


def run_sandbox(code):
    queue = multiprocessing.Queue()
    nodes._ml_os_sandbox(code, queue)
    return queue.get(timeout=5)


class TestSandboxDoesNotPromote:
    def test_sandbox_leaves_the_registry_alone(self, model_dir):
        result = run_sandbox(SCRIPT + f"\nopen({str(model_dir / 'candidate_model.onnx')!r}, 'wb').write(b'x')\n")

        assert result["status"] == "ML_TRAINING_SUCCESS"
        assert result["gate_failures"] == []
        assert result["metrics"]["fairness_score"] == 1.0
        assert not (model_dir / "model_registry.json").exists()
        assert not (model_dir / "pipeline_model.onnx").exists()

    def test_missing_candidate_is_not_replaced_by_the_champion(self, model_dir):
        (model_dir / "pipeline_model.onnx").write_bytes(b"champion")
        result = run_sandbox(SCRIPT)

        assert result["metrics"]["onnx_exported"] is False
        assert any("ONNX" in f for f in result["gate_failures"])
        assert not (model_dir / "candidate_model.onnx").exists()

    def test_script_fairness_is_ignored(self, model_dir):
        code = SCRIPT.replace('["a"] * 40 + ["b"] * 40', '["a"] * 80') + "\ncandidate_fairness = 1.0\n"
        result = run_sandbox(code)
        assert result["metrics"]["fairness_score"] is None


class TestReplayStartsFresh:
    def test_initial_state_resets_per_run_fields(self):
        state = build_initial_state({"trace_id": "tx", "pipeline_id": "p", "metrics": {"psi_score": 3.1}})
        for key in ("de_retry_count", "mle_retry_count", "de_escalation_count"):
            assert state[key] == 0
        for key in ("de_error_feedback", "mle_error_feedback", "mle_previous_code",
                    "promotion_status", "gate_failures", "candidate_metrics"):
            assert state[key] is None

    def test_replayed_thread_does_not_inherit_retry_counts(self):
        graph = StateGraph(OrchestratorState)

        async def fail_once(state):
            return {"mle_retry_count": state.get("mle_retry_count", 0) + 2,
                    "mle_error_feedback": "boom"}

        graph.add_node("work", fail_once)
        graph.set_entry_point("work")
        graph.add_edge("work", END)
        app = graph.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "tx"}}
        payload = {"trace_id": "tx", "pipeline_id": "p"}

        async def run_twice():
            await app.ainvoke(build_initial_state(payload), config)
            return await app.ainvoke(build_initial_state(payload), config)

        final = asyncio.run(run_twice())
        assert final["mle_retry_count"] == 2
