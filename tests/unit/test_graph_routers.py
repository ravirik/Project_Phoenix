"""Routing rules for the self-healing loop: retry, escalate, or hand off to the auditor."""
import pytest

from src.agents.graph import build_phoenix_runtime, route_de_execution, route_mle_execution


class TestRouteDeExecution:
    def test_success_goes_to_memory_updater(self):
        assert route_de_execution({"status": "SUCCESS", "de_retry_count": 1}) == "memory_updater"

    @pytest.mark.parametrize("retries", [0, 1, 2])
    def test_failure_retries_janitor_crew_under_limit(self, retries):
        state = {"status": "DE_FAILED", "de_retry_count": retries}
        assert route_de_execution(state) == "de_crew"

    def test_failure_stops_retrying_at_limit(self):
        assert route_de_execution({"status": "DE_FAILED", "de_retry_count": 3}) == "auditor_node"

    def test_missing_retry_count_counts_as_zero(self):
        assert route_de_execution({"status": "DE_FAILED"}) == "de_crew"

    def test_unknown_status_goes_to_auditor(self):
        assert route_de_execution({"status": "DE_CODE_GENERATED"}) == "auditor_node"


class TestRouteMleExecution:
    def test_completed_training_goes_to_auditor(self):
        assert route_mle_execution({"status": "ML_EXECUTION_COMPLETED"}) == "auditor_node"

    @pytest.mark.parametrize("retries", [0, 1, 2])
    def test_failure_retries_scientist_crew_under_limit(self, retries):
        state = {"status": "MLE_FAILED", "mle_retry_count": retries}
        assert route_mle_execution(state) == "mle_crew"

    @pytest.mark.parametrize("escalations", [0, 1])
    def test_exhausted_retries_escalate_back_to_janitor(self, escalations):
        state = {"status": "MLE_FAILED", "mle_retry_count": 3, "de_escalation_count": escalations}
        assert route_mle_execution(state) == "de_crew"

    def test_exhausted_retries_and_escalations_go_to_auditor(self):
        state = {"status": "MLE_FAILED", "mle_retry_count": 3, "de_escalation_count": 2}
        assert route_mle_execution(state) == "auditor_node"

    def test_unknown_status_goes_to_auditor(self):
        assert route_mle_execution({"status": "MLE_CODE_GENERATED"}) == "auditor_node"


def test_runtime_compiles_with_expected_nodes():
    graph = build_phoenix_runtime().get_graph()
    assert {"de_crew", "executor", "memory_updater", "mle_crew", "ml_executor",
            "auditor_node"} <= set(graph.nodes)
