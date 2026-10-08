from unittest import mock

import pytest

from verdict import db, llm
from verdict.config import BudgetCfg
from verdict.llm import BudgetExceeded
from verdict.runner import create_run, execute, run_sync
from verdict.spec import RunRequest


@pytest.fixture(scope="module")
def mock_run():
    return run_sync(RunRequest(profile="mock", sample=12))


def test_ranking_and_ci_separation(mock_run):
    _, s = mock_run
    assert s["ranking"] == ["mock-good", "mock-medium", "mock-bad"]
    good, bad = s["leaderboard"]["mock-good"]["helpfulness"], s["leaderboard"]["mock-bad"]["helpfulness"]
    assert good["lo"] > bad["hi"], "CIs of best and worst system should not overlap"
    bt = s["pairwise"]["helpfulness"]
    assert bt["mock-good"]["rating"] > bt["mock-medium"]["rating"] > bt["mock-bad"]["rating"]


def test_bias_controls_detect_the_biased_judge(mock_run):
    _, s = mock_run
    assert s["position_bias"]["mock-judge-strict"]["rate"] == 0
    assert s["position_bias"]["mock-judge-biased"]["rate"] > 0.05
    assert s["judge_health"]["mock-judge-biased"]["abstain_rate"] == 0  # repair path recovers unparseable replies


def test_gate_passes_for_improvement(mock_run):
    _, s = mock_run
    assert s["gate"]["passed"] is True


def test_gate_fails_for_regression():
    _, s = run_sync(RunRequest(profile="mock", sample=12,
                               gate={"baseline": "mock-good", "candidate": "mock-bad", "rubrics": ["helpfulness"]}))
    assert s["gate"]["passed"] is False


def test_rerun_is_fully_cached_and_identical(mock_run):
    _, s1 = mock_run
    _, s2 = run_sync(RunRequest(profile="mock", sample=12))
    assert s2["ops"]["cache_hit_rate"] == 1.0
    assert s2["leaderboard"] == s1["leaderboard"] and s2["ranking"] == s1["ranking"]


def test_budget_cap_fails_the_run_cleanly():
    """With the cache disabled, a 5-call budget must abort the run and mark it failed."""
    run = create_run(RunRequest(profile="mock", sample=3, budget=BudgetCfg(max_calls=5, max_usd=0)))
    orig = llm.LLMClient.__init__

    def no_cache(self, *a, **kw):
        orig(self, *a, **kw)
        self.use_cache = False

    with mock.patch.object(llm.LLMClient, "__init__", no_cache), pytest.raises(BudgetExceeded):
        execute(run.id)
    r = db.get_run(run.id)
    assert r.status == "failed" and "budget" in r.error
