import time

from verdict import llm
from verdict.judge import JudgeInput, judge_pointwise
from verdict.llm import CircuitBreaker, LLMClient
from verdict.runner import run_sync
from verdict.spec import RunRequest


def test_fallback_serves_when_primary_is_down(reg):
    llm._breakers.clear()
    r = judge_pointwise(LLMClient(reg), reg.judge_chain("mock-judge-flaky"), reg.rubric("safety"),
                        JudgeInput("Should I share my OTP?"), "Never share your OTP. Next step: enter it in the app.")
    assert r.score == 5.0 and r.served_model == "mock-judge-strict" and not r.abstain


def test_all_models_down_abstains_with_errors(reg):
    r = judge_pointwise(LLMClient(reg), ["mock-judge-down"], reg.rubric("safety"), JudgeInput("q"), "a")
    assert r.abstain and r.served_model is None and "mock-judge-down" in r.error


def test_circuit_breaker_opens_and_recovers(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    b = CircuitBreaker(threshold=2, cooldown_s=60)
    b.record(False)
    assert b.allow()
    b.record(False)
    assert not b.allow()          # open: fail fast
    now[0] += 61
    assert b.allow()              # half-open trial after cooldown
    b.record(True)
    assert b.failures == 0 and b.allow()


def test_breaker_skips_dead_model_without_calling_it(reg):
    llm._breakers.clear()
    client = LLMClient(reg, use_cache=False)
    for _ in range(2):
        judge_pointwise(client, reg.judge_chain("mock-judge-flaky"), reg.rubric("safety"), JudgeInput("q"), "a")
    assert not llm._breakers["mock-judge-down"].allow()


def test_run_reports_fallback_rate(reg):
    llm._breakers.clear()
    _, s = run_sync(RunRequest(profile="mock", sample=4, judges=["mock-judge-flaky", "mock-judge-biased"]))
    h = s["judge_health"]["mock-judge-flaky"]
    assert h["fallback_rate"] == 1.0 and h["abstain_rate"] == 0
    assert h["served_by"] == {"mock-judge-strict": h["judgments"]}
    assert s["judge_health"]["mock-judge-biased"]["fallback_rate"] == 0.0
    assert "mock-judge-flaky" not in s["self_preference"]  # fallback-served judgments are excluded
