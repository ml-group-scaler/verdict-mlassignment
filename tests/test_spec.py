import pytest

from verdict.spec import RunRequest, SpecError, resolve


def test_profile_resolution_pins_rubric_versions(reg):
    s = resolve(reg, RunRequest(profile="mock", sample=5))
    assert len(s.item_ids) == 5 and all("@" in r for r in s.rubrics)
    assert len(s.pairs) == 3  # all pairs of 3 systems
    assert s.item_ids == resolve(reg, RunRequest(profile="mock", sample=5)).item_ids  # seeded


def test_vs_baseline_and_external_outputs(reg):
    outs = {"cand": {"ret-001": "a", "ret-002": "b"}, "base": {"ret-001": "c", "ret-002": "d", "ret-003": "e"}}
    s = resolve(reg, RunRequest(profile="mock", systems=["base", "cand"], external_outputs=outs,
                                pairwise={"rubrics": ["helpfulness"], "pairs": "vs_baseline", "baseline": "base"},
                                gate={"baseline": "base", "candidate": "cand", "rubrics": ["helpfulness"]}))
    assert s.item_ids == ["ret-001", "ret-002"]
    assert s.pairs == [("base", "cand")] and set(s.external_systems) == {"base", "cand"}


@pytest.mark.parametrize("req", [
    RunRequest(profile="does-not-exist"),
    RunRequest(profile="mock", systems=["ghost"]),
    RunRequest(profile="mock", judges=["ghost"]),
    RunRequest(profile="mock", gate={"baseline": "mock-good", "candidate": "ghost"}),
])
def test_invalid_requests(reg, req):
    with pytest.raises(SpecError):
        resolve(reg, req)


def test_missing_api_key_fails_fast(reg, monkeypatch):
    from verdict.runner import create_run

    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(SpecError, match="NVIDIA_API_KEY"):
        create_run(RunRequest(profile="ci"))
