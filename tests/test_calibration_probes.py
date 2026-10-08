from verdict import calibration, probes
from tests.conftest import FIXTURES


def test_calibration_on_synthetic_fixture(reg):
    labels = calibration.load_labels(FIXTURES / "SYNTHETIC_test_labels.jsonl")
    assert all(l["synthetic"] for l in labels)
    rep = calibration.calibrate(reg, "support_v1", labels, ["mock-judge-strict", "mock-judge-biased"])
    pw = rep["pointwise"]["helpfulness"]
    assert pw["n_examples"] == 45
    assert pw["human_human"]["kappa"] > 0.7  # the fixture has ~20% off-by-one disagreement
    assert pw["judges"]["mock-judge-strict"]["weighted_kappa"] > 0.8
    assert "panel" in pw["judges"]
    assert rep["pairwise"]["helpfulness"]["judges"]["mock-judge-strict"]["accuracy"] == 1.0
    gate = calibration.check_gate(rep, {"helpfulness": 0.6}, judge="mock-judge-strict")
    assert gate["passed"]
    assert not calibration.check_gate(rep, {"helpfulness": 0.99}, judge="mock-judge-strict")["passed"]


def test_injection_probe_catches_susceptible_judge(reg):
    cases = [{"item_id": p["item_id"], "rubric": "helpfulness", "response": p["response"]} for p in reg.dataset("support_v1").planted]
    rep = probes.injection(reg, "support_v1", ["mock-judge-strict", "mock-judge-biased"], cases)
    assert rep["judges"]["mock-judge-strict"]["mean_delta"] == 0
    assert rep["judges"]["mock-judge-biased"]["mean_delta"] > 1


def test_planted_probe(reg):
    rep = probes.planted(reg, "support_v1", ["mock-judge-strict"])
    assert rep["judges"]["mock-judge-strict"]["discrimination"] == 1.0
    assert rep["judges"]["mock-judge-strict"]["accuracy"] >= 0.8
