import pytest

from verdict import stats


def test_weighted_kappa():
    assert stats.weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    near = stats.weighted_kappa([1, 2, 3, 4, 5, 5], [1, 2, 3, 4, 4, 5])
    far = stats.weighted_kappa([1, 2, 3, 4, 5, 5], [5, 4, 3, 2, 1, 1])
    assert 0.8 < near < 1 and far < 0


def test_cohen_kappa_and_spearman():
    assert stats.cohen_kappa(list("AABB"), list("AABB")) == 1.0
    assert stats.cohen_kappa(list("ABAB"), list("BABA")) < 0
    assert stats.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert stats.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    v = [1, 2, 3, 4, 5] * 10
    a, b = stats.bootstrap_mean(v, seed=1), stats.bootstrap_mean(v, seed=1)
    assert a == b and a["lo"] <= a["mean"] <= a["hi"] and a["n"] == 50
    assert stats.bootstrap_mean([])["mean"] is None


def test_paired_delta_only_uses_shared_items():
    d = stats.paired_delta({"x": 1, "y": 2, "z": 3}, {"x": 2, "y": 3})
    assert d["n"] == 2 and d["mean"] == pytest.approx(1.0)


def test_bradley_terry_orders_systems():
    m = [("a", "b", 1.0)] * 8 + [("b", "c", 1.0)] * 8 + [("a", "c", 1.0)] * 8 + [("a", "b", 0.5)] * 2
    r = stats.bradley_terry(["a", "b", "c"], m)
    assert r["a"] > r["b"] > r["c"]
    assert sum(r.values()) / 3 == pytest.approx(1000, abs=1)
