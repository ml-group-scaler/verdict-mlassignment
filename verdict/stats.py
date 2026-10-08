"""Statistics: bootstrap CIs, paired deltas, agreement metrics, Bradley-Terry."""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def bootstrap_mean(values: Sequence[float], n: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    v = np.asarray([x for x in values if x is not None and not math.isnan(x)], dtype=float)
    if v.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n, v.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi), "n": int(v.size)}


def paired_delta(a: dict[str, float], b: dict[str, float], n: int = 1000, seed: int = 0) -> dict:
    """Bootstrap CI of mean(b - a) over items present in both (item-level pairing)."""
    keys = sorted(set(a) & set(b))
    return bootstrap_mean([b[k] - a[k] for k in keys], n=n, seed=seed)


def weighted_kappa(x: Sequence[int], y: Sequence[int], lo: int = 1, hi: int = 5, weights: str = "quadratic") -> float | None:
    """Cohen's kappa with linear/quadratic weights for ordinal scores."""
    if len(x) != len(y) or len(x) == 0:
        return None
    k = hi - lo + 1
    obs = np.zeros((k, k))
    for a, b in zip(x, y):
        obs[int(a) - lo, int(b) - lo] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    i, j = np.meshgrid(range(k), range(k), indexing="ij")
    w = ((i - j) ** 2 if weights == "quadratic" else np.abs(i - j)) / (k - 1) ** (2 if weights == "quadratic" else 1)
    denom = (w * exp).sum()
    if denom == 0:
        return 1.0 if (w * obs).sum() == 0 else 0.0
    return float(1 - (w * obs).sum() / denom)


def cohen_kappa(x: Sequence[str], y: Sequence[str]) -> float | None:
    """Unweighted Cohen's kappa for categorical labels (e.g. A/B/tie)."""
    if len(x) != len(y) or len(x) == 0:
        return None
    cats = sorted(set(x) | set(y))
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    pe = sum((list(x).count(c) / n) * (list(y).count(c) / n) for c in cats)
    return 1.0 if pe == 1 else float((po - pe) / (1 - pe))


def _rank(v: Sequence[float]) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    order = arr.argsort(kind="mergesort")
    ranks = np.empty(len(arr))
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        ranks[order[i: j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    rx, ry = _rank(x), _rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def bradley_terry(systems: list[str], matches: list[tuple[str, str, float]], iters: int = 200) -> dict[str, float]:
    """MM-algorithm Bradley-Terry strengths from (x, y, score_x) matches, where score_x
    is 1 win / 0.5 tie / 0 loss. Returned on an Elo-like scale centred at 1000.
    A tiny prior (one virtual tie with every opponent) keeps undefeated systems finite."""
    idx = {s: i for i, s in enumerate(systems)}
    k = len(systems)
    wins = np.full((k, k), 0.0)
    games = np.zeros((k, k))
    for x, y, sx in matches:
        i, j = idx[x], idx[y]
        wins[i, j] += sx
        wins[j, i] += 1 - sx
        games[i, j] += 1
        games[j, i] += 1
    prior = (np.ones((k, k)) - np.eye(k)) * 0.5
    wins, games = wins + prior, games + 2 * prior
    p = np.ones(k)
    for _ in range(iters):
        w = wins.sum(axis=1)
        denom = np.array([sum(games[i, j] / (p[i] + p[j]) for j in range(k) if j != i) for i in range(k)])
        p_new = w / np.maximum(denom, 1e-12)
        p_new /= np.exp(np.log(p_new).mean())
        if np.allclose(p_new, p, atol=1e-9):
            p = p_new
            break
        p = p_new
    return {s: float(1000 + 400 * math.log10(p[idx[s]])) for s in systems}


def percentile(values: Sequence[float], q: float) -> float | None:
    v = [x for x in values if x is not None]
    return float(np.percentile(v, q)) if v else None
