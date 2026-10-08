"""Aggregation of a run into the leaderboard + bias/quality/ops metrics + gate verdict."""
from __future__ import annotations

from collections import defaultdict
from statistics import mean

import numpy as np
from sqlalchemy import select

from . import db, stats
from .config import GateCfg, Registry
from .judge import combine_orders, positional_to_score
from .spec import RunSpec

DISAGREE_SPREAD = 2  # judges disagree if their scores differ by >= 2 points


def _load(run_id: str) -> tuple[list[db.Generation], list[db.Judgment], list[db.LLMCall]]:
    with db.session() as s:
        gens = list(s.scalars(select(db.Generation).where(db.Generation.run_id == run_id)))
        judg = list(s.scalars(select(db.Judgment).where(db.Judgment.run_id == run_id)))
        calls = list(s.scalars(select(db.LLMCall).where(db.LLMCall.run_id == run_id)))
    return gens, judg, calls


def item_scores(judgments: list[db.Judgment]) -> tuple[dict, dict]:
    """item_score[system][rubric][item] = mean over judges; per_judge[(s, r, i)][judge] = score."""
    per_judge: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for j in judgments:
        if j.kind == "pointwise" and j.score is not None:
            per_judge[(j.system_id, j.rubric_id, j.item_id)][j.judge_id] = j.score
    item_score: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for (s, r, i), by_judge in per_judge.items():
        item_score[s][r][i] = mean(by_judge.values())
    return item_score, per_judge


def pairwise_matches(judgments: list[db.Judgment]) -> tuple[dict, dict]:
    """Combine both presentation orders per (judge, rubric, item, pair).
    Returns matches[rubric] = list of (x, y, score_x, item) averaged over judges, and flips[judge] = (n_flip, n_pairs)."""
    orders: dict[tuple, dict[str, float | None]] = defaultdict(dict)
    for j in judgments:
        if j.kind != "pairwise":
            continue
        x, y = sorted((j.system_a, j.system_b))
        key = (j.judge_id, j.rubric_id, j.item_id, x, y)
        orders[key]["x_as_a" if j.system_a == x else "x_as_b"] = positional_to_score(j.winner, j.system_a == x)
    per_item: dict[tuple, list[float]] = defaultdict(list)
    flips: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    flip_cases = []
    for (judge, rub, item, x, y), o in orders.items():
        s1, s2 = o.get("x_as_a"), o.get("x_as_b")
        sx, flipped = combine_orders(s1, s2)
        if s1 is not None and s2 is not None:
            flips[judge][1] += 1
            flips[judge][0] += int(flipped)
            if flipped:
                flip_cases.append({"judge": judge, "rubric": rub, "item": item, "x": x, "y": y})
        if sx is not None:
            per_item[(rub, item, x, y)].append(sx)
    matches: dict[str, list[tuple[str, str, float, str]]] = defaultdict(list)
    for (rub, item, x, y), vals in per_item.items():
        matches[rub].append((x, y, mean(vals), item))
    return matches, {"rates": {k: {"flips": v[0], "pairs": v[1], "rate": (v[0] / v[1]) if v[1] else None}
                               for k, v in flips.items()}, "cases": flip_cases}


def bt_with_ci(systems: list[str], matches: list[tuple[str, str, float, str]], n_boot: int, seed: int) -> dict:
    point = stats.bradley_terry(systems, [(x, y, s) for x, y, s, _ in matches])
    items = sorted({m[3] for m in matches})
    by_item: dict[str, list] = defaultdict(list)
    for m in matches:
        by_item[m[3]].append(m)
    rng = np.random.default_rng(seed)
    boots = defaultdict(list)
    for _ in range(n_boot):
        sample = [m for i in rng.choice(items, size=len(items), replace=True) for m in by_item[i]]
        for s, v in stats.bradley_terry(systems, [(x, y, sc) for x, y, sc, _ in sample], iters=100).items():
            boots[s].append(v)
    out = {}
    for s in systems:
        lo, hi = np.quantile(boots[s], [0.025, 0.975]) if boots[s] else (None, None)
        wins = [sc if x == s else 1 - sc for x, y, sc, _ in matches if s in (x, y)]
        out[s] = {"rating": point[s], "lo": float(lo) if lo is not None else None,
                  "hi": float(hi) if hi is not None else None,
                  "win_rate": mean(wins) if wins else None, "matches": len(wins)}
    return out


def evaluate_gate(gate: GateCfg, item_score: dict, n_boot: int, seed: int) -> dict:
    checks = []
    for r in gate.rubrics:
        d = stats.paired_delta(item_score.get(gate.baseline, {}).get(r, {}),
                               item_score.get(gate.candidate, {}).get(r, {}), n=n_boot, seed=seed)
        if d["mean"] is None:
            checks.append({"rubric": r, "ok": False, "rule": "no paired items", **d})
            continue
        if gate.mode == "noninferiority":
            ok = d["lo"] >= -gate.margin
            rule = f"CI lower bound {d['lo']:+.2f} >= -{gate.margin}"
        else:
            ok = not (d["hi"] < 0 and d["mean"] < -gate.margin)
            rule = f"not (CI upper {d['hi']:+.2f} < 0 and mean {d['mean']:+.2f} < -{gate.margin})"
        checks.append({"rubric": r, "ok": bool(ok), "rule": rule, **d})
    for r, floor in gate.min_scores.items():
        vals = list(item_score.get(gate.candidate, {}).get(r, {}).values())
        m = mean(vals) if vals else None
        checks.append({"rubric": f"{r} (minimum)", "ok": m is not None and m >= floor,
                       "rule": f"candidate mean {m if m is None else round(m, 2)} >= {floor}",
                       "mean": m, "lo": None, "hi": None, "n": len(vals)})
    return {"passed": all(c["ok"] for c in checks), "baseline": gate.baseline, "candidate": gate.candidate,
            "mode": gate.mode, "margin": gate.margin, "checks": checks}


def summarize(run_id: str, spec: RunSpec, reg: Registry) -> dict:
    gens, judgments, calls = _load(run_id)
    rubric_ids = [r.split("@")[0] for r in spec.rubrics]
    item_score, per_judge = item_scores(judgments)
    nb, seed = spec.bootstrap, spec.seed

    # Leaderboard (pointwise) -------------------------------------------------
    board = {}
    for s in spec.systems:
        row = {}
        for r in rubric_ids:
            row[r] = stats.bootstrap_mean(list(item_score[s][r].values()), n=nb, seed=seed)
        overall_items = [mean(item_score[s][r][i] for r in rubric_ids)
                         for i in spec.item_ids if all(i in item_score[s][r] for r in rubric_ids)]
        row["overall"] = stats.bootstrap_mean(overall_items, n=nb, seed=seed)
        board[s] = row

    # Per-judge means + panel disagreement --------------------------------------
    per_judge_means: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    acc: dict[tuple, list[float]] = defaultdict(list)
    for (s, r, i), by_j in per_judge.items():
        for j, v in by_j.items():
            acc[(j, s, r)].append(v)
    for (j, s, r), vals in acc.items():
        per_judge_means[j][s][r] = mean(vals)
    multi = {k: v for k, v in per_judge.items() if len(v) >= 2}
    disagreements = [{"system": s, "rubric": r, "item": i, "scores": v, "spread": max(v.values()) - min(v.values())}
                     for (s, r, i), v in multi.items() if max(v.values()) - min(v.values()) >= DISAGREE_SPREAD]
    disagreements.sort(key=lambda d: -d["spread"])
    judge_corr = {}
    if len(spec.judges) >= 2:
        j1, j2 = spec.judges[:2]
        pairs = [(v[j1], v[j2]) for v in multi.values() if j1 in v and j2 in v]
        if pairs:
            judge_corr = {
                "judges": [j1, j2], "n": len(pairs),
                "weighted_kappa": stats.weighted_kappa([int(a) for a, _ in pairs], [int(b) for _, b in pairs]),
                "spearman": stats.spearman([a for a, _ in pairs], [b for _, b in pairs]),
            }

    # Self-preference: judge's score on same-family systems vs the rest of the panel ---
    # (only judgments served by the judge's primary model — a fallback model has another family)
    fallback_served = {(x.system_id, x.rubric_id, x.item_id, x.judge_id) for x in judgments
                       if x.kind == "pointwise" and x.served_model and x.judge_id in reg.judges
                       and x.served_model != reg.judges[x.judge_id].model}
    self_pref = {}
    if len(spec.judges) >= 2:
        for j in spec.judges:
            fam = reg.model_for_judge(j).family
            same, other = [], []
            for (s, r, i), by_j in multi.items():
                if j not in by_j or (s, r, i, j) in fallback_served:
                    continue
                others = [v for k, v in by_j.items() if k != j]
                delta = by_j[j] - mean(others)
                (same if reg.family_of_system(s) == fam else other).append(delta)
            if same:
                self_pref[j] = {"family": fam, "delta_same_family": mean(same),
                                "delta_other": mean(other) if other else None,
                                "self_preference": mean(same) - (mean(other) if other else 0.0), "n_same": len(same)}

    # Pairwise -----------------------------------------------------------------
    matches, flips = pairwise_matches(judgments)
    pairwise = {}
    for rub, ms in matches.items():
        involved = sorted({m[0] for m in ms} | {m[1] for m in ms})
        pairwise[rub] = bt_with_ci(involved, ms, min(nb, 300), seed)

    # Ranking -----------------------------------------------------------------
    full_bt = [r for r, v in pairwise.items() if set(v) == set(spec.systems)]
    if full_bt:
        key = {s: mean(pairwise[r][s]["rating"] for r in full_bt) for s in spec.systems}
        rank_by = f"Bradley-Terry ({', '.join(full_bt)})"
    else:
        key = {s: board[s]["overall"]["mean"] or 0 for s in spec.systems}
        rank_by = "pointwise overall mean"
    ranking = sorted(spec.systems, key=lambda s: -key[s])

    # Programmatic checks (format) ------------------------------------------------
    max_words = None
    for r in spec.rubrics:
        max_words = reg.rubric(r).programmatic_checks.get("max_words", max_words)
    programmatic = {}
    for s in spec.systems:
        texts = [g.response for g in gens if g.system_id == s]
        wc = [len(t.split()) for t in texts]
        programmatic[s] = {
            "mean_words": mean(wc) if wc else None,
            "pct_within_max_words": (sum(w <= max_words for w in wc) / len(wc)) if (wc and max_words) else None,
            "max_words": max_words,
            "generation_errors": sum(1 for g in gens if g.system_id == s and g.error),
        }

    # Judge health -------------------------------------------------------------
    health = {}
    for j in spec.judges:
        js = [x for x in judgments if x.judge_id == j]
        if js:
            primary = reg.judges[j].model if j in reg.judges else None
            served = defaultdict(int)
            for x in js:
                served[x.served_model or "none (abstained)"] += 1
            answered = [x for x in js if x.served_model]
            health[j] = {"judgments": len(js), "abstain_rate": sum(x.abstain for x in js) / len(js),
                         "repair_rate": sum(x.repaired for x in js) / len(js),
                         "fallback_rate": (sum(x.served_model != primary for x in answered) / len(answered)) if answered else None,
                         "served_by": dict(served)}

    # Ops (latency / cost / cache) ------------------------------------------------
    real = [c for c in calls if not c.cache_hit and not c.error]
    n_judg = sum(1 for x in judgments if not x.abstain) or 1
    ops = {
        "llm_calls": len(calls), "cache_hits": sum(c.cache_hit for c in calls),
        "cache_hit_rate": (sum(c.cache_hit for c in calls) / len(calls)) if calls else None,
        "errors": sum(1 for c in calls if c.error),
        "latency_ms": {p: {"p50": stats.percentile([c.latency_ms for c in real if c.purpose == p], 50),
                           "p99": stats.percentile([c.latency_ms for c in real if c.purpose == p], 99),
                           "n": sum(1 for c in real if c.purpose == p)} for p in ("generate", "judge")},
        "tokens_in": sum(c.input_tokens for c in calls if not c.cache_hit),
        "tokens_out": sum(c.output_tokens for c in calls if not c.cache_hit),
        "cost_usd": sum(c.cost_usd for c in calls),
        "cost_per_judgment_usd": sum(c.cost_usd for c in calls if c.purpose in ("judge", "repair")) / n_judg,
    }

    summary = {
        "run_id": run_id, "dataset": spec.dataset, "items": len(spec.item_ids), "systems": spec.systems,
        "judges": spec.judges, "rubrics": spec.rubrics, "pairwise_rubrics": spec.pairwise_rubrics,
        "ranking": ranking, "rank_by": rank_by, "leaderboard": board,
        "per_judge": {j: dict(v) for j, v in per_judge_means.items()}, "pairwise": pairwise,
        "position_bias": flips["rates"], "flip_cases": flips["cases"][:200],
        "disagreement": {"rate": (len(disagreements) / len(multi)) if multi else None,
                         "n_cases": len(multi), "top": disagreements[:200]},
        "judge_agreement": judge_corr, "self_preference": self_pref,
        "programmatic": programmatic, "judge_health": health, "ops": ops,
    }
    if spec.gate:
        summary["gate"] = evaluate_gate(spec.gate, item_score, nb, seed)
    return summary


def item_detail(run_id: str, item_id: str, reg: Registry) -> dict:
    run = db.get_run(run_id)
    if run is None:
        raise KeyError(run_id)
    spec = RunSpec(**run.spec)
    ds = reg.dataset(spec.dataset)
    it = ds.item(item_id)
    gens, judgments, _ = _load(run_id)
    responses = {g.system_id: {"response": g.response, "error": g.error} for g in gens if g.item_id == item_id}
    point = [{"system": j.system_id, "judge": j.judge_id, "rubric": j.rubric_id, "score": j.score,
              "rationale": j.rationale, "abstain": j.abstain, "repaired": j.repaired, "served_model": j.served_model}
             for j in judgments if j.item_id == item_id and j.kind == "pointwise"]
    pair = [{"a": j.system_a, "b": j.system_b, "judge": j.judge_id, "rubric": j.rubric_id, "winner": j.winner,
             "rationale": j.rationale, "abstain": j.abstain, "served_model": j.served_model}
            for j in judgments if j.item_id == item_id and j.kind == "pairwise"]
    return {"item": it.model_dump(), "context": ds.context_for(it), "reference": ds.reference_for(it),
            "responses": responses, "pointwise": point, "pairwise": pair}
