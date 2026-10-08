"""Judge-quality calibration: re-judge human-labelled examples and measure agreement.

Label file format (JSONL, one label per line):
  pointwise: {"kind": "pointwise", "item_id", "rubric", "system_id", "response", "labeler", "score"}
  pairwise:  {"kind": "pairwise", "item_id", "rubric", "system_a", "system_b",
              "response_a", "response_b", "labeler", "winner": "A"|"B"|"tie"}
Responses are stored in the label file itself, so calibration is reproducible
without the database (this is what CI runs when a rubric changes).
"""
from __future__ import annotations

import itertools
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean, median

from . import stats
from .config import Registry, sha
from .judge import JudgeInput, judge_pairwise, judge_pointwise
from .llm import LLMClient


def load_labels(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _human_human(groups: dict, kind: str) -> dict:
    """Agreement between every pair of labelers on the examples they both labelled."""
    by_pair: dict[tuple[str, str], tuple[list, list]] = defaultdict(lambda: ([], []))
    for labels in groups.values():
        for (l1, v1), (l2, v2) in itertools.combinations(sorted(labels.items()), 2):
            by_pair[(l1, l2)][0].append(v1)
            by_pair[(l1, l2)][1].append(v2)
    ks, n = [], 0
    for a, b in by_pair.values():
        k = stats.weighted_kappa([int(x) for x in a], [int(y) for y in b]) if kind == "pointwise" else stats.cohen_kappa(a, b)
        if k is not None:
            ks.append(k)
            n += len(a)
    return {"kappa": mean(ks) if ks else None, "n_shared": n, "labeler_pairs": len(by_pair)}


def calibrate(reg: Registry, dataset_id: str, labels: list[dict], judges: list[str], concurrency: int = 6) -> dict:
    ds = reg.dataset(dataset_id)
    client = LLMClient(reg, run_id=None)

    def inp(item_id: str) -> JudgeInput:
        it = ds.item(item_id)
        return JudgeInput(it.question, ds.context_for(it), ds.reference_for(it), ds.style_guide)

    report: dict = {"dataset": dataset_id, "n_labels": len(labels), "pointwise": {}, "pairwise": {}}

    # ---- pointwise -------------------------------------------------------------
    pgroups: dict[tuple, dict[str, int]] = defaultdict(dict)  # (item, rubric, resp_hash) -> {labeler: score}
    presp: dict[tuple, str] = {}
    for l in labels:
        if l.get("kind", "pointwise") == "pointwise":
            k = (l["item_id"], l["rubric"], sha(l["response"]))
            pgroups[k][l["labeler"]] = int(l["score"])
            presp[k] = l["response"]
    keys = sorted(pgroups)
    tasks = [(k, j) for k in keys for j in judges]

    def run_point(t):  # noqa: ANN001, ANN202
        (item, rub, _), j = t
        return judge_pointwise(client, reg.judge_chain(j), reg.rubric(rub), inp(item), presp[t[0]]).score

    with ThreadPoolExecutor(concurrency) as pool:
        jscores = dict(zip(tasks, pool.map(run_point, tasks)))
    for rub in sorted({k[1] for k in keys}):
        rk = [k for k in keys if k[1] == rub]
        human = {k: round(median(pgroups[k].values())) for k in rk}
        res = {"n_examples": len(rk), "human_human": _human_human({k: pgroups[k] for k in rk}, "pointwise"), "judges": {}}
        panel: dict[tuple, list[float]] = defaultdict(list)
        for j in judges:
            pairs = [(human[k], jscores[(k, j)]) for k in rk if jscores[(k, j)] is not None]
            for k in rk:
                if jscores[(k, j)] is not None:
                    panel[k].append(jscores[(k, j)])
            res["judges"][j] = _agree(pairs)
        if len(judges) > 1:
            res["judges"]["panel"] = _agree([(human[k], round(mean(v))) for k, v in panel.items() if v])
        report["pointwise"][rub] = res

    # ---- pairwise --------------------------------------------------------------
    wgroups: dict[tuple, dict[str, str]] = defaultdict(dict)
    wresp: dict[tuple, tuple[str, str]] = {}
    for l in labels:
        if l.get("kind") == "pairwise":
            k = (l["item_id"], l["rubric"], sha([l["response_a"], l["response_b"]]))
            wgroups[k][l["labeler"]] = l["winner"]
            wresp[k] = (l["response_a"], l["response_b"])
    wkeys = sorted(wgroups)
    if wkeys:
        wtasks = [(k, j) for k in wkeys for j in judges]

        def run_pair(t):  # noqa: ANN001, ANN202
            (item, rub, _), j = t
            a, b = wresp[t[0]]
            r1 = judge_pairwise(client, reg.judge_chain(j), reg.rubric(rub), inp(item), a, b).winner
            r2 = judge_pairwise(client, reg.judge_chain(j), reg.rubric(rub), inp(item), b, a).winner
            swap = {"A": "B", "B": "A", "tie": "tie", None: None}
            if r1 is None or r2 is None:
                return r1 or swap[r2]
            return r1 if r1 == swap[r2] else "tie"  # inconsistent across orders -> tie

        with ThreadPoolExecutor(concurrency) as pool:
            jw = dict(zip(wtasks, pool.map(run_pair, wtasks)))
        for rub in sorted({k[1] for k in wkeys}):
            rk = [k for k in wkeys if k[1] == rub]
            human = {k: max(set(wgroups[k].values()), key=list(wgroups[k].values()).count) for k in rk}
            res = {"n_examples": len(rk), "human_human": _human_human({k: wgroups[k] for k in rk}, "pairwise"), "judges": {}}
            for j in judges:
                pairs = [(human[k], jw[(k, j)]) for k in rk if jw[(k, j)] is not None]
                res["judges"][j] = {
                    "n": len(pairs),
                    "accuracy": (sum(h == p for h, p in pairs) / len(pairs)) if pairs else None,
                    "kappa": stats.cohen_kappa([h for h, _ in pairs], [p for _, p in pairs]) if pairs else None,
                }
            report["pairwise"][rub] = res
    return report


def _agree(pairs: list[tuple[int, float]]) -> dict:
    if not pairs:
        return {"n": 0}
    h = [int(a) for a, _ in pairs]
    j = [int(b) for _, b in pairs]
    return {
        "n": len(pairs),
        "weighted_kappa": stats.weighted_kappa(h, j),
        "spearman": stats.spearman(h, j),
        "exact": sum(a == b for a, b in zip(h, j)) / len(h),
        "within_1": sum(abs(a - b) <= 1 for a, b in zip(h, j)) / len(h),
        "mean_judge_minus_human": mean(b - a for a, b in zip(h, j)),
    }


def check_gate(report: dict, thresholds: dict[str, float], judge: str = "panel") -> dict:
    """Fail if any rubric's weighted kappa for `judge` falls below its threshold."""
    checks = []
    for rub, thr in thresholds.items():
        jr = report["pointwise"].get(rub, {}).get("judges", {})
        k = (jr.get(judge) or next(iter(jr.values()), {})).get("weighted_kappa")
        checks.append({"rubric": rub, "kappa": k, "threshold": thr, "ok": k is not None and k >= thr})
    return {"passed": all(c["ok"] for c in checks), "checks": checks}
