"""Markdown rendering of run summaries / calibration / probes (PR comments, README)."""
from __future__ import annotations


def _f(x, nd: int = 2) -> str:  # noqa: ANN001
    return "—" if x is None else f"{x:.{nd}f}"


def _ci(d: dict) -> str:
    if not d or d.get("mean") is None:
        return "—"
    return f"{d['mean']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}]"


def run_markdown(summary: dict, title: str | None = None) -> str:
    s = summary
    rub = [r.split("@")[0] for r in s["rubrics"]]
    out = [f"## {title or 'Verdict run ' + s['run_id']}",
           f"Dataset `{s['dataset']}` · {s['items']} items · judges: {', '.join(f'`{j}`' for j in s['judges'])} · "
           f"ranked by {s['rank_by']}", ""]
    if "gate" in s:
        g = s["gate"]
        out += [f"### Merge gate: {'✅ PASSED' if g['passed'] else '❌ FAILED'}",
                f"`{g['candidate']}` vs baseline `{g['baseline']}` · mode `{g['mode']}` · margin {g['margin']}", "",
                "| check | Δ (candidate − baseline) [95% CI] | rule | result |", "|---|---|---|---|"]
        for c in g["checks"]:
            out.append(f"| {c['rubric']} | {_ci(c) if c.get('lo') is not None else _f(c.get('mean'))} | {c['rule']} | "
                       f"{'✅' if c['ok'] else '❌'} |")
        out.append("")
    pw = s.get("pairwise", {})
    pw_r = next(iter(pw), None)
    head = "| # | system | " + " | ".join(rub) + " | overall |" + (f" BT ({pw_r}) | win rate |" if pw_r else "")
    out += ["### Leaderboard (mean score 1–5, 95% bootstrap CI)", head, "|" + "---|" * (head.count("|") - 1)]
    for i, sys in enumerate(s["ranking"], 1):
        row = s["leaderboard"][sys]
        cells = [_ci(row[r]) for r in rub] + [_ci(row["overall"])]
        if pw_r:
            b = pw[pw_r].get(sys, {})
            cells += [f"{_f(b.get('rating'), 0)} [{_f(b.get('lo'), 0)}, {_f(b.get('hi'), 0)}]", _f(b.get("win_rate"))]
        out.append(f"| {i} | `{sys}` | " + " | ".join(cells) + " |")
    out += ["", "### Judge quality & bias controls",
            "| judge | position-flip rate | abstain rate | repair rate | fallback rate | self-preference Δ |",
            "|---|---|---|---|---|---|"]
    for j in s["judges"]:
        pb = s.get("position_bias", {}).get(j, {})
        h = s.get("judge_health", {}).get(j, {})
        sp = s.get("self_preference", {}).get(j, {})
        out.append(f"| `{j}` | {_f(pb.get('rate'))} ({pb.get('flips', 0)}/{pb.get('pairs', 0)}) | {_f(h.get('abstain_rate'))} | "
                   f"{_f(h.get('repair_rate'))} | {_f(h.get('fallback_rate'))} | {_f(sp.get('self_preference'))} |")
    ja = s.get("judge_agreement") or {}
    dis = s.get("disagreement", {})
    if ja:
        out.append(f"\nInter-judge agreement ({' vs '.join(ja['judges'])}): weighted κ {_f(ja.get('weighted_kappa'))}, "
                   f"Spearman {_f(ja.get('spearman'))} over {ja['n']} judgments. "
                   f"Panel disagreement (spread ≥ 2): {_f(dis.get('rate'))} of {dis.get('n_cases')} cases.")
    o = s["ops"]
    lj, lg = o["latency_ms"]["judge"], o["latency_ms"]["generate"]
    out += ["", "### Ops", f"- LLM calls: {o['llm_calls']} (cache hit rate {_f(o['cache_hit_rate'])}, errors {o['errors']})",
            f"- Judge latency p50/p99: {_f(lj['p50'], 0)} / {_f(lj['p99'], 0)} ms (n={lj['n']}); "
            f"generation p50/p99: {_f(lg['p50'], 0)} / {_f(lg['p99'], 0)} ms (n={lg['n']})",
            f"- Tokens in/out: {o['tokens_in']:,} / {o['tokens_out']:,} · cost ${o['cost_usd']:.4f} "
            f"(${o['cost_per_judgment_usd']:.6f}/judgment)"]
    return "\n".join(out) + "\n"


def calibration_markdown(rep: dict) -> str:
    out = [f"## Judge calibration vs human labels (`{rep['dataset']}`, {rep['n_labels']} labels)", "",
           "| rubric | judge | n | weighted κ | Spearman | exact | within ±1 | judge − human | human–human κ |",
           "|---|---|---|---|---|---|---|---|---|"]
    for rub, r in rep["pointwise"].items():
        hh = _f(r["human_human"]["kappa"])
        for j, m in r["judges"].items():
            out.append(f"| {rub} | `{j}` | {m.get('n', 0)} | {_f(m.get('weighted_kappa'))} | {_f(m.get('spearman'))} | "
                       f"{_f(m.get('exact'))} | {_f(m.get('within_1'))} | {_f(m.get('mean_judge_minus_human'))} | {hh} |")
    if rep["pairwise"]:
        out += ["", "| pairwise rubric | judge | n | accuracy | κ | human–human κ |", "|---|---|---|---|---|---|"]
        for rub, r in rep["pairwise"].items():
            for j, m in r["judges"].items():
                out.append(f"| {rub} | `{j}` | {m['n']} | {_f(m['accuracy'])} | {_f(m['kappa'])} | {_f(r['human_human']['kappa'])} |")
    return "\n".join(out) + "\n"


def probe_markdown(rep: dict) -> str:
    name = rep["probe"]
    if name == "planted":
        out = [f"## Planted-error probe ({rep['n_cases']} known-good/known-bad responses)", "",
               "| judge | accuracy (good≥4, bad≤2) | discrimination (good > bad) |", "|---|---|---|"]
        for j, m in rep["judges"].items():
            out.append(f"| `{j}` | {_f(m['accuracy'])} | {_f(m['discrimination'])} |")
    else:
        out = [f"## {name.capitalize()} probe ({rep['n_cases']} cases)", "",
               "| judge | mean score Δ | % of scores that rose | by rubric |", "|---|---|---|---|"]
        for j, m in rep["judges"].items():
            br = ", ".join(f"{k}: {v:+.2f}" for k, v in m["by_rubric"].items())
            out.append(f"| `{j}` | {_f(m['mean_delta'])} | {_f(m['pct_increased'])} | {br} |")
    return "\n".join(out) + "\n"
