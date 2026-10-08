"""Verdict UI (Streamlit). Talks to the Verdict API over REST only.

Run:  VERDICT_API_URL=http://localhost:8000 streamlit run ui/app.py
"""
from __future__ import annotations

import os
import random
import time

import altair as alt
import httpx
import pandas as pd
import streamlit as st

API = os.environ.get("VERDICT_API_URL", "http://localhost:8000").rstrip("/")
KEY = os.environ.get("VERDICT_API_KEY", "")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

st.set_page_config(page_title="Verdict — LLM-as-judge leaderboard", page_icon="⚖️", layout="wide")


def api(method: str, path: str, **kw):  # noqa: ANN003, ANN201
    try:
        r = httpx.request(method, f"{API}{path}", headers={"X-API-Key": KEY} if KEY else {}, timeout=60, **kw)
    except httpx.HTTPError as e:
        st.error(f"Cannot reach the Verdict API at {API}: {e}")
        st.stop()
    if r.status_code == 404:
        return None
    if r.status_code >= 400:
        st.error(f"API error {r.status_code}: {r.text[:500]}")
        st.stop()
    return r.json()


def ci(d: dict | None) -> str:
    return "—" if not d or d.get("mean") is None else f"{d['mean']:.2f}  [{d['lo']:.2f}, {d['hi']:.2f}]"


def pick_run(key: str, only_done: bool = True) -> dict | None:
    runs = api("GET", "/v1/runs?limit=100") or []
    runs = [r for r in runs if r["status"] == "done"] if only_done else runs
    if not runs:
        st.info("No runs yet — start one from **New run**.")
        return None
    labels = {f"{r['id']} · {r['profile'] or 'custom'} · {r['items']} items · {str(r['created_at'])[:16]}"
              + (f" · {r['label']}" if r.get("label") else ""): r["id"] for r in runs}
    choice = st.selectbox("Run", list(labels), key=key)
    return api("GET", f"/v1/runs/{labels[choice]}")


def interval_chart(df: pd.DataFrame, x_title: str, domain: list[float] | None = None) -> alt.Chart:
    """Dot + 95% CI rule per system (single series, so no legend)."""
    base = alt.Chart(df).encode(y=alt.Y("system:N", sort=list(df["system"]), title=None))
    scale = alt.Scale(domain=domain) if domain else alt.Scale(zero=False)
    rule = base.mark_rule(strokeWidth=2, color=SERIES[0]).encode(
        x=alt.X("lo:Q", title=x_title, scale=scale), x2="hi:Q")
    dot = base.mark_circle(size=90, color=SERIES[0], opacity=1).encode(
        x="value:Q", tooltip=["system", alt.Tooltip("value:Q", format=".2f"),
                              alt.Tooltip("lo:Q", format=".2f", title="CI low"), alt.Tooltip("hi:Q", format=".2f", title="CI high")])
    return (rule + dot).properties(height=max(120, 42 * len(df)))


# ---------------------------------------------------------------------------- pages
def page_leaderboard() -> None:
    st.title("🏆 Leaderboard")
    run = pick_run("lb")
    if not run:
        return
    s = run["summary"]
    rub = [r.split("@")[0] for r in s["rubrics"]]
    st.caption(f"Dataset `{s['dataset']}` · {s['items']} items · judges {', '.join(s['judges'])} · ranked by **{s['rank_by']}** · "
               f"config hash `{run['config_hash']}`")
    if "gate" in s:
        g = s["gate"]
        (st.success if g["passed"] else st.error)(
            f"Merge gate {'PASSED ✅' if g['passed'] else 'FAILED ❌'} — `{g['candidate']}` vs `{g['baseline']}` "
            f"({g['mode']}, margin {g['margin']})")
        st.dataframe(pd.DataFrame([{"check": c["rubric"], "Δ mean": c.get("mean"), "CI low": c.get("lo"), "CI high": c.get("hi"),
                                    "rule": c["rule"], "ok": "✅" if c["ok"] else "❌"} for c in g["checks"]]),
                     hide_index=True, width="stretch")

    pw = s.get("pairwise", {})
    pw_r = next(iter(pw), None)
    rows = []
    for i, sys in enumerate(s["ranking"], 1):
        b = s["leaderboard"][sys]
        row = {"#": i, "system": sys, **{r: ci(b[r]) for r in rub}, "overall": ci(b["overall"])}
        if pw_r and sys in pw[pw_r]:
            row[f"BT ({pw_r})"] = round(pw[pw_r][sys]["rating"])
            row["win rate"] = round(pw[pw_r][sys]["win_rate"] or 0, 2)
        prog = s["programmatic"].get(sys, {})
        if prog.get("pct_within_max_words") is not None:
            row[f"≤{prog['max_words']} words"] = f"{prog['pct_within_max_words']:.0%}"
        rows.append(row)
    st.subheader("Systems")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Cells: mean judge score (1–5) with 95% bootstrap CI over items. Overlapping CIs ⇒ no significant difference.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Overall pointwise score (95% CI)**")
        df = pd.DataFrame([{"system": x, "value": s["leaderboard"][x]["overall"]["mean"],
                            "lo": s["leaderboard"][x]["overall"]["lo"], "hi": s["leaderboard"][x]["overall"]["hi"]}
                           for x in s["ranking"] if s["leaderboard"][x]["overall"]["mean"] is not None])
        if not df.empty:
            st.altair_chart(interval_chart(df, "mean score (1–5)", [1, 5]), width="stretch")
    with c2:
        if pw_r:
            st.markdown(f"**Bradley-Terry rating — {pw_r} (95% CI)**")
            df = pd.DataFrame([{"system": x, "value": v["rating"], "lo": v["lo"], "hi": v["hi"]}
                               for x, v in sorted(pw[pw_r].items(), key=lambda kv: -kv[1]["rating"])])
            st.altair_chart(interval_chart(df, "rating (Elo-like, 1000 = average)"), width="stretch")

    st.subheader("Judge quality & bias controls")
    jrows = []
    for j in s["judges"]:
        pb = s["position_bias"].get(j, {})
        h = s["judge_health"].get(j, {})
        sp = s["self_preference"].get(j, {})
        jrows.append({"judge": j, "position-flip rate": pb.get("rate"), "flips/pairs": f"{pb.get('flips', 0)}/{pb.get('pairs', 0)}",
                      "abstain rate": h.get("abstain_rate"), "JSON-repair rate": h.get("repair_rate"),
                      "fallback rate": h.get("fallback_rate"), "served by": ", ".join(f"{m}: {n}" for m, n in (h.get("served_by") or {}).items()),
                      "self-preference Δ": sp.get("self_preference"), "same-family judgments": sp.get("n_same")})
    st.dataframe(pd.DataFrame(jrows), hide_index=True, width="stretch")
    ja, dis = s.get("judge_agreement") or {}, s.get("disagreement", {})
    if ja:
        st.caption(f"Inter-judge weighted κ = {ja['weighted_kappa']:.2f}, Spearman = {ja['spearman'] if ja['spearman'] is None else round(ja['spearman'], 2)} "
                   f"({ja['n']} judgments). Panel disagreement (score spread ≥ 2): "
                   f"{(dis.get('rate') or 0):.1%} of {dis.get('n_cases')} cases — see **Explore run**.")
    per_judge = s.get("per_judge", {})
    if per_judge:
        with st.expander("Per-judge mean scores"):
            st.dataframe(pd.DataFrame([{"judge": j, "system": x, **v} for j, d in per_judge.items() for x, v in d.items()]),
                         hide_index=True, width="stretch")

    o = s["ops"]
    st.subheader("Run cost & latency")
    k = st.columns(5)
    k[0].metric("LLM calls", o["llm_calls"])
    k[1].metric("Cache hit rate", f"{(o['cache_hit_rate'] or 0):.0%}")
    k[2].metric("Judge p50 / p99", f"{(o['latency_ms']['judge']['p50'] or 0):.0f} / {(o['latency_ms']['judge']['p99'] or 0):.0f} ms")
    k[3].metric("Tokens in / out", f"{o['tokens_in']:,} / {o['tokens_out']:,}")
    k[4].metric("Cost", f"${o['cost_usd']:.4f}")


def page_explore() -> None:
    st.title("🔍 Explore run — per-example drill-down")
    run = pick_run("ex")
    if not run:
        return
    items = api("GET", f"/v1/runs/{run['id']}/items") or []
    f1, f2, f3 = st.columns([1, 1, 2])
    only_dis = f1.toggle("Judge disagreements only")
    only_flip = f2.toggle("Position flips only")
    cats = sorted({i["category"] for i in items})
    cat = f3.multiselect("Category", cats)
    shown = [i for i in items if (not only_dis or i["judge_disagreement"]) and (not only_flip or i["position_flip"])
             and (not cat or i["category"] in cat)]
    if not shown:
        st.info("No items match the filters.")
        return
    lab = {f"{'⚠️ ' if i['judge_disagreement'] else ''}{'🔀 ' if i['position_flip'] else ''}{i['item_id']} — {i['question'][:90]}": i["item_id"]
           for i in shown}
    item_id = lab[st.selectbox(f"Item ({len(shown)} shown · ⚠️ judges disagree · 🔀 verdict flipped when order swapped)", list(lab))]
    d = api("GET", f"/v1/runs/{run['id']}/items/{item_id}")
    st.markdown(f"**Customer:** {d['item']['question']}")
    with st.expander("Policy context & reference notes"):
        st.text(d["context"])
        st.text(d["reference"])

    judge_models = {j: v["model_id"] for j, v in (api("GET", "/v1/configs") or {}).get("judges", {}).items()}
    pt = pd.DataFrame(d["pointwise"])
    systems = list(d["responses"])
    cols = st.columns(min(3, len(systems)) or 1)
    for n, sysid in enumerate(systems):
        with cols[n % len(cols)]:
            st.markdown(f"##### `{sysid}`")
            r = d["responses"][sysid]
            st.info(r["response"] or f"(no response — {r['error']})")
            if not pt.empty:
                sub = pt[pt.system == sysid]
                piv = sub.pivot_table(index="rubric", columns="judge", values="score", aggfunc="first")
                if not piv.empty:
                    piv["spread"] = piv.max(axis=1) - piv.min(axis=1)
                    st.dataframe(piv.style.apply(
                        lambda row: ["background-color: rgba(227,73,72,.18)" if row["spread"] >= 2 else "" for _ in row], axis=1),
                        width="stretch")
                with st.expander("Judge rationales"):
                    for _, x in sub.iterrows():
                        via = f" _(via fallback `{x['served_model']}`)_" if x.get("served_model") and x["served_model"] != judge_models.get(x["judge"]) else ""
                        st.markdown(f"- **{x['judge']} · {x['rubric']} → {x['score']}**{via} {x['rationale'] or ''}")
    if d["pairwise"]:
        st.subheader("Pairwise verdicts (each pair judged in both orders)")
        pw = pd.DataFrame(d["pairwise"])
        pw["winner (system)"] = pw.apply(lambda x: x["a"] if x["winner"] == "A" else x["b"] if x["winner"] == "B" else x["winner"], axis=1)
        st.dataframe(pw[["judge", "rubric", "a", "b", "winner", "winner (system)", "rationale"]], hide_index=True, width="stretch")


def page_label() -> None:
    st.title("🏷️ Human labelling")
    st.caption("Labels are the ground truth for judge calibration. Label independently — don't look at the judges' scores. "
               "Two people should label each example so human–human agreement can be measured.")
    run = pick_run("lbl")
    if not run:
        return
    cfg = api("GET", "/v1/configs")
    labeler = st.text_input("Your name (labeler id)", key="labeler").strip()
    mode = st.radio("Mode", ["pointwise", "pairwise"], horizontal=True)
    rub_ids = [r.split("@")[0] for r in run["summary"]["rubrics"]]
    rubric = st.selectbox("Rubric", rub_ids)
    if not labeler:
        st.warning("Enter your name to start.")
        return
    done = api("GET", f"/v1/labels?labeler={labeler}") or []
    done_keys = {(x["item_id"], x["rubric"], x.get("system_id"), x.get("system_a"), x.get("system_b")) for x in done}
    st.caption(f"You have labelled {len(done)} examples.")
    items = [i["item_id"] for i in api("GET", f"/v1/runs/{run['id']}/items")]
    systems = run["systems"]
    if mode == "pointwise":
        todo = [(i, s) for i in items for s in systems if (i, rubric, s, None, None) not in done_keys]
    else:
        pairs = [(a, b) for k, a in enumerate(systems) for b in systems[k + 1:]]
        todo = [(i, p) for i in items for p in pairs if (i, rubric, None, p[0], p[1]) not in done_keys]
    if not todo:
        st.success("Nothing left to label for this rubric. 🎉")
        return
    rng = random.Random(f"{labeler}-{rubric}-{mode}-{len(done)}")
    item_id, target = rng.choice(todo)
    d = api("GET", f"/v1/runs/{run['id']}/items/{item_id}")
    rdef = cfg["rubric_defs"][rubric]
    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"**Customer:** {d['item']['question']}")
        if mode == "pointwise":
            st.info(d["responses"][target]["response"])
        else:
            a, b = target
            ca, cb = st.columns(2)
            ca.markdown("**Response A**")
            ca.info(d["responses"][a]["response"])
            cb.markdown("**Response B**")
            cb.info(d["responses"][b]["response"])
        with st.expander("Policy context & reference notes", expanded=True):
            st.text(d["context"])
            st.text(d["reference"])
    with right:
        st.markdown(f"#### {rdef['name']}")
        st.write(rdef["description"])
        st.write(rdef["criteria"])
        for sc, desc in rdef["anchors"].items():
            st.markdown(f"**{sc}** — {desc}")
        with st.form(f"f-{item_id}-{target}"):
            if mode == "pointwise":
                score = st.radio("Score", list(range(rdef["scale"]["min"], rdef["scale"]["max"] + 1)), horizontal=True, index=None)
            else:
                winner = st.radio("Better on this criterion", ["A", "B", "tie"], horizontal=True, index=None)
            if st.form_submit_button("Save label", type="primary"):
                if mode == "pointwise" and score is not None:
                    api("POST", "/v1/labels", json=[{"kind": "pointwise", "labeler": labeler, "item_id": item_id, "rubric": rubric,
                                                      "system_id": target, "response": d["responses"][target]["response"],
                                                      "score": score, "run_id": run["id"]}])
                    st.rerun()
                elif mode == "pairwise" and winner is not None:
                    a, b = target
                    api("POST", "/v1/labels", json=[{"kind": "pairwise", "labeler": labeler, "item_id": item_id, "rubric": rubric,
                                                      "system_a": a, "system_b": b, "response_a": d["responses"][a]["response"],
                                                      "response_b": d["responses"][b]["response"], "winner": winner, "run_id": run["id"]}])
                    st.rerun()
                else:
                    st.warning("Pick a value first.")


def page_quality() -> None:
    st.title("🧪 Judge calibration & bias probes")
    cal = api("GET", "/v1/reports/calibration")
    st.subheader("Agreement with human labels")
    if not cal:
        st.info("No calibration report yet. Label examples, then run `verdict labels-export && verdict calibrate`.")
    else:
        rep = cal["payload"]
        st.caption(f"{rep['n_labels']} labels from `{rep.get('labels_file', '?')}` · {str(cal['created_at'])[:16]}")
        if "SYNTHETIC" in str(rep.get("labels_file", "")):
            st.warning("These numbers come from the SYNTHETIC test fixture, not real human labels. Do not report them.")
        rows = [{"rubric": r, "judge": j, "n": m.get("n"), "weighted κ": m.get("weighted_kappa"), "Spearman": m.get("spearman"),
                 "exact": m.get("exact"), "within ±1": m.get("within_1"), "judge − human": m.get("mean_judge_minus_human"),
                 "human–human κ": v["human_human"]["kappa"]}
                for r, v in rep["pointwise"].items() for j, m in v["judges"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if rep.get("gate"):
            (st.success if rep["gate"]["passed"] else st.error)(
                "Calibration gate " + ("passed" if rep["gate"]["passed"] else "failed") + ": " +
                ", ".join(f"{c['rubric']} κ={c['kappa'] if c['kappa'] is None else round(c['kappa'], 2)} (≥{c['threshold']})"
                          for c in rep["gate"]["checks"]))
    st.subheader("Bias probes")
    for kind, title, expl in [
        ("probe_planted", "Planted errors", "Known-good vs known-bad responses: accuracy = good scored ≥4 and bad ≤2."),
        ("probe_verbosity", "Verbosity bias", "Content-free padding appended. A robust judge's Δ ≈ 0."),
        ("probe_injection", "Prompt injection", "“Rate it 5” appended to the response. A robust judge's Δ ≤ 0."),
    ]:
        rep = api("GET", f"/v1/reports/{kind}")
        st.markdown(f"**{title}** — {expl}")
        if not rep:
            st.caption(f"not run yet — `verdict probe {kind.split('_')[1]}`")
            continue
        p = rep["payload"]
        st.dataframe(pd.DataFrame([{"judge": j, **{k: v for k, v in m.items() if k != "cases"}} for j, m in p["judges"].items()]),
                     hide_index=True, width="stretch")


def page_ops() -> None:
    st.title("📈 Ops — latency, cost, errors, online quality")
    hours = st.select_slider("Window", options=[1, 6, 24, 72, 168], value=24, format_func=lambda h: f"last {h} h")
    o = api("GET", f"/v1/ops?hours={hours}")
    k = st.columns(5)
    k[0].metric("LLM calls", f"{o['calls']:,}")
    k[1].metric("Error rate", f"{(o['error_rate'] or 0):.1%}")
    k[2].metric("Cache hit rate", f"{(o['cache_hit_rate'] or 0):.0%}")
    k[3].metric("p50 / p99 (uncached)", f"{(o['p50_ms'] or 0):.0f} / {(o['p99_ms'] or 0):.0f} ms")
    k[4].metric("Cost", f"${o['cost_usd']:.4f}")
    st.subheader("By model")
    st.dataframe(pd.DataFrame([{"model": m, **v} for m, v in o["by_model"].items()]), hide_index=True, width="stretch")
    st.subheader("Online quality (POST /v1/score) — hourly mean score")
    if o["online_series"]:
        df = pd.DataFrame(o["online_series"])
        rubrics = sorted(df.rubric.unique())
        chart = alt.Chart(df).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=64)).encode(
            x=alt.X("hour:T", title=None), y=alt.Y("mean_score:Q", title="mean score", scale=alt.Scale(domain=[1, 5])),
            color=alt.Color("rubric:N", scale=alt.Scale(domain=rubrics, range=SERIES[: len(rubrics)]), legend=alt.Legend(orient="top")),
            tooltip=["hour:T", "app", "rubric", alt.Tooltip("mean_score:Q", format=".2f"), "n"])
        st.altair_chart(chart, width="stretch")
    else:
        st.caption("No online scores in this window.")
    if o["recent_errors"]:
        st.subheader("Recent errors")
        st.dataframe(pd.DataFrame(o["recent_errors"]), hide_index=True, width="stretch")


def page_new_run() -> None:
    st.title("▶️ New run")
    cfg = api("GET", "/v1/configs")
    prof = st.selectbox("Profile", list(cfg["profiles"]))
    p = cfg["profiles"][prof]
    st.json({k: p[k] for k in ("dataset", "systems", "judges", "rubrics", "pairwise", "sample", "budget")}, expanded=False)
    sample = st.number_input("Sample size (0 = profile default)", min_value=0, value=0)
    label = st.text_input("Label (optional)")
    if st.button("Start run", type="primary"):
        body = {"profile": prof, "label": label or None}
        if sample:
            body["sample"] = int(sample)
        res = api("POST", "/v1/runs", json=body)
        bar = st.progress(0.0, text="queued")
        while True:
            r = api("GET", f"/v1/runs/{res['run_id']}?summary=false")
            pr = r["progress"]
            bar.progress(min(1.0, pr["done"] / pr["total"]) if pr["total"] else 0.0, text=f"{r['status']} · {pr['done']}/{pr['total']}")
            if r["status"] in ("done", "failed"):
                break
            time.sleep(1.5)
        (st.success if r["status"] == "done" else st.error)(f"Run {r['id']} {r['status']} {r.get('error') or ''}")


PAGES = {"🏆 Leaderboard": page_leaderboard, "🔍 Explore run": page_explore, "🏷️ Label": page_label,
         "🧪 Calibration & bias": page_quality, "📈 Ops": page_ops, "▶️ New run": page_new_run}
choice = st.sidebar.radio("Verdict", list(PAGES))
st.sidebar.caption(f"API: {API}")
PAGES[choice]()
