"""Verdict REST API (FastAPI). All request/response bodies are JSON.

POST /v1/runs is asynchronous: it returns 202 + run_id immediately and the background
worker executes the run. Clients (CLI, UI, the GitHub Action) poll GET /v1/runs/{id}."""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import timedelta
from statistics import mean

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from . import db, report, results, settings
from .config import registry
from .judge import JudgeInput, judge_pointwise
from .llm import LLMClient
from .runner import create_run
from .spec import RunRequest, SpecError
from .stats import percentile
from .worker import Worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
_worker: Worker | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):  # noqa: ANN201
    global _worker
    db.init()
    if settings.worker_enabled():
        _worker = Worker()
        _worker.start()
    yield
    if _worker:
        _worker.stop()


app = FastAPI(title="Verdict", version="0.1.0", lifespan=lifespan,
              description="Calibrated LLM-as-judge evaluation service + leaderboard")


def auth(x_api_key: str | None = Header(default=None)) -> None:
    expected = settings.api_key()
    if expected and x_api_key != expected:
        raise HTTPException(401, "missing or invalid X-API-Key")


def _run_view(r: db.Run, with_summary: bool = False) -> dict:
    out = {"id": r.id, "status": r.status, "profile": r.profile, "label": r.spec.get("label"),
           "dataset": r.spec.get("dataset"), "systems": r.spec.get("systems"), "judges": r.spec.get("judges"),
           "items": len(r.spec.get("item_ids", [])), "config_hash": r.config_hash,
           "progress": {"done": r.progress_done, "total": r.progress_total}, "error": r.error,
           "created_at": r.created_at, "started_at": r.started_at, "finished_at": r.finished_at}
    if r.summary and r.summary.get("gate"):
        out["gate_passed"] = r.summary["gate"]["passed"]
    if with_summary:
        out["summary"] = r.summary
    return out


@app.get("/health")
def health() -> dict:
    return {"ok": True, "worker": bool(_worker and _worker.is_alive())}


@app.get("/v1/configs", dependencies=[Depends(auth)])
def configs() -> dict:
    reg = registry()
    return {
        "profiles": {k: v.model_dump() for k, v in reg.profiles.items()},
        "systems": {k: v.model_dump() for k, v in reg.systems.items()},
        "judges": {k: {**v.model_dump(), "model_id": v.model, "model": reg.model_for_judge(k).model,
                       "chain": reg.judge_chain(k)} for k, v in reg.judges.items()},
        "rubrics": {rid: sorted(vs) for rid, vs in reg.rubric_versions.items()},
        "rubric_defs": {rid: reg.rubric(rid).model_dump() for rid in reg.rubric_versions},
    }


@app.post("/v1/runs", status_code=202, dependencies=[Depends(auth)])
def submit_run(req: RunRequest) -> dict:
    try:
        run = create_run(req)
    except (SpecError, KeyError) as e:
        raise HTTPException(422, str(e)) from e
    if _worker:
        _worker.notify()
    return {"run_id": run.id, "status": run.status, "config_hash": run.config_hash}


@app.get("/v1/runs", dependencies=[Depends(auth)])
def runs(limit: int = 50) -> list[dict]:
    return [_run_view(r) for r in db.list_runs(limit)]


@app.get("/v1/runs/{run_id}", dependencies=[Depends(auth)])
def run(run_id: str, summary: bool = True) -> dict:
    r = db.get_run(run_id)
    if r is None:
        raise HTTPException(404, "run not found")
    return _run_view(r, with_summary=summary)


@app.get("/v1/runs/{run_id}/report.md", response_class=PlainTextResponse, dependencies=[Depends(auth)])
def run_report(run_id: str) -> str:
    r = db.get_run(run_id)
    if r is None or not r.summary:
        raise HTTPException(404, "run not found or not finished")
    return report.run_markdown(r.summary, title=f"Verdict eval — run `{run_id}`" + (f" ({r.spec.get('label')})" if r.spec.get("label") else ""))


@app.get("/v1/runs/{run_id}/items", dependencies=[Depends(auth)])
def run_items(run_id: str) -> list[dict]:
    r = db.get_run(run_id)
    if r is None:
        raise HTTPException(404, "run not found")
    ds = registry().dataset(r.spec["dataset"])
    s = r.summary or {}
    dis = {d["item"] for d in s.get("disagreement", {}).get("top", [])}
    flips = {f["item"] for f in s.get("flip_cases", [])}
    return [{"item_id": i, "category": ds.item(i).category, "question": ds.item(i).question,
             "judge_disagreement": i in dis, "position_flip": i in flips} for i in r.spec["item_ids"]]


@app.get("/v1/runs/{run_id}/items/{item_id}", dependencies=[Depends(auth)])
def run_item(run_id: str, item_id: str) -> dict:
    try:
        return results.item_detail(run_id, item_id, registry())
    except KeyError as e:
        raise HTTPException(404, f"not found: {e}") from e


@app.get("/v1/leaderboard", dependencies=[Depends(auth)])
def leaderboard(dataset: str | None = None, profile: str | None = None) -> dict:
    for r in db.list_runs(500):
        if r.status == "done" and (dataset is None or r.spec.get("dataset") == dataset) \
                and (profile is None or r.profile == profile):
            return _run_view(r, with_summary=True)
    raise HTTPException(404, "no completed run matches")


class ScoreRequest(BaseModel):
    """Online scoring of one live interaction (sampled production traffic)."""
    question: str
    response: str
    context: str = ""
    reference: str = ""
    rubrics: list[str] = Field(default_factory=lambda: ["faithfulness", "helpfulness", "safety"])
    judges: list[str]
    app: str = "default"


@app.post("/v1/score", dependencies=[Depends(auth)])
def score(req: ScoreRequest) -> dict:
    reg = registry()
    for j in req.judges:
        if j not in reg.judges:
            raise HTTPException(422, f"unknown judge {j}")
    client = LLMClient(reg)
    inp = JudgeInput(req.question, req.context, req.reference, "")
    t0 = time.perf_counter()
    tasks = [(reg.rubric(r), j) for r in req.rubrics for j in req.judges]
    # all (rubric, judge) calls in parallel: latency = slowest call, not the sum
    with ThreadPoolExecutor(max_workers=len(tasks) or 1) as pool:
        res = list(pool.map(lambda t: judge_pointwise(client, reg.judge_chain(t[1]), t[0], inp, req.response), tasks))
    out, rows = {}, []
    for (rub, j), r in zip(tasks, res):
        out.setdefault(rub.id, {})[j] = {"score": r.score, "rationale": r.rationale, "abstain": r.abstain,
                                         "served_model": r.served_model}
        rows.append(db.OnlineScore(app=req.app, rubric_id=rub.id, judge_id=j, score=r.score,
                                   payload={"question": req.question[:500], "response": req.response[:2000]}))
    db.add_all(rows)
    return {"scores": out, "latency_ms": (time.perf_counter() - t0) * 1000}


class LabelIn(BaseModel):
    kind: str = "pointwise"
    labeler: str
    item_id: str
    rubric: str
    system_id: str | None = None
    response: str | None = None
    score: int | None = None
    system_a: str | None = None
    system_b: str | None = None
    response_a: str | None = None
    response_b: str | None = None
    winner: str | None = None
    run_id: str | None = None


@app.post("/v1/labels", dependencies=[Depends(auth)])
def add_labels(labels: list[LabelIn]) -> dict:
    for l in labels:
        if l.kind == "pointwise" and (l.response is None or l.score is None):
            raise HTTPException(422, "pointwise labels need response + score")
        if l.kind == "pairwise" and (l.response_a is None or l.response_b is None or l.winner not in {"A", "B", "tie"}):
            raise HTTPException(422, "pairwise labels need response_a, response_b, winner A|B|tie")
    db.add_all([db.HumanLabel(kind=l.kind, labeler=l.labeler, item_id=l.item_id, rubric_id=l.rubric,
                              payload=l.model_dump(exclude_none=True)) for l in labels])
    return {"saved": len(labels)}


@app.get("/v1/labels", dependencies=[Depends(auth)])
def get_labels(labeler: str | None = None) -> list[dict]:
    with db.session() as s:
        q = select(db.HumanLabel)
        if labeler:
            q = q.where(db.HumanLabel.labeler == labeler)
        return [{**h.payload, "id": h.id, "created_at": h.created_at} for h in s.scalars(q)]


@app.get("/v1/labels/export", response_class=PlainTextResponse, dependencies=[Depends(auth)])
def export_labels() -> str:
    with db.session() as s:
        return "\n".join(json.dumps(h.payload, ensure_ascii=False) for h in s.scalars(select(db.HumanLabel)))


@app.get("/v1/reports/{kind}", dependencies=[Depends(auth)])
def latest_report(kind: str) -> dict:
    with db.session() as s:
        r = s.scalar(select(db.Report).where(db.Report.kind == kind).order_by(db.Report.created_at.desc()).limit(1))
    if r is None:
        raise HTTPException(404, f"no {kind} report yet")
    return {"kind": kind, "created_at": r.created_at, "payload": r.payload}


@app.get("/v1/ops", dependencies=[Depends(auth)])
def ops(hours: float = Query(24, gt=0)) -> dict:
    since = db.utcnow() - timedelta(hours=hours)
    with db.session() as s:
        calls = list(s.scalars(select(db.LLMCall).where(db.LLMCall.created_at >= since)))
        online = list(s.scalars(select(db.OnlineScore).where(db.OnlineScore.created_at >= since)))
    real = [c for c in calls if not c.cache_hit and not c.error]
    by_model = {}
    for m in sorted({c.model_id for c in calls}):
        mc = [c for c in calls if c.model_id == m]
        mr = [c.latency_ms for c in real if c.model_id == m]
        by_model[m] = {"calls": len(mc), "errors": sum(1 for c in mc if c.error),
                       "cache_hits": sum(c.cache_hit for c in mc),
                       "p50_ms": percentile(mr, 50), "p99_ms": percentile(mr, 99),
                       "tokens_in": sum(c.input_tokens for c in mc if not c.cache_hit),
                       "tokens_out": sum(c.output_tokens for c in mc if not c.cache_hit),
                       "cost_usd": sum(c.cost_usd for c in mc)}
    buckets: dict[tuple, list[float]] = {}
    for o in online:
        if o.score is not None:
            key = (o.created_at.replace(minute=0, second=0, microsecond=0).isoformat(), o.app, o.rubric_id)
            buckets.setdefault(key, []).append(o.score)
    series = [{"hour": h, "app": a, "rubric": r, "mean_score": mean(v), "n": len(v)} for (h, a, r), v in sorted(buckets.items())]
    return {
        "window_hours": hours, "calls": len(calls), "errors": sum(1 for c in calls if c.error),
        "error_rate": (sum(1 for c in calls if c.error) / len(calls)) if calls else None,
        "cache_hit_rate": (sum(c.cache_hit for c in calls) / len(calls)) if calls else None,
        "p50_ms": percentile([c.latency_ms for c in real], 50), "p99_ms": percentile([c.latency_ms for c in real], 99),
        "cost_usd": sum(c.cost_usd for c in calls), "by_model": by_model, "online_series": series,
        "recent_errors": [{"model": c.model_id, "purpose": c.purpose, "error": c.error, "at": c.created_at}
                          for c in sorted((c for c in calls if c.error), key=lambda c: c.created_at)[-20:]],
    }
