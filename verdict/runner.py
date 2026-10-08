"""Executes a run: generate → pointwise judge → pairwise judge (both orders) → aggregate."""
from __future__ import annotations

import logging
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from sqlalchemy import update

from . import db, results
from .config import Registry, registry, sha
from .judge import JudgeInput, judge_pairwise, judge_pointwise
from .llm import Budget, BudgetExceeded, LLMClient
from .providers import ProviderError
from .spec import RunRequest, RunSpec, SpecError, new_run_id, resolve

log = logging.getLogger("verdict.runner")


def _check_credentials(reg: Registry, spec: RunSpec) -> None:
    """Fail fast (before queueing) if a model in the run needs an API key that is not set."""
    models = [reg.systems[s].model for s in spec.systems if s not in spec.external_systems]
    models += [reg.judges[j].model for j in spec.judges]
    missing = sorted({p.api_key_env for p in (reg.providers[reg.models[m].provider] for m in models)
                      if p.api_key_env and not os.environ.get(p.api_key_env)})
    if missing:
        raise SpecError(f"missing API key(s): {', '.join(missing)} — add them to .env")


def create_run(req: RunRequest, reg: Registry | None = None) -> db.Run:
    reg = reg or registry()
    spec = resolve(reg, req)
    _check_credentials(reg, spec)
    chash = reg.config_hash(dataset=spec.dataset, systems=spec.systems, judges=spec.judges,
                            rubrics=spec.rubrics + spec.pairwise_rubrics, external=spec.external_outputs)
    run = db.Run(id=new_run_id(), status="queued", profile=spec.profile, spec=spec.model_dump(mode="json"),
                 config_hash=chash)
    db.add_all([run])
    return run


class _Progress:
    def __init__(self, run_id: str, total: int):
        self.run_id, self.total, self.done = run_id, total, 0
        self._lock = threading.Lock()
        self._last_flush = 0.0
        with db.write() as s:
            s.execute(update(db.Run).where(db.Run.id == run_id).values(progress_total=total, progress_done=0))

    def tick(self, n: int = 1) -> None:
        with self._lock:
            self.done += n
            now = time.monotonic()
            if now - self._last_flush < 0.5 and self.done < self.total:
                return
            self._last_flush = now
            done = self.done
        with db.write() as s:
            s.execute(update(db.Run).where(db.Run.id == self.run_id).values(progress_done=done))


def _parallel(fn: Callable, tasks: list, workers: int, progress: _Progress) -> list:
    """Run fn over tasks in a thread pool; a BudgetExceeded in any task aborts the rest."""
    out: list = [None] * len(tasks)
    abort = threading.Event()

    def wrapped(i: int, t):  # noqa: ANN001
        if abort.is_set():
            return
        try:
            out[i] = fn(t)
        except BudgetExceeded:
            abort.set()
            raise
        finally:
            progress.tick()

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(wrapped, i, t) for i, t in enumerate(tasks)]
        for f in futures:
            exc = f.exception()
            if isinstance(exc, BudgetExceeded):
                for g in futures:
                    g.cancel()
                raise exc
            if exc is not None:
                raise exc
    return out


def execute(run_id: str, reg: Registry | None = None) -> dict:
    reg = reg or registry()
    run = db.get_run(run_id)
    if run is None:
        raise KeyError(run_id)
    spec = RunSpec(**run.spec)
    with db.write() as s:
        s.execute(update(db.Run).where(db.Run.id == run_id).values(status="running", started_at=db.utcnow(), error=None))
    # Re-running a run id (e.g. after a crash) starts clean; the LLM cache makes it cheap.
    with db.write() as s:
        s.query(db.Generation).filter(db.Generation.run_id == run_id).delete()
        s.query(db.Judgment).filter(db.Judgment.run_id == run_id).delete()

    try:
        summary = _execute(run_id, spec, reg)
        with db.write() as s:
            s.execute(update(db.Run).where(db.Run.id == run_id).values(
                status="done", summary=summary, finished_at=db.utcnow()))
        return summary
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        if not isinstance(e, BudgetExceeded):
            log.error("run %s failed\n%s", run_id, traceback.format_exc())
        partial = None
        try:
            partial = results.summarize(run_id, spec, reg)
        except Exception:  # noqa: BLE001
            pass
        with db.write() as s:
            s.execute(update(db.Run).where(db.Run.id == run_id).values(
                status="failed", error=msg, summary=partial, finished_at=db.utcnow()))
        raise


def _execute(run_id: str, spec: RunSpec, reg: Registry) -> dict:
    ds = reg.dataset(spec.dataset)
    client = LLMClient(reg, run_id=run_id, budget=Budget(spec.budget.max_calls, spec.budget.max_usd))
    items = [ds.item(i) for i in spec.item_ids]
    inputs = {it.id: JudgeInput(question=it.question, context=ds.context_for(it),
                                reference=ds.reference_for(it), style_guide=ds.style_guide) for it in items}
    rubrics = [reg.rubric(r) for r in spec.rubrics]
    pw_rubrics = [reg.rubric(r) for r in spec.pairwise_rubrics]
    model_systems = [s for s in spec.systems if s not in spec.external_systems]

    n_gen = len(model_systems) * len(items)
    n_point = len(spec.systems) * len(items) * len(rubrics) * len(spec.judges)
    n_pair = len(spec.pairs) * len(items) * len(pw_rubrics) * len(spec.judges) * 2
    progress = _Progress(run_id, n_gen + n_point + n_pair)

    # 1. Generation -----------------------------------------------------------
    responses: dict[tuple[str, str], str] = {}
    for s in spec.external_systems:
        for it in items:
            responses[(s, it.id)] = spec.external_outputs[s][it.id]

    def gen(task: tuple[str, str]) -> tuple[str, str, str, str | None]:
        sid, iid = task
        sc = reg.systems[sid]
        prompt = sc.prompt_text()
        system_msg = prompt.replace("{policy}", ds.policy_text) if sc.include_policy else prompt.replace("{policy}", "")
        msgs = [{"role": "system", "content": system_msg.strip()}, {"role": "user", "content": ds.item(iid).question}]
        try:
            return sid, iid, client.complete(sc.model, msgs, purpose="generate").text.strip(), None
        except ProviderError as e:
            return sid, iid, "", str(e)[:500]

    gen_rows = []
    for sid, iid, text, err in _parallel(gen, [(s, it.id) for s in model_systems for it in items], spec.concurrency, progress):
        responses[(sid, iid)] = text
        gen_rows.append(db.Generation(run_id=run_id, system_id=sid, item_id=iid, response=text, output_hash=sha(text), error=err))
    for s in spec.external_systems:
        for it in items:
            t = responses[(s, it.id)]
            gen_rows.append(db.Generation(run_id=run_id, system_id=s, item_id=it.id, response=t, output_hash=sha(t)))
    db.add_all(gen_rows)

    # 2. Pointwise -------------------------------------------------------------
    def point(task):  # noqa: ANN001, ANN202
        sid, iid, rub, jid = task
        resp = responses[(sid, iid)]
        if not resp:
            return db.Judgment(run_id=run_id, kind="pointwise", judge_id=jid, rubric_id=rub.id, rubric_version=rub.version,
                               item_id=iid, system_id=sid, abstain=True, parse_ok=False, rationale="no response (generation failed)")
        r = judge_pointwise(client, reg.judge_chain(jid), rub, inputs[iid], resp)
        return db.Judgment(run_id=run_id, kind="pointwise", judge_id=jid, rubric_id=rub.id, rubric_version=rub.version,
                           item_id=iid, system_id=sid, score=r.score, rationale=r.rationale or r.error,
                           parse_ok=r.parse_ok, repaired=r.repaired, abstain=r.abstain, served_model=r.served_model)

    tasks = [(s, it.id, r, j) for s in spec.systems for it in items for r in rubrics for j in spec.judges]
    db.add_all(_parallel(point, tasks, spec.concurrency, progress))

    # 3. Pairwise, both orders ---------------------------------------------------
    def pair(task):  # noqa: ANN001, ANN202
        a, b, iid, rub, jid = task
        ra, rb = responses[(a, iid)], responses[(b, iid)]
        if not ra or not rb:
            return db.Judgment(run_id=run_id, kind="pairwise", judge_id=jid, rubric_id=rub.id, rubric_version=rub.version,
                               item_id=iid, system_a=a, system_b=b, abstain=True, parse_ok=False, rationale="missing response")
        r = judge_pairwise(client, reg.judge_chain(jid), rub, inputs[iid], ra, rb)
        return db.Judgment(run_id=run_id, kind="pairwise", judge_id=jid, rubric_id=rub.id, rubric_version=rub.version,
                           item_id=iid, system_a=a, system_b=b, winner=r.winner, rationale=r.rationale or r.error,
                           parse_ok=r.parse_ok, repaired=r.repaired, abstain=r.abstain, served_model=r.served_model)

    tasks = [(a, b, it.id, r, j) for (x, y) in spec.pairs for it in items for r in pw_rubrics for j in spec.judges
             for (a, b) in ((x, y), (y, x))]
    db.add_all(_parallel(pair, tasks, spec.concurrency, progress))

    # 4. Aggregate ---------------------------------------------------------------
    return results.summarize(run_id, spec, reg)


def run_sync(req: RunRequest) -> tuple[str, dict]:
    """Create + execute in the current process (CLI / tests)."""
    run = create_run(req)
    return run.id, execute(run.id)
