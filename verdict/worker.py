"""In-process background worker: claims queued runs from the DB and executes them.
Runs survive restarts — anything left 'running' by a crash is re-queued on startup,
and the LLM cache makes the re-execution nearly free."""
from __future__ import annotations

import logging
import threading

from sqlalchemy import select, update

from . import db, runner

log = logging.getLogger("verdict.worker")


def recover() -> int:
    with db.write() as s:
        res = s.execute(update(db.Run).where(db.Run.status == "running").values(status="queued"))
        return res.rowcount or 0


def claim_next() -> str | None:
    with db.session() as s:
        run_id = s.scalar(select(db.Run.id).where(db.Run.status == "queued").order_by(db.Run.created_at).limit(1))
    if run_id is None:
        return None
    with db.write() as s:
        res = s.execute(update(db.Run).where(db.Run.id == run_id, db.Run.status == "queued").values(status="running"))
    return run_id if res.rowcount == 1 else None


class Worker(threading.Thread):
    def __init__(self, poll_s: float = 1.0):
        super().__init__(daemon=True, name="verdict-worker")
        self.poll_s = poll_s
        self._halt = threading.Event()
        self._wake = threading.Event()

    def notify(self) -> None:
        self._wake.set()

    def stop(self) -> None:
        self._halt.set()
        self._wake.set()

    def run(self) -> None:
        n = recover()
        if n:
            log.info("re-queued %d interrupted run(s)", n)
        while not self._halt.is_set():
            run_id = claim_next()
            if run_id is None:
                self._wake.wait(self.poll_s)
                self._wake.clear()
                continue
            log.info("executing run %s", run_id)
            try:
                runner.execute(run_id)
            except Exception as e:  # noqa: BLE001 — failure is recorded on the run row
                log.warning("run %s failed: %s", run_id, e)
