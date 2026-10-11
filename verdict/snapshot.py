"""Export / import a portable snapshot of selected runs (+ their judgments, telemetry,
reports and LLM cache) as gzipped JSONL.

Used to ship real results with the repo so a fresh deployment (e.g. Streamlit Cloud,
whose filesystem is ephemeral) boots with the live NVIDIA runs already on the leaderboard.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import DateTime, func, select

from . import db

TABLES = [db.Run, db.Generation, db.Judgment, db.LLMCall, db.Report, db.OnlineScore, db.CacheEntry]


def _row(obj) -> dict:  # noqa: ANN001
    out = {}
    for col in obj.__table__.columns:
        v = getattr(obj, col.name)
        out[col.name] = v.isoformat() if isinstance(v, datetime) else v
    return out


def export(path: Path, run_ids: list[str], include_cache: bool = True) -> dict:
    counts: dict[str, int] = {}
    with db.session() as s, gzip.open(path, "wt", encoding="utf-8") as f:
        def emit(model, rows) -> None:  # noqa: ANN001
            n = 0
            for r in rows:
                f.write(json.dumps({"table": model.__tablename__, "row": _row(r)}, ensure_ascii=False) + "\n")
                n += 1
            counts[model.__tablename__] = counts.get(model.__tablename__, 0) + n

        emit(db.Run, s.scalars(select(db.Run).where(db.Run.id.in_(run_ids))))
        for m in (db.Generation, db.Judgment, db.LLMCall):
            emit(m, s.scalars(select(m).where(m.run_id.in_(run_ids))))
        latest = s.execute(select(db.Report.kind, func.max(db.Report.id)).group_by(db.Report.kind)).all()
        emit(db.Report, s.scalars(select(db.Report).where(db.Report.id.in_([i for _, i in latest]))))
        emit(db.OnlineScore, s.scalars(select(db.OnlineScore)))
        if include_cache:
            emit(db.CacheEntry, s.scalars(select(db.CacheEntry)))
    return counts


def import_(path: Path) -> dict:
    """Insert the snapshot rows. Rows whose primary key already exists are skipped."""
    by_name = {m.__tablename__: m for m in TABLES}
    counts: dict[str, int] = {}
    batch: dict[str, list] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            model = by_name[rec["table"]]
            row = dict(rec["row"])
            for col in model.__table__.columns:
                if isinstance(col.type, DateTime) and isinstance(row.get(col.name), str):
                    row[col.name] = datetime.fromisoformat(row[col.name])
            if model is not db.Run and model is not db.CacheEntry:
                row.pop("id", None)  # autoincrement ids are re-assigned
            batch.setdefault(rec["table"], []).append(model(**row))
    with db.write() as s:
        existing_runs = set(s.scalars(select(db.Run.id)))
        existing_cache = set(s.scalars(select(db.CacheEntry.key)))
        existing_reports = {(r.kind, r.created_at.replace(tzinfo=None)) for r in s.scalars(select(db.Report))}
        existing_online = {o.created_at.replace(tzinfo=None) for o in s.scalars(select(db.OnlineScore))}
        skip_runs = {r.id for r in batch.get("runs", []) if r.id in existing_runs}
        for name in [m.__tablename__ for m in TABLES]:
            rows = batch.get(name, [])
            if name == "runs":
                rows = [r for r in rows if r.id not in skip_runs]
            elif name == "llm_cache":
                rows = [r for r in rows if r.key not in existing_cache]
            elif name in ("generations", "judgments", "llm_calls"):
                rows = [r for r in rows if r.run_id not in skip_runs]
            elif name == "reports":
                rows = [r for r in rows if (r.kind, r.created_at.replace(tzinfo=None)) not in existing_reports]
            elif name == "online_scores":
                rows = [r for r in rows if r.created_at.replace(tzinfo=None) not in existing_online]
            s.add_all(rows)
            counts[name] = len(rows)
    return counts


def seed(path: Path) -> dict | None:
    """Import the snapshot unless this exact file (by content hash) was already imported.
    Idempotent and incremental: an existing database gains the snapshot's missing runs (e.g. after a
    redeploy with a newer snapshot) without duplicating runs, cache entries, reports or online scores."""
    if not path.exists():
        return None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with db.session() as s:
        done = [r.payload.get("sha256") for r in s.scalars(select(db.Report).where(db.Report.kind == "snapshot_import"))]
    if digest in done:
        return None
    counts = import_(path)
    db.add_all([db.Report(kind="snapshot_import", payload={"sha256": digest, "file": path.name, "counts": counts})])
    return counts


seed_if_empty = seed  # backwards-compatible name
