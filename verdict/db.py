"""Storage layer (SQLAlchemy 2.0). SQLite by default; set DATABASE_URL for Postgres."""
from __future__ import annotations

import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text, create_engine, event, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .settings import database_url


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)  # queued|running|done|failed
    profile: Mapped[str | None] = mapped_column(String(64))
    spec: Mapped[dict] = mapped_column(JSON)
    config_hash: Mapped[str] = mapped_column(String(32))
    progress_done: Mapped[int] = mapped_column(Integer, default=0)
    progress_total: Mapped[int] = mapped_column(Integer, default=0)
    summary: Mapped[dict | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))  # last progress write by the executor


class Generation(Base):
    __tablename__ = "generations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(32), index=True)
    system_id: Mapped[str] = mapped_column(String(64))
    item_id: Mapped[str] = mapped_column(String(64))
    response: Mapped[str] = mapped_column(Text)
    output_hash: Mapped[str] = mapped_column(String(32), index=True)
    error: Mapped[str | None] = mapped_column(Text)


class Judgment(Base):
    __tablename__ = "judgments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(32), index=True)
    kind: Mapped[str] = mapped_column(String(16))  # pointwise | pairwise
    judge_id: Mapped[str] = mapped_column(String(64))
    rubric_id: Mapped[str] = mapped_column(String(64))
    rubric_version: Mapped[int] = mapped_column(Integer)
    item_id: Mapped[str] = mapped_column(String(64))
    system_id: Mapped[str | None] = mapped_column(String(64))  # pointwise
    system_a: Mapped[str | None] = mapped_column(String(64))  # pairwise, as shown in position A
    system_b: Mapped[str | None] = mapped_column(String(64))
    score: Mapped[float | None] = mapped_column(Float)  # pointwise score
    winner: Mapped[str | None] = mapped_column(String(8))  # pairwise: A | B | tie (positional)
    rationale: Mapped[str | None] = mapped_column(Text)
    parse_ok: Mapped[bool] = mapped_column(Boolean, default=True)
    repaired: Mapped[bool] = mapped_column(Boolean, default=False)
    abstain: Mapped[bool] = mapped_column(Boolean, default=False)
    served_model: Mapped[str | None] = mapped_column(String(64))  # primary or fallback model that answered


class LLMCall(Base):
    """One row per LLM request (including cache hits) — the observability table."""
    __tablename__ = "llm_calls"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str | None] = mapped_column(String(32), index=True)
    purpose: Mapped[str] = mapped_column(String(32))  # generate | judge | repair | probe | online | calibration
    model_id: Mapped[str] = mapped_column(String(64))
    latency_ms: Mapped[float] = mapped_column(Float)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class CacheEntry(Base):
    __tablename__ = "llm_cache"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_id: Mapped[str] = mapped_column(String(64))
    text: Mapped[str] = mapped_column(Text)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class HumanLabel(Base):
    __tablename__ = "human_labels"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(16))  # pointwise | pairwise
    labeler: Mapped[str] = mapped_column(String(64))
    item_id: Mapped[str] = mapped_column(String(64))
    rubric_id: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict] = mapped_column(JSON)  # responses, system ids, score/winner, run_id
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class OnlineScore(Base):
    __tablename__ = "online_scores"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    app: Mapped[str] = mapped_column(String(64), default="default")
    rubric_id: Mapped[str] = mapped_column(String(64))
    judge_id: Mapped[str] = mapped_column(String(64))
    score: Mapped[float | None] = mapped_column(Float)
    payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class Report(Base):
    """Calibration / bias-probe reports, so the UI can display the latest of each kind."""
    __tablename__ = "reports"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


_engine = None
_Session: sessionmaker | None = None
_write_lock = threading.Lock()  # SQLite allows one writer; serialise writes from worker threads


def init(url: str | None = None) -> None:
    global _engine, _Session
    url = url or database_url()
    is_sqlite = url.startswith("sqlite")
    _engine = create_engine(
        url,
        connect_args={"check_same_thread": False, "timeout": 30} if is_sqlite else {},
        pool_pre_ping=not is_sqlite,
    )
    if is_sqlite:
        @event.listens_for(_engine, "connect")
        def _pragmas(conn, _):  # noqa: ANN001
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()
    Base.metadata.create_all(_engine)
    _migrate(_engine)
    _Session = sessionmaker(_engine, expire_on_commit=False)


def _migrate(engine) -> None:  # noqa: ANN001
    """Additive migration: add any nullable column that exists in the models but not in the DB
    (create_all only creates missing tables). Never drops or alters existing columns."""
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            existing = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name not in existing and col.nullable:
                    conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN {col.name} {col.type.compile(engine.dialect)}'))


def _ensure() -> sessionmaker:
    if _Session is None:
        init()
    assert _Session is not None
    return _Session


@contextmanager
def session() -> Iterator[Session]:
    """Read session (no commit)."""
    s = _ensure()()
    try:
        yield s
    finally:
        s.close()


@contextmanager
def write() -> Iterator[Session]:
    """Write session: serialised, committed on success, rolled back on error."""
    with _write_lock:
        s = _ensure()()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()


def add_all(rows: list[Any]) -> None:
    if rows:
        with write() as s:
            s.add_all(rows)


def get_run(run_id: str) -> Run | None:
    with session() as s:
        return s.get(Run, run_id)


def list_runs(limit: int = 50) -> list[Run]:
    with session() as s:
        return list(s.scalars(select(Run).order_by(Run.created_at.desc()).limit(limit)))
