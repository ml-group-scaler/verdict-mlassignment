"""Process-wide settings. Everything is overridable by environment variables."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(os.environ.get("VERDICT_ROOT", Path(__file__).resolve().parent.parent))
load_dotenv(ROOT / ".env")


def database_url() -> str:
    return os.environ.get("DATABASE_URL") or f"sqlite:///{ROOT / 'verdict.db'}"


def api_key() -> str | None:
    return os.environ.get("VERDICT_API_KEY") or None


def worker_enabled() -> bool:
    return os.environ.get("VERDICT_DISABLE_WORKER", "").lower() not in {"1", "true", "yes"}
