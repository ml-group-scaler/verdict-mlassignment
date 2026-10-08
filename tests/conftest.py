import os
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="verdict-test-"))
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP / 'test.db'}"
os.environ.pop("VERDICT_API_KEY", None)

from verdict import db  # noqa: E402

db.init(os.environ["DATABASE_URL"])

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def reg():
    from verdict.config import registry

    return registry()
