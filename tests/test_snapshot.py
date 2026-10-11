from verdict import db, snapshot
from verdict.runner import run_sync
from verdict.settings import DEMO_SNAPSHOT
from verdict.spec import RunRequest


def test_snapshot_round_trip_into_empty_db(tmp_path, monkeypatch):
    run_id, summary = run_sync(RunRequest(profile="mock", sample=3, label="snap"))
    path = tmp_path / "snap.jsonl.gz"
    counts = snapshot.export(path, [run_id])
    assert counts["runs"] == 1 and counts["judgments"] > 0

    db.init(f"sqlite:///{tmp_path / 'fresh.db'}")       # a brand-new, empty database
    try:
        assert snapshot.seed_if_empty(path)["runs"] == 1
        r = db.get_run(run_id)
        assert r.status == "done" and r.summary["ranking"] == summary["ranking"]
        assert snapshot.seed_if_empty(path) is None      # non-empty DB is never re-seeded
        assert snapshot.import_(path)["runs"] == 0       # existing runs are skipped, not duplicated
    finally:
        import os
        db.init(os.environ["DATABASE_URL"])


def test_committed_demo_snapshot_is_valid(tmp_path):
    import gzip
    import json

    tables = {json.loads(l)["table"] for l in gzip.open(DEMO_SNAPSHOT, "rt")}
    assert {"runs", "generations", "judgments", "llm_calls"} <= tables
