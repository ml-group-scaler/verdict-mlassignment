"""Regenerates SYNTHETIC_test_labels.jsonl — FAKE "human" labels used ONLY by the test suite
to exercise the calibration code path. They are derived from a heuristic, not from people.
NEVER report numbers computed from this file. Real labels live in data/human_labels/labels.jsonl."""
import hashlib
import json
from pathlib import Path

from verdict.config import registry
from verdict.providers.mock import MockProvider, coverage

ds = registry().dataset("support_v1")
rows = []
for it in ds.items[:15]:
    ref = ds.reference_for(it)
    for q in ("good", "medium", "bad"):
        resp = MockProvider._answer(q, it.question)
        base = max(1, min(5, round(1 + 4 * coverage(resp, ref))))
        for labeler in ("synthetic-A", "synthetic-B"):
            h = int(hashlib.sha256(f"{labeler}{it.id}{q}".encode()).hexdigest()[:4], 16) % 10
            score = max(1, min(5, base + (1 if h == 0 else -1 if h == 1 else 0)))  # ~20% off-by-one disagreement
            rows.append({"kind": "pointwise", "synthetic": True, "item_id": it.id, "rubric": "helpfulness",
                         "system_id": f"mock-{q}", "response": resp, "labeler": labeler, "score": score})
    good, bad = MockProvider._answer("good", it.question), MockProvider._answer("bad", it.question)
    for labeler in ("synthetic-A", "synthetic-B"):
        rows.append({"kind": "pairwise", "synthetic": True, "item_id": it.id, "rubric": "helpfulness",
                     "system_a": "mock-good", "system_b": "mock-bad", "response_a": good, "response_b": bad,
                     "labeler": labeler, "winner": "A"})
out = Path(__file__).parent / "SYNTHETIC_test_labels.jsonl"
out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
print(f"wrote {len(rows)} SYNTHETIC rows to {out}")
