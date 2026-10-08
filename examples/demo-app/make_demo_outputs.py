"""Builds demo outputs files for exercising the GitHub Action locally against the mock profile:
  outputs_ok.json         candidate == a good bot      -> gate should PASS
  outputs_regression.json candidate == a degraded bot  -> gate should FAIL
Usage: python examples/demo-app/make_demo_outputs.py"""
import json
from pathlib import Path

from verdict.config import registry
from verdict.providers.mock import MockProvider

here = Path(__file__).parent
ds = registry().dataset("support_v1")
ans = {q: {it.id: MockProvider._answer(q, it.question) for it in ds.items} for q in ("good", "medium", "bad")}
(here / "outputs_ok.json").write_text(json.dumps({"baseline": ans["medium"], "candidate": ans["good"]}, indent=1, ensure_ascii=False))
(here / "outputs_regression.json").write_text(json.dumps({"baseline": ans["good"], "candidate": ans["bad"]}, indent=1, ensure_ascii=False))
print("wrote", here / "outputs_ok.json", "and", here / "outputs_regression.json")
