#!/usr/bin/env python3
"""Stdlib-only client used by the GitHub Action (no pip install on the runner).
Submits outputs to Verdict, waits for the run, writes the markdown report and sets outputs."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


def call(method: str, path: str, body: dict | None = None, raw: bool = False, retries: int = 6):
    url = os.environ["VERDICT_API_URL"].rstrip("/") + path
    headers = {"Content-Type": "application/json"}
    if os.environ.get("VERDICT_API_KEY"):
        headers["X-API-Key"] = os.environ["VERDICT_API_KEY"]
    data = json.dumps(body).encode() if body is not None else None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers, method=method), timeout=60) as r:
                payload = r.read().decode()
                return payload if raw else json.loads(payload)
        except urllib.error.HTTPError as e:
            if e.code < 500:
                sys.exit(f"Verdict API {method} {path} -> {e.code}: {e.read().decode()[:500]}")
            err = e
        except (urllib.error.URLError, TimeoutError) as e:  # free-tier hosts sleep: wait for the cold start
            err = e
        time.sleep(min(5 * 2 ** attempt, 60))
    sys.exit(f"Verdict API unreachable after {retries} attempts: {err}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-file", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--profile", default="ci")
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--gate-rubrics", default="faithfulness,helpfulness,safety")
    ap.add_argument("--min-scores", default="")
    ap.add_argument("--timeout-minutes", type=float, default=20)
    ap.add_argument("--report", default="verdict_report.md")
    ap.add_argument("--no-fail", action="store_true", help="always exit 0 (the action fails in a later step)")
    a = ap.parse_args()

    outputs = json.load(open(a.outputs_file))
    for s in (a.baseline, a.candidate):
        if s not in outputs:
            sys.exit(f"{s!r} not found in {a.outputs_file} (keys: {list(outputs)})")
    min_scores = {k: float(v) for k, v in (x.split("=") for x in a.min_scores.split(",") if x.strip())}
    body = {
        "profile": a.profile,
        "systems": [a.baseline, a.candidate],
        "external_outputs": {a.baseline: outputs[a.baseline], a.candidate: outputs[a.candidate]},
        "pairwise": {"rubrics": ["helpfulness"], "pairs": "vs_baseline", "baseline": a.baseline},
        "gate": {"baseline": a.baseline, "candidate": a.candidate, "mode": "noninferiority", "margin": a.margin,
                 "rubrics": [r.strip() for r in a.gate_rubrics.split(",") if r.strip()], "min_scores": min_scores},
        "label": os.environ.get("GITHUB_REF_NAME") or "ci",
    }
    run_id = call("POST", "/v1/runs", body)["run_id"]
    print(f"Verdict run {run_id} submitted")
    deadline = time.time() + a.timeout_minutes * 60
    while True:
        r = call("GET", f"/v1/runs/{run_id}?summary=false")
        print(f"  {r['status']} {r['progress']['done']}/{r['progress']['total']}")
        if r["status"] in ("done", "failed"):
            break
        if time.time() > deadline:
            sys.exit(f"timed out waiting for run {run_id}")
        time.sleep(5)
    if r["status"] == "failed":
        md = f"## Verdict eval failed\n\nRun `{run_id}` failed: `{r['error']}`\n"
        passed = False
    else:
        md = call("GET", f"/v1/runs/{run_id}/report.md", raw=True)
        passed = bool(call("GET", f"/v1/runs/{run_id}")["summary"]["gate"]["passed"])
    open(a.report, "w").write(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write(md)
    if os.environ.get("GITHUB_OUTPUT"):
        open(os.environ["GITHUB_OUTPUT"], "a").write(f"run-id={run_id}\npassed={'true' if passed else 'false'}\n")
    print(md)
    if not passed and not a.no_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
