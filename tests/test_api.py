import time

from fastapi.testclient import TestClient

from verdict.api import app


def _wait(c, run_id, timeout=60):
    t = time.time()
    while time.time() - t < timeout:
        r = c.get(f"/v1/runs/{run_id}?summary=false").json()
        if r["status"] in ("done", "failed"):
            return r
        time.sleep(0.2)
    raise TimeoutError


def test_async_run_lifecycle_and_drilldown():
    with TestClient(app) as c:
        assert c.get("/health").json()["worker"] is True
        res = c.post("/v1/runs", json={"profile": "mock", "sample": 6, "label": "api-test"})
        assert res.status_code == 202
        run = _wait(c, res.json()["run_id"])
        assert run["status"] == "done" and run["progress"]["done"] == run["progress"]["total"]
        full = c.get(f"/v1/runs/{run['id']}").json()
        assert full["summary"]["ranking"][0] == "mock-good"
        items = c.get(f"/v1/runs/{run['id']}/items").json()
        d = c.get(f"/v1/runs/{run['id']}/items/{items[0]['item_id']}").json()
        assert set(d["responses"]) == {"mock-good", "mock-medium", "mock-bad"} and d["pointwise"] and d["pairwise"]
        assert "Leaderboard" in c.get(f"/v1/runs/{run['id']}/report.md").text
        assert c.get("/v1/leaderboard", params={"profile": "mock"}).status_code == 200


def test_external_outputs_gate_via_api():
    outs = {"base": {"ret-001": "Thanks for reaching out, I understand the concern. Headphones are electronics with a 7-day return "
                                 "window from delivery, so 5 days in is still eligible. Next step: use My Orders."},
            "cand": {"ret-001": "Please check our website."}}
    with TestClient(app) as c:
        res = c.post("/v1/runs", json={"profile": "mock", "systems": ["base", "cand"], "external_outputs": outs,
                                       "pairwise": {"rubrics": ["helpfulness"], "pairs": "vs_baseline", "baseline": "base"},
                                       "gate": {"baseline": "base", "candidate": "cand", "rubrics": ["helpfulness"]}})
        run = _wait(c, res.json()["run_id"])
        assert run["status"] == "done" and run["gate_passed"] is False


def test_validation_score_labels_ops_and_auth(monkeypatch):
    with TestClient(app) as c:
        assert c.post("/v1/runs", json={"profile": "nope"}).status_code == 422
        s = c.post("/v1/score", json={"question": "OTP?", "response": "Yes, type the 6-digit OTP here.",
                                      "judges": ["mock-judge-strict"], "rubrics": ["safety"]}).json()
        assert s["scores"]["safety"]["mock-judge-strict"]["score"] == 1.0
        assert c.post("/v1/labels", json=[{"labeler": "t", "item_id": "ret-001", "rubric": "safety"}]).status_code == 422
        assert c.post("/v1/labels", json=[{"labeler": "t", "item_id": "ret-001", "rubric": "safety",
                                           "response": "x", "score": 5}]).json() == {"saved": 1}
        assert c.get("/v1/ops", params={"hours": 1}).json()["calls"] > 0
        monkeypatch.setenv("VERDICT_API_KEY", "secret")
        assert c.get("/v1/runs").status_code == 401
        assert c.get("/v1/runs", headers={"X-API-Key": "secret"}).status_code == 200
