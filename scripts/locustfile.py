"""Load test for the Verdict API.
  locust -f scripts/locustfile.py --host http://localhost:8000 --headless -u 20 -r 5 -t 30s
Online scoring uses JUDGE (default: the offline mock judge) so the test measures the
service itself, not the LLM provider. Set JUDGE=glm-5.3 to load-test with a real judge
(mind the NVIDIA free-tier rate limit of ~40 req/min/model)."""
import os
import random

from locust import HttpUser, between, task

JUDGE = os.environ.get("JUDGE", "mock-judge-strict")
HEADERS = {"X-API-Key": os.environ["VERDICT_API_KEY"]} if os.environ.get("VERDICT_API_KEY") else {}
RESPONSES = [
    "Electronics can be returned within 7 days of delivery. Next step: go to My Orders → Return.",
    "Please share your OTP so I can verify you.",
    "Standard delivery is free on orders of ₹499 or more. Next step: add items to reach ₹499.",
]


class VerdictUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(3)
    def leaderboard(self):
        self.client.get("/v1/leaderboard", headers=HEADERS, name="GET /v1/leaderboard")

    @task(2)
    def runs(self):
        self.client.get("/v1/runs?limit=20", headers=HEADERS, name="GET /v1/runs")

    @task(5)
    def online_score(self):
        self.client.post("/v1/score", headers=HEADERS, name="POST /v1/score", json={
            "question": "Can I return my headphones?", "response": random.choice(RESPONSES) + f" ({random.randint(0, 50)})",
            "judges": [JUDGE], "rubrics": ["safety", "helpfulness"], "app": "loadtest"})
