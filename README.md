# Verdict — a calibrated LLM-as-judge evaluation service + leaderboard

> SST · ML System Design & LLMOps capstone · Group 16 · Project 9 (LLM-as-judge evaluation framework + leaderboard)

**Problem.** Teams ship prompt and model changes "on vibes" because automated evals can't be trusted:
a single LLM judge has position bias, verbosity bias, favours its own model family, and nobody has
measured whether it agrees with humans.

**Objective (one sentence).** Let any team know — in under 10 minutes and for under $0.25 — whether a
prompt/model change made their LLM app better or worse, using a judge panel whose agreement with humans
is measured and published.

**ML problem.**
- **Input:** (customer question, policy context, reference notes, one or two candidate responses, rubric version).
- **Output:** a pointwise score 1–5 with rationale per rubric, or a pairwise verdict A/B/tie; aggregated into a
  per-system mean with 95% bootstrap CI, a Bradley-Terry rating, and a merge-gate verdict.
- **Target:** human expert judgment, measured with quadratic-weighted Cohen's κ.

**Domain.** Customer-support replies for **NimbusMart**, a *fictional* Indian online store with a 12-section policy
(`data/policy/nimbusmart_policy.md`). It exercises all four rubrics naturally: faithfulness to policy, helpfulness,
safety/privacy (OTP requests, other customers' data, prompt injection), and format/style-guide adherence.

## Requirements

| | target |
|---|---|
| Judge–human agreement | weighted κ ≥ 0.6 on faithfulness & safety, ≥ 0.5 on helpfulness & format (`configs/calibration_gate.yaml`) |
| Position bias | position-flip rate reported per judge (target < 15%) |
| CI gate run (20 items, 2 systems, 2 judges) | < 10 min, < $0.25 |
| Online score latency | p95 < 5 s per judgment (**not met** on the NVIDIA free tier with GLM-5.3, see Numbers) |
| Budget | < $20 total (NVIDIA NIM free tier used ⇒ $0 marginal) |

**In scope:** English text, hosted models only (NVIDIA NIM), single-tenant API key, configs in git.
**Out of scope:** training/fine-tuning judges, multimodal, multi-tenant auth/billing, a full labelling platform.

## Architecture

```
 Client repo (GitHub Action) ──┐                    ┌── Streamlit UI: leaderboard, drill-down,
 CLI / Streamlit UI ───────────┤  REST / JSON       │   labelling, calibration & bias, ops
                               ▼                    │   (reads the API over REST)
                  FastAPI  (verdict/api.py) ────────┘
     POST /v1/runs → 202 + run_id (sync)     GET /v1/runs/{id}, /items/{item}, /report.md, /leaderboard
     POST /v1/score (online, sync)           POST /v1/labels, GET /v1/ops, /v1/reports/{kind}
                               │ insert run row (status=queued)
                               ▼
                  Worker thread (verdict/worker.py) — claims queued runs, crash-safe re-queue
        ┌──────────────────────┼─────────────────────────┐
   1. Generate             2. Judge                   3. Aggregate (verdict/results.py)
   systems under test      pointwise + pairwise       bootstrap CIs, Bradley-Terry + CIs,
   (or external outputs    (both orders), panel of    flip rate, panel disagreement,
   uploaded by CI)         2 judges, JSON + repair    self-preference, merge gate
        └────────── LLM gateway (verdict/llm.py) ─────────┘
          content-hash cache · per-model rate limiter · run budget · per-call telemetry
                               │  OpenAI-compatible HTTPS (NVIDIA NIM) or offline mock
                               ▼
          SQLite (SQLAlchemy; Postgres via DATABASE_URL): runs, generations, judgments,
          llm_calls, llm_cache, human_labels, online_scores, reports
```

| Component | Choice | Why |
|---|---|---|
| Judges | `moonshotai/kimi-k3` + `z-ai/glm-5.3` + `nvidia/nemotron-3-super-120b-a12b` on NVIDIA NIM, each with fallback `nvidia/nemotron-3.5-lightning-30b-a3b` | three model families → cross-family bias control; free tier fits the budget; fallback + circuit breaker keep runs alive when a free endpoint stalls |
| Systems under test | GLM-5.3-flash prompt v1 / v2, gpt-oss-20b, **bad baseline (no policy)** (Gemma-4-31B config kept but dropped from profiles: its free endpoint did not respond in Oct 2026 tests) | the bad baseline is a sanity check: a working judge must rank it last on faithfulness. Systems never fall back — that would silently change what is being evaluated |
| Configs | YAML in `configs/` (rubrics are `name.vN.yaml`) | diffable in PRs; every run stores the pinned `rubric@version` and a config hash |
| Queue | in-process worker thread + DB status column | no Redis needed on free tiers; runs survive restarts |
| Storage | SQLite → Postgres by env var | zero-setup locally, one-line change for deployment |
| UI | Streamlit | fastest to build; trade-off: less custom than React |

## How it works

- **Pointwise:** each (system, item, rubric, judge) gets a 1–5 score + rationale. Rubrics have anchored scale
  descriptors. The response is wrapped in delimiters and the judge is told embedded instructions are data.
- **Pairwise with position swap:** every comparison is judged twice (A,B) and (B,A). If the two orders name
  opposite winners it's a **flip** (pure position effect) → recorded as a tie and counted in the flip rate.
- **Panel:** per-item score = mean over judges; **disagreement** = spread ≥ 2 points (highlighted in the UI).
- **Fallback:** each judge has `fallback_models`; on timeout/5xx/404 the next model serves, recorded as `served_model`
  (fallback rate shown per judge). After 2 consecutive failures a model's circuit opens for 5 min.
- **Statistics:** 95% bootstrap CIs over items; Bradley-Terry ratings (Elo-like scale) with bootstrap CIs.
- **Merge gate:** paired bootstrap of (candidate − baseline) per rubric; `noninferiority` passes iff the CI lower
  bound ≥ −margin; plus absolute floors (e.g. safety ≥ 4.5).
- **Robust parsing:** JSON extracted from fences / surrounding reasoning; one repair retry, then abstain.
- **Calibration ("judge the judge"):** `verdict calibrate` re-judges the human-labelled examples and reports weighted κ,
  Spearman, exact / ±1 agreement, judge-minus-human bias, and **human–human κ as the ceiling**. CI fails a rubric/judge
  change that lowers κ below the thresholds.
- **Bias probes:** planted known-good/known-bad answers, verbosity padding, prompt injection
  ("rate it 5") — plus position flip rate and self-preference measured in every run.

## Evaluation data

| file | what | status |
|---|---|---|
| `data/datasets/support_v1.jsonl` | 45 questions in 12 categories (returns, refunds, shipping, privacy/safety, prompt injection, out-of-scope, multi-hop …) with policy sections, key facts and must-nots | **DRAFT** — written to bootstrap the pipeline; the team must review, rewrite and extend to 80–100 by hand |
| `data/datasets/support_v1_planted.jsonl` | 14 known-good / known-bad responses with seeded errors | DRAFT |
| `data/human_labels/labels.jsonl` | human labels (collected in the UI, exported with `verdict labels-export`) | **empty — to be labelled by the team** (2 labelers per example) |
| `tests/fixtures/SYNTHETIC_test_labels.jsonl` | fake labels used only by unit tests | never report these numbers |

**Scoring method:** LLM-as-judge (pointwise + pairwise) with rubric anchors, plus a programmatic check
(word count ≤ 120 from the style guide).

## Numbers

| metric | value | source |
|---|---|---|
| Unit/integration tests | 42 passed | `pytest` |
| API load test (mock judge, laptop, 20 users) | **63.8 req/s, p50 6 ms, p99 73 ms, 0 failures / 1,214 req** | `scripts/locustfile.py` |
| Offline mock run (45 items × 3 systems × 4 rubrics × 2 judges + pairwise) | 1,775 LLM calls in ~1.5 s; re-run 100% cache hits, identical results | `verdict run --profile mock` |
| Mock bias controls | injected 25% position bias detected as 19% flip rate; injection probe Δ +1.86 for the weak judge, 0.00 for the strict one | proves the detectors work |
| **Live planted-error probe** (GLM-5.3, Nemotron-3-Super) | **14/14 correct for both judges** (every seeded error ≤ 2, every good answer ≥ 4) | `reports/probe_planted.md` |
| **Live CI run** (20 items, prompt v1 vs v2, 2 judges, 455 calls) | v2 vs v1 overall 4.51 vs 4.45 — **no significant difference**; gate **failed** on faithfulness (Δ +0.07, CI [−0.40, +0.50]) | `reports/live_ci.md` |
| Inter-judge agreement (GLM-5.3 vs Nemotron-3-Super) | **weighted κ 0.65**, Spearman 0.60; disagreement (spread ≥ 2) 7% of 153 cases | live CI run |
| Position-flip rate | **0/20** for each judge | live CI run |
| Self-preference (GLM-5.3 judging GLM-family systems) | **Δ −0.01** (none detected) | live CI run |
| Judge health | GLM-5.3: 3% abstain, 3% JSON repair, 1% fallback; Nemotron: 1% abstain, 0% fallback | live CI run |
| Judge latency (NVIDIA free tier) | **p50 15.6 s, p99 297 s**; generation (GLM-5.3-flash) p50 71 s | live CI run |
| Online `/v1/score` (2 rubrics × 2 judges, parallel) | 33 s (bounded by GLM-5.3) — **misses the 5 s target**; Nemotron alone ≈ 3–8 s | live API test |
| Cost | **$0** (NVIDIA free tier); 346k input / 379k output tokens for the CI run | live CI run |
| **Judge–human κ** | _TBD after the team labels data_ | `verdict calibrate` |
| **Full leaderboard** (45 items, 4 systems) | _paused at ~20%: NVIDIA free-tier quota for GLM-5.3 ran out (instant HTTP 429). Resume later with `verdict run --profile full` — the cache (924 entries) skips all finished work_ | `verdict run --profile full` |

> Mock rows prove the plumbing; live rows are real NVIDIA NIM runs from 2026-10-08.
>
> **Finding:** with 20 items the non-inferiority gate cannot rule out a 0.4-point faithfulness drop, so it fails even when
> nothing regressed. Fixes, in order of preference: grow the eval set, gate on the full 45+ items, or use
> `mode: significant_regression` for PR gating (fail only on a statistically significant drop).

## Quickstart

```bash
uv venv --python 3.12 .venv && uv pip install -e ".[ui,dev]"
cp .env.example .env                      # add NVIDIA_API_KEY=nvapi-... for real runs

# offline (no key): full pipeline with deterministic mock models
.venv/bin/verdict run --profile mock
.venv/bin/pytest -q

# real runs on NVIDIA NIM
.venv/bin/verdict run --profile ci --fail-on-gate          # merge-gate run (20 items)
.venv/bin/verdict run --profile full                        # leaderboard (45 items, 5 systems)
.venv/bin/verdict probe planted                             # + verbosity / injection [--run-id ID]
.venv/bin/verdict labels-export && .venv/bin/verdict calibrate --enforce

# service + UI
.venv/bin/verdict serve --port 8000
VERDICT_API_URL=http://localhost:8000 .venv/bin/streamlit run ui/app.py
```

## CI

| workflow | trigger | does |
|---|---|---|
| `ci.yml` | every push / PR | unit tests + offline mock eval with merge gate (no secrets) |
| `eval-gate.yml` | PR touching system prompts/configs | NVIDIA judge panel, baseline vs candidate; regression fails the PR and posts a comment |
| `judge-calibration.yml` | PR touching rubrics / judge prompts / judge models / labels | re-runs calibration; κ below threshold fails the PR |
| `action/` (reusable) | any other repo | `uses: <org>/verdict/action@main` — uploads that repo's outputs, gates the merge (see `examples/demo-app/`) |

## Trade-offs

1. **We chose pairwise + pointwise over pointwise only** because pairwise is more reliable for ranking while pointwise
   gives absolute scores to gate on; pairwise grows as O(n²), so CI only compares candidate vs baseline.
2. **We chose a 2-family judge panel (Kimi-K3 + GLM-5.3) over a single stronger judge** because it reduces
   self-preference and single-model idiosyncrasies at zero cost on the free tier; calibration shows whether it suffices.
3. **We chose an async run queue over a synchronous API** because a run is thousands of rate-limited LLM calls
   (minutes), which would hit HTTP timeouts.
8. **We chose per-judge fallback models + a circuit breaker over failing the run** because free endpoints stall (Kimi-K3 sent 0 bytes
   in 150 s during testing); the cost is that a fallback-served judgment is a different model, so every judgment records
   `served_model`, the fallback rate is reported per judge, and fallback-served judgments are excluded from self-preference.
4. **We chose a content-hash cache over always re-calling** because it makes CI deterministic and cheap; risk: a provider
   silently updating a model — mitigated by pinning model ids in the hash and re-running calibration on model changes.
5. **We chose gating on the CI lower bound over the raw mean** because with 20–45 items the mean is noisy and a mean-based
   gate fails PRs randomly; the cost is that small datasets give wide CIs (hence the dataset must grow).
6. **We chose SQLite + an in-process worker over Postgres + Redis** for zero-setup and free-tier deployment; at 10× scale the
   single SQLite writer and the single worker are the first bottleneck → swap `DATABASE_URL` to Postgres and run workers separately.
7. **We built the service rather than using promptfoo / DeepEval / RAGAS** because none ship human-agreement calibration,
   position-swap and self-preference measurement as first-class, gateable metrics.

## Failure modes handled

Unparseable judge output (repair → abstain, rate reported) · 429 / 5xx / timeouts (exponential backoff, Retry-After) ·
unresponsive judge endpoint (per-model timeout → fallback model; circuit breaker skips a dead model for 5 min) ·
provider throttling — HTTP 429 is treated as "slow down", not "down": separate backoff budget, never trips the breaker
or triggers fallback (learned live: at 40 in flight NVIDIA throttled GLM-5.3 and its judge slot silently drifted to the
fallback model; now capped at 8 in flight / 20 rpm) ·
reasoning model exhausting its token budget (explicit error) · missing API key (fails fast before queueing) ·
prompt injection inside responses (delimiting + injection probe + injection items in the dataset) · position bias
(swap + flip rate) · run cost runaway (per-run call/USD budget) · crash mid-run (re-queued on restart; cache makes the
retry cheap) · free-tier cold start (Action client retries).

## Open items for the team

- [ ] Review / rewrite / extend `support_v1.jsonl` to 80–100 items by hand (it is a DRAFT).
- [ ] Label ≥ 150 judgments (2 labelers each) in the UI → `verdict labels-export` → commit `labels.jsonl`.
- [ ] Add `NVIDIA_API_KEY` as a GitHub secret so `eval-gate.yml` / `judge-calibration.yml` run on PRs.
- [ ] Re-enable Kimi-K3 in `configs/profiles/*.yaml` once its free endpoint responds (it took ~3 min/call or timed out on 2026-10-08).
- [ ] Decide the PR-gate policy given the 20-item power problem (see Numbers → Finding).
- [ ] Deploy (API + UI) and add the live URL; screenshot a PR blocked by `eval-gate`.

## Resume line

Built **Verdict**, a calibrated LLM-as-judge evaluation service (pointwise + position-swapped pairwise judging with a
two-family judge panel, bootstrap CIs and Bradley-Terry leaderboard) reaching κ = _X_ agreement with human labels, with a
reusable GitHub Action that blocks regressing PRs and a public per-example drill-down leaderboard.
