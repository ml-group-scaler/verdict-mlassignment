# Verdict — 15-minute presentation outline (DRAFT for the team to adapt)

Rubric weights: Problem framing 20% · **Architecture 30%** · LLMOps depth 20% · Trade-offs 20% · Presentation 10%.
Seven speakers, ~2 min each. Every number below comes from `README.md` → *Numbers* (live NVIDIA runs, 2026-10-08/11).
Replace the TBD calibration slide once the team's human labels exist.

| # | min | speaker | slide | must say (rubric item) |
|---|---|---|---|---|
| 1 | 0:00–1:30 | S1 | **Problem** | One-sentence objective: *let a team know in < 10 min whether a prompt/model change made their LLM app better or worse, using judges whose agreement with humans is measured.* Why plain LLM judges can't be trusted: position bias, verbosity bias, self-preference, no human agreement. |
| 2 | 1:30–2:30 | S1 | **ML problem + requirements** | Input (question, policy context, reference notes, response(s), rubric@version) → output (1–5 score + rationale / A-B-tie) → aggregated (mean + 95% CI, Bradley-Terry, gate verdict). Target = human judgment (weighted κ). Requirements with numbers (κ ≥ 0.6, < 10 min CI run, < $20 budget) and in/out of scope. |
| 3 | 2:30–4:30 | S2 | **Architecture diagram** | UI → REST/JSON → FastAPI → queued run → worker → generate / judge / aggregate → LLM gateway → NVIDIA NIM (+ Droid fallback) → SQLite. Label each arrow: sync (`POST /v1/runs` → 202), async (worker), polling (`GET /v1/runs/{id}`). |
| 4 | 4:30–6:00 | S2 | **How a judgment is made** | Rubric YAML with anchors · response wrapped as data (injection-resistant) · JSON + one repair retry · pairwise judged in **both orders** (flip = position bias) · two-family panel · bootstrap CIs. |
| 5 | 6:00–7:00 | S3 | **LLM gateway** | Content-hash cache (re-run = free, deterministic), per-model rate limit + in-flight cap, circuit breaker, fallback chain that serves the **same model via Factory Droid**, budget cap, per-call telemetry. |
| 6 | 7:00–8:30 | S4 | **Results (live)** | Leaderboard: prompt v2 vs v1 = 4.53 vs 4.46, **no significant difference, replicated in two runs**. Planted-error probe 14/14 for both judges. Inter-judge κ 0.60–0.65, position flips 0–1/20, no self-preference. |
| 7 | 8:30–9:30 | S4 | **Live demo** | Streamlit link → Leaderboard → Explore run with *judge disagreements only* (e.g. `dmg-002`) → show both judges' rationales. |
| 8 | 9:30–11:00 | S5 | **LLMOps depth** | Deployment (Docker; embedded vs remote), rollout/rollback of rubric versions, monitoring table (operational, input, output, quality, drift), offline + online (`/v1/score`) eval, CI workflows + reusable GitHub Action. *Insert the screenshot of a PR blocked by `eval-gate`.* |
| 9 | 11:00–12:30 | S6 | **Trade-offs (≥ 3, "we chose X over Y because Z")** | Pairwise+pointwise over one; 2-family panel over one big judge; async queue over sync API; CI lower-bound gate over raw mean; SQLite+in-process worker over Postgres+Redis; fallback to the same model over a different one. |
| 10 | 12:30–14:00 | S7 | **Failure modes we hit live** | Kimi-K3 ~3 min/call → paused; GLM-5.3 ran out of reasoning tokens (4096 → 8192); NVIDIA 429 throttling silently drifted a judge to the fallback → separate 429 budget + quota breaker; NVIDIA GLM hangs → Droid fallback (run went from ~3.5 h to 11 min). **The gate is underpowered at 20 items** (CI [−0.38, +0.55]) → grow the eval set. |
| 11 | 14:00–15:00 | S7 | **Calibration + what's next** | Judge–human κ per rubric vs human–human κ (*TBD: fill from `reports/calibration.md`*). Next: 80–100 hand-written items, Kimi back in the panel, `significant_regression` gate mode for PRs. |

## Likely Q&A (prepare an owner for each)
- *Who judges the judge?* → human-label calibration + planted-error probe + CI gate on κ.
- *Why trust κ from ~150 labels?* → report human–human κ as the ceiling and the CI; more labels narrow it.
- *What breaks first at 10× scale?* → provider rate limits (NVIDIA ~35 rpm/model, 429s seen live) and O(n²) pairwise cost; then the single SQLite writer / in-process worker.
- *Why not one GPT-class judge?* → cost + self-preference; panel agreement is measured, not assumed.
- *Your gate failed — is v2 worse?* → no: Δ +0.12, but 20 items can't rule out a 0.4 drop; that's the gate being honest.
- *What does the fallback do to your bias stats?* → `served_model` per judgment, fallback rate per judge, fallback-served judgments excluded from self-preference.

## Peer review (participation-graded)
Write your own reviews of the two assigned teams during their talks — **AI-generated reviews get negative marks**, so this
part must be your own words.
