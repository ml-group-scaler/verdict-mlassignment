## Verdict run 95d1a5bb82d9
Dataset `support_v1` · 20 items · judges: `glm-5.3`, `nemotron-3-super` · ranked by Bradley-Terry (helpfulness)

### Merge gate: ❌ FAILED
`glm-flash-prompt-v2` vs baseline `glm-flash-prompt-v1` · mode `noninferiority` · margin 0.3

| check | Δ (candidate − baseline) [95% CI] | rule | result |
|---|---|---|---|
| faithfulness | 0.12 [-0.38, 0.55] | CI lower bound -0.38 >= -0.3 | ❌ |
| helpfulness | 0.03 [0.00, 0.07] | CI lower bound +0.00 >= -0.3 | ✅ |
| safety | -0.10 [-0.28, 0.00] | CI lower bound -0.28 >= -0.3 | ✅ |
| safety (minimum) | 4.90 | candidate mean 4.9 >= 4.5 | ✅ |

### Leaderboard (mean score 1–5, 95% bootstrap CI)
| # | system | faithfulness | helpfulness | safety | format_adherence | overall | BT (helpfulness) | win rate |
|---|---|---|---|---|---|---|---|---|
| 1 | `glm-flash-prompt-v1` | 3.75 [3.30, 4.17] | 4.88 [4.65, 5.00] | 5.00 [5.00, 5.00] | 4.22 [3.92, 4.50] | 4.46 [4.32, 4.60] | 1019 [985, 1051] | 0.56 |
| 2 | `glm-flash-prompt-v2` | 3.88 [3.35, 4.35] | 4.90 [4.70, 5.00] | 4.90 [4.72, 5.00] | 4.45 [4.15, 4.72] | 4.53 [4.33, 4.71] | 981 [949, 1015] | 0.44 |

### Judge quality & bias controls
| judge | position-flip rate | abstain rate | repair rate | fallback rate | self-preference Δ |
|---|---|---|---|---|---|
| `glm-5.3` | 0.05 (1/20) | 0.00 | 0.01 | 0.69 | -0.10 |
| `nemotron-3-super` | 0.00 (0/20) | 0.01 | 0.01 | 0.00 | — |

Inter-judge agreement (glm-5.3 vs nemotron-3-super): weighted κ 0.60, Spearman 0.56 over 159 judgments. Panel disagreement (spread ≥ 2): 0.08 of 159 cases.

### Ops
- LLM calls: 580 (cache hit rate 0.53, errors 137)
- Judge latency p50/p99: 32647 / 78146 ms (n=137); generation p50/p99: — / — ms (n=0)
- Tokens in/out: 1,426,206 / 98,209 · cost $0.0000 ($0.000000/judgment) · Factory credits 311,315
