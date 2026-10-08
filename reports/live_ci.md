## Verdict run 1ceb4cf2451f
Dataset `support_v1` · 20 items · judges: `glm-5.3`, `nemotron-3-super` · ranked by Bradley-Terry (helpfulness)

### Merge gate: ❌ FAILED
`glm-flash-prompt-v2` vs baseline `glm-flash-prompt-v1` · mode `noninferiority` · margin 0.3

| check | Δ (candidate − baseline) [95% CI] | rule | result |
|---|---|---|---|
| faithfulness | 0.07 [-0.40, 0.50] | CI lower bound -0.40 >= -0.3 | ❌ |
| helpfulness | 0.00 [-0.07, 0.07] | CI lower bound -0.07 >= -0.3 | ✅ |
| safety | -0.03 [-0.07, 0.00] | CI lower bound -0.07 >= -0.3 | ✅ |
| safety | 4.97 | candidate mean 4.97 >= 4.5 | ✅ |

### Leaderboard (mean score 1–5, 95% bootstrap CI)
| # | system | faithfulness | helpfulness | safety | format_adherence | overall | BT (helpfulness) | win rate |
|---|---|---|---|---|---|---|---|---|
| 1 | `glm-flash-prompt-v1` | 3.75 [3.27, 4.20] | 4.88 [4.65, 5.00] | 5.00 [5.00, 5.00] | 4.17 [3.83, 4.48] | 4.45 [4.29, 4.59] | 1010 [970, 1042] | 0.53 |
| 2 | `glm-flash-prompt-v2` | 3.83 [3.27, 4.30] | 4.88 [4.65, 5.00] | 4.97 [4.92, 5.00] | 4.38 [4.05, 4.70] | 4.51 [4.30, 4.71] | 990 [958, 1030] | 0.47 |

### Judge quality & bias controls
| judge | position-flip rate | abstain rate | repair rate | fallback rate | self-preference Δ |
|---|---|---|---|---|---|
| `glm-5.3` | 0.00 (0/20) | 0.03 | 0.03 | 0.01 | -0.01 |
| `nemotron-3-super` | 0.00 (0/20) | 0.01 | 0.01 | 0.00 | — |

Inter-judge agreement (glm-5.3 vs nemotron-3-super): weighted κ 0.65, Spearman 0.60 over 153 judgments. Panel disagreement (spread ≥ 2): 0.07 of 153 cases.

### Ops
- LLM calls: 455 (cache hit rate 0.00, errors 14)
- Judge latency p50/p99: 15620 / 296563 ms (n=395); generation p50/p99: 71440 / 228026 ms (n=40)
- Tokens in/out: 346,228 / 379,291 · cost $0.0000 ($0.000000/judgment)
