# Scratchpad

Working notes, open questions, and in-progress plans. Clean out when work is completed.

---

## M16 — Open Design Space — Complete (2026-03-18)

### Full results (15/15 runs, 15/15 correct)

| GT | Condition | Winner | OK? | RMSE | Gap% |
|---|---|---|---|---|---|
| GCM | closed_no_debate | Exemplar_Agent | Yes | 0.088 | 76.8 |
| GCM | closed_debate | Exemplar_Agent | Yes | 0.067 | 81.0 |
| GCM | closed_arbiter | Exemplar_Agent | Yes | 0.073 | 79.2 |
| GCM | open_debate | Exemplar_Agent | Yes | 0.084 | 71.6 |
| GCM | open_arbiter | Exemplar_Agent | Yes | 0.074 | 76.9 |
| SUSTAIN | closed_no_debate | Clustering_Agent | Yes | 0.057 | 87.7 |
| SUSTAIN | closed_debate | Clustering_Agent | Yes | 0.053 | 88.6 |
| SUSTAIN | closed_arbiter | Clustering_Agent | Yes | 0.020 | 96.0 |
| SUSTAIN | open_debate | Clustering_Agent | Yes | 0.108 | 64.1 |
| SUSTAIN | open_arbiter | Clustering_Agent | Yes | 0.100 | 70.7 |
| RULEX | closed_no_debate | Rule_Agent | Yes | 0.053 | 86.1 |
| RULEX | closed_debate | Rule_Agent | Yes | 0.168 | 58.6 |
| RULEX | closed_arbiter | Rule_Agent | Yes | 0.140 | 63.9 |
| RULEX | open_debate | Rule_Agent | Yes | 0.055 | 82.7 |
| RULEX | open_arbiter | Rule_Agent | Yes | 0.061 | 82.0 |

### Gap advantages (pp over closed_no_debate)

| GT | closed_debate | closed_arbiter | open_debate | open_arbiter |
|---|---|---|---|---|
| GCM | +4.2pp | +2.4pp | -5.2pp | +0.1pp |
| SUSTAIN | +0.9pp | **+8.3pp** | -23.6pp | -17.0pp |
| RULEX | -27.5pp | -22.2pp | -3.4pp | -4.1pp |

### Key insight: arbiter is a bias, not noise

Crux machinery steers toward similarity-based structures. This is a *feature* for
SUSTAIN/GCM and a *bug* for RULEX. Open design is the mirror bias: agent proposals
favor rule-diagnostic structures. Arbiter recovers open-design losses for similarity
models (GCM fully: -5.2pp→+0.1pp, SUSTAIN partially: -23.6pp→-17.0pp).

### Replication note (closed_arbiter)

SUSTAIN closed_arbiter replication: 95.6% gap (vs original 96.0%). Excellent consistency.

## M17 — Misspecification + Open Design — Complete (2026-03-19)

**6/6 correct.** Key findings:

| GT | open_debate | open_arbiter |
|---|---|---|
| GCM | 67.3% (42.9% recovery) | **87.8%** (85.7% recovery, best GCM ever) |
| SUSTAIN | 77.4% (0% recovery) | 72.7% (0% recovery) |
| RULEX | 57.8% (46.3% recovery) | 42.2% (0% recovery, but correct!) |

### Key insights
1. **GCM open_arbiter** — best GCM result ever (87.8%). Param recovery + arbiter
   proposals compose synergistically.
2. **Open design rescues RULEX from arbiter catastrophe** — M15 arbiter-RULEX was
   3.2% (wrong winner). M17 open_arbiter-RULEX is 42.2% (correct). Agent-proposed
   rule-diagnostic structures counteract the arbiter's similarity bias.
3. **Composition is non-additive** — effects don't add linearly across milestones.
4. **47/48 correct across M14–M17** — only M15 arbiter-RULEX fails.

### Open questions
1. Can crux bias be corrected? (diversity constraint on crux-directed selection)
2. R-IDeA as alternative OED type
3. Real data integration

---

## Failed approaches (do not repeat)

- **Pre-LOO prediction** — training and testing on same items gives GCM self-similarity bias (D11). Always use LOO.
- **Round-robin experiment selection** — doesn't optimize for discriminability, disadvantages models that are weak on common structures (RULEX on Type_VI).
- **Divergence ranking without per-model predictions** — agents can't interpret abstract divergence scores. They need to see which model wins on each structure.
- **Trusting LLM param overrides** — LLM agents invent parameter names (e.g., `w_i`). Always filter through `inspect.signature`.
- **Pure divergence-driven selection without diversity** — picks the same high-divergence structure every cycle (Type_VI). RULEX never gets tested on favorable structures.
- **EIG greedy optimization** — selects the same structure repeatedly (Phase 13). Thompson sampling (D34) fixes this.
- **Pairwise curve divergence as posterior evidence** — rewards model distinctiveness, not fit to data. Data-independent bonus distorts posterior. Removed in D35.
- **Multiplicative EIG boost for cruxes** — 2x multiplier barely shifts Thompson sampling when EIG scores cluster narrowly. Replaced with mixture distribution in D37.
- **Exact structure name matching for sampled structures** — ephemeral names change every cycle. Fixed with parameter-based fuzzy matching (D42).
