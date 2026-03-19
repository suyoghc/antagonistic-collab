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

## M17 — Misspecification + Open Design (in progress, 2026-03-18)

**Scientific question:** Do debate's parameter recovery (M15) and structure
proposal (M16) compose or interfere under the hardest regime?

### Design
- Combines M15 misspecified params with M16 open design space
- 6 new runs: 3 GTs × 2 open conditions (open_debate, open_arbiter)
- Compare against M15's 9 closed runs (already have data)
- Script: `scripts/validation/validate_m17_live.py --new-only`

### Predictions (before seeing data)
- **GCM open_debate (misspec):** Debate should help via param recovery (+3.5pp
  in M15 closed). Open design hurt GCM by -5.2pp in M16. Net effect unclear.
- **RULEX open_debate (misspec):** Best case scenario — debate's +22pp param
  recovery PLUS open design's +24pp RULEX advantage could compound.
- **SUSTAIN open_arbiter (misspec):** Worst case — arbiter hurt SUSTAIN -11.6pp
  in M15 AND open design hurt -23.6pp in M16. Could these compound?

### Open questions after M17
1. Can crux bias be corrected? (diversity constraint on crux-directed selection)
2. Arbiter debiasing: ensure cruxes also generate rule-diagnostic structures
3. R-IDeA as alternative OED type

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
