# Scratchpad

Working notes, open questions, and in-progress plans. Clean out when work is completed.

---

## M16 — Open Design Space — Phase 2 Complete (2026-03-18)

### Full results (14/15 runs, 1 error)

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
| SUSTAIN | open_arbiter | — | ERROR | — | — |
| RULEX | closed_no_debate | Rule_Agent | Yes | 0.053 | 86.1 |
| RULEX | closed_debate | Rule_Agent | Yes | 0.168 | 58.6 |
| RULEX | closed_arbiter | Rule_Agent | Yes | 0.140 | 63.9 |
| RULEX | open_debate | Rule_Agent | Yes | 0.055 | 82.7 |
| RULEX | open_arbiter | Rule_Agent | Yes | 0.061 | 82.0 |

### Gap advantages (pp over closed_no_debate)

| GT | closed_debate | closed_arbiter | open_debate | open_arbiter |
|---|---|---|---|---|
| GCM | +4.2pp | +2.4pp | -5.2pp | +0.1pp |
| SUSTAIN | +0.9pp | **+8.3pp** | -23.6pp | ERROR |
| RULEX | -27.5pp | -22.2pp | -3.4pp | -4.1pp |

### Key insight: arbiter is a bias, not noise

Crux machinery steers toward similarity-based structures. This is a *feature* for
SUSTAIN/GCM and a *bug* for RULEX. Open design is the mirror bias: agent proposals
favor rule-diagnostic structures.

### Open items

1. **Fix SUSTAIN open_arbiter bug** — "setting an array element with a sequence".
   Need to reproduce and trace.
2. **Commit results + docs** — all MDs updated, need to commit
3. **Consider next steps:**
   - Can crux bias be corrected? (diversity constraint on crux-directed selection)
   - M15+M16 combined: misspecification + open design space
   - Arbiter debiasing: ensure cruxes also generate rule-diagnostic structures

### Commits this session (prior)
1. `1529202` — `test(M15): harden M15 code paths — 19 tests + param_distance extraction`
2. `aa67cae` — `feat(M16): open design space — agents propose all structures via debate`

### Files modified this session
- `scripts/validation/validate_m16_live.py` — added arbiter parameter, 5 conditions,
  `--arbiter-only` / `--new-only` CLI flags, meta-agent creation, crux weight restore
- `CURRENT_STATE.md` — full M16 Phase 1+2 results, revised theory
- `ROADMAP.md` — M16 promoted to latest, findings updated
- `Notes/archive/DECISIONS.md` — D46 (arbiter addition to M16)

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
