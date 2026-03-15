# Scratchpad

Working notes, open questions, and in-progress plans. Clean out when work is completed.

---

## Current focus — M7 Validation Complete

5-cycle validation done: 2/3 correct (GCM, SUSTAIN). RULEX misidentified as GCM (8.2% RMSE gap — genuine model overlap). 308 tests passing.

### Open issues from validation

1. **RULEX misidentification** — GCM approximates RULEX via attention weights (Nosofsky 1991). Need structures that specifically discriminate these two (e.g., structures where rule discovery is discrete vs gradual, verbal load manipulations).

2. **EIG lacks exploration diversity** — selected same structure 5/5 times for GCM and SUSTAIN runs. Greedy EIG is locally optimal but doesn't explore. Candidate fix: structure diversity bonus or forced coverage.

3. **SUSTAIN converges too fast** — even tau=0.005 can't slow it because predictions are categorically different. Not a bug — reflects genuine model distinctiveness.

### Next steps
- EIG diversity bonus to improve experiment selection variety
- Claim-responsive debate: agents should address prior falsified claims
- New cognitive domains (memory retrieval, decision making)
- AutoRA integration for real data
- Longer runs (10+ cycles) to test cumulative reasoning

---

## Failed approaches (do not repeat)

- **Pre-LOO prediction** — training and testing on same items gives GCM self-similarity bias (D11). Always use LOO.
- **Round-robin experiment selection** — doesn't optimize for discriminability, disadvantages models that are weak on common structures (RULEX on Type_VI).
- **Divergence ranking without per-model predictions** — agents can't interpret abstract divergence scores. They need to see which model wins on each structure.
- **Trusting LLM param overrides** — LLM agents invent parameter names (e.g., `w_i`). Always filter through `inspect.signature`.
- **Pure divergence-driven selection without diversity** — picks the same high-divergence structure every cycle (Type_VI). RULEX never gets tested on favorable structures.
- **EIG greedy optimization** — selects the same structure repeatedly (Phase 13). Need diversity incentive for multi-cycle runs.
