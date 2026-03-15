# Scratchpad

Working notes, open questions, and in-progress plans. Clean out when work is completed.

---

## Current status — M8 complete

M8 Thompson sampling implemented, validated, and documented. Clean ablation (6 runs) shows both Thompson and greedy 3/3 correct. All documentation updated.

### Open issues
1. **SUSTAIN converges too fast** — even tau=0.005 can't slow it because predictions are categorically different. Not a bug — reflects genuine model distinctiveness.
2. **Debate still epiphenomenal to convergence** — parameter revisions produce replication variance (std≈0.018) but debate doesn't causally affect which model wins. Thompson stochastically selects debate-proposed novel structures, but this is random exploration, not debate-directed.

### Next steps
- Crux-directed experiment selection: bias Thompson weights toward crux-resolving experiments
- Claim-responsive debate: agents should address prior falsified claims
- New cognitive domains (memory retrieval, decision making)
- AutoRA integration for real data
- Longer runs (10+ cycles) to test cumulative reasoning
- WRITEUP.md Results section (still marked "[To be populated]")

---

## Failed approaches (do not repeat)

- **Pre-LOO prediction** — training and testing on same items gives GCM self-similarity bias (D11). Always use LOO.
- **Round-robin experiment selection** — doesn't optimize for discriminability, disadvantages models that are weak on common structures (RULEX on Type_VI).
- **Divergence ranking without per-model predictions** — agents can't interpret abstract divergence scores. They need to see which model wins on each structure.
- **Trusting LLM param overrides** — LLM agents invent parameter names (e.g., `w_i`). Always filter through `inspect.signature`.
- **Pure divergence-driven selection without diversity** — picks the same high-divergence structure every cycle (Type_VI). RULEX never gets tested on favorable structures.
- **EIG greedy optimization** — selects the same structure repeatedly (Phase 13). Thompson sampling (D34) fixes this.
- **Pairwise curve divergence as posterior evidence** — rewards model distinctiveness, not fit to data. Data-independent bonus distorts posterior. Removed in D35.
