# Scratchpad

Working notes, open questions, and in-progress plans. Clean out when work is completed.

---

## Current focus — M9 crux-directed Thompson sampling

Mixture distribution implemented and all unit tests pass (336). The crux prompt now shows the structure/condition menu so agents can produce parseable format. Pending live validation to confirm crux-directed selections appear in real runs.

### Pending
- **Live validation** — Run 5-cycle debate with GPT-4o and verify that (a) cruxes are parsed into boost specs, (b) some experiments are crux-directed, (c) correct model still wins.

### Key design choice
Mixture distribution with `crux_weight=0.3`:
- 30% of experiments are crux-directed (if active cruxes match pool entries)
- 70% follow standard EIG-weighted Thompson
- When no cruxes are active, 100% EIG-weighted (graceful fallback)

### What this changes about the debate's role
Pre-M9: debate was epiphenomenal to experiment selection. Cruxes existed but never affected which experiment ran (parsing always failed + multiplicative boost ineffective).
Post-M9: debate causally affects experiment selection. Accepted cruxes directly influence which experiment is run ~30% of the time. This is the first time the debate structure shapes the quantitative pipeline.

### Open issues
1. **SUSTAIN converges too fast** — even tau=0.005 can't slow it because predictions are categorically different. Not a bug.
2. **Agents may still write free-text crux experiments** — the prompt improvement may not be sufficient. If live validation shows parsing still fails, consider post-hoc fuzzy matching.

### Next steps (post-validation)
- Claim-responsive debate: agents should address prior falsified claims
- New cognitive domains (memory retrieval, decision making)
- AutoRA integration for real data
- Longer runs (10+ cycles) to test cumulative reasoning
- WRITEUP.md Results section

---

## Failed approaches (do not repeat)

- **Pre-LOO prediction** — training and testing on same items gives GCM self-similarity bias (D11). Always use LOO.
- **Round-robin experiment selection** — doesn't optimize for discriminability, disadvantages models that are weak on common structures (RULEX on Type_VI).
- **Divergence ranking without per-model predictions** — agents can't interpret abstract divergence scores. They need to see which model wins on each structure.
- **Trusting LLM param overrides** — LLM agents invent parameter names (e.g., `w_i`). Always filter through `inspect.signature`.
- **Pure divergence-driven selection without diversity** — picks the same high-divergence structure every cycle (Type_VI). RULEX never gets tested on favorable structures.
- **EIG greedy optimization** — selects the same structure repeatedly (Phase 13). Thompson sampling (D34) fixes this.
- **Pairwise curve divergence as posterior evidence** — rewards model distinctiveness, not fit to data. Data-independent bonus distorts posterior. Removed in D35.
- **Multiplicative EIG boost for cruxes** — 2× multiplier barely shifts Thompson sampling when EIG scores cluster narrowly. Replaced with mixture distribution in D37.
