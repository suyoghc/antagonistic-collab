# Scratchpad

Working notes, open questions, and in-progress plans. Clean out when work is completed.

---

## Current status — M12 complete and validated

M12 continuous design space complete: pool now ~427 candidates per cycle (11 base + 50 sampled × 7 conditions). 331 tests passing. Live validation: 3/3 correct, 15/15 sampled structures selected, 0% cycle overlap, gaps 77–96%.

### Key M12 insight: debate is interpretive, not directive

The computational layer (EIG + continuous sampling + Bayesian update) drives model identification with zero LLM calls. The debate layer provides interpretive value (mechanistic narratives, 80% FR rate) but 0/15 experiments came from agent proposals. This raises the question of whether debate improves identification beyond what bare EIG achieves. Three stances:
1. Debate was scaffolding — essential for designing the computational layer, now subsumed
2. Debate is complementary — computation handles selection, debate handles understanding
3. Debate needs harder problems — synthetic benchmarks are too easy; real data may reveal debate's value

### Open issues
1. **SUSTAIN converges too fast** — even tau=0.005 can't slow it because predictions are categorically different. Not a bug.
2. **Low crux-directed rate** — 1/15 experiments (6.7% vs 30% theoretical). Most cruxes reference structures already in the EIG frontier.
3. **Low format compliance** — 24/105 cruxes parsed (23%). Agents prefer semantic expressiveness over structured format.
4. **RPE never selected** — 40% of continuous samples are rule_plus_exception but EIG never picks them. Consider reducing the RPE fraction or investigating why.

### Possible next steps
- Ablation study: EIG-only (no debate) vs full system — does debate change identification outcomes?
- New cognitive domains (memory retrieval, decision making)
- AutoRA integration for real data
- Longer runs (10+ cycles) to test cumulative reasoning
- Adaptive sampling (use prior EIG results to focus sampling regions)

### WRITEUP.md status
- Sections 1–6 + Appendix A + References complete (~592 lines)
- **Deferred enhancement:** 4 items from LESSONS_LEARNED.md could strengthen the paper:
  1. Vivid confabulation example (agents interpreting constant 0.550 data) → Section 5.3
  2. Exact overclaiming numbers (claimed 0.75, actual 0.180) → Section 5.4
  3. Mock vs live crux acceptance (100% vs 15%) → Section 6.3
  4. "12 Theses" sharpest formulations (Thesis 1: "Argumentation without discriminating data is empty") → Section 6.1

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
