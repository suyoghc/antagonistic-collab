# Scratchpad

Working notes, open questions, and in-progress plans. Clean out when work is completed.

---

## Current status (2026-03-19)

M14–M17 factorial complete. 47/48 correct across all conditions.
R-IDeA standalone tested — doesn't beat EIG in either regime.

### R-IDeA + debate (in progress)

Testing whether R-IDeA experiment selection + debate parameter recovery
composes better than EIG + debate. Uses monkeypatch approach:
`validate_ridea_debate.py` swaps `bayesian_selection.select_from_pool`
at runtime with an R-IDeA version, keeping all 9 debate phases intact.
No changes to main codebase — if results are positive, integrate properly
via `_OED_TYPE` global in runner.py.

R-IDeA results (all conditions):
- Correct spec, no debate: EIG 86.9% > R-IDeA 80.5%
- Misspec, no debate: EIG 75.1% > R-IDeA 65.4%
- Misspec, R-IDeA + debate: **53.7% mean — worst condition tested**
  RULEX drops to 19.4% (vs 80.4% with EIG+debate). R-IDeA's
  representativeness term steers away from diagnostic experiments,
  preventing the visible prediction failures debate needs for param recovery.
- **Conclusion: EIG + debate (81.4%, 3.3% std) remains the gold standard.**
  Informativeness + semantic diagnosis compose; diversification + diagnosis
  are antagonistic.

### Next directions

1. **Additional complementary biases** — learning-curve-directed selection
   (targets SUSTAIN's temporal dynamics, currently underserved), falsification-directed
   selection (anti-confirmation, adaptive), random injection (Dubova-inspired debiaser).
   Learning curve selection is highest priority — data already computed, just not used
   for selection.
2. **Real data integration** — human participants via Prolific/AutoRA.
3. **Write paper** — synthetic story is self-contained through R-IDeA negative result.
4. **Griffiths connections** — see `New Ideas/tomgriffits.md` for 21 questions.

### Documentation produced this session (2026-03-19)

- M17 results: 6/6 correct (misspec + open design)
- WRITEUP.md Section 5.7: open design results with 6 new tables
- 5 publication-quality figures in `figures/`
- `New Ideas/Reflections_M17.md`: philosophy of science analysis + per-paper
  idea tracker (15 solved, 15 new)
- `New Ideas/tomgriffits.md`: 21 research questions across 8 Griffiths clusters

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
