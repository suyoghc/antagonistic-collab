# Scratchpad

Working notes, open questions, and in-progress plans. Clean out when work is completed.

---

## Current status (2026-03-20)

M14–M17 factorial complete (categorization). 47/48 correct.
R-IDeA tested — negative result in all regimes.
Decision-making domain implementation **in progress**.

### Decision-Making Domain — IN PROGRESS

**Goal:** Replicate the implicit-prior/complementary-bias findings in a second
domain (decision-making under risk) to elevate the paper from CogSci to NeurIPS.

**Three models implemented and tested (17/17 tests pass):**
- EU (Expected Utility) ↔ SUSTAIN — normative baseline, 1 param (r)
- CPT (Cumulative Prospect Theory) ↔ GCM — dominant descriptive, 5 params
- Priority Heuristic ↔ RULEX — lexicographic rules, 0-1 params

**Gamble structure registry built (76 problems):**
- 17 base diagnostic problems (certainty effect, common ratio, fourfold pattern,
  loss aversion, risk premium, PH-specific)
- 59 parametric variants (cert-vs-risky, mixed, risky pairs)

**Synthetic runner working:** 3/3 correct on base registry with clear gaps.

**Pipeline wiring — ALL THREE COMPLETE:**
1. ~~**EIG adapter**~~ ✓ — `decision_eig.py` bridges gamble predictions to
   `compute_eig()`. 7 gamble groups as candidate experiments. 3/3 correct
   winners after 5 EIG cycles. 23 tests.
2. ~~**Validation script (no-debate)**~~ ✓ — `validate_decision_m14.py`
   running computation-only baseline. Results:
   - Correct params: 3/3 (converges cycle 0-1)
   - Misspecified params: 0/3 (all wrong — stronger penalty than categorization)
3. ~~**Agent configs**~~ ✓ — `decision_agents.py` with CPT_Agent, EU_Agent,
   PH_Agent system prompts. 18 tests.

**What's next: Standalone decision debate runner (Option C)**

Architecture decision (D49): standalone runner rather than refactoring or
subclassing DebateProtocol. Rationale: zero risk to 47/48 categorization
results, and DebateProtocol is tightly coupled to categorization (LOO,
STRUCTURE_REGISTRY, learning curves, condition effects).

The standalone runner reuses domain-agnostic pieces:
- `compute_eig()` and `ModelPosterior` from bayesian_selection.py
- `decision_predictions_for_eig()` from decision_eig.py
- `generate_synthetic_choices()` from decision_runner.py
- Agent configs from decision_agents.py

New code needed (~200 lines):
1. **LLM debate round** — show agents prediction errors, ask for parameter
   diagnosis + revision proposals
2. **Parameter validation** — filter revisions through `inspect.signature`
   (pattern from categorization runner.py)
3. **Arbiter round** (optional) — identify cruxes, steer experiment selection
4. **Cycle loop** — wire: EIG select → data → score → debate → param update → repeat

Calibration notes:
- learning_rate=0.01 (calibrated: converges over 3-5 cycles, leaves room
  for debate; lr=0.1 collapses after cycle 0)
- n_subjects=30 per gamble
- 7 gamble groups as candidate experiments

**Predictions to test:**
- Debate should recover misspec (0/3 → 2-3/3) via parameter diagnosis
- Arbiter should bias toward CPT/EU (smooth gradients) over PH (discrete rules)
- LLM proposals should favor PH (easy to articulate lexicographic rules)
- If pattern replicates across both domains → principle about representational
  format, not domain content → NeurIPS paper

**Key files:**
- `antagonistic_collab/models/expected_utility.py` — EU model
- `antagonistic_collab/models/prospect_theory.py` — CPT model
- `antagonistic_collab/models/priority_heuristic.py` — Priority Heuristic model
- `antagonistic_collab/models/gamble_structures.py` — registry (76 gambles)
- `antagonistic_collab/models/decision_runner.py` — synthetic data + scoring
- `antagonistic_collab/models/decision_eig.py` — EIG adapter (7 groups, selection, posterior update)
- `antagonistic_collab/models/decision_agents.py` — agent configs + system prompts
- `scripts/validation/validate_decision_m14.py` — validation script
- `tests/test_decision_models.py` — 17 tests
- `tests/test_decision_eig.py` — 23 tests
- `tests/test_decision_agents.py` — 18 tests

**Predictions to test:**
- Arbiter should favor CPT/EU (smooth gradients) over PH (discrete rules)
- LLM proposals should favor PH (easy to describe) over CPT (hard to articulate)
- EIG should be model-agnostic
- If pattern replicates across both domains → principle about representational
  format, not domain content → NeurIPS paper

### R-IDeA — Complete (negative result)

R-IDeA results (all conditions):
- Correct spec, no debate: EIG 86.9% > R-IDeA 80.5%
- Misspec, no debate: EIG 75.1% > R-IDeA 65.4%
- Misspec, R-IDeA + debate: **53.7% mean — worst condition tested**
- **Conclusion: EIG + debate (81.4%, 3.3% std) remains gold standard.**
  Informativeness + semantic diagnosis are synergistic; diversification +
  diagnosis are antagonistic. Complementary biases must use orthogonal
  information channels, not reweight the same channel.

### Paper Strategy

- Target: **NeurIPS** (automated science / Bayesian OED community)
- Framing: implicit priors in hybrid AI systems, demonstrated across two domains
- See `New Ideas/NeurIPS.md` for full strategy
- Two-domain result elevates from CogSci to NeurIPS
- One domain = finding, two domains = principle

### Other directions (not started)

1. **GeCCo forks** — see `New Ideas/gecco_arbiter_fork.md`
   - gecco-core: can LLMs discover cognitive models from scratch?
   - gecco-supplement: is there a fourth model of categorization?
2. **Griffiths connections** — see `New Ideas/tomgriffits.md` (21 questions)
3. **Real data** — human participants via Prolific/AutoRA

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
- **R-IDeA (formal diversification)** — representativeness + de-amplification dilutes informativeness signal. R-IDeA+debate (53.7%) worse than EIG+debate (81.4%). Diversify channels, not weightings (D48).
