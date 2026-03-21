# Scratchpad

Working notes, open questions, and in-progress plans. Clean out when work is completed.

---

## Current status (2026-03-21)

M14–M17 factorial complete (categorization). 47/48 correct.
R-IDeA tested — negative result in all regimes.
Decision-making domain implementation **in progress**.

### Decision-Making Domain — IN PROGRESS

**Goal:** Replicate the implicit-prior/complementary-bias findings in a second
domain (decision-making under risk) to elevate the paper from CogSci to NeurIPS.

**Three models implemented and tested (17/17 tests pass):**
- EU (Expected Utility) — normative baseline, 1 param (r)
- CPT (Cumulative Prospect Theory) — dominant descriptive, 5 params
- Priority Heuristic — lexicographic rules, 0-1 params
- Cross-domain empirical parallels: PH↔RULEX (rule-based), EU↔GCM (moderate recovery), CPT↔SUSTAIN (abstract params resist)

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

**Standalone decision debate runner — IMPLEMENTED (D49)**

`decision_debate_runner.py` committed with 14 tests (all passing, 538/538 full suite).

Components built:
1. ~~**LLM debate round**~~ ✓ — `run_debate_round()` shows prediction errors,
   collects JSON revisions, validates params + RMSE gates
2. ~~**Parameter validation**~~ ✓ — `filter_valid_params()` validates against
   `model.default_params` keys (not inspect.signature — decision models use a
   params dict, not kwargs)
3. ~~**Cycle loop**~~ ✓ — `run_decision_debate()` wires EIG select → synthetic
   data → posterior update → debate round → param revision → repeat
4. **Arbiter round** — NOT YET IMPLEMENTED (can add later as optional layer)

**Live results — REPLICATION CONFIRMED (D50, D51)**

Decision M15 results (GPT-4o, 5 cycles, accumulated RMSE gate):

| GT | No-debate | Debate | Recovery |
|---|---|---|---|
| CPT | PH (wrong) | PH (wrong) | 51.0% |
| EU | CPT (wrong) | CPT (wrong) | 0.0% |
| PH | CPT (wrong) | **PH (correct)** | **81.8%** |

**5 cycles: 0/3 → 1/3. 10 cycles: 0/3 → 2/3.** Matches categorization M15 exactly.

| GT | No-debate | Debate (10 cyc) | Recovery |
|---|---|---|---|
| PH | Wrong | **Correct** | **100.0%** (all 3 params exact) |
| EU | Wrong | **Correct** | **75.0%** (r exact, temp off) |
| CPT | Wrong | Wrong | 28.4% (lambda_ exact, alpha/beta stuck) |

Cross-domain parallel:
- PH ↔ RULEX (rule-based, strongest recovery)
- EU ↔ GCM (recovered with more data)
- CPT ↔ SUSTAIN (abstract params resist diagnosis)

Key findings:
- Accumulated RMSE gate (D50) was critical — local-only gate let competitors game
- 10 cycles needed for EU (5 wasn't enough) — more evidence needed for simpler models
  with less distinctive predictions
- CPT alpha/beta genuinely outside LLM diagnostic capability (no prompt help given,
  matching categorization where no special prompt help was given either)

**Arbiter layer implemented (5 phases) and validated live:**

Implementation:
1. Phase 0: Interpretation text preserved in debate records (was silently discarded)
2. Phase 1: Crux protocol — identification, negotiation, finalization (reuses Crux dataclass)
3. Phase 2: Crux-directed EIG selection — crux_indices + crux_weight params
4. Phase 3: Meta-agents — decision-domain Integrator + Critic (reuses MetaAgentConfig)
5. Phase 4: Wired into run_decision_debate(enable_arbiter=True), condition="arbiter"
6. Phase 5: Validation script extended with --arbiter / --arbiter-only flags

Tests: 34 debate tests, 26 EIG tests, 561 total suite (all pass).

**Live arbiter results (GPT-4o, 10 cycles):**

| GT | No-debate | Debate (10cyc) | Arbiter (10cyc) |
|---|---|---|---|
| CPT | Wrong (→PH) | Wrong (→PH) | **Wrong** (→PH, 12.0% recovery) |
| EU | Wrong (→CPT) | **Correct** (75%) | **Wrong** (→CPT, 37.5% recovery) |
| PH | Wrong (→CPT) | **Correct** (100%) | **Correct** (78.2% recovery) |
| Score | 0/3 | 2/3 | **1/3** |

**Key finding: Arbiter bias replicates cross-domain.**
The arbiter hurts the decision domain even more than categorization (1/3 vs 8/9).
EU flipped from correct (debate) to wrong (arbiter) — crux-directed selection
steered toward CPT-favoring gambles (loss_aversion selected 4 times). PH
weakened but survived. CPT still wrong under all conditions.

Cross-domain parallel holds:
- CPT ↔ SUSTAIN: abstract params resist in all conditions
- EU ↔ GCM: arbiter hurts (categorization: arbiter helped GCM +5pp; decision: arbiter breaks EU)
- PH ↔ RULEX: arbiter weakens recovery (cat: -55pp wrong winner; decision: -22pp but still correct)

**Nuance vs categorization:** In categorization, the arbiter helped similarity
models (GCM, SUSTAIN) and hurt rule models (RULEX). In the decision domain,
the arbiter helps the complex descriptive model (CPT posterior rises) at the
expense of all simpler models. The bias is toward *complexity*, not just
*similarity* — crux-directed experiments probe where complex models make
distinctive predictions, disadvantaging simpler models.

**What's next:**
- Write up two-domain arbiter results for NeurIPS
- CPT could be a target for prompt enrichment study (separate from main result)

Calibration notes:
- learning_rate=0.01, n_subjects=30, 7 gamble groups
- RMSE gate: strict improvement, accumulated observations

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
- `antagonistic_collab/models/decision_debate_runner.py` — standalone debate runner
- `tests/test_decision_debate.py` — 14 tests

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
