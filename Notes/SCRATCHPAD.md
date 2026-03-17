# Scratchpad

Working notes, open questions, and in-progress plans. Clean out when work is completed.

---

## Phase 1a — Mimicry sweep (COMPLETE)

### Result: No true mimicry exists

Script: `scripts/m15_mimicry_sweep.py`

Swept parameter grids for all 3 models across 7 base structures (Shepard I-VI,
five_four). At every parameter setting tested, each model's predictions remain
closer to its own ground truth than to any competitor. The models are structurally
too different for parameter changes to make them indistinguishable.

Closest mimicry distances (all still far from threshold):
- GCM → SUSTAIN: c=6.0, RMSE=0.2430 (vs self: 0.0580) — 4× gap
- SUSTAIN → GCM: r=12.0/eta=0.04, RMSE=0.2438 (vs self: 0.0573) — 4× gap
- RULEX → SUSTAIN: err_tol=0.05/p_single=0.3, RMSE=0.3258 (vs self: 0.2119) — 1.5× gap

**Implication:** Model identification will always succeed (correct model always
closest to ground truth). But misspecification narrows the gap, making identification
fragile. M15 tests whether debate improves identification *quality* (wider gap,
lower RMSE) not whether it flips the *winner*.

**Why this is still scientifically meaningful:** In real-world settings with noisy
data (M17), a narrow gap can easily flip. Demonstrating that debate widens the gap
via parameter recovery would show debate's value as a robustness mechanism.

---

## Active: Phase 1b — Competition-based gap sweep

### Goal
Generate synthetic data from each ground-truth model, then score all three models
against that data with the correct model deliberately misspecified. Measure how much
misspecification narrows the winner's gap. Select settings where the gap is small
enough that parameter recovery matters.

### Script: extend `scripts/m15_mimicry_sweep.py`

### Method
For each ground truth (GCM, SUSTAIN, RULEX):
1. Generate synthetic data using ground-truth params (via `_synthetic_runner()` logic)
2. Score all 3 models against that data using LOO
3. Correct model uses misspecified params; competitors use their own defaults
4. Compute RMSE and gap (difference between correct model and best competitor)
5. Sweep misspecification levels to find settings that meaningfully narrow the gap

### Why debate helps and EIG alone can't
EIG assumes fixed params — optimizes for model discrimination, not parameter
estimation. Three debate-specific capabilities address this:
1. **Parameter revision** — agents propose new values via `sync_params_from_theory()`
2. **Diagnosis** — agents reason about *why* predictions are wrong
3. **Directed experimentation** — claim-directed selection tests whether revisions helped

This is Pitt & Myung's (2004) point: parameter estimation and model selection must
happen together. EIG does selection. Debate does estimation.

### Ground-truth parameters
- GCM: `c=4.0, r=1, gamma=1.0` (from `_synthetic_runner()`)
- SUSTAIN: `r=9.01, beta=1.252, d=16.924, eta=0.092`
- RULEX: `p_single=0.5, p_conj=0.3, error_tolerance=0.1, seed=42`

---

## Phase 1b — Competition sweep (COMPLETE)

### Result: Misspecification narrows gaps but never flips winner

Generated synthetic data from each ground truth, scored all 3 models with
the correct model deliberately misspecified. No setting flips the winner.

| Ground Truth | Baseline gap | Worst misspec setting | Narrowed gap |
|---|---|---|---|
| GCM | 60.7% | c=0.5 | 27.9% |
| SUSTAIN | 65.4% | r=3.0, eta=0.15 | 28.5% |
| RULEX | 81.9% | err_tol=0.25, p_single=0.3 | 15.7% |

RULEX most vulnerable: gap drops from 82% → 16%. Best test case for Phase 2.

### Recommended misspecification levels for Phase 2
- **GCM**: c=0.5 (broadest similarity, 28% gap)
- **SUSTAIN**: r=3.0, eta=0.15 (broadest clusters + fast learning, 29% gap)
- **RULEX**: error_tolerance=0.25, p_single=0.3 (lenient rules + limited search, 16% gap)

### What Phase 2 tests
Debate vs no-debate at these settings. Debate should widen the gap back toward
baseline via parameter recovery. No-debate stays stuck with wrong params.
RULEX is the strongest test case (most room to improve, most degradation).

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
- **Exact structure name matching for sampled structures** — ephemeral names change every cycle. Fixed with parameter-based fuzzy matching (D42).
