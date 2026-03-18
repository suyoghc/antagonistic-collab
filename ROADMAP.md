# Roadmap

## Latest: M15 — Model misspecification (Phase 2 complete)

**First causal demonstration that debate adds value.** Under parameter misspecification,
the core debate loop (agents interpreting results + proposing param revisions) improves
identification gap by +3.5pp (GCM) to +22.4pp (RULEX) via 60-86% parameter recovery.
The arbiter layer (cruxes + meta-agents) is net negative — it distorts experiment
selection. M14 showed debate was epiphenomenal under correct specification; M15 shows
it's causally necessary under misspecification.

See [CURRENT_STATE.md](CURRENT_STATE.md) for full results and analysis.

### M14 — Debate→Computation Feedback Loop (complete)

Closed the loop between LLM debate output and the computational scoring pipeline.
Debate adds no value when models are fully specified (M13 ablation: 18/18 correct
without debate). Full M14 details: [Notes/archive/TASKS.md](Notes/archive/TASKS.md)

---

## Future Milestones

### M15 — Model misspecification (Phase 2 complete)

**Scientific question:** Can LLM agents, through debate, identify that their model's
parameters are wrong and propose corrections — and does this happen better with debate
than without?

**Literature grounding:** Pitt, Myung & Zhang (2004) established that model selection
and parameter estimation are entangled — evaluating models at wrong parameter values
gives misleading fit, and models can mimic each other at certain parameter settings.
Cavagnaro, Myung & Pitt (2010, ADO) showed that experiment selection should marginalize
over parameter uncertainty, not assume fixed params. Our EIG currently assumes fixed
params, so misspecified agents select experiments that are informative for the *wrong*
parameter regime.

**Why debate should help here (and EIG alone can't):**
EIG selects experiments assuming fixed parameters — it optimizes for model
discrimination, not parameter estimation. When params are wrong, EIG picks
experiments informative for the wrong regime. Debate adds three capabilities
EIG lacks:
1. **Parameter revision** — agents propose new values via `sync_params_from_theory()`
2. **Diagnosis** — agents reason about *why* predictions are wrong ("overpredicting
   on Type_VI means my attention weights are too uniform")
3. **Directed experimentation** — claim-directed selection tests whether revisions helped

This is Pitt & Myung's point: parameter estimation and model selection must happen
together. EIG does selection. Debate does estimation. M15 tests their combination.

**Design (three phases):**

Phase 1a — Mimicry sweep (complete, `scripts/m15_mimicry_sweep.py`):
- Swept parameter grids for all 3 models across 7 base structures
- **Finding: no true mimicry exists.** At every parameter setting tested, each
  model's predictions remain closer to its own ground truth than to any competitor.
  GCM, SUSTAIN, and RULEX are structurally too different for parameter changes alone
  to make them indistinguishable. This explains M13's 18/18 correct.

Phase 1b — Competition-based sweep (complete, `scripts/m15_mimicry_sweep.py --phase 1b`):
- Generated synthetic data from each GT, scored all models with correct model misspecified
- **Finding: misspecification never flips the winner but narrows gaps substantially.**
  GCM 61%→28%, SUSTAIN 65%→29%, RULEX 82%→16%. RULEX most vulnerable.

Calibrated misspecification settings for Phase 2:
- GCM: c=0.5 (gap narrows to 28%)
- SUSTAIN: r=3.0, eta=0.15 (gap narrows to 29%)
- RULEX: error_tolerance=0.25, p_single=0.3 (gap narrows to 16%)

Phase 2 — Three-way comparison (complete, `scripts/validation/validate_m15_live.py`):
- All agents start with gap-calibrated wrong params from Phase 1b
- Ground truth uses correct params in `_synthetic_runner()`
- Three conditions: no-debate (computational only), debate without arbiter,
  debate with arbiter (cruxes + meta-agents + claim-directed selection)
- Measure: RMSE, gap, param revisions accepted, param recovery toward GT

**Full 9-run results (GPT-4o, 5 cycles each):**

| GT | No-debate gap | Debate gap | Arbiter gap | Best |
|---|---|---|---|---|
| GCM | 74.4% | 77.9% (+3.5pp) | 79.3% (+4.9pp) | Arbiter |
| SUSTAIN | 87.7% | 85.8% (-1.9pp) | 76.1% (-11.6pp) | No-debate |
| RULEX | 58.0% | 80.4% (+22.4pp) | 3.2% (-54.7pp, **wrong winner**) | Debate |

**Correct winner: 8/9.** Only arbiter-RULEX fails.

**Key findings:**
- Debate without arbiter helps on GCM (+3.5pp) and RULEX (+22.4pp) via param recovery
  (85.7% and 60.3% respectively). Neutral on SUSTAIN (0% recovery — misspecification
  doesn't produce enough prediction error to trigger revisions).
- Arbiter consistently degrades gap on SUSTAIN (-11.6pp) and RULEX (-54.7pp). Helps
  slightly on GCM (+4.9pp). Root cause on RULEX: meta-agents distort divergence mapping,
  shifting experiment selection toward non-discriminative structures.
- Plain debate (no meta-agents) is the best overall configuration under misspecification.

**Architecture:** Existing `default_agent_configs()` → patch with bad params.
Existing `sync_params_from_theory()` + `validate_param_revision()` handle recovery.
No new pipeline machinery needed — just a validation script + param sweep script.

**Key references:**
- Pitt, Myung & Zhang (2004). Toward a method of selecting among computational
  models of cognition. *Psychological Review*, 109, 472–491.
- Cavagnaro, Myung, Pitt & Kujala (2010). Adaptive design optimization: A mutual
  information-based approach. *Neural Computation*, 22, 887–905.
- Wagenmakers, Ratcliff, Gomez & Iverson (2004). Assessing model mimicry using the
  parametric bootstrap. *JMP*, 48, 28–50.

### M16 — Open design space (code complete, pending validation)
Remove structure registry. Force agents to propose every experiment via debate.
Only `temporary_structures` from agent proposals enter the EIG pool.
Tests whether debate generates diagnostic experiments that EIG alone can't discover.

**Implementation:** `--design-space open` CLI flag. New `run_structure_proposal()`
phase runs between divergence mapping and EIG selection. Agents propose 2-3 structures
per cycle; valid proposals accumulate in `protocol.temporary_structures`. Pool grows
from ~6-9 structures on cycle 0 to ~15-24 by cycle 4. Fallback: if all proposals
invalid, seeds with Type_I + Type_VI.

**Three conditions:** closed_no_debate (M14 baseline), closed_debate (standard),
open_debate (agent-proposed only). Validation: `scripts/validation/validate_m16_live.py`.

**Tests:** `TestOpenDesignSpace` (5 tests) in `tests/test_bugfixes.py`.

### Conditions where debate may causally matter
- Model misspecification — **confirmed (M15).** Debate improves gap by +3.5pp (GCM),
  +22.4pp (RULEX) via parameter recovery. Neutral when misspecification is invisible (SUSTAIN).
- Non-enumerated design space (LLM agents propose novel structures)
- Ambiguous data (real human data with noise, individual differences)
- Explanation for humans (goal is understanding, not just identification)

See [Notes/archive/LESSONS_LEARNED.md](Notes/archive/LESSONS_LEARNED.md) for 40 theses
on LLM-mediated scientific debate.
