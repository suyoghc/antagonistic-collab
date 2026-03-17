# Roadmap

## Latest: M14 — Debate→Computation Feedback Loop (complete)

Closed the loop between LLM debate output and the computational scoring pipeline:
claim-directed experiment selection, validated parameter revisions, and claim
auto-resolution. All 3 interventions fire end-to-end (94 fuzzy matches, 12
claim-directed selections, 7/33 param rejections, 45 claims resolved).

**Key finding:** Debate adds no value when models are complete, data is synthetic,
and the design space is enumerable. The computational pipeline alone identifies the
correct model in all conditions (M13 ablation: 18/18 correct). Next step: test where
models are *incomplete*.

Full M14 results and earlier milestone details: [Notes/archive/TASKS.md](Notes/archive/TASKS.md)

---

## Future Milestones

### M15 — Model misspecification (in progress)

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

Phase 2 — Debate vs no-debate comparison (LLM calls):
- All agents start with gap-calibrated wrong params from Phase 1b
- Ground truth uses correct params in `_synthetic_runner()`
- Debate condition: agents propose param revisions via `sync_params_from_theory()`,
  gated by `validate_param_revision()`
- No-debate condition: params stay fixed, computational pipeline only
- Measure: RMSE, gap, number of param revisions accepted, cycles to recovery

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

### M16 — Open design space (proposed)
Remove structure registry. Force agents to propose every experiment via debate.
Only `temporary_structures` from agent proposals enter the EIG pool.
Tests whether debate generates diagnostic experiments that EIG alone can't discover.

### Conditions where debate may causally matter
- Model misspecification (models need LLM-proposed param adaptation)
- Non-enumerated design space (LLM agents propose novel structures)
- Ambiguous data (real human data with noise, individual differences)
- Explanation for humans (goal is understanding, not just identification)

See [Notes/archive/LESSONS_LEARNED.md](Notes/archive/LESSONS_LEARNED.md) for 40 theses
on LLM-mediated scientific debate.
