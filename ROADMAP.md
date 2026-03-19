# Roadmap

## Latest: M17 — Misspecification + Open Design (complete)

**6/6 correct under double stress.** Combines M15 misspecification with M16 open design.
Key finding: composition is non-additive and model-dependent. GCM open_arbiter achieves
best-ever GCM result (87.8%) via synergy between param recovery (85.7%) and arbiter-guided
proposals. Open design rescues RULEX from arbiter catastrophe — M15 arbiter-RULEX was the
project's only wrong winner (3.2%); M17 open_arbiter-RULEX is correct (42.2%).
47/48 correct across M14–M17.

### M16 — Open Design Space (complete)

**15/15 correct.** Arbiter is a bias, not noise: helps SUSTAIN (+8.3pp, 96% gap),
hurts RULEX (-22pp). Open design is the mirror bias: helps RULEX (+24pp), hurts
SUSTAIN (-24pp). Computation alone most reliable across all model types.

See [CURRENT_STATE.md](CURRENT_STATE.md) for full results and analysis.

### M15 — Model misspecification (complete)

First causal demonstration that debate adds value. Under parameter misspecification,
the core debate loop improves gap by +3.5pp (GCM) to +22.4pp (RULEX) via 60-86%
parameter recovery. Arbiter layer net negative under misspecification (revised by M16:
the arbiter is model-biased, not universally harmful).

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

### M16 — Open design space (Phase 2 complete)

**Scientific question:** Does debate add value for experiment *design*? Does the arbiter
help or hurt under correct specification?

**Design:** 2×2+1 factorial (closed/open × debate/arbiter + no-debate baseline).
15 runs total, 15/15 correct.

**Key findings:**
- Arbiter is model-biased: +8.3pp SUSTAIN (best ever), +2.4pp GCM, -22pp RULEX
- Open design is the mirror bias: +24pp RULEX recovery, -24pp SUSTAIN
- Arbiter recovers open-design losses for GCM (open_arbiter +0.1pp vs open_debate -5.2pp)
- Computation alone most reliable across all model types (76-88% gap)

**Open:** Crux debiasing. R-IDeA as alternative OED. Real data integration.

**Tests:** `TestOpenDesignSpace` (5 tests) in `tests/test_bugfixes.py`.
Validation: `scripts/validation/validate_m16_live.py`.

### M-future — R-IDeA as alternative OED type

**Scientific question:** Does R-IDeA's multi-objective acquisition (representativeness +
informativeness + de-amplification) reduce the model-type biases that M16 identified,
and does it interact differently with debate than EIG does?

**Motivation:** EIG optimizes for informativeness only. M16 showed this is model-agnostic
when the design space is adequate, but every other component (arbiter, open design)
introduces model-type bias. Tang, Sloman & Kaski (2025, R-IDeA) add two terms:
representativeness (broad coverage) and de-amplification (don't amplify existing errors).
These directly target the failure modes M16 documented.

**Design:** Add R-IDeA as a new OED type alongside EIG in the experiment selection
pipeline, not replacing it. Run the existing 5-condition factorial (closed/open ×
debate/arbiter + no-debate baseline) with R-IDeA scoring instead of EIG. Compare
per-GT gaps and model-type bias patterns.

**Key comparisons:**
- R-IDeA no-debate vs. EIG no-debate: does de-amplification reduce per-GT variance?
- R-IDeA + debate vs. EIG + debate: does R-IDeA change when debate helps vs. hurts?
- R-IDeA + open design: does representativeness term counteract narrative narrowing?

**Implementation:** Add an `oed_type` parameter to the experiment selection pipeline
(values: `"eig"`, `"r_idea"`). R-IDeA scoring requires: (1) a representativeness
metric over the candidate pool, (2) the existing EIG term, (3) a de-amplification
estimate based on current residuals. The pipeline selects from the same candidate
pool — only the scoring changes.

**Connection:** See [New Ideas/Sloman - Bayesian OED under Misspecification.md] for
full analysis. Also related: GBOED (Barlas, Sloman & Kaski 2025) could be a third
OED type targeting robustness to misspecified noise distributions.

### Conditions where debate causally matters (updated through M17)
- Model misspecification — **confirmed (M15).** Debate improves gap by +3.5pp (GCM),
  +22.4pp (RULEX) via parameter recovery. Neutral when misspecification is invisible (SUSTAIN).
- Non-enumerated design space — **partially confirmed (M16).** Agent proposals help RULEX
  (+24pp over closed_debate) when registry lacks diagnostic structures. But hurt SUSTAIN (-24pp).
- Arbiter as model-biased tool — **confirmed as bias, not noise (M16).** Crux machinery
  helps similarity models (SUSTAIN +8pp, GCM +2pp), hurts rule models (RULEX -22pp).
- Composition under double stress — **confirmed (M17).** Misspec + open design + arbiter
  produces best-ever GCM (87.8%) via synergy. Open design rescues RULEX from arbiter
  catastrophe (3.2% wrong → 42.2% correct). Complementary biases partially cancel.
- Ambiguous data (real human data with noise, individual differences) — untested
- Explanation for humans (goal is understanding, not just identification) — untested

See [Notes/archive/LESSONS_LEARNED.md](Notes/archive/LESSONS_LEARNED.md) for 40 theses
on LLM-mediated scientific debate.
