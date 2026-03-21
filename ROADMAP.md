# Roadmap

## In Progress: Decision-Making Domain Extension

Extending the framework to a second domain (decision-making under risk) to test
whether the implicit-prior / complementary-bias findings generalize. Three models
implemented: Expected Utility (↔ SUSTAIN), Cumulative Prospect Theory (↔ GCM),
Priority Heuristic (↔ RULEX). Full computational pipeline wired: gamble registry
(76 problems), EIG adapter (7 groups), agent configs, validation script.

No-debate baseline results:
- Correct params: 3/3 (all identified by cycle 0-1)
- Misspecified params: 0/3 (all wrong — stronger penalty than categorization)

Standalone decision debate runner built (D49 — Option C): `decision_debate_runner.py`
with 14 tests. Components: LLM debate round, param validation via
model.default_params, RMSE gate, full cycle loop. Arbiter layer not yet added.

Next: live LLM experiments under misspecification to test debate recovery.

If the same bias pattern replicates → NeurIPS paper on implicit priors in hybrid
AI systems as a domain-general principle.

### R-IDeA — tested, negative result

R-IDeA (formal diversification) underperforms EIG in all regimes. R-IDeA+debate
(53.7%) is worst condition tested — diversification starves debate's parameter
recovery mechanism. Lesson: complementary biases must use orthogonal information
channels, not reweight the same channel.

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

### R-IDeA as alternative OED type (tested — negative result)

**Scientific question:** Does R-IDeA's multi-objective acquisition (representativeness +
informativeness + de-amplification) reduce the model-type biases that M16 identified?

**Answer: No.** R-IDeA underperforms EIG in all regimes tested:
- Correct spec, no debate: EIG 86.9% > R-IDeA 80.5%
- Misspec, no debate: EIG 75.1% > R-IDeA 65.4%
- Misspec, R-IDeA + debate: 53.7% — **worst condition tested**. RULEX drops to 19.4%
  (vs 80.4% with EIG+debate) because representativeness dilutes visible prediction
  failures that debate needs for parameter recovery.

**Key insight:** Informativeness and semantic diagnosis are synergistic — debate needs
the most informative experiments to see prediction failures and diagnose parameter errors.
Diversifying experiment selection (R-IDeA) is antagonistic to diagnosis — it starves the
mechanism that actually works. EIG + debate (81.4%, 3.3% std) remains the gold standard
under misspecification.

Scripts: `antagonistic_collab/ridea.py`, `scripts/validation/validate_ridea.py`,
`scripts/validation/validate_ridea_debate.py`. Tests: `tests/test_ridea.py` (15 tests).

### M-future — Additional complementary biases

**Scientific question:** Can we add selection mechanisms that target currently underserved
model types (especially SUSTAIN) to further reduce model-type variance?

**Motivation:** The current system has three biases: EIG (model-agnostic, item-level
predictions), arbiter cruxes (favors similarity models via gradient-visible disagreement),
and LLM proposals (favors rule models via linguistically nameable structures). SUSTAIN's
most distinctive feature — cluster recruitment dynamics, order-sensitive learning — is not
specifically targeted by any component.

**Candidate mechanisms:**

| Component | Sees | Favors | Fills gap |
|---|---|---|---|
| EIG | Item-level prediction divergence | Model-agnostic | Baseline |
| Arbiter cruxes | Smooth gradient disagreement | GCM/SUSTAIN | Similarity models |
| LLM proposals | Nameable structures | RULEX | Rule models |
| **Learning curve selection** | Temporal dynamics | SUSTAIN | Currently underserved |
| **Falsification-directed** | Leader weakness | Adaptive | Anti-confirmation bias |
| **Random injection** | Everything uniformly | None | Universal debiaser |

**Learning curve selection** is highest priority: we already compute learning curves
for scoring but don't use them for experiment *selection*. RULEX shows sudden jumps
(rule discovery), SUSTAIN shows cluster recruitment steps, GCM shows smooth gradual
learning. Selecting experiments that maximize learning-curve divergence would directly
target SUSTAIN's distinctive mechanism.

**Falsification-directed selection** targets experiments where the current posterior leader
is weakest. Adaptive per-cycle, systematically anti-confirmatory.

**Random injection** (Dubova-inspired): small probability of random experiment each cycle
breaks systematic bias. Theoretically grounded, trivial to implement.

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
