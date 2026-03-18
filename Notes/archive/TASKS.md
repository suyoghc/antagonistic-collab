# Tasks

## Current: M14 — Close Debate→Computation Feedback Loop (COMPLETE)

### M14 Tasks
- [x] **Claim-directed experiment selection** — `claims_to_boost_specs()` converts untested testable claims to boost specs for EIG selection. Validates against full pool universe (registry + parametric + sampled + temporary). Merged with crux specs in `run_full_pool_selection()`.
- [x] **Validated parameter revisions** — `validate_param_revision()` computes RMSE with current vs proposed params on last experiment. Gate in `sync_params_from_theory()` rejects degradations > 0.01.
- [x] **Claim auto-resolution** — `resolve_claims_from_data()` resolves untested claims matching executed experiment. Hybrid strategy: pattern parsing for explicit predictions, RMSE fallback otherwise. Wired into `run_cycle()` after execution.
- [x] **Claim field normalization** — `normalize_claim_fields()` fuzzy-matches LLM free-text to registry keys ("Shepard Type I" → "Type_I", "high noise" → "high_noise"). Gated by `_NORMALIZE_CLAIMS` flag.
- [x] **Fuzzy structure matching** — `_fuzzy_match_sampled_structure()` parses parameters from ephemeral sampled structure names and finds nearest match in current pool by type, dims, and separation/exceptions. Gated by `_FUZZY_STRUCTURE_MATCH` flag.
- [x] **40 new tests** across 5 test classes (TestClaimDirectedSelection, TestValidatedParamRevisions, TestClaimAutoResolution, TestClaimFieldNormalization, TestFuzzySampledStructureMatch). 428 total passing.
- [x] **Live validation** — 3/3 correct with all interventions firing
- [x] **Ablation** — 3/3 correct with crux_weight=0

### M14 Validation Results (with normalization + fuzzy matching, GPT-4o, 2026-03-17)

| GT | Winner | RMSE | Gap% | Claims | Resolved | Cruxes |
|---|---|---|---|---|---|---|
| GCM | Exemplar_Agent | 0.087 | 77.8% | 44 | 15 | 31 (14 acc) |
| SUSTAIN | Clustering_Agent | 0.061 | 87.9% | 36 | 15 | 32 (8 acc) |
| RULEX | Rule_Agent | 0.233 | 39.0% | 40 | 15 | 32 (13 acc) |

### M14 Ablation Results (crux_weight=0, GPT-4o, 2026-03-17)

| GT | Winner | RMSE | Gap% | Claims | Resolved |
|---|---|---|---|---|---|
| GCM | Exemplar_Agent | 0.102 | 70.3% | 40 | 15 |
| SUSTAIN | Clustering_Agent | 0.056 | 87.6% | 36 | 15 |
| RULEX | Rule_Agent | 0.326 | 17.7% | 41 | 15 |

### M14 Intervention Activity
- **Fuzzy structure match**: 94 cross-cycle matches
- **Claim-directed selection**: fired 12 times (1→5 claims/cycle, accumulating)
- **Param validation gate**: 33 events, 7 rejected (21%) — all on Exemplar_Agent
- **Claim auto-resolution**: 45 resolved (6 confirmed, 39 falsified)

### M14 Key Findings
- **Feedback loop mechanically complete.** All 3 interventions fire as designed.
- **Debate still does not beat no-debate on synthetic benchmarks.** M13 no-debate baseline (avg RMSE 0.066) outperforms M14 full (avg RMSE 0.127).
- **Param validation is the strongest intervention.** Blocked 21% of proposed revisions, preventing RMSE degradation of 0.02–0.10 each.
- **Claim auto-resolution heavily falsification-skewed.** 39/45 falsified. LLM agents overclaim consistently.
- **Claim-directed selection mixed.** Helps GCM (0.087 vs 0.102 ablation) but may steer toward narratively interesting rather than computationally diagnostic experiments.
- **The boundary is empirically established.** Debate adds no value when models are complete, data is synthetic, and design space is enumerable.

### M14 Commits
- (pending commit)

---

## Previous: M13 — Debate Ablation Study (COMPLETE)

### M13 Tasks
- [x] **Experiment framework** — `antagonistic_collab/experiment.py`: `ExperimentCondition` dataclass, `load_experiment()` YAML parser, `run_condition()` global-setter + runner, `run_experiment()` grid runner + comparison table. Rejects unknown YAML keys to prevent silent misconfig.
- [x] **No-debate mode** — `_NO_DEBATE` global in runner.py. `run_cycle()` skips all LLM phases (commitment, divergence, cruxes, interpretation, critique, audit). `run_execution()` computes predictions with default params, no LLM calls. `client=None` works.
- [x] **CLI integration** — `--no-debate`, `--experiment`, `--merge` flags in `__main__.py` and `runner.py`
- [x] **3×2 ablation config** — `experiments/debate_ablation.yaml`: No-Debate / Debate-No-Arbiter / Debate+Arbiter × Thompson / Greedy × 3 ground truths = 18 conditions
- [x] **Merge utility** — `merge_summaries()` combines multiple summary.json files, prints unified comparison table
- [x] **15 new tests** (TestExperimentFramework, TestNoDebateMode, TestDebateAblationConfig), 388 total passing
- [x] **Ablation run** — 18/18 completed, all 18 correct winners
- [x] Re-run failed `greedy_debate_RULEX` (connection error) — Rule_Agent wins, RMSE=0.053, gap=88.9%

### M13 Ablation Results (3×2, GPT-4o, 2026-03-16)

| Condition | GCM→Exemplar | RULEX→Rule | SUSTAIN→Clustering |
|---|---|---|---|
| thompson_no_debate | 0.088, 76.8% | 0.053, 86.1% | 0.057, 87.7% |
| thompson_debate_no_arbiter | 0.080, 76.7% | 0.078, 81.4% | 0.056, 87.5% |
| thompson_debate+arbiter | 0.081, 79.9% | 0.051, 86.9% | 0.077, 83.8% |
| greedy_no_debate | 0.065, 87.6% | 0.053, 90.0% | 0.016, 97.4% |
| greedy_debate_no_arbiter | 0.068, 84.7% | 0.172, 66.7% | 0.016, 97.4% |
| greedy_debate+arbiter | 0.071, 85.3% | 0.053, 88.9% | 0.021, 96.7% |

Summary by debate level:
- **none**: 6/6 correct, avg RMSE=0.055, avg gap=87.6%, avg time=368s
- **debate (no arbiter)**: 6/6 correct, avg RMSE=0.078, avg gap=82.4%, avg time=1315s
- **debate+arbiter**: 6/6 correct, avg RMSE=0.059, avg gap=86.8%, avg time=1107s

Key findings:
- **Debate is epiphenomenal on synthetic benchmarks.** 18/18 correct. No-debate has the best RMSE and gap while running 3-4× faster.
- **Debate without arbiter actively hurts** — LLM param_overrides introduce noise; debate output is disconnected from scoring pipeline.
- **Arbiter partially recovers** — crux-directed selection compensates for noise, but still doesn't beat no-debate.
- **Greedy > Thompson** when signal is strong (gap 88.2% vs 83.0%), but Thompson retains healthy uncertainty (entropy 0.001–0.032 vs 0.000).

### M13 Commits
- `938bb43` feat: experiment framework + no-debate ablation mode

---

## Future Tasks

### M15 — Model Misspecification (proposed)
Test whether debate helps when models start with wrong parameters. Deliberately misspecify default_params (wrong attention weights, wrong sensitivity). The "correct" model needs LLM-proposed parameter revisions via sync_params_from_theory() to recover. Compare debate vs no-debate.

### M16 — Open Design Space (proposed)
Remove structure registry. Force agents to propose every experiment via debate. Only temporary_structures from agent proposals enter EIG pool. Tests whether debate generates diagnostic experiments EIG alone can't discover.

### Conditions where debate may causally matter
- **Model misspecification** — models need parameter adaptation from LLM reasoning
- **Non-pre-enumerated design space** — LLM agents propose genuinely novel structures
- **Ambiguous data** — interpretation changes what you test next (real human data)
- **Explanation for humans** — the goal is understanding, not just winner identification

---

## Completed: M12 — Continuous Design Space Parameterization (DONE)

### M12 Tasks
- [x] **Config + CLI** — `design_space: continuous` in default_config.yaml, `--design-space {base,richer,continuous}` CLI flag, `--n-continuous-samples`, `_DESIGN_SPACE` and `_N_CONTINUOUS_SAMPLES` module globals, `--no-richer-design-space` kept as deprecated alias
- [x] **Sampling function** — `_sample_continuous_structures(n_samples, seed)`: ~60% linear_separable (dims 2-8, sep 0.5-4.0), ~40% rule_plus_exception (dims 3-8, exc 1-4). Deterministic per seed, cycle-dependent seeds (42 + cycle × 1000)
- [x] **Pool generation** — `generate_full_candidate_pool(design_space=, n_continuous_samples=, continuous_seed=)` replaces `richer=True|False`. Three modes: base (55), richer (168), continuous (~427)
- [x] **Resolution** — `protocol.sampled_structures` attr + 4 merge points in debate_protocol.py. All code paths resolve sampled structures
- [x] **Bug fix** — attention_weights dimension mismatch on high-dimensional structures (shapes (8,) vs (3,))
- [x] **16 new tests** (TestContinuousDesignSpace), 331 total passing
- [x] **Documentation** — D40 in DECISIONS.md, REPORT.md 2.2/3.26/4.7/5 updated, all docs updated
- [x] **Live validation** — 3/3 correct. 15/15 sampled structures selected. 0% cycle overlap. Gaps 77–96%

### M12 Validation Results (continuous, GPT-4o, 2026-03-16)

| Ground Truth | Winner | Correct? | RMSE | Gap | Sampled/5 | Cycle Overlap | FR% |
|---|---|---|---|---|---|---|---|
| GCM | Exemplar_Agent | Yes | 0.092 | 76.8% | 5/5 | 0–2% | 80% |
| SUSTAIN | Clustering_Agent | Yes | 0.022 | 95.8% | 5/5 | 0–2% | 80% |
| RULEX | Rule_Agent | Yes | 0.048 | 87.4% | 5/5 | 0–2% | 80% |

### M12 Commits
- `acc3438` feat(M12): continuous design space parameterization
- `6c2fe62` fix: attention_weights dimension mismatch on high-dimensional structures

---

## Completed: M11 — Richer Design Spaces (DONE)

### M11 Tasks
- [x] **Config + CLI** — `no_richer_design_space: false` in default_config.yaml, `--no-richer-design-space` CLI flag, `_RICHER_DESIGN_SPACE` module global, wired through config.py key_map
- [x] **Parametric structures** — 13 new structures: 7 linear_separable variants (separation × dimensionality), 6 rule_plus_exception variants (dimensionality × exception count). All generated at import time with deterministic seeds, pass `validate_novel_structure()`
- [x] **Interpolated conditions** — 2 new conditions: `moderate_attention` (midpoint low/high), `mild_noise` (between baseline/high_noise)
- [x] **Pool generation** — `generate_full_candidate_pool(richer=True|False)` includes parametric entries when enabled. Pool expands from 55 to 168 candidates
- [x] **Resolution** — `_synthetic_runner()` and `compute_model_predictions()` both resolve parametric structures and conditions via merged lookups
- [x] **14 new tests** (TestRicherDesignSpaces): config/CLI/global (3), structure validity (4), condition validity (3), pool generation (2), synthetic runner resolution (2). 315 total passing
- [x] **Documentation** — D39 in DECISIONS.md, REPORT.md 2.2/3.25/4.7/5 updated, PLANNING.md M11 milestone, this file
- [x] **Live validation** — 3/3 correct. 15/15 experiments used parametric structures. EIG strongly prefers parametric linear_separable variants.

### M11 Validation Results (richer=true, GPT-4o, 2026-03-16)

| Ground Truth | Winner | Correct? | RMSE | Gap | Param-S | Param-C | FR% |
|---|---|---|---|---|---|---|---|
| GCM | Exemplar_Agent | Yes | 0.075 | 75.8% | 5/5 | 3/5 | 80% |
| SUSTAIN | Clustering_Agent | Yes | 0.022 | 95.6% | 5/5 | 1/5 | 73% |
| RULEX | Rule_Agent | Yes | 0.053 | 83.7% | 5/5 | 1/5 | 80% |

### M11 Commits
- `2a5823b` feat(M11): richer design spaces — parametric structures + interpolated conditions
- `faf2a37` docs: M11 richer design spaces across all project documentation

### M11 Literature
- Myung, J. I. & Pitt, M. A. (2009). Optimal experimental design for model discrimination. *Psychological Review, 116*(3), 499–518.
- Cavagnaro, D. R., Myung, J. I., Pitt, M. A., & Kujala, J. V. (2010). Adaptive design optimization: A mutual information-based approach to model discrimination in cognitive science. *Neural Computation, 22*(4), 887–905.

---

## Completed: M10 — Claim-Responsive Debate (DONE)

### M10 Tasks
- [x] **Config + CLI** — `no_claim_responsive: false` in default_config.yaml, `--no-claim-responsive` CLI flag, `_CLAIM_RESPONSIVE` module global, wired through config.py key_map
- [x] **Falsified claims directive** — When `_CLAIM_RESPONSIVE` is true and agent has falsified claims, inject `### FALSIFIED CLAIMS` block into interpretation prompt listing each falsified claim with evidence. Agents must respond with revise/explain/abandon via `"falsified_response"` JSON field
- [x] **7 new tests** (TestClaimResponsiveDebate): config default, CLI flag, module global, directive present when enabled, absent when disabled, absent when no falsified claims, JSON field instruction present
- [x] **Documentation** — D38 in DECISIONS.md, REPORT.md 4.4/4.7 updated, WRITEUP.md 6.3/6.4 updated with Reflexion citation, PLANNING.md M10 milestone
- [x] **Live validation** — 3/3 correct across all ground truths. 80% falsified_response rate. Agents use revise/explain/abandon actions.

### M10 Commits
- `cb9c69a` feat(M10): claim-responsive debate — agents must address falsified claims
- `e199090` docs: M10 claim-responsive debate across all project documentation

---

## Completed: M9 — Crux-Directed Thompson Sampling (DONE)

### M9 Commits
- `7e26048` fix: 3 Codex review bugs — hardcoded credential, mock crux matching, batch mode leak
- `466d1c0` feat(M9): crux-directed Thompson sampling — debate now affects experiment selection
- `e8fbb00` docs: M9 crux-directed Thompson sampling across all project documentation

---

## Completed: M8 — Thompson Sampling + Codex Fixes Round 6 (DONE)

### M8 Commits
- `8ff7793` feat(M8): Thompson sampling for experiment selection + literature review
- `4bb559e` fix: 3 Codex review bugs — curve bonus, novel structure execution, divergence map

---

## Completed: M7 — Likelihood Tempering + Codex Fixes (DONE)

### M7 Commits
- `0c43faf` feat(M7): likelihood tempering, arbiter-v0.1 toggle, and config file
- `b79829c` fix: 5 Codex review bugs — ground-truth leakage, novel structures, LOO mismatch, RULEX curves, n_subjects threading
- `9138b42` fix(M7): calibrate tempering — tau=0.005, clip [0.05, 0.95]

---

## Completed: M6 — arbiter-v0.1 Integration (DONE)

### M6 Commits
- `3109d14` feat(M6a): MetaAgentConfig
- `dfc6ed2` feat(M6b.1): Crux dataclass + EpistemicState crux management
- `201ff06` feat(M6b.2): run_crux_identification
- `35485d8` feat(M6b.3): run_crux_negotiation
- `fde4949` feat(M6b.4): finalize_cruxes
- `e1920a4` feat(M6b.5): crux_boost_specs in select_from_pool
- `f49818c` feat(M6b.6): Wire crux phases into run_cycle
- `7fb5de3` feat(M6e): Conflict Map
- `d7b8ca6` feat(M6d): Pre-registration output
- `be91b7b` feat(M6c): HITL checkpoints
- `2a57937` fix: coerce new_predictions to list in summary_for_agent

## Completed: M5 — Close Debate Feedback Loops (DONE)

### M5 Commits
- `1d12fde` feat(M5/7.1): Parameter revision persistence
- `4625d53` feat(M5/7.4): Structured claim ledger
- `84852bb` feat(M5/7.2): Critique-as-falsification
- `f61eec4` feat(M5/7.3): Debate-informed EIG weighting + lint/format cleanup
