# Tasks

## Current Milestone: M7 — Likelihood Tempering + Codex Fixes (DONE)

### M7 Tasks
- [x] **M7: Likelihood Tempering** — Add `learning_rate` (tau) parameter to Bayesian posterior updates to prevent posterior collapse. Multiplies log-likelihoods by tau ∈ (0, 1] before adding to prior. Threaded through `ModelPosterior.update()`, `compute_eig()`, `select_from_pool()`, `select_experiment()`, `update_posterior_from_experiment()`. Default tau=0.2 with `--no-tempering` to disable. `--no-arbiter` toggle for ARBITER features. YAML config file with layered precedence. 10 tests.
- [x] **Codex Review Fixes** — 5 bugs identified by Codex automated review (D31):
  - [x] Fix #1: Ground-truth leakage in curve evidence → pairwise divergence
  - [x] Fix #2: Novel structure silent fallback → merge temporary_structures
  - [x] Fix #3: Synthetic data LOO mismatch → LOO in _synthetic_runner
  - [x] Fix #4: RULEX curve missing exceptions → use predict() not _evaluate_rule()
  - [x] Fix #5: n_subjects not threaded → data.get("n_subjects", default)
  - 5 new tests (306 total passing)
- [x] **Tempering Calibration** (D32) — tau=0.2 still caused collapse; recalibrated to tau=0.005 + prediction clip [0.05, 0.95]. Live validation: entropy=0.635 after cycle 0, EIG=0.233 on cycle 1, correct winner. 2 new tests (308 total).

### M7 Commits
- `0c43faf` feat(M7): likelihood tempering, ARBITER toggle, and config file
- `b79829c` fix: 5 Codex review bugs — ground-truth leakage, novel structures, LOO mismatch, RULEX curves, n_subjects threading
- `9138b42` fix(M7): calibrate tempering — tau=0.005, clip [0.05, 0.95]

---

## Completed: M6 — ARBITER Integration (DONE)

### M6 Tasks
- [x] **M6a: MetaAgentConfig** — `MetaAgentConfig` dataclass (name, role, system_prompt), `create_default_meta_agents()` factory, Integrator + Critic role-specific prompts, wired into `run_interpretation_debate()`. 8 tests.
- [x] **M6b: Crux Negotiation** — `Crux` dataclass + `EpistemicState` methods (`add_crux`, `get_active_cruxes`, `resolve_crux`, `crux_summary`), `run_crux_identification()`, `run_crux_negotiation()`, `finalize_cruxes()`, `crux_boost_specs` in `select_from_pool()`, wired into `run_cycle()`. 32 tests.
- [x] **M6e: Conflict Map** — `category` field on `DebateClaim`, `conflict_map_summary()` on `EpistemicState`, injected into interpretation prompts. 6 tests.
- [x] **M6d: Pre-registration Output** — `generate_preregistration()` produces prediction tables, adjudication criteria, active cruxes, prior accuracy. 4 tests.
- [x] **M6c: HITL Checkpoints** — `hitl_checkpoint()` with auto-continue in batch mode, `--hitl-checkpoints` CLI flag. 4 tests.
- [x] **Bugfix: dict new_predictions crash** — `summary_for_agent()` crashes when LLM returns `new_predictions` as dict. Coerce to list before slicing. 2 regression tests.
- [x] **M6 Live Validation** — 5-cycle runs with all 3 ground truths (GPT-4o via Princeton, all M6 features enabled): GCM✓ (36.4% gap), SUSTAIN✓ (45.6% gap), RULEX✓ (67.6% gap). 3/3 correct winners.

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

### M6 Live Validation Results (GPT-4o, 2026-03-15)

| Ground Truth | Winner | RMSE | Gap% | Cruxes (accepted/total) | Claims (falsified/total) | Time |
|---|---|---|---|---|---|---|
| GCM | Exemplar_Agent | 0.1512 | 36.4% | 4/34 | 14/41 | 431s |
| SUSTAIN | Clustering_Agent | 0.2700 | 45.6% | 7/32 | 15/39 | 439s |
| RULEX | Rule_Agent | 0.1187 | 67.6% | 4/35 | 15/41 | 467s |

### M6 Key Findings
- **Crux negotiation is selective**: ~100 cruxes proposed, 15 accepted (15%). Real LLMs reject most proposals (mock validation showed 100% acceptance).
- **Falsification dominates confirmation**: 44 claims falsified vs 1 confirmed across all 3 runs. System works as a falsification engine.
- **Posterior collapse after cycle 0–1**: EIG drops to ~0, making later cycles uninformative. Main architectural weakness.
- **Winning theories need fewer revisions**: Rule_Agent made 0 revisions and won RULEX by 67.6%. Losing agents revise futilely.
- **RULEX shows non-monotonic dynamics**: Posterior initially favored Exemplar_Agent, flipped to Rule_Agent by cycle 2 — self-correction via structural variation.
- **Meta-agents contribute substantively**: Critic identifies weakest arguments, Integrator synthesizes across theories. Neither overrides Bayesian machinery.

---

## Completed: M5 — Close Debate Feedback Loops (DONE)

### M5 Tasks
- [x] **7.1 Parameter revision persistence** — `sync_params_from_theory()` copies `theory.model_params` → `agent_config.default_params` after each interpretation phase. Filters through `inspect.signature`. 6 tests.
- [x] **7.4 Structured claim ledger** — `DebateClaim` dataclass + `claim_ledger` on `EpistemicState`. Methods: `add_claim`, `update_claim_status`, `get_active_claims`, `stale_claims`, `claims_summary_for_agent`. Claims parsed from agent JSON, shown in interpretation prompts. 8 tests.
- [x] **7.2 Critique-as-falsification** — `verify_prediction_claim()` runs actual model computation when agents claim prediction values. FALSE CLAIMs flagged and recorded in ledger. 5 tests.
- [x] **7.3 Debate-informed EIG weighting** — `select_from_pool()` accepts `focus_pair` + `pair_boost`. Extracts contested pairs from ledger or posterior. 5 tests.
- [x] **M5 Validation** — 5-cycle runs with all 3 ground truths (GPT-4o via Princeton): GCM✓, SUSTAIN✓, RULEX✓. All correct winners.
- [x] **Replication variance** — 4× GCM runs: Exemplar RMSE std=0.018 (was 0.000 pre-M5). Debate now causally affects outcomes.
- [x] **Critique-as-falsification activity** — ~45 FALSE CLAIMs detected across 6 runs, 1 verified. Agents consistently overclaim model accuracy.

### M5 Commits
- `1d12fde` feat(M5/7.1): Parameter revision persistence
- `4625d53` feat(M5/7.4): Structured claim ledger
- `84852bb` feat(M5/7.2): Critique-as-falsification
- `f61eec4` feat(M5/7.3): Debate-informed EIG weighting + lint/format cleanup

---

## Completed: M4 — Analysis & write-up

### M4 Tasks
- [x] **Analyze EIG selection patterns** — five_four/fast universal cycle-0 pick; RULEX run shifts to Type_I/low_attention; EIG repeats optimal structure rather than diversifying (LESSONS 9.1)
- [x] **Analyze novel structures** — 21 proposed across 15 cycles, none selected by EIG. Narratively interesting but not statistically optimal. Registry structures sufficient (LESSONS 9.3)
- [x] **Per-cycle convergence trajectories** — GCM/SUSTAIN immediate (cycle 0), RULEX 2-cycle lag then flip. Posterior monotonic after convergence (LESSONS 9.2)
- [x] **Theory revision patterns** — correct theories stable, incorrect revise progressively (Lakatos-compatible). RULEX most rigid (LESSONS 9.5)
- [x] **Interpretation debate quality audit** — data citation weak (posteriors as proxy), critiques mixed (mechanism-aware but numerically ungrounded), behavioral adaptation limited (same talking points repeat), novel structure rationale poor (not rooted in divergence). Forcing function of adversarial critique helps in later cycles. (LESSONS 9.6)
- [x] **Replication runs** — 3× each condition (full_pool): RMSE values identical across replicates (zero variance). Pipeline is fully deterministic — EIG selection, synthetic data, and model predictions are all seeded. LLM debate text varies but doesn't affect RMSE. (REPORT 3.5)
- [x] **Write-up** — REPORT.md: intro (antagonistic collaboration + 3 models), methods (EIG, learning curves, 55 candidates, 2 modes), results (6 runs, all correct, 10 findings), discussion (architecture thesis, specification gap, debate limitations). 2026-03-14
- [x] **Cross-LLM comparison** — GPT-4o vs Sonnet vs Opus (9 runs). Correct model wins in all 9/9. RMSE varies slightly via param_overrides but outcomes identical. Framework is LLM-agnostic. (D26, LESSONS 9.8, REPORT 3.6)

---

## Completed: M3 — Validate convergence

### Blocking
- [x] **Fix synthetic data generator** — experiments now produce different data depending on category structure and conditions (D6 → D7)
- [x] **Constrain agent proposals to structure library** — agents pick from STRUCTURE_REGISTRY menu (11 structures)

### High Priority
- [x] **Expand scoring beyond mean_accuracy** — item_accuracies now merged into scoring dict, per-item predictions scored
- [x] **Vary model params by experimental conditions** — CONDITION_EFFECTS maps 5 conditions to model param overrides

### Next (M2 cont.)
- [x] **Agents call their models for predictions** — run `model.predict()` during execution phase instead of LLM-guessed numbers (D8)

### Up Next (M3)
- [x] Validate convergence: run 3-cycle debate where the true model's agent accumulates best RMSE — **confirmed** (Exemplar_Agent RMSE=0.0776, 3.6x gap)
- [x] Run with each model as ground truth (GCM, SUSTAIN, RULEX) and compare — **GCM and SUSTAIN correct, RULEX fails due to unfavorable structure selection** (D11, Phase 4 findings)
- [x] Fix self-prediction bias — leave-one-out cross-validation in compute_model_predictions and compute_divergence_map (D11)
- [x] Assess whether critique quality degrades over cycles — critiques remain substantive but formulaic ("my model can also predict that" is common). Agents revise theories "progressively" but critiques don't drive structure diversity. The audit phase detects no convergence collapse. Main issue: critiques target proposals but don't influence moderator selection.
- [x] Run longer debates (5+ cycles) to check whether RMSE gap widens monotonically — **GCM: yes (15.1% gap at 5 cycles), SUSTAIN: yes (32.2%), RULEX: gap only appears with diversity penalty (1.8%)**
- [x] **Make the debate influence outcomes** — divergence-driven experiment selection picks the most diagnostic structure (D12)
- [x] **Investigate GCM flexibility confound** — With diversity penalty, Rule_Agent now wins on RULEX-generated data (RMSE 0.433 vs 0.441). Gap is small (1.8%) vs GCM/SUSTAIN (15-32%) — this is genuine model flexibility, not a system bug. GCM approximates rule-like behavior through attention weights (consistent with Nosofsky 1991).
- [x] **Bayesian information-gain experiment selection** — Replaced heuristic diversity penalty with principled adaptive design (D18). `ModelPosterior` maintains Bayesian posterior over models; `compute_eig()` uses Monte Carlo expected information gain to select experiments. Heuristic retained as `--selection heuristic` fallback. 12 new tests, 158 total passing.

- [x] **Debate-as-hypothesis-generator architecture (D19)** — Three-phase refactor: (A) Full-pool EIG over 55 candidates replaces agent proposals, (B) Learning curves as second evidence channel, (C) Novel structure generation from LLM debate. Legacy 9-phase flow preserved as `--mode legacy`. 31 new tests, 189 total passing.
- [x] **Fix full_pool mode phase desync (D20)** — `advance_phase()` uses `current_phase` for transitions; skipping phases 3-5 in full_pool mode left the state machine at EXPERIMENT_PROPOSAL instead of HUMAN_ARBITRATION. Fix: `skip_to_phase(HUMAN_ARBITRATION)` before EIG advance. Integration test added. 190 tests passing.
- [x] **Validate full_pool mode end-to-end** — 2-cycle real run with Princeton/GPT-4o confirms correct convergence: Exemplar_Agent wins (RMSE 0.139) when GCM is ground truth. EIG selects different structures across cycles without diversity penalty.

### Up Next (M3 cont. — full_pool integration gaps)
- [x] **Wire learning curves into run_execution()** — compute `compute_learning_curve_predictions()` during execution, pass curves to `update_posterior_from_experiment()`, store on `protocol.state.last_execution_curves` (D23)
- [x] **Feed novel structures back into EIG pool** — after `run_interpretation_debate()`, validate and register novel structures in `protocol.temporary_structures` for next cycle's `generate_full_candidate_pool()` (D23)
- [x] **Include learning curve + posterior context in interpretation debate** — agents see curve comparison table (pattern, final accuracy, max jump, onset block) in their interpretation prompt (D23)
- [x] **5-cycle comparative validation** — full_pool vs legacy for all 3 ground truths. Full_pool correct in all 3 with larger gaps: GCM 34% (vs 37%), SUSTAIN 42% (vs 34%), RULEX 68% (vs 2.4%). Learning curves solved the GCM-RULEX discrimination problem.
- [x] **Prompt novel structure generation** — added few-shot example (diagonal_xor), constraints (4-32 items, ≤8 dims, ≥2 cats), strategic guidance to interpretation prompt (D24)

### Queued (from Codex review, 2026-03-13)
- [x] **[P1] Implement Phase 5 (Design Revision)** — agents now revise proposals based on critiques, updating design_spec via state.revise_proposal()
- [x] **[P2] Fix moderator reject path** — rejection now loops back to proposal→critique→revision→arbitration (up to 3 attempts), rejected proposals marked with status="rejected"
- [x] **[P2] SUSTAIN.predict_learning_curve() ignores test_items/test_labels** — fixed: now evaluates test items at each block boundary (matches GCM contract)
- [x] **[P3] Fix --demo order-sensitivity** — `"--demo" in sys.argv` works regardless of position
- [x] **Show concrete model predictions in divergence ranking** — ranked summary now shows per-model predicted accuracy (e.g., "Exemplar_Agent: 0.75, Rule_Agent: 0.80, Clustering_Agent: 0.50") so agents can see their advantage per structure

### Done (Codex review round 4, 2026-03-14)
- [x] **RULEX ground truth non-deterministic** — _synthetic_runner() missing seed=42 for RULEX.predict() (D22)
- [x] **RULEX predict_learning_curve() drops seed** — find_best_rule() call missing seed forwarding (D22)
- [x] **Redundant model_predicted/predicted keys** — removed duplicate key in execution messages (D22)
- [x] **.gitignore gap for .md transcripts** — added debate_cycle_*.md pattern (D22)
- [ ] **(Latent) GCM.fit() self-prediction bias** — same D11 bug exists in fit(), not used in pipeline currently

### Done (Codex review round 3, 2026-03-14)
- [x] **[P1] Override fallback drops condition params** — fallback now uses condition-applied params, not bare defaults (D21)
- [x] **[P1] Scalar addresses_critiques crashes design revision** — coerce to list before iterating (D21)
- [x] **[P2] Invalid approval input escapes retry loop** — set rejected=True on out-of-range approve (D21)
- [x] **[P2] SUSTAIN partial block duplicate label** — use enumerate index instead of arithmetic formula (D21)

### Done (Codex review round 2, 2026-03-14)
- [x] **Malformed param override values crash execution** — try/except fallback to default params on ValueError/TypeError
- [x] **SUSTAIN drops final partial block** — predict_learning_curve now includes partial block, matches GCM/RULEX curve length

### Done (code review fixes, 2026-03-13)
- [x] Fix format crash: :.3f applied to 'N/A' string on missing mean_accuracy (runner.py:647)
- [x] Validate design_spec is dict before .get() calls (runner.py:593)
- [x] GCM: r=0 now raises ValueError instead of ZeroDivisionError (gcm.py:58)
- [x] GCM: incomplete bias dict now raises ValueError instead of KeyError (gcm.py:108)
- [x] SUSTAIN: zero lambdas produce mean(dim_sim) instead of NaN (sustain.py:74)
- [x] EpistemicState.to_json() creates parent directories (epistemic_state.py:717)
- [x] NaN guard in divergence map computation (debate_protocol.py:530)
- [x] Warning for unknown condition names in compute_model_predictions
- [x] SUSTAIN.predict_learning_curve() now evaluates test_items at block boundaries (P2 fix)
- [x] call_agent() retries up to 3 times on transient API errors (exponential backoff)

### Done
- [x] Fix P1 crashes and data-integrity bugs — 2026-03-11 → 2026-03-12
- [x] Add Princeton/Portkey backend — 2026-03-12 16:40
- [x] Add Markdown reports + auto-naming — 2026-03-12 21:30
- [x] Fix duplicate JSON + empty leaderboard — 2026-03-12 21:41
- [x] Fix batch-mode moderator bias (always picking first proposal) — 2026-03-12 22:39
- [x] Delete buggy run outputs, re-run with rotation fix — confirmed rotation works — 2026-03-13
- [x] Create DECISIONS.md, CHATLOG.md, TASKS.md — 2026-03-13
- [x] Create SCRATCHPAD.md, PLANNING.md — 2026-03-13
