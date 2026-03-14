# Tasks

## Current Milestone: M3 — Validate convergence

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
- [ ] **5-cycle comparative validation** — run `--mode full_pool` vs `--mode legacy` for all 3 ground truth models, compare convergence speed and final RMSE gap
- [ ] **Prompt novel structure generation** — add few-shot examples to interpretation debate prompt showing valid `novel_structure` format to encourage agents to propose new structures

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
