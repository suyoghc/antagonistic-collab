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
- [ ] Assess whether critique quality degrades over cycles (circular "my model can also predict that" pattern)
- [ ] Run longer debates (5+ cycles) to check whether RMSE gap widens monotonically
- [x] **Make the debate influence outcomes** — divergence-driven experiment selection picks the most diagnostic structure (D12)
- [ ] **Investigate GCM flexibility confound** — GCM outperforms RULEX on RULEX-generated data (4.7% gap after 4 rounds of improvements). May need structures specifically designed to disadvantage GCM, or longer runs to accumulate signal.

### Queued (from Codex review, 2026-03-13)
- [x] **[P1] Implement Phase 5 (Design Revision)** — agents now revise proposals based on critiques, updating design_spec via state.revise_proposal()
- [x] **[P2] Fix moderator reject path** — rejection now loops back to proposal→critique→revision→arbitration (up to 3 attempts), rejected proposals marked with status="rejected"
- [x] **[P2] SUSTAIN.predict_learning_curve() ignores test_items/test_labels** — fixed: now evaluates test items at each block boundary (matches GCM contract)
- [ ] **[P3] Fix --demo order-sensitivity** — `sys.argv[1] == "--demo"` fails when other flags precede it
- [x] **Show concrete model predictions in divergence ranking** — ranked summary now shows per-model predicted accuracy (e.g., "Exemplar_Agent: 0.75, Rule_Agent: 0.80, Clustering_Agent: 0.50") so agents can see their advantage per structure

### Done (Codex review, 2026-03-14)
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
