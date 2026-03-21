# Decision Log

Tracks what was changed, why, what alternatives were considered, and what's still open.

---

## D1: Initial bug fixes (commits 5e3d726 → e41ae99) — 2026-03-11 23:04 → 2026-03-12 12:55

**Problem:** Multiple P1 crashes and data-integrity bugs found during code review.
**Key fixes:**
- JSON parser couldn't handle nested braces → replaced regex with brace-depth parser
- Critique provenance: critiques were sprayed to all proposals → now matched by title
- numpy int64 keys broke JSON serialization → recursive sanitization
- NaN/None values crashed leaderboard formatting → guards added
- Shared-reference mutation in theory revisions → deep copy snapshots

**Alternative considered:** None — these were clear bugs with clear fixes.

---

## D2: Princeton AI Sandbox backend (commits 296112f → 2624d9f) — 2026-03-12 16:40 → 20:56

**Problem:** Needed to run debates with GPT-4o through Princeton's compute allocation.
**Decision:** Add `--backend princeton` using Portkey gateway (api.portkey.ai).
**Alternative considered:** Direct Azure OpenAI — tried first but Princeton uses Portkey, not raw Azure.

---

## D3: Markdown reports and auto-naming (commit be4f44b) — 2026-03-12 21:30

**Problem:** Debate output was only JSON transcripts — hard to read.
**Decision:** Add per-cycle `.md` transcripts and a `summary.md` with leaderboard + theory trajectories. Auto-generate output dirs as `runs/True_{model}_LLM_{llm}_COLLAB_{agents}_{NN}/`.
**Also fixed:** Cumulative transcript bug — each cycle's JSON was accumulating all prior messages.

---

## D4: Duplicate JSON in markdown + empty leaderboard (commit 8f837d6) — 2026-03-12 21:41

**Problem:** Cycle markdown had duplicate JSON blocks. Prediction leaderboard was always empty because agents predicted with arbitrary metric keys while scoring expected `mean_accuracy`.
**Decision:** Remove duplicate JSON rendering; explicitly instruct agents to include `mean_accuracy` in prediction prompt.

---

## D5: Batch-mode moderator rotation (commit c193d0a) — 2026-03-12 22:39

**Problem:** `run_human_arbitration()` in batch mode always picked `approve 0`. Since agents propose in list order, Exemplar_Agent always won all 3 cycles.
**Decision:** Round-robin by prior approvals + critique-count tiebreaker.
**Alternatives considered:**
- Random selection — simple but not reproducible/explainable
- LLM-based selection — adds API cost, non-deterministic
- Weighted scoring formula — over-engineered for the current prototype

**Result:** Verified in a 3-cycle run: Cycle 0 → Exemplar, Cycle 1 → Rule, Cycle 2 → Clustering.

---

## D6: Synthetic data always returns 0.550 (OPEN) — 2026-03-13

**Problem:** Every experiment produces `mean_accuracy = 0.550` regardless of the LLM's proposed design. The debate cannot converge because the data never varies.

**Root cause (2 parts):**
1. LLM-designed experiments aren't parsed into `stimuli`/`labels` format — generator always falls back to Shepard Type II with fixed params and seed.
2. Even if parsing worked, model params are hardcoded (e.g., GCM always uses `c=4.0`), so varying the category structure is the only lever.

**Options under consideration:**
- **(A) Map proposals to existing structure library** — LLM picks from Shepard I-VI, 5-4, rule-plus-exception, linear-separable instead of inventing structures. Reliable, already implemented.
- **(B) Parse LLM-designed structures** — Translate LLM's freeform specs to stimuli/labels. Brittle, LLM format varies each time.
- **(C) Vary params by experimental conditions** — Map conditions (e.g., "cognitive load") to param perturbations. Makes same structure produce different data.
- **(D) Pre-compute experiment menu from divergence map** — Run all models on all structures, present results where they disagree most. Agents debate which experiment is most informative.

**Leaning toward:** A + C (structure library + condition-to-param mapping). Smallest change, biggest impact.

**Status:** Resolved — see D7.

---

## D7: Structure registry + condition mapping + item-level scoring — 2026-03-13

**Problem:** D6 — every experiment produced identical data (mean_accuracy=0.550).

**Decision:** Implemented Option A + C:
1. `STRUCTURE_REGISTRY` — 11 structures (Shepard I-VI, 5-4, rule+exception ×2, linear-separable ×2) built from existing `category_structures.py` functions.
2. `CONDITION_EFFECTS` — 5 conditions (baseline, low_attention, high_attention, fast_presentation, high_noise) mapping to per-model parameter overrides.
3. Rewrote `_synthetic_runner()` to look up `structure_name` from registry, apply condition overrides, use deterministic per-experiment seeds (md5-based).
4. Fixed scoring filter in `runner.py` to merge `item_accuracies` into actual dict for per-item scoring.
5. Updated prompts: experiment proposal shows structure/condition menus, prediction prompt guides item-level predictions, divergence context ranks structures by divergence.

**Alternatives considered:** Option B (parse LLM-designed structures) — too brittle. Option D (pre-computed menu) — deferred, could be added later.

**Result:** 9 new tests, 82 total passing. Different structures/conditions/cycles now produce genuinely different data. However, 3-cycle validation run showed Clustering_Agent beating Exemplar_Agent despite GCM being ground truth — because agents guess predictions rather than running their models. See LESSONS_LEARNED.md 2.1.

---

## D8: Agents must call their models for predictions (NEXT) — 2026-03-13

**Problem:** LLMs write down numerical predictions by reasoning about what their model would predict, rather than running `model.predict()`. This confounds RMSE scoring with LLM calibration quality, not model fit.

**Decision:** Agents should call their actual models during the prediction phase. The runner should run `model.predict()` with the agent's stated params on the approved structure and use those outputs as the predicted pattern.

**Decision:** Added `compute_model_predictions()` to `DebateProtocol` — takes an agent config, structure name, and condition, runs the agent's model on the structure with condition overrides, returns per-item P(correct label) and mean_accuracy. Modified `run_execution()` so the LLM provides reasoning and confidence while the system computes predictions via `model.predict()`. RULEX uses seed=42 for determinism. Model name mapping: `model_class.name.split()[0]` → CONDITION_EFFECTS key.

**Files changed:** `debate_protocol.py` (new method), `runner.py` (prediction loop + prompt), `tests/test_bugfixes.py` (7 new tests).

**Result:** 91 tests pass (including param_overrides fix from Codex review). Predictions are now model-computed, not LLM-guessed. Validated with 3-cycle run: Exemplar_Agent wins with RMSE=0.0776, 3.6x lower than nearest competitor (Rule_Agent=0.2755, Clustering_Agent=0.3528). See LESSONS_LEARNED.md 3.1–3.4.

**Status:** Done.

---

## D9: Code review bug fixes (8 issues) — 2026-03-13

**Problem:** Comprehensive code review identified 10+ issues across runner.py, models, debate_protocol.py, and epistemic_state.py. Most were crash-level bugs triggered by edge cases (invalid params, missing data, malformed LLM output).

**Decision:** Fixed the 8 highest-impact issues:
1. **runner.py:647** — `:.3f` format applied to string `'N/A'` → conditional formatting
2. **runner.py:593** — `design_spec.get()` without dict check → validate type first
3. **gcm.py:58** — `r=0` causes ZeroDivisionError → raise ValueError with message
4. **gcm.py:108** — incomplete bias dict causes KeyError → validate keys, raise ValueError
5. **sustain.py:74** — zero lambdas cause NaN (division by zero) → fallback to mean(dim_sim)
6. **epistemic_state.py:717** — `to_json()` doesn't create parent dirs → add `os.makedirs()`
7. **debate_protocol.py:530** — NaN in model probs corrupts divergence map → `nan_to_num(0.5)`
8. **debate_protocol.py:574** — unknown condition name silently uses defaults → add warning

**Tests:** 13 new regression tests, 104 total passing. Also fixed a weak source-inspection test that checked for single-quoted string literal instead of the actual semantic content.

**Alternatives considered:** For SUSTAIN zero-lambdas, considered raising an error, but `mean(dim_sim)` is the mathematically sensible degenerate case (uniform attention → unweighted average).

**Status:** Done.

---

## D10: SUSTAIN predict_learning_curve fix + call_agent retry — 2026-03-13

**Problem 1 (P2):** `SUSTAIN.predict_learning_curve()` accepted `test_items` and `test_labels` parameters but ignored them. It reported training accuracy from `trial_log["correct"]` instead of evaluating held-out test items at each block boundary. This broke the contract shared with `GCM.predict_learning_curve()`.

**Decision:** Rewrote the method to simulate learning incrementally (training on items up to each block boundary), then classify `test_items` using the current cluster state. This matches GCM's approach: at each block, test all held-out items and report accuracy.

**Trade-off:** The new implementation re-trains from scratch at each block boundary (SUSTAIN is order-sensitive, so we can't just append). This is O(blocks × sequence_length) instead of O(sequence_length), but correctness > speed for validation runs.

**Problem 2:** `call_agent()` raised exceptions on any API error (network, rate limit, timeout), crashing the entire multi-cycle debate. No retry logic existed.

**Decision:** Added exponential backoff retry (up to 3 attempts, 1s/2s/4s waits). `ValueError` from empty responses (content filter) is NOT retried — only transient errors.

**Tests:** 4 new regression tests, 108 total passing.

**Status:** Done.

---

## D11: Leave-one-out prediction to fix self-prediction bias — 2026-03-13

**Problem:** Multi-model validation showed SUSTAIN (Clustering_Agent) winning even when GCM or RULEX was the ground truth model. Root cause: `compute_model_predictions()` trained and tested models on the same items. For GCM, predicting item i with item i in the training set gives distance=0, similarity=1.0 — a self-match that produces near-binary predictions for every item. These over-confident predictions don't match noisy synthetic data (binomial sampling), while SUSTAIN's softer cluster-based predictions accidentally fit noise better.

**Decision:** Implement leave-one-out (LOO) cross-validation — when predicting item i, exclude item i from the training set. Applied to both `compute_model_predictions()` and `compute_divergence_map()`. LOO is standard practice in the GCM literature (Nosofsky 1986).

**Alternatives considered:**
- (A) Hold-out split — discards data, reduces training set quality
- (B) Rank correlation instead of RMSE — changes metric semantics, doesn't fix root cause
- (C) Noise-free evaluation (compare to ground truth model, not noisy data) — useful as secondary metric but doesn't address the overfitting

**Verification:** With LOO, GCM on `rule_plus_exception_1exc` (c=6.0) produces varied per-item predictions (0.13–0.87) instead of all-identical values. Full/LOO predictions differ for every item, confirming self-matches are excluded.

**Tests:** 4 new regression tests, 112 total passing.

**Status:** Done. Re-run multi-model validation to confirm correct agents now win.

---

## D12: Divergence-driven experiment selection — 2026-03-14

**Problem:** Batch-mode arbitration used round-robin agent rotation to pick which experiment to run. This ignored the diagnostic value of different category structures. RULEX consistently lost because Type_VI (where it's weakest — no simple rule exists) was always selected in cycle 0 by Exemplar_Agent. Only 1 of 3 experiments tested a rule-favorable structure.

**Decision:** Replaced round-robin with divergence-driven selection. The moderator now:
1. Computes `compute_divergence_map()` over all 11 structures in `STRUCTURE_REGISTRY`
2. For each proposal, looks up its `structure_name` in the divergence map
3. Picks the proposal whose structure has the highest max pairwise divergence between models
4. Falls back to critique count on ties (more scrutinized = more refined)

Also expanded `compute_divergence_map()` to use the full `STRUCTURE_REGISTRY` (was only Shepard I–VI + five_four). This ensures divergence map keys match registry keys directly.

**Alternatives considered:**
- (A) Weighted random sampling by divergence — adds noise, harder to debug
- (B) LLM moderator selects based on divergence context — adds API cost and latency, LLM may not optimize well
- (C) Pre-computed optimal sequence — inflexible, doesn't adapt to debate state

**Tests:** Updated `test_batch_mode_rotates_proposals` → `test_batch_mode_divergence_driven`, fixed div map key lookups, 115 total passing.

**Status:** Done. RULEX gap narrowed from 16.7% to 5.5% but Exemplar_Agent still won. See D13.

---

## D13: Concrete model predictions in divergence ranking + param filter — 2026-03-14

**Problem 1:** Agents saw divergence scores (e.g., "linear_separable_2d — 0.619") but couldn't tell which model wins on each structure. Rule_Agent kept proposing Type_II (lowest divergence, 0.160) because it couldn't see that RULEX dominates on linear_separable structures.

**Decision:** Added per-model predicted accuracy to the ranked summary:
```
1. `linear_separable_2d` — max divergence = 0.619
   Predicted accuracy: Exemplar_Agent: 0.55, Rule_Agent: 0.65, Clustering_Agent: 0.50
```
Also updated the proposal prompt to explicitly direct agents toward structures where their model has the highest accuracy.

**Problem 2:** LLM agents proposed param_overrides with invented parameter names (e.g., `w_i` for GCM) that crashed `model.predict()` with `TypeError`. Discovered during RULEX validation run _05.

**Decision:** Filter params using `inspect.signature(model.predict)` before calling — only pass keys that the method actually accepts. Valid overrides still work, invalid ones are silently dropped.

**Tests:** 7 new regression tests, 122 total passing.

**Status:** Done. Re-running RULEX validation to check if agents now propose better structures.

---

## D14: Malformed param override fallback + SUSTAIN partial block — 2026-03-14

**Problem 1 (Codex P1):** `compute_model_predictions()` filters unknown param *keys* but not malformed *values*. An LLM override like `attention_weights=[0.5,0.5,0.5]` on a 4D structure causes `ValueError: operands could not be broadcast together with shapes (3,) (4,)`. Similarly, `c="high"` causes `TypeError`. One bad suggestion aborts the entire execution phase.

**Decision:** Wrap `model.predict()` in `try/except (ValueError, TypeError)`. On first failure, switch to default params for all remaining items in the LOO loop. This avoids silently corrupting individual item predictions while still completing the run.

**Problem 2 (Codex P2):** `SUSTAIN.predict_learning_curve()` iterates `range(block_size, len(seq)+1, block_size)`, which skips the final partial block when `len(seq) % block_size != 0`. For a 10-item sequence with block_size=4, SUSTAIN produces 2 blocks (at 4, 8) while GCM produces 3 (at 4, 8, 10). This makes curve lengths disagree between models.

**Decision:** After building the complete-block list, append `len(training_sequence)` if there's a remainder. Now all three models produce curves of the same length.

**Tests:** 6 new regression tests, 128 total passing.

**Status:** Done.

---

## D15: Implement Phase 5 (Design Revision) — 2026-03-14

**Problem:** Phase 5 was a placeholder — `PhaseResult(phase=Phase.DESIGN_REVISION, outputs={})`. Critiques were generated in Phase 4 but never led to revised proposals. The full provenance chain (proposal → critique → revision) was broken.

**Decision:** Implemented `run_design_revision()`:
1. For each proposal with critiques, call the proposing agent with critique context
2. Agent returns revised JSON with `structure_name`, `condition`, `changes`, `addresses_critiques`
3. If valid, call `state.revise_proposal()` to update the design_spec and log the revision
4. If LLM output is unparseable or references invalid structures, keep the original proposal

**Design choices:**
- Only revise proposals that received critiques — uncritiqued proposals stay as-is
- Require `addresses_critiques` indices to maintain provenance; invalid indices fall back to addressing all
- Include structure menu in prompt so agents can switch structures
- Revised `design_spec` replaces the original, so Phase 6 (moderator) sees the revised version

**Tests:** 5 new regression tests, 133 total passing.

**Status:** Done. The 9-phase debate loop is now structurally complete.

---

## D16: Fix moderator reject path (P2) — 2026-03-14

**Problem:** In interactive mode, typing "reject" at the moderator prompt printed a placeholder message ("In full version, this loops back.") and fell through. No experiment was approved, so `run_execution` found nothing and skipped. The cycle was wasted.

**Decision:** Two changes:
1. `run_human_arbitration()` now sets `outputs["rejected"] = True` when the moderator rejects, signaling to `run_cycle()`.
2. `run_cycle()` wraps Phases 3–6 (proposal → critique → revision → arbitration) in a retry loop. On rejection, old proposals are marked `status="rejected"` and the loop restarts from Phase 3 via `skip_to_phase()`. Capped at `MAX_REJECT_RETRIES = 2` (3 total attempts). After exhausting retries, proceeds with no approved experiment (execution phase already handles this gracefully).

**Alternatives considered:**
- (A) Loop inside `run_human_arbitration` only — can't re-run proposals/critiques, just re-prompts the moderator with the same proposals
- (B) No retry cap — risks infinite loops if moderator keeps rejecting
- (C) Auto-approve after max retries — masks the moderator's objection; better to skip the cycle visibly

**Tests:** 7 new regression tests, 140 total passing.

**Status:** Done.

---

## D17: Structure diversity penalty in experiment selection — 2026-03-14

**Problem:** 5-cycle validation runs revealed that divergence-driven selection picks the same high-divergence structure every cycle. In the RULEX validation, Type_VI was selected 4/5 times — a structure where RULEX has no advantage (no simple rule exists). Clustering_Agent won 12/15 experiment selections across all 3 runs. Rule_Agent never won a single selection (0/15).

**Validation results (before fix):**

| Ground Truth | Winner | RMSE | Correct? |
|---|---|---|---|
| GCM | Exemplar_Agent | 0.342 | YES (15.1% gap) |
| SUSTAIN | Clustering_Agent | 0.344 | YES (32.2% gap) |
| RULEX | Clustering_Agent | 0.500 | NO (0.2% gap, random) |

**Root cause:** Raw divergence doesn't account for information gain. Testing Type_VI a 4th time adds no new information but keeps winning because its raw divergence score is highest.

**Decision:** Add a diversity penalty — structures tested in prior cycles get their effective divergence halved per prior use: `effective_div = raw_div / 2^n_prior`. This means:
- First test: full divergence
- Second test: half divergence
- Third test: quarter divergence

Untested structures with moderate divergence can now beat heavily-tested high-divergence structures.

**Alternatives considered:**
- (A) Ban repeated structures entirely — too aggressive, same structure with different conditions can be informative
- (B) Random selection weighted by divergence — loses determinism, harder to debug
- (C) Round-robin agent rotation (reverted D12) — doesn't optimize for discriminability

**Tests:** 3 new regression tests, 145 total passing.

**Verification (5-cycle RULEX re-run with diversity penalty):**

| Metric | Before (no diversity) | After (two-tier penalty) |
|---|---|---|
| Rule_Agent RMSE | 0.504 (3rd) | **0.433 (1st)** |
| Exemplar_Agent RMSE | 0.501 (2nd) | 0.441 (2nd) |
| Clustering_Agent RMSE | 0.500 (1st) | 0.515 (3rd) |
| Gap (1st vs 2nd) | 0.2% | 1.8% |
| Structures tested | 2 unique | 4+ unique |

Rule_Agent now wins. Gap is small (1.8%) vs GCM (15.1%) and SUSTAIN (32.2%) — this is the genuine GCM flexibility confound, not a system bug.

**Status:** Done.

---

## D18: Bayesian information-gain experiment selection — 2026-03-14

**Problem:** The heuristic diversity penalty (D17) halves divergence per prior use, which doesn't account for what was *learned* from prior experiments. It can't reason about which untested structure would best discriminate between models given accumulated evidence. For example, it penalizes Type_VI equally after observing it once regardless of whether that observation was highly informative or not.

**Decision:** Replace heuristic with principled Bayesian adaptive design:

1. **`ModelPosterior`** dataclass — stores log-probabilities over 3 models, supports Bayesian update, serialization, entropy computation
2. **`compute_log_likelihood()`** — item-level binomial log-PMF, clips predictions to [0.01, 0.99] to avoid -inf
3. **`compute_eig()`** — Monte Carlo expected information gain: for each model as hypothetical ground truth (weighted by prior), simulates 200 datasets, computes posterior entropy, returns `EIG = H_current - E[H_posterior]`
4. **`select_experiment()`** — evaluates EIG for each candidate proposal, returns best index + all scores
5. **`update_posterior_from_experiment()`** — after execution, computes log-likelihoods under all models and updates posterior

**Integration points:**
- `EpistemicState.model_posterior` field stores serialized posterior across cycles
- `run_human_arbitration()` (batch mode) uses EIG to select experiments, prints EIG ranking
- `run_execution()` updates posterior after scoring predictions, prints P(model) + entropy
- `summary_for_agent()` shows Bayesian posterior to agents for context
- `--selection bayesian|heuristic` flag (default: bayesian) for backward compatibility

**Alternatives considered:**
- (A) Improve heuristic penalty formula — doesn't solve the fundamental problem of not reasoning about information
- (B) Mutual information instead of EIG — equivalent under Jensen's inequality, but EIG is more intuitive
- (C) Analytical EIG (conjugate models) — binomial-beta conjugate exists but requires per-model per-item tracking, Monte Carlo is simpler and generalizes

**Tests:** 12 new regression tests, 158 total passing. Covers: prior initialization, posterior update, serialization roundtrip, log-likelihood ordering, clipping, EIG non-negativity, EIG near zero when certain, EIG discriminability ordering, determinism with seed, select_experiment validity, EpistemicState field + JSON roundtrip.

**Status:** Done. Pending: 5-cycle validation runs to compare with heuristic results.

---

## D19: Debate-as-hypothesis-generator architecture — 2026-03-14

**Problem:** With Bayesian EIG implemented (D18), the LLM debate is redundant for experiment *selection* — EIG can search 55 candidates (11 structures × 5 conditions) faster and better than 3 agents proposing from a menu. The debate should shift to where LLMs genuinely add value: generating hypotheses, interpreting results, detecting confounds, and proposing novel experiments beyond the fixed registry. Additionally, the 1.8% GCM-RULEX gap (D17) suggests a second evidence channel is needed to break ties.

**Decision:** Three-phase refactor:

**Phase A: Full-pool EIG + Interpretation Debate**
1. `generate_full_candidate_pool()` — all 55 structure×condition pairs
2. `select_from_pool()` — EIG over full pool without ExperimentRecords
3. `run_full_pool_selection()` — creates+approves winning experiment, prints top-10 EIG landscape
4. `run_interpretation_debate()` — agents produce structured JSON: interpretation, confounds, hypotheses, optional novel_structure proposals, optional theory revision
5. `run_interpretation_critique()` — agents challenge each other's interpretations
6. `run_cycle()` accepts `mode="full_pool"|"legacy"` — legacy preserves 9-phase flow
7. `EpistemicState.agent_hypotheses` field stores hypotheses across cycles

**Phase B: Learning Curves as Second Evidence Channel**
1. `DebateProtocol.compute_learning_curve_predictions()` — runs all 3 models' `predict_learning_curve()` on a structure with n_epochs passes
2. `extract_curve_features()` — extracts: final_accuracy, onset_block, max_jump, n_big_jumps, monotonic, mean_slope, learning_pattern (gradual|sudden|stepwise)
3. `update_posterior_from_experiment()` extended: optional `learning_curves` parameter; curve RMSE converted to log-likelihood-like score, weighted at 0.5× accuracy evidence

**Phase C: Novel Structure Generation**
1. `validate_novel_structure()` — checks 2D stimuli, matching labels, 4–32 items, ≤8 dims, ≥2 categories
2. `DebateProtocol.temporary_structures` dict — per-cycle storage for novel structures from agents
3. `generate_full_candidate_pool()` accepts `extra_structures` parameter
4. `run_full_pool_selection()` merges temporary_structures into pool

**New phase flow (full_pool mode):**
```
Cycle N:
  1. Commitment (cycle 0 only)
  2. Divergence Mapping
  3. Full-Pool Bayesian Selection (replaces phases 3-6)
  4. Execution (+ learning curve comparison)
  5. Interpretation Debate (replaces fire-and-forget interpretation)
  6. Interpretation Critique (new)
  7. Audit
```

**Alternatives considered:**
- (A) Keep proposals but make EIG a tiebreaker — still wastes LLM calls on experiment selection when EIG does it better
- (B) Remove LLM from the loop entirely — misses the value of hypothesis generation and confound detection
- (C) Separate learning curves into an independent system — loses the integration with Bayesian posterior updates

**Tests:** 31 new regression tests (12 Phase A + 9 Phase B + 10 Phase C), 189 total passing. ruff clean.

**Status:** Done. Validated end-to-end with 2-cycle Princeton/GPT-4o run (see D20). Pending: 5-cycle comparative runs and wiring learning curves into execution.

---

## D20: Fix full_pool mode phase state machine desync — 2026-03-14

**Problem:** Full_pool mode's `run_cycle()` called `advance_phase()` after EIG selection, but the phase state machine was still at `EXPERIMENT_PROPOSAL` (from the divergence mapping advance). Since `advance_phase()` uses `self.current_phase` — not the result's phase — to determine the next state, it transitioned to `ADVERSARIAL_CRITIQUE` instead of `EXECUTION`. The state machine never reached `AUDIT`, so `advance_cycle()` never fired and the cycle counter stayed at 0. Discovered by integration test; unit tests didn't catch it because they tested individual phase functions in isolation.

**Decision:** Insert `skip_to_phase(Phase.HUMAN_ARBITRATION)` before the full_pool selection's `advance_phase()` call. This restores the correct transition chain: HUMAN_ARBITRATION → EXECUTION → INTERPRETATION → AUDIT → advance_cycle().

**Alternatives considered:**
- (A) Add a separate transition map for full_pool mode — more complex, duplicates logic, higher maintenance burden
- (B) Make `advance_phase()` accept an explicit "from_phase" override — changes the core API, risks downstream breakage
- (C) Skip directly to EXECUTION after selection (bypassing advance_phase entirely) — loses the phase history record

**Tests:** Added `TestFullPoolIntegration` — 2-cycle end-to-end integration test with mocked LLM, verifying the complete pipeline (commitment → divergence → EIG → execution → interpretation debate → critique → audit) for both cycles. 190 total passing.

**Verification:** Real 2-cycle run with Princeton/GPT-4o, GCM ground truth:
- Cycle 0: EIG selected `five_four / fast_presentation`, Exemplar_Agent RMSE=0.170
- Cycle 1: EIG selected `Type_I / low_attention`, Exemplar_Agent RMSE=0.139 (cumulative)
- Correct agent wins with 2.1x gap over second place

**Status:** Done.

---

## D21: Codex review round 3 — 4 bug fixes — 2026-03-14

**Source:** Codex automated review flagged 4 issues (2 P1, 2 P2).

**Bug 1 (P1): Override fallback drops condition params.**
`compute_model_predictions()` builds params as defaults → condition → LLM overrides. On malformed LLM override (e.g. wrong-shape attention_weights), the fallback used bare `agent_config.default_params`, silently dropping the condition (e.g. `low_attention` sets GCM `c=1.5` but fallback reverted to `c=3.0`).
**Fix:** Build `fallback_params` from defaults + condition overrides (excluding LLM overrides), so the condition survives.

**Bug 2 (P1): Scalar `addresses_critiques` crashes design revision.**
`run_design_revision()` iterates `addresses_critiques` as a list. LLM returns `"addresses_critiques": 1` (truthy scalar) → `TypeError: 'int' object is not iterable`.
**Fix:** Coerce non-list values to `[value]` before iterating.

**Bug 3 (P2): Invalid `approve 99` escapes retry loop.**
Out-of-range index nulls `idx` but doesn't set `rejected = True`. The outer `run_cycle()` retry loop checks `outputs["rejected"]`, which is absent, so it breaks out — cycle continues with no approved experiment.
**Fix:** Added `else: rejected = True` after the `if idx is not None` block.

**Bug 4 (P2): SUSTAIN partial block duplicate label.**
`predict_learning_curve()` computes block label as `(block_end // block_size) - 1`. For partial final block (e.g. `block_end=3, block_size=2`), gives `0` — same as the prior complete block.
**Fix:** Use `enumerate` index instead of arithmetic formula.

**Tests:** 4 regression tests, 194 total passing, ruff clean.

**Status:** Done.

---

## D22: Codex review round 4 — RULEX determinism, cleanup — 2026-03-14

**Source:** Codex automated review flagged 7 issues. Fixed 4, deferred 1, dismissed 2.

**Bug 1: RULEX ground truth non-deterministic.**
`_synthetic_runner()` called `RULEX.predict()` without a seed. RULEX's `find_best_rule()` uses a stochastic search (`rng = np.random.default_rng(seed)`), so `seed=None` produces different results each run. `compute_model_predictions()` already had `seed=42` for RULEX — the synthetic runner was the oversight.
**Fix:** Add `"seed": 42` to RULEX params in `_synthetic_runner()`.

**Bug 2: RULEX `predict_learning_curve()` drops seed.**
`predict_learning_curve()` merges `**params` into `p` dict, but calls `find_best_rule()` with explicit keyword args that omit `seed`. Even if `seed=42` is in `p`, it's silently dropped.
**Fix:** Add `seed=p.get("seed")` to the `find_best_rule()` call.

**Bug 4: Redundant `model_predicted`/`predicted` keys.**
Execution prediction messages contained both keys with the same value.
**Fix:** Removed `model_predicted`, kept `predicted`.

**Bug 5: .gitignore gap.**
Pattern covered `debate_cycle_*.json` but not `.md` transcripts.
**Fix:** Added `debate_cycle_*.md`.

**Deferred:**
- Bug 7 (latent): `GCM.fit()` has the same self-prediction bias fixed in D11, but `fit()` is unused in the pipeline. Logged in TASKS.md.

**Dismissed:**
- Bug 3 (module-level globals): Valid design concern but large refactor with low ROI for current single-process architecture.
- Bug 6 (`__init__.py` code): These are import re-exports, not business logic. Standard Python packaging practice despite CLAUDE.md's general rule.

**Tests:** 4 regression tests, 198 total passing, ruff clean.

**Status:** Done.

---

## D23: Wire learning curves + novel structures into pipeline — 2026-03-14

**Problem:** Phase B (learning curves) and Phase C (novel structures) from D19 were implemented but not connected to the live pipeline. `run_execution()` didn't compute curves, `update_posterior_from_experiment()` was called without the `learning_curves` param, interpretation debate didn't show curve data to agents, and novel structures proposed by agents were captured but never validated/registered.

**Decision:** Four integration changes:
1. `run_execution()` computes learning curves after synthetic data, extracts features, passes `learning_curves=` to Bayesian update, stores on `protocol.state.last_execution_curves`
2. `run_interpretation_debate()` includes learning curve comparison in agent context (pattern/final_accuracy/max_jump/onset_block)
3. `run_interpretation_debate()` validates novel structures via `validate_novel_structure()` and registers valid ones in `protocol.temporary_structures`
4. `compute_learning_curve_predictions()` checks `temporary_structures` in addition to `STRUCTURE_REGISTRY`

**Alternatives:**
- Could have deferred novel structure registration to a separate PR — decided to bundle since the code changes are minimal and independent
- Could have weighted curve evidence differently — kept 0.5x default from D19 Phase B design

**Tests:** 7 new regression tests, 205 total passing, ruff clean.

**Status:** Done.

---

## D24: Few-shot novel structure examples in interpretation prompt — 2026-03-14

**Problem:** Agents could propose novel structures but had no guidance on the required format, constraints, or strategic design principles. The prompt only said `"novel_structure": null or {"name": "...", "stimuli": [[...]], "labels": [...]}`.

**Decision:** Added to the interpretation debate prompt:
- Explicit constraints: 4-32 items, ≤8 dimensions, ≥2 categories
- Concrete example: `diagonal_xor` with 6 items, 2D stimuli, binary labels
- Strategic guidance: "design structures that exploit weaknesses in competing models" with specific suggestions (random categories to challenge rule models, high-dimensional stimuli to challenge exemplar models)

**Alternatives:**
- Multiple examples covering different structure types — decided one clear example is enough; more would bloat the prompt
- Parameterized template — over-engineering for current needs

**Tests:** 1 new regression test, 206 total passing, ruff clean.

**Status:** Done.

---

## D25: Fix summary_for_agent crash on non-string new_predictions — 2026-03-14

**Problem:** `summary_for_agent()` in `epistemic_state.py` crashed with `TypeError: sequence item 0: expected str instance, dict found` during the audit phase. Root cause: LLM agents return structured `new_predictions` (e.g., `{"item": "Type_I", "accuracy": 0.9}`) during interpretation debate revisions, but `summary_for_agent()` assumed predictions were strings and called `'; '.join(...)` on them.

**Fix:** Coerce `new_predictions` items to `str()` before joining: `preds = [str(p) for p in latest["new_predictions"][:2]]`.

**Impact:** This crash blocked all full_pool mode runs from completing cycle 0 → audit. All 3 full_pool validation runs (GCM, SUSTAIN, RULEX) failed at this point.

**Tests:** 1 regression test, 207 total passing, ruff clean.

**Status:** Done.

---

## D26: Cross-LLM validation — GPT-4o vs Claude Sonnet vs Claude Opus — 2026-03-15

**Question:** Does the framework's convergence depend on which LLM serves as the agent backbone?

**Method:** 9 runs total — 3 ground truths × 3 LLMs (GPT-4o via Princeton/Portkey, Claude Sonnet 4, Claude Opus 4). All full_pool mode, 5 cycles, Bayesian EIG selection.

**Results:** Correct model wins in all 9/9 runs. SUSTAIN RMSE identical across all 3 LLMs (0.270). GCM and RULEX show small variation from LLM-proposed param_overrides (GCM: 0.143–0.159, RULEX: 0.148–0.213).

**Key finding:** The one surviving LLM→RMSE feedback path is `param_overrides` during execution. Different LLMs propose different parameter tweaks, creating minor RMSE variation, but never enough to change the winner. Framework is LLM-agnostic for convergence.

**Cost:** ~$1/run (Sonnet), ~$5/run (Opus), ~$1/run (GPT-4o). 68 LLM calls per 5-cycle run.

**Status:** Done.

---

## D27: M5 — Close debate feedback loops — 2026-03-15

**Problem:** M4 analysis revealed debate is epiphenomenal to RMSE: replication runs have zero variance, cross-LLM comparison produces identical winners. Four feedback loops are broken: (1) theory params never propagate to agent_config, (2) param_overrides ephemeral, (3) agent_hypotheses never read, (4) critique content doesn't affect selection.

**Solution:** Implemented 4 features (FEATURES.md 7.1–7.4):
1. **Parameter revision persistence** — `sync_params_from_theory()` copies revised params after interpretation, filtered through `inspect.signature`
2. **Structured claim ledger** — `DebateClaim` dataclass tracks testable predictions across cycles, statuses updated after execution
3. **Critique-as-falsification** — `verify_prediction_claim()` fact-checks agent assertions against actual model computation
4. **Debate-informed EIG weighting** — `select_from_pool()` boosts EIG for candidates distinguishing contested model pairs

**Alternatives considered:** Could have tackled param_overrides persistence instead of theory params, but theory params are the more principled path (LLM reasons about theory, system validates). Could have used a penalty instead of a boost for EIG, but boosting the contested pair is more targeted.

**Validation:** 3/3 ground truths correct. 4× GCM replication: RMSE std=0.018 (was 0.000). ~45 FALSE CLAIMs detected. 24 new tests (231 total).

**Impact:** Debate now causally affects RMSE. First non-zero replication variance in project history.

**Status:** Done.

---

## D28: M6 — arbiter-v0.1 Integration: role-specialized agents & crux negotiation — 2026-03-15

**Problem:** M5 closed feedback loops but debate structure remained flat: all agents share the same prompt template, there's no mechanism for identifying decisive questions, and no structured output summarizing predictions before experiments run. The ARBITER framework (Kachergis et al.) provides an architecture with role-specialized meta-agents, crux-based negotiation, conflict maps, and pre-registration. Our implementation is versioned as **arbiter-v0.1** to distinguish it from future iterations that may change agent roster, model set, or debate protocol structure.

**Solution:** Implemented 5 features (54 tests, 11 commits):

1. **MetaAgentConfig (M6a)** — `MetaAgentConfig` dataclass with role-specific prompts. Integrator synthesizes across theories; Critic challenges weakest argument. Meta-agents respond after theory agents in interpretation debate but don't trigger parameter revisions.

2. **Crux Negotiation (M6b, 6 sub-commits)** — `Crux` dataclass with lifecycle (proposed→accepted→resolved→rejected). `run_crux_identification()`: each agent proposes 1-2 cruxes. `run_crux_negotiation()`: agents accept/reject/counter-propose. `finalize_cruxes()`: 2+ supporters → accepted. Active cruxes converted to `crux_boost_specs` for EIG boosting.

3. **Conflict Map (M6e)** — `category` field on `DebateClaim`, `conflict_map_summary()` groups claims by structure/condition, shows where models make conflicting predictions. Injected into interpretation prompts.

4. **Pre-registration (M6d)** — `generate_preregistration()` produces prediction tables (each model's predicted accuracy per structure), adjudication criteria, active cruxes, prior accuracy. Saved per cycle.

5. **HITL Checkpoints (M6c)** — `hitl_checkpoint()` at crux finalization, EIG selection, and pre-registration. Auto-continues in batch mode, prompts in interactive mode.

**Bugfix:** `summary_for_agent()` crashed with `KeyError: slice(None, 2, None)` when `new_predictions` was a dict. Coerced to list before slicing (`2a57937`).

**Alternatives considered:**
- Could have made meta-agents own computational models — decided against it; meta-agents add value through cross-theory synthesis, not prediction
- Could have used a lower crux acceptance threshold (1 supporter) — chose 2 to prevent rubber-stamping; validation confirmed 15% acceptance rate with real LLMs
- Could have replaced `focus_pair` with `crux_boost_specs` — kept both; they serve complementary purposes (posterior-based vs debate-based)

**Validation:** Live 5-cycle runs with GPT-4o for all 3 ground truths:
- GCM → Exemplar_Agent (RMSE 0.1512, gap 36.4%) ✓
- SUSTAIN → Clustering_Agent (RMSE 0.2700, gap 45.6%) ✓
- RULEX → Rule_Agent (RMSE 0.1187, gap 67.6%) ✓

**Key findings:**
- Falsification dominates: 44 claims falsified, 1 confirmed across all 3 runs
- Crux negotiation is selective: 15% acceptance rate (vs 100% in mock)
- Posterior collapse remains the main bottleneck: EIG≈0 after cycle 0–1
- Winning theories need fewer revisions (Rule_Agent: 0 revisions, 67.6% gap)
- Meta-agents contribute substantively but don't override Bayesian machinery

**Impact:** Full arbiter-v0.1 architecture operational. System now has role specialization, focused debate via cruxes, conflict tracking, and pre-registered predictions. 287 tests total.

**Status:** Done.

---

## D29: Posterior collapse as primary architectural bottleneck — 2026-03-15

**Problem:** M6 live validation revealed that the Bayesian posterior collapses to certainty (P≈1.0) after cycle 0 or 1 in 2 of 3 runs (GCM, SUSTAIN). When the posterior is certain, EIG=0 for all candidates, making remaining cycles uninformative. Crux boost can't overcome zero EIG.

**Observation:** RULEX is the exception — posterior initially favored Exemplar_Agent, then self-corrected by cycle 2 when five_four→Type_I structural variation provided disambiguating evidence. This non-monotonic trajectory demonstrates the value of structural diversity, but the system doesn't actively seek it once the posterior collapses.

**Open question:** How to keep later cycles informative?

**Options under consideration:**
- **Posterior tempering** — raise log-probs to a power <1 to prevent collapse, ensuring later cycles still have non-zero EIG
- **Entropy-based re-exploration** — when posterior entropy drops below threshold, force exploration of untested structures regardless of EIG
- **Multi-hypothesis tracking** — maintain a particle set of posteriors to preserve uncertainty
- **Crux-driven override** — when accepted cruxes exist but EIG=0, run the crux's discriminating experiment regardless

**Status:** Resolved — see D30.

---

## D30: Likelihood tempering to fix posterior collapse — 2026-03-15

**Problem:** D29 identified posterior collapse as the primary architectural bottleneck. Binomial log-likelihood with n_subjects=20 across ~10 items generates ~10 nats of evidence per experiment. After 2 experiments, log-odds reach ~50 nats (ratio ~5×10²¹). EIG=0 for all remaining candidates, making later cycles uninformative.

**Decision:** Likelihood tempering (power posteriors): multiply log-likelihoods by a `learning_rate` (tau) in (0, 1] before adding to the prior. This is well-established in Bayesian statistics (Grünwald 2012 "Safe Bayesian", Bissiri et al. 2016, Miller & Dunson 2019).

**Implementation:**
1. `ModelPosterior.update()` — new `learning_rate` param, validates 0 < lr ≤ 1, applies `self.log_probs += learning_rate * log_likelihoods`
2. `compute_eig()` — new `learning_rate` param, applied in simulated posterior updates
3. `select_from_pool()`, `select_experiment()` — thread `learning_rate` to `compute_eig()`
4. `update_posterior_from_experiment()` — thread to `posterior.update()`, record in history
5. `runner.py` — `_LEARNING_RATE = 1.0` global, `--learning-rate` CLI flag, wired through 3 call sites
6. `__main__.py` — `--learning-rate` in `_build_argparser()`

**Alternatives considered:**
- (A) Entropy-based re-exploration — when H(posterior) < threshold, force untested structures. Addresses symptom not cause.
- (B) Multi-hypothesis tracking (particle posteriors) — preserves more uncertainty but much more complex. Over-engineering for current needs.
- (C) Crux-driven override — run crux experiments when EIG=0. Useful but doesn't fix the posterior itself.

**Tests:** 9 new tests in `TestLikelihoodTempering`: tempered slower than untempered, ordering preserved, backward compatibility at tau=1.0, EIG changes with learning_rate, EIG nonzero after tempered updates, history records learning_rate, select_from_pool threads parameter, CLI parsed, input validation. 296 total passing.

**Default:** tau=1.0 preserves all existing behavior. Recommended tau=0.1–0.3 for synthetic data.

**Status:** Done. Pending live validation with --learning-rate 0.2.

---

## D31: Codex review — 5 bug fixes across Bayesian selection, debate protocol, and RULEX — 2026-03-15

**Problem:** Codex automated review identified 5 bugs in the codebase. All were real issues, not false positives.

**Fixes:**

1. **Ground-truth leakage in curve evidence** (`bayesian_selection.py`): `update_posterior_from_experiment()` used `data["ground_truth_model"]` to select a reference curve when computing curve-based evidence, leaking the answer key into posterior updates. **Fix:** Replaced single-model curve comparison with pairwise curve divergence — compute mean L1 divergence between each pair of models' curves, then score each model by how distinct its curve is from others. No ground-truth reference needed.

2. **Novel structure silent fallback** (`debate_protocol.py`): `compute_model_predictions()` only searched `STRUCTURE_REGISTRY`, silently falling back to Type_II when a novel structure from `temporary_structures` was requested. **Fix:** Merge `self.temporary_structures` with `STRUCTURE_REGISTRY` before lookup: `all_structures = {**STRUCTURE_REGISTRY, **self.temporary_structures}`.

3. **Synthetic data LOO mismatch** (`debate_protocol.py`): `_synthetic_runner()` used full-set predictions (train on all items, predict all items) while the scoring path used LOO predictions. This meant synthetic data was easier than it should be, biasing evidence. **Fix:** Changed `_synthetic_runner()` to use LOO: `loo_stimuli = np.delete(stimuli, i, axis=0)`, `loo_labels = np.delete(labels, i)` for each item.

4. **RULEX curve missing exceptions** (`models/rulex.py`): `predict_learning_curve()` called `_evaluate_rule()` which only applies the rule, ignoring stored exceptions. On `rule_plus_exception` structures, this underestimates RULEX accuracy. **Fix:** Replaced `_evaluate_rule()` loop with `self.predict()` loop so `p_exception` contributes to curve accuracy.

5. **n_subjects not threaded** (`bayesian_selection.py`): `update_posterior_from_experiment()` ignored `data["n_subjects"]`, always using the default of 20. **Fix:** Added `n_subjects = data.get("n_subjects", n_subjects)` before the binomial likelihood computation.

**Tests:** 5 new tests in `test_codex_fixes.py` (306 total passing):
- Curve evidence independent of ground_truth_model (compares log_probs)
- Novel structure produces 4-item predictions (not 8-item Type_II)
- Synthetic runner matches scoring path within 0.35 tolerance
- RULEX curve final accuracy > 0.5 on rule_plus_exception structure
- n_subjects=100 produces lower entropy posterior than default

**Alternatives considered:** None — all were clear bugs with clear fixes.

**Status:** Done.

---

## D32: Tempering calibration — tau=0.005, clip [0.05, 0.95] — 2026-03-15

**Problem:** M7 live validation with tau=0.2 showed posterior still collapsing to entropy=0.0000 on cycle 0. All EIG=0.0000 on cycle 1. Tempering was implemented but not calibrated.

**Root cause:** Two compounding issues:
1. **Near-binary model predictions**: SUSTAIN produces predictions like 0.0005/0.999, which — even clipped to [0.01, 0.99] — generate per-item LL gaps of ~53 nats. Across 20 items × n=30 subjects = ~1000 nats total range.
2. **tau=0.2 insufficient**: 0.2 × 1000 = 200 nats. Probability ratio e^200 ≈ 10^87 — still effectively infinite.

**Decision:** Two-pronged fix:
1. Widen prediction clip from [0.01, 0.99] to [0.05, 0.95] in `compute_log_likelihood` and `compute_eig`. No cognitive model should predict with >95% confidence on individual items.
2. Lower default tau from 0.2 to 0.005. Calibrated so that after 1 experiment: H≈0.6 (meaningful uncertainty), after 5 experiments: H≈0.02 (strong convergence). Updated in runner.py, __main__.py, and default_config.yaml.

**Validation:** 2-cycle run with GCM ground truth:
- Cycle 0: P(GCM)=0.73, entropy=0.635 (was 0.000 before fix)
- Cycle 1: P(GCM)=0.90, entropy=0.325 (was 0.000), EIG=0.233 (was 0.000)
- Correct winner with gradual convergence across cycles

**Tests:** 2 new tests in `TestPredictionClipping`: clip boundary verification, posterior non-collapse integration test. 308 total passing.

**Status:** Done.

---

## D33: M7 5-cycle validation — 2/3 correct, RULEX misidentification — 2026-03-15

**Setup:** 5-cycle runs with all 3 ground truths (GPT-4o via Princeton, tau=0.005, all M6+M7 features enabled).

**Results:**

| Ground Truth | Winner | Correct? | RMSE gap | Entropy trajectory |
|---|---|---|---|---|
| GCM | Exemplar_Agent | YES | 81.1% | 0.64→0.33→0.13→0.03→0.00 |
| RULEX | Exemplar_Agent | NO | 8.2% | 0.65→0.69→0.70→0.48→0.16 |
| SUSTAIN | Clustering_Agent | YES | 97.1% | 0.22→0.01→0.00→0.00→0.00 |

**Key findings:**

1. **Tempering works as intended.** GCM shows textbook gradual convergence: entropy drops monotonically from 0.64→0.00 over 5 cycles. EIG remains nonzero through cycle 4 (0.029). This is the first validation where later cycles are genuinely informative.

2. **RULEX misidentification reveals genuine model overlap.** GCM and RULEX produce very similar predictions on the structures the system selects (linear_separable_4d, nonlinear_complex_5d). RMSE gap is only 8.2% (0.370 vs 0.403). The posterior oscillated — RULEX led on cycles 0 and 2, GCM on cycles 1, 3, 4. This mirrors the known theoretical result that GCM can approximate rule-like behavior through attention weights (Nosofsky 1991).

3. **SUSTAIN is trivially identifiable.** Its predictions are so distinctive (RMSE=0.018 vs 0.631) that posterior collapses by cycle 2 even with tau=0.005. SUSTAIN's near-binary predictions create a unique signature.

4. **Experiment selection lacks diversity.** GCM run selected linear_separable_4d 5/5 times. SUSTAIN run selected it 5/5 times. Only RULEX showed variation (alternating with nonlinear_complex_5d). EIG concentrates on a single structure.

**Comparison to M6 validation (tau=1.0, no tempering):**

| Metric | M6 (no tempering) | M7 (tau=0.005) |
|---|---|---|
| Correct winners | 3/3 | 2/3 |
| Posterior collapse | Cycle 0 | Cycle 2-4 (gradual) |
| EIG on cycle 1 | 0.000 | 0.233 (GCM) |
| RULEX posterior dynamics | Locked cycle 0 | Oscillated 3 times |
| RMSE gap (RULEX gt) | 67.6% | 8.2% |

**Analysis:** The RULEX RMSE gap dropped from 67.6% (M6) to 8.2% (M7). In M6, posterior collapse locked the winner early and the high gap was an artifact of the structure selected on cycle 0. In M7, with tempering allowing exploration, the system discovers that GCM and RULEX are hard to discriminate — which is the scientifically correct conclusion. The 2/3 result is arguably more honest than 3/3.

**Status:** Logged. RULEX misidentification resolved by Thompson sampling (D34).

---

## D34: Thompson sampling for experiment selection — 2026-03-15

**Problem:** Greedy `argmax(EIG)` selects the same experiment every cycle (D33 finding #4). GCM and SUSTAIN runs selected `five_four/fast_presentation` 5/5 times. This prevents exploration of structures that could discriminate hard model pairs (GCM-RULEX).

**Decision:** Replace greedy selection with Thompson sampling — sample experiments proportional to EIG scores. Default strategy; greedy preserved via `--selection-strategy greedy`.

**Alternatives considered:**
1. Ad-hoc diversity bonus (penalize recently selected structures) — rejected per CLAUDE.md rule: prefer established methods over ad-hoc solutions
2. Epsilon-greedy (random with probability ε) — simpler but doesn't use EIG information for exploration
3. Myopic Posterior Sampling (Kandasamy et al. 2019) — more principled but requires computing per-model EIG; deferred as future enhancement

**Implementation:** New `_select_index()` helper in `bayesian_selection.py`. When `strategy="thompson"`, samples from EIG scores as weights; falls back to uniform when all EIG=0. Config option `selection_strategy`, CLI flag `--selection-strategy`.

**Outcome (preliminary, pre-bugfix validation):**
- **3/3 correct** including RULEX (was 2/3 with greedy)
- Thompson explored 3 novel structures that greedy never selected
- Diverged from greedy in cycles 2-3 when EIG scores were closer
- GCM run: 2 unique structures (vs 1 with greedy); RULEX: explored `complex_conjunctive`

**Status:** Implemented, preliminary validation done. Clean validation pending (post-bugfix).

---

## D35: Codex review round 6 — curve bonus, novel structures, divergence map — 2026-03-15

**Problem (P1 — curve bonus):** The pairwise curve divergence bonus in `update_posterior_from_experiment()` was data-independent — it measured inter-model curve distinctiveness, not fit to observed data. With 3 models, the bonus is asymmetric: the model with the most distinctive curves gets a larger bonus regardless of observations. This distorts the posterior toward "distinctiveness" rather than evidence.

**Fix:** Remove the curve bonus entirely. No observed learning curve data exists in the synthetic framework, so there is no valid comparison target. Curves remain available for agent interpretation.

**Problem (P1 — novel structures):** `_synthetic_runner()` only checked `STRUCTURE_REGISTRY`, silently falling back when a novel (agent-proposed) structure was selected. Now urgent because Thompson sampling actually selects novel structures.

**Fix:** `_synthetic_runner()` now checks `{**STRUCTURE_REGISTRY, **self.temporary_structures}`.

**Problem (P2 — divergence map):** `compute_divergence_map()` used only `STRUCTURE_REGISTRY` when called without explicit structures dict. Legacy mode divergence mapping omitted novel structures.

**Fix:** Default structures dict is `{**STRUCTURE_REGISTRY, **self.temporary_structures}`.

**Status:** Fixed, 4 regression tests, 322 total passing.

---

## D36: M8 clean ablation — Thompson vs greedy, 3 ground truths × 2 strategies — 2026-03-15

**Question:** Does Thompson sampling improve model identification compared to greedy EIG selection? How do they compare on accuracy, convergence, and structural diversity?

**Setup:** 6 runs total — 3 ground truths (GCM, RULEX, SUSTAIN) × 2 strategies (thompson, greedy). All runs: full_pool mode, GPT-4o via Princeton, tau=0.005, 5 cycles. Post-bugfix (D35: curve bonus removed, novel structures executable, divergence map includes temporaries).

**Results:**

| Ground Truth | Strategy | Correct? | Winner RMSE | Unique structs | Novel structs | Final entropy |
|---|---|---|---|---|---|---|
| GCM | Thompson | Yes | 0.085 | 5 | 3 | 0.12 |
| GCM | Greedy | Yes | 0.077 | 2 | 0 | 0.01 |
| RULEX | Thompson | Yes | 0.189 | 4 | 2 | 0.16 |
| RULEX | Greedy | Yes | 0.050 | 2 | 0 | 0.06 |
| SUSTAIN | Thompson | Yes | 0.022 | 3 | 1 | 0.00 |
| SUSTAIN | Greedy | Yes | 0.018 | 1 | 0 | 0.00 |

**Key findings:**

1. **Both 3/3 correct.** Post-bugfix, RULEX misidentification from M7 is resolved for both strategies. The curve bonus removal (D35) was the critical fix — the data-independent bonus had distorted RULEX posterior.

2. **Thompson explores 4× more.** 12 unique structures (6 novel) vs 3 unique (0 novel). First time novel structures selected and executed in the framework.

3. **Greedy converges tighter.** Final entropies 0.00–0.06 vs 0.00–0.16. Winner RMSE slightly lower. Expected: greedy concentrates evidence on the single most informative experiment.

4. **The tradeoff is real.** Thompson trades ~50% convergence tightness for ~4× structural diversity. Neither dominates. Thompson is default because exploration is more scientifically valuable (tests more structures, exercises novel structure pipeline).

5. **Debate contributes novel structures, not experiment strategy.** Thompson's random exploration selects agent-proposed structures; debate provides the proposals. But the selection is stochastic, not semantically directed.

**Decision:** Thompson is the default strategy. Greedy preserved via `--selection-strategy greedy` for users who prioritize convergence speed.

**Status:** Done.

---

## D37: Crux-directed Thompson sampling — debate affects experiment selection — 2026-03-15

**Problem:** The crux-to-experiment pipeline had two structural failures:
1. **Parsing failure**: `cruxes_to_boost_specs()` expected `structure/condition` format, but LLM agents wrote free-text descriptions. Zero boost specs were produced across all M6/M7/M8 runs (100+ cruxes proposed, 0 parsed).
2. **Ineffective mechanism**: Even if parsing worked, the multiplicative EIG boost (2×) barely shifted Thompson's sampling distribution when EIG scores were narrowly clustered (e.g., 0.18–0.23).

**Decision:** Replace multiplicative boost with a **mixture distribution**: with probability `crux_weight` (default 0.3), sample uniformly from crux-matching candidates; otherwise sample from standard EIG-weighted Thompson.

**Implementation:**
1. `_select_index()` — new `crux_indices` and `crux_weight` params. Mixture coin flip → uniform crux selection or standard EIG sampling.
2. `select_from_pool()` — new `crux_weight` param. Computes `crux_indices` from `crux_boost_specs`. Removed multiplicative boost.
3. Crux prompt — now shows available structures and conditions with format example.
4. `cruxes_to_boost_specs()` — validates against known structures/conditions, strips whitespace, adds `crux_id`.
5. Runner logging — `crux_directed: bool` and `crux_id` in transcript messages.
6. Config — `crux_weight: 0.3` in default_config.yaml, `--crux-weight` CLI flag.

**Alternatives considered:**
- Fix multiplicative boost only — doesn't solve distribution problem when scores cluster
- Additive bonus — requires tuning relative to absolute EIG values
- Higher multiplicative factor (10×, 100×) — brute force, not principled

**Tests:** 11 new tests in `TestCruxDirectedThompson`, 336 total passing.

**Live validation (2026-03-15):**

| Ground Truth | Winner | Correct? | RMSE | Gap | Cruxes parsed | Crux-directed |
|---|---|---|---|---|---|---|
| GCM | Exemplar_Agent | Yes | 0.084 | 74.7% | 11/34 | 1/5 |
| RULEX | Rule_Agent | Yes | 0.050 | 83.9% | 7/34 | 0/5 |
| SUSTAIN | Clustering_Agent | Yes | 0.033 | 93.1% | 6/37 | 0/5 |

Key outcome: Before M9, 0 crux boost specs were parsed across all runs (100+ cruxes proposed). After M9, 24 parseable specs across 3 runs. 1 crux-directed experiment (GCM run: `rule_plus_exception_1exc/high_noise` from `crux_004`). Debate now causally affects experiment selection — the crux pipeline is operational for the first time.

**Literature context:** The convergence between crux-directed and EIG-driven selection was predicted by Corcoran, Hohwy & Friston (2023, *Neuron*) who argue adversarial collaboration and Bayesian optimal design should be unified. Low format compliance (23%) is consistent with Tam et al. (EMNLP 2024) showing format restrictions degrade LLM reasoning, and Zhou et al. (2023) IFEval showing <80% compliance for all models.

**Status:** Done.

---

### D38: Claim-responsive debate — agents must address falsified claims (2026-03-15)

**Problem:** Agents systematically ignore their own falsification record. The claim ledger tracks falsified claims and injects summaries into interpretation prompts, but agents do not spontaneously engage with this information. They repeat the same 2–3 talking points across all 5 cycles within a run (LESSONS Phase 9, Principle 7), producing multi-cycle debate that lacks cumulative scientific reasoning. The 45:1 false-to-verified prediction ratio (M6) makes the problem quantitatively stark: agents make bold claims, those claims are tested and falsified, and the agents proceed as though the falsification never occurred.

**Decision:** Add a claim-responsive directive to the interpretation debate prompt. When an agent has falsified claims and `_CLAIM_RESPONSIVE` is true (default), the prompt includes a structured `### FALSIFIED CLAIMS` block that:
- Lists each falsified claim with its evidence (e.g., "actual=0.350")
- Requires the agent to respond with one of: revise (adjust theory/predictions), explain (argue the falsification is misleading), or abandon (drop the claim)
- Requests a `"falsified_response"` JSON field with structured reasoning per claim

**Alternatives considered:**
- **Fine-tuning on falsification response:** Would produce more natural engagement but requires training infrastructure and model-specific tuning. Prompt engineering is model-agnostic.
- **Penalizing agents with falsified claims in posterior:** Conflates agent behavior with model quality. An agent may overclaim while its model is correct.
- **Automatic theory revision on falsification:** Removes agent judgment. The "explain" option preserves the possibility that a falsification is misleading (boundary condition, confound) — a legitimate scientific response.
- **Doing nothing (status quo):** The claim summary already appears in prompts. But LESSONS Phase 9 showed agents do not engage with it voluntarily. Explicit directives are needed.

**Literature context:** The design is inspired by Shinn et al.'s Reflexion (NeurIPS 2023), which demonstrated that LLM agents improve significantly when given structured linguistic feedback about prior failures. Claim-responsive debate is a targeted form of reflexion: instead of general "reflect on what went wrong," agents address specific falsified predictions. The revise/explain/abandon trichotomy maps onto AGM belief revision operators (Alchourrón, Gärdenfors & Makinson, 1985). The requirement to respond to disconfirming evidence is a basic norm in dialogical argumentation theory (Walton, 1998).

**Config:** `no_claim_responsive: false` in default_config.yaml (on by default). CLI: `--no-claim-responsive`. Follows the same `no_*` pattern as `no_arbiter` and `no_tempering`.

**Tests:** 7 new tests (TestClaimResponsiveDebate), 343 total passing.

**Live validation (2026-03-15):**

| Ground Truth | Winner | Correct? | RMSE | Gap | Falsified claims | FR rate |
|---|---|---|---|---|---|---|
| GCM | Exemplar_Agent | Yes | 0.071 | 79.3% | 12 | 80% |
| SUSTAIN | Clustering_Agent | Yes | 0.018 | 96.6% | 14 | 80% |
| RULEX | Rule_Agent | Yes | 0.166 | 51.8% | 15 | 80% |

Key outcome: Agents engage with falsified claims at 80% rate (12/15 theory interpretations include structured `falsified_response` fields; 3 missing are cycle-0 where no claims yet exist). All three response actions observed: revise (adjusting parameters), explain (attributing to confounds), abandon (1 instance). "Explain" dominates — consistent with Lakatos's auxiliary hypothesis shielding. Overclaiming persists (claimed 0.65–0.85, actual 0.10–0.50) but agents now confront their failures rather than ignoring them.

**Status:** Done.

---

## D39: Richer design spaces — parametric structures + interpolated conditions — 2026-03-16

**Problem:** The fixed 11-structure × 5-condition registry (55 candidates) constrains EIG to a discrete search space. Optimal experimental design performs best with continuous or near-continuous design spaces (Myung & Pitt 2009; Cavagnaro et al. 2010). Intermediate parameter values (e.g., cluster separation=1.5 between the existing 2.0 entries, or moderate attention between low and high) may reveal model differences that extreme fixed values mask.

**Decision:** Add 13 parametric structures and 2 interpolated conditions, expanding the pool from 55 to 168 candidates. Parametric structures are generated at import time using the same `linear_separable()` and `rule_plus_exception()` generators already used for the base registry, with different parameter combinations. Interpolated conditions are hand-specified midpoints of existing condition parameters.

**Structures added:**
- `linear_separable_{2,3,4,6}d_sep{1.0,1.5,2.5,3.0}` — 7 variants filling in separation × dimensionality gaps
- `rule_plus_exception_{3,5,6}d_{1,2,3}exc` — 6 variants filling in dimensionality × exception count gaps

**Conditions added:**
- `moderate_attention` — midpoint of low_attention (c=1.5) and high_attention (c=6.0): GCM c=3.5, SUSTAIN r=7.5, RULEX p_single=0.5
- `mild_noise` — between baseline and high_noise: GCM c=3.0, SUSTAIN η=0.07/r=7.0, RULEX tol=0.15

**Alternatives considered:**
1. **Continuous parameterization** (sample random parameters per cycle) — More principled for OED but breaks reproducibility and makes debugging harder. Deferred.
2. **Only expand structures, not conditions** — Conditions affect all models through parameter overrides; interpolated conditions provide cheap diagnostic information.
3. **Larger expansion** (50+ parametric structures) — Diminishing returns. EIG computation scales linearly with pool size; 168 is 3× larger, keeping selection under 1 minute.

**Implementation:**
- `PARAMETRIC_STRUCTURES` and `PARAMETRIC_CONDITIONS` dicts in `debate_protocol.py` (generated at import time)
- `generate_full_candidate_pool(richer=True|False)` in `bayesian_selection.py`
- `_synthetic_runner()` and `compute_model_predictions()` resolve parametric entries via merged lookups
- Config: `no_richer_design_space: false`, CLI: `--no-richer-design-space`, global: `_RICHER_DESIGN_SPACE`
- 14 tests (TestRicherDesignSpaces), 315 total passing

**Live validation (2026-03-16):**

| Ground Truth | Winner | Correct? | RMSE | Gap | Param-S | Param-C |
|---|---|---|---|---|---|---|
| GCM | Exemplar_Agent | Yes | 0.075 | 75.8% | 5/5 | 3/5 |
| SUSTAIN | Clustering_Agent | Yes | 0.022 | 95.6% | 5/5 | 1/5 |
| RULEX | Rule_Agent | Yes | 0.053 | 83.7% | 5/5 | 1/5 |

Key outcome: EIG strongly prefers parametric structures — 15/15 experiments selected parametric linear_separable variants. The expanded pool provides diagnostic intermediate-separation stimuli that the fixed registry lacked. Parametric conditions selected less frequently (5/15), suggesting conditions are less informative than structural variation for model discrimination.

**Status:** Done. Superseded by M12 continuous parameterization.

---

## D40: Continuous design space parameterization — 2026-03-16

**Problem:** M11's fixed parametric grid (168 candidates) expanded the pool but structures were still the same every cycle. EIG exclusively selected linear_separable variants with intermediate parameters, suggesting the diagnostic sweet spot is a continuous region, not a fixed set of points. The same 168 candidates are re-evaluated every cycle — no exploration of new parameter regions.

**Decision:** Replace the fixed grid with continuous sampling from parameter ranges. Each cycle draws 50 fresh structures via `_sample_continuous_structures()`: ~60% linear_separable (n_dims ∈ {2,...,8}, separation ∈ Uniform(0.5, 4.0)), ~40% rule_plus_exception (n_dims ∈ {3,...,8}, n_exceptions ∈ {1,...,4}). Seeds are cycle-dependent (42 + cycle × 1000). Config: tri-state `design_space` (base/richer/continuous) replaces boolean `no_richer_design_space`.

**Alternatives considered:**
1. **Keep M11's fixed grid with more points** — Diminishing returns; more points in a fixed grid don't explore new regions across cycles.
2. **Adaptive sampling based on prior EIG results** — More sophisticated but premature; simple uniform sampling already outperforms the fixed grid.
3. **Only linear_separable** (since RPE is never selected) — RPE serves as exploration insurance for edge cases where rule structures might be diagnostic.

**Implementation:**
- `_sample_continuous_structures(n_samples, seed)` in `debate_protocol.py`
- `generate_full_candidate_pool(design_space=, n_continuous_samples=, continuous_seed=)` in `bayesian_selection.py`
- `protocol.sampled_structures` dict + 4 resolution merge points
- Config: `design_space: continuous`, `n_continuous_samples: 50`
- CLI: `--design-space {base,richer,continuous}`, `--n-continuous-samples`
- `--no-richer-design-space` kept as deprecated alias → `design_space: base`
- 16 new tests (TestContinuousDesignSpace), 331 total passing
- Bug fix: attention_weights dimension mismatch on high-dimensional structures

**Live validation (2026-03-16):**

| Ground Truth | Winner | Correct? | RMSE | Gap | Sampled/5 | Cycle Overlap |
|---|---|---|---|---|---|---|
| GCM | Exemplar_Agent | Yes | 0.092 | 76.8% | 5/5 | 0–2% |
| SUSTAIN | Clustering_Agent | Yes | 0.022 | 95.8% | 5/5 | 0–2% |
| RULEX | Rule_Agent | Yes | 0.048 | 87.4% | 5/5 | 0–2% |

Key findings:
1. **15/15 sampled structures selected** — EIG exclusively prefers continuous samples over fixed registry
2. **0% cycle overlap** — different cycles explore genuinely different parameter regions
3. **All linear_separable** — separation sweet spot at 0.68–2.13, dimensionality 4–8D
4. **RULEX gap improved** (87.4% vs M11's 83.7%) — continuous sampling finds more diagnostic experiments
5. **Debate layer is interpretive, not directive** — 0/15 experiments from agent proposals; debate provides mechanistic narratives (80% FR rate) but doesn't influence experiment selection

**Status:** Done.

---

## D41: Debate ablation study — experiment framework + 3×2 ablation — 2026-03-16

**Problem:** After 9 milestones of progressive improvements, debate's causal contribution remained unclear. The computational pipeline (EIG + Bayesian posterior + model predictions) drove all identification outcomes, while debate output was disconnected from the scoring pipeline. Is debate epiphenomenal, or does it contribute in ways not captured by RMSE?

**Decision:** Build a reusable experiment framework and run a 3×2 ablation: No-Debate / Debate-No-Arbiter / Debate+Arbiter × Thompson / Greedy × 3 ground truths = 18 conditions. The "no debate" mode runs only the computational pipeline with zero LLM calls, providing a clean computational-only baseline.

**Alternatives considered:**
1. **2×2 ablation (debate/no-debate only)** — Original plan, but conflates base debate with arbiter-v0.1 features. User pointed out we need to separate debate-without-arbiter from debate-with-arbiter to estimate each contribution independently.
2. **Ablate individual features** — Too many combinations (claim-responsive, crux-directed, meta-agents, etc.). The 3-level debate factor captures the main architectural distinction.
3. **Compare across milestones** — Confounded by other changes between milestones (tempering, continuous design, etc.). Same-config ablation is cleaner.

**Implementation:**
- `antagonistic_collab/experiment.py` (NEW): `ExperimentCondition` dataclass, `load_experiment()` YAML parser (rejects unknown keys), `run_condition()` (saves/restores globals), `run_experiment()` (grid runner + comparison table), `merge_summaries()` (combine multiple runs)
- `antagonistic_collab/runner.py`: `_NO_DEBATE` global, `run_cycle()` skips all LLM phases when set, `run_execution()` computes predictions with default params
- `experiments/debate_ablation.yaml`: 3×2 config
- `experiments/debate_no_arbiter.yaml`: supplemental 2-condition config for separate runs
- CLI: `--no-debate`, `--experiment`, `--merge` flags
- 15 new tests (TestExperimentFramework, TestNoDebateMode, TestDebateAblationConfig)

**Ablation results (18/18 completed):**

| Debate Level | Correct | Avg RMSE | Avg Gap | Avg Time |
|---|---|---|---|---|
| None | 6/6 | 0.055 | 87.6% | 368s |
| Debate (no arbiter) | 6/6 | 0.078 | 82.4% | 1315s |
| Debate + Arbiter | 6/6 | 0.059 | 86.8% | 1107s |

**Key findings:**
1. **Debate is epiphenomenal on synthetic benchmarks.** All 17/17 conditions identify the correct winner regardless of debate level. No-debate has the best RMSE and gap while running 3-4× faster.
2. **Debate without arbiter actively hurts.** LLM param_overrides introduce noise into model predictions. Without crux-directed selection to compensate, this noise degrades discrimination.
3. **Arbiter partially recovers.** Crux-directed Thompson sampling and meta-agent oversight compensate for param_override noise, but still don't beat the computational-only baseline.
4. **The structural gap is architectural.** Debate output doesn't feed back into EIG or predictions. For debate to help, the loop must be closed.

**Future directions (logged in TASKS.md):**
- Model misspecification (models need LLM-proposed param adaptation)
- Non-pre-enumerated design space (LLM agents propose novel structures EIG can't discover)
- Ambiguous data (real human data with noise, individual differences)
- Explanation for humans (goal is understanding, not just identification)

**Status:** Done. All 18/18 conditions complete.

---

## D42: Fuzzy parameter matching for ephemeral sampled structures — 2026-03-16

**Problem:** Claim-directed experiment selection never fired during live validation because sampled structures are regenerated each cycle with different seeds. A claim from cycle 0 targeting `sampled_ls_8d_sep0.99_13` won't exist in cycle 1's pool — the name encodes a random index that changes every cycle.

**Decision:** Implement parameter-based fuzzy matching as a fallback in `claims_to_boost_specs()`. When a claim targets a sampled structure name not in the current pool:
1. Parse the encoded parameters from the name (type=ls/rpe, dims, separation/exceptions)
2. Find the closest match in the current pool by: same type → closest dims (10× weight) → closest separation or n_exceptions

Gated by `_FUZZY_STRUCTURE_MATCH = True` (on by default, toggleable).

**Alternatives considered:**
1. **Persist sampled structures across cycles** — Would break the continuous design space sampling that M12 introduced. Each cycle should explore new regions of parameter space.
2. **Store claims by parameters instead of names** — Would require changing the `DebateClaim` dataclass and all claim-producing code. Too invasive for the payoff.
3. **Exact parameter matching only** — Too strict; separation values are continuous floats unlikely to repeat exactly. Nearest-neighbor is more robust.

**Implementation:**
- `_parse_sampled_params(name)`: regex parser for `sampled_ls_{dims}d_sep{sep}_{idx}` and `sampled_rpe_{dims}d_{exc}exc_{idx}`
- `_fuzzy_match_sampled_structure(name, pool)`: nearest-neighbor by dims (10× weight) + parameter distance
- Modified `claims_to_boost_specs()`: fallback to fuzzy match when exact structure lookup fails
- 12 new tests in `TestFuzzySampledStructureMatch`

**Status:** Implemented, 428/428 tests pass. Live validation: 94 fuzzy matches fired across 3 ground truths. Done.

---

## D43: M14 final assessment — debate epiphenomenal even with closed feedback loop — 2026-03-17

**Problem:** M13 showed debate is epiphenomenal on synthetic benchmarks. M14 closed the debate→computation feedback loop with 3 interventions (claim-directed selection, validated param revisions, claim auto-resolution) + 2 fixes (normalization, fuzzy structure matching). Does the closed loop change the outcome?

**Finding:** No. Despite all interventions firing mechanically (94 fuzzy matches, 12 claim-directed selections, 7/33 param rejections, 45 claims resolved), M14 full does not beat the M13 no-debate baseline:

| Condition | Avg RMSE | Avg Gap% |
|---|---|---|
| M13 No-Debate (Thompson) | 0.066 | 83.5% |
| M13 Debate+Arbiter (Thompson) | 0.070 | 83.2% |
| M14 Full (feedback loop) | 0.127 | 68.2% |
| M14 Ablation (crux_weight=0) | 0.161 | 58.5% |

**Interpretation:** The computational pipeline (EIG + Bayesian posterior + model predictions) is sufficient when models are fully specified and data is clean. Debate adds noise rather than signal because:
1. Claim-directed selection steers toward narratively interesting but computationally non-diagnostic experiments
2. LLM agents overclaim consistently (87% of claims falsified), so claim boosting amplifies bad hypotheses
3. Param validation prevents the worst damage but can't add information the computation doesn't already have

**What M14 *did* establish:**
- Param validation is causally beneficial (prevents 0.02–0.10 RMSE degradation per blocked revision)
- The feedback loop mechanism works — it just operates in a regime where it's unnecessary
- The boundary between computation-sufficient and debate-needed regimes is now empirically characterized

**Next:** Test in the debate-needed regime — model misspecification (M15), open design space (M16), or real data (M17).

**Status:** M14 complete. This is a precise negative result, not a failure — it establishes where the boundary is.

---

## D44: M15 test hardening + param_distance() extraction — 2026-03-17

**Problem:** M15 Phase 2 results are validated via `validate_m15_live.py` (real LLM calls), but the underlying code paths have no unit tests. Codex flagged this as a gap.

**Decision:** Added 19 unit tests in 4 classes to `tests/test_bugfixes.py`:
- `TestValidateNovelStructure` (8 tests) — gates every structure entering the EIG pool
- `TestCruxIdBugfix` (3 tests) — regression for the `spec.get("crux_id")` fix
- `TestParamDistance` (5 tests) — normalized Euclidean distance for param recovery
- `TestParamRecoveryFlow` (3 tests) — param patching + recovery computation

Also extracted `_param_distance()` from `validate_m15_live.py` → `runner.py::param_distance()` (public API) so it's importable and testable. Validation script imports from runner.

**Alternatives:** Could have tested by importing from the script via sys.path manipulation. Chose to extract to runner.py since it belongs with other param-related functions (`validate_param_revision`, `sync_params_from_theory`).

**Status:** 19/19 tests pass. Full suite no regressions.

---

## D45: M16 open design space implementation — 2026-03-17

**Problem:** M14 showed debate is epiphenomenal when the design space is pre-enumerated. M15 showed debate helps under misspecification. M16 tests a different axis: what if the design space itself requires debate? Remove the registry so agents must propose all structures.

**Decision:** Four-part implementation:
1. `generate_full_candidate_pool()` gains `"open"` mode — empty structures dict, pool comes entirely from `extra_structures` (agent proposals)
2. New `run_structure_proposal()` function — runs between divergence mapping and EIG selection. Each agent proposes 2-3 structures via LLM. Valid proposals stored in `protocol.temporary_structures`
3. `run_full_pool_selection()` — empty-pool fallback seeds Type_I + Type_VI
4. Interpretation prompt gains open-mode directive encouraging structure proposals (second proposal opportunity per cycle)

Three conditions: closed_no_debate (M14 baseline), closed_debate (standard), open_debate (agent-proposed only). No arbiter in any condition (M15 showed arbiter hurts).

**Alternatives:** Could have had agents propose structures without dedicated phase (only during interpretation). Chose dedicated phase for clearer signal and more proposals.

**Risks:**
- LLM structures may be low quality → mitigated by `validate_novel_structure()` + fallback
- Small pool reduces EIG effectiveness → by design; tests whether debate adds value
- Agents may propose redundant structures → prompt shows existing pool; could add dedup

**Status:** Phase 1 complete (9 runs, 9/9 correct). Phase 2 (arbiter conditions) in progress.

---

## D46: M16 Phase 2 — Adding arbiter conditions to open design space — 2026-03-18

**Problem:** M16 Phase 1 dropped the arbiter based on M15's finding that meta-agents
distort experiment selection. But M15's arbiter failure was specific: under
misspecification, meta-agents optimized for argumentative richness instead of model
discrimination. M16 asks a different question under correct specification — and the
arbiter's crux machinery might interact differently with an open design space. In
open_debate, agents propose structures based on narrative reasoning; cruxes could
redirect proposals toward actual points of model disagreement.

**Decision:** Expand from 3 to 5 conditions (2×2+1 factorial):
- closed_no_debate (baseline, already run)
- closed_debate (already run)
- **closed_arbiter** (NEW: curated registry + cruxes + meta-agents)
- open_debate (already run)
- **open_arbiter** (NEW: agent-proposed structures + cruxes + meta-agents)

Arbiter conditions enable `_ARBITER=True`, `_CRUX_WEIGHT=0.3`, and instantiate
`create_default_meta_agents()` (Integrator + Critic). Updated
`validate_m16_live.py` with `arbiter` parameter, `--arbiter-only` / `--new-only`
CLI flags.

**Alternatives considered:** Could run only open_arbiter (the most novel combination).
Chose to also run closed_arbiter for completeness — it separates the arbiter's effect
on experiment *selection from registry* vs experiment *design from scratch*.

**Results (14/15 complete, SUSTAIN open_arbiter errored):**

| GT | closed_arbiter gap | open_arbiter gap | vs baseline |
|---|---|---|---|
| GCM | 79.2% | 76.9% | +2.4pp, +0.1pp |
| SUSTAIN | **96.0%** | ERROR | **+8.3pp** (best ever) |
| RULEX | 63.9% | 82.0% | -22.2pp, -4.1pp |

**Answers to key questions:**
1. Closed_arbiter helps SUSTAIN substantially (+8.3pp, best result across all milestones)
   and GCM modestly (+2.4pp). Still hurts RULEX (-22.2pp) but less than M15 (-54.7pp).
2. Open_arbiter recovers open_debate losses for GCM (+0.1pp vs -5.2pp). Neutral on RULEX
   (82.0% vs 82.7%). SUSTAIN errored.
3. Arbiter's M15 failure was NOT a general property — it's a model-type-dependent bias.
   Crux machinery steers toward similarity-based structures: helps SUSTAIN/GCM, hurts RULEX.

**Key insight:** The arbiter is a bias, not noise. It favors similarity-based model
discrimination. Open design is the mirror bias — agent proposals favor rule-diagnostic
structures. Neither is universally good or bad; each helps the model type whose
diagnostic structures it naturally generates.

**Status:** Complete (1 error pending fix — fixed in 0c4837c, re-run filled the cell:
SUSTAIN open_arbiter 70.7% gap, correct winner).

## D47: M17 — Misspecification + Open Design Space — 2026-03-19

**Problem:** M15 showed debate helps under misspecification via parameter recovery.
M16 showed open design helps RULEX via rule-diagnostic proposals. But these were
tested independently. The hardest regime — wrong parameters AND no curated registry —
was untested. This is also closest to real scientific practice. Do the two mechanisms
compose or interfere?

**Decision:** Run 6 new conditions: 3 GTs × 2 open conditions (open_debate,
open_arbiter) with M15's misspecified params. Compare against existing M15 (closed,
misspec) and M16 (open, correct spec) data. Script: `validate_m17_live.py --new-only`.

**Alternatives considered:** Could re-run all 15 conditions (5 per GT) for a complete
M17 factorial. Chose 6 new runs only since closed conditions under misspecification
replicate M15 data, and the scientific question is specifically about the open
conditions.

**Results (6/6 correct):**

| GT | open_debate gap | open_arbiter gap | Param recovery |
|---|---|---|---|
| GCM | 67.3% | **87.8%** (best GCM ever) | 42.9% / 85.7% |
| SUSTAIN | 77.4% | 72.7% | 0% / 0% |
| RULEX | 57.8% | 42.2% (correct!) | 46.3% / 0% |

**Key findings:**
1. GCM open_arbiter (87.8%) — best GCM result across all milestones. Param recovery
   (85.7%) and arbiter-guided proposals compose synergistically.
2. RULEX open_arbiter (42.2%) — correct winner, vs M15 arbiter-RULEX (3.2%, wrong).
   Open design rescues RULEX from arbiter catastrophe by providing rule-diagnostic
   structures that counteract the similarity bias.
3. Composition is non-additive: GCM benefits from synergy, RULEX open_debate loses
   M15's param recovery advantage (60.3% → 46.3%), SUSTAIN is roughly stable.
4. 47/48 correct across M14–M17. Only M15 arbiter-RULEX (closed, misspec) fails.

**Insight:** The complementary biases of arbiter (similarity-favoring) and open
design (rule-favoring) partially cancel under the hardest regime, producing better
results than either alone for GCM and rescuing RULEX from catastrophe. Objectivity
through composition of complementary biases, not through a single unbiased component.

**Status:** Complete.

## D48: R-IDeA as alternative OED — negative result — 2026-03-19

**Problem:** M16 showed every component carries implicit model-type priors. Tang,
Sloman & Kaski (2025, R-IDeA) propose a principled multi-objective acquisition
(representativeness + informativeness + de-amplification) that could reduce these
biases formally rather than through accidental composition. Would R-IDeA beat EIG?

**Decision:** Implement R-IDeA as standalone module (`ridea.py`), test head-to-head
against EIG under correct specification and misspecification, then test R-IDeA + debate
combination via monkeypatch of `select_from_pool()`.

**Results:**

| Condition | EIG | R-IDeA |
|---|---|---|
| Correct spec, no debate | 86.9% | 80.5% |
| Misspec, no debate | 75.1% | 65.4% |
| Misspec, + debate | **81.4%** | **53.7%** |

R-IDeA underperforms EIG in all regimes. R-IDeA + debate is the worst condition
tested — RULEX drops to 19.4% (vs 80.4% with EIG+debate). Root cause:
representativeness term steers away from the most informative experiments,
preventing the visible prediction failures that debate needs to trigger parameter
recovery (RULEX recovery: 0% vs 60.3%).

**Key insight:** Informativeness (EIG) and semantic diagnosis (debate) are
synergistic — debate needs maximally informative experiments to see failures.
Diversification (R-IDeA) and diagnosis are antagonistic — diversification dilutes
the signal. You cannot substitute formal multi-objective optimization for semantic
diagnosis under misspecification.

**Alternatives considered:** Different R-IDeA weights (α, β). Current: α=0.3, β=0.3.
Could tune, but the direction is clear — any weight on representativeness/de-amplification
dilutes informativeness.

**Status:** Complete. Negative result. R-IDeA not integrated into main pipeline.

## D49: Decision-domain debate runner — standalone (Option C) — 2026-03-20

**Problem:** The decision-making domain computational pipeline is complete (3/3
correct under correct spec, 0/3 under misspecification). To test whether debate
recovers misspecification via parameter diagnosis — the core NeurIPS experiment —
we need a debate loop for decision models. Three options evaluated:

- **Option A (refactor DebateProtocol):** Make DebateProtocol domain-agnostic via
  dependency injection. Maximum reuse, but touches the core protocol that produces
  the 47/48 categorization results. Medium-high regression risk. ~4-6 hours.
- **Option B (subclass DebateProtocol):** Override 3-4 methods. Reuses phase machine
  and LLM orchestration. Low risk but fragile — inherits 3000 lines of categorization
  baggage (LOO, learning curves, condition effects, context language). ~2-3 hours.
- **Option C (standalone runner):** New module reusing only domain-agnostic pieces
  (compute_eig, ModelPosterior, decision_eig, decision_runner). Zero risk to
  categorization pipeline. ~3-4 hours.

**Decision:** Option C. Three reasons:
1. The 47/48 categorization results are the core asset — don't risk them.
2. We're testing a hypothesis (does the bias pattern replicate?), not building
   a permanent multi-domain framework. If the pattern replicates, a future
   refactor (Option A) will be easier because the standalone runner makes
   domain-specific requirements explicit.
3. Most computational work is already done. Only new code: LLM debate round
   (~100-150 lines), parameter validation (~50 lines), optional arbiter.

**Coupling analysis:** DebateProtocol is tightly coupled to categorization via:
- `compute_model_predictions()` assumes LOO over category structures
- `_synthetic_runner()` hardcodes STRUCTURE_REGISTRY, CONDITION_EFFECTS
- `compute_learning_curve_predictions()` — irrelevant for decisions
- Context generators reference "accuracy," "structure names," LOO language

Domain-agnostic pieces (reusable without changes):
- `compute_eig()`, `ModelPosterior`, `compute_log_likelihood()` from bayesian_selection.py
- `EpistemicState` from epistemic_state.py
- Phase machine logic (but not worth extracting for one use)

**Alternatives considered:** Options A and B above.

**Status:** Complete. Runner built, tested (16 tests), and validated live.

## D50: Accumulated RMSE gate for decision debate — 2026-03-21

**Problem:** First live experiment (decision M15) showed 0/3 no-debate → 1/3
debate, but the debate mechanism was unhealthy. Competitor agents were gaming
the RMSE gate by proposing revisions that improved fit on the current 2-3
gambles while hurting global fit. EU-GT debate run was perversely worse than
no-debate: PH_Agent won at 77% (vs CPT at 61% without debate) because PH
accepted 8/8 revisions that improved local RMSE but didn't represent genuine
parameter recovery. Additionally, neutral revisions (identical RMSE) were
being accepted, letting agents change params without proving improvement.

**Root cause:** Two compounding issues:
1. RMSE validation used only current cycle's 2-3 gambles, not accumulated data
2. `tolerance=0.01` with `<=` accepted equal or nearly-equal RMSE

**Decision:** Two fixes:
1. Thread accumulated observations through `run_decision_debate()` →
   `run_debate_round()`. RMSE gate now validates against ALL gambles observed
   across all cycles, preventing local overfitting.
2. Changed to strict improvement: `revised_rmse < baseline_rmse` (tolerance=0.0,
   `<` instead of `<=`).

**Results after fix:**
- Acceptance rate dropped from 79% to 49% (much more selective)
- EU debate no longer perverse (PH 77% → CPT 50%, matching no-debate)
- CPT lambda_ recovery improved: 1.2→1.9 (before) → 1.2→2.25 exact GT (after)
- PH win stronger: 58% → 75.3% posterior
- Still 1/3 debate correct (PH via 81.8% param recovery)

**Key lesson:** "Calibrate against real pipeline quantities, not toy tests"
(CLAUDE.md) applies to RMSE gates too. Validating against 2-3 gambles is the
decision-domain equivalent of unit-testing individual phases — only end-to-end
validation catches the real failure mode.

**Alternatives considered:**
- Only revise losing agents: rejected — we don't know who's GT
- Minimum RMSE improvement threshold (e.g., 5%): simpler but less principled
  than accumulated validation. Could revisit if accumulated gate proves too strict.

## D51: Decision M15 first results — debate partially replicates — 2026-03-21

**Problem:** Does the categorization M15 finding (debate recovers misspecification
via parameter diagnosis) replicate in the decision-making domain?

**Results (GPT-4o, 5 cycles, with D50 accumulated RMSE gate):**

| GT | No-debate | Debate | Recovery | Correct? |
|---|---|---|---|---|
| CPT | PH wins (wrong) | PH wins (wrong) | 51.0% | No |
| EU | CPT wins (wrong) | CPT wins (wrong) | 0.0% | No |
| PH | CPT wins (wrong) | **PH wins (correct)** | **81.8%** | Yes |

**Overall: 0/3 → 1/3.** Partial replication. Compare to categorization M15:
0/3 → 2/3 (RULEX +22pp, GCM +3.5pp).

**Diagnosis per model:**
- **PH (success):** Rule-based model with 3 discrete params. LLM diagnosed
  errors and recovered `outcome_threshold_frac` (0.3→0.1, exact GT) and
  `prob_threshold` (0.25→0.1, exact GT). `phi` partially recovered (1.5→1.0,
  GT 0.5). Parallels RULEX in categorization — rule-based models are easiest
  for LLMs to diagnose because the parameter-to-behavior mapping is intuitive.
- **CPT (partial):** 5-param model. `lambda_` recovered exactly (1.2→2.25),
  `gamma_pos/neg` moved toward GT, but `alpha/beta` (value function curvature)
  never diagnosed. LLMs can articulate "more loss aversion needed" but struggle
  with "the value function should be less concave" — the parameter-to-prediction
  mapping is too abstract.
- **EU (failure):** 2-param model but no revisions accepted on accumulated data.
  EU's misspecification (r=0.1 vs 0.5) makes its predictions too similar to
  CPT, so it never generates enough prediction error for the LLM to diagnose.

**Scientific conclusion:** Debate's parameter recovery mechanism depends on
the LLM's ability to map prediction errors to parameter changes. This works
for parameters with intuitive behavioral interpretations (loss aversion,
threshold levels) but fails for abstract mathematical parameters (value
function curvature, risk aversion). This is a domain-general principle about
the representational format of parameters, not domain content.

**Update (10-cycle run):** 0/3 → **2/3 with 10 cycles.** EU flipped — `r`
recovered exactly (0.1→0.5). PH achieved 100% recovery (all 3 params exact).
CPT still wrong (alpha/beta never diagnosed, 28.4% recovery).

This now matches categorization M15 exactly: 0/3 → 2/3.
- PH ↔ RULEX (rule-based, full/strong recovery)
- EU ↔ GCM (recovered with sufficient data)
- CPT ↔ SUSTAIN (abstract params resist diagnosis)

**Status:** Replication confirmed at 10 cycles. Two-domain result ready for
NeurIPS framing.
