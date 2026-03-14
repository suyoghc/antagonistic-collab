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
