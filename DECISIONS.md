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
