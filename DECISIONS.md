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

**Result:** 9 new tests, 82 total passing. Different structures/conditions/cycles now produce genuinely different data.
