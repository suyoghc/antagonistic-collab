# Chat Log

Human-readable summary of each Claude Code session on this project.

---

## Session 1 — 2026-03-11 ~23:00

**Transcript:** `4b9a651e-e17f-4b50-84b7-820aacf6103e.jsonl`
**Commits:** `5e3d726`, `7d2e854`

**What we did:**
- Initial code review of the antagonistic collaboration framework
- Fixed first round of bugs: critique provenance (sprayed to all proposals), category structure KeyError fallback, off-by-one in transcript filenames, non-deterministic divergence maps, 0.0 RMSE display, format crash on non-numeric accuracy, packaging issues
- Fixed packaging layout (moved source into `antagonistic_collab/` subdirectory), replaced regex JSON parser with brace-depth parser, guarded `--cycles 0` edge case

**Key discussion:**
- Identified that the codebase had multiple crash-level bugs preventing any successful run
- Decided to fix bugs bottom-up before attempting to run debates

---

## Session 2 — 2026-03-12 ~12:00–14:30

**Transcript:** `425a9658-4c99-45f8-9fe5-358a346504a8.jsonl`
**Commits:** `e41ae99`, `f4198a5`

**What we did:**
- Fixed numpy serialization (int64 keys broke JSON)
- Rewrote tests to exercise production code paths instead of mocking internals
- Fixed 16 additional bugs across 5 modules: NaN in correlations, missing dict keys, shared-reference mutation in theory revisions, empty API responses, negative critique indices, EOFError on stdin
- Cleaned all ruff lint warnings

**Key discussion:**
- Shifted from "make it not crash" to "make data integrity correct"
- Tests expanded from basic to 57 passing

---

## Session 3 — 2026-03-12 ~16:00–16:36

**Transcript:** `ae8634fd-2da7-42fe-afc4-f201b765a573.jsonl`
**Commits:** `296112f`

**What we did:**
- Added Princeton AI Sandbox as alternative LLM backend (`--backend princeton`)
- Initial implementation used Azure OpenAI SDK
- 7 regression tests for both Anthropic and OpenAI code paths

**Key discussion:**
- Needed GPT-4o access through Princeton's compute allocation
- Decided on a `--backend` CLI flag rather than auto-detection

---

## Session 4 — 2026-03-12 ~20:00–22:28

**Transcript:** `c6ea8aa1-d5a0-4e8d-9efc-7ac7fd8f960e.jsonl`
**Commits:** `2624d9f`, `be4f44b`, `8f837d6`

**What we did:**
- Fixed Princeton backend: switched from AzureOpenAI to Portkey gateway (api.portkey.ai)
- Added Markdown reports: per-cycle `.md` transcripts + `summary.md` with leaderboard and theory trajectories
- Added auto-naming for output dirs (`runs/True_{model}_LLM_{llm}_COLLAB_{agents}_{NN}/`)
- Fixed cumulative transcript bug (each cycle's JSON was accumulating all prior messages)
- Fixed duplicate JSON in cycle markdown
- Fixed empty prediction leaderboard (agents weren't using `mean_accuracy` key)
- Ran first batch debates (4 runs, `_01` through `_04`), discovered batch-mode bias
- Planned the moderator rotation fix (D5)

**Key discussion:**
- First successful end-to-end 3-cycle debate (`_04`)
- Discovered that all 3 experiments were proposed by Exemplar_Agent — batch mode always picks `approve 0`
- Analyzed the run and designed the round-robin + critique tiebreaker strategy

---

## Session 5 — 2026-03-12 22:28 → 2026-03-13 ~07:16 (current)

**Transcript:** `dea55001-6188-40c3-bf20-d30fdae94c19.jsonl`
**Commits:** `c193d0a`

**What we did:**
- Implemented batch-mode rotation fix (TDD: 3 failing tests → implementation → all 73 pass)
- Deleted buggy runs from prior sessions
- Saved Princeton API key to `.env`
- Re-ran 3-cycle debate with rotation fix — confirmed each agent gets one experiment per cycle
- Analyzed convergence: discovered that all experiments return `mean_accuracy = 0.550` regardless of design
- Diagnosed root cause: LLM designs aren't parsed into stimuli/labels, generator always falls back to Shepard Type II with fixed params/seed
- Discussed 4 solution options (A: structure library, B: parse LLM designs, C: param variation, D: pre-computed menu)
- Created DECISIONS.md, TASKS.md, CHATLOG.md to establish project documentation

**Key discussion:**
- The debate machinery works (rotation, critique, theory revision) but produces no signal because data never varies
- Leaning toward Option A+C (structure library + condition-to-param mapping) as next fix
- This is the main blocker for meaningful convergence

---

## Session 6 — 2026-03-13

**What we did:**
- Implemented Phase 2: model-sensitive synthetic data (D7)
- Added `STRUCTURE_REGISTRY` (11 structures) and `CONDITION_EFFECTS` (5 conditions) to `debate_protocol.py`
- Rewrote `_synthetic_runner()`: looks up structure by name, applies condition overrides, uses md5-based per-experiment seeds, returns metadata
- Fixed scoring filter in `runner.py`: item_accuracies merged into actual dict for per-item scoring
- Updated experiment proposal prompt with structure/condition menus
- Updated prediction prompt with item-level guidance
- Enhanced `_divergence_context()` with ranked structure list
- 9 new TDD regression tests, 82 total passing, ruff clean
- Ran 3-cycle validation debate (Princeton/GPT-4o): data now varies (0.666, 0.473, 0.651 across cycles)
- Discovered next blocker: Clustering_Agent beats Exemplar_Agent despite GCM being ground truth — agents guess predictions instead of running models (D8)
- Logged findings to LESSONS_LEARNED.md Phase 2 (2.1–2.3)

**Key discussion:**
- Phase 2 data pipeline fix works: different structures/conditions/cycles produce different data
- But prediction scoring is confounded by LLM calibration — agents need to call `model.predict()` for predictions
- Agents also underuse divergence ranking (nobody picked 5-4 despite highest divergence)
- Next: agents call their models for predictions, then re-validate with 4-cycle run

---

## Session 7 — 2026-03-13

**Commits:** `dba1489`, `7b0b787`

**What we did:**
- Implemented Phase 3: model-computed predictions (D8)
- Added `compute_model_predictions()` to `DebateProtocol` — runs agent's model on structure with condition overrides, returns per-item P(correct label)
- Modified `run_execution()`: LLM provides reasoning+confidence, system computes predictions via `model.predict()`
- Fixed P1 bug from Codex review: `param_overrides` from LLM response were ignored and `default_params` recorded instead of actual params
- Queued 4 remaining Codex findings (P1–P3) in TASKS.md
- Ran 3-cycle validation debate (Princeton/GPT-4o, `--true-model GCM`)
- **Result: Exemplar_Agent wins with RMSE=0.0776, 3.6x lower than Rule_Agent (0.2755) and 4.5x lower than Clustering_Agent (0.3528)**
- Logged findings to LESSONS_LEARNED.md Phase 3 (3.1–3.4)
- 91 tests passing, ruff clean

**Key discussion:**
- Phase 3 fixes the core blocker from Phase 2: RMSE now measures model fit, not LLM calibration
- The correct agent wins decisively — monotonic separation across all 3 cycles
- LLM reasoning is still valuable for interpretation; the fix was separating semantics (LLM) from numerics (model)
- 5-4 structure still never selected despite highest divergence ranking
- Milestone M3 core hypothesis confirmed; ready for multi-model validation (SUSTAIN, RULEX as ground truth)

---

*This log is maintained manually. Update it at the end of each session.*
