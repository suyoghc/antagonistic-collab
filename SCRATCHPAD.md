# Scratchpad

Working notes, open questions, and in-progress plans. Clean out when work is completed.

---

## Current focus — 2026-03-13

Code review complete. Fixed 10 bugs across 6 files (D9, D10). 108 tests passing.

### Where to pick up next session:
1. **Multi-model validation** — run `--true-model SUSTAIN` and `--true-model RULEX` to verify Clustering_Agent and Rule_Agent win respectively
2. **Longer debates** — run 5+ cycles with `--true-model GCM` to check if RMSE gap widens monotonically
3. **Concrete divergence display** — queued feature to show per-model predicted accuracies in divergence ranking
4. **Critique quality assessment** — check whether "my model can also predict that" pattern persists
5. **Remaining Codex review items** — P1 (Phase 5 placeholder), P2 (reject path), P3 (--demo flag)

### Key files for context:
- `FEATURES.md` — full inventory of scientifically meaningful design choices
- `LESSONS_LEARNED.md` — Phase 1–3 findings with data tables
- `runs/summary.md` — latest Phase 3 validation run results
- Latest commits: `dba1489` (Phase 3 impl), `7b0b787` (param_overrides fix), `c732ce8` (validation results), `5b03200` (FEATURES.md)

---

## Failed approaches (do not repeat)

*(none yet)*
