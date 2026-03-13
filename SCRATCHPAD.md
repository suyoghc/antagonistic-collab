# Scratchpad

Working notes, open questions, and in-progress plans. Clean out when work is completed.

---

## Current focus — 2026-03-13

Phase 3 validated. Exemplar_Agent wins with RMSE=0.0776 when GCM is ground truth (3.6x gap over nearest competitor). M3 core hypothesis confirmed.

### Where to pick up next session:
1. **Multi-model validation** — run `--true-model SUSTAIN` and `--true-model RULEX` to verify Clustering_Agent and Rule_Agent win respectively
2. **Longer debates** — run 5+ cycles with `--true-model GCM` to check if RMSE gap widens monotonically
3. **Concrete divergence display** — queued feature to show per-model predicted accuracies in divergence ranking (agents ignore abstract scores)
4. **Critique quality assessment** — check whether "my model can also predict that" pattern persists or improves with model-computed predictions
5. **Codex review fixes** — 4 queued items (P1–P3), none blocking M3 validation

### Key files for context:
- `FEATURES.md` — full inventory of scientifically meaningful design choices
- `LESSONS_LEARNED.md` — Phase 1–3 findings with data tables
- `runs/summary.md` — latest Phase 3 validation run results
- Latest commits: `dba1489` (Phase 3 impl), `7b0b787` (param_overrides fix), `c732ce8` (validation results), `5b03200` (FEATURES.md)

---

## Failed approaches (do not repeat)

*(none yet)*
