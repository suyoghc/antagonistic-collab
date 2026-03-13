# Tasks

## Current Milestone: M2 — Make data meaningful

### Blocking
- [x] **Fix synthetic data generator** — experiments now produce different data depending on category structure and conditions (D6 → D7)
- [x] **Constrain agent proposals to structure library** — agents pick from STRUCTURE_REGISTRY menu (11 structures)

### High Priority
- [x] **Expand scoring beyond mean_accuracy** — item_accuracies now merged into scoring dict, per-item predictions scored
- [x] **Vary model params by experimental conditions** — CONDITION_EFFECTS maps 5 conditions to model param overrides

### Up Next (M3)
- [ ] Validate convergence: run 3+ cycle debates where the true model's agent should accumulate best RMSE over cycles
- [ ] Assess whether critique quality degrades over cycles (circular "my model can also predict that" pattern)
- [ ] Run with each model as ground truth (GCM, SUSTAIN, RULEX) and compare

### Done
- [x] Fix P1 crashes and data-integrity bugs — 2026-03-11 → 2026-03-12
- [x] Add Princeton/Portkey backend — 2026-03-12 16:40
- [x] Add Markdown reports + auto-naming — 2026-03-12 21:30
- [x] Fix duplicate JSON + empty leaderboard — 2026-03-12 21:41
- [x] Fix batch-mode moderator bias (always picking first proposal) — 2026-03-12 22:39
- [x] Delete buggy run outputs, re-run with rotation fix — confirmed rotation works — 2026-03-13
- [x] Create DECISIONS.md, CHATLOG.md, TASKS.md — 2026-03-13
- [x] Create SCRATCHPAD.md, PLANNING.md — 2026-03-13
