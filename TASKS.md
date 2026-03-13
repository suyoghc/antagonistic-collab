# Tasks

## Current Milestone: M2 — Make data meaningful

### Blocking
- [ ] **Fix synthetic data generator** — experiments must produce different data depending on category structure and conditions (see DECISIONS.md D6)
- [ ] **Constrain agent proposals to structure library** — agents pick from Shepard I-VI, 5-4, rule+exception, linear-separable instead of inventing freeform specs

### High Priority
- [ ] **Expand scoring beyond mean_accuracy** — score on item-level accuracy patterns where models actually diverge
- [ ] **Vary model params by experimental conditions** — map conditions like "cognitive load" to parameter perturbations

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
