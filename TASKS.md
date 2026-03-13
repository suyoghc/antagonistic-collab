# Tasks

## Current Milestone: Make debates produce meaningful convergence

### Blocking
- [ ] **Fix synthetic data generator** — experiments must produce different data depending on category structure and conditions (see DECISIONS.md D6)

### Up Next
- [ ] Validate convergence: run 3+ cycle debates where the true model's agent should accumulate best RMSE over cycles
- [ ] Consider whether critique quality degrades over cycles (agents repeat the same "my model can also predict that" pattern)

### Done
- [x] Fix P1 crashes and data-integrity bugs — 2026-03-11 → 2026-03-12
- [x] Add Princeton/Portkey backend — 2026-03-12 16:40
- [x] Add Markdown reports + auto-naming — 2026-03-12 21:30
- [x] Fix duplicate JSON + empty leaderboard — 2026-03-12 21:41
- [x] Fix batch-mode moderator bias (always picking first proposal) — 2026-03-12 22:39
- [x] Delete buggy run outputs, re-run with rotation fix — confirmed rotation works — 2026-03-13
