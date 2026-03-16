# TASKS

## Current Milestone: M14 — Close Debate→Computation Feedback Loop

### Completed
- [x] Claim-directed experiment selection (`claims_to_boost_specs()` + merge in `run_full_pool_selection()`)
- [x] Validated parameter revisions (`validate_param_revision()` + gate in `sync_params_from_theory()`)
- [x] Claim auto-resolution (`resolve_claims_from_data()` + wire into `run_cycle()`)
- [x] 15 new tests across 3 test classes (all passing)
- [x] Lint/format clean (`ruff check` + `ruff format`)
- [x] Full test suite: 403/403 passing

### Open
- [ ] Live validation: 3 ground truths × 5 cycles, compare RMSE/gap against M12 baseline
- [ ] Ablation: compare with/without claim-directed selection (set crux_weight=0)
- [ ] Smoke test: 2-cycle debate with GCM ground truth, verify claim specs appear in EIG output

---

## Previous Milestones

### M13 — Debate Ablation Study (complete)
- [x] Experiment framework (YAML configs, condition runner, comparison table)
- [x] No-debate mode (_NO_DEBATE global, computational-only pipeline)
- [x] 3×2 ablation: No-Debate / Debate-No-Arbiter / Debate+Arbiter × Thompson / Greedy
- [x] Ablation results: debate is epiphenomenal on synthetic benchmarks (18/18 correct, no-debate best RMSE)

---

## Future Tasks

### Conditions where debate may causally matter
See `memory/project_debate_ablation_future.md` for scenarios where the now-closed
feedback loop should produce measurable improvements: model misspecification,
non-enumerated design space, ambiguous data, explanation for humans.
