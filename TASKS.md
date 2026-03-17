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
- [ ] Re-run live validation with normalization enabled (claim-directed selection should now fire)

### Completed (validation)
- [x] Smoke test: 2-cycle GCM, all 3 interventions fire (PASS)
- [x] Live validation: 3/3 correct (GCM 0.076/81%, SUSTAIN 0.063/88%, RULEX 0.226/41%)
- [x] Ablation (crux_weight=0): 3/3 correct (GCM 0.087/77%, SUSTAIN 0.125/68%, RULEX 0.053/87%)
- [x] Diagnosis: claim-directed selection never fired (LLM free-text didn't match registry keys)
- [x] Fix: normalize_claim_fields() fuzzy-matches LLM output to registry keys (13 tests)
- [x] Param validation gate blocked 6/33 bad revisions (18%) — only intervention with clear causal impact pre-normalization

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
