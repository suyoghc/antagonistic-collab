# Planning

High-level project roadmap and architectural notes.

---

## Project goal

Build an antagonistic collaboration framework where LLM agents representing competing cognitive theories (GCM, RULEX, SUSTAIN) debate through a structured protocol, propose experiments, and converge toward the theory that best explains the data — mimicking the scientific method.

## Architecture

```
runner.py              — CLI entry point, LLM calls, phase orchestration
debate_protocol.py     — 9-phase protocol logic, synthetic data generation
epistemic_state.py     — Theory commitments, experiments, predictions, scoring
bayesian_selection.py  — Bayesian EIG experiment selection (D18)
models/
  gcm.py               — Generalized Context Model (Nosofsky 1986)
  sustain.py           — SUSTAIN clustering model (Love, Medin & Gureckis 2004)
  rulex.py             — RULEX rule-based model (Nosofsky, Palmeri & McKinley 1994)
  category_structures.py — Shepard types, 5-4 structure, rule+exception, etc.
```

## Phase flow (per cycle)

1. Commitment — agents declare theoretical commitments
2. Divergence mapping — identify where models disagree most
3. Experiment proposal — each agent proposes a discriminating experiment
4. Adversarial critique — agents critique each other's proposals (2 rounds)
5. Design revision — agents revise proposals based on critiques
6. Human arbitration — moderator selects experiment (batch: EIG or heuristic)
7. Execution — synthetic data generated, predictions scored
8. Interpretation — agents interpret results, propose theory revisions
9. Audit — impartial auditor summarizes cycle, checks convergence

### Full-pool mode (`--mode full_pool`)
1. Commitment (cycle 0 only)
2. Divergence mapping
3. Full-pool Bayesian selection — EIG over all 55+ candidates (no LLM calls)
4. Execution + learning curve comparison
5. Interpretation debate — structured JSON: interpretation, confounds, hypotheses, novel structures
6. Interpretation critique — agents challenge each other
7. Audit

## Milestones

### M1: Make it run (DONE)
- Fix crashes, packaging, serialization
- Add Princeton backend
- Add reporting (markdown + summary)

### M2: Make data meaningful (DONE)
- Fix synthetic data generator to produce experiment-sensitive results
- Constrain agent proposals to executable structures
- Expand scoring beyond mean_accuracy
- Agents call their models for predictions (not LLM-guessed)

### M3: Validate convergence (DONE)
- Run debates where true model's agent should win — **DONE** (GCM, SUSTAIN, RULEX all correct)
- Measure whether RMSE gap grows over cycles — **DONE** (15.1% GCM, 32.2% SUSTAIN, 1.8% RULEX at 5 cycles with legacy)
- Check whether critique quality improves or degrades — **DONE** (stable but formulaic)
- Full-pool EIG mode validated end-to-end — **DONE** (2-cycle, correct convergence, 36% fewer LLM calls)
- Learning curves wired into execution + Bayesian update — **DONE** (D23)
- Novel structures validated/registered from interpretation debate — **DONE** (D23)
- Novel structure prompting with few-shot examples — **DONE** (D24)
- 5-cycle comparative validation (full_pool vs legacy, all 3 ground truths) — **DONE** (all correct, full_pool RULEX gap 68% vs legacy 2.4%)

### M4: Analysis & write-up (DONE)
- [x] EIG selection patterns — five_four/fast universal cycle-0 pick; RULEX shifts to Type_I (LESSONS 9.1)
- [x] Novel structures — 21 proposed, none selected by EIG (LESSONS 9.3)
- [x] Posterior convergence — GCM/SUSTAIN immediate, RULEX 2-cycle lag (LESSONS 9.2)
- [x] Debate quality audit — data citation weak, critiques mixed, no cumulative learning (LESSONS 9.6)
- [x] Replication — zero variance; pipeline fully deterministic (LESSONS 9.7)
- [x] Write-up — REPORT.md: intro, methods, results (3.1–3.9), discussion, conclusion
- [x] Theory revision — Lakatos-compatible (LESSONS 9.5)
- [x] Legacy vs full_pool comparison — debate hurts experiment selection (LESSONS 9.4)

### M5: Close debate feedback loops (DONE)
- Parameter revision persistence, critique-as-falsification, debate-informed EIG, claim ledger
- Replication variance went from 0.000 to std=0.018

### M6: ARBITER integration (DONE)
- MetaAgentConfig, crux negotiation, conflict map, pre-registration, HITL checkpoints
- Live validation: 3/3 correct (GCM 36.4%, SUSTAIN 45.6%, RULEX 67.6%)
- Identified posterior collapse as primary bottleneck (D29)

### M7: Likelihood tempering (DONE)
- [x] Add `learning_rate` (tau) param to `ModelPosterior.update()`, `compute_eig()`, `select_from_pool()`, `select_experiment()`, `update_posterior_from_experiment()`
- [x] Wire `_LEARNING_RATE` global through `runner.py` call sites
- [x] Add `--learning-rate` CLI flag, `--no-tempering`, `--no-arbiter` toggles
- [x] YAML config file with layered precedence (built-in → user config → CLI)
- [x] 5 Codex review bug fixes: ground-truth leakage, novel structures, LOO mismatch, RULEX curves, n_subjects threading
- [x] Calibrated tau: 0.2 → 0.005 + prediction clip [0.01, 0.99] → [0.05, 0.95] (D32)
- [x] Live validation: entropy=0.635 after cycle 0 (was 0.000), EIG=0.233 on cycle 1 (was 0.000)
- [x] 308 tests passing

### M8: Thompson sampling + Codex fixes round 6 (DONE)
- [x] Configurable `selection_strategy`: thompson (default) or greedy (D34)
- [x] Thompson sampling literature review: Section 4.5 in WRITEUP.md (10 references)
- [x] 3 Codex fixes (D35): curve bonus removed, novel structure execution, divergence map consistency
- [x] Clean ablation: Thompson vs greedy, 3 ground truths × 2 strategies = 6 runs
- [x] Results: both 3/3 correct post-bugfix; Thompson 12 unique structures (6 novel) vs greedy 3 unique (0 novel)
- [x] 322 tests passing

### M9: Crux-directed Thompson sampling (DONE)
- [x] Mixture distribution in `_select_index` — crux_weight probability of crux-directed selection
- [x] Fix crux identification prompt — show structure/condition menu with format example
- [x] Fix `cruxes_to_boost_specs` — validate against known structures, strip whitespace
- [x] Config + CLI: `crux_weight: 0.3`, `--crux-weight` flag
- [x] Crux-directed logging in transcript (crux_directed, crux_id fields)
- [x] 3 Codex fixes (D37b): hardcoded credential, mock crux matching, batch mode leak
- [x] 336 tests passing
- [x] Live validation: 3/3 correct, 24 parseable crux specs (was 0), 1 crux-directed experiment

### M10: Claim-responsive debate (DONE — code; pending live validation)
- [x] Config: `no_claim_responsive: false` (default on), CLI: `--no-claim-responsive`
- [x] Interpretation prompt injects `FALSIFIED CLAIMS` directive when agent has falsified claims
- [x] Agents must respond with revise/explain/abandon per claim via `"falsified_response"` JSON field
- [x] 7 tests (TestClaimResponsiveDebate), 343 total passing
- [ ] Live validation: compare debate quality with/without claim-responsive across 3 ground truths
- Literature: Shinn et al. (2023) Reflexion, AGM belief revision (Alchourrón et al. 1985)

## Key constraints

- Synthetic data only (no real experiments) — but must be model-sensitive
- LLM agents via API (Anthropic or Princeton/Portkey)
- Batch mode must be fully automated (no human input)
- All results must be reproducible (deterministic seeds where possible)
