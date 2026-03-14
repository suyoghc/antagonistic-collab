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

### M3: Validate convergence (NEARLY COMPLETE)
- Run debates where true model's agent should win — **DONE** (GCM, SUSTAIN, RULEX all correct)
- Measure whether RMSE gap grows over cycles — **DONE** (15.1% GCM, 32.2% SUSTAIN, 1.8% RULEX at 5 cycles)
- Check whether critique quality improves or degrades — **DONE** (stable but formulaic)
- Full-pool EIG mode validated end-to-end — **DONE** (2-cycle, correct convergence, 36% fewer LLM calls)
- Learning curves wired into execution + Bayesian update — **DONE** (D23)
- Novel structures validated/registered from interpretation debate — **DONE** (D23)
- Novel structure prompting with few-shot examples — **DONE** (D24)
- 5-cycle comparative validation (full_pool vs legacy, all 3 ground truths) — **DONE** (all correct, full_pool RULEX gap 68% vs legacy 2.4%)

### M4: Multi-model ground truth
- Run with each model as ground truth (GCM, SUSTAIN, RULEX)
- Compare convergence patterns across conditions
- Write up findings

## Key constraints

- Synthetic data only (no real experiments) — but must be model-sensitive
- LLM agents via API (Anthropic or Princeton/Portkey)
- Batch mode must be fully automated (no human input)
- All results must be reproducible (deterministic seeds where possible)
