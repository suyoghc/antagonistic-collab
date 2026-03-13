# Planning

High-level project roadmap and architectural notes.

---

## Project goal

Build an antagonistic collaboration framework where LLM agents representing competing cognitive theories (GCM, RULEX, SUSTAIN) debate through a structured protocol, propose experiments, and converge toward the theory that best explains the data — mimicking the scientific method.

## Architecture

```
runner.py          — CLI entry point, LLM calls, phase orchestration
debate_protocol.py — 9-phase protocol logic, synthetic data generation
epistemic_state.py — Theory commitments, experiments, predictions, scoring
models/
  gcm.py           — Generalized Context Model (Nosofsky 1986)
  sustain.py       — SUSTAIN clustering model (Love, Medin & Gureckis 2004)
  rulex.py         — RULEX rule-based model (Nosofsky, Palmeri & McKinley 1994)
  category_structures.py — Shepard types, 5-4 structure, rule+exception, etc.
```

## Phase flow (per cycle)

1. Commitment — agents declare theoretical commitments
2. Divergence mapping — identify where models disagree most
3. Experiment proposal — each agent proposes a discriminating experiment
4. Adversarial critique — agents critique each other's proposals (2 rounds)
5. Human arbitration — moderator selects experiment (batch: round-robin)
6. Execution — synthetic data generated, predictions scored
7. Interpretation — agents interpret results, propose theory revisions
8. Audit — impartial auditor summarizes cycle, checks convergence

## Milestones

### M1: Make it run (DONE)
- Fix crashes, packaging, serialization
- Add Princeton backend
- Add reporting (markdown + summary)

### M2: Make data meaningful (CURRENT)
- Fix synthetic data generator to produce experiment-sensitive results
- Constrain agent proposals to executable structures
- Expand scoring beyond mean_accuracy

### M3: Validate convergence
- Run debates where true model's agent should win
- Measure whether RMSE gap grows over cycles
- Check whether critique quality improves or degrades

### M4: Multi-model ground truth
- Run with each model as ground truth (GCM, SUSTAIN, RULEX)
- Compare convergence patterns across conditions
- Write up findings

## Key constraints

- Synthetic data only (no real experiments) — but must be model-sensitive
- LLM agents via API (Anthropic or Princeton/Portkey)
- Batch mode must be fully automated (no human input)
- All results must be reproducible (deterministic seeds where possible)
