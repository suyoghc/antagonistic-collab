# Roadmap

## Latest: M14 — Debate→Computation Feedback Loop (complete)

Closed the loop between LLM debate output and the computational scoring pipeline:
claim-directed experiment selection, validated parameter revisions, and claim
auto-resolution. All 3 interventions fire end-to-end (94 fuzzy matches, 12
claim-directed selections, 7/33 param rejections, 45 claims resolved).

**Key finding:** Debate adds no value when models are complete, data is synthetic,
and the design space is enumerable. The computational pipeline alone identifies the
correct model in all conditions (M13 ablation: 18/18 correct). Next step: test where
models are *incomplete*.

Full M14 results and earlier milestone details: [Notes/archive/TASKS.md](Notes/archive/TASKS.md)

---

## Future Milestones

### M15 — Model misspecification (proposed)
Test whether debate helps when models start with wrong parameters.
Deliberately misspecify default_params (wrong attention weights, wrong sensitivity).
The "correct" model needs LLM-proposed parameter revisions via `sync_params_from_theory()`
to recover. Compare debate vs no-debate: does param revision from debate close the gap?

### M16 — Open design space (proposed)
Remove structure registry. Force agents to propose every experiment via debate.
Only `temporary_structures` from agent proposals enter the EIG pool.
Tests whether debate generates diagnostic experiments that EIG alone can't discover.

### Conditions where debate may causally matter
- Model misspecification (models need LLM-proposed param adaptation)
- Non-enumerated design space (LLM agents propose novel structures)
- Ambiguous data (real human data with noise, individual differences)
- Explanation for humans (goal is understanding, not just identification)

See [Notes/archive/LESSONS_LEARNED.md](Notes/archive/LESSONS_LEARNED.md) for 40 theses
on LLM-mediated scientific debate.
