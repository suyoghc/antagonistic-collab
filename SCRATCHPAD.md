# Scratchpad

Working notes, open questions, and in-progress plans. Clean out when work is completed.

---

## Current focus — 2026-03-13

Main blocker: synthetic data generator returns identical results for every experiment (D6 in DECISIONS.md).

### Plan for fixing P1+P2 (structure library + param variation)

**Idea:** Instead of letting LLMs invent freeform experiment specs, constrain them to pick from the existing category structure library. Map experimental conditions (e.g., "cognitive load", "presentation order") to model parameter perturbations.

**Open questions:**
- How should the structure menu be presented to agents? As a list in the prompt? As part of the divergence map output?
- Should agents be allowed to specify parameter ranges, or should the system map conditions to params automatically?
- How many distinct data patterns can the current models produce across the existing structures? Need to verify there's actually enough variance.
- Should the random seed vary per experiment, or stay fixed for reproducibility?

**Not yet started.** Awaiting approval to proceed.

---

## Failed approaches (do not repeat)

*(none yet)*
