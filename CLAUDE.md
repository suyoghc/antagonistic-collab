# CLAUDE.md

> Best practices derived from Poldrack, *Better Code, Better Science* (2024–2026)
> and Bridgeford et al., "Ten Simple Rules for AI-Assisted Coding in Science" (2025).

---

## Session Start

- At the start of every session, read PLANNING.md, TASKS.md, SCRATCHPAD.md,
  DECISIONS.md, and CHATLOG.md (if they exist) before doing any work.
- Check TASKS.md for the current milestone and pick up where the last session
  left off.
- If context usage exceeds ~50%, proactively suggest summarising progress to
  SCRATCHPAD.md, committing, and clearing context.

---

## Problem Framing

- Before writing any code, think through the problem. Articulate inputs,
  expected outputs, constraints, and edge cases.
- Ask clarifying questions when requirements are ambiguous. Do not guess at
  intent.
- Propose an implementation plan and wait for approval before proceeding on
  non-trivial tasks.

---

## Test-Driven Development

Follow Red → Green → Refactor strictly:

1. **Red**: Write failing tests that describe the desired behaviour.
2. Run the tests. **Confirm they fail.** Do not proceed until they do.
3. **Green**: Write the minimal code to make the tests pass.
4. **Refactor**: Clean up the implementation.

Rules:
- Never create mock or placeholder implementations just to make tests pass.
- Never modify tests to match broken code. Only change a test if there is a
  genuine error in the test itself.
- When a bug is found, write a regression test that catches it before fixing it.
- When generating tests, include edge cases: boundary conditions, type errors,
  malformed inputs, empty inputs, extreme values.

---

## Code Quality

- Write clean, modular code. Prefer short, single-purpose functions over long
  ones.
- Use clear, descriptive variable and function names.
- Prefer reliance on widely-used, well-documented packages. Avoid obscure
  dependencies.
- Do not put code in `__init__.py` files.
- Run linters/formatters (e.g. `ruff check` and `ruff format` for Python)
  before considering a task complete.
- When asked to improve code, make focused, incremental changes — never
  refactor broadly without explicit instruction.

---

## Solving Problems

- Fix root causes. Do not generate workarounds that avoid the actual problem.
  If you catch yourself saying "here's a quick workaround", stop and solve the
  real issue instead.
- If you are stuck or going in circles, say so. Suggest clearing context and
  restarting with a revised approach rather than continuing to thrash.
- Before restarting, note the failed approach in SCRATCHPAD.md so you do not
  repeat it.
- When stuck on a difficult problem, think step by step. Reason through what
  might be going wrong before generating more code.

---

## Version Control

- Commit after every successful, coherent set of changes.
- Write detailed, descriptive commit messages.
- Do NOT add "Co-Authored-By" lines to commit messages.
- Prefer working in feature branches so reverts are clean.
- Do not overwrite or delete user-authored content (READMEs, config, prompts)
  without explicit permission.

---

## Context & Documentation

- Keep TASKS.md updated: mark completed tasks, add newly discovered tasks.
- Use SCRATCHPAD.md to record plans, open questions, and progress notes.
  Clean it out when work is completed.
- Keep DECISIONS.md updated: when a non-trivial design choice is made, log it
  with a timestamp, the problem, the decision, alternatives considered, and
  outcome. Number entries sequentially (D1, D2, …).
- Keep CHATLOG.md updated: at the end of each session, append a summary with
  the date, what was done, what was committed, and key discussion points.
- If a `problems_tbd.md` file exists, read it and work through open items.
  Only mark a problem as fixed after the user confirms the fix.
- When generating project files (PRD.md, PLANNING.md, etc.), produce drafts
  and flag areas that need human review rather than assuming your choices are
  final.

---

## What NOT To Do

- Do not claim all tests pass when functional code has not been written.
- Do not silently change files outside the scope of the current task.
- Do not fabricate test data or placeholder functions that merely satisfy test
  structure without validating real logic.
- Do not make confident claims about performance, correctness, or compatibility
  that you cannot verify. State uncertainty explicitly.
- Do not attempt broad "improvements" unless specifically asked. Unsolicited
  refactors introduce risk.

---

## Session Checklist

```
□  Read PLANNING.md, TASKS.md, SCRATCHPAD.md, DECISIONS.md, CHATLOG.md
□  Identify current milestone
□  Write failing tests first
□  Confirm tests fail
□  Implement minimal code to pass tests
□  Review own output for correctness
□  Run linter / formatter
□  Commit with descriptive message
□  Monitor context usage; suggest summarise + clear at ~50%
□  Update TASKS.md, SCRATCHPAD.md, DECISIONS.md
□  Update CHATLOG.md with session summary before ending
```
