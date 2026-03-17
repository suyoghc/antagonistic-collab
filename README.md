# Antagonistic Collaboration

**Automated adversarial scientific debate for theory advancement.**

A framework where AI agents — each committed to a competing scientific theory — debate experiment design, critique each other's proposals, and have their predictions scored against data. Bayesian experiment selection finds maximally informative experiments; LLM agents interpret results and generate hypotheses.

## Architecture

```
Cycle N:
  1. Divergence mapping — where do models disagree?
  2. Bayesian EIG selection — search ~427 candidates for max information gain
  3. Execution — item-level scoring + learning curve comparison
  4. Interpretation debate — agents interpret results, propose hypotheses
  5. Interpretation critique — agents challenge each other
  6. Audit — convergence check
```

**Dual-layer design**: Bayesian EIG selects experiments and models generate quantitative predictions (computation layer). LLM agents interpret results, identify confounds, and revise theories (semantic layer). Agents can't just *assert* predictions — they must *call their model*.

## Quickstart

```bash
git clone https://github.com/suyoghc/antagonistic-collab.git
cd antagonistic-collab
pip install -r requirements.txt
```

**Demo** (no API key needed):
```bash
python -m antagonistic_collab --demo
```

**Full run** (recommended):
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m antagonistic_collab --batch --cycles 5 --true-model GCM --mode full_pool --selection bayesian
```

**Interactive** (human moderator):
```bash
python -m antagonistic_collab --cycles 3 --true-model SUSTAIN
```

Run `python -m antagonistic_collab --help` for all CLI options.

## Project status

- [**Current results and conclusions**](CURRENT_STATE.md) — where we are as of M14
- [**Roadmap**](ROADMAP.md) — next milestones (M15: model misspecification, M16: open design space)
- [**Development history**](Notes/archive/INDEX.md) — 14 milestones of docs, decisions, and lessons learned

## References

- Nosofsky, R. M. (1986). Attention, similarity, and the identification-categorization relationship. *JEP: General*, 115(1), 39-57.
- Love, B. C., Medin, D. L., & Gureckis, T. M. (2004). SUSTAIN: A network model of category learning. *Psychological Review*, 111(2), 309-332.
- Nosofsky, R. M., Palmeri, T. J., & McKinley, S. C. (1994). Rule-plus-exception model of classification learning. *Psychological Review*, 101(1), 53-79.
- Shepard, R. N., Hovland, C. I., & Jenkins, H. M. (1961). Learning and memorization of classifications. *Psychological Monographs*, 75(13), 1-42.

## License

MIT
