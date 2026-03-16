# Antagonistic Collaboration

**Automated adversarial scientific debate for theory advancement.**

A framework where AI agents — each committed to a competing scientific theory — debate experiment design, critique each other's proposals, and have their predictions scored against data. Bayesian experiment selection finds maximally informative experiments; LLM agents interpret results and generate hypotheses.

## Why this exists

Current approaches to automated science (e.g., [AutoRA](https://autoresearch.github.io/autora/), [AI Scientist](https://github.com/SakanaAI/AI-Scientist)) run a single agent through a research pipeline. But science advances through **adversarial discourse** — competing theories propose different experiments, critique each other's designs, and update in light of shared evidence.

This framework operationalizes that process with a **dual-layer architecture**:

- **Computational layer**: Bayesian expected information gain (EIG) selects experiments from ~427 candidates (continuous sampling from parameter ranges); executable models generate quantitative predictions; learning curves provide a second evidence channel
- **Semantic layer**: LLM agents interpret results, identify confounds, propose novel structures, and revise theories in natural language

## Current domain: Human categorization

The prototype uses category learning as a testbed because it has:

- **Deep, contested model landscape**: Exemplar models (GCM), rule-based models (RULEX), clustering models (SUSTAIN) — debated for 40+ years
- **Rich experimental design space**: 11 base structures + 50 sampled per cycle × 7 conditions ≈ 427 candidate experiments
- **Known critical experiments**: Shepard types, Medin & Schaffer's 5-4 structure — benchmarks for evaluating whether the system produces scientifically sensible proposals

## Architecture

The framework supports two operating modes:

**Full-pool mode** (`--mode full_pool`) — recommended:
```
Cycle N:
  1. Commitment (cycle 0 only)
  2. Divergence mapping
  3. Bayesian EIG selection — searches all ~427 candidates (no LLM calls)
  4. Execution — item-level scoring + learning curve comparison
  5. Interpretation debate — agents interpret results, propose hypotheses
  6. Interpretation critique — agents challenge each other
  7. Audit — convergence check
```

**Legacy mode** (`--mode legacy`) — 9-phase flow with LLM-driven experiment proposals:
```
  commit → diverge → propose → critique → revise → arbitrate → execute → interpret → audit
```

**Key design choice**: Agents can't just *assert* what their theory predicts — they must *call their model*. Predictions come from `model.predict()`, not LLM guessing. Arguments are backed by computation.

## Quickstart

```bash
# Clone and install
git clone https://github.com/suyoghc/antagonistic-collab.git
cd antagonistic-collab
pip install -r requirements.txt
```

### 1. Demo — formal layer only (no API key needed)

```bash
python -m antagonistic_collab --demo
```

Runs models, divergence mapping, and epistemic state tracker with synthetic data.

### 2. Full-pool mode with Bayesian EIG (recommended)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m antagonistic_collab --batch --cycles 5 --true-model GCM --mode full_pool --selection bayesian
```

EIG selects the most informative experiment each cycle. Agents interpret results and propose hypotheses. No human input needed.

### 3. Legacy mode — interactive with human moderator

```bash
python -m antagonistic_collab --cycles 3 --true-model SUSTAIN
```

Full 9-phase debate protocol. You act as the human moderator, approving or editing proposals at each cycle.

### CLI options

```
--cycles N              Number of debate cycles (default: 1)
--true-model MODEL      Ground truth for synthetic data: GCM, SUSTAIN, or RULEX
--batch                 Non-interactive mode (for automated runs)
--mode MODE             full_pool (EIG + interpretation debate) or legacy (9-phase)
--selection METHOD      bayesian (EIG) or heuristic (diversity penalty)
--backend BACKEND       anthropic (default) or princeton (Azure OpenAI via Portkey)
--model MODEL_ID        LLM model string (default: claude-sonnet-4-20250514)
--design-space MODE     base (55), richer (168), or continuous (sampled, default)
--n-continuous-samples  Structures sampled per cycle in continuous mode (default: 50)
--critique-rounds N     Adversarial critique rounds per cycle (default: 2)
--output-dir DIR        Where to save transcripts and state
```

## Project structure

```
antagonistic_collab/
├── __init__.py
├── __main__.py              # Entry point: python -m antagonistic_collab
├── runner.py                # LLM debate runner, phase orchestration
├── debate_protocol.py       # Phase state machine, synthetic data, agent prompts
├── epistemic_state.py       # Theory commitments, predictions, scoring
├── bayesian_selection.py    # Bayesian EIG experiment selection
├── demo.py                  # Formal layer demo (no API key)
└── models/
    ├── gcm.py               # Generalized Context Model (Nosofsky, 1986)
    ├── sustain.py            # SUSTAIN (Love, Medin & Gureckis, 2004)
    ├── rulex.py              # RULEX (Nosofsky, Palmeri & McKinley, 1994)
    └── category_structures.py  # 11 structures: Shepard I-VI, 5-4, etc.
tests/
    └── test_bugfixes.py     # 331 tests
Notes/                       # Analysis, decisions, lessons learned
```

## Key results

Validated across 43 runs (3 ground truths × multiple milestones/modes, 5 cycles each). The correct model's agent wins in 42/43 runs (one M7 RULEX misidentification due to model overlap, resolved in M8):

| Ground Truth | Mode | Winner | RMSE | Gap |
|---|---|---|---|---|
| GCM | full_pool | Exemplar_Agent | 0.161 | 34% |
| SUSTAIN | full_pool | Clustering_Agent | 0.270 | 42% |
| RULEX | full_pool | Rule_Agent | 0.119 | 68% |
| GCM | legacy | Exemplar_Agent | 0.255 | 37% |
| SUSTAIN | legacy | Clustering_Agent | 0.361 | 34% |
| RULEX | legacy | Rule_Agent | 0.429 | 2.4% |

**M6 arbiter-v0.1 live validation** (GPT-4o, role-specialized agents + crux negotiation + conflict maps + pre-registration):

| Ground Truth | Winner | RMSE | Gap | Cruxes | Claims Falsified |
|---|---|---|---|---|---|
| GCM | Exemplar_Agent | 0.151 | 36.4% | 4/34 accepted | 14 |
| SUSTAIN | Clustering_Agent | 0.270 | 45.6% | 7/32 accepted | 15 |
| RULEX | Rule_Agent | 0.119 | 67.6% | 4/35 accepted | 15 |

**Key findings**:
- **Learning curves** solved the hardest discrimination problem — RULEX gap went from 2.4% (legacy) to 68% (full_pool). GCM approximates RULEX's final accuracy but can't mimic its sudden learning dynamics.
- **LLM-agnostic**: Correct model wins in 9/9 cross-LLM runs (GPT-4o, Claude Sonnet, Claude Opus).
- **Debate now affects outcomes**: Post-M5, replication RMSE std=0.018 (was 0.000). Parameter revision persistence closes the debate→prediction feedback loop.
- **Falsification engine**: 44 claims falsified vs 1 confirmed across M6 runs. Convergence occurs by ruling out wrong theories, not confirming the right one.
- **Crux selectivity**: 15% acceptance rate with real LLMs (vs 100% in mock). Accepted cruxes map to genuine theoretical fault lines.
- **Posterior collapse**: Primary architectural bottleneck — EIG≈0 after cycle 0–1, making later cycles uninformative despite crux negotiation.
- **Winning theories need fewer revisions**: Rule_Agent made 0 revisions and won RULEX by 67.6%. Losing agents revise futilely (Lakatos-compatible).

See [Notes/REPORT.md](Notes/REPORT.md) for the full write-up and [Notes/LESSONS_LEARNED.md](Notes/LESSONS_LEARNED.md) for 35 theses on LLM-mediated scientific debate.

## Current version: arbiter-v0.1

The current debate architecture (**arbiter-v0.1**) consists of:
- **3 theory agents**: Exemplar_Agent (GCM), Rule_Agent (RULEX), Clustering_Agent (SUSTAIN)
- **2 meta-agents**: Integrator (cross-theory synthesis), Critic (weakest-argument identification)
- **Debate protocol**: commitment → divergence mapping → crux negotiation → EIG selection → execution → interpretation debate → interpretation critique → audit
- **Feedback paths**: parameter revision persistence, claim-responsive debate (revise/explain/abandon), crux-directed Thompson sampling, critique-as-falsification
- **Computational layer**: Bayesian EIG over continuous design space (~427 candidates/cycle), likelihood tempering (tau=0.005), Thompson sampling, learning curves as second evidence channel

Future versions may change the agent roster, model set, debate protocol structure, or the division of labor between computation and debate.

## What's next

- [x] ~~Close debate feedback loops~~ — parameter revision persistence, structured claim ledger, critique-as-falsification, debate-informed EIG weighting (M5, done)
- [x] ~~Compare LLM backbones~~ — Claude Sonnet/Opus vs GPT-4o: correct model wins in 9/9 runs, framework is LLM-agnostic (M4, done)
- [x] ~~arbiter-v0.1 integration~~ — role-specialized meta-agents, crux negotiation, conflict maps, pre-registration output, HITL checkpoints (M6, done — 3/3 correct, 36–68% gaps)
- [x] ~~Address posterior collapse~~ — likelihood tempering (M7), Thompson sampling (M8), crux-directed selection (M9)
- [x] ~~Claim-responsive debate~~ — agents must address falsified claims (M10, 80% FR rate)
- [x] ~~Richer design spaces~~ — parametric structures + interpolated conditions (M11, superseded by M12)
- [x] ~~Continuous design space~~ — fresh samples each cycle from parameter ranges (M12, 15/15 sampled selected, 0% overlap)
- [x] ~~Ablation study~~ — 3×2 ablation (No-Debate / Debate-No-Arbiter / Debate+Arbiter × Thompson / Greedy): 18/18 correct, debate epiphenomenal on synthetic benchmarks (M13)
- [ ] Close debate→computation feedback loop — debate output currently disconnected from scoring pipeline
- [ ] Additional cognitive domains — memory retrieval, associative learning, decision making
- [ ] AutoRA integration — real data collection via Prolific
- [ ] Longer runs (10+ cycles) — test cumulative reasoning at longer horizons

## Key references

- Nosofsky, R. M. (1986). Attention, similarity, and the identification-categorization relationship. *JEP: General*, 115(1), 39–57.
- Love, B. C., Medin, D. L., & Gureckis, T. M. (2004). SUSTAIN: A network model of category learning. *Psychological Review*, 111(2), 309–332.
- Nosofsky, R. M., Palmeri, T. J., & McKinley, S. C. (1994). Rule-plus-exception model of classification learning. *Psychological Review*, 101(1), 53–79.
- Shepard, R. N., Hovland, C. I., & Jenkins, H. M. (1961). Learning and memorization of classifications. *Psychological Monographs*, 75(13), 1–42.
- Mellers, B., Hertwig, R., & Kahneman, D. (2001). Do frequency representations eliminate conjunction effects? *Psychological Science*, 12(4), 269–275.

## License

MIT
