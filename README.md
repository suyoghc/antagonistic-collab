# Antagonistic Collaboration

**Automated adversarial scientific debate for theory advancement.**

A framework where AI agents — each committed to a competing scientific theory — debate experiment design, critique each other's proposals, and have their predictions scored against data. A human moderator arbitrates.

> *"A simulation of the way we do science: a theory-driven debate between agents, with pairwise comparisons and rejection of models that fail."*

## Why this exists

Current approaches to automated science (e.g., [AutoRA](https://autoresearch.github.io/autora/), [AI Scientist](https://github.com/SakanaAI/AI-Scientist)) run a single agent through a research pipeline. But science advances through **adversarial discourse** — competing theories propose different experiments, critique each other's designs, and update in light of shared evidence.

This framework operationalizes that process with a **dual-layer architecture**:

- **Semantic layer**: LLM agents argue in natural language — interpreting results, identifying confounds, proposing theory modifications
- **Formal layer**: Executable computational models generate quantitative predictions — ensuring arguments are grounded in actual model behavior, not LLM confabulation

## Current domain: Human categorization

The prototype uses category learning as a testbed because it has:

- **Deep, contested model landscape**: Exemplar models (GCM), rule-based models (RULEX), clustering models (SUSTAIN), and hybrids — debated for 40+ years
- **Rich experimental design space**: Stimulus dimensionality, category structure, training regime, transfer tests, dependent measures
- **Known critical experiments**: Shepard types, Medin & Schaffer's 5-4 structure, COVIS dissociations — benchmarks for evaluating whether the system produces scientifically sensible proposals

## Architecture

```
┌───────────────────────────────────────────┐
│           DEBATE PROTOCOL                 │
│  9 phases per cycle:                      │
│  commit → diverge → propose → critique    │
│  → revise → arbitrate → execute →         │
│  interpret → audit                        │
├───────────────────────────────────────────┤
│        EPISTEMIC STATE TRACKER            │
│  Theories, predictions, disputes,         │
│  established facts, leaderboard           │
├───────────────────────────────────────────┤
│           FORMAL LAYER                    │
│  GCM · SUSTAIN · RULEX                    │
│  Category structures · Divergence maps    │
└───────────────────────────────────────────┘
```

**Key design choice**: Agents can't just *assert* what their theory predicts — they must *call their model*. When the Exemplar agent claims GCM predicts a smooth generalization gradient, it runs `gcm.predict_generalization_gradient()`. When the Clustering agent critiques, it runs SUSTAIN on the same conditions and shows a different pattern. Arguments are backed by computation.

## Quickstart

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/antagonistic-collab.git
cd antagonistic-collab
pip install -r requirements.txt
```

### 1. Demo — formal layer only (no API key needed)

```bash
python -m antagonistic_collab --demo
```

This runs the models, divergence mapping, and epistemic state tracker with synthetic data. Use this to verify the formal layer works and to see model predictions on Shepard types, 5-4, etc.

### 2. Live debate — interactive with human moderator

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m antagonistic_collab
```

This runs the full 9-phase debate protocol. Three agents (Exemplar, Rule, Clustering) argue about experiment design. You act as the human moderator, approving or editing proposals at each cycle. Full transcripts and epistemic state are saved to JSON after each cycle.

### 3. Batch mode — no human input needed

```bash
python -m antagonistic_collab --batch --cycles 3 --true-model SUSTAIN
```

Auto-approves the first proposal at each cycle. Use this for systematic comparisons (e.g., running 10 debates with different ground-truth models) or on compute clusters.

### Full CLI options

```
--cycles N              Number of debate cycles (default: 1)
--true-model MODEL      Ground truth for synthetic data: GCM, SUSTAIN, or RULEX
--batch                 Non-interactive mode (auto-approve proposals)
--model MODEL_ID        LLM model string (default: claude-sonnet-4-20250514)
--critique-rounds N     Adversarial critique rounds per cycle (default: 2)
--output-dir DIR        Where to save transcripts and state
```

### Running on a cluster (e.g., Della)

The LLM debate requires outbound HTTPS access to `api.anthropic.com`, which most HPC clusters block. Two options:

1. **If your cluster allows outbound HTTPS** (or you have a proxy): Run in batch mode directly.
   ```bash
   sbatch --wrap="python -m antagonistic_collab --batch --cycles 5 --true-model GCM"
   ```

2. **If not**: Split the workflow. Run the debate on a local machine, then run model fitting / evaluation sweeps on the cluster. The `models/` module and `epistemic_state.py` have no API dependency and run anywhere with numpy/scipy.

## Project structure

```
antagonistic_collab/
├── __init__.py
├── __main__.py              # Entry point: python -m antagonistic_collab
├── runner.py                # LLM debate runner (needs API key)
├── debate_protocol.py       # 9-phase state machine + agent system prompts
├── epistemic_state.py       # Cumulative knowledge tracker
├── demo.py                  # Formal layer demo (no API key)
└── models/
    ├── __init__.py
    ├── gcm.py               # Generalized Context Model (Nosofsky, 1986)
    ├── sustain.py            # SUSTAIN (Love, Medin & Gureckis, 2004)
    ├── rulex.py              # RULEX (Nosofsky, Palmeri & McKinley, 1994)
    └── category_structures.py  # Shepard types, 5-4, generators
```

## What's working now

- [x] GCM, SUSTAIN, and RULEX as callable models with `predict()`, `predict_learning_curve()`, and `fit()` interfaces
- [x] Standard category structures (Shepard 6 types, 5-4, rule-plus-exception, linear separable)
- [x] Automatic divergence mapping across models × structures
- [x] Epistemic state tracker with theory registration, prediction registry, dispute tracking, and cumulative leaderboard
- [x] Term glossary on each theory — maps natural language terms to specific model parameters (eliminates terminology confusion)
- [x] ModelClaim dataclass — structured, verifiable claims about model predictions (eliminates straw-manning)
- [x] Critique → revision provenance chain — every proposal revision must link back to the critique(s) it addresses
- [x] Progressive vs. degenerative theory revision tracking (Lakatos) with `theory_trajectory()` computation
- [x] 9-phase debate protocol with phase specs, context generators, and transition logic
- [x] Synthetic experiment runner (ground-truth model generates noisy data)
- [x] Agent system prompts for the categorization domain
- [x] **LLM runner**: Interactive debate via Claude API — raw calls, no framework, human moderator in terminal
- [x] Transcript and epistemic state saved to JSON after each cycle

## What's next

- [ ] **Run the first debate** and iterate on system prompts based on transcript quality
- [ ] **Three-condition comparison**: single-agent vs. multi-agent (no formal layer) vs. multi-agent (with formal layer)
- [ ] **Tool use**: Expose models as callable tools so agents invoke them mid-debate (not just pre-computed divergence maps)
- [ ] **Convergence monitor**: Detect premature agreement in audit phase, inject disruptions
- [ ] **AutoRA integration**: Plug in AutoRA's experiment runners for real data collection via Prolific
- [ ] **Evaluation harness**: Blind comparison rated by domain experts
- [ ] **Additional domains**: Cross-situational word learning, working memory, recognition memory

## Key references

- Nosofsky, R. M. (1986). Attention, similarity, and the identification-categorization relationship. *JEP: General*, 115(1), 39–57.
- Love, B. C., Medin, D. L., & Gureckis, T. M. (2004). SUSTAIN: A network model of category learning. *Psychological Review*, 111(2), 309–332.
- Nosofsky, R. M., Palmeri, T. J., & McKinley, S. C. (1994). Rule-plus-exception model of classification learning. *Psychological Review*, 101(1), 53–79.
- Shepard, R. N., Hovland, C. I., & Jenkins, H. M. (1961). Learning and memorization of classifications. *Psychological Monographs*, 75(13), 1–42.
- Musslick, S., et al. AutoRA: Automated Research Assistant. https://autoresearch.github.io/autora/

## License

MIT
