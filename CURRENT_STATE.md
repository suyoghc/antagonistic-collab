# Current State (M15 Phase 2 complete)

## M15 — Model misspecification (Phase 2 complete)

**Scientific question:** Can LLM agents, through debate, identify that their model's
parameters are wrong and propose corrections — and does this happen better with debate
than without?

### Phase 2 results — Full 9-run matrix (GPT-4o, 5 cycles each)

| GT | Condition | Winner | Correct? | RMSE | Gap | Param Recovery |
|---|---|---|---|---|---|---|
| GCM | No-debate | Exemplar_Agent | Yes | 0.100 | 74.4% | 0% |
| GCM | Debate | Exemplar_Agent | Yes | 0.085 | 77.9% | 85.7% |
| GCM | Arbiter | Exemplar_Agent | Yes | 0.074 | 79.3% | 85.7% |
| SUSTAIN | No-debate | Clustering_Agent | Yes | 0.057 | 87.7% | 0% |
| SUSTAIN | Debate | Clustering_Agent | Yes | 0.063 | 85.8% | 0% |
| SUSTAIN | Arbiter | Clustering_Agent | Yes | 0.108 | 76.1% | 22.8% |
| RULEX | No-debate | Rule_Agent | Yes | 0.176 | 58.0% | 0% |
| RULEX | Debate | Rule_Agent | Yes | 0.077 | 80.4% | 60.3% |
| RULEX | Arbiter | Exemplar_Agent | **No** | 0.393 | 3.2% | 30.9% |

**Correct winner: 8/9.** Only arbiter-RULEX fails.

### Gap advantage over no-debate baseline

| GT | Debate | Arbiter |
|---|---|---|
| GCM | +3.5pp | +4.9pp |
| SUSTAIN | -1.9pp | -11.6pp |
| RULEX | **+22.4pp** | **-54.7pp** |

### Key findings

- **Debate without arbiter helps on GCM and RULEX**, where misspecification produces
  visible prediction errors that trigger param revision. Gap widens +3.5pp (GCM) and
  +22.4pp (RULEX). Param recovery: 85.7% (GCM), 60.3% (RULEX).
- **Debate is neutral-to-harmful on SUSTAIN.** SUSTAIN's misspecification (r=3.0,
  eta=0.15) doesn't produce enough prediction error to trigger revisions — 0% recovery
  in both debate conditions. Debate adds LLM noise without compensating param recovery.
- **Arbiter consistently degrades gap** relative to no-debate on SUSTAIN (-11.6pp) and
  RULEX (-54.7pp, wrong winner). Helps slightly on GCM (+4.9pp). Root cause on RULEX:
  meta-agents distorted divergence mapping, causing experiment selection to favor
  non-discriminative linearly-separable structures over RULEX-diagnostic
  rule-plus-exception structures.
- **Param recovery only fires when misspecification is visible.** GCM and RULEX show
  strong recovery; SUSTAIN shows none. The LLM agents need to *see* prediction failures
  before proposing revisions.

Scripts: `scripts/m15_mimicry_sweep.py`, `scripts/validation/validate_m15_live.py`

## Scientific conclusions

### The debate layer vs the arbiter layer

M14 showed debate was **epiphenomenal** under correct specification — the computational
pipeline alone identified the correct model in 18/18 conditions. M15 shows debate is
**causally necessary** under misspecification — but only the *core debate loop*, not the
full arbiter machinery.

Two distinct layers contribute differently:

1. **Debate layer** (agents interpreting results + proposing param revisions via
   `sync_params_from_theory()`): adds genuine causal value. LLM agents see prediction
   failures, diagnose the cause, and propose corrections. This is what EIG alone can't
   do — EIG selects experiments, debate estimates parameters. Pitt & Myung's (2004)
   point confirmed empirically: parameter estimation and model selection must happen
   together.

2. **Arbiter layer** (cruxes + meta-agents + claim-directed selection): net negative.
   Meta-agents (Integrator, Critic) optimize for argumentative richness, not model
   discrimination. They distort divergence mapping, degrading experiment selection.
   On RULEX this was catastrophic — all linearly-separable structures selected instead
   of diagnostic rule-plus-exception structures, producing wrong winner.

**The sweet spot is debate without arbiter** — agents + interpretation + param revision,
no meta-agents. This improves gap by +3.5pp (GCM) to +22.4pp (RULEX) via parameter
recovery of 60-86%.

### M14
The computational pipeline (Bayesian EIG + model predictions + learning curves) is
**causally sufficient** to identify the correct model on synthetic benchmarks with
fully-specified models. Debate adds interpretive value but does not improve
identification accuracy.

Established by M13 ablation: 18/18 correct across 3 debate conditions ×
3 ground truths × 2 selection strategies. No-debate achieved the best RMSE (0.055)
and gap (87.6%) while running 3-4x faster.

## M14 results (GPT-4o, 5 cycles, full debate + computation feedback loop)

| Ground Truth | Winner | RMSE | Gap | Claims | Resolved |
|---|---|---|---|---|---|
| GCM | Exemplar_Agent | 0.087 | 78% | 44 | 15 |
| SUSTAIN | Clustering_Agent | 0.061 | 88% | 36 | 15 |
| RULEX | Rule_Agent | 0.233 | 39% | 40 | 15 |

All 3 feedback interventions fire: fuzzy structure matching (94 matches),
claim-directed experiment selection (12 firings), parameter validation (7/33
rejected), claim auto-resolution (45 resolved, 39 falsified vs 6 confirmed).

## Key findings across milestones

- **Learning curves are the key discriminator.** RULEX gap: 2.4% (item-level only, M5) → 68% (with learning curves, M6). GCM approximates RULEX's final accuracy but not its sudden learning dynamics.
- **LLM-agnostic.** Correct model wins in 9/9 cross-LLM runs: GPT-4o, Claude Sonnet, Claude Opus (M4).
- **Falsification-dominated.** 39/45 claims falsified vs 6 confirmed (M14). The system converges by ruling out wrong theories, not confirming the right one.
- **Param validation is the strongest M14 intervention.** Blocked 21% of proposed revisions. Prevents RMSE degradation of 0.02–0.10 per blocked revision.
- **Debate helps under misspecification (M15).** First causal demonstration: debate widens gap +22pp on RULEX, +3.5pp on GCM via parameter recovery. But arbiter hurts — meta-agents distort experiment selection.

## What's next

See [ROADMAP.md](ROADMAP.md) — M16 (open design space), possible arbiter ablations.

## Full history

See [Notes/archive/INDEX.md](Notes/archive/INDEX.md) for development docs across 14 milestones.
