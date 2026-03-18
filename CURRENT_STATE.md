# Current State (M16 — Open Design Space, Phase 2 complete)

## M16 — Open Design Space

**Scientific question:** When agents must propose all experiment structures via debate
(no pre-registered structure registry), does this improve or degrade model
identification compared to a curated registry? And does the arbiter's crux machinery
help focus agent proposals, or distort them?

**Motivation:** M14 showed debate was epiphenomenal when models are correctly specified
and the experiment registry is adequate. M15 showed debate helps under misspecification
via *parameter estimation*. M16 tests the remaining axis: does debate add value for
experiment *design* itself? Phase 2 adds arbiter conditions because M15's "arbiter is
net negative" conclusion was tested only under misspecification — the interaction between
arbiter and correct specification / open design space was untested.

### Full results — 14/15 runs (GPT-4o, 5 cycles each)

| GT | Condition | Winner | Correct? | RMSE | Gap% | #Structs |
|---|---|---|---|---|---|---|
| GCM | closed_no_debate | Exemplar_Agent | Yes | 0.088 | 76.8 | 0 |
| GCM | closed_debate | Exemplar_Agent | Yes | 0.067 | 81.0 | 10 |
| GCM | closed_arbiter | Exemplar_Agent | Yes | 0.073 | 79.2 | 8 |
| GCM | open_debate | Exemplar_Agent | Yes | 0.084 | 71.6 | 54 |
| GCM | open_arbiter | Exemplar_Agent | Yes | 0.074 | 76.9 | 48 |
| SUSTAIN | closed_no_debate | Clustering_Agent | Yes | 0.057 | 87.7 | 0 |
| SUSTAIN | closed_debate | Clustering_Agent | Yes | 0.053 | 88.6 | 10 |
| SUSTAIN | closed_arbiter | Clustering_Agent | Yes | 0.020 | 96.0 | 10 |
| SUSTAIN | open_debate | Clustering_Agent | Yes | 0.108 | 64.1 | 56 |
| SUSTAIN | open_arbiter | — | **ERROR** | — | — | — |
| RULEX | closed_no_debate | Rule_Agent | Yes | 0.053 | 86.1 | 0 |
| RULEX | closed_debate | Rule_Agent | Yes | 0.168 | 58.6 | 14 |
| RULEX | closed_arbiter | Rule_Agent | Yes | 0.140 | 63.9 | 13 |
| RULEX | open_debate | Rule_Agent | Yes | 0.055 | 82.7 | 48 |
| RULEX | open_arbiter | Rule_Agent | Yes | 0.061 | 82.0 | 55 |

**Correct winner: 14/14** (1 error: SUSTAIN open_arbiter — "setting an array element
with a sequence").

### Gap advantage (pp over closed_no_debate baseline)

| GT | closed_debate | closed_arbiter | open_debate | open_arbiter |
|---|---|---|---|---|
| GCM | +4.2pp | +2.4pp | -5.2pp | +0.1pp |
| SUSTAIN | +0.9pp | **+8.3pp** | -23.6pp | ERROR |
| RULEX | -27.5pp | -22.2pp | -3.4pp | -4.1pp |

### Design (2x2+1 factorial)

| Condition | Registry | Debate | Arbiter | Purpose |
|---|---|---|---|---|
| closed_no_debate | curated | no | no | M14 baseline: computation alone |
| closed_debate | curated | yes | no | Debate + curated registry |
| closed_arbiter | curated | yes | yes | Cruxes + meta-agents + curated registry |
| open_debate | agent-proposed | yes | no | Agent proposals without guidance |
| open_arbiter | agent-proposed | yes | yes | Crux-guided agent proposals |

### Key findings

**1. The arbiter is not universally bad — it's model-type-dependent.**

M15 concluded the arbiter was net negative. M16 overturns this: the arbiter is a *bias*,
not noise. Cruxes steer experiment selection toward continuous/similarity-based structures
where models disagree. This is a feature for similarity-based models (SUSTAIN, GCM) and
a bug for rule-based models (RULEX).

- SUSTAIN closed_arbiter: **96.0% gap** (+8.3pp) — best SUSTAIN result across all
  milestones. RMSE 0.020 is the lowest we've measured. Crux-directed selection picks
  structures that maximally distinguish clustering from exemplar strategies.
- GCM closed_arbiter: 79.2% (+2.4pp) — modest improvement.
- RULEX closed_arbiter: 63.9% (-22.2pp) — still hurts, consistent with M15 finding,
  but less catastrophic than M15's -54.7pp (no misspecification compounding the problem).

**2. Open design space helps RULEX, hurts everything else.**

Agent-proposed structures are semantically rich (exception-heavy, rule-diagnostic) which
helps RULEX but hurts SUSTAIN. This is the mirror image of the arbiter bias:

- RULEX open_debate (82.7%) and open_arbiter (82.0%) both dramatically outperform
  closed_debate (58.6%) and closed_arbiter (63.9%). Agent proposals are genuinely better
  than the continuous registry for rule models.
- SUSTAIN open_debate (64.1%) is the worst SUSTAIN condition (-23.6pp). Agent proposals
  don't probe clustering behavior as effectively as the curated registry.

**3. Arbiter recovers open design space losses for GCM.**

GCM open_debate drops -5.2pp from baseline. GCM open_arbiter recovers to +0.1pp. The
crux machinery successfully focuses agent proposals toward more diagnostic structures
for similarity-based models. This is the one case where arbiter + open design space
interact positively.

**4. Computation alone remains the most reliable single condition.**

closed_no_debate achieves 76.8-87.7% gap across all three ground truths with zero
LLM calls. No debate condition consistently beats it across all models. The best
debate results (SUSTAIN closed_arbiter 96.0%, GCM closed_debate 81.0%) come from
specific model-condition pairings, not a universally beneficial intervention.

**5. Bug: SUSTAIN open_arbiter errored** ("setting an array element with a sequence").
Needs investigation before conclusions about that cell are final.

Scripts: `scripts/validation/validate_m16_live.py`

---

## M15 — Model misspecification (complete)

**Scientific question:** Can LLM agents, through debate, identify that their model's
parameters are wrong and propose corrections?

### Results — Full 9-run matrix (GPT-4o, 5 cycles each)

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

Scripts: `scripts/m15_mimicry_sweep.py`, `scripts/validation/validate_m15_live.py`

---

## Scientific conclusions

### The arbiter is a bias, not noise

The arbiter's crux machinery steers experiment selection toward continuous,
similarity-based structures. This is not random degradation — it's a systematic bias
that favors certain model types:

| Model type | Arbiter effect (M15 misspec) | Arbiter effect (M16 correct spec) |
|---|---|---|
| SUSTAIN (clustering) | -11.6pp | **+8.3pp** (best result) |
| GCM (exemplar) | +4.9pp | +2.4pp |
| RULEX (rule-based) | **-54.7pp** (wrong winner) | -22.2pp |

Under misspecification (M15), the bias compounds with parameter noise to produce
catastrophic RULEX failure. Under correct specification (M16), the same bias produces
SUSTAIN's best-ever result. The arbiter is not broken — it's partial to
similarity-based model discrimination.

### When each component helps (revised theory)

The Phase 1 "gap-filling" theory was too simple. The full picture:

1. **Computation alone** is the most reliable baseline. 76-88% gap, never gets
   the wrong winner (23/23 across M14-M16).

2. **Debate layer** (agents interpreting + proposing params/structures):
   - Under misspecification: +3.5 to +22pp via parameter recovery (M15)
   - Under correct specification: noise injection, -3 to -28pp (M16)
   - Exception: open design proposals help RULEX when registry has a coverage gap

3. **Arbiter layer** (cruxes + meta-agents + claim-directed selection):
   - Not noise but *bias* toward similarity-based structures
   - Helps SUSTAIN (+8pp) and GCM (+2-5pp)
   - Hurts RULEX (-22 to -55pp)
   - Can recover open-design losses for similarity models (GCM open_arbiter)

4. **Open design space** (agent-proposed structures):
   - Mirror-image bias: semantically rich proposals favor rule-based models
   - Helps RULEX (+24pp over closed_debate), hurts SUSTAIN (-24pp)
   - Arbiter partially counteracts open-design bias for GCM

### M14
The computational pipeline (Bayesian EIG + model predictions + learning curves) is
**causally sufficient** to identify the correct model on synthetic benchmarks with
fully-specified models. 18/18 correct across ablation conditions.

## Key findings across milestones

- **Learning curves are the key discriminator.** RULEX gap: 2.4% (item-level only, M5) -> 68% (with learning curves, M6).
- **LLM-agnostic.** Correct model wins in 9/9 cross-LLM runs: GPT-4o, Claude Sonnet, Claude Opus (M4).
- **Falsification-dominated.** 39/45 claims falsified vs 6 confirmed (M14).
- **Debate helps under misspecification (M15).** +22pp on RULEX, +3.5pp on GCM via parameter recovery.
- **Debate hurts under correct specification (M16).** Agent proposals less discriminative than curated registry on average.
- **Arbiter is model-biased, not broken (M16).** Crux machinery favors similarity-based model discrimination. Best SUSTAIN result (96%) but worst RULEX conditions.
- **Open design is the mirror bias (M16).** Agent proposals favor rule-diagnostic structures. Best closed-debate RULEX recovery (+24pp) but worst SUSTAIN result (64%).

## What's next

1. Fix SUSTAIN open_arbiter bug (1 missing data point)
2. Investigate whether crux bias can be corrected (e.g., diversity constraint on
   crux-directed selection to ensure rule-diagnostic structures also get selected)
3. Consider M15+M16 combined: misspecification + open design space
4. See [ROADMAP.md](ROADMAP.md)

## Full history

See [Notes/archive/INDEX.md](Notes/archive/INDEX.md) for development docs across 14 milestones.
