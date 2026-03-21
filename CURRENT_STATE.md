# Current State (M17 — Misspecification + Open Design, complete)

## M17 — Misspecification + Open Design

**Scientific question:** When agents start with wrong parameters AND must propose all
experiment structures via debate, do parameter recovery (M15's win) and structure
proposal (M16's RULEX win) compose or interfere?

**Motivation:** M15 showed debate helps under misspecification via parameter recovery
(+22pp RULEX). M16 showed open design helps RULEX via rule-diagnostic proposals (+24pp).
M17 tests the hardest regime: both challenges simultaneously. This is also closest to
real scientific practice, where neither parameters nor experiment designs are pre-specified.

### Results — 6/6 correct (GPT-4o, 5 cycles each)

| GT | Condition | Winner | Correct? | RMSE | Gap% | #Structs | Param Recovery |
|---|---|---|---|---|---|---|---|
| GCM | open_debate | Exemplar_Agent | Yes | 0.114 | 67.3 | 48 | 42.9% |
| GCM | open_arbiter | Exemplar_Agent | Yes | 0.047 | **87.8** | 58 | 85.7% |
| SUSTAIN | open_debate | Clustering_Agent | Yes | 0.085 | 77.4 | 48 | 0% |
| SUSTAIN | open_arbiter | Clustering_Agent | Yes | 0.098 | 72.7 | 51 | 0% |
| RULEX | open_debate | Rule_Agent | Yes | 0.189 | 57.8 | 51 | 46.3% |
| RULEX | open_arbiter | Rule_Agent | Yes | 0.214 | 42.2 | 57 | 0% |

### Cross-milestone comparison

| GT | Condition | M17 (misspec+open) | M15 (misspec+closed) | M16 (correct+open) |
|---|---|---|---|---|
| GCM | open_debate | 67.3% | 77.9% (debate) | 71.6% |
| GCM | open_arbiter | **87.8%** | 79.3% (arbiter) | 76.9% |
| SUSTAIN | open_debate | 77.4% | 85.8% (debate) | 64.1% |
| SUSTAIN | open_arbiter | 72.7% | 76.1% (arbiter) | 70.7% |
| RULEX | open_debate | 57.8% | 80.4% (debate) | 82.7% |
| RULEX | open_arbiter | 42.2% | 3.2% (arbiter, **wrong**) | 82.0% |

### Key findings

**1. GCM open_arbiter (87.8%) — best GCM result across all milestones.**

Parameter recovery (85.7%) and arbiter-guided open proposals compose synergistically.
Better than M15 arbiter (79.3%, closed registry) AND M16 open_arbiter (76.9%, correct
params). The arbiter steers proposals toward similarity-diagnostic structures, and debate
recovers the misspecified sensitivity parameter — the two mechanisms reinforce rather than
interfere. RMSE 0.047 is the second-lowest measured (after SUSTAIN closed_arbiter 0.020).

**2. Open design rescues RULEX from arbiter catastrophe.**

M15 arbiter-RULEX (3.2%, wrong winner) was the project's only incorrect identification.
M17 open_arbiter-RULEX (42.2%, correct) shows the open design space partially counteracts
the arbiter's similarity bias by providing rule-diagnostic structures that the closed
registry lacks. The arbiter still hurts RULEX (42.2% vs open_debate 57.8%) but no longer
causes wrong-winner failure.

**3. Composition is non-additive and model-dependent.**

The effects of misspecification and open design do not simply add:
- GCM: arbiter + misspec + open > either alone (synergy)
- SUSTAIN: open_debate under misspec (77.4%) > open_debate correct spec (64.1%)
- RULEX: open_debate under misspec (57.8%) < M15 debate (80.4%) — param recovery
  weakened by open design's structure diversity

**4. Parameter recovery is modulated by design space.**

| GT | M15 debate recovery | M17 open_debate recovery | M17 open_arbiter recovery |
|---|---|---|---|
| GCM | 85.7% | 42.9% | 85.7% |
| SUSTAIN | 0% | 0% | 0% |
| RULEX | 60.3% | 46.3% | 0% |

GCM fully recovers under open_arbiter but only partially under open_debate. RULEX
recovery degrades from 60.3% (M15 closed) to 46.3% (M17 open_debate) and drops to 0%
under open_arbiter. The arbiter appears to redirect agent attention from parameter
revision toward structure-level reasoning.

**5. 6/6 correct — the system is robust under double stress.**

Even the hardest condition (RULEX open_arbiter: wrong params + arbiter bias + open
design) produces the correct winner. Combined with M14 (18/18), M15 (8/9), and M16
(15/15), the system achieves 47/48 correct across all factorial conditions.

Scripts: `scripts/validation/validate_m17_live.py`

---

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

### Full results — 15/15 runs (GPT-4o, 5 cycles each)

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
| SUSTAIN | open_arbiter | Clustering_Agent | Yes | 0.100 | 70.7 | 54 |
| RULEX | closed_no_debate | Rule_Agent | Yes | 0.053 | 86.1 | 0 |
| RULEX | closed_debate | Rule_Agent | Yes | 0.168 | 58.6 | 14 |
| RULEX | closed_arbiter | Rule_Agent | Yes | 0.140 | 63.9 | 13 |
| RULEX | open_debate | Rule_Agent | Yes | 0.055 | 82.7 | 48 |
| RULEX | open_arbiter | Rule_Agent | Yes | 0.061 | 82.0 | 55 |

**Correct winner: 15/15.**

### Gap advantage (pp over closed_no_debate baseline)

| GT | closed_debate | closed_arbiter | open_debate | open_arbiter |
|---|---|---|---|---|
| GCM | +4.2pp | +2.4pp | -5.2pp | +0.1pp |
| SUSTAIN | +0.9pp | **+8.3pp** | -23.6pp | -17.0pp |
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

**3. Arbiter recovers open design space losses for similarity models.**

The arbiter partially counteracts the open-design penalty for both GCM and SUSTAIN:
- GCM: open_debate -5.2pp → open_arbiter +0.1pp (full recovery)
- SUSTAIN: open_debate -23.6pp → open_arbiter -17.0pp (+6.6pp recovery, still negative)

The crux machinery focuses agent proposals toward similarity-diagnostic structures,
which is exactly what open design proposals lack. Recovery is complete for GCM but
only partial for SUSTAIN, where the open-design penalty is larger.

**4. Computation alone remains the most reliable single condition.**

closed_no_debate achieves 76.8-87.7% gap across all three ground truths with zero
LLM calls. No debate condition consistently beats it across all models. The best
debate results (SUSTAIN closed_arbiter 96.0%, GCM closed_debate 81.0%) come from
specific model-condition pairings, not a universally beneficial intervention.

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

| Model type | M15 misspec closed | M16 correct closed | M16 correct open | M17 misspec open |
|---|---|---|---|---|
| SUSTAIN (clustering) | -11.6pp | **+8.3pp** (best) | -17.0pp | -15.0pp |
| GCM (exemplar) | +4.9pp | +2.4pp | +0.1pp | **+13.4pp** (best GCM) |
| RULEX (rule-based) | **-54.7pp** (wrong) | -22.2pp | -4.1pp | -15.8pp (but correct) |

M17 reveals the interaction: under misspecification + open design, the arbiter's
similarity bias is partially counteracted by the open design space's rule-diagnostic
bias. RULEX open_arbiter (42.2%, correct) vs M15 arbiter (3.2%, wrong) — the open
design space rescued RULEX from catastrophe. GCM open_arbiter (87.8%) achieves the
best GCM result ever via synergy between param recovery and arbiter-guided proposals.

### When each component helps (revised theory)

The Phase 1 "gap-filling" theory was too simple. The full picture:

1. **Computation alone** is the most reliable baseline. 76-88% gap, never gets
   the wrong winner (24/24 across M14-M16).

2. **Debate layer** (agents interpreting + proposing params/structures):
   - Under misspecification: +3.5 to +22pp via parameter recovery (M15)
   - Under correct specification: noise injection, -3 to -28pp (M16)
   - Exception: open design proposals help RULEX when registry has a coverage gap

3. **Arbiter layer** (cruxes + meta-agents + claim-directed selection):
   - Not noise but *bias* toward similarity-based structures
   - Helps SUSTAIN (+8pp) and GCM (+2-5pp) under correct specification
   - Hurts RULEX (-22 to -55pp) under closed design
   - Under misspec + open design (M17): synergy with GCM (+13.4pp, best ever),
     rescues RULEX from wrong winner (3.2% → 42.2%)

4. **Open design space** (agent-proposed structures):
   - Mirror-image bias: semantically rich proposals favor rule-based models
   - Helps RULEX (+24pp over closed_debate), hurts SUSTAIN (-24pp)
   - **Counteracts arbiter bias**: M15 arbiter-RULEX was catastrophic (wrong winner),
     M17 arbiter-RULEX with open design is correct (42.2%)

5. **Composition is non-additive (M17):**
   - Misspec + open compose synergistically for GCM arbiter (87.8%, best ever)
   - Open design rescues RULEX from arbiter catastrophe under misspec
   - But open design weakens RULEX param recovery (60% → 46% → 0%)

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
- **Arbiter is model-biased, not broken (M16).** Crux machinery favors similarity-based model discrimination. Best SUSTAIN result (96%) but worst RULEX conditions. Recovers open-design losses for similarity models.
- **Open design is the mirror bias (M16).** Agent proposals favor rule-diagnostic structures. Best closed-debate RULEX recovery (+24pp) but worst SUSTAIN result (64%). 15/15 correct across all conditions.
- **Composition is non-additive (M17).** Misspec + open design: 6/6 correct. GCM arbiter achieves best-ever 87.8% via synergy. Open design rescues RULEX from arbiter catastrophe (3.2% wrong → 42.2% correct). 47/48 correct across M14–M17.
- **R-IDeA cannot substitute for debate (negative result).** Formal multi-objective OED (representativeness + informativeness + de-amplification) underperforms EIG in all regimes. R-IDeA + debate (53.7% mean) is worse than EIG + debate (81.4%) because diversifying experiments dilutes the visible prediction failures debate needs for parameter recovery. Informativeness and semantic diagnosis are synergistic; diversification and diagnosis are antagonistic.

## What's next

1. **Decision-making domain (pipeline wired)** — EU, CPT, Priority Heuristic models
   (17/17 tests), gamble registry (76 problems), EIG adapter (23 tests), agent configs
   (18 tests), validation script all complete. No-debate baseline: 3/3 correct (correct
   params), 0/3 correct (misspecified params). Remaining: wire debate/arbiter conditions,
   run live LLM experiments. NeurIPS target.
2. Write up two-domain results as NeurIPS submission
3. GeCCo forks (gecco-core, gecco-supplement) — model discovery + adjudication
4. Real data integration (human participants via Prolific/AutoRA)
5. See [ROADMAP.md](ROADMAP.md)

## Full history

See [Notes/archive/INDEX.md](Notes/archive/INDEX.md) for development docs across 14 milestones.
