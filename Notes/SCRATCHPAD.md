# Scratchpad

Working notes, open questions, and in-progress plans. Clean out when work is completed.

---

## M15 Phase 2 — Complete (2026-03-17)

### Full 9-run results

| GT | Condition | Winner | Correct? | RMSE | Gap | Recovery |
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

### Analysis

**Debate without arbiter is the best configuration under misspecification:**
- GCM: +3.5pp gap, 85.7% param recovery (c: 0.5 → near GT 4.0)
- RULEX: +22.4pp gap, 60.3% param recovery (strongest effect)
- SUSTAIN: -1.9pp gap, 0% recovery — neutral (misspecification invisible to agents)

**Arbiter consistently degrades performance:**
- GCM: +4.9pp (only case where arbiter helps)
- SUSTAIN: -11.6pp
- RULEX: -54.7pp, wrong winner

**Why SUSTAIN shows 0% recovery:** SUSTAIN's misspecification (r=3.0, eta=0.15 vs GT
r=9.01, eta=0.092) doesn't produce enough prediction error to trigger LLM revision
proposals. The agents need to *see* prediction failures before proposing param changes.
SUSTAIN still wins easily (87.7% gap even with misspec) because its computational
advantage on clustering-diagnostic structures is robust to these param changes.

**RULEX arbiter failure — root cause (detailed):**
Experiment selection divergence:
- No-debate & Debate: selected **rule-plus-exception** (`rpe`) structures (4/5 cycles)
  — RULEX-diagnostic. Posterior converged to 99.5%.
- Arbiter: selected only **linearly-separable** (`ls`) structures (5/5 cycles) —
  non-discriminative. Posterior eroded from 94% → 69%.

Mechanism: meta-agents (Integrator, Critic) influenced interpretation debate, changing
agents' divergence mapping → different EIG landscape → Thompson sampling favored `ls`
over `rpe`. Crux-directed selection was NOT the cause (0/5 were crux-directed).

### Bugfix during M15

`runner.py:1278` — `spec["crux_id"]` KeyError when claim-directed boost specs (which
lack `crux_id`) matched the selected experiment. Fixed to `spec.get("crux_id")`.
Surfaced during GCM arbiter run.

### M14→M15 synthesis

M14 showed debate was **epiphenomenal** under correct specification — the computational
pipeline alone identified the correct model in 18/18 conditions. M15 shows debate is
**causally necessary** under misspecification, but only the *core debate loop*:

- **Debate layer** (agents + interpretation + param revision): adds causal value.
  LLM agents see prediction failures, diagnose the cause, propose corrections via
  `sync_params_from_theory()`. This is what EIG alone can't do — EIG does selection,
  debate does estimation. Pitt & Myung (2004) confirmed empirically.
- **Arbiter layer** (cruxes + meta-agents): net negative. Meta-agents optimize for
  argumentative richness, not model discrimination. They distort divergence mapping,
  degrading experiment selection quality.

The sweet spot: **debate without arbiter.**

### Open questions

1. **Is the arbiter effect robust?** Single run per condition. Need replications to
   separate signal from Thompson sampling stochasticity.
2. **Would removing meta-agents but keeping cruxes help?** Current arbiter bundles
   cruxes + meta-agents. The problem is meta-agent influence on divergence mapping,
   not crux-directed selection itself.
3. **Why does arbiter help on GCM but hurt everywhere else?** GCM's misspecification
   (c=0.5) may interact differently with meta-agent interpretations.
4. **Can param recovery be triggered on SUSTAIN?** Current misspecification (r=3.0,
   eta=0.15) is too mild — 0% recovery. Need stronger misspecification or explicit
   prompting to diagnose param errors.

---

## Failed approaches (do not repeat)

- **Pre-LOO prediction** — training and testing on same items gives GCM self-similarity bias (D11). Always use LOO.
- **Round-robin experiment selection** — doesn't optimize for discriminability, disadvantages models that are weak on common structures (RULEX on Type_VI).
- **Divergence ranking without per-model predictions** — agents can't interpret abstract divergence scores. They need to see which model wins on each structure.
- **Trusting LLM param overrides** — LLM agents invent parameter names (e.g., `w_i`). Always filter through `inspect.signature`.
- **Pure divergence-driven selection without diversity** — picks the same high-divergence structure every cycle (Type_VI). RULEX never gets tested on favorable structures.
- **EIG greedy optimization** — selects the same structure repeatedly (Phase 13). Thompson sampling (D34) fixes this.
- **Pairwise curve divergence as posterior evidence** — rewards model distinctiveness, not fit to data. Data-independent bonus distorts posterior. Removed in D35.
- **Multiplicative EIG boost for cruxes** — 2× multiplier barely shifts Thompson sampling when EIG scores cluster narrowly. Replaced with mixture distribution in D37.
- **Exact structure name matching for sampled structures** — ephemeral names change every cycle. Fixed with parameter-based fuzzy matching (D42).
