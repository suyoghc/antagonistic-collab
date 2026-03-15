# Features That Shape Debate Outcomes

Scientifically meaningful design choices in the adversarial collaboration
framework. Each feature, if varied, could alter which theory wins, how fast
convergence occurs, or whether the debate produces genuine scientific insight.

---

## 1. Ground Truth & Data Generation

### 1.1 True model (`--true-model`)
**What it does:** Selects which cognitive model (GCM, SUSTAIN, RULEX) generates
synthetic experimental data.
**Current default:** GCM
**Why it matters:** This is the single largest lever. It determines which agent's
predictions will match reality. Validated: when true_model=GCM, Exemplar_Agent
wins with 3.6x RMSE gap.
**Open question:** Does the system converge equally well for all three ground
truths, or is one model easier to identify than others?

### 1.2 Structure registry (STRUCTURE_REGISTRY)
**What it does:** Defines the 11 category structures agents can propose
experiments on (Shepard I–VI, 5-4, rule+exception ×2, linear-separable ×2).
**Where:** `debate_protocol.py` L40–54
**Why it matters:** The menu composition determines the debate's search space.
A registry biased toward rule-friendly structures (more Type I variants) would
systematically advantage Rule_Agent. The current mix is roughly balanced, but
agents never pick 5-4 despite its highest divergence (0.556).
**Open question:** Would adding more structures increase discriminability, or
does the current set already cover the interesting regions?

### 1.3 Condition effects (CONDITION_EFFECTS)
**What it does:** Maps 5 experimental conditions (baseline, low/high attention,
fast presentation, high noise) to per-model parameter overrides.
**Where:** `debate_protocol.py` L78–104
**Why it matters:** Conditions are the main way the same structure produces
different data. The magnitude of parameter perturbations determines how
revealing each condition is. For example, `low_attention` sets GCM's c from 3.0
to 1.5 — a large shift that significantly changes predictions.
**Open question:** Are the current perturbation magnitudes well-calibrated?
Could conditions be tuned to maximize cross-model discrimination?

### 1.4 Synthetic noise (n_subjects + binomial sampling)
**What it does:** Adds realistic sampling noise to model predictions. Each
item's accuracy is drawn from Binomial(n_subjects, p_correct).
**Where:** `debate_protocol.py` L812–826
**Current default:** n_subjects=30 (from agent's design spec), deterministic
seed from hash(cycle, structure, condition)
**Why it matters:** Low n_subjects (5–10) produces noisy data where models
can't be distinguished. High n_subjects (100+) produces tight estimates where
even small model differences become detectable. The seed ensures reproducibility
but means repeated experiments give similar (not identical) results.

---

## 2. Model Parameters & Predictions

### 2.1 Agent default parameters
**What it does:** Each agent starts with fixed model parameters that determine
its predictions before any overrides.
**Where:** `debate_protocol.py` L273–297

| Agent | Key params | Effect |
|-------|-----------|--------|
| Exemplar (GCM) | c=3.0, r=1, gamma=1.0 | Moderate sensitivity, city-block distance |
| Rule (RULEX) | p_single=0.5, p_conj=0.3, error_tolerance=0.1 | Balanced rule search, strict acceptance |
| Clustering (SUSTAIN) | r=9.01, beta=1.252, d=16.924, eta=0.092 | Sharp focus, moderate learning rate |

**Why it matters:** These defaults determine each model's "prior" predictions.
If GCM's default c=3.0 happens to be close to the ground-truth c=4.0, GCM
looks good. If SUSTAIN's default eta=0.092 is far from optimal, SUSTAIN looks
worse than it should. The defaults encode implicit assumptions about the
cognitive regime being studied.

### 2.2 Parameter layering (defaults → condition → overrides)
**What it does:** Agent defaults are overridden by condition effects, which are
overridden by agent-requested param_overrides from the LLM.
**Where:** `debate_protocol.py` L573–579, `runner.py` L624–637
**Why it matters:** This layering determines whether agents can adapt their
models to the data. If param_overrides are ignored (the bug we fixed), agents
are stuck with defaults + condition effects. With overrides, agents can request
theoretically motivated parameter changes.

### 2.3 RULEX stochasticity
**What it does:** RULEX's rule search is inherently stochastic — different
random seeds find different rules for the same structure.
**Where:** `models/rulex.py` L223–304, forced seed=42 in
`compute_model_predictions`
**Why it matters:** We force seed=42 for determinism, but this means RULEX
always finds the same rule for the same structure. In reality, different
learners might find different rules. This could underestimate RULEX's
variability and overstate its confidence.

### 2.4 SUSTAIN order-dependence
**What it does:** SUSTAIN processes training items sequentially — presentation
order affects cluster recruitment and final representation.
**Where:** `models/sustain.py` L162–271
**Why it matters:** Currently, `compute_model_predictions` calls `predict()`
which runs a full training simulation in list order. Different orderings would
produce different predictions. This is a genuine model property (not a bug),
but the system doesn't exploit it — agents can't specify presentation order
in their proposals.

---

## 3. Scoring & Accountability

### 3.1 Prediction metric (RMSE)
**What it does:** Scores predictions as root-mean-squared error between
predicted and actual item accuracies.
**Where:** `epistemic_state.py` L387–410
**Why it matters:** RMSE penalizes large errors quadratically — a 0.2 error
costs 4x as much as a 0.1 error. An alternative metric (correlation) would
only check rank-order agreement, which is much more lenient. Under correlation,
an agent that gets the pattern right but the scale wrong would score well.
Under RMSE, it would score poorly.

### 3.2 Item-level vs. mean-level scoring
**What it does:** Predictions include per-item accuracies (item_0 through
item_N) plus mean_accuracy. Scoring uses all shared keys.
**Where:** `runner.py` L670–677
**Why it matters:** Item-level scoring exposes where models actually disagree.
Two models might predict the same mean_accuracy (0.75) but disagree sharply
on which items are easy vs. hard. Item-level scoring catches this; mean-only
scoring would miss it entirely.

### 3.3 Prediction leaderboard visibility
**What it does:** After each execution phase, all agents see the cumulative
RMSE leaderboard showing who's winning.
**Where:** `runner.py` L678–687, `epistemic_state.py` L427–442
**Why it matters:** Visible scores create competitive pressure. Agents can see
if they're losing and may adjust strategy (propose experiments that favor their
model, make conservative predictions). Hiding the leaderboard until the end
would remove this pressure, potentially changing proposal strategy.

### 3.4 Lakatos trajectory tracking
**What it does:** Classifies theory revisions as "progressive" (generates new
testable predictions) or "degenerative" (patches post-hoc without new
predictions).
**Where:** `epistemic_state.py` L444–541
**Why it matters:** This is the system's operationalization of Lakatos's
methodology of scientific research programmes. Progressive revisions signal
a healthy theory; degenerative revisions signal a dying one. However, the
implementation is incomplete — it does NOT verify whether new predictions are
subsequently tested or confirmed (L537–538 marked TODO). Agents can game it
by making speculative predictions that are never tested.

---

## 4. Debate Dynamics

### 4.1 Critique rounds (`--critique-rounds`)
**What it does:** Controls how many rounds of adversarial critique occur per
cycle.
**Current default:** 2
**Why it matters:** More rounds give agents more opportunities to identify
genuine weaknesses — but they can also lead to circular "my model can also
predict that" arguments (see LESSONS_LEARNED 1.4). There may be diminishing
returns or even negative returns beyond some threshold.

### 4.2 Batch-mode arbitration (round-robin + critique tiebreak)
**What it does:** In batch mode, selects which proposal to run by rotating
across agents (fewest prior approvals first), breaking ties by critique count
(more critiques = more refined).
**Where:** `runner.py` L486–511
**Why it matters:** This replaces human scientific judgment with a deterministic
rule. The round-robin ensures fairness but removes the moderator's ability to
strategically test unfavorable theories. A different selection rule (e.g.,
pick the proposal targeting the current leader's weakness) would change debate
dynamics significantly.

### 4.3 Design revision phase (placeholder)
**What it does:** Currently skipped — agents revise implicitly during critique
rounds.
**Where:** `runner.py` L862–865
**Why it matters:** Without explicit revision, critiques don't lead to improved
proposals. The approved experiment is the original, uncorrected proposal.
Implementing this phase would let agents incorporate critiques before execution,
potentially producing more diagnostic experiments.

### 4.4 Reject path (non-functional)
**What it does:** In interactive mode, the moderator can type "reject" but
nothing happens — the cycle advances anyway.
**Where:** `runner.py` L552–553
**Why it matters:** A functional reject path would let the moderator force
better experiments when all proposals are weak. Without it, every cycle runs
an experiment regardless of quality.

### 4.5 Agent system prompts
**What it does:** Each agent gets a ~300-word prompt encoding its theoretical
commitments, what it should argue for, what it struggles with, and how to
critique opponents.
**Where:** `debate_protocol.py` L145–270
**Why it matters:** The prompts shape agent behavior profoundly. Key design
choices embedded in prompts:
- "Be honest about what your theory struggles with" — aspirational but not
  enforced
- Specific critique strategies (e.g., "challenge SUSTAIN on order-dependence")
- What experiments to advocate for (e.g., "non-linearly-separable categories")

Changing prompt framing (e.g., emphasizing "find decisive evidence" vs. "defend
your theory") would shift agent behavior from scientific to adversarial.

---

## 5. Information Presentation

### 5.1 Divergence ranking
**What it does:** Before proposing experiments, agents see structures ranked by
maximum prediction divergence across models.
**Where:** `debate_protocol.py` L565–586
**Why it matters:** Intended to guide agents toward maximally diagnostic
experiments. But agents consistently ignore it, choosing structures by narrative
familiarity instead (see LESSONS_LEARNED 2.4, 3.4). The ranking shows only
abstract divergence scores (e.g., "max divergence = 0.556").

### 5.2 Concrete model predictions in divergence display (NOT YET IMPLEMENTED)
**What it would do:** Instead of abstract scores, show each model's predicted
accuracy per structure (e.g., "GCM: 0.85, RULEX: 0.40, SUSTAIN: 0.44").
**Why it matters:** Concrete predictions are harder to ignore than abstract
divergence numbers. This addresses the root cause of 5.1 — agents don't know
what the divergence means for their model's behavior. Showing that RULEX
predicts 0.40 on 5-4 while GCM predicts 0.85 makes the diagnostic value
immediately obvious.

### 5.3 Epistemic state summary
**What it does:** At each phase, agents see: active theories, prediction
leaderboard, established facts, open disputes, recent experiments.
**Where:** `epistemic_state.py` L573–684
**Why it matters:** This is the shared context that grounds the debate. It
determines what agents know about the current state of play. Omitting the
leaderboard would remove competitive pressure. Omitting prior experiments
would prevent agents from building on earlier results.

### 5.4 Term glossary (anti-straw-manning)
**What it does:** Each theory defines key terms operationally (e.g.,
"attention" → "w_i, dimensional weight, sum to 1").
**Where:** `epistemic_state.py` L37–43
**Why it matters:** Prevents agents from attacking a caricature of the opposing
theory. Without the glossary, "attention" in an exemplar model and "attention"
in a clustering model could mean very different things, leading to spurious
disagreements.

---

## 6. LLM Configuration

### 6.1 LLM model (`--model`)
**What it does:** Selects which LLM all agents use for reasoning.
**Current default:** claude-sonnet (Anthropic) or gpt-4o (Princeton)
**Why it matters:** Model quality affects critique rigor, proposal creativity,
and interpretation depth. A weaker model might produce circular critiques;
a stronger model might identify genuinely novel experimental designs.

### 6.2 Temperature
**What it does:** Controls LLM sampling randomness.
**Current default:** 0.7
**Where:** `runner.py` L72
**Why it matters:** Higher temperature (0.9+) produces more creative but less
reliable proposals. Lower temperature (0.3) produces focused, repetitive
proposals. The current 0.7 balances exploration and reliability.

### 6.3 Max tokens
**What it does:** Limits LLM response length.
**Current default:** 4096
**Where:** `runner.py` L71
**Why it matters:** Long responses allow detailed justification; short responses
force conciseness. Truncated responses can lose the JSON block entirely,
causing parse failures.

---

## Summary: Features Most Likely to Alter Outcomes

Ranked by estimated impact on which theory wins and how fast convergence occurs:

1. **Ground truth model** — determines the answer
2. **Structure registry composition** — determines the search space
3. **Agent default parameters** — determines prior predictions
4. **Condition effect magnitudes** — determines how revealing experiments are
5. **Divergence display format** — determines whether agents pick diagnostic experiments
6. **Prediction metric (RMSE vs correlation)** — determines what "winning" means
7. **Critique rounds** — determines depth of adversarial pressure
8. **Batch arbitration rule** — determines which experiments get run
9. **LLM model quality** — determines reasoning quality across all phases
10. **Lakatos trajectory completeness** — determines whether revision tracking is meaningful
