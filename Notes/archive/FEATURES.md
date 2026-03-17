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

### 4.3 Design revision phase
**What it does:** Agents revise proposals based on critiques, updating
design_spec via `state.revise_proposal()`.
**Where:** `runner.py`
**Why it matters:** Critiques now lead to improved proposals before execution.
Implemented in M3 (P1 fix). Addresses the original placeholder gap.

### 4.4 Reject path
**What it does:** In interactive mode, moderator can reject all proposals,
triggering a loop back to proposal→critique→revision→arbitration (up to 3
attempts). Rejected proposals marked with status="rejected".
**Where:** `runner.py`
**Why it matters:** Moderator can force better experiments when all proposals
are weak. Implemented in M3 (P2 fix).

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

## 7. Debate Feedback Features (Implemented — M5, validated 2026-03-15)

These features close the 4 broken feedback loops identified in M4 analysis,
making the debate causally relevant to outcomes. Pre-M5, replication runs had
zero RMSE variance; post-M5, std=0.018 across 4× GCM replicates. All 3 ground
truths still correctly identified. ~45 FALSE CLAIMs detected across validation
runs, confirming critique-as-falsification is active.

### 7.1 Parameter revision persistence
**What it does:** When an agent revises its theory (updating `model_params` in
the epistemic state), the revised parameters propagate back to
`agent_config.default_params` so they affect the *next* cycle's predictions.
**Status:** Implemented (commit `1d12fde`). `sync_params_from_theory()` in
`runner.py` copies params after each interpretation phase, filtering through
`inspect.signature` to reject invalid keys. 6 tests.
**Why it matters:** This is the most direct feedback path from debate to RMSE.
If an agent interprets results and concludes "my attention parameter should be
higher", that insight should actually change predictions. Currently, agents
revise parameters every cycle but predictions never change — the debate is
shouting into a void.
**Implementation:** After each interpretation phase, copy
`theory.model_params` → `agent_config.default_params` for the corresponding
agent. Filter through `inspect.signature` to reject invalid keys.
**Expected impact:** High. This makes the LLM's scientific reasoning directly
affect model fitness. Cross-LLM RMSE variation should increase (currently
param_overrides is the only LLM→RMSE path, and it's ephemeral).

### 7.2 Critique-as-falsification
**What it does:** When an agent claims "my model can also predict that outcome",
the system runs the actual computation and checks whether the claim is true.
**Status:** Implemented (commit `84852bb`). `verify_prediction_claim()` in
`debate_protocol.py` compares claimed vs actual predictions. FALSE CLAIMs
recorded in ledger. ~45 false claims detected across M5 validation. 5 tests.
**Why it matters:** Unverified claims degrade debate quality. An agent claiming
its model predicts something should be held to that claim. This would surface
genuine model limitations rather than allowing agents to hand-wave.
**Implementation:** Parse critique claims containing prediction assertions.
Run `compute_model_predictions()` for the claimed structure/condition/params.
Compare claimed vs. actual prediction. Flag false claims in the epistemic
state and penalize repeat offenders.
**Expected impact:** Medium. Improves debate quality and honesty. Does not
directly change experiment selection or RMSE, but prevents agents from making
unfounded defensive arguments that waste debate cycles.

### 7.3 Debate-informed EIG weighting
**What it does:** When agents identify a contested model pair during
interpretation (e.g., "GCM and RULEX make similar predictions on Type I"),
the system upweights EIG for candidates that discriminate between those
specific models.
**Status:** Implemented (commit `f61eec4`). `select_from_pool()` accepts
`focus_pair` + `pair_boost` (default 1.5×). Contested pairs extracted from
claim ledger or posterior (closest probabilities). 5 tests.
**Why it matters:** EIG optimizes for maximum posterior entropy reduction
across all models. But after a few cycles, the real contest may be between
two specific models. Debate can identify this before the posterior does
(agents reason about model mechanisms, not just fit statistics). Focusing
EIG on the contested pair would accelerate convergence.
**Implementation:** After interpretation, extract contested model pairs from
agent hypotheses. In `select_experiment()`, multiply EIG by a boost factor
(e.g., 1.5×) for candidates where the contested pair has high divergence.
**Expected impact:** Medium-high. Could accelerate convergence by 1–2 cycles
for hard discrimination problems (RULEX took 2 extra cycles in validation).

### 7.4 Structured claim ledger
**What it does:** Maintains a per-agent registry of empirical claims made
during debate, tracking which claims have been tested, confirmed, or
falsified by subsequent experiments.
**Status:** Implemented (commit `4625d53`). `DebateClaim` dataclass in
`epistemic_state.py` with `claim_ledger` on `EpistemicState`. Claims parsed
from agent JSON, statuses updated after execution. Summary injected into
interpretation prompts. 8 tests.
**Why it matters:** Without a claim ledger, the debate has no memory. Agents
can't build on prior arguments or be held accountable for failed predictions.
The ledger would force cumulative scientific progress — each cycle's debate
builds on what was established before, rather than starting fresh.
**Implementation:** After each debate phase, extract testable claims (structure
of: "model X will predict Y on structure Z"). Store in ledger with status
(untested/confirmed/falsified). At the start of each interpretation phase,
show agents the ledger with updated statuses. Flag stale claims (>2 cycles
untested).
**Expected impact:** Medium. Improves debate coherence and prevents repetitive
arguments. Indirect effect on RMSE through better-informed parameter revisions
and more targeted hypotheses.

---

## 8. arbiter-v0.1 Features (Implemented — M6, validated 2026-03-15)

> **arbiter-v0.1** is the current version of the debate architecture. It includes:
> 3 theory agents (Exemplar_Agent/GCM, Rule_Agent/RULEX, Clustering_Agent/SUSTAIN),
> 2 meta-agents (Integrator, Critic), crux-based negotiation, conflict maps,
> pre-registration, claim-responsive debate, and HITL checkpoints. Agents debate
> in natural language but predictions come from executable models (`model.predict()`).
> The computational layer (Bayesian EIG, Thompson sampling, likelihood tempering,
> continuous design space) drives experiment selection; debate provides interpretation
> and hypothesis generation. Future versions may change the agent roster, model set,
> debate protocol structure, or the division of labor between computation and debate.

### 8.1 Role-specialized meta-agents (`MetaAgentConfig`)
**What it does:** Adds Integrator and Critic meta-agents with role-specific system prompts.
The Integrator synthesizes across all three theory agents' responses; the Critic challenges
the weakest argument.
**Where:** `debate_protocol.py` (MetaAgentConfig), `runner.py` (create_default_meta_agents,
INTEGRATOR_PROMPT, CRITIC_PROMPT)
**Why it matters:** Role specialization focuses agent effort. Integrator finds convergence;
Critic finds weakness. Neither overrides Bayesian machinery — their value is qualitative
(narrative structure, human readability).
**M6 result:** 10 meta-agent responses per 5-cycle run. Critic consistently identifies
weakest argument (often from agents with P≈0 posterior). Integrator synthesizes areas
of agreement.

### 8.2 Crux negotiation (`Crux` dataclass + 3-phase protocol)
**What it does:** Three-phase protocol: identification (agents propose 1-2 cruxes per cycle),
negotiation (accept/reject/counter-propose), finalization (2+ supporters → accepted).
Active cruxes boost EIG for matching candidates via `crux_boost_specs`.
**Where:** `epistemic_state.py` (Crux, add_crux, get_active_cruxes, resolve_crux, crux_summary),
`runner.py` (run_crux_identification, run_crux_negotiation, finalize_cruxes, cruxes_to_boost_specs),
`bayesian_selection.py` (crux_boost_specs param)
**Why it matters:** Focuses debate on decisive questions rather than rehashing every disagreement.
With real LLMs, only 15% of proposed cruxes are accepted — agents genuinely reject unpersuasive proposals.
**M6 result:** ~100 cruxes proposed across 3 runs, 15 accepted. Accepted cruxes map to real
theoretical fault lines in cognitive science. Crux boost is active but constrained by
posterior collapse (can't boost zero EIG).
**Open question:** Would crux-driven experiment overrides (bypass EIG when cruxes exist) help?

### 8.3 Conflict map (`conflict_map_summary()`)
**What it does:** Groups claims by structure and condition, showing where models agree and disagree.
Claims carry a `category` field (prediction/mechanism/scope/general). Injected into interpretation prompts.
**Where:** `epistemic_state.py` (conflict_map_summary, DebateClaim.category)
**Why it matters:** Provides structured view of disagreement. Agents can see which predictions are
contested vs settled, and where their theory is vulnerable.
**M6 result:** 40-68 lines per run. Dominated by falsified prediction claims.

### 8.4 Pre-registration (`generate_preregistration()`)
**What it does:** Generates prediction tables (each model's predicted accuracy per structure),
adjudication criteria, active cruxes, and prior accuracy. Saved per cycle.
**Where:** `runner.py` (generate_preregistration)
**Why it matters:** Scientific rigor — predictions committed before experiments run. Enables
post-hoc analysis of which agents' predictions were closest.
**M6 result:** Pre-registration files generated for cycles 1-4 in each run.

### 8.5 HITL checkpoints (`hitl_checkpoint()`)
**What it does:** Optional breakpoints at crux finalization, EIG selection, and pre-registration.
Auto-continues in batch mode; prompts human in interactive mode.
**Where:** `runner.py` (hitl_checkpoint), `__main__.py` (--hitl-checkpoints flag)
**Why it matters:** Enables human oversight at key decision points without requiring fully interactive mode.
**M6 result:** Auto-continued in all validation runs (batch mode). Not yet tested in interactive mode.

---

## Summary: Features Most Likely to Alter Outcomes

Ranked by estimated impact on which theory wins and how fast convergence occurs:

1. **Ground truth model** — determines the answer
2. **Structure registry composition** — determines the search space
3. **Agent default parameters** — determines prior predictions
4. **Condition effect magnitudes** — determines how revealing experiments are
5. **Parameter revision persistence** (7.1) — implemented M5; makes debate causally relevant to RMSE
6. **Divergence display format** — determines whether agents pick diagnostic experiments
7. **Debate-informed EIG weighting** (7.3) — implemented M5; focus pair boosting active
8. **Prediction metric (RMSE vs correlation)** — determines what "winning" means
9. **Critique-as-falsification** (7.2) — implemented M5; ~45 FALSE CLAIMs detected in validation
10. **Structured claim ledger** (7.4) — implemented M5; claims tracked and verified across cycles
11. **Critique rounds** — determines depth of adversarial pressure
12. **Batch arbitration rule** — determines which experiments get run (legacy mode only)
13. **LLM model quality** — validated as non-critical: 3 LLMs produce identical outcomes (9/9 correct)
14. **Lakatos trajectory completeness** — determines whether revision tracking is meaningful

15. **Crux negotiation** (8.2) — implemented M6; 15% acceptance rate, focuses debate on decisive questions
16. **Role-specialized meta-agents** (8.1) — implemented M6; Integrator + Critic shape debate narrative
17. **Conflict map** (8.3) — implemented M6; structured disagreement view in interpretation prompts
18. **Pre-registration** (8.4) — implemented M6; predictions committed before experiments run

**Key finding (M4):** Features 1–4 fully determine outcomes in the current system.
**Update (M5, 2026-03-15):** Features 5, 7, 9, 10 now implemented. Replication
variance is non-zero (std=0.018 vs 0.000 pre-M5). Debate causally affects RMSE
through parameter revision persistence. All 3 ground truths still correctly
identified. Cross-LLM comparison (GPT-4o, Sonnet, Opus) confirms: 9/9 correct.
**Update (M6, 2026-03-15):** Features 15–18 now implemented. arbiter-v0.1 features
enrich debate quality (crux selectivity, meta-agent synthesis, conflict tracking)
but do not alter convergence mechanism. Posterior collapse (D29) was the primary
bottleneck — crux boost cannot overcome zero EIG. 3/3 correct with 36–68% gaps.
**Update (M7, 2026-03-15):** Posterior collapse fixed via likelihood tempering
(tau=0.005, clip [0.05, 0.95]). Entropy=0.635 after cycle 0 (was 0.000), EIG=0.233
on cycle 1 (was 0.000). Multi-cycle debate is now load-bearing — experiment
selection adapts to evolving posterior across cycles.
**Update (M13, 2026-03-16):** 3×2 debate ablation (18/18 correct) confirms debate
is epiphenomenal on synthetic benchmarks. No-debate achieves best RMSE (0.055) and
gap (87.6%) while running 3-4× faster. Debate without arbiter-v0.1 actively hurts
(0.078); arbiter-v0.1 partially recovers (0.059) but doesn't beat no-debate. Features
1–4 remain fully sufficient. The debate→computation feedback loop is architecturally
open — closing it is the key task for making debate non-epiphenomenal.

---

## 9. Posterior Convergence Control (Implemented — M7, 2026-03-15)

### 9.1 Likelihood tempering (`--learning-rate`)
**What it does:** Multiplies log-likelihoods by tau ∈ (0, 1] before Bayesian
posterior updates. tau=1.0 is standard Bayesian update; tau<1 slows convergence,
keeping the posterior spread across models longer and maintaining nonzero EIG.
**Where:** `bayesian_selection.py` (`ModelPosterior.update`, `compute_eig`,
`select_from_pool`, `select_experiment`, `update_posterior_from_experiment`),
`runner.py` (`_LEARNING_RATE` global, 3 call sites, `--learning-rate` CLI),
`__main__.py` (`--learning-rate` flag)
**Current default:** tau=0.005 (calibrated for synthetic data)
**Why it matters:** M6 validation showed posterior collapse to P≈1.0 after 1–2
experiments, making EIG=0 for all remaining candidates. Crux boost can't overcome
zero EIG. Tempering keeps later cycles informative by preventing the posterior
from concentrating too rapidly. Well-established in Bayesian statistics (Grünwald
2012, Bissiri et al. 2016, Miller & Dunson 2019).
**Calibration (D32):** Initial tau=0.2 was insufficient — SUSTAIN's near-binary
predictions (0.0005/0.999) create ~1000 nat LL range, and 0.2 × 1000 = 200 nats
still overwhelms. Two-pronged fix: (1) widen prediction clip from [0.01, 0.99]
to [0.05, 0.95], (2) lower tau from 0.2 to 0.005. Calibrated empirically for
gradual convergence: 1 cycle → H=0.64, 2 cycles → H=0.32, 5 cycles → H=0.02.
**M7 result:** Live validation with tau=0.005: P(GCM)=0.73→0.90 over 2 cycles,
entropy=0.635→0.325, EIG=0.233 on cycle 1 (was 0.000 with tau=0.2). Correct
winner identified. 12 tests across `TestLikelihoodTempering`, `TestConfig`, and
`TestPredictionClipping` (308 total passing).
**Toggles:** `--no-tempering` sets tau=1.0 (standard Bayesian). `--no-arbiter`
disables arbiter-v0.1 features (crux negotiation, meta-agents, conflict map).
Both configurable via YAML config file with layered precedence.

### 9.2 Experiment selection strategy (`--selection-strategy`)
**What it does:** Controls how experiments are chosen from EIG scores. `thompson`
(default) samples proportional to EIG scores, naturally balancing exploration and
exploitation. `greedy` always picks argmax(EIG), which is locally optimal but
selects the same experiment every cycle.
**Where:** `bayesian_selection.py` (`_select_index`, `select_from_pool`,
`select_experiment`), `runner.py` (`_SELECTION_STRATEGY` global, 2 call sites),
`__main__.py` (`--selection-strategy` flag), `default_config.yaml`
**Current default:** thompson
**Why it matters:** Greedy EIG selected the same structure 5/5 cycles for GCM and
SUSTAIN (D33). Thompson sampling (Russo & Van Roy 2018, Kandasamy et al. 2019)
addresses this by exploring underexplored design regions without ad-hoc diversity
bonuses. In clean ablation (D36), Thompson explored 12 unique structures (6 novel
agent-proposed) across 3 ground truths vs 3 unique (0 novel) for greedy.
**M8 result:** Both strategies 3/3 correct post-bugfix. Thompson maintains higher
entropy longer (later cycles informative). Greedy gets lower winner RMSE by
hammering the single best structure. Thompson is more robust — was correct even
with pre-bugfix curve bonus that caused greedy RULEX misidentification in M7.

### 9.3 Crux-directed selection (`--crux-weight`)
**What it does:** Controls the probability of crux-directed experiment selection in
Thompson sampling. With probability `crux_weight`, the system samples uniformly from
candidates matching active cruxes; with probability `1 - crux_weight`, it uses
standard EIG-weighted Thompson. This makes debate causally relevant to experiment
selection: accepted cruxes (from arbiter-v0.1 negotiation) directly influence which
experiment is run.
**Where:** `bayesian_selection.py` (`_select_index` crux mixture, `select_from_pool`
crux_indices computation), `runner.py` (`_CRUX_WEIGHT` global, crux-directed
logging), `__main__.py` (`--crux-weight` flag), `default_config.yaml`
**Current default:** 0.3 (30% crux-directed, 70% EIG-weighted)
**Why it matters:** Before M9, cruxes existed but never affected experiment selection.
The old multiplicative EIG boost was doubly broken: (1) agents wrote free-text
descriptions instead of `structure/condition` format, so parsing always failed;
(2) even when fixed, a 2× multiplier barely shifts Thompson's sampling distribution.
The mixture distribution guarantees debate has causal influence: when cruxes are
active, ~30% of experiments will test the specific questions agents negotiated.
**Pre-M9 crux pipeline failure:** Zero boost specs produced across all M6/M7/M8
validation runs. 100+ cruxes proposed, 0 parsed into actionable boost specs.
**M9 fixes:** Prompt now shows structure/condition menu; parsing validates against
known structures and strips whitespace; mixture replaces multiplicative boost.
