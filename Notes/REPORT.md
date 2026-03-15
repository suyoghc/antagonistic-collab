# Antagonistic Collaboration via LLM Debate: Can AI Agents Resolve Scientific Disputes?

**Phase: M6 — ARBITER Integration (updated)**
**Date: 2026-03-15** (originally 2026-03-14; updated with M5 and M6 validation results)

---

## Abstract

We present an antagonistic collaboration framework in which three LLM agents — each representing a competing theory of human category learning — debate through a structured protocol, propose experiments, and converge toward the theory that best explains synthetic data. The three models are the Generalized Context Model (GCM; Nosofsky 1986), SUSTAIN (Love, Medin & Gureckis 2004), and RULEX (Nosofsky, Palmeri & McKinley 1994). We compare two architectures: a *legacy* mode where LLM agents propose experiments through adversarial debate, and a *full-pool* mode where Bayesian expected information gain (EIG) selects experiments while agents shift to interpreting results and generating hypotheses. Across 6 validation runs (3 ground truths × 2 modes, 5 cycles each), the correct model's agent wins in every condition. Full-pool mode achieves dramatically better discrimination for hard model pairs (RULEX gap: 2.4% legacy vs. 68% full-pool), driven by learning curves as a second evidence channel. Cross-LLM comparison (GPT-4o, Claude Sonnet, Claude Opus) confirms the framework is LLM-agnostic (9/9 correct). After closing four broken feedback loops (M5), debate now causally affects RMSE through parameter revision persistence (replication std=0.018, previously 0.000), and critique-as-falsification reveals that agents overclaim model accuracy by 3–5×. M6 adds ARBITER-inspired architecture: role-specialized meta-agents (Integrator, Critic), crux-based negotiation for focusing debate on decisive questions, conflict maps, and pre-registration output. Live M6 validation with GPT-4o achieves 3/3 correct with decisive gaps (36–68%), reveals the system operates as a falsification engine (44 claims falsified, 1 confirmed), and identifies posterior collapse as the primary architectural bottleneck. We distill 12 theses on what LLM-mediated scientific debate can and cannot do.

---

## 1. Introduction

### 1.1 The problem: theory adjudication in cognitive science

Category learning is one of the oldest research domains in cognitive science, with multiple competing theoretical accounts that have coexisted for decades. Three of the most prominent are:

- **The Generalized Context Model (GCM)** — an exemplar-based approach where classification depends on summed similarity to all stored instances of each category (Nosofsky 1986). The model is notable for its flexibility: attention weights allow it to approximate rule-like behavior by concentrating on diagnostic dimensions.

- **SUSTAIN** — a clustering model where categories are represented by adaptive clusters that recruit and update through learning (Love, Medin & Gureckis 2004). Its key property is order-sensitivity: learning depends on the sequence in which items are encountered, with new clusters recruited when prediction fails.

- **RULEX** — a rule-plus-exception model where learners search stochastically for single-dimension or conjunctive rules, then memorize items that violate the discovered rule (Nosofsky, Palmeri & McKinley 1994). Its key property is discreteness: learning transitions are sudden (rule discovery), not gradual.

These models make overlapping predictions for many category structures, making it difficult to adjudicate between them from a single experiment. The traditional approach — designing carefully targeted experiments to distinguish pairs of models — is slow, expert-dependent, and susceptible to narrative bias (researchers tend to design experiments that favor their preferred theory).

### 1.2 Antagonistic collaboration

Mellers, Hertwig & Kahneman (2001) proposed *adversarial collaboration* as a method for resolving disputes between competing theoretical camps: proponents of each theory jointly design a decisive experiment and agree in advance on what outcomes would adjudicate between their accounts. This framework has been used in psychology and behavioral economics, but it requires significant coordination between researchers.

We ask: **can LLM agents serve as proxies for the competing theorists, automating the adversarial collaboration process?** Each agent is given a theoretical commitment, access to its model's predictions, and a structured debate protocol. The system runs iteratively: agents propose experiments, critique each other's proposals, execute the winning experiment against a known ground-truth model, interpret results, and revise their theories.

### 1.3 Why LLMs?

LLMs offer three potential advantages for this task:

1. **Theoretical fluency** — they can articulate and reason about model mechanisms, generate hypotheses, and interpret results in natural language.
2. **No sunk-cost bias** — unlike human researchers, LLM agents have no career investment in their assigned theory.
3. **Scalability** — many debate cycles can run automatically without human intervention.

The question is whether these advantages translate into genuine scientific convergence, or whether LLM debate produces merely the *appearance* of scientific reasoning without its substance.

---

## 2. Methods

### 2.1 Cognitive models

Each agent is backed by a computational model that generates quantitative predictions.

**GCM (Generalized Context Model).** Classifies a test stimulus by computing its similarity to all stored training exemplars, then applying the Luce choice rule. Key parameters: sensitivity `c` ∈ [0.1, 20.0] (higher = sharper similarity gradient), per-dimension attention weights (sum to 1), distance metric `r` ∈ {1, 2} (city-block or Euclidean), response scaling `γ` ≥ 1. Predictions are computed via leave-one-out cross-validation to avoid self-similarity bias (D11).

**SUSTAIN (Supervised and Unsupervised STratified Adaptive Incremental Network).** Classifies by computing similarity to adaptive clusters, with new clusters recruited when prediction fails. Key parameters: attentional focus `r` (default 9.01), cluster competition `β` (default 1.252), decision consistency `d` (default 16.924), learning rate `η` (default 0.092), recruitment threshold `τ` (default 0.0). Order-dependent: learning depends on presentation sequence.

**RULEX (Rule-Plus-Exception Model).** Classifies by stochastic search for single-dimension or conjunctive rules, then memorizes exceptions. Key parameters: `p_single` (probability of testing a single-dimension rule, default 0.5), `p_conj` (conjunctive rule probability, default 0.3), `p_exception` (exception retrieval probability, default 0.8), `max_search_steps` (default 50), `error_tolerance` (default 0.1). Rule discovery is discrete — learning transitions are sudden, not gradual.

All three models implement `predict(stimulus, training_items, training_labels) → {probabilities, ...}` for item-level classification and `predict_learning_curve(training_sequence, test_items, test_labels, block_size) → [block_results]` for incremental learning dynamics.

### 2.2 Category structures and conditions

The experiment space consists of **11 category structures** × **5 experimental conditions** = 55 candidate experiments.

**Structures** (from STRUCTURE_REGISTRY):
- Shepard Types I–VI (Shepard, Hovland & Jenkins 1961) — canonical structures varying from single-dimension rules (Type I) to all-exceptions parity problems (Type VI)
- Medin & Schaffer (1978) five-four structure — 9 items with complex category boundary
- Rule-plus-exception structures (1 and 2 exceptions per category)
- Linear separable structures (2D and 4D Gaussian clusters)

**Conditions** (from CONDITION_EFFECTS): baseline, low_attention, high_attention, fast_presentation, high_noise. Each condition maps to model-specific parameter perturbations (e.g., low_attention: GCM c=1.5, SUSTAIN r=3.0, RULEX p_single=0.3).

### 2.3 Debate protocol

The framework implements two operating modes.

**Legacy mode** (9 phases per cycle):
1. **Commitment** — agents declare theoretical commitments and model parameters
2. **Divergence mapping** — system computes pairwise prediction divergence (LOO) across all structures
3. **Experiment proposal** — each agent proposes a discriminating experiment from the structure registry
4. **Adversarial critique** — agents critique each other's proposals (2 rounds)
5. **Design revision** — agents revise proposals based on critiques
6. **Human arbitration** — moderator selects the winning experiment (in batch mode: EIG or heuristic)
7. **Execution** — synthetic data generated from ground-truth model; all models' predictions computed and scored (RMSE on item-level accuracy)
8. **Interpretation** — agents interpret results and propose theory revisions
9. **Audit** — impartial auditor summarizes cycle and checks convergence

Phases 3–6 may loop up to 3 times if the moderator rejects all proposals.

**Full-pool mode** (modified flow):
1. **Commitment** (cycle 0 only)
2. **Divergence mapping**
3. **Full-pool Bayesian selection** — EIG computed over all 55+ candidates (no LLM calls); replaces phases 3–6
4. **Execution** — standard scoring + learning curve computation for all 3 models
5. **Interpretation debate** — agents produce structured JSON (interpretation, confounds, hypotheses, optional novel structure proposals)
6. **Interpretation critique** — agents challenge each other's interpretations
7. **Audit**

### 2.4 Bayesian experiment selection

The Bayesian module maintains a posterior distribution P(model | data) across the three models, initialized as uniform (1/3 each).

**Expected Information Gain (EIG):** For each candidate experiment, EIG is computed via Monte Carlo sampling (`n_sim=200`, `n_subjects=20`):

```
EIG(experiment) = H(prior) − E[H(posterior | data)]
```

For each hypothetical ground truth model (weighted by current prior), `n_sim` synthetic datasets are simulated via binomial sampling. Each dataset is scored under all three models via item-level log-likelihood, the posterior is updated, and the resulting entropy is averaged. The experiment with the highest EIG is selected.

**Posterior update** combines two evidence sources:
1. **Accuracy evidence** (primary): item-level log-likelihoods from `binom.logpmf(observed, n_subjects, predicted)` for each model
2. **Curve evidence** (secondary, weight=0.5): RMSE between each model's predicted learning curve and the ground-truth curve, converted to log-likelihood via `curve_ll = −curve_rmse × n_subjects × curve_weight`

### 2.5 Learning curves as a second evidence channel

Each model's `predict_learning_curve()` produces a block-by-block accuracy trajectory on held-out test items. The three models predict qualitatively different dynamics:

- **GCM** → *gradual* improvement (more stored exemplars → better similarity computation)
- **RULEX** → *sudden* jumps (discrete rule discovery)
- **SUSTAIN** → *stepwise* increases (cluster recruitment events)

Curve features are extracted automatically: `final_accuracy`, `onset_block`, `max_jump`, `learning_pattern` (gradual/sudden/stepwise). These features are included in the interpretation debate context so agents can reason about curve shapes.

### 2.6 Novel structure generation

During interpretation debate, agents may propose novel category structures beyond the 11 in the registry. Proposed structures are validated (4–32 items, ≤8 dimensions, ≥2 categories) and, if valid, added to the candidate pool for the next cycle's EIG computation. Agents receive few-shot examples and strategic guidance for structure design.

### 2.7 Synthetic data generation

For each selected experiment, the ground-truth model generates "observed" data:
1. Train the ground-truth model on the structure's stimuli and labels
2. For each item, compute P(correct | model) via LOO cross-validation
3. Simulate observed accuracy via binomial sampling with `n_subjects=20`
4. Deterministic seeding (md5 hash of structure + condition + cycle) ensures reproducibility

### 2.8 LLM backend

All validation runs used GPT-4o via Princeton/Portkey gateway. Each agent receives a role-specific system prompt defining its theoretical commitments, model description, and output format. API calls include exponential-backoff retry (up to 3 attempts).

### 2.9 Debate feedback features (M5)

Four features close the feedback loops between debate and the quantitative pipeline:

1. **Parameter revision persistence** — `sync_params_from_theory()` copies theory params revised during interpretation back to `agent_config.default_params`, filtered through `inspect.signature` to reject invalid keys. This is the primary debate→RMSE feedback path.

2. **Structured claim ledger** — `DebateClaim` dataclass tracks testable predictions across cycles. Claims are parsed from agent JSON during interpretation, and statuses are updated (confirmed/falsified) after execution. Active claims are injected into subsequent interpretation prompts.

3. **Critique-as-falsification** — `verify_prediction_claim()` runs the actual model computation when an agent claims a specific prediction value, comparing claimed vs actual output. FALSE CLAIMs are flagged and recorded in the ledger.

4. **Debate-informed EIG weighting** — `select_from_pool()` accepts a `focus_pair` (the two models with closest posterior probabilities, or the most-disputed pair in the claim ledger) and multiplies EIG by 1.5× for candidates where those models have high prediction divergence.

### 2.10 ARBITER features (M6)

M6 adds five ARBITER-inspired features:

1. **Role-specialized meta-agents** — `MetaAgentConfig` defines Integrator and Critic roles with dedicated system prompts. The Integrator synthesizes across all three theory agents' responses; the Critic identifies and challenges the weakest argument. Meta-agents respond after theory agents in interpretation debate but do not trigger parameter revisions or model predictions.

2. **Crux negotiation** — A three-phase protocol between divergence mapping and experiment selection. *Identification*: each theory agent proposes 1–2 cruxes — questions whose answer would change their mind. *Negotiation*: agents accept, reject, or counter-propose each other's cruxes. *Finalization*: cruxes with 2+ supporters become active. Active cruxes are converted to `crux_boost_specs` that multiply EIG for matching experiment candidates, focusing experiment selection on decisive questions.

3. **Conflict map** — `conflict_map_summary()` groups claims by structure and condition, showing where models agree and disagree. Claims carry a `category` field (prediction/mechanism/scope/general). The conflict map is injected into interpretation prompts to help agents engage with the structure of disagreement.

4. **Pre-registration** — `generate_preregistration()` produces prediction tables (each model's predicted accuracy per tested structure), adjudication criteria (what RMSE gap counts as decisive), active cruxes and their resolution criteria, and claimed vs actual accuracy for prior cycles. Generated at the start of each cycle after cycle 0.

5. **HITL checkpoints** — `hitl_checkpoint()` at crux finalization, EIG selection, and pre-registration. Auto-continues in batch mode; prompts human moderator in interactive mode. Controlled by `--hitl-checkpoints` CLI flag.

### 2.11 Validation protocol

Six M4 runs: 3 ground truths × 2 modes, each 5 cycles. Nine cross-LLM runs: 3 ground truths × 3 LLMs. Three M5 validation runs: 3 ground truths, full_pool with GPT-4o. Four M5 replication runs: GCM ground truth, full_pool with GPT-4o (for variance analysis). Three M6 live validation runs: 3 ground truths, full_pool with GPT-4o, all ARBITER features enabled. 287 automated tests verify framework correctness.

---

## 3. Results

### 3.1 Primary outcome: correct model identification

All 6 validation runs correctly identified the ground-truth model's agent as the winner.

| Ground Truth | Mode | Winner | RMSE | 2nd Place | RMSE | Gap |
|---|---|---|---|---|---|---|
| GCM | full_pool | Exemplar_Agent | 0.161 | Clustering_Agent | 0.242 | 34% |
| GCM | legacy | Exemplar_Agent | 0.255 | Clustering_Agent | 0.404 | 37% |
| SUSTAIN | full_pool | Clustering_Agent | 0.270 | Rule_Agent | 0.465 | 42% |
| SUSTAIN | legacy | Clustering_Agent | 0.361 | Rule_Agent | 0.546 | 34% |
| RULEX | full_pool | Rule_Agent | 0.119 | Clustering_Agent | 0.366 | 68% |
| RULEX | legacy | Rule_Agent | 0.429 | Exemplar_Agent | 0.440 | 2.4% |

Full-pool mode produces lower absolute RMSE for the correct agent (0.119–0.270 vs 0.255–0.429) and larger discrimination gaps in 2 of 3 conditions.

### 3.2 Learning curves solve the GCM-RULEX discrimination problem

The most dramatic difference between modes is for RULEX ground truth: legacy achieves only a 2.4% gap (Rule_Agent RMSE 0.429 vs Exemplar_Agent 0.440), while full-pool achieves 68% (0.119 vs 0.366).

This is because GCM can approximate RULEX's final accuracy on most structures by concentrating attention weights on the diagnostic dimension (consistent with Nosofsky 1991). The models produce similar *endpoint* predictions but different *learning dynamics*: GCM improves gradually as more exemplars are stored; RULEX jumps suddenly when a rule is discovered. Learning curve evidence (curve_weight=0.5 in the Bayesian update) breaks this tie.

### 3.3 Bayesian EIG experiment selection

Full-pool mode selects experiments via EIG from the full candidate pool:

| Ground Truth | Cycle 0 | Cycle 1 | Cycle 2 | Cycle 3 | Cycle 4 |
|---|---|---|---|---|---|
| GCM | five_four/fast | five_four/fast | five_four/fast | five_four/fast | five_four/fast |
| SUSTAIN | five_four/fast | Type_I/fast | five_four/baseline | five_four/baseline | five_four/baseline |
| RULEX | five_four/fast | Type_I/low_attn | Type_I/low_attn | Type_I/low_attn | Type_I/low_attn |

`five_four / fast_presentation` is universally highest-EIG in cycle 0 — the five-four structure has the most items (9) and most complex boundary, producing maximal model disagreement.

For RULEX ground truth, EIG shifts to `Type_I / low_attention` after cycle 0. This is a simple single-dimension rule structure where RULEX excels (RMSE 0.06 vs GCM's 0.35). The Bayesian system correctly identifies that the initial experiment (five_four) misleads the posterior, and selects a maximally corrective follow-up.

Legacy mode selects more diverse structures (agents propose different experiments each cycle) but this diversity is not strategically optimal — it is driven by narrative preferences rather than information gain.

### 3.4 Posterior convergence dynamics

| Ground Truth | Cycle 0 P(correct) | Cycle 1 | Cycle 2 | Cycle 3 | Cycle 4 |
|---|---|---|---|---|---|
| GCM | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| SUSTAIN | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| RULEX | 5.5×10⁻⁵ | 5.5×10⁻⁵ | **0.9998** | 1.0000 | 1.0000 |

GCM and SUSTAIN are immediately identifiable (posterior collapses at cycle 0). RULEX requires 2 cycles: the initial five_four experiment favors GCM, but EIG finds Type_I/low_attention in cycle 1, and the posterior flips in cycle 2. This demonstrates the Bayesian system working as designed — initial evidence can mislead, but adaptive experiment selection eventually finds the correct model.

### 3.5 Replication: zero variance in full-pool mode

Three replication runs per ground truth (9 total) revealed that full-pool mode RMSE values are **perfectly deterministic** — identical across replicates to 4 decimal places:

| Ground Truth | Rep 1 RMSE | Rep 2 RMSE | Rep 3 RMSE | Winner |
|---|---|---|---|---|
| GCM | 0.1587 | 0.1587 | 0.1587 | Exemplar_Agent |
| SUSTAIN | 0.2701 | 0.2701 | 0.2701 | Clustering_Agent |
| RULEX | 0.1580 | 0.1580 | 0.1580 | Rule_Agent |

This is because the entire quantitative pipeline is deterministic: EIG selection (same prior → same experiment), synthetic data generation (md5-seeded), and model predictions (deterministic given parameters and structure). The only source of variation is the LLM interpretation text, which does not feed back into RMSE computation.

This result strengthens finding 3.1 (correct model always wins) — the outcome is not sensitive to stochastic variation because there is none. It also demonstrates the sharp separation between the computational and LLM components: the Bayesian machinery determines the quantitative result; the LLM debate produces varying natural-language interpretations that do not affect convergence.

### 3.6 Cross-LLM comparison: GPT-4o vs Claude Sonnet vs Claude Opus

To test whether convergence depends on the LLM backbone, we ran full-pool mode with three different models (5 cycles each, all 3 ground truths = 9 runs):

| Ground Truth | GPT-4o Winner (RMSE) | Sonnet Winner (RMSE) | Opus Winner (RMSE) |
|---|---|---|---|
| GCM | Exemplar (0.159) | Exemplar (0.159) | Exemplar (0.143) |
| SUSTAIN | Clustering (0.270) | Clustering (0.270) | Clustering (0.270) |
| RULEX | Rule (0.158) | Rule (0.148) | Rule (0.213) |

The correct model wins in all 9/9 runs. SUSTAIN RMSE is identical across all 3 LLMs (0.270). GCM and RULEX show small variation (0.143–0.159 and 0.148–0.213 respectively) due to LLM-proposed `param_overrides` during execution — the one surviving code path where LLM output affects RMSE.

The framework is LLM-agnostic for convergence. The choice of backbone affects interpretation quality and parameter proposals, but not which model wins.

### 3.8 Novel structure generation

Across 3 full-pool runs (15 cycles), agents proposed 21 novel structures:

| Category | Count | Examples |
|---|---|---|
| Random/unstructured | 5 | random_assignment, randomized_no_rule |
| Complex conjunctive | 4 | complex_conjunction, noisy_xor |
| Multimodal/subgroup | 5 | multimodal_subgroups, overlapping_clusters |
| Attention/order-based | 3 | order_dependency_test, nonverbal_complex |
| Other | 4 | noisy_or, staggered_overlap, asymmetric_complex |

None of the 21 novel structures were selected by EIG. The Bayesian selector consistently preferred registry structures (five_four, Type_I) — either because the 11 registry structures already span the relevant discrimination space, or because LLM-proposed structures are narratively interesting but not statistically optimal.

### 3.9 Theory revision patterns

| Theory | Revisions when TRUE model | Revisions when NOT true |
|---|---|---|
| GCM (Exemplar) | 0 (stable) | 1–4 (progressive) |
| RULEX (Rule) | 0–1 | 0–1 (rigid) |
| SUSTAIN (Clustering) | 0 (stable) | 2–4 (progressive) |

Correct theories do not revise — their predictions already match the data. Incorrect theories revise progressively (adapting parameters, adjusting claims) but never degeneratively. This is a Lakatos-compatible outcome. RULEX is notably revision-resistant even when wrong, consistent with its rigid rule-based structure having fewer free parameters.

### 3.10 Interpretation debate quality

We audited all 30 debate cycles across 6 runs on four dimensions:

| Dimension | Rating | Finding |
|---|---|---|
| Data citation accuracy | Weak | Agents cite posterior probabilities but rarely reference item-level predictions, RMSE values, or learning curve shapes |
| Critique quality | Mixed | Structurally substantive (cite mechanisms, name parameters) but numerically ungrounded ("model flexibility allows post-hoc fitting" without specifying which parameters diverge) |
| Behavioral adaptation | Limited | Same 2–3 talking points repeat across all 5 cycles within a run; no cumulative learning from prior data |
| Novel structure rationale | Poor | Proposals not rooted in actual model divergence; duplicate existing structures with condition permutations |

The adversarial critique forcing function does produce improvement in later cycles (more specific proposals after 3+ cycles of critique pressure), but the debate does not generate cumulative scientific reasoning.

### 3.11 M5: Debate feedback loops — debate now affects outcomes

After closing four broken feedback loops (Section 2.9), we re-ran validation with GPT-4o via Princeton (5 cycles, full_pool mode):

| Ground Truth | Winner | RMSE | Posterior | Correct? |
|---|---|---|---|---|
| GCM | Exemplar_Agent | 0.1836 | 1.0000 | Yes |
| SUSTAIN | Clustering_Agent | 0.2687 | 1.0000 | Yes |
| RULEX | Rule_Agent | 0.1580 | 1.0000 | Yes |

All 3 ground truths correctly identified, consistent with M4 results. The key new finding is replication variance.

### 3.12 Post-M5 replication: non-zero variance

Four GCM replication runs with identical settings:

| Run | Exemplar RMSE | Clustering RMSE | Rule RMSE |
|---|---|---|---|
| Initial | 0.1836 | 0.2123 | 0.3280 |
| Rep 1 | 0.1587 | 0.2424 | 0.3379 |
| Rep 2 | 0.1832 | 0.2571 | 0.3628 |
| Rep 3 | 0.2082 | 0.2590 | 0.3558 |
| **Std Dev** | **0.0177** | **0.0189** | **0.0153** |

Pre-M5, all replicates produced identical RMSE (Section 3.5). Post-M5, std ≈ 0.018. The mechanism: different LLM runs propose different parameter revisions during interpretation, and those revisions now persist into subsequent cycles via `sync_params_from_theory()`. The correct winner is preserved across all runs — the variance is within-winner, not winner-changing.

### 3.13 Critique-as-falsification: agents overclaim by 3–5×

Across 6 validation runs, `verify_prediction_claim()` checked agent assertions during the critique phase:

| Metric | Value |
|---|---|
| Total prediction claims checked | ~46 |
| FALSE CLAIMs (discrepancy > 0.1) | ~45 |
| Verified claims (discrepancy ≤ 0.1) | 1 |
| Typical claimed accuracy | 0.65–0.90 |
| Typical actual accuracy | 0.10–0.48 |

Agents systematically overclaim their model's performance. When an agent asserts "my model predicts 0.75 on Type I / high attention," the actual computed prediction is typically 0.10–0.18. The one verified claim: Rule_Agent predicted 0.600 on five_four / baseline; actual was 0.544 (within tolerance).

This 45:1 false-to-verified ratio quantifies the gap between LLM mechanistic intuition ("exemplars handle this well") and computational reality. Agents reason correctly about mechanisms but cannot estimate quantitative consequences.

### 3.14 Claim ledger and EIG weighting

The structured claim ledger accumulates ~3 claims per agent per cycle (from interpretation debate) plus critique claims. Most claims remain "untested" because they reference conditions not subsequently selected by EIG. The focus pair mechanism activates after cycle 0 but has modest impact in 5-cycle runs because the posterior typically collapses quickly. The RULEX run shows the most interesting trajectory: focus pair correctly identifies Exemplar–Rule as the contested pair during cycles 1–2, when discrimination matters most.

### 3.15 M6: ARBITER live validation — role-specialized agents and crux negotiation

M6 adds ARBITER-inspired architecture (meta-agents, crux negotiation, conflict maps, pre-registration) and validates with GPT-4o on all three ground truths, 5 cycles each.

| Ground Truth | Winner | RMSE | Gap% | Cruxes (accepted/total) | Claims (falsified/confirmed/untested) | Time |
|---|---|---|---|---|---|---|
| GCM | Exemplar_Agent | 0.1512 | 36.4% | 4/34 | 14/1/26 | 431s |
| SUSTAIN | Clustering_Agent | 0.2700 | 45.6% | 7/32 | 15/0/24 | 439s |
| RULEX | Rule_Agent | 0.1187 | 67.6% | 4/35 | 15/0/26 | 467s |

3/3 correct, all decisive margins.

### 3.16 The system as a falsification engine

The claim ledger reveals a striking asymmetry: across all three M6 runs, 44 claims were falsified, 1 confirmed, and 76 remain untested. Agents make bold predictions during interpretation debate; experiments consistently disprove them; the Bayesian posterior accumulates evidence against wrong theories. Even the winning agent rarely makes predictions conservative enough to survive empirical test.

This is the debate protocol's emergent methodology: convergence occurs not by proving the winner right, but by proving the losers wrong. The one confirmed claim (Rule_Agent predicting mean_accuracy=0.600 on Type_IV/low_attention, actual 0.500) is the exception that proves the rule.

### 3.17 Crux negotiation is selective with real LLMs

Approximately 100 cruxes were proposed across all three M6 runs; only 15 were accepted (15% acceptance rate). In mock validation with deterministic responses, acceptance was 100% — agents rubber-stamped every crux. Real GPT-4o agents reject proposals they find unpersuasive, producing genuine selectivity.

Accepted cruxes cluster around real theoretical fault lines: "Do people store individual exemplars or use abstract rules?" (exemplar vs rule debate), "The role of presentation order in category learning" (relevant to SUSTAIN's order-sensitivity), "The necessity of cluster recruitment in learning complex structures" (SUSTAIN's core mechanism). These are precisely the questions that cognitive scientists disagree about.

However, the crux→experiment pipeline is loose. Crux boost multiplies EIG for matching candidates, but when EIG≈0 due to posterior collapse, there is nothing to boost. The crux mechanism's full potential is constrained by the posterior collapse problem.

### 3.18 Posterior collapse: the primary architectural bottleneck

Two of three runs (GCM, SUSTAIN) achieve posterior certainty (P≈1.0) after cycle 0. Once the posterior collapses, EIG=0 for all candidates, making cycles 2–4 uninformative regardless of crux boost. The system spends most of its compute budget running experiments that cannot change the outcome.

RULEX is the exception and the most scientifically interesting case. The posterior initially favors Exemplar_Agent after the five_four experiment (cycle 0), then self-corrects by cycle 2 once Type_I structures provide disambiguating evidence. This non-monotonic trajectory demonstrates the system's capacity for self-correction — but only when posterior uncertainty survives long enough for structural variation to take effect.

### 3.19 Winning theories need fewer revisions

| Ground Truth | Winner's Revisions | Losers' Total Revisions |
|---|---|---|
| GCM | 2 (Exemplar_Agent) | 5 |
| SUSTAIN | 1 (Clustering_Agent) | 6 |
| RULEX | 0 (Rule_Agent) | 5 |

Theories aligned with ground truth require less parameter adjustment. In the RULEX run, Rule_Agent made zero revisions and won by 67.6%, while Clustering_Agent made 3 futile revisions trying to accommodate evidence it could not explain. This is Lakatos's criterion for progressive vs degenerative research programs, emerging naturally from the adversarial structure: robust theoretical cores resist falsification without auxiliary adjustments, while misaligned theories revise progressively but cannot close the gap.

### 3.20 Meta-agent contributions

Each M6 run produced 10 meta-agent responses (5 Integrator, 5 Critic, one per cycle). The Integrator synthesized across all three theory agents' responses, identifying areas of convergence and divergence. The Critic consistently identified the weakest argument among theory agents — typically challenging agents whose posterior probability had collapsed to zero but who continued making strong claims.

Meta-agents did not override the Bayesian machinery. Their primary value is qualitative: they structure the debate's narrative, making human review more efficient by highlighting the key tensions and weaknesses. This is the intended division of labor — computation for quantitative adjudication, LLMs for semantic synthesis.

---

## 4. Discussion

### 4.1 The architecture thesis: computation for selection, LLMs for interpretation

The central finding is a division of labor. Bayesian EIG experiment selection outperforms LLM-proposed experiments because it searches the full candidate space mathematically, while agents select experiments that tell good stories. The legacy RULEX run never tested Type_I (the single structure where RULEX dominates GCM) — agents proposed narratively familiar structures instead.

However, LLMs contribute genuine interpretive value. Agent reasoning correctly identifies mechanisms ("GCM's attention weights concentrate on the diagnostic dimension"), connects predictions to theory ("SUSTAIN predicts order effects through cluster recruitment"), and produces human-readable explanations of Bayesian results. The LLM's qualitative reasoning was sound even when its numerical predictions were wrong (Phase 2 vs Phase 3).

**Recommendation:** Use formal computation for experiment selection and evidence accumulation. Use LLMs for interpretation, hypothesis generation, and human-facing explanation.

### 4.2 Learning curves as the critical discriminator

GCM and RULEX produce similar final accuracies across most structures because GCM can approximate rule-like behavior through attention weights (Nosofsky 1991). This is a genuine property of GCM's flexibility, not a system bug. Final accuracy alone cannot distinguish them.

Learning curves break the tie because the models predict qualitatively different dynamics: GCM gradual, RULEX sudden, SUSTAIN stepwise. The curve evidence (weighted at 0.5× accuracy in the Bayesian update) increased the RULEX discrimination gap from 2.4% to 68% — the single largest improvement in the project.

**Implication for cognitive science:** Empirical studies that rely only on endpoint accuracy may systematically underestimate model differences. Learning dynamics carry diagnostic information orthogonal to final performance.

### 4.3 The specification gap

The most fundamental bottleneck was translating between LLM reasoning and computational specification. Agents produced scientifically sophisticated experiment designs ("non-linearly-separable categories with family resemblance structure, 3 conditions, 120 subjects") that the computational backend could not execute. The backend needed `{"stimuli": np.array, "labels": np.array}`.

The solution was a constrained menu (structure registry + condition effects) that preserves agent choice while ensuring executability. This is likely a general challenge for LLM-in-the-loop scientific systems: the translation layer between natural language and formal specification must be designed explicitly.

### 4.4 What debate does and doesn't do

**What debate contributes:**
- Adversarial critique as a forcing function — pressures agents to refine proposals in later cycles
- Theory revision pressure — incorrect theories accommodate evidence progressively (Lakatos-compatible)
- Human-readable mechanistic explanations of model behavior
- Parameter revisions that causally affect subsequent predictions (M5 — replication std=0.018)
- Crux negotiation that identifies genuine theoretical fault lines (M6 — 15% acceptance rate, accepted cruxes map to real scientific disagreements)
- Role-specialized synthesis: Integrator identifies convergence, Critic identifies weakest arguments (M6)
- Falsification as an emergent methodology: 44:1 falsified-to-confirmed ratio (M6)

**What debate does not contribute:**
- Experiment selection quality (EIG dominates; LLM proposals are narrative-driven)
- Cumulative scientific reasoning (agents repeat talking points across cycles despite the claim ledger)
- Data-grounded argumentation (posteriors cited as proxy; item-level data ignored)
- Calibrated quantitative predictions (agents overclaim accuracy by 3–5× when fact-checked)
- Overcoming posterior collapse (crux boost is active but powerless when EIG=0)

Pre-M5, the debate was entirely epiphenomenal to RMSE — replication variance was zero, and convergence was driven by the Bayesian machinery alone. Post-M5, parameter revision persistence creates a modest but real causal link: different LLM runs produce different parameter revisions, which produce different model predictions. Post-M6, the ARBITER architecture enriches debate quality (cruxes, conflict maps, meta-agents) but does not fundamentally alter the convergence mechanism, which remains Bayesian. The debate's primary value is qualitative — mechanistic narratives, structured disagreement, and human-readable explanations — augmented by M6's role specialization and crux-driven focus.

### 4.5 Posterior collapse as the primary bottleneck

M6 validation reveals a structural problem: the Bayesian posterior collapses to certainty after 1–2 experiments, leaving remaining cycles with EIG≈0. This renders crux boost, meta-agent guidance, and later debate cycles uninformative. The posterior concentration is correct — the first five_four experiment often provides overwhelming evidence — but it eliminates the system's ability to explore structural questions raised by crux negotiation.

The RULEX run demonstrates what happens when collapse is delayed: the posterior initially favors the wrong model, then self-corrects when structural variation provides disambiguating evidence. This non-monotonic trajectory is the system's most scientifically valuable behavior, and it only occurs when uncertainty survives long enough for the debate to matter.

Potential solutions: posterior tempering (prevent collapse by raising log-probs to a power <1), entropy-based re-exploration (force untested structures when entropy is low), or crux-driven overrides (run a crux's discriminating experiment regardless of EIG).

### 4.6 Limitations

1. **Posterior collapse.** The most urgent limitation. EIG≈0 after cycle 0–1 in 2 of 3 ground truths makes later cycles uninformative. Crux negotiation identifies decisive questions, but the posterior is already certain, so there's nothing to decide. This limits the debate to 1–2 genuinely informative cycles despite running 5.

2. **Modest debate impact.** Post-M5, replication variance is non-zero but small (std≈0.018 on RMSE≈0.18, ~10% coefficient of variation). Post-M6, ARBITER features enrich debate quality but do not fundamentally alter convergence. The Bayesian machinery still dominates.

3. **Synthetic data only.** The framework validates whether correct models are identifiable in principle, not whether the models are correct accounts of human behavior. Extending to real experimental data would require a lab-automation interface.

4. **Three models only.** The framework currently implements GCM, SUSTAIN, and RULEX. Generalization to other model families (neural networks, Bayesian cognitive models) is architecturally straightforward but untested.

5. **LLM-agnostic convergence.** Cross-LLM comparison (Section 3.6) shows correct model wins regardless of backbone, which validates robustness but also suggests the LLM is currently a replaceable component. The debate quality differences between GPT-4o, Sonnet, and Opus do not translate into convergence differences.

6. **No human evaluation of debate quality.** Our quality audit was systematic but not blind. Expert evaluation of whether agent reasoning constitutes genuine scientific reasoning would strengthen the findings.

7. **Claim ledger underutilized.** Agents don't spontaneously engage with the claim ledger or conflict map in their interpretations despite injection into prompts. The 44:1 falsification ratio suggests agents make bold claims but don't learn from falsification.

### 4.7 Future directions

1. **Address posterior collapse** — posterior tempering, entropy-based re-exploration, or multi-hypothesis tracking to keep later cycles informative
2. **Claim-responsive debate** — agents should explicitly address their prior claims ("I previously predicted X, which was falsified; I now revise to Y") rather than repeating generic talking points
3. **Longer runs (10+ cycles)** — assess whether novel structures eventually outperform registry structures as the registry space is exhausted, and whether the claim ledger produces cumulative reasoning at longer horizons
4. **Cross-domain generalization** — apply the framework to other multi-model disputes in cognitive science (memory models, decision-making theories)
5. **Real data integration** — AutoRA + Prolific for closing the loop with human participants
6. **Crux-driven experiment override** — when accepted cruxes exist but EIG=0, bypass Bayesian selection and run the crux's discriminating experiment directly

---

## 5. Conclusion

Antagonistic collaboration via LLM debate can successfully identify the correct model from competing theories. The mechanism of convergence is primarily Bayesian computation, but M5's feedback loop closures demonstrate that debate can causally affect outcomes — parameter revisions proposed during interpretation now persist into subsequent predictions, producing non-zero replication variance for the first time. M6's ARBITER integration adds role-specialized meta-agents, crux-based negotiation, conflict maps, and pre-registration output, enriching debate quality while preserving correct convergence (3/3 ground truths, 36–68% gaps). The system operates as a falsification engine: 44 claims falsified vs 1 confirmed across all M6 runs. Crux negotiation is genuinely selective with real LLMs (15% acceptance rate), and winning theories require fewer parameter revisions than losing theories (Lakatos-compatible). The primary architectural bottleneck is posterior collapse — the Bayesian posterior concentrates too quickly, leaving later cycles uninformative despite active crux negotiation. The optimal architecture separates computation (experiment selection, posterior update, learning curves) from language (interpretation, hypothesis generation, explanation), but connects them through validated feedback paths (parameter persistence, claim verification, crux-driven EIG boosting). The framework demonstrates both the promise and the current limits of LLMs in the scientific method: they identify genuine theoretical fault lines through crux negotiation, synthesize across competing accounts through meta-agents, and produce human-readable mechanistic narratives — but they cannot yet learn cumulatively from evidence, calibrate their quantitative expectations, or overcome the system's tendency toward premature certainty.

---

## References

- Love, B. C., Medin, D. L., & Gureckis, T. M. (2004). SUSTAIN: A network model of category learning. *Psychological Review, 111*(2), 309–332.
- Medin, D. L., & Schaffer, M. M. (1978). Context theory of classification learning. *Psychological Review, 85*(3), 207–238.
- Mellers, B., Hertwig, R., & Kahneman, D. (2001). Do frequency representations eliminate conjunction effects? An exercise in adversarial collaboration. *Psychological Science, 12*(4), 269–275.
- Nosofsky, R. M. (1986). Attention, similarity, and the identification–categorization relationship. *Journal of Experimental Psychology: General, 115*(1), 39–57.
- Nosofsky, R. M. (1991). Tests of an exemplar model for relating perceptual classification and recognition memory. *Journal of Experimental Psychology: Human Perception and Performance, 17*(1), 3–27.
- Nosofsky, R. M., Palmeri, T. J., & McKinley, S. C. (1994). Rule-plus-exception model of classification learning. *Psychological Review, 101*(1), 53–79.
- Shepard, R. N., Hovland, C. I., & Jenkins, H. M. (1961). Learning and memorization of classifications. *Psychological Monographs: General and Applied, 75*(13), 1–42.
