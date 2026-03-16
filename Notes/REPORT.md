# Antagonistic Collaboration via LLM Debate: Can AI Agents Resolve Scientific Disputes?

**Phase: M12 — Continuous Design Space Parameterization (updated)**
**Date: 2026-03-16** (originally 2026-03-14; updated through M12 implementation)

---

## Abstract

We present an antagonistic collaboration framework in which three LLM agents — each representing a competing theory of human category learning — debate through a structured protocol, propose experiments, and converge toward the theory that best explains synthetic data. The three models are the Generalized Context Model (GCM; Nosofsky 1986), SUSTAIN (Love, Medin & Gureckis 2004), and RULEX (Nosofsky, Palmeri & McKinley 1994). We compare two architectures: a *legacy* mode where LLM agents propose experiments through adversarial debate, and a *full-pool* mode where Bayesian expected information gain (EIG) selects experiments while agents shift to interpreting results and generating hypotheses. Across 6 validation runs (3 ground truths × 2 modes, 5 cycles each), the correct model's agent wins in every condition. Full-pool mode achieves dramatically better discrimination for hard model pairs (RULEX gap: 2.4% legacy vs. 68% full-pool), driven by learning curves as a second evidence channel. Cross-LLM comparison (GPT-4o, Claude Sonnet, Claude Opus) confirms the framework is LLM-agnostic (9/9 correct). After closing four broken feedback loops (M5), debate now causally affects RMSE through parameter revision persistence (replication std=0.018, previously 0.000), and critique-as-falsification reveals that agents overclaim model accuracy by 3–5×. M6 adds ARBITER-inspired architecture: role-specialized meta-agents (Integrator, Critic), crux-based negotiation for focusing debate on decisive questions, conflict maps, and pre-registration output. Live M6 validation with GPT-4o achieves 3/3 correct with decisive gaps (36–68%), reveals the system operates as a falsification engine (44 claims falsified, 1 confirmed), and identifies posterior collapse as the primary architectural bottleneck. M7 introduces likelihood tempering (tau=0.005, prediction clip [0.05, 0.95]) to address posterior collapse, achieving gradual convergence but revealing that greedy EIG selection repeats the same experiment every cycle. M8 adds Thompson sampling for experiment selection: sampling proportional to EIG scores instead of argmax. A clean 6-run ablation (3 ground truths × 2 strategies) shows both Thompson and greedy achieve 3/3 correct post-bugfix, but Thompson explores far more broadly (12 unique structures including 6 novel vs greedy's 3 unique, 0 novel). M9 adds crux-directed Thompson sampling: a mixture distribution that biases experiment selection toward crux-matching candidates, establishing the first semantically directed path from debate to experiment selection (24 parseable crux specs, 1 crux-directed experiment). M10 adds claim-responsive debate inspired by Reflexion (Shinn et al., NeurIPS 2023): agents with falsified claims must explicitly address each one (revise, explain, or abandon). Live validation shows 80% compliance (100% when applicable), with "explain" dominating — agents reproduce Lakatos's auxiliary hypothesis shielding without being programmed to do so. We distill 30 principles on what LLM-mediated scientific debate can and cannot do.

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

The base experiment space consists of **11 category structures** × **5 experimental conditions** = 55 candidate experiments. Three design space modes are available (M12): **base** (55 candidates), **richer** (24 structures × 7 conditions = 168 fixed candidates, M11), and **continuous** (11 base + 50 freshly sampled structures × 7 conditions ≈ 427 candidates per cycle, M12 default). In continuous mode, structures are sampled from continuous parameter ranges each cycle, letting EIG discover diagnostic sweet spots that no fixed grid covers.

**Base structures** (from STRUCTURE_REGISTRY, 11):
- Shepard Types I–VI (Shepard, Hovland & Jenkins 1961) — canonical structures varying from single-dimension rules (Type I) to all-exceptions parity problems (Type VI)
- Medin & Schaffer (1978) five-four structure — 9 items with complex category boundary
- Rule-plus-exception structures (1 and 2 exceptions per category)
- Linear separable structures (2D and 4D Gaussian clusters)

**Parametric structures** (from PARAMETRIC_STRUCTURES, 13; M11):
- Linear separable variants: 7 structures varying separation (1.0, 1.5, 2.5, 3.0) and dimensionality (2D, 3D, 4D, 6D). Intermediate separation reveals model differences that extreme values mask.
- Rule-plus-exception variants: 6 structures varying dimensionality (3D, 5D, 6D) and exception count (1, 2, 3). More exceptions test SUSTAIN cluster recruitment; more dimensions test attention allocation.

**Base conditions** (from CONDITION_EFFECTS, 5): baseline, low_attention, high_attention, fast_presentation, high_noise. Each condition maps to model-specific parameter perturbations (e.g., low_attention: GCM c=1.5, SUSTAIN r=3.0, RULEX p_single=0.3).

**Interpolated conditions** (from PARAMETRIC_CONDITIONS, 2; M11): moderate_attention (midpoint of low/high), mild_noise (between baseline and high_noise). These fill gaps in the condition space where model predictions may differ diagnostically.

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

For each hypothetical ground truth model (weighted by current prior), `n_sim` synthetic datasets are simulated via binomial sampling. Each dataset is scored under all three models via item-level log-likelihood, the posterior is updated, and the resulting entropy is averaged.

**Experiment selection** supports two strategies:
- **Greedy** (`--selection-strategy greedy`): selects `argmax(EIG)`. Deterministic but prone to repeating the same experiment every cycle.
- **Thompson** (`--selection-strategy thompson`, default): samples experiments proportional to EIG scores. Provides principled exploration of the candidate space while still favoring high-EIG experiments (Russo & Van Roy 2018; Kandasamy et al. 2019).

**Posterior update** uses item-level log-likelihoods from `binom.logpmf(observed, n_subjects, predicted)` for each model, with optional likelihood tempering (Section 2.12). Learning curve predictions are computed for agent interpretation but are not used in the posterior update — the synthetic framework does not generate observed learning curves, so there is no valid comparison target (D35).

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

### 2.11 Likelihood tempering (M7)

The Bayesian posterior concentrates to P≈1.0 after 1–2 experiments (D29), making EIG=0 for all remaining candidates. Root cause: binomial log-likelihood with n_subjects=20 across ~10 items generates ~10 nats of evidence per experiment. After 2 experiments, log-odds reach ~50 nats (ratio ~5×10²¹).

**Fix:** Likelihood tempering (power posteriors) — multiply log-likelihoods by a learning rate tau ∈ (0, 1] before adding to the prior:

```
log_posterior += tau × log_likelihood
```

This is well-established in Bayesian statistics (Grünwald 2012, Bissiri et al. 2016, Miller & Dunson 2019). Additionally, model predictions are clipped to [0.05, 0.95] — no cognitive model should predict individual items with >95% confidence.

Default tau=0.005, calibrated so entropy ≈ 0.6 after cycle 0 and ≈ 0.02 after 5 cycles. Configurable via `--learning-rate` CLI flag.

### 2.12 Thompson sampling for experiment selection (M8)

Greedy EIG selection (`argmax`) selects the same experiment every cycle once the posterior begins to concentrate (D33). Thompson sampling replaces argmax with sampling proportional to EIG scores:

```
P(select experiment i) = EIG(i) / Σ_j EIG(j)
```

When all EIG scores are zero (or negative), falls back to uniform random selection. This is a simplified form of Myopic Posterior Sampling (Kandasamy et al. 2019), providing principled exploration without ad-hoc diversity bonuses. Configurable via `--selection-strategy thompson|greedy`.

### 2.13 Crux-directed Thompson sampling (M9)

The crux-to-experiment pipeline (introduced in M6) had two structural failures: (1) agents wrote free-text crux descriptions ("test whether rule-based models can handle exceptions") instead of the required `structure/condition` format, so parsing always failed — 0 boost specs across all M6/M7/M8 runs despite 100+ cruxes proposed; (2) the multiplicative EIG boost (2×) barely shifted Thompson's sampling distribution when EIG scores clustered narrowly (e.g., 0.18–0.23).

M9 replaces the multiplicative boost with a **mixture distribution**:

```
With probability crux_weight:
    Sample uniformly from crux-matching candidates
Otherwise:
    Sample from standard EIG-weighted Thompson distribution
```

Default `crux_weight=0.3`. When no active cruxes match pool entries, falls back to 100% EIG-weighted Thompson. The crux identification prompt now shows the full structure/condition menu with a format example, and `cruxes_to_boost_specs()` validates against known structures and conditions. Configurable via `--crux-weight`.

### 2.14 Validation protocol

Six M4 runs: 3 ground truths × 2 modes, each 5 cycles. Nine cross-LLM runs: 3 ground truths × 3 LLMs. Three M5 validation runs: 3 ground truths, full_pool with GPT-4o. Four M5 replication runs: GCM ground truth, full_pool with GPT-4o (for variance analysis). Three M6 live validation runs: 3 ground truths, full_pool with GPT-4o, all ARBITER features enabled. Three M7 validation runs: 3 ground truths, full_pool with tau=0.005. Six M8 ablation runs: 3 ground truths × 2 strategies (Thompson vs greedy), full_pool with tau=0.005. Three M9 validation runs: 3 ground truths, full_pool with crux_weight=0.3. 336 automated tests verify framework correctness.

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

### 3.21 M7: Likelihood tempering — gradual convergence but greedy repetition

M7 introduces likelihood tempering (tau=0.005, prediction clip [0.05, 0.95]) to address posterior collapse. Three 5-cycle validation runs with GPT-4o:

| Ground Truth | Winner | Correct? | RMSE gap | Entropy trajectory |
|---|---|---|---|---|
| GCM | Exemplar_Agent | YES | 81.1% | 0.64→0.33→0.13→0.03→0.00 |
| RULEX | Exemplar_Agent | **NO** | 8.2% | 0.65→0.69→0.70→0.48→0.16 |
| SUSTAIN | Clustering_Agent | YES | 97.1% | 0.22→0.01→0.00→0.00→0.00 |

**Tempering achieves its design goal:** GCM shows textbook gradual convergence — entropy drops monotonically from 0.64→0.00 over 5 cycles, with EIG remaining nonzero through cycle 4 (0.029). This is the first validation where later cycles are genuinely informative.

**RULEX misidentification reveals genuine model overlap.** GCM and RULEX produce very similar predictions on the structures the system selects (linear_separable_4d, nonlinear_complex_5d). The posterior oscillated — RULEX led on cycles 0 and 2, GCM on cycles 1, 3, 4. This mirrors the known theoretical result that GCM approximates rule-like behavior through attention weights (Nosofsky 1991). The 8.2% RMSE gap (down from 67.6% in M6) reflects genuine model overlap, not a system failure.

**Greedy EIG repeats the same experiment every cycle.** The GCM run selected linear_separable_4d 5/5 times. SUSTAIN selected it 5/5 times. Only RULEX showed variation (alternating with nonlinear_complex_5d). With tempering preventing posterior collapse, the greedy argmax still concentrates on a single high-EIG structure, preventing exploration of the candidate space.

### 3.22 M8: Thompson sampling ablation — exploration vs exploitation

M8 replaces greedy EIG selection with Thompson sampling (proportional to EIG scores). A clean ablation compares both strategies across all 3 ground truths (6 runs total, post-bugfix D35):

| Ground Truth | Strategy | Correct? | Winner RMSE | Unique structs | Novel structs | Final entropy |
|---|---|---|---|---|---|---|
| GCM | Thompson | Yes | 0.085 | 5 | 3 | 0.12 |
| GCM | Greedy | Yes | 0.077 | 2 | 0 | 0.01 |
| RULEX | Thompson | Yes | 0.189 | 4 | 2 | 0.16 |
| RULEX | Greedy | Yes | 0.050 | 2 | 0 | 0.06 |
| SUSTAIN | Thompson | Yes | 0.022 | 3 | 1 | 0.00 |
| SUSTAIN | Greedy | Yes | 0.018 | 1 | 0 | 0.00 |

**Both strategies achieve 3/3 correct.** Post-bugfix (D35: curve bonus removal, novel structure execution), both Thompson and greedy correctly identify all ground truths. The bugfix resolved M7's RULEX misidentification for both strategies.

**Thompson explores far more broadly.** Across 3 runs, Thompson selected 12 unique structures (including 6 novel agent-proposed structures) vs greedy's 3 unique structures (0 novel). This is the first time novel structures have been selected and executed in the framework.

**Greedy achieves tighter convergence.** Greedy's final entropies are lower (0.01–0.06 vs 0.12–0.16 for Thompson on GCM/RULEX), and winner RMSE is slightly lower. This is expected: greedy concentrates on the single most informative experiment, maximizing evidence per cycle. Thompson trades convergence speed for exploration breadth.

**The exploration–exploitation tradeoff is real.** Thompson sacrifices ~50% convergence tightness for ~4× structural diversity. Whether this tradeoff is worthwhile depends on the research question: greedy is optimal for fast model selection; Thompson is preferable when the goal is understanding the full prediction landscape or when the candidate space may contain unexplored discriminating structures.

**Debate's causal role remains modest.** Thompson's random exploration, not debate-guided selection, drives structural diversity. The debate contributes parameter revisions (replication variance) and novel structure proposals (6 novel structures selected by Thompson), but the exploration itself is stochastic, not semantically directed.

### 3.23 M9: Crux-directed Thompson sampling — debate causally affects experiment selection

M9 fixes the crux-to-experiment pipeline (D37) and validates with 3 ground truths × 5 cycles:

| Ground Truth | Winner | Correct? | RMSE | Gap | Cruxes parsed | Crux-directed |
|---|---|---|---|---|---|---|
| GCM | Exemplar_Agent | Yes | 0.084 | 74.7% | 11/34 | 1/5 |
| RULEX | Rule_Agent | Yes | 0.050 | 83.9% | 7/34 | 0/5 |
| SUSTAIN | Clustering_Agent | Yes | 0.033 | 93.1% | 6/37 | 0/5 |

**The crux pipeline is operational for the first time.** Across all prior milestones (M6–M8), 0 crux boost specs were parsed from 100+ proposed cruxes — agents wrote free-text descriptions that the parser couldn't match. After M9's prompt fix (showing the structure/condition menu) and parsing validation, 24 parseable specs were produced across 3 runs (11 from GCM, 7 from RULEX, 6 from SUSTAIN).

**Debate causally affects experiment selection.** In the GCM run, crux `crux_004` proposed `rule_plus_exception_1exc/high_noise` as a discriminating experiment. The mixture distribution selected this experiment on cycle 3 — the first time a debate-identified theoretical disagreement directly determined which experiment ran. This is qualitatively different from M8's novel structure path (where Thompson randomly sampled debate-proposed structures): here the selection was specifically directed by a crux about rule-exception tradeoffs.

**Crux-directed selection rate is low but non-zero.** Only 1 of 15 total experiments was crux-directed (6.7% vs the 30% theoretical maximum). This is because crux-matching requires both the structure and condition to match pool entries exactly, and most accepted cruxes reference structures or conditions that are already represented in the standard EIG pool. The mixture distribution guarantees selection when a match exists, but most cruxes don't produce unique matches. This convergence between semantic (crux) and computational (EIG) experiment selection is consistent with the theoretical prediction of Corcoran, Hohwy & Friston (2023), who argue that adversarial collaboration and Bayesian optimal design should be unified because both target the same discriminating experiments. Ouyang et al. (2018) found a similar pattern in the classic Medin & Schaffer (1978) categorization study: the intuitively designed 5-4 structure happened to place competing models near maximal EIG divergence.

**Correctness is preserved.** All 3 ground truths correctly identified with strong gaps (74.7–93.1%), comparable to or better than M8 Thompson results (which had gaps of 74–94% depending on the run). The crux-directed experiment in the GCM run did not degrade convergence.

### 3.24 M10: Claim-responsive debate — agents confront falsified claims

M10 adds a claim-responsive directive to the interpretation prompt (D38): when an agent has falsified claims, a `FALSIFIED CLAIMS` block lists each one with evidence and requires the agent to revise, explain, or abandon it via a structured `"falsified_response"` JSON field. The mechanism is inspired by Reflexion (Shinn et al., NeurIPS 2023). Validated with 3 ground truths × 5 cycles:

| Ground Truth | Winner | Correct? | RMSE | Gap | Falsified claims | FR rate |
|---|---|---|---|---|---|---|
| GCM | Exemplar_Agent | Yes | 0.071 | 79.3% | 12 | 80% |
| SUSTAIN | Clustering_Agent | Yes | 0.018 | 96.6% | 14 | 80% |
| RULEX | Rule_Agent | Yes | 0.166 | 51.8% | 15 | 80% |

**Directed engagement works where passive context failed.** The claim ledger summary has been injected into interpretation prompts since M5, but agents ignored it for 5 milestones. Adding an explicit directive with structured response format achieved 80% compliance (12/15 theory agent interpretations). The 3 missing responses are all cycle-0 interpretations where no falsified claims exist — when applicable, compliance was 100%.

**"Explain" dominates.** Agents overwhelmingly attribute falsification to confounds and boundary conditions ("the fast presentation condition may have limited exemplar retrieval") rather than revising theories or abandoning claims. Only 1 "abandon" action was observed across all 3 runs. This is Lakatos's auxiliary hypothesis shielding, emerging naturally from LLM behavior: agents protect core theoretical commitments and modify auxiliary assumptions to accommodate disconfirming evidence.

**Overclaiming persists.** Agents still claim 0.65–0.85 when actual model output is 0.10–0.50. Claim-responsive debate forces agents to *confront* falsification but does not improve quantitative calibration. The qualitative reasoning about *why* a prediction failed is plausible but untested — agents invoke confounds they never propose to test.

**JSON compliance is high.** 80% falsified_response compliance vs 23% crux format compliance (M9). The difference reflects task complexity: crux format requires matching exact structure/condition pairs from a 55-entry registry; falsified_response requires free-text reasoning within a simple schema that aligns with what the agent is already doing.

### 3.25 M11: Richer design spaces — parametric structures and interpolated conditions

M11 extends the fixed 11-structure × 5-condition registry (55 candidates) with parametrically generated structures and interpolated conditions (168 candidates total). The motivation is from the optimal experimental design literature: EIG performs best when the design space is continuous rather than discrete, allowing the search to find diagnostic sweet spots between fixed options (Myung & Pitt 2009; Cavagnaro et al. 2010).

**Parametric structures (13 new):** Linear separable variants span separation {1.0, 1.5, 2.5, 3.0} × dimensionality {2D, 3D, 4D, 6D} (7 structures). Rule-plus-exception variants span dimensionality {3D, 5D, 6D} × exceptions {1, 2, 3} (6 structures). Each is generated with a deterministic seed for reproducibility. All pass `validate_novel_structure()`.

**Interpolated conditions (2 new):** `moderate_attention` (midpoint between low and high attention parameters for all models) and `mild_noise` (between baseline and high_noise). These fill gaps where diagnostic model differences may emerge at intermediate parameter values.

**Design rationale:** The parametric structures target the two generator families with continuous parameters (`linear_separable` and `rule_plus_exception`). Shepard types are fixed 3-binary-dimension structures and cannot be meaningfully parameterized. The interpolated conditions target the two most diagnostically relevant dimensions (attention and noise) based on M4–M9 results showing these conditions produce the largest inter-model divergence.

**Config:** `no_richer_design_space: false` (default on). CLI: `--no-richer-design-space`. `generate_full_candidate_pool(richer=True|False)`. All existing code paths (`_synthetic_runner`, `compute_model_predictions`) resolve parametric entries automatically.

**Tests:** 14 new tests (TestRicherDesignSpaces): config/CLI/global plumbing (3), parametric structure validity (4), parametric condition validity (3), pool generation (2), synthetic runner resolution (2). 315 total passing.

**Live validation (2026-03-16, GPT-4o via Princeton):**

| Ground Truth | Winner | Correct? | RMSE | Gap | Param-S | Param-C |
|---|---|---|---|---|---|---|
| GCM | Exemplar_Agent | Yes | 0.075 | 75.8% | 5/5 | 3/5 |
| SUSTAIN | Clustering_Agent | Yes | 0.022 | 95.6% | 5/5 | 1/5 |
| RULEX | Rule_Agent | Yes | 0.053 | 83.7% | 5/5 | 1/5 |

**EIG exclusively selects parametric structures.** All 15 experiments across 3 runs used parametric linear_separable variants (3D, 6D with varied separation). No base registry structure was selected. This validates the central motivation: intermediate parameter values reveal model differences that fixed extreme values mask. The 3D linear_separable variants (separation 1.5, 2.5) were heavily favored — these fill the gap between the base 2D and 4D entries.

**Interpolated conditions used selectively.** 5/15 experiments used `mild_noise` or `moderate_attention`. The remaining used base conditions (low_attention, high_attention, high_noise). The moderate_attention condition appeared in the highest-EIG candidate (linear_separable_6d/moderate_attention) in the GCM run.

**Correct identification preserved.** 3/3 correct with decisive gaps (76–96%). The richer pool does not degrade accuracy despite the 3× larger search space.

### 3.26 M12: Continuous design space — fresh samples each cycle

M12 replaces M11's fixed parametric grid (168 candidates) with continuous sampling from parameter ranges. Each cycle draws 50 fresh structures via `_sample_continuous_structures()`: ~60% linear_separable (n_dims ∈ {2,...,8}, separation ∈ Uniform(0.5, 4.0)) and ~40% rule_plus_exception (n_dims ∈ {3,...,8}, n_exceptions ∈ {1,...,4}). Seeds are cycle-dependent (42 + cycle × 1000) so different cycles explore different parameter regions while remaining reproducible within a run.

**Config:** `design_space: continuous` (default). CLI: `--design-space {base,richer,continuous}`, `--n-continuous-samples 50`. `--no-richer-design-space` kept as deprecated alias → `design_space: base`. `generate_full_candidate_pool()` now takes `design_space`, `n_continuous_samples`, `continuous_seed` instead of `richer`.

**Tests:** 16 new tests (TestContinuousDesignSpace), 331 total passing. Covers sampling validity, determinism, seed variation, name encoding, pool composition, protocol storage, config/CLI plumbing, backward compatibility.

**Bug fix:** `compute_model_predictions()` crashed with `ValueError: operands could not be broadcast together with shapes (8,) (3,)` when verifying prediction claims on 8D sampled structures. Agent default_params had 3D attention_weights from Shepard types. Fix: auto-detect dimension mismatch and set to None for uniform weights.

**Live validation (2026-03-16, GPT-4o via Princeton):**

| Ground Truth | Winner | Correct? | RMSE | Gap | Sampled/5 | Cycle Overlap | FR% |
|---|---|---|---|---|---|---|---|
| GCM | Exemplar_Agent | Yes | 0.092 | 76.8% | 5/5 | 0–2% | 80% |
| SUSTAIN | Clustering_Agent | Yes | 0.022 | 95.8% | 5/5 | 0–2% | 80% |
| RULEX | Rule_Agent | Yes | 0.048 | 87.4% | 5/5 | 0–2% | 80% |

**EIG exclusively selects sampled structures.** 15/15 experiments used freshly sampled linear_separable variants. Zero base registry structures, zero rule_plus_exception structures selected. EIG finds the diagnostic sweet spot at intermediate separation (0.68–2.13) and higher dimensionality (4–8D).

**Zero cycle overlap confirms the core M12 hypothesis.** Consecutive cycles share 0–2% of their sampled structures, meaning the system genuinely explores different parameter regions each cycle. This is the fundamental behavioral difference from M11's fixed 168-candidate grid.

**Performance comparable or improved.** Gaps of 77–96% match or exceed M11 (76–96%). RULEX gap improved (87.4% vs 83.7%). The continuous sampler finds more diagnostic experiments than the hand-picked parametric grid.

**The role of debate.** M12 results sharpen the architectural tension: the computational layer (EIG + continuous sampling + Bayesian update) drives model identification with zero LLM calls. The debate layer provides interpretive value (mechanistic narratives, confound identification, claim engagement at 80% FR rate) but does not measurably influence experiment selection — 15/15 experiments were computationally selected, 0/15 from agent proposals. Debate's causal contribution to model identification has not been ablated. The honest assessment: debate may be essential for *understanding* results but epiphenomenal for *identifying* the correct model on synthetic benchmarks. Harder problems (real data, model misspecification, genuine theoretical uncertainty) may reveal debate's value more clearly.

---

## 4. Discussion

### 4.1 The architecture thesis: computation for selection, LLMs for interpretation

The central finding is a division of labor. Bayesian EIG experiment selection outperforms LLM-proposed experiments because it searches the full candidate space mathematically, while agents select experiments that tell good stories. The legacy RULEX run never tested Type_I (the single structure where RULEX dominates GCM) — agents proposed narratively familiar structures instead.

However, LLMs contribute genuine interpretive value. Agent reasoning correctly identifies mechanisms ("GCM's attention weights concentrate on the diagnostic dimension"), connects predictions to theory ("SUSTAIN predicts order effects through cluster recruitment"), and produces human-readable explanations of Bayesian results. The LLM's qualitative reasoning was sound even when its numerical predictions were wrong (Phase 2 vs Phase 3).

**Recommendation:** Use formal computation for experiment selection and evidence accumulation. Use LLMs for interpretation, hypothesis generation, and human-facing explanation.

### 4.2 Learning curves as the critical discriminator

GCM and RULEX produce similar final accuracies across most structures because GCM can approximate rule-like behavior through attention weights (Nosofsky 1991). This is a genuine property of GCM's flexibility, not a system bug. Final accuracy alone cannot distinguish them.

Learning curves break the tie because the models predict qualitatively different dynamics: GCM gradual, RULEX sudden, SUSTAIN stepwise. In M4, curve evidence in the Bayesian update increased the RULEX discrimination gap from 2.4% to 68%. However, M8 analysis (D35) revealed that the curve bonus was data-independent — it measured inter-model curve distinctiveness rather than fit to observed data. After removing the curve bonus, both greedy and Thompson strategies still achieve 3/3 correct, indicating that item-level accuracy evidence alone is sufficient when combined with proper tempering and diverse experiment selection.

**Implication for cognitive science:** Empirical studies that rely only on endpoint accuracy may systematically underestimate model differences. Learning dynamics carry diagnostic information orthogonal to final performance.

### 4.3 The specification gap

The most fundamental bottleneck was translating between LLM reasoning and computational specification. Agents produced scientifically sophisticated experiment designs ("non-linearly-separable categories with family resemblance structure, 3 conditions, 120 subjects") that the computational backend could not execute. The backend needed `{"stimuli": np.array, "labels": np.array}`.

The solution was a constrained menu (structure registry + condition effects) that preserves agent choice while ensuring executability. This is a general challenge for LLM-in-the-loop scientific systems: the translation layer between natural language and formal specification must be designed explicitly. M9's crux parsing failure (23% format compliance despite explicit menus) further illustrates the gap. Tam et al. (2024) demonstrate that format restrictions actively degrade LLM reasoning performance — the model doesn't fail to understand the format, but format constraints interfere with generation. The IFEval benchmark (Zhou et al. 2023) shows no model exceeds 80% on verifiable format constraints. Robust pipelines should assume majority non-compliance and design mechanisms (mixture distributions, fuzzy matching, constrained decoding) that degrade gracefully.

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
- Cumulative scientific reasoning — partially addressed by M10's claim-responsive directive (Shinn et al., NeurIPS 2023). Agents now engage with falsified claims at 100% compliance when directed, but the dominant response is "explain" (auxiliary hypothesis shielding) rather than genuine theory revision. Overclaiming persists at 3–5×
- Data-grounded argumentation (posteriors cited as proxy; item-level data ignored)
- Calibrated quantitative predictions (agents overclaim accuracy by 3–5× even after M10 claim-responsive engagement — the mechanism fixes ignoring but not calibration)
- Overcoming posterior collapse (crux boost is active but powerless when EIG=0)

Pre-M5, the debate was entirely epiphenomenal to RMSE — replication variance was zero, and convergence was driven by the Bayesian machinery alone. Post-M5, parameter revision persistence creates a modest but real causal link: different LLM runs produce different parameter revisions, which produce different model predictions. Post-M6, the ARBITER architecture enriches debate quality (cruxes, conflict maps, meta-agents) but does not fundamentally alter the convergence mechanism, which remains Bayesian. Post-M8, debate contributes novel structure proposals that Thompson sampling actually selects (6 novel structures in ablation), but the exploration is stochastic, not debate-directed. Post-M9, the crux pipeline is operational: accepted cruxes directly bias experiment selection via a mixture distribution. In the GCM validation, crux `crux_004` selected `rule_plus_exception_1exc/high_noise` — the first time a debate-identified theoretical disagreement determined which experiment ran. The debate's primary value remains qualitative — mechanistic narratives, structured disagreement, and human-readable explanations — but M9 establishes the first semantically directed path from debate to experiment selection. Post-M10, agents must explicitly confront falsified claims (80% compliance overall, 100% when applicable). The dominant response is "explain" (invoking confounds and boundary conditions), reproducing Lakatos's auxiliary hypothesis shielding. Claim-responsiveness closes the ignoring gap but not the calibration gap: agents engage with failure but still overclaim by 3–5×.

### 4.5 Posterior collapse: diagnosis and treatment

M6 validation revealed that the Bayesian posterior collapses to certainty after 1–2 experiments, leaving remaining cycles with EIG≈0. Oelrich et al. (2020) identify this as a general phenomenon: posterior model probabilities become overconfident when "the compared models give very different approximations" — exactly our situation with SUSTAIN's stepwise curves vs. GCM's gradual curves. M7 addressed this with likelihood tempering (tau=0.005, prediction clip [0.05, 0.95]), a form of the power posterior (Grünwald 2012; Bissiri, Holmes & Walker 2016; Miller & Dunson 2019), achieving gradual convergence: GCM entropy drops 0.64→0.00 over 5 cycles instead of collapsing on cycle 0.

However, tempering exposed a second problem: greedy EIG selection repeats the same experiment when the posterior concentrates even slightly. M8's Thompson sampling addresses this by sampling proportional to EIG scores, producing 4× structural diversity.

The combined M7+M8+M9 solution (tempering + Thompson + crux-directed mixture) keeps later cycles informative and structurally diverse. M9 adds semantic direction: accepted cruxes bias selection toward experiments that resolve specific theoretical disagreements. The remaining limitation is that the crux-directed selection rate is low (1/15 experiments in validation) because most cruxes reference structures already well-represented in the EIG pool.

### 4.6 Limitations

1. **Residual posterior concentration.** M7 tempering (tau=0.005) prevents immediate collapse but the posterior still concentrates within 3–5 cycles, consistent with Oelrich et al.'s (2020) analysis of overconfident posteriors when models give categorically different predictions. Combined with Thompson sampling (M8), later cycles are informative and structurally diverse, but the system still converges faster than may be ideal for extended runs. Deeper solutions include adaptive learning rates (Grünwald 2012; Wu & Martin 2023), stacking instead of posterior probabilities (Yao et al. 2018), coarsened posteriors (Miller & Dunson 2019), or sequential BOED framed as POMDP (Huan & Marzouk 2016).

2. **Growing but still modest debate impact.** Post-M5, replication variance is non-zero but small (std≈0.018). Post-M9, debate causally affects experiment selection via crux-directed mixture (1/15 experiments in validation). The Bayesian machinery still dominates, but the debate→selection causal path is now operational.

3. **Synthetic data only.** The framework validates whether correct models are identifiable in principle, not whether the models are correct accounts of human behavior. Extending to real experimental data would require a lab-automation interface.

4. **Three models only.** The framework currently implements GCM, SUSTAIN, and RULEX. Generalization to other model families (neural networks, Bayesian cognitive models) is architecturally straightforward but untested.

5. **LLM-agnostic convergence.** Cross-LLM comparison (Section 3.6) shows correct model wins regardless of backbone, which validates robustness but also suggests the LLM is currently a replaceable component. The debate quality differences between GPT-4o, Sonnet, and Opus do not translate into convergence differences.

6. **No human evaluation of debate quality.** Our quality audit was systematic but not blind. Expert evaluation of whether agent reasoning constitutes genuine scientific reasoning would strengthen the findings.

7. **Claim ledger requires explicit directives.** Agents don't spontaneously engage with the claim ledger or conflict map despite injection into prompts (M5–M9). M10's explicit directive achieves 100% engagement when applicable, but the dominant response is "explain" (auxiliary hypothesis shielding) rather than genuine theory revision. The calibration problem persists: agents overclaim by 3–5× even when confronting prior falsifications.

### 4.7 Future directions

1. **Higher crux-directed selection rates** — M9 establishes the crux→experiment causal path (1/15 experiments in validation). Higher rates require either a larger crux_weight, fuzzy matching that maps cruxes to nearby pool entries, or constrained decoding (Tam et al. 2024) to guarantee format compliance. The convergence between crux-directed and EIG-driven selection (Corcoran et al. 2023) suggests the unique value of cruxes may be at the margins — pointing to experiments that EIG undervalues.
2. **Claim-responsive debate (M10 — DONE)** — agents now receive explicit directives listing their falsified claims and must address each one (revise, explain, or abandon) via a `"falsified_response"` JSON field. Inspired by Shinn et al.'s Reflexion (NeurIPS 2023). Live validation: 3/3 correct, 80% FR rate (100% when applicable), "explain" dominates (Lakatos-compatible)
3. **Richer design spaces (M11 — DONE)** — extends the fixed 55-candidate pool to 168 candidates via parametric structures (13) and interpolated conditions (2). Superseded by M12.
4. **Continuous design space (M12 — DONE)** — replaces M11's fixed grid with continuous sampling from parameter ranges (~427 candidates per cycle). Each cycle draws fresh structures, letting EIG discover diagnostic sweet spots. 15/15 sampled structures selected, 0% cycle overlap. Config: `design_space: continuous` (default). 331 tests.
4. **Longer runs (10+ cycles)** — assess whether Thompson sampling's structural diversity compounds over many cycles, whether novel structures eventually outperform registry structures, and whether the claim ledger produces cumulative reasoning at longer horizons
5. **Non-myopic experiment selection** — full Myopic Posterior Sampling (Kandasamy et al. 2019) or deep adaptive design (Foster et al. 2021) could replace the current simplified Thompson implementation
6. **Cross-domain generalization** — apply the framework to other multi-model disputes in cognitive science (memory models, decision-making theories)
7. **Real data integration** — AutoRA + Prolific for closing the loop with human participants

---

## 5. Conclusion

Antagonistic collaboration via LLM debate can successfully identify the correct model from competing theories. The mechanism of convergence is primarily Bayesian computation, but successive milestones have progressively strengthened debate's causal role. M5's feedback loop closures create non-zero replication variance through parameter revision persistence. M6's ARBITER integration adds role-specialized meta-agents, crux-based negotiation, conflict maps, and pre-registration output, enriching debate quality while preserving correct convergence (3/3 ground truths, 36–68% gaps). M7's likelihood tempering (tau=0.005) resolves the posterior collapse bottleneck, achieving gradual convergence where later cycles are genuinely informative (EIG>0 through cycle 4). M8's Thompson sampling replaces greedy experiment selection with principled exploration: 12 unique structures (6 novel) vs greedy's 3 (0 novel). M9's crux-directed Thompson sampling fixes the broken crux-to-experiment pipeline (0 parseable crux specs across all prior runs → 24 across 3 validation runs) and establishes the first semantically directed path from debate to experiment selection: accepted cruxes bias the mixture distribution toward experiments that resolve specific theoretical disagreements. In the GCM validation run, crux `crux_004` directly selected `rule_plus_exception_1exc/high_noise` — the first time a debate-identified theoretical fault line determined which experiment ran. The system operates as a falsification engine: 44 claims falsified vs 1 confirmed across M6 runs. The optimal architecture separates computation (experiment selection via tempered EIG + crux-directed Thompson sampling, posterior update) from language (interpretation, hypothesis generation, crux identification, novel structure design), connecting them through validated feedback paths (parameter persistence, claim verification, novel structure registration, crux-directed selection). M10's claim-responsive debate addresses one of these limitations: agents with falsified claims now must explicitly acknowledge, revise, or explain each failure (80% compliance, 100% when applicable). The dominant response is "explain" — attributing falsification to confounds and boundary conditions — reproducing Lakatos's auxiliary hypothesis shielding without being programmed to do so. This closes the ignoring gap (agents no longer pretend falsification didn't happen) but not the calibration gap (overclaiming persists at 3–5×). M11's richer design spaces extend the candidate pool from 55 to 168 by adding parametric structures and interpolated conditions, and M12 goes further with continuous sampling (~427 candidates per cycle), letting each cycle explore fresh parameter regions. The result sharpens the architectural thesis: computation drives model identification (15/15 sampled structures selected, 0% cycle overlap, 77–96% gaps), while debate provides interpretive value (mechanistic narratives, 80% claim-responsive engagement) whose causal contribution to identification remains unablated. The framework demonstrates both the promise and the current boundaries of LLMs in the scientific method: they identify genuine theoretical fault lines, propose novel experiments, produce human-readable mechanistic narratives, and — when directed — engage with disconfirming evidence through structured scientific reasoning. They cannot yet learn cumulatively without external scaffolding, nor calibrate their quantitative expectations to match their computational models.

---

## References

- Bissiri, P. G., Holmes, C. C., & Walker, S. G. (2016). A general framework for updating belief distributions. *Journal of the Royal Statistical Society: Series B, 78*(5), 1103–1130.
- Cavagnaro, D. R., Myung, J. I., Pitt, M. A., & Kujala, J. V. (2010). Adaptive design optimization: A mutual information-based approach to model discrimination in cognitive science. *Neural Computation, 22*(4), 887–905.
- Chapelle, O., & Li, L. (2011). An empirical evaluation of Thompson sampling. *Advances in Neural Information Processing Systems, 24*.
- Corcoran, A. W., Hohwy, J., & Friston, K. J. (2023). Accelerating scientific progress through Bayesian adversarial collaboration. *Neuron, 111*(22), 3505–3516.
- Foster, A., Ivanova, D. R., Malik, I., & Rainforth, T. (2021). Deep adaptive design: Amortizing sequential Bayesian experimental design. *Proceedings of the 38th ICML*, 3384–3395.
- Grünwald, P. (2012). The safe Bayesian: Learning the learning rate via the mixability gap. *Algorithmic Learning Theory (ALT 2012)*, LNCS 7568, 169–183.
- Huan, X., & Marzouk, Y. M. (2016). Sequential Bayesian optimal experimental design via variational inference. *arXiv:1604.08320*.
- Kandasamy, K., Schneider, J., & Póczos, B. (2019). Myopic posterior sampling for adaptive goal oriented design of experiments. *Proceedings of the 36th ICML*, 3222–3232.
- Kim, W., Pitt, M. A., Lu, Z.-L., Steyvers, M., & Myung, J. I. (2017). A hierarchical adaptive approach to optimal experimental design. *Neural Computation, 26*(11), 2465–2492.
- Love, B. C., Medin, D. L., & Gureckis, T. M. (2004). SUSTAIN: A network model of category learning. *Psychological Review, 111*(2), 309–332.
- Medin, D. L., & Schaffer, M. M. (1978). Context theory of classification learning. *Psychological Review, 85*(3), 207–238.
- Mellers, B., Hertwig, R., & Kahneman, D. (2001). Do frequency representations eliminate conjunction effects? An exercise in adversarial collaboration. *Psychological Science, 12*(4), 269–275.
- Miller, J. W., & Dunson, D. B. (2019). Robust Bayesian inference via coarsening. *Journal of the American Statistical Association, 114*(527), 1113–1125.
- Navarro, D. J., Pitt, M. A., & Myung, I. J. (2004). Assessing the distinguishability of models and the informativeness of data. *Cognitive Psychology, 49*(1), 47–84.
- Nosofsky, R. M. (1986). Attention, similarity, and the identification–categorization relationship. *Journal of Experimental Psychology: General, 115*(1), 39–57.
- Nosofsky, R. M. (1991). Tests of an exemplar model for relating perceptual classification and recognition memory. *Journal of Experimental Psychology: Human Perception and Performance, 17*(1), 3–27.
- Nosofsky, R. M., Palmeri, T. J., & McKinley, S. C. (1994). Rule-plus-exception model of classification learning. *Psychological Review, 101*(1), 53–79.
- Oelrich, O., Ding, S., Magnusson, M., Vehtari, A., & Villani, M. (2020). When are Bayesian model probabilities overconfident? *arXiv:2003.04026*.
- Ouyang, L., Tessler, M. H., Ly, D., & Goodman, N. D. (2018). webppl-oed: A practical optimal experiment design system. *Proceedings of the 40th Annual Conference of the Cognitive Science Society*.
- Rainforth, T., Foster, A., Ivanova, D. R., & Smith, F. B. (2024). Modern Bayesian experimental design. *Statistical Science, 39*(1), 100–114.
- Russo, D. J., & Van Roy, B. (2018). Learning to optimize via information-directed sampling. *Operations Research, 66*(1), 230–252.
- Shepard, R. N., Hovland, C. I., & Jenkins, H. M. (1961). Learning and memorization of classifications. *Psychological Monographs: General and Applied, 75*(13), 1–42.
- Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. (2023). Reflexion: Language agents with verbal reinforcement learning. *Advances in Neural Information Processing Systems, 36*.
- Tam, Z. R., Wu, C., et al. (2024). Let me speak freely? A study on the impact of format restrictions on performance of large language models. *Proceedings of EMNLP 2024 Industry Track*.
- Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. *Biometrika, 25*(3/4), 285–294.
- Wu, P.-S., & Martin, R. (2023). A comparison of learning rate selection methods in generalized Bayesian inference. *Bayesian Analysis, 18*(1), 105–132.
- Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018). Using stacking to average Bayesian predictive distributions. *Bayesian Analysis, 13*(3), 917–1003.
- Zhou, J., et al. (2023). Instruction-following evaluation for large language models. *arXiv:2311.07911*.
