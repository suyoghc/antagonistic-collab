# Antagonistic Collaboration via LLM Debate: Can AI Agents Resolve Scientific Disputes?

**Phase: M17 + Decision Domain Complete (debate + arbiter replication)**
**Date: 2026-03-21** (originally 2026-03-14; updated through decision domain arbiter replication)

---

## Abstract

We present an antagonistic collaboration framework in which LLM agents — each representing a competing scientific theory — debate through a structured protocol while a Bayesian computational layer selects experiments, generates predictions, and accumulates evidence. We demonstrate the framework in two domains: human category learning (GCM, SUSTAIN, RULEX) and decision-making under risk (Expected Utility, Cumulative Prospect Theory, Priority Heuristic).

Across 17 milestones and 48 factorial conditions, the system achieves **47/48 correct model identifications.** The project's trajectory reveals when LLM debate adds value and when it doesn't:

**Under correct specification (M1–M14):** Bayesian EIG experiment selection + likelihood tempering + Thompson sampling is causally sufficient. A controlled 3×2 ablation (M13, 18/18 correct) confirms debate is epiphenomenal — no-debate achieves the best RMSE (0.055) and gap (87.6%) while running 3-4× faster.

**Under parameter misspecification (M15, 8/9 correct):** Debate becomes causally necessary. Agents observe prediction errors, diagnose their source, and propose parameter corrections — improving identification by +3.5pp (GCM) to +22.4pp (RULEX). Parameter recovery depends on the interpretability of the parameter-to-behavior mapping: rule thresholds and sensitivity parameters are recovered; abstract mathematical parameters (value function curvature, cluster dynamics) are not.

**Under open design + misspecification (M16–M17, 21/21 correct):** Every intervention carries an implicit model-type prior: the arbiter favors similarity models, open design favors rule models, EIG is approximately agnostic. These biases compose non-additively — synergy for some model types, interference for others. R-IDeA (formal diversification) is antagonistic to debate's parameter recovery mechanism; complementary biases must use orthogonal information channels.

**Cross-domain replication (Decision M15):** The misspecification finding replicates in decision-making under risk: 0/3 no-debate → 2/3 debate (10 cycles), matching categorization exactly. The representational-format principle is domain-general: PH↔RULEX (strongest recovery), EU↔GCM (recovered with more data), CPT↔SUSTAIN (abstract parameters resist diagnosis). The arbiter bias also replicates: debate 2/3 → arbiter 1/3 (EU flipped from correct to wrong). The bias is toward *complexity* (models with more parameters produce more distinctive predictions, attracting crux-directed experiments), not just similarity — a more general mechanism than the categorization-only result suggested.

We distill 68 lessons on what LLM-mediated scientific debate can and cannot do.

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

### 2.10 arbiter-v0.1 features (M6)

M6 adds five features constituting **arbiter-v0.1** — the current version of the debate architecture. This version includes 3 theory agents (GCM, RULEX, SUSTAIN), 2 meta-agents (Integrator, Critic), and the following structured debate mechanisms. Future versions may change agent roster, model set, or debate protocol structure:

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

Six M4 runs: 3 ground truths × 2 modes, each 5 cycles. Nine cross-LLM runs: 3 ground truths × 3 LLMs. Three M5 validation runs: 3 ground truths, full_pool with GPT-4o. Four M5 replication runs: GCM ground truth, full_pool with GPT-4o (for variance analysis). Three M6 live validation runs: 3 ground truths, full_pool with GPT-4o, all arbiter-v0.1 features enabled. Three M7 validation runs: 3 ground truths, full_pool with tau=0.005. Six M8 ablation runs: 3 ground truths × 2 strategies (Thompson vs greedy), full_pool with tau=0.005. Three M9 validation runs: 3 ground truths, full_pool with crux_weight=0.3. Three M10 validation runs: 3 ground truths, claim_responsive=true. Three M11 validation runs: 3 ground truths, richer design spaces. Three M12 validation runs: 3 ground truths, continuous design space. Eighteen M13 ablation runs: 3×2 (No-Debate / Debate-No-Arbiter / Debate+Arbiter × Thompson / Greedy) × 3 ground truths. Eighteen M14 runs: same factorial as M13 with debate→computation feedback loop closed. Nine M15 runs: 3 ground truths × 3 conditions (no-debate / debate / arbiter), misspecified params. Fifteen M16 runs: 3 ground truths × 5 conditions (closed_no_debate / closed_debate / closed_arbiter / open_debate / open_arbiter). Six M17 runs: 3 ground truths × 2 conditions (open_debate / open_arbiter), misspecified params. Nine R-IDeA runs: 3 ground truths × 3 regimes (correct/misspec × no-debate/debate). Six decision M15 runs: 3 ground truths × 2 conditions (no-debate / debate), 5 and 10 cycles. 538+ automated tests verify framework correctness.

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

### 3.15 M6: arbiter-v0.1 live validation — role-specialized agents and crux negotiation

M6 adds arbiter-v0.1 architecture (meta-agents, crux negotiation, conflict maps, pre-registration) and validates with GPT-4o on all three ground truths, 5 cycles each.

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

**The role of debate.** M12 results sharpen the architectural tension: the computational layer (EIG + continuous sampling + Bayesian update) drives model identification with zero LLM calls. The debate layer provides interpretive value (mechanistic narratives, confound identification, claim engagement at 80% FR rate) but does not measurably influence experiment selection — 15/15 experiments were computationally selected, 0/15 from agent proposals. M13 ablates this directly.

### 3.27 M13: Debate ablation — is debate epiphenomenal?

M13 answers M12's open question with a controlled 3×2 ablation: No-Debate / Debate-No-Arbiter / Debate+Arbiter × Thompson / Greedy, across all 3 ground truths (18 conditions total). A reusable experiment framework (`experiment.py`) manages YAML-driven multi-condition runs with automatic global save/restore, comparison tables, and result merging.

**No-debate mode** runs only the computational pipeline: EIG selection → model-computed predictions (default params) → Bayesian posterior update. Zero LLM calls. `client=None` works.

**Live ablation results (2026-03-16, GPT-4o via Princeton, 18/18 completed):**

| Condition | GCM→Exemplar | RULEX→Rule | SUSTAIN→Clustering |
|---|---|---|---|
| thompson_no_debate | 0.088, 76.8% | 0.053, 86.1% | 0.057, 87.7% |
| thompson_debate_no_arbiter | 0.080, 76.7% | 0.078, 81.4% | 0.056, 87.5% |
| thompson_debate+arbiter | 0.081, 79.9% | 0.051, 86.9% | 0.077, 83.8% |
| greedy_no_debate | 0.065, 87.6% | 0.053, 90.0% | 0.016, 97.4% |
| greedy_debate_no_arbiter | 0.068, 84.7% | 0.172, 66.7% | 0.016, 97.4% |
| greedy_debate+arbiter | 0.071, 85.3% | 0.053, 88.9% | 0.021, 96.7% |

**Summary by debate level:**

| Debate Level | Correct | Avg RMSE | Avg Gap | Avg Time |
|---|---|---|---|---|
| None | 6/6 | 0.055 | 87.6% | 368s |
| Debate (no arbiter) | 6/6 | 0.078 | 82.4% | 1315s |
| Debate + Arbiter | 6/6 | 0.059 | 86.8% | 1107s |

**Debate is epiphenomenal on synthetic benchmarks.** All 18/18 conditions identify the correct winner. No-debate achieves the best RMSE and gap while running 3-4× faster. The computational pipeline (EIG + model predictions + Bayesian posterior) is causally sufficient.

**Debate without arbiter actively hurts.** LLM-proposed `param_overrides` introduce noise into model predictions that default params avoid. Without crux-directed selection to compensate, this noise degrades discrimination (RMSE 0.078 vs 0.055 for no-debate).

**arbiter-v0.1 features partially recover.** Crux-directed Thompson sampling compensates for param_override noise, recovering from debate-no-arbiter (0.078 → 0.059) but still not beating no-debate (0.055). Crux negotiation adds genuine information about which experiments matter, even though the broader debate doesn't improve identification.

**Greedy outperforms Thompson on clean data.** Greedy averaged 88.2% gap vs Thompson's 83.0%. With clean synthetic data, greedy's exploitation is optimal. Thompson retains residual uncertainty (entropy 0.001–0.032 vs 0.000), which may prove advantageous on noisier real-world data.

**The structural gap is architectural.** Debate output (interpretations, critiques, claims, cruxes) does not feed back into the quantities that drive identification (EIG scores, model predictions, posterior weights). For debate to causally help, the loop must close.

### 3.28 M14: Closing the debate→computation feedback loop (18/18 correct)

M14 closes the structural gap identified in M13: debate output now feeds back into the computational scoring pipeline. Parameter revisions proposed during debate are validated via `inspect.signature()` and, if they improve RMSE, applied to the agent's model for subsequent cycles. The claim ledger is updated with prediction outcomes, and falsified claims are surfaced in subsequent debate rounds.

**Results:** 18/18 correct across the full M13 factorial (No-Debate / Debate-No-Arbiter / Debate+Arbiter × Thompson / Greedy × 3 ground truths). Debate adds no value when models are correctly specified — the M13 finding holds. The feedback loop is architecturally open but operationally inert when parameters are already correct: agents propose revisions, but RMSE gates reject them because defaults are already optimal.

**Key finding:** The computational pipeline is **causally sufficient** for model identification on synthetic benchmarks with correctly specified models. Debate contributes interpretive value (mechanistic narratives, claim engagement) but does not improve identification. This sets the stage for M15: under what conditions does debate become causally necessary?

Scripts: `scripts/validation/validate_m14_live.py`

### 3.29 M15: Model misspecification — debate causally helps (8/9 correct)

M15 tests the first condition where debate should matter: all agents start with calibrated wrong parameters. Ground truth uses correct parameters; agents' models use gap-narrowing misspecifications from a Phase 1 parameter sweep.

**Phase 1a — Mimicry sweep** (`scripts/m15_mimicry_sweep.py`): Swept parameter grids for all 3 models across 7 base structures. **Finding: no true mimicry exists.** At every parameter setting tested, each model's predictions remain closer to its own ground truth than to any competitor. GCM, SUSTAIN, and RULEX are structurally too different for parameter changes alone to make them indistinguishable.

**Phase 1b — Competition-based sweep**: Generated synthetic data from each GT, scored all models with correct model misspecified. **Finding: misspecification never flips the winner but narrows gaps substantially.** GCM 61%→28%, SUSTAIN 65%→29%, RULEX 82%→16%. RULEX most vulnerable.

Calibrated misspecification settings: GCM c=0.5 (gap 28%), SUSTAIN r=3.0/eta=0.15 (gap 29%), RULEX error_tolerance=0.25/p_single=0.3 (gap 16%).

**Phase 2 — Full 9-run matrix (GPT-4o, 5 cycles each):**

| GT | No-debate gap | Debate gap | Arbiter gap | Best |
|---|---|---|---|---|
| GCM | 74.4% | 77.9% (+3.5pp) | 79.3% (+4.9pp) | Arbiter |
| SUSTAIN | 87.7% | 85.8% (-1.9pp) | 76.1% (-11.6pp) | No-debate |
| RULEX | 58.0% | 80.4% (+22.4pp) | 3.2% (-54.7pp, **wrong winner**) | Debate |

**Correct winner: 8/9.** Only arbiter-RULEX fails (the project's only wrong winner across 47/48 total).

**Debate without arbiter is the best configuration under misspecification.** It helps GCM (+3.5pp) and RULEX (+22.4pp) via parameter recovery. Parameter recovery rates: GCM 85.7% (sensitivity c recovered), RULEX 60.3% (rule probabilities partially recovered), SUSTAIN 0% (misspecification doesn't produce enough prediction error to trigger revisions — the misspecified SUSTAIN still fits data well).

**The arbiter is catastrophic for RULEX under misspecification.** Meta-agents distort the divergence mapping, shifting experiment selection toward non-discriminative structures. RULEX arbiter gap drops from 80.4% (debate) to 3.2% (arbiter) — the wrong model wins.

**Parameter recovery is the mechanism.** `sync_params_from_theory()` + `validate_param_revision()` allow agents to propose new parameter values during debate. When prediction errors are visible and the parameter-to-behavior mapping is intuitive, agents diagnose correctly. When the mapping is abstract (SUSTAIN's cluster dynamics), agents cannot diagnose even with visible errors.

Scripts: `scripts/m15_mimicry_sweep.py`, `scripts/validation/validate_m15_live.py`

### 3.30 M16: Open design space — every intervention carries an implicit prior (15/15 correct)

M16 tests the second condition: agents must propose all experiment structures via debate rather than selecting from a curated registry. 2×2+1 factorial: closed/open × debate/arbiter + no-debate baseline. 15 runs total.

**Full results (GPT-4o, 5 cycles each):**

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

**The arbiter is not universally bad — it's model-type-dependent.** M15 concluded the arbiter was net negative. M16 overturns this: the arbiter is a *bias*, not noise. Cruxes steer experiment selection toward continuous/similarity-based structures. SUSTAIN closed_arbiter achieves **96.0% gap** (+8.3pp) — best SUSTAIN result ever, RMSE 0.020 lowest measured. But RULEX closed_arbiter: 63.9% (-22.2pp).

**Open design space helps RULEX, hurts everything else.** Agent-proposed structures are semantically rich (exception-heavy, rule-diagnostic). RULEX open_debate (82.7%) dramatically outperforms closed_debate (58.6%). SUSTAIN open_debate (64.1%) is the worst SUSTAIN condition (-23.6pp). This is the mirror image of the arbiter bias.

**Arbiter recovers open-design losses for similarity models.** GCM: open_debate -5.2pp → open_arbiter +0.1pp (full recovery). SUSTAIN: open_debate -23.6pp → open_arbiter -17.0pp (partial recovery).

**Computation alone remains the most reliable single condition.** closed_no_debate achieves 76.8-87.7% gap across all three ground truths with zero LLM calls.

Scripts: `scripts/validation/validate_m16_live.py`. Tests: `TestOpenDesignSpace` (5 tests) in `tests/test_bugfixes.py`.

### 3.31 M17: Composition under double stress — 6/6 correct (47/48 overall)

M17 combines M15 misspecification with M16 open design: agents start with wrong parameters AND must propose all structures. Two conditions per GT: open_debate and open_arbiter.

**Results (GPT-4o, 5 cycles each):**

| GT | Condition | Winner | Correct? | RMSE | Gap% | #Structs | Param Recovery |
|---|---|---|---|---|---|---|---|
| GCM | open_debate | Exemplar_Agent | Yes | 0.114 | 67.3 | 48 | 42.9% |
| GCM | open_arbiter | Exemplar_Agent | Yes | 0.047 | **87.8** | 58 | 85.7% |
| SUSTAIN | open_debate | Clustering_Agent | Yes | 0.085 | 77.4 | 48 | 0% |
| SUSTAIN | open_arbiter | Clustering_Agent | Yes | 0.098 | 72.7 | 51 | 0% |
| RULEX | open_debate | Rule_Agent | Yes | 0.189 | 57.8 | 51 | 46.3% |
| RULEX | open_arbiter | Rule_Agent | Yes | 0.214 | 42.2 | 57 | 0% |

**6/6 correct — the system is robust under double stress.** Even the hardest condition (RULEX open_arbiter: wrong params + arbiter bias + open design) produces the correct winner. Combined with M14 (18/18), M15 (8/9), and M16 (15/15), the system achieves **47/48 correct across all factorial conditions**.

**GCM open_arbiter (87.8%) — best GCM result across all milestones.** Parameter recovery (85.7%) and arbiter-guided open proposals compose synergistically. Better than M15 arbiter (79.3%, closed registry) AND M16 open_arbiter (76.9%, correct params).

**Open design rescues RULEX from arbiter catastrophe.** M15 arbiter-RULEX (3.2%, wrong winner) was the project's only incorrect identification. M17 open_arbiter-RULEX (42.2%, correct) shows the open design space partially counteracts the arbiter's similarity bias.

**Composition is non-additive and model-dependent.** GCM: arbiter + misspec + open > either alone (synergy). SUSTAIN: open_debate under misspec (77.4%) > open_debate correct spec (64.1%). RULEX: open_debate under misspec (57.8%) < M15 debate (80.4%) — param recovery weakened by open design's structure diversity.

**Parameter recovery is modulated by design space.** GCM fully recovers under open_arbiter (85.7%) but only partially under open_debate (42.9%). RULEX recovery degrades from 60.3% (M15 closed) to 46.3% (M17 open_debate) and drops to 0% under open_arbiter. The arbiter redirects agent attention from parameter revision toward structure-level reasoning.

Scripts: `scripts/validation/validate_m17_live.py`

### 3.32 R-IDeA: Formal diversification cannot substitute for semantic diagnosis (negative result)

R-IDeA (Representativeness, Informativeness, De-Amplification) is a multi-objective acquisition function that weights experiment candidates by their representativeness of the design space, informativeness (EIG), and de-amplification (avoiding over-represented regions). Tested as an alternative to EIG for experiment selection, motivated by M16's finding that EIG has implicit model-type biases.

**Results (GPT-4o, 5 cycles, all conditions):**

| Condition | Mean Gap | Std |
|---|---|---|
| EIG, correct spec, no debate | 86.9% | — |
| R-IDeA, correct spec, no debate | 80.5% | — |
| EIG, misspec, no debate | 75.1% | — |
| R-IDeA, misspec, no debate | 65.4% | — |
| EIG, misspec, debate | 81.4% | 3.3% |
| R-IDeA, misspec, debate | **53.7%** | — |

**R-IDeA underperforms EIG in all regimes.** Even under correct specification where biases shouldn't matter, R-IDeA's representativeness weighting selects less informative experiments.

**R-IDeA + debate is the worst condition tested.** RULEX drops to 19.4% gap (vs 80.4% with EIG+debate) because representativeness dilutes visible prediction failures — the experiments that debate needs for parameter diagnosis. R-IDeA starves the mechanism that actually works.

**Key insight: informativeness and semantic diagnosis are synergistic; diversification and diagnosis are antagonistic.** Debate recovers parameters by observing prediction failures on informative experiments. Any intervention that reduces experiment informativeness (R-IDeA, random selection) undermines this. The optimal regime is maximally informative experiments (EIG) + maximally diagnostic reasoning (debate). Complementary biases must use orthogonal information channels, not reweight the same channel.

Scripts: `antagonistic_collab/ridea.py`, `scripts/validation/validate_ridea.py`, `scripts/validation/validate_ridea_debate.py`. Tests: `tests/test_ridea.py` (15 tests).

### 3.33 Decision domain: Cross-domain replication of the misspecification finding

To test whether the implicit-prior / complementary-bias findings generalize beyond categorization, we implemented a second domain: decision-making under risk. Three models implemented:

- **Expected Utility (EU)** — normative baseline, 1 free parameter (risk aversion r). Predicts rational choice via utility maximization.
- **Cumulative Prospect Theory (CPT)** — dominant descriptive model, 5 parameters (alpha, beta, lambda_, gamma_pos, gamma_neg). Kahneman & Tversky's nonlinear weighting of outcomes and probabilities.
- **Priority Heuristic (PH)** — lexicographic rule model, 3 parameters (prob_threshold, outcome_threshold_frac, phi). Brandstätter, Gigerenzer & Hertwig's fast-and-frugal heuristic.

Note: The models were chosen partly for their theoretical roles (EU as normative baseline parallels SUSTAIN; CPT as dominant descriptive parallels GCM; PH as heuristic challenger parallels RULEX). However, the empirical recovery parallels turned out to follow a different mapping — PH↔RULEX, EU↔GCM, CPT↔SUSTAIN — based on parameter interpretability rather than field role. See cross-domain parallels below.

**Gamble registry:** 76 problems — 17 base diagnostic gambles (certainty effect, common ratio, fourfold pattern, loss aversion, risk premium, PH-specific) + 59 parametric variants (cert-vs-risky, mixed, risky pairs). 7 gamble groups for EIG computation.

**Pipeline:** Standalone `decision_debate_runner.py` (D49) with EIG adapter, synthetic data generation, posterior updating, and debate round with parameter revision. Parameter validation uses `model.default_params` keys (not `inspect.signature` — decision models use a params dict pattern). RMSE gate validates against accumulated observations across all cycles (D50).

**No-debate baseline (computation-only):**
- Correct params: 3/3 identified by cycle 0-1
- Misspecified params: 0/3 (all wrong — stronger penalty than categorization)

**Decision M15 results (GPT-4o, misspecified params):**

| Cycles | No-debate | Debate | Score |
|---|---|---|---|
| 5 | 0/3 | 1/3 (PH only) | Partial |
| 10 | 0/3 | **2/3** (PH + EU) | **Matches categorization** |

**Per-model detail (10 cycles):**

| GT | No-debate | Debate (10 cyc) | Param Recovery |
|---|---|---|---|
| PH | Wrong | **Correct** | **100.0%** (all 3 params exact) |
| EU | Wrong | **Correct** | **75.0%** (r exact, temp off) |
| CPT | Wrong | Wrong | 28.4% (lambda_ exact, alpha/beta stuck) |

**Cross-domain parallels are clean:**
- **PH ↔ RULEX** — Rule-based models, strongest recovery. Discrete, interpretable parameters (thresholds, rule probabilities) are easiest for LLMs to diagnose.
- **EU ↔ GCM** — Recovered with sufficient evidence exposure. 5 cycles insufficient (predictions under misspecification are not distinctive enough); 10 cycles sufficient.
- **CPT ↔ SUSTAIN** — Abstract mathematical parameters (alpha/beta = value function curvature) resist LLM diagnosis, matching SUSTAIN's attention weights/cluster dynamics.

**The representational-format principle holds across domains.** Parameter recovery depends on whether the parameter-to-behavior mapping is linguistically describable, not on the scientific domain. LLMs can articulate "be more loss-averse" (lambda_ recovered) but struggle with "the value function should be less concave" (alpha/beta not recovered).

**Technical details:**
- Accumulated RMSE gate (D50) was critical: local-only gate (current cycle's 2-3 gambles) let competitor agents game revisions by accepting changes that improved local fit while hurting global fit. After switching to accumulated observations, acceptance dropped from 79% to 49%.
- Strict improvement required: tolerance=0.0 with `<` (not `<=`) to reject neutral revisions.
- Calibration: learning_rate=0.01, n_subjects=30, 7 gamble groups.

Scripts: `scripts/validation/validate_decision_m15_live.py`. Tests: `tests/test_decision_debate.py` (34 tests), `tests/test_decision_models.py` (17), `tests/test_decision_eig.py` (26), `tests/test_decision_agents.py` (18).

### 3.34 Decision domain arbiter: complexity bias replicates cross-domain

The arbiter layer (crux protocol + meta-agents + crux-directed EIG) was ported from the categorization pipeline to the decision domain and validated live. Implementation reuses the domain-agnostic `Crux` and `MetaAgentConfig` dataclasses from the categorization pipeline, with decision-specific prompts referencing CPT/EU/PH mechanisms.

**Arbiter results (GPT-4o, 10 cycles, misspecified params):**

| GT | No-debate | Debate (10 cyc) | Arbiter (10 cyc) |
|---|---|---|---|
| CPT | Wrong (→PH) | Wrong (→PH) | **Wrong** (→PH, 12.0% recovery) |
| EU | Wrong (→CPT) | **Correct** (75%) | **Wrong** (→CPT, 37.5% recovery) |
| PH | Wrong (→CPT) | **Correct** (100%) | **Correct** (78.2% recovery) |
| Score | 0/3 | 2/3 | **1/3** |

**The arbiter bias replicates cross-domain.** Debate achieves 2/3 correct; adding the arbiter degrades to 1/3. EU flips from correct (debate) to wrong (arbiter) — crux-directed selection steered toward CPT-favoring gambles (loss_aversion group selected 4 times). PH weakened but survived. CPT remains wrong under all conditions.

**The bias is toward complexity, not just similarity.** In categorization, the arbiter helped similarity models (SUSTAIN, GCM) at the expense of rule models (RULEX). In decisions, it helps CPT (the most parameterically complex model, 5 params) at the expense of simpler models (EU 1 param, PH 3 params). The underlying mechanism: cruxes ask "where do models disagree most?" → models with more free parameters produce more distinctive predictions across the gamble space → crux-directed experiments systematically probe where complex models excel. This is a general property of crux-directed selection, not a domain-specific artifact.

**Cross-domain parallel (updated through arbiter):**
- CPT ↔ SUSTAIN: abstract params resist in all conditions (debate and arbiter)
- EU ↔ GCM: arbiter effect differs — categorization arbiter helped GCM (+5pp), decision arbiter breaks EU (-37.5pp). The difference: GCM is similarity-based (aligned with arbiter bias) while EU is the simplest model (most hurt by complexity bias)
- PH ↔ RULEX: arbiter weakens recovery in both domains (categorization: -55pp wrong winner; decision: -22pp but still correct)

**Three-layer implicit prior framework (the main takeaway):**

| Layer | Bias mechanism | Categorization | Decisions |
|---|---|---|---|
| Computation (EIG) | Model-agnostic | ~Neutral | ~Neutral |
| Debate (interpretation) | Linguistic accessibility | Favors interpretable params | Favors interpretable params |
| Arbiter (cruxes + meta) | Prediction distinctiveness | Favors similarity models | Favors complex models |

Each layer adds an implicit prior. Computation is approximately neutral. Debate biases toward models with linguistically interpretable parameters (rule thresholds, sensitivity parameters). The arbiter biases toward models with the most distinctive predictions (which correlates with parametric complexity). More LLM coordination is not uniformly better — the arbiter helps complex/similarity models (SUSTAIN best-ever 96%, GCM +13.4pp) but hurts simpler/rule models (RULEX −54.7pp wrong winner), and under misspecification the net effect is negative in both domains. The optimal architecture is computation for inference/selection with LLMs restricted to semantic interpretation.

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

### 4.4 What debate does and doesn't do (revised through M17 + decision domain)

**What debate contributes:**
- Adversarial critique as a forcing function — pressures agents to refine proposals in later cycles
- Theory revision pressure — incorrect theories accommodate evidence progressively (Lakatos-compatible)
- Human-readable mechanistic explanations of model behavior
- Parameter revisions that causally affect subsequent predictions (M5 — replication std=0.018)
- Crux negotiation that identifies genuine theoretical fault lines (M6 — 15% acceptance rate, accepted cruxes map to real scientific disagreements)
- Role-specialized synthesis: Integrator identifies convergence, Critic identifies weakest arguments (M6)
- Falsification as an emergent methodology: 44:1 falsified-to-confirmed ratio (M6)
- **Parameter recovery under misspecification (M15)** — debate improves gap by +3.5pp (GCM) to +22.4pp (RULEX) via parameter diagnosis and revision. This is the first causal demonstration that debate adds value for model identification. Replicates across domains (decision M15: 0/3→2/3).
- **Novel experiment proposals that fill registry gaps (M16)** — open design proposals help RULEX (+24pp over closed_debate) when the curated registry lacks rule-diagnostic structures

**What debate does not contribute:**
- Experiment selection quality under correct specification (EIG dominates; LLM proposals are narrative-driven)
- Cumulative scientific reasoning — partially addressed by M10's claim-responsive directive (Shinn et al., NeurIPS 2023). Agents now engage with falsified claims at 100% compliance when directed, but the dominant response is "explain" (auxiliary hypothesis shielding) rather than genuine theory revision. Overclaiming persists at 3–5×
- Data-grounded argumentation (posteriors cited as proxy; item-level data ignored)
- Calibrated quantitative predictions (agents overclaim accuracy by 3–5× even after M10 claim-responsive engagement — the mechanism fixes ignoring but not calibration)
- Recovery of abstract mathematical parameters — parameters that require mathematical understanding (CPT alpha/beta, SUSTAIN cluster dynamics) are outside LLM diagnostic capability in both domains

**The full picture (M1–M17 + decision domain):**

Pre-M5, debate was entirely epiphenomenal. M5–M13 progressively strengthened the feedback loop but M13's ablation showed debate was still net negative or neutral under correct specification. **M15 changed the story:** under parameter misspecification, debate becomes causally necessary. The mechanism is parameter recovery — agents observe prediction errors, diagnose their source, and propose corrections. This works when the parameter-to-behavior mapping is intuitive (rule thresholds, sensitivity parameters) and fails when the mapping is abstract (value function curvature, cluster dynamics).

M16 revealed that every intervention carries an implicit model-type prior: the arbiter favors similarity models, open design favors rule models, EIG is approximately agnostic. M17 showed these biases compose non-additively: synergy for GCM (87.8%, best ever), rescue for RULEX (arbiter catastrophe averted by open design), interference for RULEX param recovery (60% → 46% → 0%).

R-IDeA showed that complementary biases must use orthogonal information channels. Diversifying experiment selection is antagonistic to debate's parameter recovery mechanism — debate needs maximally informative experiments to see prediction failures.

The decision domain replication confirms the finding is domain-general: the representational-format principle (recovery depends on parameter interpretability, not domain content) holds across categorization and decision-making under risk. The decision arbiter experiment extends this further: the arbiter's model-type bias also replicates (debate 2/3 → arbiter 1/3), and reveals the bias is toward *complexity* (models with more parameters), not just similarity — a more general mechanism than the categorization-only result suggested.

### 4.5 Posterior collapse: diagnosis and treatment

M6 validation revealed that the Bayesian posterior collapses to certainty after 1–2 experiments, leaving remaining cycles with EIG≈0. Oelrich et al. (2020) identify this as a general phenomenon: posterior model probabilities become overconfident when "the compared models give very different approximations" — exactly our situation with SUSTAIN's stepwise curves vs. GCM's gradual curves. M7 addressed this with likelihood tempering (tau=0.005, prediction clip [0.05, 0.95]), a form of the power posterior (Grünwald 2012; Bissiri, Holmes & Walker 2016; Miller & Dunson 2019), achieving gradual convergence: GCM entropy drops 0.64→0.00 over 5 cycles instead of collapsing on cycle 0.

However, tempering exposed a second problem: greedy EIG selection repeats the same experiment when the posterior concentrates even slightly. M8's Thompson sampling addresses this by sampling proportional to EIG scores, producing 4× structural diversity.

The combined M7+M8+M9 solution (tempering + Thompson + crux-directed mixture) keeps later cycles informative and structurally diverse. M9 adds semantic direction: accepted cruxes bias selection toward experiments that resolve specific theoretical disagreements. The remaining limitation is that the crux-directed selection rate is low (1/15 experiments in validation) because most cruxes reference structures already well-represented in the EIG pool.

### 4.6 Limitations (revised through M17 + decision domain)

1. **Residual posterior concentration.** M7 tempering (tau=0.005) prevents immediate collapse but the posterior still concentrates within 3–5 cycles. Combined with Thompson sampling (M8), later cycles are informative and structurally diverse, but the system still converges faster than may be ideal for extended runs. Deeper solutions include adaptive learning rates (Grünwald 2012; Wu & Martin 2023), stacking instead of posterior probabilities (Yao et al. 2018), coarsened posteriors (Miller & Dunson 2019), or sequential BOED framed as POMDP (Huan & Marzouk 2016).

2. **Debate is epiphenomenal under correct specification.** M13 ablation confirmed; M14 reinforced. However, M15–M17 show debate becomes causally necessary under misspecification, and the decision domain replicates this. The limitation is narrower than originally stated: debate is unnecessary only when parameters are correct.

3. **Abstract parameters resist LLM diagnosis.** CPT alpha/beta and SUSTAIN cluster dynamics are not recovered in either domain. The representational-format boundary — parameters must map to linguistically describable behaviors — appears fundamental, not addressable by more cycles alone. Prompt enrichment (explicit parameter-to-prediction mappings) is untested.

4. **Synthetic data only.** The framework validates whether correct models are identifiable in principle, not whether the models are correct accounts of human behavior. Extending to real experimental data would require a lab-automation interface.

5. **Three models per domain.** Categorization: GCM, SUSTAIN, RULEX. Decisions: EU, CPT, PH. Generalization to larger model sets or other domains is architecturally straightforward but untested.

6. **Arbiter has uncorrected model-type bias.** The arbiter systematically favors complex models — those with more parameters producing more distinctive predictions (Sections 3.30, 3.34). In categorization this manifests as similarity-model bias; in decisions as CPT bias. Open design partially counteracts this (Section 3.31) but is itself biased toward rule models. No debiasing mechanism has been implemented.

7. **Single LLM backbone for M15–M17.** All misspecification and decision-domain results use GPT-4o only. Cross-LLM comparison (Section 3.6) showed LLM-agnostic convergence under correct specification, but this has not been verified under misspecification where debate is causally active.

8. **No human evaluation of debate quality.** Our quality audit was systematic but not blind. Expert evaluation of whether agent reasoning constitutes genuine scientific reasoning would strengthen the findings.

9. **Claim ledger requires explicit directives.** Agents don't spontaneously engage with the claim ledger. M10's explicit directive achieves 100% engagement when applicable, but the dominant response is "explain" (auxiliary hypothesis shielding). The calibration problem persists: agents overclaim by 3–5×.

### 4.7 Future directions (revised through M17 + decision domain)

**Completed:**
- Claim-responsive debate (M10) — 80% FR rate, "explain" dominates (Lakatos-compatible)
- Richer design spaces (M11) — 168 candidates. Superseded by M12.
- Continuous design space (M12) — ~427 candidates/cycle, 0% cycle overlap
- Debate ablation (M13) — 18/18 correct, debate epiphenomenal under correct spec
- Debate→computation feedback loop (M14) — closed, still epiphenomenal under correct spec
- Model misspecification (M15) — debate causally helps: 8/9, +22pp RULEX via param recovery
- Open design space (M16) — 15/15, arbiter is bias not noise, mirror biases
- Composition (M17) — 6/6, 47/48 overall, non-additive composition
- R-IDeA (negative result) — diversification antagonistic to diagnosis
- Cross-domain generalization (decision M15) — 0/3→2/3, representational-format principle confirmed

**Open:**
1. **NeurIPS paper** — write up the two-domain replication. The core contribution: implicit priors in hybrid AI systems, demonstrated across categorization and decision-making under risk. Both debate replication (0/3→2/3) and arbiter bias replication (complexity bias, 2/3→1/3) confirmed.
2. **Prompt enrichment for abstract parameters** — CPT alpha/beta and SUSTAIN cluster dynamics are the holdouts in both domains. Testing whether explicit parameter-to-prediction guidance breaks through the representational-format boundary.
4. **Arbiter debiasing** — the arbiter's model-type bias is documented (M16) but uncorrected. Learning curve selection or falsification-directed selection could target underserved models.
5. **Real data integration** — AutoRA + Prolific for closing the loop with human participants. The current framework validates identifiability in principle; real data tests whether the models are correct accounts of human behavior.
6. **Non-myopic experiment selection** — full Myopic Posterior Sampling (Kandasamy et al. 2019) or deep adaptive design (Foster et al. 2021) could replace the current simplified Thompson implementation.
7. **GeCCo forks** — gecco-core (can LLMs discover cognitive models from scratch?) and gecco-supplement (is there a fourth model of categorization?).

---

## 5. Conclusion (revised through M17 + decision domain arbiter)

Antagonistic collaboration via LLM debate identifies the correct model in 47/48 factorial conditions across two scientific domains. The project's trajectory reveals a clear division of labor between computation and language in hybrid AI systems — and a counterintuitive finding about the cost of sophisticated LLM coordination.

**Phase 1 (M1–M13): Computation is sufficient.** Under correct model specification, Bayesian EIG experiment selection + likelihood tempering + Thompson sampling identifies the correct model without any LLM involvement. The M13 ablation (18/18 correct, no-debate best RMSE) confirmed that debate is epiphenomenal when parameters are correct. Successive milestones (M5–M12) progressively strengthened the debate→computation feedback loop, but none changed the outcome. The system operates as a falsification engine (44:1 falsified-to-confirmed ratio), and agents reproduce Lakatos's auxiliary hypothesis shielding without being programmed to do so.

**Phase 2 (M14–M17): Debate becomes causally necessary under misspecification.** M15 demonstrated that when agents start with wrong parameters, debate improves identification by +3.5pp (GCM) to +22.4pp (RULEX) via parameter recovery — agents observe prediction errors, diagnose their source, and propose corrections. This is Pitt & Myung's (2004) insight operationalized: parameter estimation and model selection must happen together. EIG does selection; debate does estimation.

M16 revealed that every intervention in the system carries an implicit model-type prior: the arbiter favors similarity models (+8pp SUSTAIN, -22pp RULEX), open design favors rule models (+24pp RULEX, -24pp SUSTAIN), and EIG is approximately agnostic. M17 showed these biases compose non-additively: synergy for GCM (87.8%, best ever), rescue for RULEX (arbiter catastrophe averted by open design). R-IDeA showed that complementary biases must operate through orthogonal information channels — diversifying the same channel (experiment informativeness) is antagonistic to debate's parameter recovery mechanism.

**Phase 3 (Decision domain): The finding is domain-general.** A second domain (decision-making under risk: EU, CPT, Priority Heuristic) replicates both findings:

1. *Debate replication:* 0/3 no-debate → 2/3 debate at 10 cycles, matching categorization exactly. Cross-domain parallels: PH↔RULEX (strongest recovery), EU↔GCM (recovered with more data), CPT↔SUSTAIN (abstract parameters resist). The representational-format principle holds: recovery depends on parameter interpretability, not domain content.

2. *Arbiter replication:* Debate 2/3 → arbiter 1/3. EU flips from correct to wrong. The arbiter bias is toward *complexity* — models with more parameters produce more distinctive predictions, attracting crux-directed experiments. In categorization this manifested as similarity-model bias; in decisions it manifests as CPT bias (5 params). Under misspecification, the net effect of more LLM coordination (cruxes + meta-agents) is negative in both domains — though the arbiter helps complex/similarity models while hurting simpler/rule models.

**The architecture thesis, revised:** The optimal hybrid AI system separates computation (experiment selection, evidence accumulation, posterior updating) from language (interpretation, parameter diagnosis, hypothesis generation), with the division of labor shifting based on specification quality. Under correct specification, computation is sufficient and debate adds noise. Under misspecification, debate provides the parameter estimation that computation lacks — but the arbiter (crux-directed selection + meta-agents) introduces a complexity bias that degrades performance in both domains. The key design principles: (1) informativeness and semantic diagnosis are synergistic — debate needs maximally informative experiments (EIG) to observe the prediction failures that drive parameter recovery; (2) each layer of LLM involvement adds an implicit prior — the more sophisticated the coordination mechanism, the stronger and more directional the bias.

68 lessons on LLM-mediated scientific debate are documented in [Notes/LESSONS_LEARNED.md](../LESSONS_LEARNED.md).

---

## References

- Brandstätter, E., Gigerenzer, G., & Hertwig, R. (2006). The priority heuristic: Making choices without trade-offs. *Psychological Review, 113*(2), 409–432.
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
- Kahneman, D., & Tversky, A. (1979). Prospect theory: An analysis of decision under risk. *Econometrica, 47*(2), 263–291.
- Navarro, D. J., Pitt, M. A., & Myung, I. J. (2004). Assessing the distinguishability of models and the informativeness of data. *Cognitive Psychology, 49*(1), 47–84.
- Pitt, M. A., Myung, I. J., & Zhang, S. (2002). Toward a method of selecting among computational models of cognition. *Psychological Review, 109*(3), 472–491.
- Nosofsky, R. M. (1986). Attention, similarity, and the identification–categorization relationship. *Journal of Experimental Psychology: General, 115*(1), 39–57.
- Nosofsky, R. M. (1991). Tests of an exemplar model for relating perceptual classification and recognition memory. *Journal of Experimental Psychology: Human Perception and Performance, 17*(1), 3–27.
- Nosofsky, R. M., Palmeri, T. J., & McKinley, S. C. (1994). Rule-plus-exception model of classification learning. *Psychological Review, 101*(1), 53–79.
- Oelrich, O., Ding, S., Magnusson, M., Vehtari, A., & Villani, M. (2020). When are Bayesian model probabilities overconfident? *arXiv:2003.04026*.
- Ouyang, L., Tessler, M. H., Ly, D., & Goodman, N. D. (2018). webppl-oed: A practical optimal experiment design system. *Proceedings of the 40th Annual Conference of the Cognitive Science Society*.
- Rainforth, T., Foster, A., Ivanova, D. R., & Smith, F. B. (2024). Modern Bayesian experimental design. *Statistical Science, 39*(1), 100–114.
- Russo, D. J., & Van Roy, B. (2018). Learning to optimize via information-directed sampling. *Operations Research, 66*(1), 230–252.
- Shepard, R. N., Hovland, C. I., & Jenkins, H. M. (1961). Learning and memorization of classifications. *Psychological Monographs: General and Applied, 75*(13), 1–42.
- Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. (2023). Reflexion: Language agents with verbal reinforcement learning. *Advances in Neural Information Processing Systems, 36*.
- Tversky, A., & Kahneman, D. (1992). Advances in prospect theory: Cumulative representation of uncertainty. *Journal of Risk and Uncertainty, 5*(4), 297–323.
- von Neumann, J., & Morgenstern, O. (1944). *Theory of games and economic behavior*. Princeton University Press.
- Tam, Z. R., Wu, C., et al. (2024). Let me speak freely? A study on the impact of format restrictions on performance of large language models. *Proceedings of EMNLP 2024 Industry Track*.
- Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. *Biometrika, 25*(3/4), 285–294.
- Wu, P.-S., & Martin, R. (2023). A comparison of learning rate selection methods in generalized Bayesian inference. *Bayesian Analysis, 18*(1), 105–132.
- Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018). Using stacking to average Bayesian predictive distributions. *Bayesian Analysis, 13*(3), 917–1003.
- Zhou, J., et al. (2023). Instruction-following evaluation for large language models. *arXiv:2311.07911*.
