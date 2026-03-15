# Antagonistic Collaboration via LLM Debate: Can AI Agents Resolve Scientific Disputes?

**Phase: M4 — Analysis & Write-up**
**Date: 2026-03-14**

---

## Abstract

We present an antagonistic collaboration framework in which three LLM agents — each representing a competing theory of human category learning — debate through a structured protocol, propose experiments, and converge toward the theory that best explains synthetic data. The three models are the Generalized Context Model (GCM; Nosofsky 1986), SUSTAIN (Love, Medin & Gureckis 2004), and RULEX (Nosofsky, Palmeri & McKinley 1994). We compare two architectures: a *legacy* mode where LLM agents propose experiments through adversarial debate, and a *full-pool* mode where Bayesian expected information gain (EIG) selects experiments while agents shift to interpreting results and generating hypotheses. Across 6 validation runs (3 ground truths × 2 modes, 5 cycles each), the correct model's agent wins in every condition. Full-pool mode achieves dramatically better discrimination for hard model pairs (RULEX gap: 2.4% legacy vs. 68% full-pool), driven by learning curves as a second evidence channel. We find that LLM agents add value for mechanistic interpretation but not for experiment selection, and that adversarial debate does not produce cumulative scientific reasoning within the current architecture.

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

### 2.9 Validation protocol

Six runs: 3 ground truths (GCM, SUSTAIN, RULEX) × 2 modes (full_pool, legacy), each 5 cycles. The correct agent should achieve the lowest RMSE in every condition. 207 automated tests verify framework correctness.

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

### 3.5 Novel structure generation

Across 3 full-pool runs (15 cycles), agents proposed 21 novel structures:

| Category | Count | Examples |
|---|---|---|
| Random/unstructured | 5 | random_assignment, randomized_no_rule |
| Complex conjunctive | 4 | complex_conjunction, noisy_xor |
| Multimodal/subgroup | 5 | multimodal_subgroups, overlapping_clusters |
| Attention/order-based | 3 | order_dependency_test, nonverbal_complex |
| Other | 4 | noisy_or, staggered_overlap, asymmetric_complex |

None of the 21 novel structures were selected by EIG. The Bayesian selector consistently preferred registry structures (five_four, Type_I) — either because the 11 registry structures already span the relevant discrimination space, or because LLM-proposed structures are narratively interesting but not statistically optimal.

### 3.6 Theory revision patterns

| Theory | Revisions when TRUE model | Revisions when NOT true |
|---|---|---|
| GCM (Exemplar) | 0 (stable) | 1–4 (progressive) |
| RULEX (Rule) | 0–1 | 0–1 (rigid) |
| SUSTAIN (Clustering) | 0 (stable) | 2–4 (progressive) |

Correct theories do not revise — their predictions already match the data. Incorrect theories revise progressively (adapting parameters, adjusting claims) but never degeneratively. This is a Lakatos-compatible outcome. RULEX is notably revision-resistant even when wrong, consistent with its rigid rule-based structure having fewer free parameters.

### 3.7 Interpretation debate quality

We audited all 30 debate cycles across 6 runs on four dimensions:

| Dimension | Rating | Finding |
|---|---|---|
| Data citation accuracy | Weak | Agents cite posterior probabilities but rarely reference item-level predictions, RMSE values, or learning curve shapes |
| Critique quality | Mixed | Structurally substantive (cite mechanisms, name parameters) but numerically ungrounded ("model flexibility allows post-hoc fitting" without specifying which parameters diverge) |
| Behavioral adaptation | Limited | Same 2–3 talking points repeat across all 5 cycles within a run; no cumulative learning from prior data |
| Novel structure rationale | Poor | Proposals not rooted in actual model divergence; duplicate existing structures with condition permutations |

The adversarial critique forcing function does produce improvement in later cycles (more specific proposals after 3+ cycles of critique pressure), but the debate does not generate cumulative scientific reasoning.

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

**What debate does not contribute:**
- Experiment selection quality (0% of value — EIG dominates)
- Cumulative scientific reasoning (agents repeat talking points across cycles)
- Data-grounded argumentation (posteriors cited as proxy; item-level data ignored)

The debate produces the *form* of scientific discourse — agents argue, cite mechanisms, revise theories — but not its *function*. The convergence is driven entirely by the Bayesian machinery. Whether this reflects a fundamental limitation of current LLMs or a fixable prompt engineering problem is an open question.

### 4.5 Limitations

1. **Single runs per condition.** Without replication, we cannot estimate variance in RMSE gaps. The large effect sizes (34–68%) suggest robustness, but formal confidence intervals require replicate runs.

2. **Synthetic data only.** The framework validates whether correct models are identifiable in principle, not whether the models are correct accounts of human behavior. Extending to real experimental data would require a lab-automation interface.

3. **Three models only.** The framework currently implements GCM, SUSTAIN, and RULEX. Generalization to other model families (neural networks, Bayesian cognitive models) is architecturally straightforward but untested.

4. **GPT-4o as the agent backbone.** Debate quality may differ with other LLMs. The specification gap and cumulative reasoning limitations may be model-specific.

5. **No human evaluation of debate quality.** Our quality audit was systematic but not blind. Expert evaluation of whether agent reasoning constitutes genuine scientific reasoning would strengthen the findings.

### 4.6 Future directions

1. **Enforce numerical citation requirements** — agents must cite 3+ specific item predictions that diverge, not just posterior probabilities
2. **Structured claim tracking** — maintain a per-agent claim registry that is updated each cycle, flagging stale claims that haven't been revised despite contradicting data
3. **Longer runs (10+ cycles)** — assess whether novel structures eventually outperform registry structures as the registry space is exhausted
4. **Cross-domain generalization** — apply the framework to other multi-model disputes in cognitive science (memory models, decision-making theories)
5. **Hybrid mode** — EIG for selection, but with agents interpreting the EIG landscape to generate hypotheses about why certain experiments are informative (bridging computation and reasoning)

---

## 5. Conclusion

Antagonistic collaboration via LLM debate can successfully identify the correct model from competing theories, but the mechanism of convergence is Bayesian computation, not argumentation. The optimal architecture separates computation (experiment selection, posterior update, learning curves) from language (interpretation, hypothesis generation, explanation). LLM agents add genuine value as scientific narrators — translating statistical evidence into mechanistic understanding — but they do not yet function as autonomous scientific reasoners who learn cumulatively from evidence. The framework demonstrates both the promise and the current limits of LLMs in the scientific method.

---

## References

- Love, B. C., Medin, D. L., & Gureckis, T. M. (2004). SUSTAIN: A network model of category learning. *Psychological Review, 111*(2), 309–332.
- Medin, D. L., & Schaffer, M. M. (1978). Context theory of classification learning. *Psychological Review, 85*(3), 207–238.
- Mellers, B., Hertwig, R., & Kahneman, D. (2001). Do frequency representations eliminate conjunction effects? An exercise in adversarial collaboration. *Psychological Science, 12*(4), 269–275.
- Nosofsky, R. M. (1986). Attention, similarity, and the identification–categorization relationship. *Journal of Experimental Psychology: General, 115*(1), 39–57.
- Nosofsky, R. M. (1991). Tests of an exemplar model for relating perceptual classification and recognition memory. *Journal of Experimental Psychology: Human Perception and Performance, 17*(1), 3–27.
- Nosofsky, R. M., Palmeri, T. J., & McKinley, S. C. (1994). Rule-plus-exception model of classification learning. *Psychological Review, 101*(1), 53–79.
- Shepard, R. N., Hovland, C. I., & Jenkins, H. M. (1961). Learning and memorization of classifications. *Psychological Monographs: General and Applied, 75*(13), 1–42.
