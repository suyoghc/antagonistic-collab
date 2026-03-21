# Implicit Priors in Hybrid AI Systems: How Representational Format Biases Automated Experiment Design

**Target venue:** NeurIPS / ICML (automated scientific discovery track)

**Target community:** Bayesian OED (Myung, Pitt, Cavagnaro, Foster, Rainforth), automated science (AI Scientist, AutoRA, GeCCo), multi-agent LLM systems

---

## Core Claim

Every component of a hybrid LLM-computation system embeds a systematic bias toward certain model types, determined by the component's representational format. These biases are identifiable, replicable across domains, and — counterintuitively — compound rather than cancel when layers of LLM coordination are added.

**The one-sentence version:** More sophisticated LLM coordination makes automated experiment design worse, not better, because each coordination mechanism introduces an implicit prior that biases which scientific theories get a fair test.

---

## Why This Matters (for this audience)

Hybrid systems combining LLMs with formal computation are proliferating: AI Scientist (Lu et al. 2024), AutoRA (Griffiths et al.), GeCCo (Dubova et al.), Corcoran et al. (2023). All assume that adding LLM reasoning to Bayesian OED improves experiment design. We show this assumption is wrong in a precise, characterizable way — and provide a diagnostic framework for when LLM involvement helps vs. hurts.

This is not a domain-specific finding about cognitive models. It's a general property of any system where LLM reasoning interfaces with formal computation for experiment selection.

---

## Key Results to Present

### Result 1: Computation suffices under correct specification

Bayesian EIG + Thompson sampling identifies the correct model in 100% of well-specified conditions (24/24 categorization, 3/3 decision-making) with zero LLM calls. Controlled ablation (M13, 18/18) confirms debate is epiphenomenal. No-debate achieves best RMSE (0.055) and gap (87.6%) while running 3-4× faster.

**For this audience:** Confirms that Bayesian OED works as advertised (Cavagnaro et al. 2010, Foster et al. 2021). The LLM adds no value when the formal method's assumptions hold.

### Result 2: LLM reasoning becomes causally necessary under misspecification

When model parameters are wrong, EIG selects experiments informative for the wrong regime. LLM agents diagnose prediction failures and propose corrections: +3.5pp (GCM) to +22.4pp (RULEX) via parameter recovery. Replicates across domains (0/3→2/3 in both categorization and decision-making).

**For this audience:** This is Pitt & Myung's (2004) insight operationalized — parameter estimation and model selection must happen together. EIG does selection; LLM debate does estimation. Neither alone is sufficient under misspecification.

### Result 3: Three-layer implicit prior framework

Each architectural layer introduces a specific, identifiable bias:

| Layer | Mechanism | Bias | Evidence |
|---|---|---|---|
| Computation (EIG) | Exhaustive evaluation | ~Model-agnostic | Neutral across all conditions |
| Debate (LLM interpretation) | Linguistic accessibility | Favors interpretable parameters | Rule models recover best (RULEX +22pp, PH +100%) |
| Arbiter (crux-directed selection) | Prediction distinctiveness | Favors complex models | CPT posterior rises; EU flips correct→wrong |

This replicates across both domains. The biases arise from representational format, not domain content.

### Result 4: More coordination → worse performance

| Condition | Categorization | Decision-making |
|---|---|---|
| Computation only | 0/3 (misspec) | 0/3 |
| + Debate | 2/3 | 2/3 |
| + Debate + Arbiter | 1/3 (RULEX wrong) | 1/3 (EU wrong) |

Adding the arbiter layer (crux-directed selection + meta-agents) degrades performance in both domains. The mechanism: cruxes ask "where do models disagree most?" → complex models (more parameters) produce more distinctive predictions → experiment selection is pulled toward the complex model's home turf. This is a structural property of divergence-based experiment selection.

### Result 5: Complementary biases must use orthogonal information channels

R-IDeA (representativeness + informativeness + de-amplification) reweights the same information channel EIG already optimizes. R-IDeA + debate (53.7%) is worse than EIG + debate (81.4%) — the worst condition tested. RULEX drops to 19.4% because representativeness dilutes the visible prediction failures debate needs for parameter recovery.

**Design principle:** Diversify information CHANNELS (item predictions, temporal dynamics, linguistic structure), not information WEIGHTINGS within a single channel.

### Result 6: Biases compose non-additively

Under double stress (misspecification + open design space, M17):
- GCM arbiter: 87.8% — best result ever. Param recovery + arbiter-guided proposals synergize.
- RULEX arbiter: 42.2% — arbiter catastrophe (3.2%, wrong winner in M15) rescued by open design's counterbalancing rule-model bias.

Complementary biases can cancel (GCM arbiter + open design), but only when they are genuinely orthogonal.

---

## Paper Structure

### 1. Introduction (1.5 pages)
- Hybrid LLM-computation systems for automated science are proliferating
- Assumption: adding LLM reasoning improves experiment design
- We show: each LLM component introduces an implicit prior that biases which theories get a fair test
- Demonstrated across two scientific domains with positive, negative, and replication results
- The counterintuitive finding: more sophisticated coordination → worse performance

### 2. Related Work (1 page)
- **Bayesian OED:** Myung & Pitt (2009), Cavagnaro et al. (2010), Foster et al. (2021), Rainforth et al. (2024)
- **Misspecification in OED:** Pitt, Myung & Zhang (2002), Sloman et al. (active learning bias)
- **Multi-agent LLM debate:** Du et al. (2023), Liang et al. (2023) — but none ground debate in formal computation
- **Automated science:** AI Scientist (Lu et al. 2024), AutoRA (Griffiths), GeCCo (Dubova et al.)
- **Adversarial collaboration:** Mellers, Hertwig & Kahneman (2001), Corcoran et al. (2023)

### 3. Framework (1.5 pages)
- Three-layer architecture: computation (EIG) → debate (LLM interpretation + param recovery) → arbiter (crux-directed selection + meta-agents)
- Each layer independently ablatable
- Debate cycle: EIG select → generate data → posterior update → LLM interpret → param revision → repeat
- Arbiter additions: crux identification/negotiation → crux-directed EIG mixture → meta-agent synthesis
- Two domains: categorization (GCM, SUSTAIN, RULEX) and decision-making under risk (EU, CPT, PH)

### 4. Experiments (3 pages)

#### 4.1 Domain 1: Human Categorization
- Models (brief specs, full in appendix): GCM (exemplar), SUSTAIN (clustering), RULEX (rule-based)
- Experiment space: 11 structures × 5 conditions, continuous sampling, open design
- Factorial (M14-M17): 48 runs, 47/48 correct
- Key table: gap by condition (no-debate / debate / arbiter) × ground truth

#### 4.2 Domain 2: Decision-Making Under Risk
- Models: EU (1 param), CPT (5 params), PH (3 params)
- Experiment space: 76 gambles in 7 groups
- M15 replication: 0/3 → 2/3 (debate), → 1/3 (arbiter)
- Cross-domain parallel table: PH↔RULEX, EU↔GCM, CPT↔SUSTAIN

#### 4.3 Negative Control: R-IDeA
- Multi-objective diversification fails: 53.7% (worst) vs 81.4% (EIG+debate best)
- Mechanism: diversification starves debate's parameter recovery

### 5. Analysis: The Implicit Prior Framework (1.5 pages)
- Three-layer bias table (the main contribution)
- The representational-format principle: bias tracks format, not domain
- Cross-domain replication as evidence for generality
- When to use each component: decision framework based on specification quality
- The "less is more" principle: optimal architecture ≠ all layers active

### 6. Discussion & Limitations (1 page)
- Synthetic data necessary for measuring implicit priors (known ground truth required)
- Two domains — same structural pattern = principle, not coincidence
- Abstract parameters (CPT alpha/beta, SUSTAIN clusters) are a fundamental boundary
- Implications for hybrid system design: characterize your biases before deploying
- Future: real data, more domains, debiasing mechanisms

### Appendices
- Full model specifications (both domains)
- Complete factorial results tables (all 48 + 9 conditions)
- R-IDeA implementation details
- Debate transcript examples showing parameter recovery
- Arbiter crux selection patterns

---

## Anticipated Reviewer Questions

**"Why synthetic data?"**
Because you need known ground truth to measure implicit priors. With real data, you can't tell whether the system's answer is biased or correct. This is the diagnostic that should be run BEFORE deploying on real data.

**"Only two domains?"**
Two domains with the same structural pattern is the minimum for claiming generality. The bias tracks representational format (smooth vs discrete, complex vs simple), not domain content. A third domain would strengthen but isn't required.

**"How does this compare to AutoRA / AI Scientist?"**
AutoRA uses Bayesian OED without LLM reasoning — optimal under correct specification, suboptimal under misspecification. AI Scientist is single-agent with no adversarial structure or computational model grounding. We are the first to characterize when LLM reasoning adds value to formal OED.

**"Is the arbiter result just a bad implementation?"**
No — the bias is structural. Crux-directed selection will always favor models with more distinctive predictions (more parameters). We demonstrate the mechanism (loss_aversion selected 4×, CPT posterior rises) and replicate across domains. A "better" arbiter implementation would need to explicitly counteract this bias.

**"Could you just tune the crux weight?"**
The crux weight (0.3) determines how often crux-directed selection overrides EIG. Setting it to 0 recovers pure EIG (which we already test as no-arbiter). The problem isn't the weight — it's that the cruxes themselves systematically identify experiments favoring complex models. Any positive crux weight introduces the bias.

---

## Figures

1. **Architecture diagram** — three-layer system with bias labels
2. **Factorial heatmap** — gap% across all conditions × ground truths (categorization)
3. **Cross-domain replication** — side-by-side bar charts (cat vs decision, no-debate/debate/arbiter)
4. **Parameter recovery by interpretability** — scatter plot of recovery% vs parameter interpretability rating
5. **R-IDeA comparison** — bar chart showing degradation
6. **Crux selection bias** — which gamble groups / structures get selected under arbiter vs EIG
