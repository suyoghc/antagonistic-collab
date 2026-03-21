# When Models Are Wrong: LLM-Assisted Parameter Recovery in Bayesian Optimal Experiment Design

**Target venue:** PNAS / BBS / Journal of Mathematical Psychology

**Target community:** Computational/mathematical psychology — Pitt, Myung, Cavagnaro, Navarro, Lee, Wagenmakers, Griffiths

---

## Core Claim

Standard Bayesian OED (Expected Information Gain) assumes models are correctly specified. Under parameter misspecification — the realistic case — EIG selects experiments informative for the wrong parameter regime, and model selection degrades. We demonstrate that LLM agents can diagnose and correct misspecified parameters through semantic reasoning about prediction failures, recovering +3.5 to +22pp in model discriminability. This capability depends on the *interpretability* of the parameter-to-behavior mapping: rule thresholds and sensitivity parameters are recovered; abstract mathematical parameters (value function curvature, cluster dynamics) are not.

**The one-sentence version:** LLM debate provides the parameter estimation that Bayesian OED lacks under misspecification, but only for parameters whose effects on behavior are linguistically describable.

---

## Why This Matters (for this audience)

Pitt, Myung & Zhang (2002) established that model selection and parameter estimation are entangled — you can't do one without the other. Cavagnaro et al. (2010) showed that optimal experiment design should marginalize over parameter uncertainty. In practice, most OED implementations (including our own EIG) assume fixed parameters. When those parameters are wrong:

1. EIG picks experiments informative for the wrong regime
2. The "correct" model may look wrong because its predictions are generated from wrong parameters
3. A different model may fit better at the wrong parameter values (Pitt & Myung's mimicry concern)

This is the gap between OED theory (marginalize over parameters) and OED practice (assume fixed parameters). We show that LLM-mediated debate can partially close this gap — and characterize exactly where it succeeds and fails.

**Connection to ADO:** Cavagnaro et al.'s Adaptive Design Optimization marginalizes over parameter uncertainty analytically. Our LLM debate achieves something similar but via a fundamentally different mechanism — semantic diagnosis of prediction failures rather than numerical marginalization. The advantage: it works for models where the parameter space is too complex for analytic marginalization (SUSTAIN's 5+ params, CPT's 5 params). The disadvantage: it depends on parameter interpretability.

---

## Key Results to Present

### Result 1: No mimicry, but misspecification narrows gaps substantially

Phase 1 mimicry sweep across 3 categorization models × 7 structures × parameter grids:
- **No true mimicry exists.** At every parameter setting, each model's predictions remain closer to its own GT than to any competitor. GCM, SUSTAIN, and RULEX are structurally too different.
- **But misspecification narrows discriminability severely:** GCM 61%→28%, SUSTAIN 65%→29%, RULEX 82%→16%.

This is consistent with Navarro, Pitt & Myung's (2004) finding that model distinguishability depends on parameterization. Mimicry is rare in practice, but misspecification still degrades selection.

### Result 2: EIG fails under misspecification; LLM debate recovers

Categorization M15 (3 GT × 3 conditions, GPT-4o, 5 cycles):

| GT | No-debate gap | Debate gap | Δ | Param recovery |
|---|---|---|---|---|
| GCM | 74.4% | 77.9% | +3.5pp | 85.7% (c recovered) |
| SUSTAIN | 87.7% | 85.8% | -1.9pp | 0% (invisible misspec) |
| RULEX | 58.0% | 80.4% | +22.4pp | 60.3% (rule probs) |

**Mechanism:** Agents observe prediction errors on EIG-selected experiments, reason about what parameter change would fix them ("overpredicting on Type_VI means sensitivity is too high"), and propose corrections validated by an RMSE gate against accumulated data.

### Result 3: Recovery depends on parameter interpretability, not domain

Decision M15 (EU, CPT, Priority Heuristic; 10 cycles):

| GT | No-debate | Debate | Recovery |
|---|---|---|---|
| PH | Wrong | **Correct** | 100% (thresholds exact) |
| EU | Wrong | **Correct** | 75% (r exact, temp off) |
| CPT | Wrong | Wrong | 28.4% (lambda_ yes, alpha/beta no) |

Cross-domain parallel:

| Recovery level | Categorization | Decision-making | Common factor |
|---|---|---|---|
| Strong (60-100%) | RULEX (rule probs) | PH (thresholds) | Discrete, linguistically describable |
| Moderate (75-86%) | GCM (sensitivity c) | EU (risk aversion r) | Single continuous param with clear behavioral effect |
| Weak (0-28%) | SUSTAIN (clusters) | CPT (alpha/beta) | Abstract mathematical; no intuitive behavior mapping |

**The representational-format principle:** Parameter recovery depends on whether the parameter-to-behavior mapping is linguistically describable, not on the scientific domain. LLMs can articulate "be more loss-averse" (lambda_ recovered) but struggle with "the value function should be less concave" (alpha/beta not recovered).

### Result 4: The RMSE gate is critical — local validation lets agents game revisions

First decision M15 run validated revisions against only current-cycle gambles (2-3 observations). Competitor agents accepted 79% of revisions — many improved local fit while hurting global fit. EU-GT debate was perversely worse than no-debate because PH accepted 8/8 locally-valid but globally-harmful revisions.

After switching to accumulated observations (all gambles seen so far), acceptance dropped to 49% and the perverse outcome disappeared. This mirrors Cavagnaro et al.'s insight that OED must consider cumulative evidence, not just the current trial.

### Result 5: Experiment selection carries implicit model-type priors

The arbiter layer (crux-directed experiment selection) introduces a systematic bias toward complex models:

| Condition | Categorization (M15) | Decision-making |
|---|---|---|
| No-debate | 0/3 correct | 0/3 |
| Debate | 2/3 | 2/3 |
| Arbiter | 1/3 (RULEX wrong) | 1/3 (EU wrong) |

Crux-directed selection asks "where do models disagree most?" — and models with more parameters disagree more, creating a complexity bias. This is a new form of the "experimenter bias" problem: the method for selecting discriminating experiments itself biases which model wins.

### Result 6: Formal diversification makes things worse (R-IDeA negative result)

R-IDeA (representativeness + informativeness + de-amplification) was tested as a fairness correction for EIG. Results:

| Condition | Mean gap |
|---|---|
| EIG + debate | 81.4% |
| EIG alone | 75.1% |
| R-IDeA alone | 65.4% |
| R-IDeA + debate | **53.7%** (worst) |

R-IDeA's representativeness term steers away from informative experiments, preventing the visible prediction failures that debate needs for parameter diagnosis. The representativeness objective and the diagnostic mechanism are antagonistic.

---

## Paper Structure

### 1. Introduction (1.5 pages)
- Bayesian OED assumes correct specification (Cavagnaro et al. 2010; Foster et al. 2021)
- Under misspecification, EIG optimizes for the wrong parameter regime
- Pitt & Myung (2002): parameter estimation and model selection are entangled
- We demonstrate a new mechanism for joint estimation+selection: LLM semantic diagnosis
- The mechanism has a precise boundary: parameter interpretability

### 2. Background (1.5 pages)
- **OED for model selection:** EIG, ADO, mutual information (Myung & Pitt 2009)
- **The misspecification problem:** Pitt, Myung & Zhang (2002), White (1982)
- **Model mimicry:** Wagenmakers et al. (2004), Navarro et al. (2004)
- **LLM capabilities:** Not reviewing multi-agent literature (wrong audience). Focus on LLMs' demonstrated capacity for scientific reasoning (Boiko et al. 2023)
- **Adversarial collaboration:** Mellers et al. (2001), Corcoran et al. (2023)

### 3. Models and Methods (2 pages)

#### 3.1 Cognitive Models (Domain 1: Categorization)
- GCM (Nosofsky 1986) — exemplar-based, key params: sensitivity c, attention weights
- SUSTAIN (Love et al. 2004) — clustering, key params: r, β, d, η, τ
- RULEX (Nosofsky et al. 1994) — rule-plus-exception, key params: p_single, p_conj, error_tolerance

#### 3.2 Decision Models (Domain 2: Decision-Making Under Risk)
- EU (von Neumann & Morgenstern 1944) — risk aversion r
- CPT (Tversky & Kahneman 1992) — alpha, beta, lambda_, gamma_pos, gamma_neg
- PH (Brandstätter et al. 2006) — prob_threshold, outcome_threshold_frac, phi

#### 3.3 Experiment Selection
- EIG over candidate experiments (structures × conditions / gamble groups)
- Thompson sampling for exploration-exploitation balance
- Bayesian posterior via likelihood tempering (tau=0.005)

#### 3.4 LLM Debate Protocol
- Each agent observes prediction errors on selected experiments
- Agents propose parameter revisions with justification
- RMSE gate: revision accepted only if RMSE improves against accumulated data
- Validated against model's own parameter space (reject hallucinated params)

#### 3.5 Misspecification Design
- Phase 1: parameter sweep to find gap-narrowing misspecifications
- Calibrated settings: GCM c=0.5 (gap 61%→28%), SUSTAIN r=3.0/η=0.15 (65%→29%), RULEX error_tolerance=0.25/p_single=0.3 (82%→16%)
- Analogous calibration for decision models
- Ground truth always uses correct parameters

### 4. Results (3 pages)

#### 4.1 No Mimicry, But Narrowed Gaps
- Mimicry sweep results
- Misspecification narrows but doesn't flip winners in isolation

#### 4.2 Parameter Recovery Under Debate
- Full M15 categorization results (3 GT × 3 conditions)
- Which parameters recovered, which didn't, and why
- The RMSE gate mechanism

#### 4.3 Cross-Domain Replication
- Decision M15 results
- Cross-domain parallel table
- The representational-format principle

#### 4.4 Experiment Selection Bias
- Arbiter results in both domains
- The complexity bias mechanism
- R-IDeA negative result

#### 4.5 Composition Under Double Stress
- M17 results (misspec + open design)
- Non-additive composition: synergy (GCM 87.8%) vs interference (RULEX param recovery degrades)

### 5. The Representational-Format Boundary (1 page)
- Parameters with intuitive behavioral effects → LLM recovery works
- Parameters with abstract mathematical effects → LLM recovery fails
- This is not a limitation of current LLMs — it's a structural property of linguistic reasoning about formal models
- Connection to Pitt & Myung's insight: joint estimation+selection, but with a representational constraint
- Implications: which model comparison questions can automated OED answer vs. which require human expertise?

### 6. General Discussion (1.5 pages)
- **For the OED community:** Misspecification is the norm, not the exception. Any OED system that assumes fixed parameters will underperform when parameters are wrong. LLM diagnosis is a practical, deployable solution for the class of parameters that are linguistically accessible.
- **For model comparison:** The implicit prior finding (experiment selection methods carry model-type biases) is a new methodological concern. Even well-designed OED can be biased if the selection criterion favors certain model architectures.
- **Limitations:** Synthetic data, two domains, single LLM backbone (GPT-4o), abstract parameters not recovered
- **Future:** ADO-style analytic marginalization vs LLM semantic diagnosis (complementary?), real human data, prompt enrichment for abstract parameters

### References
Pitt, Myung & Zhang (2002); Cavagnaro et al. (2010); Navarro et al. (2004); Wagenmakers et al. (2004); Nosofsky (1986, 1991); Love et al. (2004); Nosofsky et al. (1994); Tversky & Kahneman (1979, 1992); Brandstätter et al. (2006); Foster et al. (2021); Rainforth et al. (2024); Mellers et al. (2001); Corcoran et al. (2023); Myung & Pitt (2009); White (1982); Oelrich et al. (2020)

---

## Anticipated Reviewer Questions

**"Why not just use ADO (Cavagnaro et al.) which marginalizes over parameters properly?"**
ADO requires analytic or tractable numerical marginalization over the parameter space. For models like SUSTAIN (5+ params with nonlinear interactions) or CPT (5 params with reference-dependent utilities), this is computationally prohibitive. LLM diagnosis is an alternative that works where analytic marginalization doesn't — but with the interpretability constraint.

**"Is this just prompt engineering?"**
No. The parameter recovery mechanism has three components: (1) EIG selects informative experiments that expose prediction errors, (2) agents reason about what parameter change would fix the error, (3) an RMSE gate validates proposals against accumulated data. Removing any component breaks recovery. The mechanism is architectural, not prompt-dependent.

**"Could a human do this better?"**
Almost certainly yes — especially for abstract parameters. The point is not that LLMs are optimal diagnosticians but that they provide a *scalable* mechanism for the joint estimation+selection that Pitt & Myung identified as necessary. The representational-format boundary tells you when to use automated diagnosis vs. human expertise.

**"The misspecification is calibrated — is this realistic?"**
We chose gap-narrowing misspecifications to stress-test the system. In practice, misspecification may be milder (parameters slightly off) or more severe (wrong functional form). Our results bound the intermediate case. The key finding — recovery depends on interpretability, not severity — should hold across the spectrum.

**"Why only 3 models per domain?"**
These are the dominant competing models in each field. Adding a fourth (e.g., DIVA for categorization, regret theory for decisions) is architecturally trivial but would require additional calibration. The representational-format principle should hold with more models — it depends on parameter properties, not model count.

---

## Figures

1. **Misspecification narrows gaps** — bar chart showing gap% at correct vs misspecified params for all 6 models
2. **Parameter recovery timeline** — cycle-by-cycle param values converging toward GT (GCM c, RULEX probs, PH thresholds)
3. **Recovery vs interpretability** — scatter plot of all 14 parameters (both domains) by recovery% vs interpretability rating
4. **Cross-domain parallel** — paired bar charts showing categorization↔decision recovery rates
5. **RMSE gate effect** — before/after accumulated gate: acceptance rate and winner correctness
6. **Arbiter experiment selection bias** — which experiments are selected under EIG vs crux-directed
