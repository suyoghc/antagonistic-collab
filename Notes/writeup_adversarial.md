# What Automated Adversarial Collaboration Reveals About Scientific Methodology

**Target venue:** BJPS / Philosophy of Science / BBS (target article) / Perspectives on Psychological Science

**Target community:** Philosophy of science, metascience, Lakatos/Kuhn/Longino scholars, adversarial collaboration practitioners

---

## Core Claim

By automating the adversarial collaboration process — replacing human scientists with LLM agents backed by computational models — we create a "model organism" for studying the structure of scientific debate. The automation strips away sociological factors (career incentives, status, interpersonal dynamics) while preserving epistemic ones (theory commitment, evidence interpretation, experiment selection). What remains reveals structural features of scientific methodology that are difficult to observe in human-mediated science.

**The one-sentence version:** Automated adversarial collaboration reproduces Lakatos's research programme dynamics, reveals that every experiment selection method carries an implicit epistemic value, and shows that more sophisticated coordination between competing theorists can degrade rather than improve scientific convergence.

---

## Why This Matters (for this audience)

Mellers, Hertwig & Kahneman (2001) introduced adversarial collaboration as a practical tool for resolving scientific disputes. Since then, it has been applied in psychology (Bateman et al. 2005, Silberzahn et al. 2018), behavioral economics, and neuroscience. But adversarial collaboration remains rare because it is expensive, socially difficult, and dependent on the willingness of researchers to participate.

We automate the process: LLM agents are assigned theoretical commitments, given access to computational models, and run through a structured debate protocol. This has three philosophical payoffs:

1. **Controlled ablation of scientific methodology.** We can systematically add and remove components (experiment selection, parameter revision, meta-analysis, crux negotiation) and measure their causal effect on convergence. You cannot do this with human scientists.

2. **Large-N replication.** We run 48+ adversarial collaborations under controlled conditions. Human adversarial collaborations produce N=1 results per scientific question.

3. **Observation of methodology "in the wild."** LLM agents are not programmed to follow any philosophy of science. They exhibit Lakatosian dynamics, falsificationist patterns, and implicit epistemic values *spontaneously*. This provides evidence about whether these philosophical frameworks describe something structural about theory-mediated reasoning, rather than being merely normative prescriptions.

---

## Key Findings

### Finding 1: The system is a falsification engine

Across all runs, the ratio of falsified to confirmed claims is 44:1 (M6 analysis, extended through M14). Agents make bold predictions; experiments consistently disprove them. Convergence occurs not by proving the winner right, but by proving the losers wrong.

This is striking because no agent was instructed to follow Popperian methodology. The adversarial structure — agents defending competing theories against shared experimental evidence — produces falsification-dominated dynamics as an emergent property. The mechanism is structural: when multiple theories are simultaneously tested against the same data, most predictions will be wrong for most theories. The asymmetry between falsification and confirmation is not a methodological choice but a statistical inevitability of multi-theory testing.

**Philosophical implication:** Falsificationism may describe a structural feature of competitive theory evaluation, not a normative ideal that scientists should aspire to. The question shifts from "should scientists try to falsify?" to "do competitive evaluation structures produce falsification whether or not anyone intends it?"

### Finding 2: Lakatos's auxiliary hypothesis shielding emerges spontaneously

When agents confront falsified predictions, the dominant response is "explain" (attribute to confounds, boundary conditions, measurement limitations) rather than "revise" (change theoretical commitments) or "abandon" (retract claims). In M10's claim-responsive debate, this ratio was approximately 85:10:5 (explain:revise:abandon).

Typical example:
> "The lower-than-expected accuracy suggests that the fast presentation condition may have limited exemplar retrieval, not that the exemplar model is wrong."

This is Lakatos's protective belt in action: agents protect core theoretical commitments by modifying auxiliary assumptions. But the agents were not programmed to do this — they were given a theory and asked to interpret data. The shielding behavior emerges from the combination of (a) a committed theoretical stance and (b) the generative capacity to produce ad hoc explanations.

**Philosophical implication:** Auxiliary hypothesis shielding may not require psychological commitment (career investment, ego, reputation). It may be a structural consequence of combining theoretical commitment with explanatory fluency. LLMs have no careers to protect, yet they shield as readily as human scientists. This suggests Lakatos's framework captures something about the *logic* of theory defense, not just the *psychology* of scientists.

### Finding 3: Crux negotiation maps to real theoretical fault lines

When agents propose decisive experiments (cruxes), the accepted cruxes cluster around genuine scientific disagreements:
- "Do people store individual exemplars or use abstract rules?"
- "The role of presentation order in category learning"
- "The necessity of cluster recruitment in learning complex structures"

These are precisely the questions that cognitive scientists have debated for decades. The agents identified them independently, from their assigned theoretical commitments and the structure of the models. Acceptance rate was 15% (15/100 proposed cruxes accepted) — agents are genuinely selective, rejecting proposals they find unpersuasive.

**Philosophical implication:** The structure of theoretical disagreement may be more constrained than the sociology of science suggests. Two entities with no knowledge of the field's history, given only the formal models, converge on the same fault lines that human scientists have identified. This is consistent with Longino's (1990) view that rational disagreement is structured by the logical relationships between theories, not just by social factors.

### Finding 4: Every experiment selection method carries an implicit epistemic value

The project's most novel finding for philosophy of science: the method you use to select experiments embeds a systematic bias toward certain types of theories.

| Selection method | Representational format | Implicit bias |
|---|---|---|
| EIG (Bayesian information gain) | Numerical prediction comparison | ~Theory-agnostic |
| Crux-directed (arbiter) | "Where do theories disagree most?" | Favors complex theories (more parameters → more distinctive predictions) |
| LLM experiment proposals | Linguistically describable structures | Favors theories with discrete, nameable mechanisms (rules) |
| Diversified (R-IDeA) | Representativeness of design space | Favors no one but starves other mechanisms |

These biases are not noise — they are systematic, replicable across two scientific domains, and predictable from the selection method's representational format. The bias toward complex theories in crux-directed selection is particularly striking: asking "where do theories disagree most?" seems epistemically neutral, but it systematically advantages theories with more free parameters because those theories generate more distinctive predictions.

**Philosophical implication:** Experiment selection is not epistemically neutral. Every methodology for choosing "the most informative experiment" carries implicit values about what counts as informative, and these values systematically favor certain theory types. This connects to Longino's (2002) argument that epistemic values are embedded in methodology, but provides a concrete, measurable demonstration in a controlled setting.

This also connects to Duhem's thesis in a new way: not only is any single experiment insufficient to adjudicate between theories (because auxiliary hypotheses can always be modified), but the *method for selecting experiments* itself influences which theory appears to win. The experimenter's bias is built into the method, not just the experimenter.

### Finding 5: More sophisticated coordination → worse convergence

The most counterintuitive finding:

| Coordination level | Correct identifications |
|---|---|
| Computation only (no LLM) | 0/3 (under misspecification) |
| + LLM debate | 2/3 |
| + LLM debate + arbiter (cruxes + meta-agents) | 1/3 |

Adding a meta-level coordination mechanism (identifying cruxes, synthesizing across theories, directing experiments toward decisive tests) degrades performance. The mechanism: the meta-level coordination asks "what experiments would resolve the disagreement?" — but this question has an implicit bias toward theories that disagree more distinctively, which correlates with parametric complexity.

This replicates across both domains. In categorization, the arbiter's complexity bias caused the only wrong identification in 48 runs. In decision-making, it flipped a correct result (EU) to incorrect.

**Philosophical implication:** The finding challenges the assumption that more structured scientific methodology leads to better outcomes. Mellers et al.'s adversarial collaboration framework includes meta-level agreements about what experiments would be decisive. Our results suggest that this meta-level negotiation can itself introduce bias. The "impartial arbiter" is not impartial — the act of seeking decisive experiments privileges theories that make distinctive predictions, which are typically the more complex ones.

This connects to the Occam's razor debate: if experiment selection methods systematically favor complex theories, then the methodological playing field is already tilted against parsimony. A simple theory must overcome not just the evidence but also the experiment selection bias that generates evidence in the complex theory's favor.

### Finding 6: Parameter recovery reveals the limits of linguistic reasoning about formal models

Under parameter misspecification, LLM agents can diagnose and correct parameters whose effects on behavior are linguistically describable (rule thresholds: "be more lenient"; sensitivity: "be less selective"). They cannot diagnose parameters whose effects require mathematical understanding (value function curvature: "make the function less concave"; cluster recruitment thresholds).

This boundary is consistent across two domains. It is not a limitation of current LLMs — it reflects a structural mismatch between linguistic reasoning and mathematical parameter spaces.

**Philosophical implication:** This provides empirical evidence for the "two cultures" in theory evaluation. Some theoretical claims are naturally expressible in natural language ("this model predicts that people use rules") and others require formal notation ("the probability weighting function has curvature parameter alpha=0.88"). LLM-mediated adversarial collaboration works for the first class and fails for the second. Human-mediated adversarial collaboration presumably spans both — but the linguistic component may be doing more work than we realize.

---

## Paper Structure

### 1. Introduction: Adversarial Collaboration as a Method and as an Object of Study (2 pages)
- Mellers et al. (2001): the promise of adversarial collaboration
- Why it remains rare: social cost, coordination difficulty, N=1 per question
- Our contribution: automating the process to study it — not to replace human science, but to reveal the structural properties of competitive theory evaluation
- Preview of findings: falsification as emergent property, Lakatos validated, implicit epistemic values in experiment selection

### 2. The Automated Framework (2 pages)
- LLM agents assigned theoretical commitments (not told which theory is correct)
- Computational models generating quantitative predictions (the agents' "laboratories")
- Bayesian OED selecting experiments (the "impartial" experiment designer)
- Structured debate protocol: prediction → observation → interpretation → revision
- Crux negotiation: identifying decisive experiments by mutual agreement
- Meta-agents: third-party synthesis (Integrator) and critique (Critic)
- Two domains: categorization (GCM/SUSTAIN/RULEX) and decision-making (EU/CPT/PH)

### 3. The System as a Falsification Engine (1.5 pages)
- 44:1 falsification-to-confirmation ratio
- Falsification as structural consequence of multi-theory competition
- Winning theories need fewer revisions: Lakatos's progressive vs degenerative programmes
- Implications for Popper: falsification is not a methodology but a statistical property

### 4. Auxiliary Hypothesis Shielding Without Psychological Commitment (1.5 pages)
- Explain/revise/abandon ratios (85:10:5)
- LLMs have no careers, no ego, no reputation — yet they shield
- Examples of shielding strategies
- Lakatos's protective belt as a property of committed reasoning + explanatory fluency
- Connection to Kuhn: are paradigm shifts necessary because shielding is structurally inevitable?

### 5. The Discovery of Implicit Epistemic Values (2.5 pages — the main contribution)
- Every experiment selection method carries a bias
- The representational format determines the bias
- Worked examples:
  - EIG: why it's approximately neutral (exhaustive numerical comparison)
  - Crux-directed: why it favors complex theories (more parameters → more distinctive predictions)
  - LLM proposals: why they favor discrete theories (linguistically nameable = rule-based)
- Cross-domain replication: same biases in categorization and decision-making
- Connection to Longino's contextual empiricism: epistemic values embedded in methodology
- Connection to Duhem: experiment selection as a new form of the problem

### 6. The Coordination Paradox (1.5 pages)
- More structured methodology → worse outcomes
- The "impartial arbiter" is not impartial
- Crux-directed selection privileges complexity
- Implications for adversarial collaboration practice: meta-level agreements can introduce bias
- Connection to Occam's razor: if methodology favors complexity, parsimony requires explicit protection

### 7. The Representational-Format Boundary (1 page)
- Linguistic reasoning succeeds for linguistically describable parameters
- Fails for mathematically abstract parameters
- This is structural, not a limitation of current technology
- Implications for "two cultures" in science: natural language vs formal notation
- What human-mediated adversarial collaboration adds: mathematical reasoning

### 8. Discussion (1.5 pages)
- What automated adversarial collaboration can and cannot tell us about human science
- The "model organism" analogy: controlled, replicable, ablatable, but not fully realistic
- Missing: social dynamics, career incentives, genuine creativity, multi-year timescales
- What's preserved: theory commitment, evidence interpretation, experiment selection, convergence dynamics
- Future: using the framework to test specific philosophical claims about scientific methodology

---

## Anticipated Objections

**"LLM agents are not scientists."**
Correct. They lack genuine understanding, creativity, social motivation, and career stakes. But they share with human scientists: assigned theoretical commitments, access to empirical evidence, the capacity to generate explanations, and structured argumentation. The analogy is to model organisms in biology: *E. coli* is not a human, but studying bacterial genetics reveals principles that generalize. We study the structural properties of competitive theory evaluation in a controlled setting.

**"The findings might just reflect LLM training data."**
LLMs were trained on text that includes scientific reasoning, so their behavior may reflect learned patterns of scientific discourse. But this makes the findings *more* interesting, not less: if Lakatosian dynamics are present in scientific discourse generally, and LLMs reproduce those patterns, this is evidence that the patterns are robust features of theory-mediated reasoning, not artifacts of specific historical contingencies.

**"You're conflating epistemic values with statistical artifacts."**
The complexity bias in crux-directed selection is both: it's a statistical property (more parameters → more distinctive predictions) that functions as an epistemic value (experiment selection favors complexity). The philosophical point is precisely that epistemic values can emerge from apparently neutral statistical procedures. You don't have to *intend* a bias toward complexity for your methodology to produce one.

**"Adversarial collaboration in practice is much richer than this."**
Agreed. Human adversarial collaborations involve multi-year negotiations, creative experiment design, genuine theoretical innovation, and social dynamics. Our framework captures only the structural skeleton: commitment, evidence, interpretation, revision. But this is the point — by stripping away the sociological, we can see what remains. If falsification dominance and auxiliary hypothesis shielding appear even without social factors, they are structural, not sociological.

**"Isn't the 'more coordination → worse' finding just a bad arbiter?"**
We show the mechanism is structural, not implementational. Crux-directed selection will always favor theories with more distinctive predictions because that's what "disagreement" means for parametrically richer models. Any arbiter that asks "where do theories disagree most?" will have this bias. The finding is about the logic of seeking decisive experiments, not about our particular implementation.

---

## Key References (Philosophy)

- Duhem, P. (1906/1954). *The Aim and Structure of Physical Theory*
- Kuhn, T. S. (1962). *The Structure of Scientific Revolutions*
- Lakatos, I. (1970). Falsification and the methodology of scientific research programmes
- Longino, H. (1990). *Science as Social Knowledge*
- Longino, H. (2002). *The Fate of Knowledge*
- Mellers, B., Hertwig, R., & Kahneman, D. (2001). Do frequency representations eliminate conjunction effects? An exercise in adversarial collaboration
- Popper, K. (1959). *The Logic of Scientific Discovery*
- Stanford, P. K. (2006). *Exceeding Our Grasp: Science, History, and the Problem of Unconceived Alternatives*
- van Fraassen, B. (1980). *The Scientific Image*
- Bateman, I., et al. (2005). Testing competing models of loss aversion (adversarial collaboration)
- Corcoran, A. W., Hohwy, J., & Friston, K. J. (2023). Accelerating scientific progress through Bayesian adversarial collaboration

## Key References (Cognitive Science Methods)

- Cavagnaro, D. R., Myung, J. I., Pitt, M. A., & Kujala, J. V. (2010). Adaptive design optimization
- Pitt, M. A., Myung, I. J., & Zhang, S. (2002). Toward a method of selecting among computational models of cognition
- Navarro, D. J., Pitt, M. A., & Myung, I. J. (2004). Assessing the distinguishability of models
- Wagenmakers, E.-J., et al. (2004). Assessing model mimicry using the parametric bootstrap
- Silberzahn, R., et al. (2018). Many analysts, one data set

---

## Figures

1. **Falsification ratio** — bar chart of claims falsified vs confirmed across all runs
2. **Auxiliary hypothesis shielding** — distribution of explain/revise/abandon responses
3. **Implicit bias table** — the 4-row table from Finding 4, visualized as a matrix
4. **The coordination paradox** — stacked bar showing computation → debate → arbiter degradation
5. **Cross-domain replication** — side-by-side showing same bias pattern in two domains
6. **Crux acceptance network** — which cruxes were proposed vs accepted, clustered by theoretical fault line
