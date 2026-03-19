# Reflections on M14–M17 in Context of Prior Paper Analysis

*Written after completing the M14–M17 factorial (47/48 correct across 4 milestones)*

---

## Dubova — "Against Theory-Motivated Experimentation"

Dubova's headline claim — that theory-motivated experiment selection performs *worse* than random — is **largely vindicated but with a boundary condition she didn't predict**.

M16 is the cleanest test. Closed_no_debate (EIG-only, no agent involvement) achieves 76-88% gap uniformly. Open_debate (agents propose all experiments) achieves 58-83% — worse on average, exactly as Dubova predicts. The mechanism is exactly her "vicious cycle": agents propose narratively compelling structures, which narrows the evidence base toward structures that confirm familiar patterns.

**But the boundary condition matters.** For RULEX, agent proposals *outperform* the curated registry because they generate rule-diagnostic structures (exception-heavy, conjunction-based) that the continuous-structure registry lacks. Dubova's model assumes the formal method has access to the complete design space. When it doesn't — when the registry has a coverage gap — theory-motivated selection fills the gap. This is the constructive resolution: theory-motivated selection is harmful when it *replaces* optimal selection but beneficial when it *supplements* it.

M17 sharpens this further: under misspecification, SUSTAIN open_debate (77.4%) beats M16 open_debate (64.1%). Why? Misspecification creates visible prediction errors, which motivate agents to propose *different* structures than they would under correct specification. Dubova's narrowing cycle assumes agents are confident in their theories. When agents are uncertain (because their predictions keep failing), the cycle breaks. **Misspecification is the antidote to Dubova's trap.**

---

## Sloman — Bayesian OED under Misspecification

Sloman's work is the most technically precise match to what we found. Three connections:

**1. Active learning bias is real and we measured it.** Sloman (2022) showed that Bayesian adaptive OED under misspecification amplifies rather than corrects model errors. M15 demonstrates this empirically: under misspecification, EIG-only (no-debate) selects experiments informative for the *wrong* parameter regime. RULEX's no-debate gap (58%) is low precisely because EIG picks structures that are diagnostic for the wrong sensitivity setting. Debate's parameter recovery (+22pp) is the empirical counterpart to Sloman's theoretical fix — correcting the parameters that EIG assumes are correct.

**2. R-IDeA's three objectives map onto our hybrid architecture, imperfectly.** R-IDeA optimizes representativeness + informativeness + de-amplification simultaneously. Our system achieves something similar through composition: the registry provides representativeness (broad coverage), EIG provides informativeness, and agent proposals partially address de-amplification (by testing structures the current model fails on). But we lack R-IDeA's *formal* de-amplification term — our version is accidental, arising from the complementary biases of arbiter and open design. M17's GCM open_arbiter (87.8%) might be the empirical ceiling of what informal composition can achieve; R-IDeA could potentially match this *without* needing the specific arbiter+open combination to happen to cancel.

**3. The nuisance parameter framework predicts our implicit priors.** Sloman's UAI 2024 work shows that nuisance parameters don't just add noise — they systematically bias estimation of target parameters through adaptive selection. Each of our components has a "target" and "nuisance": the arbiter targets maximum disagreement (useful) but its nuisance effect is similarity-structure concentration (harmful for RULEX). Open design targets expressive structures (useful) but its nuisance effect is linguistic-accessibility bias (harmful for SUSTAIN). The framework predicts that you can't eliminate these nuisance effects — you can only balance them across components.

**The practical implication for the project:** Implementing R-IDeA as an alternative OED type would let us test whether formal multi-objective optimization achieves the same bias-cancellation that we currently get from the *accidental* composition of arbiter + open design — but reliably and without needing to know which biases exist in advance.

---

## Breithaupt/Crockett — Narrative Side-Taking

The M16 open-design results are the strongest evidence for Breithaupt's thesis that narrative compels side-taking independent of evidence quality. When agents propose experiment structures, the *names they give* reveal the bias: "exception_heavy_structure," "conjunctive_rule_with_exceptions," "rule_versus_similarity_conflict." These are narratively organized around discrete, nameable structural properties — the kind of features that make for compelling stories about how models differ.

M17 adds a new dimension: **narrative bias interacts with epistemic state**. Under correct specification, agents are confident and their proposals are narrowly narrative (M16 open_debate: -23.6pp for SUSTAIN). Under misspecification, agents' predictions fail, creating epistemic humility that broadens their proposals (M17 open_debate: only -10.3pp for SUSTAIN vs baseline). Crockett's point about motivated reasoning applies specifically to the confident case — when agents have no reason to doubt their theory, narrative dominates. When their predictions visibly fail, evidence partially overrides narrative.

The computation-only pipeline is the anti-Breithaupt condition: it has no narrative, no protagonist, no antagonist. It evaluates structures by a metric that doesn't privilege any story. This is why it's the most reliable component — and why it's also the least creative. Breithaupt would predict exactly this tradeoff: narrative enables discovery (agents propose structures computation would never generate) at the cost of systematic bias (those proposals favor linguistically accessible models).

---

## GeCCo — LLM-Generated Cognitive Models

M16 makes a specific prediction about GeCCo that can now be stated precisely: **GeCCo should rediscover RULEX-like models more easily than SUSTAIN-like models**, because rule-plus-exception mechanisms are linguistically describable ("if dimension 1 > threshold, category A, except items 3 and 7") while cluster-recruitment mechanisms require continuous parameter interactions that are harder to express in code comments or docstrings.

M17's finding that parameter recovery is modulated by design space connects to GeCCo's ablation result (iterative feedback is the strongest driver). GeCCo's feedback loop changes the *model itself* — always useful because the model is the thing being optimized. Our feedback loop changes the *context* (parameters, experiment structures) — useful only when the context has a gap. This explains why GeCCo's ablation effect (beta = -17.04) is so much stronger than debate's typical effect in our system.

The integration path is now clearer: GeCCo discovers candidate models, our system adjudicates between them. But M16-M17 warn that the *adjudication* must use formal OED (not narrative-driven selection) to avoid biasing the comparison toward whichever model type is more linguistically accessible.

---

## AlphaEvolve — Evolutionary Algorithm Discovery

The complementary-bias-composition principle from M17 has a direct parallel in AlphaEvolve's Flash/Pro ensemble: cheap models for breadth, expensive models for depth. Our system composes arbiter (depth — focused on high-divergence regions) with open design (breadth — diverse agent proposals). M17 shows this composition can produce results better than either alone (GCM 87.8%).

But the deeper connection is about **what kind of problem each system solves**. AlphaEvolve searches vast combinatorial spaces where the evaluation function is known and cheap. Our system searches small model spaces where the evaluation function itself is contested (which experiments to use IS the question). AlphaEvolve's tight loop works because evaluation is objective. Our system's messy composition works because evaluation is perspective-dependent — and multiple perspectives partially cancel each other's biases.

M17's finding that misspecification + open design can *help* (GCM 87.8%) is something AlphaEvolve's architecture cannot replicate. Evolution doesn't benefit from starting with wrong parameters. But adversarial debate does — because wrong parameters create visible prediction failures that motivate exploration. This is the specific capability that separates explanation-seeking systems (ours) from explanation-indifferent systems (AlphaEvolve).

---

## OED Literature — Myung, Pitt, Navarro

The OED comparison is now complete across four regimes:

| Regime | EIG-only | + Debate | + Arbiter | + Open Design |
|---|---|---|---|---|
| Correct spec (M14/M16) | **Best** (76-88%) | Noise (-7.5pp mean) | Biased (+8/-22pp) | Biased (-11pp mean) |
| Misspec (M15) | Degraded (58-88%) | **Helps** (+8pp mean) | Model-dependent | n/a |
| Misspec + Open (M17) | n/a | Mixed | **Synergy** (GCM 87.8%) | Rescue (RULEX) |

The central finding, stated in OED terms: **formal OED is approximately model-agnostic because it evaluates exhaustively, but its model-agnosticism degrades under misspecification** (Sloman's active learning bias). Theory-motivated alternatives embed implicit model-type priors, but these priors are *complementary* across mechanisms, and their composition under misspecification can outperform formal OED alone.

This challenges the OED literature's implicit assumption that more formal = more reliable. Under the hardest conditions (M17), the hybrid system's messy composition (EIG + debate + arbiter + open design) produced 87.8% on GCM — better than EIG-only has ever achieved on any model. The formalism is necessary (EIG provides the backbone), but insufficient under misspecification without the informal mechanisms that address parameter error and design-space coverage.

Navarro's (2019) point that methodology constrains conclusions now has four regime-specific instantiations. The practical recommendation for the OED community: evaluate design methods not just for average performance but for *fairness across model types*. A method that achieves 96% on one model type and 3% on another is not merely variable — it's systematically biased.

---

## The Meta-Principle: The Medium of Reasoning as Implicit Prior

Across all six papers, a single pattern emerges: **every reasoning mechanism can only "see" certain kinds of model differences, and this determines which models it favors.**

- **Computation (EIG)** reasons through *information theory* — format-neutral, model-agnostic, but blind to design-space coverage gaps.
- **Crux reasoning (arbiter)** reasons through *points of disagreement* — favors models with visible gradients in continuous space (similarity-based models).
- **Linguistic reasoning (LLM agents)** reasons through *natural language descriptions* — favors models whose diagnostic structures have clear verbal handles (rule-based models).
- **Evolutionary search (AlphaEvolve)** reasons through *fitness landscape* — favors models whose performance differences are smooth and gradient-following.
- **Iterative feedback (GeCCo)** reasons through *model modification* — always useful because it changes the object being evaluated, not just the evaluation context.

No single mechanism is unbiased. Objectivity is not the absence of perspective but the triangulation of multiple perspectives (Longino, 1990). The practical design principle: build hybrid systems from components with *different* biases, and measure fairness across model types, not just average accuracy.

Dubova warns that theory-motivation narrows. Sloman formalizes why. Breithaupt explains the narrative mechanism. GeCCo shows that model-level feedback avoids the problem entirely. AlphaEvolve shows that explanation-indifferent search avoids it differently. And our M14–M17 factorial shows that composing mechanisms with complementary biases can approximate the model-agnosticism that no single mechanism achieves alone — with the caveat that this composition is currently accidental, not principled.

The next step — implementing R-IDeA — would test whether *principled* multi-objective optimization achieves the same bias-cancellation more reliably than our current accidental composition.
