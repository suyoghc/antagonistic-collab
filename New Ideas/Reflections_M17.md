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

---

# Updated Ideas per Paper (post-M17)

Status key: ~~Strikethrough~~ = solved/substantially addressed, with milestone noted.

---

## Dubova — Theory-Motivated Experimentation

### Prior ideas

1. ~~**Q1: Does adversarial structure rescue theory-motivated experimentation from Dubova's trap?** (Solved, M16. Answer: largely confirmed — agent-selected experiments underperform OED on average. Boundary condition: RULEX agents outperform OED when registry has coverage gap.)~~
2. **Q2: Is the illusion of progress measurable in LLM agents?** Measure whether agents' expressed confidence correlates with actual RMSE improvement. Raw data exists (87% falsification rate, 3-5x overclaiming) but standalone analysis not yet done.
3. ~~**Q3: Does the narrowing pattern vary by ground truth?** (Solved, M16. SUSTAIN -23.6pp, GCM -5.2pp, RULEX -3.4pp. Narrowing cost is inversely proportional to linguistic accessibility of diagnostic structures.)~~
4. ~~**Q4: Can adversarial revision serve as anti-narrowing?** (Partially solved, M16. Arbiter redirects narrowing rather than eliminating it. Combined open+arbiter partially neutralizes both biases.)~~
5. **Q5: Is there a cycle threshold where narrowing overtakes signal?** Run 10+ cycle debates looking for crossover. Not yet tested.
6. **Q6: Does the dual-process architecture interact with Dubova's finding?** Not directly tested as standalone question.
7. ~~**Q7: Does combining theory-motivated selection with formal OED beat both alone?** (Solved, M17. Answer: yes, for GCM — open_arbiter under misspecification achieves 87.8%, better than either OED-only or agent-only. But composition is model-dependent, not universal.)~~

### New M17 ideas

8. **Q8: Misspecification as Dubova-trap breaker.** M17 shows SUSTAIN open_debate (77.4%) beats M16 open_debate (64.1%) — misspecification *reduces* narrowing by making agents uncertain. Test systematically: vary misspecification severity and measure narrowing. Prediction: moderate misspecification produces the least-narrow proposals because agents are uncertain enough to explore but not so lost that they propose randomly.

9. **Q9: Composition order effects.** Does it matter whether agents first recover parameters (M15 mechanism) then propose structures (M16 mechanism), or vice versa? The current system interleaves both within each cycle. A sequential design (first N cycles closed for param recovery, then switch to open) might outperform interleaved composition.

---

## Sloman — Bayesian OED under Misspecification

### Prior ideas

1. **Q1: Does R-IDeA's de-amplification reduce model-type bias?** Not yet tested. Highest priority integration.
2. **Q2: Can GBOED replace standard Bayesian inference?** Not yet tested.
3. ~~**Q3: Is debate-driven parameter correction complementary to R-IDeA?** (Partially solved, M17. M17 shows debate param correction + open design compose synergistically for GCM. This is the empirical precursor — R-IDeA would replace the "accidental" composition with principled multi-objective optimization.)~~
4. ~~**Q4: Does active learning bias predict which GTs suffer most?** (Partially solved, M15+M17. RULEX suffers most from arbiter bias under misspec; SUSTAIN suffers most from open-design bias. Pattern consistent with Sloman's prediction that models with narrow diagnostic regions are most vulnerable to active learning bias.)~~
5. **Q5: Can SIM's social interpolation replace LLM belief revision?** Not yet tested.
6. **Q6: Does the nuisance parameter framework predict implicit priors?** Not yet tested as a formal diagnostic, though M14-M17 data validates the framework empirically.
7. **Q7: How does excess capacity learning interact with identifiability?** Not yet tested.

### Prior integration ideas

1. **Idea 1: R-IDeA as alternative OED type.** Not yet implemented. Still highest priority.
2. **Idea 2: GBOED for posterior update.** Not yet implemented.
3. **Idea 3: Active learning bias diagnostic.** Not yet implemented.
4. ~~**Idea 4: Misspecification as bridge between computation and debate.** (Partially solved, M15+M17. The empirical core now exists: M15 shows debate helps under misspec, M17 shows composition amplifies the benefit. What remains is the formal characterization + joint paper.)~~
5. **Idea 5: Negative transfer → negative debate.** Not yet tested.
6. **Idea 6: Parameter estimation diagnostics.** Not yet tested.
7. **Idea 7: LLM agents as testbed for active learning bias.** Not formally tested as standalone.
8. **Idea 8: Architectural prescription for automating science.** Not yet written up.

### New M17 ideas

9. **Q8: Does R-IDeA replicate M17's accidental synergy?** M17's GCM open_arbiter (87.8%) arises from accidental composition of complementary biases. R-IDeA formally optimizes representativeness + informativeness + de-amplification. Test: does R-IDeA on the M17 regime (misspec + open) match or beat 87.8% *without* needing to know which biases exist? This is the critical test of principled vs. accidental composition.

10. **Q9: Bias-aware OED.** M14-M17 shows every component has a model-type-dependent effect. Design a meta-OED that tracks per-model-type diagnosticity and ensures minimum coverage across all model pairs, not just maximum average EIG. This is OED Q7 restated in Sloman's framework.

11. **Q10: Parameter recovery × design space interaction.** M17 shows param recovery is modulated by design space (GCM: 85.7% closed, 42.9% open_debate, 85.7% open_arbiter; RULEX: 60.3% closed, 46.3% open_debate, 0% open_arbiter). Does R-IDeA's de-amplification term predict this interaction? If de-amplification reduces the parameter estimation error that motivates debate-driven recovery, the two mechanisms may be substitutable rather than complementary.

---

## Breithaupt/Crockett — Narrative Side-Taking

### Prior ideas

1. **Q1: Does narrative coherence predict persuasion independent of RMSE?** Not yet tested as standalone analysis.
2. ~~**Q2: Do agents exhibit side-taking escalation?** (Solved, M14-M16. Yes: 87% falsification rate, persistent 3-5x overclaiming, dominant "explain" response to falsification, only 1 "abandon" across all runs. Breithaupt's escalation at epistemic level, bounded by computational backstop.)~~
3. **Q3: Can Crockett's outrage dynamics be operationalized?** Not yet tested.
4. **Q4: Is the glossary an effective anti-straw-manning intervention?** Not yet tested.
5. ~~**Q5: Narrative-driven vs. divergence-driven experiment selection.** (Solved, M16. Confirmed: LLM proposals are systematically biased toward linguistically describable structures. Effect sizes: RULEX +24pp, SUSTAIN -24pp.)~~
6. ~~**Q6: Does dual-process architecture resolve the tension?** (Partially solved, M14-M16. The architecture contains but doesn't eliminate narrative bias — narrative coercion is real but computational backstop prevents it from corrupting final results.)~~

### New M17 ideas

7. **Q7: Epistemic humility as narrative disruptor.** M17 shows misspecification reduces narrative narrowing (SUSTAIN open_debate: 77.4% under misspec vs 64.1% under correct spec). Breithaupt's model predicts narrative compels side-taking when agents are *confident*. When predictions visibly fail, narrative grip loosens. Test: vary agents' awareness of their own prediction failures and measure narrative diversity of proposals. Prediction: agents who see their RMSE get worse propose more diverse structures than agents shielded from feedback.

8. **Q8: The arbiter as narrator.** The arbiter's crux machinery is itself a narrative device — it says "the critical question is X" and focuses all agents on that question. M17 shows this narrative focus helps when the narrator's perspective aligns with the ground truth (GCM: +20.5pp) and hurts when it doesn't (RULEX: -15.6pp). This is Breithaupt's moderator-as-empathizer: the arbiter "takes sides" by choosing which disagreements matter.

---

## GeCCo — LLM-Generated Cognitive Models

### Prior ideas

1. **Idea 1: GeCCo as model generator for debate.** Not yet tested.
2. **Idea 2: Adversarial refinement of GeCCo models.** Not yet tested.
3. **Idea 3: GeCCo + EIG closed loop.** Not yet tested.
4. **Idea 4: BIC-based model posterior.** Not yet tested.
5. ~~**Idea 5: Ablation methodology.** (Partially solved, M14-M17. Component-level ablation done: no-debate, debate, arbiter, open design across correct/misspec regimes. 47/48 run factorial reveals signed, model-type-dependent effects per component. Remaining: finer-grained ablation of claim ledger, parameter persistence, critique phase individually.)~~
6. **Idea 6: Overclaiming metric exported to GeCCo.** Not yet tested.
7. **Idea 7: Adversarial GeCCo with competing generators.** Not yet tested.
8. **Idea 8: Ground-truth recovery with debate.** Not yet tested.
9. **Idea 9: GeCCo on categorization domain.** Not yet tested. M16 prediction sharpened: RULEX recovery > SUSTAIN recovery.
10. **Idea 10: Learning curves as shared discriminator.** Not yet tested.

### New M17 ideas

11. **Idea 11: GeCCo models as fourth agent under misspecification.** M17's regime (wrong params + open design) is the ideal testbed for GeCCo-generated models: if GeCCo discovers a novel model, does it survive adversarial debate under the hardest conditions? The linguistic accessibility bias predicts GeCCo will generate RULEX-like models that benefit from open design but are vulnerable to arbiter bias — exactly the interaction M17 documents.

12. **Idea 12: GeCCo's feedback loop vs. debate's feedback loop — composition test.** GeCCo changes the *model itself*; debate changes the *context* (params + structures). M17 shows context-change helps under misspecification. Does model-change (GeCCo) + context-change (debate) compose, or is one strictly better? Run GeCCo iterative refinement alongside debate-driven parameter recovery on the same models and measure whether the combination exceeds either alone.

13. **Idea 13: Predict GeCCo's linguistic accessibility bias from M16-M17 data.** M16 quantified the bias: +24pp for linguistically accessible models, -24pp for continuous models. Use these effect sizes to predict which GeCCo-generated models will be over/under-represented before running GeCCo on categorization data. If predictions hold, this validates the linguistic accessibility bias as a general property of LLM-mediated science.

---

## AlphaEvolve — Evolutionary Algorithm Discovery

### Prior ideas

1. **Idea 1: AlphaEvolve generates models, debate adjudicates.** Not yet tested.
2. **Idea 2: Evolve the evaluation function.** Not yet tested. M15-M16 update warned: evolving the evaluator doesn't escape implicit priors.
3. **Idea 3: Evolve EIG approximation heuristics.** Not yet tested.
4. ~~**Idea 4: Ensemble diversity (Flash+Pro → Theory+Meta agents).** (Partially solved, M16-M17. The complementary-bias-composition principle is validated: arbiter (depth, similarity bias) + open design (breadth, rule bias) partially cancel. The specific Flash/Pro cost differentiation is not implemented, but the architectural insight is confirmed.)~~
5. **Idea 5: Population-based debate — evolve agent strategies.** Not yet tested.
6. **Idea 6: Import prompt sampling into crux negotiation.** Not yet tested.
7. **Idea 7: Adversarial evolution with competing populations.** Not yet tested.
8. **Idea 8: Evolve cognitive models from scratch on Shepard types.** Not yet tested. Highest priority.
9. ~~**Idea 9: Evolutionary parameter refinement to close feedback loop.** (Partially superseded, M15. Debate-driven parameter recovery works through semantic diagnosis of visible failures — 85.7% recovery for GCM, 60.3% for RULEX. M17 confirms param recovery still works under open design. Evolutionary refinement addresses a different regime: when failures are NOT semantically visible. The two mechanisms are complementary, not substitutable.)~~
10. **Idea 10: Meta-evolution of debate protocols.** Not yet tested.

### New M17 ideas

11. **Idea 11: Evolve the bias-composition architecture.** M17 shows that the *specific combination* of arbiter + open design + misspec produces synergy for GCM (87.8%) but not RULEX (42.2%). An evolutionary approach could search over combinations of components (which phases to include, what crux_weight, open vs closed, etc.) optimizing for minimum variance across model types rather than maximum average gap. The fitness function is "fairness" (min gap across GTs), not "accuracy" (mean gap).

12. **Idea 12: AlphaEvolve for crux debiasing.** The arbiter's similarity bias arises from how cruxes are identified (divergence-seeking). Evolve the crux identification prompt or the crux→experiment mapping to minimize model-type variance. AlphaEvolve's mutation-evaluation loop could discover crux formulations that produce more balanced experiment selection without manual engineering of diversity constraints.

---

## OED Literature — Myung, Pitt, Navarro

### Prior ideas

1. ~~**Q1: OED as benchmark for debate quality.** (Solved, M13+M16. OED-selected achieves 76-88% gap; agent-selected achieves 58-83%.)~~
2. ~~**Q2: OED vs narrative convergence speed.** (Solved, M16. OED wins on average; agents win for RULEX when registry has coverage gap.)~~
3. ~~**Q3: Three conditions (free choice / OED menu / OED only).** (Solved, M16. Maps to open_debate / closed_debate / closed_no_debate. Result: OED-only ≥ OED-menu > free-choice on average, reversed for RULEX.)~~
4. **Q4: Lakatos-optimal design.** Not yet tested.
5. ~~**Q5: Why agents resist divergence ranking.** (Solved, M16. Structural mismatch: structures LLMs can articulate ≠ structures that maximize information gain. Not motivational resistance but representational incompatibility.)~~
6. ~~**Q6: Adaptive Bayesian OED.** (Solved, M8-M12. EIG + posterior update + Thompson sampling is the computational backbone.)~~
7. **Q7: Fairness-aware OED ensuring minimum diagnosticity per model pair.** Proposed by M16 update. Not yet implemented.

### New M17 ideas

8. **Q8: OED under double stress as new benchmark.** M17 establishes a new benchmark regime: misspec + open design. The current best (87.8% GCM open_arbiter) was achieved by accidental bias composition. A proper fairness-aware OED should achieve comparable or better performance *uniformly* across model types. This is the test: can formal OED methods match the peak of accidental composition while avoiding its troughs (RULEX 42.2%)?

9. **Q9: Non-myopic OED for bias mitigation.** Current EIG is myopic (one-step lookahead). Under M17's regime, the first few experiments lock in parameter estimates that constrain later cycles. Non-myopic OED (Foster et al., 2021) could plan experiment sequences that first recover parameters (high-divergence structures) then discriminate models (balanced structures). The M17 finding that param recovery is modulated by design space suggests that experiment *ordering* matters — a sequential plan could separate the parameter-recovery and model-discrimination phases.

10. **Q10: Quantify the "accidental composition premium."** Across M14-M17, the best single-component result for each GT differs from the best multi-component result. Compute the gap: how much does composition add beyond the best single mechanism? If the premium is consistently positive, it argues for hybrid architectures even when principled methods (R-IDeA) exist. If the premium is zero or negative for some GTs, it argues for replacing accidental composition with principled optimization.
