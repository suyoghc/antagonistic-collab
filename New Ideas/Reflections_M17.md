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

1. ~~**Q1: Does adversarial structure rescue theory-motivated experimentation from Dubova's trap?** (Solved, M16.) M16's 2×2+1 factorial directly tests this: closed_no_debate (EIG-only) achieves 76-88% gap across all GTs, while open_debate (agents propose all structures) achieves 58-83%. Dubova is largely correct — theory-motivated selection underperforms formal OED on average. But a boundary condition she didn't predict emerged: for RULEX, agents outperform OED (82.7% vs 58.6% closed_debate) because they propose rule-diagnostic structures the curated registry lacks. Theory-motivated selection hurts when it *replaces* optimal selection, but helps when it *supplements* a design space with coverage gaps.~~
2. **Q2: Is the illusion of progress measurable in LLM agents?** Measure whether agents' expressed confidence correlates with actual RMSE improvement. Raw data exists (87% falsification rate, 3-5x overclaiming) but standalone analysis not yet done.
3. ~~**Q3: Does the narrowing pattern vary by ground truth?** (Solved, M16.) The open_debate condition measures narrowing cost directly as pp below the computational baseline. SUSTAIN pays -23.6pp because its diagnostic structures (continuous similarity gradients) are narratively inaccessible to LLMs. GCM pays -5.2pp (moderate — some similarity structures are nameable). RULEX pays only -3.4pp because rule-plus-exception structures are exactly what LLMs naturally propose. The narrowing cost is inversely proportional to the linguistic accessibility of each model's diagnostic structures.~~
4. ~~**Q4: Can adversarial revision serve as anti-narrowing?** (Partially solved, M16+M17.) The arbiter's crux-directed selection was designed to counteract narrowing by steering experiments toward high-disagreement regions. M16 shows it redirects narrowing rather than eliminating it: it helps SUSTAIN (+8.3pp) by steering toward similarity-diagnostic structures, but hurts RULEX (-22.2pp) by steering *away* from rule-diagnostic structures. M17 shows that combining open design (rule-biased) with arbiter (similarity-biased) partially neutralizes both biases — GCM open_arbiter achieves 87.8% via complementary cancellation. Anti-narrowing works not through a single mechanism but through composing mechanisms with opposite biases.~~
5. **Q5: Is there a cycle threshold where narrowing overtakes signal?** Run 10+ cycle debates looking for crossover. Not yet tested.
6. **Q6: Does the dual-process architecture interact with Dubova's finding?** Not directly tested as standalone question.
7. ~~**Q7: Does combining theory-motivated selection with formal OED beat both alone?** (Solved, M17.) M17's open_arbiter condition combines agent-proposed structures (theory-motivated) with crux-directed EIG scoring (formal OED). For GCM under misspecification, this achieves 87.8% — better than EIG-only (74.4% no-debate baseline), agent-only (67.3% open_debate), or either M15/M16 single-regime result. The composition is non-additive and model-dependent: synergistic for GCM (param recovery + arbiter + open proposals all contribute), neutral for SUSTAIN, and still net-negative for RULEX (42.2%). The answer is: yes, combination beats either alone, but only for models whose diagnostics align with the combined bias profile.~~

### New M17 ideas

8. **Q8: Misspecification as Dubova-trap breaker.** M17 shows SUSTAIN open_debate (77.4%) beats M16 open_debate (64.1%) — misspecification *reduces* narrowing by making agents uncertain. Test systematically: vary misspecification severity and measure narrowing. Prediction: moderate misspecification produces the least-narrow proposals because agents are uncertain enough to explore but not so lost that they propose randomly.

9. **Q9: Composition order effects.** Does it matter whether agents first recover parameters (M15 mechanism) then propose structures (M16 mechanism), or vice versa? The current system interleaves both within each cycle. A sequential design (first N cycles closed for param recovery, then switch to open) might outperform interleaved composition.

---

## Sloman — Bayesian OED under Misspecification

### Prior ideas

1. **Q1: Does R-IDeA's de-amplification reduce model-type bias?** Not yet tested. Highest priority integration.
2. **Q2: Can GBOED replace standard Bayesian inference?** Not yet tested.
3. ~~**Q3: Is debate-driven parameter correction complementary to R-IDeA?** (Partially solved, M17.) M17 demonstrates that debate-driven parameter correction composes with structure-level interventions (open design + arbiter) to produce synergy — GCM open_arbiter achieves 87.8% with 85.7% parameter recovery, exceeding any single-mechanism result. This establishes the empirical precursor: debate fixes parameters while the design-space intervention fixes structure coverage. R-IDeA's de-amplification term targets the same structure-coverage problem formally. What remains is testing whether R-IDeA + debate outperforms accidental composition + debate — i.e., whether principled structure coverage is better than our current arbiter/open-design hack.~~
4. ~~**Q4: Does active learning bias predict which GTs suffer most?** (Partially solved, M15+M17.) Sloman's framework predicts that models with narrow diagnostic regions are most vulnerable to active learning bias because adaptive selection concentrates on the wrong region under misspecification. Our data confirms this pattern across regimes: RULEX suffers most from arbiter bias (its diagnostic structures are discrete rule-exceptions, which the continuous-favoring arbiter avoids — -54.7pp in M15, -15.6pp in M17). SUSTAIN suffers most from open-design bias (its diagnostics require continuous similarity gradients that LLMs can't articulate — -23.6pp in M16). GCM is least vulnerable because its diagnostics span both continuous and discrete spaces. The per-GT vulnerability pattern is predictable from each model's diagnostic-region width.~~
5. **Q5: Can SIM's social interpolation replace LLM belief revision?** Not yet tested.
6. **Q6: Does the nuisance parameter framework predict implicit priors?** Not yet tested as a formal diagnostic, though M14-M17 data validates the framework empirically.
7. **Q7: How does excess capacity learning interact with identifiability?** Not yet tested.

### Prior integration ideas

1. **Idea 1: R-IDeA as alternative OED type.** Not yet implemented. Still highest priority.
2. **Idea 2: GBOED for posterior update.** Not yet implemented.
3. **Idea 3: Active learning bias diagnostic.** Not yet implemented.
4. ~~**Idea 4: Misspecification as bridge between computation and debate.** (Partially solved, M15+M17.) The central thesis — that misspecification is the regime where computation alone fails and debate becomes causally necessary — is now empirically established across two milestones. M15 showed debate adds +3.5 to +22pp under misspecification via parameter recovery, while hurting under correct specification. M17 showed that the combination of misspecification + open design + arbiter produces the best-ever GCM result (87.8%), demonstrating that debate's value scales with the *severity* of the information gap computation must bridge. What remains is the formal characterization (mapping our empirical findings onto Sloman's active learning bias framework) and the joint paper framing.~~
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
2. ~~**Q2: Do agents exhibit side-taking escalation?** (Solved, M14-M16.) Across all M14-M16 validation runs, agents exhibit Breithaupt's side-taking pattern at the epistemic level: 87% of auto-resolved claims are falsified, overclaiming persists at 3-5x throughout all cycles (agents predict 0.65-0.85 when actual model output is 0.10-0.50), the dominant response to falsification is "explain" (auxiliary hypothesis shielding à la Lakatos), and only 1 "abandon" action was observed across 45+ theory interpretations. Agents escalate their commitment to their assigned theory regardless of accumulating counter-evidence. However, the computational backstop (Bayesian posterior) converges correctly regardless — the narrative layer escalates while the computational layer converges, demonstrating that dual-process architecture structurally contains Breithaupt's effect.~~
3. **Q3: Can Crockett's outrage dynamics be operationalized?** Not yet tested.
4. **Q4: Is the glossary an effective anti-straw-manning intervention?** Not yet tested.
5. ~~**Q5: Narrative-driven vs. divergence-driven experiment selection.** (Solved, M16.) M16's open_debate condition is pure narrative-driven selection (agents propose all structures); closed_no_debate is pure divergence-driven (EIG scores the registry). The comparison reveals structural bias, not just preference: agents name their proposals "exception_heavy_structure," "conjunctive_rule_with_exceptions," "rule_versus_similarity_conflict" — organized around discrete, nameable properties that favor RULEX (+24pp over closed_debate) while being unable to articulate the continuous parameter variations that would probe SUSTAIN (-24pp). This is not motivational resistance to the divergence ranking but representational incompatibility — LLMs literally cannot describe the structures that maximize information gain for similarity-based models.~~
6. ~~**Q6: Does dual-process architecture resolve the tension?** (Partially solved, M14-M16.) The dual-process architecture (LLM for narrative/interpretation, computation for predictions/scoring) does not *eliminate* narrative bias but *contains* it to the design-selection layer. Agents' narrative escalation (Q2) and linguistically biased proposals (Q5) are real and measurable. But the computational backstop — Bayesian posterior updated on model-computed predictions, not agent assertions — converges correctly in 47/48 runs regardless of narrative quality. The architecture resolves the tension not by making agents unbiased but by ensuring their biases cannot corrupt the quantitative outcome. The one failure (M15 arbiter-RULEX) occurred when the arbiter's structural bias was so extreme that even the computational layer couldn't overcome the resulting evidence base.~~

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
5. ~~**Idea 5: Ablation methodology.** (Partially solved, M14-M17.) GeCCo's systematic ablation (removing iterative feedback, domain knowledge, etc.) inspired the M14-M17 factorial design. The 47/48-run matrix across 4 milestones constitutes the most thorough component ablation in the project: no-debate vs debate vs arbiter, closed vs open design, correct vs misspecified params. Each component has a signed, model-type-dependent effect (e.g., arbiter: +8.3pp SUSTAIN, -22.2pp RULEX under M16 closed). What remains is finer-grained ablation *within* the debate phase: isolating the causal contribution of the claim ledger, parameter persistence, critique rounds, and crux negotiation individually.~~
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
4. ~~**Idea 4: Ensemble diversity (Flash+Pro → Theory+Meta agents).** (Partially solved, M16-M17.) AlphaEvolve's Flash/Pro ensemble uses cheap models for breadth and expensive models for depth. Our system implements the analogous principle at the component level: open design provides breadth (diverse agent proposals, rule-biased), the arbiter provides depth (focused crux-directed selection, similarity-biased). M17 validates that this complementary-bias composition produces results exceeding either component alone (GCM 87.8%). The specific cost-based model differentiation (cheap LLM for theory agents, expensive for meta-agents) is not yet implemented, but the architectural insight — compose components with *different* biases rather than optimizing a single component — is empirically confirmed across 47/48 runs.~~
5. **Idea 5: Population-based debate — evolve agent strategies.** Not yet tested.
6. **Idea 6: Import prompt sampling into crux negotiation.** Not yet tested.
7. **Idea 7: Adversarial evolution with competing populations.** Not yet tested.
8. **Idea 8: Evolve cognitive models from scratch on Shepard types.** Not yet tested. Highest priority.
9. ~~**Idea 9: Evolutionary parameter refinement to close feedback loop.** (Partially superseded, M15+M17.) The original idea was to use evolution to refine model parameters because debate was epiphenomenal (M13). M15 showed debate-driven parameter recovery works through a different mechanism — semantic diagnosis of visible prediction failures ("my model overpredicts on Type_VI, so my attention weights must be too uniform") — achieving 85.7% recovery for GCM and 60.3% for RULEX. M17 confirms this mechanism persists under open design (85.7% GCM recovery in open_arbiter). Evolutionary refinement would address a complementary regime: when misspecification doesn't produce *visible* prediction failures (SUSTAIN shows 0% recovery across M15 and M17 because its misspecification is "invisible" at the item level). The two mechanisms are complementary, not substitutable — debate for diagnosable failures, evolution for opaque ones.~~
10. **Idea 10: Meta-evolution of debate protocols.** Not yet tested.

### New M17 ideas

11. **Idea 11: Evolve the bias-composition architecture.** M17 shows that the *specific combination* of arbiter + open design + misspec produces synergy for GCM (87.8%) but not RULEX (42.2%). An evolutionary approach could search over combinations of components (which phases to include, what crux_weight, open vs closed, etc.) optimizing for minimum variance across model types rather than maximum average gap. The fitness function is "fairness" (min gap across GTs), not "accuracy" (mean gap).

12. **Idea 12: AlphaEvolve for crux debiasing.** The arbiter's similarity bias arises from how cruxes are identified (divergence-seeking). Evolve the crux identification prompt or the crux→experiment mapping to minimize model-type variance. AlphaEvolve's mutation-evaluation loop could discover crux formulations that produce more balanced experiment selection without manual engineering of diversity constraints.

---

## OED Literature — Myung, Pitt, Navarro

### Prior ideas

1. ~~**Q1: OED as benchmark for debate quality.** (Solved, M13+M16.) M13's 18-run ablation showed computation-only (EIG) achieves the best average RMSE (0.055) and gap (87.6%). M16's 15-run factorial confirms: closed_no_debate (EIG-only) achieves 76-88% gap uniformly across all three GTs with zero LLM calls. Agent-selected experiments (open_debate) achieve 58-83% — worse on average by 7.5pp. OED is the clear quantitative benchmark against which all debate contributions are measured.~~
2. ~~**Q2: OED vs narrative convergence speed.** (Solved, M16.) OED wins on average across all GTs: closed_no_debate outperforms open_debate by 5-24pp depending on model type. But the relationship reverses for RULEX specifically: open_debate (82.7%) beats closed_debate (58.6%) because agents propose rule-diagnostic structures that the curated registry lacks. The registry was designed around the Shepard types and continuous linear-separable variants — adequate for similarity models but missing the exception-heavy structures where RULEX is most distinctive.~~
3. ~~**Q3: Three conditions (free choice / OED menu / OED only).** (Solved, M16.) The original proposal maps directly onto M16's conditions: free choice = open_debate (agents propose and EIG scores), OED menu = closed_debate (agents interpret but EIG selects from registry), OED only = closed_no_debate (no agents). Result: OED-only ≥ OED-menu > free-choice on average (86.1% ≥ 58.6-88.6% > 64.1-82.7%), but reversed for RULEX where free-choice outperforms OED-menu because agents generate structures the menu lacks.~~
4. **Q4: Lakatos-optimal design.** Not yet tested.
5. ~~**Q5: Why agents resist divergence ranking.** (Solved, M16.) The original hypothesis was that agents resist the divergence ranking for motivational reasons (preferring narratively familiar structures). M16 reveals a deeper answer: the mismatch is *representational*, not motivational. LLM agents cannot articulate the continuous parameter variations (e.g., "stimulus separation of 1.41 in 2D") that maximize information gain for similarity-based models. They naturally generate discrete, nameable structures ("rule with exceptions") because that's what language affords. Agents don't resist the ranking — they literally cannot propose the structures that would top it for certain model types. This is a Sapir-Whorf effect on experimental design.~~
6. ~~**Q6: Adaptive Bayesian OED.** (Solved, M8-M12.) Fully implemented as the computational backbone: Bayesian EIG with posterior updating and Thompson sampling selects experiments from the candidate pool each cycle. M8 introduced Thompson sampling to resolve greedy repetition; M9 added crux-directed mixture; M11-M12 expanded to continuous design spaces with parametric structures. The resulting system is the "computation alone" baseline (closed_no_debate) that achieves 76-88% gap and serves as the benchmark for all subsequent milestones.~~
7. **Q7: Fairness-aware OED ensuring minimum diagnosticity per model pair.** Proposed by M16 update. Not yet implemented.

### New M17 ideas

8. **Q8: OED under double stress as new benchmark.** M17 establishes a new benchmark regime: misspec + open design. The current best (87.8% GCM open_arbiter) was achieved by accidental bias composition. A proper fairness-aware OED should achieve comparable or better performance *uniformly* across model types. This is the test: can formal OED methods match the peak of accidental composition while avoiding its troughs (RULEX 42.2%)?

9. **Q9: Non-myopic OED for bias mitigation.** Current EIG is myopic (one-step lookahead). Under M17's regime, the first few experiments lock in parameter estimates that constrain later cycles. Non-myopic OED (Foster et al., 2021) could plan experiment sequences that first recover parameters (high-divergence structures) then discriminate models (balanced structures). The M17 finding that param recovery is modulated by design space suggests that experiment *ordering* matters — a sequential plan could separate the parameter-recovery and model-discrimination phases.

10. **Q10: Quantify the "accidental composition premium."** Across M14-M17, the best single-component result for each GT differs from the best multi-component result. Compute the gap: how much does composition add beyond the best single mechanism? If the premium is consistently positive, it argues for hybrid architectures even when principled methods (R-IDeA) exist. If the premium is zero or negative for some GTs, it argues for replacing accidental composition with principled optimization.
