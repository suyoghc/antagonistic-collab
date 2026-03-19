# Tom Griffiths — Connection Points to Adversarial Collaboration

*Brainstorming research questions that connect our M14–M17 findings to
Griffiths' research program. Organized by clusters of his work.*

---

## 1. Resource Rationality and Bounded Agents

Griffiths' most distinctive contribution: "irrational" behavior is optimal
given computational constraints (Lieder & Griffiths, 2020, *BBS*;
Griffiths, Lieder & Goodman, 2015, *Topics in Cognitive Science*).

Our LLM agents are themselves resource-rational agents — they can't do
exact Bayesian inference, so they approximate via debate, narrative, and
heuristic reasoning. The question is whether their approximation strategy
is *well-calibrated* to the problem.

**Q1: Can the arbiter's model-type bias be derived as resource-rational?**
The arbiter steers toward continuous similarity structures because
divergence is most *visible* there (smooth gradients → easy to detect
disagreement). A resource-rational agent that minimizes computation per
unit of information gained would do exactly this — continuous structures
are cheaper to evaluate for disagreement than discrete structures where
differences are localized to specific boundary items. The bias might not be
a bug but an optimal approximation under bounded divergence computation.
If this framing holds, the question becomes: what computational cost
structure would produce a *model-agnostic* approximation?

**Q2: Are LLM agents' narrative proposals resource-rational experiment design?**
LLM agents propose "exception_heavy_structure" instead of
"sampled_ls_2d_sep1.41_28" because natural language is their computational
medium — discrete nameable structures are cheaper to reason about than
continuous parameter variations. Griffiths' framework predicts this: the
agent's proposal bias reflects the cost structure of linguistic reasoning.
Can we derive the specific +24pp RULEX / -24pp SUSTAIN bias from a
resource-rational model of language-mediated experiment design?

**Q3: Does the "debate is epiphenomenal under correct specification"
finding follow from resource rationality?** Under correct specification,
the computational pipeline already solves the problem optimally (76-88%
gap). Debate adds cognitive cost without information gain — a
resource-rational system should skip it. Under misspecification, the
computation is stuck (wrong parameter regime), so the additional cost of
debate is justified by the information it provides (parameter recovery).
The regime-dependence of debate's value might be derivable from a
cost-benefit analysis.

---

## 2. Models of Categorization as Approximations

Sanborn, Griffiths & Navarro (2010, *Psych Review*) argued that GCM,
prototype models, and SUSTAIN-like clustering are different *approximation
algorithms* for the same Bayesian ideal — exemplar = store all data,
prototype = maximum-likelihood summary, clustering = particle filter.

This directly intersects our project because we treat GCM, SUSTAIN, and
RULEX as genuinely different theories. If Griffiths is right that they're
approximations to the same computation, our project's question shifts.

**Q4: Does the adversarial debate framework find evidence for or against
Griffiths' unification?** Our M15 Phase 1a mimicry sweep found "no true
mimicry" — at every parameter setting tested, each model's predictions
remain closer to its own ground truth than to any competitor. GCM, SUSTAIN,
and RULEX are structurally too different for parameter changes alone to
make them indistinguishable. This challenges the strong version of the
approximation-to-same-ideal thesis — if they were really the same
computation at different approximation levels, mimicry should be possible.
RULEX (rule-based) sits outside Griffiths' unification entirely, which may
explain why it's the model with the most distinctive behavior across all
our milestones.

**Q5: Can a Bayesian nonparametric model (Griffiths' Dirichlet Process
mixture) serve as a fourth agent?** Griffiths, Canini, Sanborn & Navarro
(2007) showed that a hierarchical Dirichlet process subsumes GCM and
SUSTAIN as special cases. Adding this model as a fourth debate agent
would directly test whether the "unifying" model outcompetes the
specialized models in adversarial adjudication — or whether it's too
flexible (overfitting) to win on Bayesian model comparison criteria.

**Q6: Do the model-type biases we found correspond to
approximation-quality differences?** Under Griffiths' framework, GCM
(store-all-data) is the most faithful approximation, SUSTAIN (particle
filter) is intermediate, and prototype (collapse to mean) is the coarsest.
Our arbiter helps GCM and SUSTAIN but hurts RULEX. If the arbiter's bias
tracks approximation faithfulness (more faithful approximations produce
predictions that diverge more smoothly → easier for crux machinery to
detect), then the model-type bias is actually an *approximation-quality*
bias. This would be a novel connection between the computational-level
analysis (Griffiths) and the algorithmic-level selection bias (our work).

---

## 3. AutoRA and Automated Experiment Design

Griffiths' lab built AutoRA (Musslick et al., 2024), the closest existing
system to ours: closed-loop automation with theorist components (model
fitting), experimentalist components (BOED-style selection), and experiment
runners. Also: Musslick et al. (2025, *PNAS*) on automating behavioral
science.

**Q7: What does adversarial debate add beyond AutoRA's architecture?**
AutoRA treats model comparison as a computational optimization problem
without an argumentative layer. Our M14 confirms this works under correct
specification (18/18 correct, best RMSE). But M15 shows debate adds value
under misspecification (+22pp RULEX) via parameter recovery — a capability
AutoRA's purely computational loop cannot provide. The question for
Griffiths: should AutoRA incorporate a "debate" module for
misspecification-prone domains? Or is there a principled computational
alternative (like marginalizing over parameter uncertainty, as in
Cavagnaro et al., 2010)?

**Q8: Can AutoRA's experimentalist module be plugged into our debate
framework?** Our EIG computation is a simplified version of what AutoRA's
experimentalist does. Replacing our EIG with AutoRA's more sophisticated
BOED (which can handle continuous design spaces natively, use gradient-based
optimization, and marginalize over nuisance parameters) could test whether
the implicit model-type biases we found persist under a more principled
experiment selection method.

**Q9: Does the "implicit model-type prior" finding generalize to AutoRA?**
We found that every component of our system carries an implicit bias toward
certain model types. Does AutoRA's experimentalist also carry such biases?
If AutoRA uses EIG with a fixed set of candidate experiments (like our
curated registry), it should show the same registry-dependent bias we
documented. This would be a contribution from our project to AutoRA: a
diagnostic for checking whether automated experiment selection is
model-type-fair.

---

## 4. Cultural Evolution and Iterated Learning

Griffiths & Kalish (2007, *Cognitive Science*) showed that when Bayesian
learners form a transmission chain, the distribution of outputs converges
to the learners' prior. The math: iterated learning is a Markov chain
whose stationary distribution is the agents' shared prior.

Our debate cycles are structurally analogous to transmission chains.

**Q10: Do debate cycles converge to agents' priors, and does this explain
the arbiter's bias?** Each cycle, agents interpret results and propose new
experiments. Their proposals pass through the EIG filter and the arbiter's
crux machinery. Over 5 cycles, the selected experiments should converge
toward the stationary distribution of the agent+arbiter "prior." The
arbiter's similarity-structure bias IS the prior being revealed through
iterated selection. Griffiths' iterated learning math could formalize this:
what prior over experiment types does the arbiter's crux mechanism
implicitly encode?

**Q11: Is the open design space a noisier transmission channel?** In
iterated learning, adding noise to the transmission channel shifts the
stationary distribution toward the prior faster. Open design (where agents
must reinvent structures each cycle) is noisier than closed design (where
structures are fixed). Griffiths' framework predicts: open design should
converge to the agents' linguistic prior faster, producing stronger
model-type bias. This matches our data — open_debate has stronger biases
(+24pp RULEX, -24pp SUSTAIN) than closed_debate (+4pp GCM, -28pp RULEX).

**Q12: Can transmission chain experiments with human participants validate
the LLM bias pattern?** Run actual humans in a serial reproduction task
where they describe category structures to each other. Does the linguistic
prior revealed by human transmission chains match the LLM agents'
proposals? If LLMs and humans share a linguistic-accessibility bias toward
discrete rule-like structures, this validates Griffiths' "LLMs as models
of cognition" thesis in a specific, testable way.

---

## 5. LLMs as Cognitive Models / Tools for Cognitive Science

Griffiths (2023, *Trends in Cognitive Sciences*) argues AI systems are most
useful to cognitive science not as models of human cognition but as
benchmarks revealing what's distinctive about human thought.

**Q13: Are LLM agents' debate behaviors informative about human scientific
reasoning?** Our agents exhibit: 87% overclaiming, persistent side-taking
escalation (Breithaupt-like), dominant "explain" responses to falsification
(Lakatos-like auxiliary hypothesis shielding), and narrative bias toward
discrete structures. Do human scientists show the same patterns? If so, the
LLM debate is a cheap simulacrum of adversarial collaboration that could be
used to pre-test experimental designs before running expensive human
adversarial collaborations (like Cowan et al., 2020's multi-year working
memory debate).

**Q14: Can the "implicit model-type prior" finding be tested in human
experimenters?** We showed that LLM agents propose structures biased toward
linguistically describable properties. Griffiths' framework suggests humans
should show the same bias (language constrains thought about experiments,
per Whorfian effects). Run a study where human experimenters design
categorization experiments: do they also over-propose rule-diagnostic
structures and under-propose similarity-gradient structures?

**Q15: Does resource-rational meta-cognition predict when LLM debate adds
value?** Griffiths, Callaway et al. (2019) argue that the key to
understanding intelligence is understanding *meta-cognition* — how agents
decide what to think about. Our finding maps: under correct specification,
the meta-rational decision is "don't debate, just compute." Under
misspecification, the meta-rational decision is "debate to diagnose
parameter errors." Can Griffiths' meta-reasoning framework predict the
crossover point — at what level of misspecification does debate's expected
value exceed its computational cost?

---

## 6. Meta-Learning and Learning to Learn

Grant, Finn, Levine, Darrell & Griffiths (2018) showed MAML is equivalent
to hierarchical Bayesian inference — the meta-learned initialization is
the prior. McCoy et al. (2023) showed meta-learned biases recapitulate
human linguistic universals.

**Q16: Are LLM debate strategies meta-learned from scientific text?** LLMs
were trained on scientific papers, reviews, and debates. Their debate
behavior — overclaiming, auxiliary hypothesis shielding, narrative
familiarity bias — may be meta-learned strategies that were optimal
*in the training distribution* (human scientific practice) but suboptimal
for our computational setting. The gap between LLM meta-learned priors
and the task-specific optimum is precisely what resource rationality
measures. Can we quantify this gap?

**Q17: Can meta-learning improve the arbiter?** Rather than hand-designing
crux identification rules, meta-learn the arbiter's selection strategy
across many model-comparison tasks. The meta-learned arbiter should
discover that different model types need different experiment types —
i.e., it should learn the model-type-fairness that our current arbiter
lacks. Griffiths' MAML-as-hierarchical-Bayes framing suggests the
meta-learned prior over experiment selection strategies would encode
the same information as our empirically discovered complementary-bias
structure.

---

## 7. Social Learning and Collective Intelligence

Griffiths' work on how groups aggregate information, including sensitivity
to source independence (Whalen, Griffiths & Buchsbaum, 2018) and social
learning strategies.

**Q18: Is the debate structure an optimal information aggregation mechanism?**
Our system has three agents with correlated information (they all see the
same divergence map and prediction data). Griffiths' work on social
learning suggests that naive aggregation of correlated sources leads to
overconfidence. The arbiter's role as a meta-agent that synthesizes across
agents should, in principle, discount correlated evidence. Does it? Or
does the arbiter naively aggregate, explaining why its bias compounds
rather than cancels?

**Q19: Does adversarial structure outperform cooperative structure for model
selection?** Griffiths' collective intelligence work studies how group
structure affects accuracy. Our agents are adversarial (each advocates for
its model). Would cooperative agents (all trying to find the correct model
without partisan commitment) perform better or worse? The adversarial
structure forces exploration (agents must defend different positions), but
Dubova showed theory-motivation narrows evidence. A Griffiths-style social
learning model could predict the optimal degree of adversarialness.

---

## 8. Building Machines That Think With People

Collins et al. (2024) propose "thought partners" — AI systems that
collaborate with humans by sharing representations and jointly solving
problems.

**Q20: Is our framework a "thought partner" for experimental design?**
The system produces not just a model identification result but a legible
scientific narrative: debate transcripts, prediction records, falsification
ledgers, crux negotiations. A human scientist reading the M17 GCM
open_arbiter transcript gets an explanation for *why* GCM won and *what
experiments were most diagnostic*. This is Collins et al.'s vision
instantiated: the machine doesn't replace the scientist, it structures
the reasoning process. Can we formalize what the "thought partnership"
adds beyond the computational result alone?

**Q21: What is the right human-AI division of labor for scientific model
comparison?** Our finding — "LLM for semantics, computation for numerics"
— is an empirically derived answer to a question Griffiths has been asking
theoretically. The specific division: computation handles experiment
selection, posterior updating, and model predictions; LLMs handle
interpretation, parameter diagnosis, and structure proposal. Griffiths'
resource rationality framework provides the theoretical justification: each
component handles the subtask where its approximation strategy is cheapest
relative to the information gained. Can this be formalized as a
resource-rational division of cognitive labor?

---

## Summary: Highest-Priority Questions for Collaboration

If I had to pick 5 questions to lead a conversation with Griffiths:

1. **Q4** — Does our "no mimicry" finding challenge or refine the
   approximation-to-same-ideal thesis (Sanborn, Griffiths & Navarro 2010)?
2. **Q7** — What should AutoRA learn from our adversarial debate findings,
   especially the M15 misspecification result?
3. **Q10** — Can iterated learning math formalize the arbiter's model-type
   bias as convergence to an implicit prior?
4. **Q1** — Is the arbiter's bias resource-rational (optimal given bounded
   divergence computation)?
5. **Q15** — Can meta-reasoning predict the crossover point where debate's
   expected value exceeds its cost?
