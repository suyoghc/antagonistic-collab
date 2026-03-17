# Automated Antagonistic Collaboration: AI Agents as Adversarial Scientists

*Working draft — last updated March 2026*

---

## 1. Introduction

Science advances through the competition of ideas. In the ideal hypothetico-deductive process, theories are continually tested against each other, with the most successful surviving (Popper, 1959). In practice, however, competing theories often develop in parallel silos — separate labs, separate methods, separate languages — and rarely meet in direct confrontation (Peters, Blohm, Haefner et al., 2025). When they do meet, the process is slow: adversarial collaborations between human scientists unfold over years and require extraordinary coordination (Mellers, Hertwig & Kahneman, 2001; Clark & Tetlock, 2022).

We propose **automated antagonistic collaboration**: a framework in which AI agents, each formally committed to a competing scientific theory, engage in structured adversarial debate to design maximally informative experiments. The agents argue in natural language but are grounded by executable computational models — they cannot merely *assert* what their theory predicts, they must *demonstrate* it. A human moderator arbitrates, preserving the judgment calls that make science more than optimization.

This work sits at the intersection of three literatures that have not yet been connected:

1. **Adversarial collaboration** in psychology and cognitive science (Kahneman, 2003; Clark & Tetlock, 2022; Peters et al., 2025), which documents the epistemic benefits of structured disagreement but relies entirely on human scientists.
2. **Automated scientific discovery** (Lu et al., 2024; Musslick et al., AutoRA), which automates the research pipeline but lacks adversarial structure — a single agent or cooperative pair runs through ideation, experimentation, and analysis.
3. **Multi-agent debate in LLMs** (Du et al., 2023; Liang et al., 2023), which shows that multiple model instances critiquing each other can improve factual accuracy and reasoning, but has not been applied to scientific theory arbitration.

The gap is clear: no existing system combines adversarial discourse structure with formal model grounding and human-in-the-loop moderation. We fill that gap.

### 1.1 Why this matters beyond a technical contribution

The dominant approach to automated experiment design in psychology — Adaptive Design Optimization (ADO; Myung & Pitt, 2009; Cavagnaro et al., 2010) — is powerful but operates under restrictive assumptions. ADO optimizes designs to discriminate between a pre-specified set of models (the M-closed setting; Bernardo & Smith, 1994), typically in pairwise comparisons. Real science is M-open: the true generating process may not be any model under consideration, models can be revised mid-investigation, and the space of relevant theories is not fixed in advance.

Antagonistic collaboration is inherently M-open. Agents can propose new models during debate, challenge each other's auxiliary assumptions, and surface theory modifications that no single-agent system would consider. The adversarial structure also produces a legible scientific narrative — a reader can follow the reasoning, see where theories were challenged, and understand why a particular experiment was chosen. ADO produces an optimal design; our system produces an optimal design *with a transparent justification*.

### 1.2 Scope of the current work

We demonstrate the framework in the domain of **human categorization** — a testbed chosen for three reasons:

- The model landscape is deep and contested: exemplar models (GCM; Nosofsky, 1986), rule-based models (RULEX; Nosofsky, Palmeri & McKinley, 1994), clustering models (SUSTAIN; Love, Medin & Gureckis, 2004), and hybrids have been debated for over 40 years.
- The experimental design space is rich: stimulus dimensionality, category structure, training regime, transfer tests, and dependent measures provide many degrees of freedom for creative experiment design.
- Critical experiments are known: Shepard et al.'s (1961) six types, the Medin & Schaffer (1978) 5-4 structure, and COVIS dissociations serve as benchmarks against which we can evaluate whether the system produces scientifically sensible proposals.

---

## 2. Architecture

### 2.1 The core problem: two layers, not one

A naive implementation would have LLM agents debating experiment design in natural language, occasionally emitting structured proposals. But the entire point of formal models in science is that they *constrain* what you are allowed to argue. When an exemplar-theory agent claims "my model predicts a gradual learning curve," that claim should come from actually running the Generalized Context Model on the proposed conditions — not from the LLM generating a plausible-sounding prediction.

This motivates a **dual-layer architecture**:

- **Semantic layer.** LLM agents reason in natural language: interpreting results, identifying confounds, proposing theory modifications, arguing about auxiliary assumptions. This is where scientific *judgment* happens — the kind of reasoning that cannot be fully formalized.
- **Formal layer.** Executable computational models (GCM, SUSTAIN, RULEX) generate quantitative predictions, fit parameters to data, and compute model comparison metrics. This is where arguments get *grounded*.

The agents sit at the interface. When an agent makes a claim about what its theory predicts, it calls its model. When it critiques an opponent's experiment proposal, it can run its own model under the proposed conditions and demonstrate quantitatively that the design does not discriminate as well as claimed. The LLM's role is to reason about things the models cannot capture — plausibility of assumptions, ecological validity, what the results *mean* — while the formal layer prevents the debate from drifting into pure rhetoric.

**Design rationale.** The alternative — pure language debate without model grounding — fails for a specific reason: LLMs are prone to generating confident-sounding but quantitatively wrong predictions about model behavior. In preliminary tests, an ungrounded agent representing GCM would assert predictions that the actual GCM does not make for the given parameter settings. The formal layer is not an optional enhancement; it is a necessary constraint on the integrity of the debate.

### 2.2 Phase structure rather than round-robin

Real scientific disputes do not proceed by "your turn, my turn." They have distinct epistemic phases with different goals. We implement each debate cycle as a state machine with nine phases:

1. **Commitment registration.** Each agent articulates its theory's core claims and registers the executable model(s) that instantiate them. This produces a structured object including core claims, callable models, parameter ranges, and explicit auxiliary assumptions. Crucially, an agent can register *multiple models* for the same theory — there is generally a one-to-many relationship between a theoretical stance and the models that could instantiate it.

2. **Divergence mapping.** Not adversarial yet. All agents' models are run on a grid of plausible experimental conditions (e.g., Shepard's six category types, the 5-4 structure). The system automatically identifies where predictions diverge most. This produces a quantitative *divergence map* — a structured representation of which conditions would be most diagnostic. This phase ensures the subsequent debate is grounded in actual model behavior rather than agents' intuitions about where theories disagree.

3. **Experiment proposal.** Given the divergence map, each agent proposes a design that would be maximally diagnostic from its theoretical perspective. Proposals must include both a formal specification (category structure, conditions, DVs) and a natural-language rationale.

4. **Adversarial critique.** The key phase. Each agent attacks opponents' proposals on multiple dimensions: demonstrating (by running its model) that under alternative auxiliary assumptions, its theory *also* predicts the expected pattern; identifying confounds; showing quantitatively that the proposed design does not discriminate as well as claimed. This is where the formal layer earns its keep — critiques backed by model runs carry more weight than verbal objections.

5. **Design revision.** Agents revise proposals in light of critique. The moderator can intervene to break deadlocks or introduce practical constraints.

6. **Human arbitration.** The moderator selects, edits, or synthesizes a final design. This is where scientific judgment about feasibility, ethics, ecological validity, and novelty enters — considerations that agents cannot fully evaluate.

7. **Execution.** Before data are revealed, each agent *must* register a quantitative prediction. These predictions are logged in the epistemic state tracker and will be scored against actual results. The experiment is then run (synthetic data in the current prototype, real data via Prolific/AutoRA in future work).

8. **Interpretation and revision.** Agents interpret results through their theoretical lens. Crucially, this phase allows *theory modification*: an agent whose predictions were disconfirmed can propose a specific revision to its model. The revision must be formally registered — no stealth changes. The epistemic state tracker logs what changed and why.

9. **Audit.** Automated summary: what was established this cycle? Which predictions were accurate? What theories were revised? What disputes remain open? What should the next cycle focus on?

**Design rationale.** The round-robin alternative (used by Du et al., 2023, and most multi-agent debate frameworks) treats all conversational turns as equivalent. But proposing an experiment, critiquing a proposal, and interpreting data are fundamentally different epistemic activities with different goals and different required outputs. The phased structure ensures each activity gets appropriate attention and produces structured artifacts rather than unstructured chat messages.

### 2.3 Epistemic state tracker

This is the architectural component most conspicuously absent from prior multi-agent systems. A "shared scratchpad" that is merely a chat history is not cumulative — it is just long. The epistemic state tracker maintains a structured, evolving record of:

- **Theories**: name, agent, core claims, registered models (as callables), parameter ranges, auxiliary assumptions, revision history, status (active / modified / abandoned).
- **Experiments**: design spec, rationale, critiques received, moderator edits, data, per-agent interpretations.
- **Prediction registry**: for each experiment, each agent's quantitative prediction, the actual outcome, and a score. This is the accountability mechanism — it creates a cumulative record of which agents' models predicted the data well and which did not, making it very difficult for an agent to quietly revise its theory without acknowledging the failure.
- **Established facts**: claims that all agents accept.
- **Live disputes**: claims on which agents disagree, with each agent's position and proposed resolution.

At the start of each phase, agents receive a `summary_for_agent()` view of the epistemic state — not a raw transcript but an organized summary of what has been established, what is disputed, and how their past predictions have fared. This is what makes the debate cumulative rather than repetitive.

**Design rationale.** The prediction registry deserves special emphasis. In human adversarial collaborations, a common failure mode is *goalpost-shifting*: after seeing data, a theorist reinterprets their prediction or modifies their theory in a way that was not committed to in advance (Clark & Tetlock, 2022). The prediction registry prevents this by requiring agents to commit to specific quantitative predictions *before* data are revealed. Theory modifications are permitted but must be registered as such — the system distinguishes between a prediction that was correct and a theory that was revised post hoc to accommodate the data.

### 2.4 The formal layer: model implementations

We implement three categorization models as callable Python objects with a shared interface:

**Generalized Context Model (GCM; Nosofsky, 1986).** The exemplar model. Classification of a stimulus is based on its summed similarity to all stored training instances, weighted by learned dimensional attention. Parameters: sensitivity (c), attention weights (w_i per dimension), distance metric (r). Key predictions: sensitivity to specific training items, smooth generalization gradients, good performance on non-linearly-separable categories. The model stores all instances — no information is lost during learning.

**SUSTAIN (Love, Medin & Gureckis, 2004).** The clustering model. Learners maintain a flexible number of clusters, recruited adaptively when prediction fails. Parameters: attentional focus (r), cluster competition (beta), response determinism (d), learning rate (eta). Key predictions: sensitivity to within-category structure, order-dependent learning, cluster recruitment at surprising items. The number of clusters is not fixed — it emerges from the interaction between category structure and learning dynamics.

**RULEX (Nosofsky, Palmeri & McKinley, 1994).** The rule-based model. Learners search for simple verbalizable rules, then store exceptions to those rules. Parameters: rule-search probabilities (p_single, p_conj), exception memorization probability (p_exception). Key predictions: strong advantage for rule-defined categories (Shepard Type I), difficulty with categories lacking simple rules (Type VI), discrete learning transitions (rule discovery is all-or-nothing).

Each model implements:
- `predict(stimulus, training_items, training_labels, **params)` → response probabilities
- `predict_learning_curve(training_sequence, test_items, **params)` → accuracy over training
- `fit(training_items, labels, response_data)` → best-fitting parameters and fit metrics

This shared interface is what allows agents to invoke any model during debate — the Exemplar Agent can run SUSTAIN to check whether the Clustering Agent's claims are quantitatively correct, and vice versa.

### 2.5 How adversarial debate improves on single-agent experiment design

The adversarial structure buys three specific things that a single prompted agent cannot provide:

**Exploration of the auxiliary assumption space.** When a single agent designs an experiment, it implicitly commits to one set of auxiliary assumptions. Multiple agents with different theoretical commitments will challenge each other's auxiliaries, surfacing designs that are robust to a wider range of assumptions. This directly addresses the model mimicry problem documented by Pitt, Myung and colleagues: models in categorization are notoriously flexible, and a design that discriminates under one parameterization may fail under another.

**M-open by construction.** A single agent optimizing to discriminate between pre-specified models is inherently M-closed. Adversarial agents can propose new models mid-debate — "what if the mechanism is neither pure exemplar storage nor pure rule search, but a mixture?" — and the formal layer can immediately register and test the new model. This is how real science works: the model space is not fixed in advance.

**Legibility.** The debate transcript is a scientific narrative. A reader can follow the reasoning, see where theories were challenged, and understand how they responded. A single-agent optimization produces an experiment design with no visible reasoning chain. For publication and for scientific trust, this matters enormously.

### 2.6 Failure modes engineered out

Human adversarial collaborations have well-documented failure modes (Kahneman, 2003; Clark & Tetlock, 2022; Cowan et al., 2020). Some of these are harmful without qualification. Three in particular are structurally eliminated by the architecture rather than merely mitigated.

#### Straw-manning

In traditional scientific discourse, Scholar A claims "Scholar B would predict X," designs six studies showing not-X, and Scholar B responds "I would not have predicted X" (Clark & Tetlock, 2022). The problem is that Scholar A's representation of Scholar B's theory is a caricature — simplified, weakened, or distorted in ways that make it easier to refute. This is not always deliberate; researchers working within their own framework may genuinely misunderstand the commitments and predictions of a competing theory.

*How the system eliminates it.* Each agent's theory is registered as a formal object: core claims, auxiliary assumptions, and — critically — an executable model. When the Exemplar Agent critiques the Clustering Agent's proposal, it cannot misrepresent what SUSTAIN predicts, because SUSTAIN is a callable function. The agent can run `sustain.predict()` on the proposed experimental conditions and see the actual predictions. If the Exemplar Agent claims "the clustering model predicts no order effect," the system can verify this against the model's actual behavior. The opponent's theory is not a verbal summary that can be distorted; it is code that can be executed.

This eliminates straw-manning at two levels. First, agents cannot mischaracterize what a competing model predicts because the model is right there. Second, agents cannot caricature the *scope* of a competing theory — the theory commitment object explicitly lists core claims, auxiliary assumptions, and the conditions under which the theory applies. The formal registration makes each theory's commitments transparent and verifiable.

#### Terminology confusion

Different theoretical traditions in psychology routinely use the same terms to mean different things, or different terms to mean the same thing (Cowan et al., 2020). In the working memory literature, "central executive" means something different to Baddeley's framework than "focus of attention" does to Cowan's, yet they sometimes get treated as equivalent. In categorization, "attention" in the GCM refers to dimensional weighting parameters that redistribute sensitivity across stimulus dimensions; in SUSTAIN, it refers to receptive field tuning that sharpens or broadens cluster activation. These are related but not identical mechanisms, and debates that conflate them are debates about words, not about cognition.

*How the system eliminates it.* The shared model interface forces operational definitions. When the Exemplar Agent says "my model predicts that attention shifts toward the diagnostic dimension," this means, specifically, that the attention weight parameter $w_i$ for that dimension increases. When the Clustering Agent says "my model predicts attention shifts toward the diagnostic dimension," this means that $\lambda_i$ (the receptive field tuning parameter) for that dimension increases. These are different mathematical operations with different behavioral consequences, and the formal layer makes the difference explicit.

More broadly, any theoretical claim that can be stated in natural language must, in this system, be grounded in a model operation. "Learning is gradual" means the `predict_learning_curve()` output shows a monotonically increasing accuracy function with no discontinuities. "Exceptions are stored" means the model's response to exception items differs from its response to rule-following items by a specific amount. The formal layer does not ban imprecise language — agents still argue in natural language — but it provides a resolution mechanism. When two agents appear to disagree, the moderator can ask each to demonstrate the claim by calling their model. If the models produce the same output despite the verbal disagreement, the dispute is terminological, not substantive.

#### Lack of record-keeping

Kahneman's collaboration with Hertwig was plagued by disputes about what predictions had been committed to in advance. The published protocol from Mellers, Hertwig & Kahneman (2001) explicitly requires the arbiter to maintain detailed records, but in practice this is a burden that scales poorly — especially in extended collaborations like Cowan et al.'s multi-year, multi-lab project, where the sheer volume of proposals, revisions, and interim results exceeds what any human record-keeper can track reliably.

*How the system eliminates it.* The epistemic state tracker is a structured database, not a notebook. Every theory registration, every prediction, every critique, every data point, every theory revision is logged with timestamps, authorship, and context. The prediction registry specifically addresses Kahneman's concern: predictions are committed to formally before data are revealed, and they cannot be edited after the fact. The revision log on each theory commitment object tracks exactly what changed and when, so the distinction between a prediction that was confirmed, a prediction that was disconfirmed, and a theory that was revised post hoc is always recoverable.

This is not merely more efficient record-keeping — it is a qualitatively different kind of record. In a human collaboration, the record is retrospective: someone writes down what happened after the fact, filtered through memory and interpretation. In this system, the record is constitutive: the structured state *is* the collaboration. Nothing happens outside the tracker. There is no possibility of an off-the-record conversation where commitments are informally adjusted, because the agents have no off-the-record channel.

The cumulative nature of this record also enables analyses that would be impractical in human collaborations. The prediction leaderboard — a running tally of each agent's prediction accuracy across all experiments — emerges automatically from the logged data. In a human AC, computing this would require going back through years of correspondence to reconstruct who predicted what. In this system, it is a single function call.

---

## 3. Evaluation plan

### 3.1 George's empirical question

A key empirical question raised during the initial conception of this project: is it better to prompt a single AI agent with "design the best experiment to discriminate theories of X," or to assign different theoretical stances to k different agents and have them propose and discuss alternatives? At what k does the adversarial approach break down?

We will compare:
- **(a) Single-agent baseline.** A single LLM prompted with full knowledge of all three theories, asked to design the most informative experiment.
- **(b) k-agent adversarial debate without human moderator.** k = 2 and k = 3 agents, each committed to one theory, debating via the full phased protocol but with an automated moderator.
- **(c) k-agent adversarial debate with human moderator.** Same as (b) but with a human scientist arbitrating proposals and directing the debate.

### 3.2 Metrics

- **Proposal quality.** Blind evaluation by domain experts (categorization researchers) rating experiment proposals on: novelty, diagnostic value, feasibility, and scientific importance. We aim for 5–10 expert raters.
- **Divergence recovered.** Does the system converge on designs that target the known regions of maximal model disagreement? The divergence mapping phase provides a quantitative benchmark.
- **Retrospective validation.** Given the state of knowledge circa 1985 (pre-ALCOVE, pre-SUSTAIN), does the adversarial debate converge on something resembling the experiments that actually advanced the field in the 1990s?
- **Debate quality.** Is the adversarial discourse scientifically coherent? Do agents make valid arguments? Do critiques identify real problems? Human evaluation of transcript quality.
- **Prediction accuracy.** Cumulative prediction registry scores across debate cycles — which agents' models best predict the data, and does this converge over time?

### 3.3 Domain validation

Human categorization was chosen *because* the answers are known. The system's proposals can be evaluated against 40 years of empirical literature. The claim is not that the system discovers something new in categorization — it is that the adversarial process produces better experiment designs than non-adversarial alternatives, validated in a domain where we know the ground truth. Generalization to domains where the answer is not known (cross-situational word learning, working memory, recognition memory) is future work.

---

## 4. Related work

### 4.1 Adversarial collaboration in science

Adversarial collaboration was named by Kahneman (2003), though the practice dates to at least Latham & Locke (1988). The canonical example is Mellers, Hertwig & Kahneman (2001), in which Kahneman and Hertwig — advocates of competing accounts of the conjunction fallacy — collaborated under an arbiter (Mellers) to design and run a joint experiment. Clark & Tetlock (2022) provide the most comprehensive recent review, arguing that adversarial collaboration should become a norm in psychology. They document both benefits (fairer tests, reduced straw-manning, new evidence valued by both sides) and challenges (coordination costs, trust barriers, the difficulty of finding neutral arbiters).

The Cognitive Computational Neuroscience (CCN) conference has operationalized adversarial collaboration through Generative Adversarial Collaborations (GACs; Peters et al., 2025; Blohm et al., 2024). GACs focus on the *pre-experimental* stages — identifying disagreements, building common language, and devising plans to resolve debates — and have run 15 teams to date. Our work automates and extends this process: AI agents can conduct the pre-experimental debate rapidly and at scale, while the human moderator retains the judgment calls that GACs identify as critical.

### 4.2 Automated scientific discovery

A growing number of systems aim to automate parts of the scientific process. We organize them by how much *discourse structure* they preserve — that is, whether the system produces not just an output (an experiment, a paper) but a legible record of the reasoning that led to it.

#### Systems with no discourse structure

**The AI Scientist** (Lu et al., 2024; v2: Lu et al., 2025) automates the full research lifecycle — ideation, experimentation, writing, and even reviewing — within a single-agent pipeline. The system generates research ideas by prompting an LLM, executes code to test them, and produces a complete paper. The v2 adds tree search over experiment strategies, improving depth of exploration. However, the AI Scientist has no adversarial structure and no mechanism for competing theoretical commitments. Its "ideas" are generated by one agent brainstorming, not by multiple agents with different commitments debating. There is no formal model layer constraining what the agent can claim — predictions are LLM outputs, not model runs. And there is no cumulative state across research cycles; each paper is generated independently. The result is a system that can produce a plausible-looking paper but cannot simulate the *process* of scientific discourse through which theories are tested against each other.

**Karpathy's autoresearch** occupies the minimal end of the spectrum: a single agent, one metric, one file, running experiments overnight. It illustrates what clean closed-loop experimentation looks like at minimum viable complexity, but has no theory-driven component at all — it optimizes a metric without any representation of *why* one approach should work better than another.

#### Systems with cooperative but not adversarial structure

**AutoRA** (Musslick et al.) is the closest existing infrastructure to what we are building. It implements an autonomous empirical research cycle with a theorist component that proposes models and an experimentalist component that designs follow-up experiments. It interfaces with real data collection platforms (Prolific, Firebase) and includes vetted experimentalist modules for falsification-based design, model disagreement, uncertainty sampling, and novelty detection. The modularity is genuine — these modules map onto considerations that our agents reason about.

However, AutoRA's architecture differs from ours in three structural ways. First, its theorist and experimentalist are *cooperative* agents working toward a shared goal. There is no mechanism for assigning competing theoretical commitments or for agents to argue against each other's proposals. Second, the system assumes a pre-specified set of models — the experimentalist modules optimize to discriminate among models that are fixed at the start. This is the M-closed setting (Bernardo & Smith, 1994). In our system, agents can propose model modifications and entirely new models mid-debate, operating in the M-open setting. Third, AutoRA runs a *pipeline*, not a *conversation*. There is no running shared record of proposals and critiques that compounds over rounds. The scientific plausibility of an output depends partly on whether that cumulative discourse structure is preserved — whether a reader can follow the chain of reasoning from initial disagreement through experiment design to interpretation.

Despite these differences, AutoRA's periphery — experiment runners, data collection infrastructure, experimentalist modules — is valuable and potentially complementary. Our adversarial debate layer could feed approved experiment designs into AutoRA's execution infrastructure. The agents could also invoke AutoRA's experimentalist modules as tools during debate (e.g., calling the falsification sampler to quantitatively back up a critique). This integration is a planned future step, facilitated by the fact that Younes Strittmatter, a developer of AutoRA, is accessible as a potential collaborator.

**ERDOS / minimum-overlap** (Together Computer) demonstrates multi-agent reasoning for mathematical problem-solving. Multiple agents collaborate to prove a conjecture, with each agent exploring different proof strategies. This is closer in spirit to our approach — multiple agents with different strategies — but it lacks the adversarial component. The agents are trying to converge on a shared solution, not to stress-test competing theories. It also operates in formal mathematics rather than empirical science, so there is no experiment design, no data, and no prediction-evaluation loop.

#### What discourse structure buys you

The key differentiator of our system is not that it uses multiple agents — several of these systems do — but that it preserves the *discourse structure* of adversarial scientific reasoning. This means:

**Cumulative state.** The epistemic state tracker maintains a structured record that evolves across cycles. Each cycle inherits what was established in previous cycles: which predictions were accurate, which theories were revised, which disputes remain open. The AI Scientist generates each paper from scratch; AutoRA's pipeline has state but no discourse; our system's state *is* the discourse — the cumulative record of proposals, critiques, predictions, and interpretations.

**Transparent reasoning chains.** When our system produces an experiment proposal, it comes with a full provenance: which agent proposed it, what theoretical rationale they gave, what critiques were raised by opponents, how those critiques were addressed, what the moderator changed, and what quantitative predictions each agent committed to. The AI Scientist's experiments emerge from a single agent's reasoning, visible only in its chain-of-thought. ADO's designs emerge from a utility function, with no reasoning trace at all.

**Accountability over time.** The prediction registry creates a longitudinal record of which theories are progressing and which are degenerating — Lakatos's distinction, operationalized computationally. After several cycles, the leaderboard shows not just which model fits best at one time point, but the *trajectory*: is this theory getting better at predicting new data, or is it being patched repeatedly to accommodate results it didn't anticipate? This kind of meta-scientific question is answerable from the structured state but not from any pipeline output.

### 4.3 Multi-agent debate in LLMs

Du et al. (2023) showed that multiple LLM instances debating their responses improves factual accuracy and mathematical reasoning. Liang et al. (2023) extended this with persona-assigned agents (the MAD framework), most similar in spirit to our theoretical stance assignment. However, a critical review (ICLR 2025 blog) found that multi-agent debate does not consistently outperform single-agent strategies, especially when controlling for compute. Our system differs in that agents are not debating factual questions with known answers but engaging in scientific theory arbitration where the formal layer provides a grounding mechanism absent from generic debate. The prediction registry and epistemic state tracker also introduce accountability structures that generic debate frameworks lack.

### 4.4 Optimal experimental design

Adaptive Design Optimization (Myung & Pitt, 2009; Cavagnaro et al., 2010, 2011) is the gold standard for automated experiment design in cognitive psychology. ADO uses Bayesian decision theory to identify maximally informative designs for discriminating between models. It has been successfully applied to retention, categorization, risky choice, and other domains. Our system can be understood as an expansive, M-open generalization of ADO: rather than optimizing a utility function over a fixed model set, adversarial agents search the design space through argumentation, with the formal layer providing the quantitative grounding that ADO's utility function provides. The adversarial structure also naturally addresses the model mimicry problem (Wagenmakers et al., 2004; Pitt et al., 2006): agents critique each other by demonstrating that flexible models can accommodate the proposed data pattern under alternative parameterizations — a concern that ADO handles through complexity penalties but not through the kind of targeted, theory-specific critique that adversarial debate enables.

### 4.5 Sequential experiment design and the exploration–exploitation tradeoff

Our system selects experiments by maximizing one-step (myopic) expected information gain (EIG) — the standard approach in ADO (Cavagnaro et al., 2010). This greedy strategy is locally optimal but can fail in sequential settings: it selects the same high-EIG experiment repeatedly when one candidate dominates, leaving informative regions of the design space unexplored. In our validation, greedy EIG selected the identical structure in 5/5 cycles for two of three ground truths (Section 3.3 of the technical report). This is a known limitation of myopic BOED.

The sequential experiment design problem is formally a partially observable Markov decision process (POMDP): the hidden state is the true model, actions are experiment choices, observations are experimental outcomes, and the reward is cumulative information gain over all remaining cycles (Huan & Marzouk, 2016). Exact solutions are intractable because they require planning over an exponentially growing tree of future observations. Three lines of work address this.

**Non-myopic Bayesian optimal experimental design.** Huan & Marzouk (2016) formulate sequential BOED as a POMDP and solve it via dynamic programming for low-dimensional problems, demonstrating that non-myopic designs substantially outperform greedy selection when early experiments have downstream consequences. Foster, Ivanova, Malik & Rainforth (2021) extend this with deep adaptive design (DAD), using amortized variational inference to learn design policies that implicitly plan ahead. Rainforth, Foster, Ivanova & Smith (2024) provide a comprehensive review of modern BOED, noting that the gap between myopic and non-myopic strategies grows with the number of remaining experiments — precisely the setting where our multi-cycle framework operates.

**Thompson sampling as approximate non-myopic design.** Thompson sampling (Thompson, 1933) — selecting actions by sampling from the posterior rather than maximizing expected utility — provides a computationally cheap approximation to non-myopic planning. Originally developed for multi-armed bandits, it has been shown to achieve near-optimal regret bounds while naturally balancing exploration and exploitation (Russo & Van Roy, 2018; Chapelle & Li, 2011). The key insight is that sampling proportional to posterior model probabilities automatically explores: uncertain models get sampled sometimes, not always, leading to experiment choices that probe underexplored regions of design space.

Kandasamy, Schneider & Póczos (2019) make this connection explicit for experimental design: their Myopic Posterior Sampling (MPS) algorithm samples the "true" model from the posterior, then selects the experiment with highest information gain *assuming that sample is correct*. Because different samples lead to different optimal experiments, MPS explores diverse designs without any ad-hoc diversity bonus. They prove that MPS achieves the same asymptotic convergence as greedy EIG while exploring more broadly in finite samples — exactly the property our system lacks.

**Application to cognitive model discrimination.** Kim, Pitt, Lu, Steyvers & Myung (2017) demonstrate non-myopic experimental planning specifically for cognitive model comparison, showing that look-ahead designs recover the true model faster than greedy EIG when models are partially overlapping — the situation we encounter with GCM and RULEX, where exemplar-based and rule-based representations make similar endpoint predictions but differ in learning dynamics. Cavagnaro, Myung, Pitt & Kujala (2010) established EIG as the standard criterion for model discrimination in cognitive science; the Thompson sampling extension preserves EIG as the local objective while addressing the exploration problem that greedy optimization cannot.

For our system, Thompson sampling offers a principled upgrade path: rather than selecting `argmax(EIG)`, sample experiments proportional to EIG scores (or sample the ground-truth model from the posterior and select the best experiment for that model). This requires no architectural changes — only replacing the `argmax` in `select_from_pool()` with a softmax sample — but addresses the greedy repetition problem identified in validation. The approach is grounded in established theory (Russo & Van Roy, 2018; Kandasamy et al., 2019) rather than an ad-hoc diversity bonus.

### 4.6 Bayesian adversarial collaboration and the crux-EIG convergence

Corcoran, Hohwy & Friston (2023) argue that adversarial collaboration and Bayesian optimal experimental design should be unified: both seek experiments that maximally discriminate between competing theories, one through qualitative identification of "crux" disagreements and the other through quantitative information gain. Our M9 validation provides an empirical test of this prediction: when LLM agents identify theoretical cruxes and propose discriminating experiments, those experiments largely overlap with what Bayesian EIG already selects. Of 24 parseable crux-directed experiment proposals, most referenced structures already in the EIG frontier. Only 1 of 15 experiments was uniquely crux-directed, suggesting the unique value of semantic experiment design lies at the margins — pointing to experiments that EIG undervalues.

This convergence has a precursor. Ouyang, Tessler, Ly & Goodman (2018) analyzed the classic Medin & Schaffer (1978) 5-4 categorization study and found that the intuitively designed category structure happened to place competing models near maximal EIG divergence. Expert intuition converged with computational optimality — though the authors present this as a fortunate case, not the general rule. Myung & Pitt (2009) argue that expert intuition is generally *unreliable* for model discrimination as models become more complex. Valentin et al. (2024) extend this with neural mutual information estimation, showing computational optimal design outperforms intuition-based design.

Our finding bridges these results: LLM agents' semantic crux reasoning converges with EIG not because of domain expertise per se, but because the constrained structure registry is small enough that both approaches identify the same discriminating candidates. Whether this convergence holds in richer design spaces — where expert knowledge about model mechanisms might reveal discriminating conditions that EIG's Monte Carlo search misses — is an open empirical question.

### 4.7 Overconfident posteriors and the limits of tempering

Oelrich, Ding, Magnusson, Vehtari & Villani (2020) identify conditions under which Bayesian model probabilities become overconfident: when compared models give "very different approximations of the data-generating process." This precisely describes the SUSTAIN case: stepwise learning curves are qualitatively different from GCM's gradual curves, producing likelihood ratios so extreme that even aggressive tempering (tau=0.005) cannot prevent rapid concentration.

Our likelihood tempering is a form of the power posterior (Grünwald, 2012; Bissiri, Holmes & Walker, 2016). Several principled alternatives exist for choosing the tempering rate: SafeBayes learns the learning rate adaptively from data (Grünwald, 2012); the c-posterior calibrates tempering via distributional tolerance (Miller & Dunson, 2019); Wu & Martin (2023) compare selection methods and find generalized posterior calibration outperforms others. Stacking (Yao, Vehtari, Simpson & Gelman, 2018) abandons posterior model probabilities entirely in favor of weights calibrated to predictive accuracy. Navarro, Pitt & Myung's (2004) "landscaping" technique could pre-assess which experimental conditions produce distinguishable vs. overlapping predictions, allowing the framework to focus BOED on conditions where models are harder to distinguish.

### 4.8 LLM format compliance and the specification gap

M9's crux parsing achieved 23% format compliance despite explicit menus and format examples. This is consistent with the broader literature on LLM instruction following. Tam et al. (2024) demonstrate that format restrictions actively *degrade* reasoning performance — the model doesn't fail to parse the format, but format constraints interfere with the generation process. The IFEval benchmark (Zhou et al., 2023) shows no model exceeds 80% on verifiable format constraints. Schall & de Melo (2025) find that instruction-tuned models drop 17% accuracy under constrained decoding because they are trained to paraphrase helpfully, which conflicts with rigid format adherence.

For LLM-in-the-loop scientific systems, this has a design implication: reliable structured output requires constrained decoding (token-level enforcement) or deterministic post-processing (fuzzy matching), not prompt engineering alone. Our mixture distribution design is robust to low compliance because it works with even 1 parseable crux per run — a pattern of graceful degradation that should guide other LLM→computation pipelines.

---

## 5. Results

### 5.1 Model identification accuracy

The framework's primary objective is to correctly identify the ground-truth model from among three competitors. Across 43 validation runs spanning milestones M4–M12 — encompassing legacy and full-pool modes, three ground truths (GCM, SUSTAIN, RULEX), three LLM backbones (GPT-4o, Claude Sonnet, Claude Opus), two selection strategies (greedy, Thompson), and crux-directed experiment selection — the correct model was identified in 39 of 40 runs.

**Table 1. Model identification across milestones.**

| Milestone | Mode | Runs | Correct | Discrimination gap | Notes |
|---|---|---|---|---|---|
| M4 | full_pool | 3 | 3/3 | 34–68% | Baseline full-pool validation |
| M4 | legacy | 3 | 3/3 | 2.4–37% | RULEX gap only 2.4% |
| M5 | full_pool | 3 | 3/3 | 36–68% | Post-feedback-loop closure |
| M6 | full_pool | 3 | 3/3 | 36–68% | arbiter-v0.1 features enabled |
| M7 | full_pool | 3 | 2/3 | 8.2–97.1% | RULEX misidentified |
| M8 | full_pool (greedy) | 3 | 3/3 | — | Post-bugfix |
| M8 | full_pool (Thompson) | 3 | 3/3 | — | Post-bugfix |
| M9 | full_pool (crux-directed) | 3 | 3/3 | 75–93% | Crux pipeline operational |
| M10 | full_pool (claim-responsive) | 3 | 3/3 | 52–97% | 80% FR rate |
| M11 | full_pool (richer design) | 3 | 3/3 | 76–96% | 15/15 parametric structures |
| M12 | full_pool (continuous) | 3 | 3/3 | 77–96% | 15/15 sampled, 0% cycle overlap |
| Cross-LLM | full_pool | 9 | 9/9 | — | GPT-4o, Sonnet, Opus |

The single misidentification occurred in M7, where RULEX ground truth was identified as GCM. This reflects genuine model overlap rather than a system failure: GCM approximates rule-like behavior by concentrating attention weights on the diagnostic dimension (Nosofsky, 1991), and the structures selected by greedy EIG in this run (linear_separable_4d, nonlinear_complex_5d) fell in regions where the two models' predictions were nearly indistinguishable. The posterior oscillated across cycles — RULEX led on cycles 0 and 2, GCM on cycles 1, 3, and 4 — producing only an 8.2% RMSE gap. M8 resolved the misidentification through a bugfix to the curve scoring mechanism and the addition of Thompson sampling, which selected more diverse structures that exposed the models' differing learning dynamics.

Discrimination gaps ranged from 2.4% (legacy-mode RULEX in M4, the worst case) to 97.1% (M7 SUSTAIN, where stepwise learning curves made SUSTAIN qualitatively distinct from both competitors). Full-pool mode consistently produced larger gaps than legacy mode because Bayesian experiment selection identified the most discriminating structures rather than relying on agents' narratively-driven proposals. The legacy RULEX run, for instance, never tested Type_I — the single structure where RULEX dominates GCM — because no agent proposed it.

**Cross-LLM robustness.** Nine runs with three different LLM backbones all identified the correct model (9/9). SUSTAIN RMSE was identical across all three backbones (0.270), while GCM and RULEX showed small variation (0.143–0.159 and 0.148–0.213 respectively) attributable to LLM-proposed parameter overrides — the one surviving code path where LLM output enters the scoring pipeline. The framework's convergence behavior is LLM-agnostic: the choice of backbone affects interpretation quality and parameter proposals but not which model wins.

### 5.2 Bayesian experiment selection: EIG, posterior dynamics, and the exploration problem

The transition from agent-driven to Bayesian experiment selection is the framework's most consequential architectural choice. This section traces how experiment selection evolved across milestones as successive limitations were identified and addressed.

**EIG-selected experiments.** In the initial full-pool validation (M4), EIG universally selected `five_four/fast_presentation` on cycle 0 — the Medin & Schaffer (1978) five-four structure has the most items (9) and the most complex category boundary, producing maximal model disagreement. When RULEX was the ground truth, EIG shifted to `Type_I/low_attention` after cycle 0, a simple single-dimension rule structure where RULEX excels (RMSE 0.06 vs GCM's 0.35). The Bayesian system correctly identified that the initial five_four experiment produced misleading evidence — the posterior initially favored GCM — and selected a maximally corrective follow-up. For GCM and SUSTAIN ground truths, the same five_four experiment was sufficiently discriminating that EIG selected it repeatedly across all five cycles.

**Posterior collapse.** Prior to likelihood tempering (M4–M6), the posterior collapsed to P≈1.0 for the correct model after a single experiment in two of three ground-truth conditions (GCM, SUSTAIN). Only the RULEX case, where the initial experiment was misleading, maintained uncertainty long enough for adaptive selection to operate. With n_subjects=20 and ~10 items per experiment, each experiment generates ~10 nats of log-likelihood evidence; after two experiments, log-odds reach ~50 nats (ratio ~5×10²¹), making posterior recovery from an incorrect initial classification effectively impossible.

M7 introduced likelihood tempering (τ=0.005, prediction clip [0.05, 0.95]), achieving the design goal of gradual convergence. For GCM ground truth, entropy dropped monotonically from 0.64 to 0.00 over five cycles, with EIG remaining non-zero through cycle 4 (0.029) — the first validation where later cycles were genuinely informative. The RULEX misidentification in M7 revealed, however, that tempering alone is insufficient when the selected structures happen to fall in regions of model overlap.

**The greedy repetition problem.** Tempering preserved uncertainty but exposed a second limitation: greedy EIG selection (`argmax`) repeated the same experiment every cycle when a single candidate dominated. The GCM and SUSTAIN M7 runs both selected linear_separable_4d on all five cycles. Only the RULEX run, where the posterior oscillated between models, produced structural variation.

**Thompson sampling resolves exploration–exploitation.** M8 replaced `argmax` with sampling proportional to EIG scores.

**Table 2. Thompson vs greedy ablation (M8, post-bugfix).**

| Ground truth | Strategy | Correct? | Winner RMSE | Unique structures | Novel structures | Final entropy |
|---|---|---|---|---|---|---|
| GCM | Thompson | Yes | 0.085 | 5 | 3 | 0.12 |
| GCM | Greedy | Yes | 0.077 | 2 | 0 | 0.01 |
| RULEX | Thompson | Yes | 0.189 | 4 | 2 | 0.16 |
| RULEX | Greedy | Yes | 0.050 | 2 | 0 | 0.06 |
| SUSTAIN | Thompson | Yes | 0.022 | 3 | 1 | 0.00 |
| SUSTAIN | Greedy | Yes | 0.018 | 1 | 0 | 0.00 |

Both strategies achieved 3/3 correct identification after the M8 bugfix. The tradeoff is interpretable: greedy achieves tighter convergence (lower final entropy, lower winner RMSE) by concentrating on the single most informative experiment, while Thompson explores 4× more broadly — 12 unique structures including 6 novel agent-proposed structures across three runs, compared to greedy's 3 unique structures with 0 novel. This is the first time novel structures proposed by agents during debate were actually selected and executed. Greedy is optimal for fast model selection when the candidate pool is known to be sufficient; Thompson is preferable when the goal includes understanding the prediction landscape or when undiscovered discriminating structures may exist.

**Crux-directed selection.** M9 completed the experiment selection architecture by connecting debate-identified theoretical disagreements to the selection mechanism via a mixture distribution. Of the 100+ cruxes proposed across prior milestones (M6–M8), zero produced parseable experiment specifications — agents wrote free-text descriptions ("test whether rule-based models can handle exceptions") that the parser could not match to pool entries. After M9's prompt redesign (showing the full structure/condition menu with a format example) and validation against known entries, 24 parseable specifications were produced across three runs (11 from the GCM run, 7 from RULEX, 6 from SUSTAIN). The mixture distribution (crux_weight=0.3) selected one crux-directed experiment in 15 total: in the GCM run, crux `crux_004` proposed `rule_plus_exception_1exc/high_noise` as a discriminating experiment for the rule-exception tradeoff, and this experiment was selected on cycle 3. This is qualitatively different from M8's path, where Thompson randomly sampled agent-proposed novel structures; here, a specific theoretical disagreement identified through debate directly determined which experiment ran.

### 5.3 The progressive strengthening of debate's causal role

A central question for the framework is whether adversarial debate contributes causally to scientific outcomes, or whether it is epiphenomenal — generating plausible-sounding discourse while the Bayesian machinery does all the work. The developmental trajectory across milestones reveals a gradual transition from epiphenomenal to genuinely causal, though the Bayesian layer remains dominant. Crucially, this trajectory was not planned — each milestone was a response to a specific failure or limitation discovered in the previous one (see Appendix A for the full development timeline).

**M4 baseline: debate is epiphenomenal.** The discovery was accidental. Replication runs — intended to measure sensitivity to stochastic variation in LLM outputs — instead revealed there was no variation to measure. Nine M4 replication runs (three per ground truth) produced identical RMSE values to four decimal places across replicates. The entire quantitative pipeline — EIG computation, experiment selection, synthetic data generation, model predictions, posterior update — is deterministic given the same prior. Different LLM runs produced different interpretive text, but this text did not feed back into any computation that affected RMSE or model identification. Debate was, in the strongest possible sense, causally inert with respect to outcomes.

**M5: first non-zero variance.** The M4 replication result prompted a systematic audit of debate feedback loops, which identified four broken connections: parameter revisions were computed but never persisted, agent claims were never verified against model outputs, hypotheses were generated but never read back, and critique content did not affect experiment selection. Closing these four loops created the first causal pathway from debate to outcomes. Four replication runs of the GCM condition produced RMSE standard deviation of 0.018 (previously 0.000). The mechanism: different LLM runs proposed different parameter revisions during interpretation, and those revisions persisted into subsequent cycles via `sync_params_from_theory()`. The correct winner was preserved across all runs — the variance was within-winner, not winner-changing — but the quantitative trajectory now depended on what the agents said.

**M6: enriched debate, broken pipeline.** M5 established that debate *could* affect outcomes. The natural next question was whether richer debate structures would produce stronger effects. Inspired by the ARBITER framework (Kachergis et al.), M6 implemented **arbiter-v0.1** — adding role-specialized meta-agents (Integrator, Critic), crux-based negotiation, conflict maps, and pre-registration. Crux negotiation was genuinely selective: 15% acceptance rate with real LLMs versus 100% with deterministic mock agents. Accepted cruxes mapped to theoretical fault lines that cognitive scientists actually disagree about ("Do people store individual exemplars or use abstract rules?"; "The role of presentation order in category learning"). However, the crux-to-experiment pipeline was structurally broken: zero parseable boost specifications were produced from over 100 proposed cruxes across all M6 runs, because agents wrote free-text crux descriptions that the parser could not match to pool entries. Debate quality improved substantially — rich structured output, genuine selectivity, interpretive synthesis — but debate did not causally influence which experiments were run.

**M8: novel structures enter the candidate pool.** M6 validation revealed posterior collapse as the primary bottleneck (EIG≈0 after cycle 0–1), which M7 addressed with likelihood tempering. But tempering exposed a second problem: greedy EIG repeated the same experiment every cycle. Thompson sampling (M8), chosen over an ad-hoc diversity bonus to follow established methods (Kandasamy et al., 2019), resolved this — and produced an unexpected side effect. Thompson sampling's stochastic exploration selected six novel agent-proposed structures across three runs — the first time debate-generated experimental designs were actually executed. However, this path was not semantically directed: Thompson sampled from the candidate pool proportional to EIG scores, and novel structures entered the pool through an automatic registration mechanism, not through directed selection based on theoretical reasoning. Debate contributed the raw material (proposed structures) but computation chose whether to use it.

**M9: the first semantically directed causal path.** The crux-to-experiment pipeline had been nominally present since M6 but was silently broken: over 100 cruxes were proposed across M6–M8 runs, and zero were parsed into experiment specifications, because agents wrote free-text descriptions while the parser expected structured `structure/condition` format. The system appeared to work — cruxes were proposed, negotiated, and accepted — but the downstream connection was inert. M9's redesigned mixture distribution connected debate-identified cruxes to experiment selection for the first time. Across three validation runs, 24 parseable crux specifications were produced and one crux-directed experiment was selected (`rule_plus_exception_1exc/high_noise` in the GCM run, cycle 3). This completes a causal chain from semantic reasoning to experiment selection: agents debated theoretical disagreements → proposed cruxes about what evidence would be decisive → the parser mapped these to pool entries → the mixture distribution selected a crux-matching experiment that would not have been selected by EIG-weighted Thompson alone.

**Table 3. Progressive strengthening of debate's causal role.**

| Milestone | Mechanism | Evidence | Effect on outcomes |
|---|---|---|---|
| M4 | None | Replication variance = 0.000 | Epiphenomenal |
| M5 | Parameter persistence | Replication std = 0.018 | RMSE affected by debate content |
| M6 | Crux negotiation | 15% acceptance; 0 parseable specs | Quality enriched; no experiment effect |
| M8 | Novel structure pool | 6 novel structures executed | Debate designs explored (stochastically) |
| M9 | Crux-directed mixture | 24 specs parsed; 1 crux-directed expt | First semantic debate→experiment path |
| M10 | Claim-responsive directive | 100% compliance when applicable | Agents confront falsified claims |
| M11 | Richer design space | 15/15 parametric structures selected | EIG exploits intermediate separations |

The trajectory shows debate's causal influence growing from zero to a genuine, if still modest, directed contribution. The Bayesian machinery remains the primary driver of convergence — correct identification occurs even without any debate (greedy EIG alone achieves 3/3 correct) — but debate progressively enriches what the system explores and why.

### 5.4 The system as a falsification engine

The claim ledger reveals a striking asymmetry in how the system converges: it identifies the correct model primarily by falsifying the incorrect ones, not by confirming the winner.

**Falsification ratio.** Across three M6 validation runs, 44 agent claims were falsified by the critique-as-falsification mechanism, 1 was confirmed, and 76 remained untested. Agents make bold predictions during interpretation debate; experiments consistently disprove them; the Bayesian posterior accumulates evidence against wrong theories. Even the winning agent rarely made predictions conservative enough to survive empirical test. The one confirmed claim — Rule_Agent predicting mean accuracy of 0.600 on Type_IV/low_attention, against an actual 0.500 — fell within the 0.1 tolerance threshold but was still substantially off.

**Agent overclaiming.** The critique-as-falsification mechanism exposed systematic overconfidence: across six validation runs, approximately 45 of 46 checked prediction claims were false (discrepancy > 0.1). Typical claimed accuracy ranged from 0.65 to 0.90; typical actual accuracy ranged from 0.10 to 0.48. This 45:1 false-to-verified ratio quantifies the gap between LLM mechanistic intuition ("exemplars handle this well") and computational reality. Agents reason correctly about *mechanisms* — they accurately describe how attention weights or cluster recruitment operate — but cannot estimate the *quantitative consequences* of those mechanisms for specific experimental conditions. This dissociation between qualitative understanding and quantitative calibration is consistent with the general finding that LLMs perform well on verbal reasoning tasks but poorly on tasks requiring precise numerical estimation.

**Theory revision patterns.** The revision dynamics exhibit a pattern consistent with Lakatos's (1978) distinction between progressive and degenerative research programmes:

| Ground truth | Winning agent's revisions | Losing agents' total revisions |
|---|---|---|
| GCM | 0–2 | 4–6 |
| RULEX | 0–1 | 3–5 |
| SUSTAIN | 0–1 | 5–6 |

Correct theories are stable: their predictions already match the data, requiring no parameter adjustment. Incorrect theories revise progressively — adjusting parameters, accommodating evidence, narrowing scope claims — but cannot close the gap, because the wrong model generates systematically wrong predictions regardless of parameterization. This is the operational analogue of Lakatos's criterion: theories aligned with the data exhibit a progressive research programme (new predictions confirmed without auxiliary adjustments), while misaligned theories exhibit a degenerating one (repeated adjustments that fail to improve predictive accuracy). RULEX is notably revision-resistant even when it is the wrong model — its rigid rule-based structure has fewer free parameters to adjust, making it simultaneously harder to fit and harder to revise. In the M6 RULEX run, Rule_Agent made zero revisions and won by 67.6%, while Clustering_Agent made three futile revisions trying to accommodate evidence it could not explain.

### 5.5 Debate quality and interpretive contributions

While debate's causal contribution to experiment selection grew across milestones, its primary value throughout was qualitative: producing interpretive, explanatory, and structurally informative discourse that no computation layer alone could generate.

**Debate quality audit.** We audited all 30 debate cycles across six M4 runs on four dimensions:

| Dimension | Assessment | Finding |
|---|---|---|
| Data citation | Weak | Agents cite posterior probabilities but rarely reference item-level predictions, RMSE values, or learning curve shapes |
| Critique quality | Mixed | Structurally substantive — agents cite mechanisms and name parameters — but numerically ungrounded ("model flexibility allows post-hoc fitting" without specifying which parameters diverge) |
| Behavioral adaptation | Limited | Same 2–3 talking points repeat across all 5 cycles within a run; no cumulative learning from prior experimental data |
| Novel structure rationale | Poor | Proposals not rooted in actual model divergence; often duplicate existing structures with minor condition permutations |

Adversarial critique pressure produces improvement in later cycles: proposals become more specific after 3+ rounds of cross-examination. But the debate does not exhibit cumulative scientific reasoning — earlier cycles' evidence is not integrated into later cycles' arguments. The claim ledger and conflict map were injected into interpretation prompts, but agents did not spontaneously engage with them.

**Novel structures.** Across 15 full-pool cycles in three M4 runs, agents proposed 21 novel category structures: 5 random/unstructured, 4 complex conjunctive, 5 multimodal/subgroup, 3 attention/order-based, and 4 other. Under greedy EIG selection, none were selected — the Bayesian selector consistently preferred registry structures (five_four, Type_I) that already spanned the relevant discrimination space. Under Thompson sampling (M8), six novel structures were selected and executed for the first time. The value of novel structure proposals lies not in superior discriminating power — EIG ranks them below registry structures — but in exploring regions of the design space that greedy selection would never reach.

**Meta-agent contributions.** Each M6 run produced 10 meta-agent responses (5 Integrator, 5 Critic). The Integrator synthesized across all three theory agents' interpretations, identifying points of convergence and divergence. The Critic consistently targeted the weakest argument — typically challenging agents whose posterior probability had collapsed but who continued asserting their model's superiority. Meta-agents did not override the Bayesian machinery; their value was qualitative, structuring the debate's narrative for human review.

**Genuine interpretive value.** Despite the quality limitations documented above, the debate transcripts contain contributions that computation alone cannot produce:

- *Mechanism identification.* Agents correctly identify the computational mechanisms responsible for model behavior ("GCM's attention weights concentrate on the diagnostic dimension, approximating rule-like behavior").
- *Theory-prediction connections.* Agents articulate why specific experimental outcomes follow from theoretical commitments ("SUSTAIN predicts order effects because cluster recruitment depends on presentation sequence").
- *Human-readable explanation.* The interpretation phase translates Bayesian posteriors, RMSE scores, and learning curve features into narrative explanations accessible to non-technical readers.

These contributions constitute genuine scientific reasoning in the sense that they connect formal model behavior to theoretical meaning — the operation that distinguishes scientific explanation from curve fitting.

### 5.6 Deviations from the evaluation plan

Section 3 outlined an evaluation plan designed before the system was built. The architecture evolved substantially during development, making some planned evaluations inapplicable and replacing them with more informative alternatives.

**Not conducted:**

*Single-agent vs multi-agent comparison* (Section 3.1). The original question — whether k adversarial agents outperform a single prompted agent — was not tested because the architecture evolved away from agent-driven experiment selection. In the final system, Bayesian EIG selects experiments regardless of whether one or three agents are debating; the agents' role shifted to interpretation, crux identification, and novel structure proposal. The operative comparison became EIG-only vs EIG + debate-informed mechanisms (crux-directed selection, novel structure registration, parameter revision persistence), not single vs multi-agent.

*Expert evaluation of proposals* (Section 3.2). Blind evaluation by independent domain experts was not conducted. The audit in Section 5.5 was systematic but not blind, and was carried out by the authors.

*Retrospective validation* (Section 3.2). We did not test whether the system, given the state of knowledge circa 1985, would converge on experiments resembling those that actually advanced the field. This remains a compelling future evaluation.

**Why these deviations occurred.** The shift from legacy mode (where agents propose experiments) to full-pool mode (where EIG selects experiments) rendered the single-agent vs multi-agent comparison less informative than originally envisioned. Expert evaluation was deferred due to prioritization of architecture development. Retrospective validation requires careful reconstruction of the 1985 model landscape, which proved beyond the current project's scope.

**What was done instead.** The actual ablations — greedy vs Thompson (Section 5.2), with and without crux-directed selection (Section 5.3), cross-LLM backbone comparison (Section 5.1) — are more informative for the evolved architecture than the originally planned comparisons. They directly test the mechanisms through which the computational and language layers interact, which is the core scientific question the system raises.

---

## 6. Discussion

### 6.1 The architecture thesis revisited: computation for numerics, language for meaning

The introduction framed this project around a specific question: can AI agents, each committed to a competing scientific theory, collaborate adversarially to identify the correct model? The results support a more nuanced answer than a simple affirmative. The correct model is identified reliably — 33 of 34 runs across six milestones, three ground truths, three LLM backbones, and multiple selection strategies — but the mechanism of convergence is overwhelmingly computational. The language layer's contribution is real but structurally different from what was initially envisioned.

**Experiment selection belongs to computation.** This conclusion was forced by the development process, not assumed in advance. The original architecture had agents propose experiments through debate (legacy mode). Two failures drove the pivot: first, early runs produced identical data regardless of agent proposals because the LLM-designed experiments could not be parsed into executable specifications — every experiment silently fell back to the same default structure. Second, even after implementing a constrained structure registry, agents' proposed experiments were narratively compelling but statistically suboptimal: they never selected Type_I (where RULEX dominates) because it is too simple to make an interesting argument about. Bayesian EIG dominates agent-driven experiment selection on every metric relevant to model identification. Legacy mode (where agents propose experiments through debate) achieved correct identification in all six M4 runs, but with gaps as low as 2.4% — perilously close to indistinguishable. Full-pool mode (where EIG selects from all 55 candidates) achieved gaps of 34–93%, driven by its ability to identify structures like Type_I/low_attention that agents never proposed because they are not narratively compelling. The exploration problem with greedy EIG (selecting the same experiment repeatedly) was solved by Thompson sampling — a computational fix — not by agent proposals.

**M13 confirms: debate is epiphenomenal on synthetic benchmarks.** The 3×2 ablation (No-Debate / Debate-No-Arbiter / Debate+Arbiter × Thompson / Greedy, 18/18 conditions correct) settles this question definitively for the current domain. Removing debate entirely produces the best RMSE (0.055) and gap (87.6%) while running 3-4× faster. Debate without arbiter-v0.1 features actively hurts (RMSE 0.078) because LLM param_overrides introduce noise. arbiter-v0.1 features partially recover (0.059) via crux-directed selection but still don't beat the computational-only baseline. The debate→computation feedback loop is architecturally open: debate output doesn't feed back into EIG, predictions, or the posterior. What debate adds is qualitative — mechanism identification, theory-prediction connections, human-readable explanations, structured records of reasoning — and, as of M9, occasional directed exploration based on semantic understanding of theoretical disagreements. Whether debate helps on harder problems (model misspecification, real data, explanation goals) remains untested.

This pattern suggests a general design principle for hybrid LLM-computation scientific systems: use computation for anything that can be formulated as an optimization problem (experiment selection, posterior update, model fitting), and use language models for anything that requires semantic understanding (interpretation, explanation, hypothesis generation, identifying what questions matter). The boundary is not fixed — crux-directed experiment selection sits precisely at the interface — but the principle provides a useful default. The crux-EIG convergence documented in Section 4.6 adds a subtlety: in constrained design spaces, the semantic and computational approaches identify the same discriminating experiments because the candidate pool is small enough for both to reach the same frontier. The unique contribution of semantic reasoning may emerge primarily in richer design spaces where domain knowledge about model mechanisms could reveal discriminating conditions that Monte Carlo search would miss.

### 6.2 What adversarial collaboration gains from automation

Section 2.6 argued that three failure modes of human adversarial collaboration — straw-manning, terminology confusion, and lack of record-keeping — are structurally eliminated by the architecture. The results bear this out: across 34 runs, no agent misrepresented a competing model's predictions (the model is callable code, not a verbal description), no terminological dispute arose (all theoretical terms are operationally grounded in model parameters), and the epistemic state tracker maintained a complete record of every prediction, critique, and revision.

Beyond eliminating failure modes, automation adds capabilities that human adversarial collaborations cannot match. Five debate-experiment cycles complete in 7–8 minutes; human adversarial collaborations unfold over months (Mellers et al., 2001) to years (Cowan et al., 2020). The Bayesian module evaluates all 55 candidate experiments per cycle — a search space no human researcher could explore systematically. The computational pipeline is deterministic given the same prior, enabling ablation studies (greedy vs Thompson, with vs without cruxes) that would require decades to conduct with human collaborators. And 34 validation runs were completed in the time it would take to organize a single human collaboration meeting.

What automation loses is the theoretical creativity of human adversarial collaborations. The debate transcripts are coherent but not creative: agents accurately describe model mechanisms, identify genuine fault lines, and produce structurally sound arguments, but they do not generate the kind of novel theoretical insight that characterizes the best human scientific discourse. The 21 novel structure proposals are variations on known themes (random assignment, complex conjunctions, multimodal subgroups) rather than genuinely novel experimental paradigms. And the debate quality ceiling documented in Section 5.5 — repetitive talking points, no cumulative learning, systematic overclaiming — suggests that current LLMs cannot sustain the progressive, evidence-responsive reasoning that makes human adversarial collaborations scientifically valuable over extended timescales.

The arbiter-v0.1 crux negotiation mechanism partially bridges this gap. With 15% of proposed cruxes accepted by real LLM agents (compared to 100% with deterministic mock agents), the negotiation exhibits genuine selectivity. Accepted cruxes map to theoretical fault lines that cognitive scientists actually disagree about. This is not rubber-stamping — the agents exercise judgment about which disagreements are worth pursuing — though substantial engineering (M6–M9) was required before that judgment could influence experimental outcomes.

### 6.3 Limitations

Several limitations constrain the generality of these results.

*Synthetic data only.* All validation used synthetic data generated from the ground-truth model. This tests whether the framework can identify the correct model when one of the three candidates is in fact correct (the M-closed setting of Bernardo & Smith, 1994). It does not test whether the models are correct accounts of human behavior, whether the framework can detect model misspecification, or whether it performs well in the M-open setting where no candidate is fully correct. Extending to real experimental data is the most important next step.

*Three models with shared interfaces.* GCM, SUSTAIN, and RULEX share a common computational interface. Models with fundamentally different output types — neural network activations, qualitative predictions, process-level observables — would require interface adaptation. The framework's generalization beyond models that produce item-level classification probabilities is architecturally straightforward but empirically untested.

*Deferred evaluations.* The single-agent vs multi-agent comparison, expert evaluation, and retrospective validation outlined in Section 3 were not conducted (Section 5.6). The absence of a single-agent baseline is particularly notable: it leaves open whether the adversarial structure provides value above a single well-prompted agent with access to the same Bayesian machinery.

*Residual posterior concentration.* Even with likelihood tempering (τ=0.005), the posterior concentrates within 3–5 cycles. Section 4.7 discusses this limitation and its relationship to the literature on overconfident Bayesian model probabilities (Oelrich et al., 2020). The combined tempering + Thompson solution keeps later cycles informative, but the system still converges faster than may be ideal for extended runs. A related subtlety: the initial tempering parameter (τ=0.2) was calibrated against unit tests with convenient small values but failed against real model predictions, where SUSTAIN's near-binary outputs (0.0005/0.999) generated ~1000 nats of evidence per experiment. The correct value (τ=0.005) required empirical calibration against actual pipeline quantities — a general lesson for hybrid systems with tunable hyperparameters.

*Silent failures in multi-component pipelines.* The crux-to-experiment pipeline was nominally present from M6 but silently broken for three milestones: over 100 cruxes were proposed, negotiated, and accepted, but zero were parsed into experiment specifications. The system appeared to function because the surrounding components (EIG selection, posterior update) operated independently. Similarly, M6's perfect 3/3 identification may have been partly an artifact of posterior collapse locking the answer on cycle 0, rather than evidence that the arbiter-v0.1 features contributed to convergence. Both cases illustrate that plausible-looking outputs can mask broken mechanisms in multi-component systems — a risk that grows with architectural complexity.

*Debate quality ceiling.* Agents do not learn cumulatively from evidence across cycles. Overclaiming is systematic (45:1 false-to-verified prediction ratio), and agents do not spontaneously engage with their own falsification record. M10's claim-responsive directive closes the ignoring gap: when explicitly directed to address falsified claims, agents comply at 100% (12/12 eligible interpretations; the remaining 3 are cycle-0 where no claims yet exist). However, the response is predominantly *explanation* rather than revision — agents attribute falsification to confounds and boundary conditions, reproducing Lakatos's auxiliary hypothesis shielding without being programmed to do so. Only 1 "abandon" action was observed across 45 theory interpretations. Overclaiming persists (agents still predict 0.65–0.85 when actual model output is 0.10–0.50), indicating that claim-responsiveness addresses ignoring but not calibration. The underlying challenge remains: LLMs produce competent single-turn analysis but cannot calibrate their quantitative expectations to match their computational models, even when confronting prior failures.

*LLM as replaceable component.* Cross-LLM comparison (9/9 correct across three backbones with substantially different capabilities) validates robustness but also implies the language layer is underutilized. If convergence is identical regardless of LLM quality, the architecture is not leveraging what better models could contribute.

*Single domain.* All results are from category learning with three specific models. Whether the framework generalizes to other multi-model disputes in cognitive science, or to domains beyond cognitive science, is unknown.

### 6.4 Future directions

*Real data integration.* The most important extension is closing the loop with human participants. The framework's computational backend requires only that models produce probability predictions for experimental stimuli — the same interface AutoRA (Musslick et al.) already supports. Integration with Prolific or Firebase for data collection would transform the system from a model identification tool into a genuine automated research assistant. This is the single extension that would most increase the project's scientific value, and it would move the framework from the M-closed to the M-open setting, where the additional information in debate (novel hypotheses, model modification proposals) could become genuinely valuable.

*Richer design spaces (M11 — implemented and validated).* M11 extends the fixed 11-structure registry to 24 structures and the 5-condition set to 7, expanding the candidate pool from 55 to 168. The extension uses parametric variants of existing generators — `linear_separable` with varied separation and dimensionality, `rule_plus_exception` with varied exception count and dimensionality — plus interpolated conditions (moderate attention, mild noise). Live validation: 3/3 correct (GCM 75.8%, SUSTAIN 95.6%, RULEX 83.7% gaps). EIG strongly prefers the parametric structures: 15/15 experiments across all three ground truths selected parametric linear_separable variants, confirming that intermediate separations provide diagnostic information the fixed registry lacked. This is a first step toward the continuous design space that optimal experimental design theory recommends (Myung & Pitt, 2009; Cavagnaro et al., 2010). A fully generative design space — where stimuli are parameterized rather than enumerated — remains a future direction that would test the limits of exhaustive EIG evaluation and potentially create conditions where semantic reasoning identifies discriminating conditions that computational search misses.

*Claim-responsive debate.* Prior to M10, agents did not engage with their falsification record despite it appearing in every interpretation prompt since M5. M10 addresses this by adding a structured directive: when an agent has falsified claims, a `FALSIFIED CLAIMS` block lists each one with evidence and requires the agent to revise, explain, or abandon it. The mechanism is inspired by Reflexion (Shinn et al., 2023), which demonstrated that LLM agents improve significantly when given structured linguistic feedback about prior failures — claim-responsive debate applies this principle to scientific argumentation, where the "failure" is a falsified empirical prediction. The revise/explain/abandon trichotomy maps onto belief revision operators from AGM theory (Alchourrón, Gärdenfors & Makinson, 1985): contraction (abandon), revision (revise), or shielding via auxiliary hypothesis (explain).

Live validation (3 ground truths × 5 cycles, GPT-4o) confirms the mechanism works: 100% compliance when agents have falsified claims (12/12 eligible interpretations include structured `falsified_response` fields). The dominant response is "explain" — agents invoke confounds, boundary conditions, and parameter sensitivity rather than revising core theoretical commitments. This reproduces Lakatos's (1978) description of auxiliary hypothesis shielding in degenerating research programmes, emerging from LLM behavior without being programmed. "Revise" actions typically adjust parameter predictions rather than core claims; only one "abandon" was observed across all runs. The mechanism closes the ignoring gap (agents can no longer pretend falsification didn't happen) but not the calibration gap (overclaiming persists at 3–5×). Whether requiring agents to design follow-up experiments testing their proposed confounds would push debate toward genuinely cumulative reasoning is a natural next step.

*Non-myopic experiment selection.* The current system selects experiments myopically (one-step EIG). Full Myopic Posterior Sampling (Kandasamy et al., 2019) or deep adaptive design (Foster et al., 2021) would plan over the remaining experiment budget, potentially selecting early experiments that are individually less informative but set up more discriminating future experiments. The gap between myopic and non-myopic strategies grows with the number of remaining experiments (Rainforth et al., 2024), making this increasingly important for longer runs.

*Cross-domain generalization.* The framework's core mechanisms — adversarial debate, Bayesian experiment selection, crux negotiation — are domain-general. Applying the system to working memory models (where Cowan et al., 2020, conducted a multi-year human adversarial collaboration) or decision-making theories would test whether the architecture's strengths and limitations are specific to category learning or reflect general properties of LLM-mediated scientific reasoning.

*The single-agent question.* The comparison between single-agent and multi-agent configurations (Section 3.1) was deferred but remains important. The proper control is a single agent with access to all three models, the same Bayesian machinery, and the same computational budget — measuring whether adversarial structure adds value beyond what a well-prompted monolithic agent can achieve. Given the LLM-agnostic convergence result (Section 5.1), the answer likely depends more on interpretive quality and novel hypothesis generation than on identification accuracy.

---

## Appendix A. Development timeline

The framework was developed iteratively over 25 sessions (March 11–15, 2026), with each milestone responding to limitations discovered in the previous one. This appendix documents the key decisions and their motivations, curated from the project's decision log (37 entries) and session transcripts. We include it because the development process itself illustrates how hybrid LLM-computation systems are debugged — many of the scientific findings reported in Section 5 emerged from diagnosing engineering failures.

### Pre-M4: Making the system run (Sessions 1–13, Decisions D1–D25)

**The constant-data problem.** The initial architecture let LLM agents design experiments in free text ("non-linearly-separable categories with family resemblance structure, 3 conditions, 120 subjects"). The computational backend could not parse these descriptions into executable stimuli, so every experiment silently fell back to the same default structure. Every run produced mean_accuracy = 0.550 regardless of what agents proposed. *Resolution:* A constrained structure registry (11 structures × 5 conditions = 55 candidates) that preserved agent choice while ensuring executability (D6–D7). This was the first instance of a pattern that recurred throughout development: LLM outputs require a validated translation layer before entering computation.

**The self-prediction bias.** After fixing the data pipeline, multi-model validation showed the wrong model winning: SUSTAIN (Clustering_Agent) beat GCM (Exemplar_Agent) even when GCM was the ground truth. Root cause: GCM predicted each item by comparing it to all training items *including itself*, producing distance=0 and near-binary predictions that could not match noisy synthetic data. Leave-one-out cross-validation — standard in the GCM literature (Nosofsky, 1986) — resolved this (D11).

**LLMs guess rather than compute.** Even with LOO scoring, the wrong agent sometimes won because predictions were LLM-generated, not model-computed. An agent asked to predict its model's accuracy would reason qualitatively ("exemplars handle this well") and produce a number (0.75) that bore no relationship to the model's actual output (0.18). *Resolution:* Separate the prediction pipeline — the LLM provides reasoning and confidence, the system calls `model.predict()` for the actual numbers (D8). This decision crystallized the architecture thesis: LLMs for semantics, computation for numerics.

**Information design matters for LLM agents.** Providing agents with divergence scores (e.g., "linear_separable_2d — 0.619") did not change their behavior; they continued proposing narratively familiar structures. Showing per-model predictions ("RULEX: 0.65, GCM: 0.55 on linear_separable_2d") immediately changed proposals. LLMs respond to concrete, actionable information, not abstract metrics (D13).

### M4: Baseline validation (Session 13–14, D25)

Six validation runs (3 ground truths × 2 modes) established that the system correctly identifies the ground-truth model in all conditions. Learning curves solved the GCM-RULEX discrimination problem (gap: 2.4% legacy → 68% full-pool). Replication runs revealed zero variance — the discovery that debate was epiphenomenal.

### M5: Closing feedback loops (Session 16, D27)

Systematic audit of the M4 pipeline identified four broken connections between debate and computation. Parameter revision persistence was the highest-impact fix: theory revisions proposed during interpretation now persisted into subsequent prediction cycles. Replication variance became non-zero (std=0.018) for the first time. *Alternatives rejected:* Persisting `param_overrides` (ephemeral by design) rather than theory-level revisions; penalty-based rather than boost-based EIG weighting.

### M6: arbiter-v0.1 integration (Session 17, D28–D29)

Motivated by the ARBITER/CRUCIBLE architecture (Kachergis et al.), M6 implemented arbiter-v0.1 — adding role-specialized meta-agents, crux negotiation, conflict maps, and pre-registration. Live validation achieved 3/3 correct with decisive gaps. Key discoveries: the system operates as a falsification engine (44:1 ratio); crux negotiation is genuinely selective (15% acceptance); but posterior collapse makes later cycles uninformative (EIG≈0 after cycle 0–1). *Alternatives rejected for meta-agents:* Giving meta-agents their own computational models (role is synthesis, not prediction). *For crux acceptance:* Threshold of 1 supporter (chose 2 to prevent rubber-stamping).

### M7: Likelihood tempering (Sessions 18–21, D30–D33)

Posterior collapse — the M6 bottleneck — was addressed with likelihood tempering (power posteriors). Initial calibration failed: τ=0.2 worked in unit tests but not with real model predictions, where SUSTAIN's near-binary outputs generated ~1000 nats of evidence per experiment. Empirical calibration against actual pipeline quantities gave τ=0.005, achieving gradual convergence (entropy 0.64→0.00 over 5 cycles). Full validation: 2/3 correct. The RULEX "failure" revealed genuine GCM-RULEX overlap, producing oscillating posteriors — arguably a more honest result than M6's 3/3, which may have been an artifact of posterior collapse locking the cycle-0 answer. *Alternatives rejected:* Entropy-based re-exploration (addresses symptom, not cause); particle posteriors (over-engineering); crux-driven override (useful but doesn't fix the posterior).

### M8: Thompson sampling (Sessions 22–23, D34–D36)

Greedy EIG repeated the same experiment 5/5 cycles once the posterior began to concentrate. Thompson sampling (sampling proportional to EIG scores) was chosen over an ad-hoc diversity penalty per the project's rule of preferring established methods (Kandasamy et al., 2019). Clean ablation: both strategies 3/3 correct; Thompson explored 4× more broadly (12 unique structures, 6 novel, vs greedy's 3 unique, 0 novel). A concurrent bugfix (removing a data-independent curve bonus that distorted posteriors) was the actual fix for M7's RULEX misidentification — both strategies benefited.

### M9: Crux-directed selection (Sessions 24–25, D37)

The crux-to-experiment pipeline had been silently broken since M6: agents wrote free-text crux descriptions that the parser could not match to pool entries. Zero of 100+ cruxes produced experiment specifications across three milestones. *Two fixes:* (1) Show agents the full structure/condition menu with format examples; (2) replace the multiplicative EIG boost (ineffective when scores cluster) with a mixture distribution guaranteeing crux-directed selection when matches exist. Result: 24 parseable specifications, 1 crux-directed experiment — the first semantically directed path from debate to experiment selection. *Alternatives rejected:* Fixing only the multiplicative boost (doesn't solve the distribution problem); higher boost factors (brute force, not principled).

### M10: Claim-responsive debate (Session 27, D38)

The claim ledger had been injected into interpretation prompts since M5, but agents ignored it for five milestones — zero spontaneous engagement with falsified claims despite the information being visible. Inspired by Reflexion (Shinn et al., 2023), M10 adds an explicit directive: agents with falsified claims must address each one via revise, explain, or abandon. Live validation: 3/3 correct (GCM 79.3%, SUSTAIN 96.6%, RULEX 51.8% gaps), 100% compliance when applicable (12/12 eligible interpretations). The dominant response is "explain" — agents invoke confounds and boundary conditions, reproducing Lakatos's auxiliary hypothesis shielding without being programmed to do so. Only 1 "abandon" across all runs. Overclaiming persists (0.65–0.85 claimed vs 0.10–0.50 actual). *Alternatives rejected:* Fine-tuning (model-specific, not portable); posterior penalty for falsified agents (conflates agent behavior with model quality); automatic revision (removes agent judgment). *Key insight:* Passive context injection is qualitatively different from explicit directives — the same information in the same prompt produces zero engagement passively and 100% engagement when accompanied by "you MUST address each claim."

### M11: Richer design spaces (Session 28, D39)

The 55-candidate pool (11 structures × 5 conditions) constrained EIG to a discrete search space. Optimal experimental design theory suggests continuous design spaces find diagnostic sweet spots that fixed registries miss (Myung & Pitt, 2009; Cavagnaro et al., 2010). M11 adds 13 parametric structures (7 linear_separable variants spanning separation × dimensionality, 6 rule_plus_exception variants spanning dimensionality × exception count) and 2 interpolated conditions (moderate_attention, mild_noise), expanding the pool to 168 candidates. All parametric entries use the same generators as the base registry with different parameters and deterministic seeds. `_synthetic_runner()` and `compute_model_predictions()` resolve them automatically. Live validation: 3/3 correct (GCM 75.8%, SUSTAIN 95.6%, RULEX 83.7% gaps). EIG strongly prefers parametric structures — 15/15 experiments selected parametric linear_separable variants, confirming that intermediate separations provide diagnostic information the fixed registry lacked. Parametric conditions selected 5/15 times. *Alternatives rejected:* Fully continuous parameterization (more principled but breaks reproducibility and complicates debugging); larger expansion with 50+ entries (diminishing returns; EIG scales linearly with pool size). *Key design choice:* Only the two generator families with continuous parameters (`linear_separable` and `rule_plus_exception`) are parameterized. Shepard types are fixed 3-binary-dimension structures and cannot be meaningfully varied.

---

## References

Ashby, F. G., Alfonso-Reese, L. A., Turken, A. U., & Waldron, E. M. (1998). A neuropsychological theory of multiple systems in category learning. *Psychological Review*, 105(3), 442–481.

Bernardo, J. M., & Smith, A. F. M. (1994). *Bayesian Theory*. New York: Wiley.

Blohm, G., Peters, B., Haefner, R., Isik, L., Kriegeskorte, N., Lieberman, J. S., Ponce, C. R., Roig, G., & Peters, M. A. K. (2024). Generative adversarial collaborations: A practical guide for conference organizers and participating scientists. *arXiv:2402.12604*.

Bissiri, P. G., Holmes, C. C., & Walker, S. G. (2016). A general framework for updating belief distributions. *Journal of the Royal Statistical Society: Series B*, 78(5), 1103–1130.

Cavagnaro, D. R., Myung, J. I., Pitt, M. A., & Kujala, J. V. (2010). Adaptive design optimization: A mutual information-based approach to model discrimination in cognitive science. *Neural Computation*, 22(4), 887–905.

Cavagnaro, D. R., Pitt, M. A., & Myung, J. I. (2011). Model discrimination through adaptive experimentation. *Psychonomic Bulletin & Review*, 18(1), 204–210.

Chapelle, O., & Li, L. (2011). An empirical evaluation of Thompson sampling. *Advances in Neural Information Processing Systems*, 24.

Clark, C. J., & Tetlock, P. E. (2022). Adversarial collaboration: The next science reform. In C. L. Frisby, R. E. Redding, W. T. O'Donohue, & S. O. Lilienfeld (Eds.), *Political Bias in Psychology* (pp. 905–927). Springer.

Corcoran, A. W., Hohwy, J., & Friston, K. J. (2023). Accelerating scientific progress through Bayesian adversarial collaboration. *Neuron*, 111(22), 3505–3516.

Cowan, N., Belletier, C., Doherty, J. M., Jaroslawska, A. J., Rhodes, S., Forsberg, A., Naveh-Benjamin, M., Barrouillet, P., Camos, V., & Logie, R. H. (2020). How do scientific views change? Notes from an extended adversarial collaboration. *Perspectives on Psychological Science*, 15(4), 1011–1025.

Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023). Improving factuality and reasoning in language models through multiagent debate. *ICML 2024*.

Foster, A., Ivanova, D. R., Malik, I., & Rainforth, T. (2021). Deep adaptive design: Amortizing sequential Bayesian experimental design. *Proceedings of the 38th International Conference on Machine Learning*, 3384–3395.

Grünwald, P. (2012). The safe Bayesian: Learning the learning rate via the mixability gap. *Algorithmic Learning Theory (ALT 2012)*, LNCS 7568, 169–183.

Huan, X., & Marzouk, Y. M. (2016). Sequential Bayesian optimal experimental design via variational inference. *arXiv:1604.08320*.

Kahneman, D. (2003). Experiences of collaborative research. *American Psychologist*, 58(9), 723–730.

Kandasamy, K., Schneider, J., & Póczos, B. (2019). Myopic posterior sampling for adaptive goal oriented design of experiments. *Proceedings of the 36th International Conference on Machine Learning*, 3222–3232.

Kim, W., Pitt, M. A., Lu, Z.-L., Steyvers, M., & Myung, J. I. (2017). A hierarchical adaptive approach to optimal experimental design. *Neural Computation*, 26(11), 2465–2492.

Lakatos, I. (1978). *The Methodology of Scientific Research Programmes: Philosophical Papers, Volume 1*. Cambridge University Press.

Liang, T., He, Z., Jiao, W., Wang, X., Wang, Y., Wang, R., Yang, Y., Tu, Z., & Shi, S. (2023). Encouraging divergent thinking in large language models through multi-agent debate. *arXiv:2305.19118*.

Love, B. C., Medin, D. L., & Gureckis, T. M. (2004). SUSTAIN: A network model of category learning. *Psychological Review*, 111(2), 309–332.

Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., & Ha, D. (2024). The AI Scientist: Towards fully automated open-ended scientific discovery. *arXiv:2408.06292*.

Lu, C., Hu, E. J., Lange, R. T., Foerster, J., & Clune, J. (2025). The AI Scientist-v2: Workshop-level automated scientific discovery via agentic tree search. *Sakana AI Technical Report*.

Medin, D. L., & Schaffer, M. M. (1978). Context theory of classification learning. *Psychological Review*, 85(3), 207–238.

Mellers, B., Hertwig, R., & Kahneman, D. (2001). Do frequency representations eliminate conjunction effects? An exercise in adversarial collaboration. *Psychological Science*, 12(4), 269–275.

Miller, J. W., & Dunson, D. B. (2019). Robust Bayesian inference via coarsening. *Journal of the American Statistical Association*, 114(527), 1113–1125.

Myung, J. I., & Pitt, M. A. (2009). Optimal experimental design for model discrimination. *Psychological Review*, 116(3), 499–518.

Navarro, D. J., Pitt, M. A., & Myung, I. J. (2004). Assessing the distinguishability of models and the informativeness of data. *Cognitive Psychology*, 49(1), 47–84.

Nosofsky, R. M. (1986). Attention, similarity, and the identification-categorization relationship. *Journal of Experimental Psychology: General*, 115(1), 39–57.

Nosofsky, R. M. (1991). Tests of an exemplar model for relating perceptual classification and recognition memory. *Journal of Experimental Psychology: Human Perception and Performance*, 17(1), 3–27.

Nosofsky, R. M., Palmeri, T. J., & McKinley, S. C. (1994). Rule-plus-exception model of classification learning. *Psychological Review*, 101(1), 53–79.

Oelrich, O., Ding, S., Magnusson, M., Vehtari, A., & Villani, M. (2020). When are Bayesian model probabilities overconfident? *arXiv:2003.04026*.

Ouyang, L., Tessler, M. H., Ly, D., & Goodman, N. D. (2018). webppl-oed: A practical optimal experiment design system. *Proceedings of the 40th Annual Conference of the Cognitive Science Society*.

Peters, B., Blohm, G., Haefner, R., Isik, L., Kriegeskorte, N., Lieberman, J. S., Ponce, C. R., Roig, G., & Peters, M. A. K. (2025). Generative adversarial collaborations: A new model of scientific discourse. *Trends in Cognitive Sciences*, 29(1), 1–4.

Pitt, M. A., Kim, W., Navarro, D. J., & Myung, J. I. (2006). Global model analysis by parameter space partitioning. *Psychological Review*, 113(1), 57–83.

Rainforth, T., Foster, A., Ivanova, D. R., & Smith, F. B. (2024). Modern Bayesian experimental design. *Statistical Science*, 39(1), 100–114.

Russo, D. J., & Van Roy, B. (2018). Learning to optimize via information-directed sampling. *Operations Research*, 66(1), 230–252.

Pitt, M. A., Myung, I. J., & Zhang, S. (2002). Toward a method of selecting among computational models of cognition. *Psychological Review*, 109(3), 472–491.

Schall, J. D., & de Melo, G. (2025). The hidden cost of structure: How constrained decoding affects language model performance. *Proceedings of RANLP 2025*.

Shepard, R. N., Hovland, C. I., & Jenkins, H. M. (1961). Learning and memorization of classifications. *Psychological Monographs*, 75(13), 1–42.

Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. (2023). Reflexion: Language agents with verbal reinforcement learning. *Advances in Neural Information Processing Systems*, 36.

Tam, Z. R., Wu, C., et al. (2024). Let me speak freely? A study on the impact of format restrictions on performance of large language models. *Proceedings of EMNLP 2024 Industry Track*.

Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. *Biometrika*, 25(3/4), 285–294.

Valentin, S., Kleinegesse, S., Bramley, N. R., Series, P., Gutmann, M. U., & Lucas, C. G. (2024). Designing optimal behavioral experiments using machine learning. *eLife*, 13, e86224.

Wagenmakers, E.-J., Ratcliff, R., Gomez, P., & Iverson, G. J. (2004). Assessing model mimicry using the parametric bootstrap. *Journal of Mathematical Psychology*, 48(1), 28–50.

Wu, P.-S., & Martin, R. (2023). A comparison of learning rate selection methods in generalized Bayesian inference. *Bayesian Analysis*, 18(1), 105–132.

Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018). Using stacking to average Bayesian predictive distributions. *Bayesian Analysis*, 13(3), 917–1003.

Zhou, J., et al. (2023). Instruction-following evaluation for large language models. *arXiv:2311.07911*.
