# Automated Antagonistic Collaboration: AI Agents as Adversarial Scientists

**Suyog Chandramouli & George Kachergis**

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

---

## 5. Results

*[To be populated as experiments are run.]*

### 5.1 Divergence mapping results

### 5.2 Debate transcript analysis

### 5.3 Expert evaluation of proposals

### 5.4 Single-agent vs. multi-agent comparison

### 5.5 Retrospective validation

---

## 6. Discussion

*[To be developed.]*

---

## References

Ashby, F. G., Alfonso-Reese, L. A., Turken, A. U., & Waldron, E. M. (1998). A neuropsychological theory of multiple systems in category learning. *Psychological Review*, 105(3), 442–481.

Bernardo, J. M., & Smith, A. F. M. (1994). *Bayesian Theory*. New York: Wiley.

Blohm, G., Peters, B., Haefner, R., Isik, L., Kriegeskorte, N., Lieberman, J. S., Ponce, C. R., Roig, G., & Peters, M. A. K. (2024). Generative adversarial collaborations: A practical guide for conference organizers and participating scientists. *arXiv:2402.12604*.

Cavagnaro, D. R., Myung, J. I., Pitt, M. A., & Kujala, J. V. (2010). Adaptive design optimization: A mutual information-based approach to model discrimination in cognitive science. *Neural Computation*, 22(4), 887–905.

Cavagnaro, D. R., Pitt, M. A., & Myung, J. I. (2011). Model discrimination through adaptive experimentation. *Psychonomic Bulletin & Review*, 18(1), 204–210.

Clark, C. J., & Tetlock, P. E. (2022). Adversarial collaboration: The next science reform. In C. L. Frisby, R. E. Redding, W. T. O'Donohue, & S. O. Lilienfeld (Eds.), *Political Bias in Psychology* (pp. 905–927). Springer.

Cowan, N., Belletier, C., Doherty, J. M., Jaroslawska, A. J., Rhodes, S., Forsberg, A., Naveh-Benjamin, M., Barrouillet, P., Camos, V., & Logie, R. H. (2020). How do scientific views change? Notes from an extended adversarial collaboration. *Perspectives on Psychological Science*, 15(4), 1011–1025.

Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023). Improving factuality and reasoning in language models through multiagent debate. *ICML 2024*.

Kahneman, D. (2003). Experiences of collaborative research. *American Psychologist*, 58(9), 723–730.

Liang, T., He, Z., Jiao, W., Wang, X., Wang, Y., Wang, R., Yang, Y., Tu, Z., & Shi, S. (2023). Encouraging divergent thinking in large language models through multi-agent debate. *arXiv:2305.19118*.

Love, B. C., Medin, D. L., & Gureckis, T. M. (2004). SUSTAIN: A network model of category learning. *Psychological Review*, 111(2), 309–332.

Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., & Ha, D. (2024). The AI Scientist: Towards fully automated open-ended scientific discovery. *arXiv:2408.06292*.

Lu, C., Hu, E. J., Lange, R. T., Foerster, J., & Clune, J. (2025). The AI Scientist-v2: Workshop-level automated scientific discovery via agentic tree search. *Sakana AI Technical Report*.

Medin, D. L., & Schaffer, M. M. (1978). Context theory of classification learning. *Psychological Review*, 85(3), 207–238.

Mellers, B., Hertwig, R., & Kahneman, D. (2001). Do frequency representations eliminate conjunction effects? An exercise in adversarial collaboration. *Psychological Science*, 12(4), 269–275.

Myung, J. I., & Pitt, M. A. (2009). Optimal experimental design for model discrimination. *Psychological Review*, 116(3), 499–518.

Nosofsky, R. M. (1986). Attention, similarity, and the identification-categorization relationship. *Journal of Experimental Psychology: General*, 115(1), 39–57.

Nosofsky, R. M., Palmeri, T. J., & McKinley, S. C. (1994). Rule-plus-exception model of classification learning. *Psychological Review*, 101(1), 53–79.

Peters, B., Blohm, G., Haefner, R., Isik, L., Kriegeskorte, N., Lieberman, J. S., Ponce, C. R., Roig, G., & Peters, M. A. K. (2025). Generative adversarial collaborations: A new model of scientific discourse. *Trends in Cognitive Sciences*, 29(1), 1–4.

Pitt, M. A., Kim, W., Navarro, D. J., & Myung, J. I. (2006). Global model analysis by parameter space partitioning. *Psychological Review*, 113(1), 57–83.

Pitt, M. A., Myung, I. J., & Zhang, S. (2002). Toward a method of selecting among computational models of cognition. *Psychological Review*, 109(3), 472–491.

Shepard, R. N., Hovland, C. I., & Jenkins, H. M. (1961). Learning and memorization of classifications. *Psychological Monographs*, 75(13), 1–42.

Wagenmakers, E.-J., Ratcliff, R., Gomez, P., & Iverson, G. J. (2004). Assessing model mimicry using the parametric bootstrap. *Journal of Mathematical Psychology*, 48(1), 28–50.
