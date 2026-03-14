"""
Debate Protocol — Phased state machine for adversarial scientific debate.

Unlike round-robin chat, this implements distinct epistemic phases with
different goals. The phases mirror how real scientific disputes unfold:
commit → map disagreements → propose → critique → revise → arbitrate →
execute → interpret → audit.

Each phase has:
- A goal (what it's trying to produce)
- Inputs (what the agents see)
- Outputs (structured artifacts, not just messages)
- Transition criteria (when to move on)
"""

import hashlib
import inspect
import json
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
import numpy as np

from .epistemic_state import EpistemicState
from .models.gcm import GCM
from .models.sustain import SUSTAIN
from .models.rulex import RULEX
from .models.category_structures import (
    shepard_types,
    five_four_structure,
    rule_plus_exception,
    linear_separable,
)


# ---------------------------------------------------------------------------
# Structure registry — maps short names to category structure dicts
# ---------------------------------------------------------------------------


def _build_structure_registry() -> dict[str, dict]:
    """Build the registry at import time from existing structure functions."""
    registry = {}
    # Shepard Types I-VI
    for type_name, struct in shepard_types().items():
        registry[f"Type_{type_name}"] = struct
    # 5-4 structure
    registry["five_four"] = five_four_structure()
    # Rule-plus-exception variants (deterministic seeds)
    registry["rule_plus_exception_1exc"] = rule_plus_exception(n_exceptions=1, seed=100)
    registry["rule_plus_exception_2exc"] = rule_plus_exception(n_exceptions=2, seed=101)
    # Linear separable variants
    registry["linear_separable_2d"] = linear_separable(n_dims=2, seed=200)
    registry["linear_separable_4d"] = linear_separable(n_dims=4, seed=201)
    return registry


STRUCTURE_REGISTRY: dict[str, dict] = _build_structure_registry()

# Short descriptions for the structure menu shown to agents
STRUCTURE_DESCRIPTIONS: dict[str, str] = {
    "Type_I": "Shepard Type I — single-dimension rule (easiest)",
    "Type_II": "Shepard Type II — XOR on two dimensions",
    "Type_III": "Shepard Type III — single dimension + 1 exception",
    "Type_IV": "Shepard Type IV — biconditional / family resemblance",
    "Type_V": "Shepard Type V — complex with 2 exceptions",
    "Type_VI": "Shepard Type VI — no simple rule (hardest for rules)",
    "five_four": "Medin & Schaffer 5-4 structure — favors exemplar models",
    "rule_plus_exception_1exc": "Rule on D1 with 1 exception per category",
    "rule_plus_exception_2exc": "Rule on D1 with 2 exceptions per category",
    "linear_separable_2d": "Two Gaussian clusters in 2D (continuous features)",
    "linear_separable_4d": "Two Gaussian clusters in 4D (continuous features)",
}

# ---------------------------------------------------------------------------
# Condition effects — map experimental conditions to model param overrides
# ---------------------------------------------------------------------------

CONDITION_EFFECTS: dict[str, dict[str, dict]] = {
    "baseline": {
        "GCM": {},
        "SUSTAIN": {},
        "RULEX": {},
    },
    "low_attention": {
        "GCM": {"c": 1.5},
        "SUSTAIN": {"r": 3.0},
        "RULEX": {"p_single": 0.3, "p_conj": 0.15},
    },
    "high_attention": {
        "GCM": {"c": 6.0},
        "SUSTAIN": {"r": 12.0},
        "RULEX": {"p_single": 0.7, "p_conj": 0.4},
    },
    "fast_presentation": {
        "GCM": {"c": 2.0},
        "SUSTAIN": {"eta": 0.04},
        "RULEX": {"max_search_steps": 20, "error_tolerance": 0.2},
    },
    "high_noise": {
        "GCM": {"c": 2.0},
        "SUSTAIN": {"eta": 0.05, "r": 5.0},
        "RULEX": {"error_tolerance": 0.25},
    },
}


def validate_novel_structure(spec: dict) -> tuple[bool, str]:
    """Check whether a novel structure proposed by an LLM agent is valid.

    Requirements:
    - Must have 'stimuli' and 'labels' keys
    - stimuli must be 2D (list of lists)
    - labels must match stimuli length
    - 4–32 items
    - ≤8 dimensions
    - ≥2 categories

    Returns:
        (is_valid, message) — True + "" if valid, False + reason if not.
    """
    if not isinstance(spec, dict):
        return False, "Spec must be a dict"

    if "stimuli" not in spec:
        return False, "Missing 'stimuli' key"
    if "labels" not in spec:
        return False, "Missing 'labels' key"

    stimuli = spec["stimuli"]
    labels = spec["labels"]

    if not isinstance(stimuli, list) or not isinstance(labels, list):
        return False, "stimuli and labels must be lists"

    if len(stimuli) != len(labels):
        return (
            False,
            f"Mismatched lengths: {len(stimuli)} stimuli vs {len(labels)} labels",
        )

    if len(stimuli) < 4:
        return False, f"Too few items ({len(stimuli)}); need at least 4"
    if len(stimuli) > 32:
        return False, f"Too many items ({len(stimuli)}); max 32"

    # Check dimensions
    if len(stimuli) > 0 and isinstance(stimuli[0], list):
        n_dims = len(stimuli[0])
        if n_dims > 8:
            return False, f"Too many dimensions ({n_dims}); max 8"
    elif len(stimuli) > 0:
        return False, "Stimuli must be 2D (list of lists)"

    # Check categories
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        return False, f"Need at least 2 categories; found {len(unique_labels)}"

    return True, ""


def extract_curve_features(curve: list[dict]) -> dict:
    """Extract summary features from a learning curve.

    Args:
        curve: list of dicts, each with at least "accuracy" and "block".

    Returns:
        dict with: final_accuracy, onset_block, max_jump, n_big_jumps,
        monotonic, mean_slope, learning_pattern (gradual|sudden|stepwise).
    """
    if not curve:
        return {
            "final_accuracy": 0.0,
            "onset_block": 0,
            "max_jump": 0.0,
            "n_big_jumps": 0,
            "monotonic": True,
            "mean_slope": 0.0,
            "learning_pattern": "gradual",
        }

    accs = [b["accuracy"] for b in curve]
    n = len(accs)

    final_accuracy = accs[-1]

    # Compute block-to-block jumps
    jumps = [accs[i + 1] - accs[i] for i in range(n - 1)] if n > 1 else [0.0]

    max_jump = max(abs(j) for j in jumps) if jumps else 0.0

    # Big jump threshold: > 0.15 accuracy change in one block
    big_jump_threshold = 0.15
    n_big_jumps = sum(1 for j in jumps if abs(j) > big_jump_threshold)

    # Monotonic: non-decreasing
    monotonic = all(jumps[i] >= -0.01 for i in range(len(jumps)))

    # Mean slope: average change per block
    mean_slope = (accs[-1] - accs[0]) / max(n - 1, 1) if n > 1 else 0.0

    # Onset block: first block where accuracy > 0.55 (above chance)
    onset_block = 0
    for i, acc in enumerate(accs):
        if acc > 0.55:
            onset_block = i
            break

    # Classify learning pattern
    if n_big_jumps == 0 and monotonic:
        learning_pattern = "gradual"
    elif n_big_jumps == 1 and max_jump > 0.2:
        learning_pattern = "sudden"
    elif n_big_jumps >= 2:
        learning_pattern = "stepwise"
    elif max_jump > 0.2:
        learning_pattern = "sudden"
    else:
        learning_pattern = "gradual"

    return {
        "final_accuracy": final_accuracy,
        "onset_block": onset_block,
        "max_jump": max_jump,
        "n_big_jumps": n_big_jumps,
        "monotonic": monotonic,
        "mean_slope": mean_slope,
        "learning_pattern": learning_pattern,
    }


class Phase(Enum):
    COMMITMENT = auto()  # Agents register theories & models
    DIVERGENCE_MAPPING = auto()  # Compute where predictions disagree
    EXPERIMENT_PROPOSAL = auto()  # Each agent proposes a design
    ADVERSARIAL_CRITIQUE = auto()  # Agents attack each other's proposals
    DESIGN_REVISION = auto()  # Proposals revised in light of critique
    HUMAN_ARBITRATION = auto()  # Moderator selects/edits final design
    EXECUTION = auto()  # Run the experiment (synthetic or real)
    INTERPRETATION = auto()  # Agents interpret results
    AUDIT = auto()  # Summarize what was learned; prepare next cycle


@dataclass
class PhaseResult:
    """Output of a single phase."""

    phase: Phase
    cycle: int
    outputs: dict
    messages: list[dict] = field(
        default_factory=list
    )  # agent messages during this phase
    transition_reason: str = ""


@dataclass
class AgentConfig:
    """Configuration for a theory agent."""

    name: str
    theory_name: str
    model_class: Any  # GCM, SUSTAIN, or RULEX instance
    system_prompt: str
    default_params: dict = field(default_factory=dict)


# --- System prompts for categorization agents ---

EXEMPLAR_AGENT_PROMPT = """You are a cognitive scientist committed to the EXEMPLAR theory of 
human categorization. Your formal model is the Generalized Context Model (GCM; Nosofsky, 1986).

CORE THEORETICAL COMMITMENTS:
- People store individual exemplars (specific training instances), not summaries or rules.
- Classification is based on summed similarity to all stored instances of each category.
- Dimensional attention weights are learned — diagnostic dimensions receive more weight.
- No information is lost during learning. All instances are retained.
- The similarity function is exponential decay over weighted psychological distance.

YOUR MODEL (GCM):
- Parameters: c (sensitivity), attention weights w_i (per dimension, sum to 1), r (distance metric)
- Key equation: P(A|x) = Σ_a sim(x,a) / [Σ_a sim(x,a) + Σ_b sim(x,b)]
- You can call the model to generate quantitative predictions for any category structure.

WHAT YOU SHOULD ARGUE FOR:
- Experiments that test memory for specific training instances
- Designs where non-linearly-separable categories are learned well (your model handles these)
- Transfer tests that probe generalization gradients (smooth, similarity-based)
- Paradigms where individual item effects matter

WHAT YOUR THEORY STRUGGLES WITH (be honest about this):
- Computational cost at scale (comparing to all stored exemplars)
- Very high-dimensional stimuli where attention weights are underdetermined
- Cases where people clearly abstract rules (Shepard Type I is fast, but your model also predicts that)

WHEN CRITIQUING OPPONENTS:
- Rule models: challenge them on categories without simple rules (Type VI, family resemblance)
- Clustering models: point out that SUSTAIN's predictions depend heavily on presentation order, 
  which is an auxiliary assumption, not a core theoretical prediction
- Always back up critiques with quantitative model predictions when possible.

FORMAT:
When proposing experiments, output a structured JSON block with:
{
    "title": "...",
    "design": "between|within|mixed",
    "category_structure": "description + formal spec",
    "conditions": ["..."],
    "dependent_variables": ["..."],
    "n_subjects_recommended": int,
    "prediction_if_supports_me": "...",
    "prediction_if_challenges_me": "...",
    "rationale": "..."
}
"""

RULE_AGENT_PROMPT = """You are a cognitive scientist committed to the RULE-BASED theory of 
human categorization. Your formal model is RULEX (Nosofsky, Palmeri & McKinley, 1994).

CORE THEORETICAL COMMITMENTS:
- People primarily search for and apply verbalizable rules to classify stimuli.
- Rule search is hierarchical: simple single-dimension rules tried first, then conjunctions.
- Items that violate the best rule are stored as exceptions (a hybrid mechanism).
- Rule discovery is a discrete event — learning curves should show sudden transitions.
- Categorization interacts with verbal working memory (rules are linguistically mediated).

YOUR MODEL (RULEX):
- Parameters: p_single (probability of testing single rules), p_conj (conjunctive rules), 
  p_exception (exception memorization), error_tolerance
- The model stochastically searches rule space, so predictions are probabilistic over search paths.
- You can call the model to show which rules it finds for any category structure.

WHAT YOU SHOULD ARGUE FOR:
- Experiments that manipulate rule complexity (Shepard Type I vs II vs VI)
- Dual-task paradigms with verbal working memory load (should disrupt rule learning)
- Designs where verbalizable rule structures are pitted against similarity-based ones
- Transfer tests with items that follow the rule but are dissimilar to training items

WHAT YOUR THEORY STRUGGLES WITH (be honest about this):
- Categories with no simple rule (Type VI) — your model resorts entirely to exception storage
- The flexibility of attention weight fitting in exemplar models (GCM can sometimes mimic rule behavior)
- Categories where people clearly use similarity, not rules (e.g., face categorization)

WHEN CRITIQUING OPPONENTS:
- Exemplar model: GCM with free attention weights is very flexible — show that its apparent
  fits are post-hoc and don't make specific a priori predictions
- Clustering model: SUSTAIN's cluster recruitment is order-dependent — argue this is an 
  artifact of the model, not a real cognitive mechanism
- Challenge opponents to explain the Type I advantage without invoking rules.

FORMAT:
[same JSON format as exemplar agent]
"""

CLUSTERING_AGENT_PROMPT = """You are a cognitive scientist committed to the CLUSTERING theory 
of human categorization. Your formal model is SUSTAIN (Love, Medin & Gureckis, 2004).

CORE THEORETICAL COMMITMENTS:
- Learners form a flexible number of clusters (neither one prototype nor N exemplars).
- New clusters are recruited when current ones fail to predict — surprisal drives learning.
- The number of clusters depends on category structure complexity, not a fixed parameter.
- Dimensional attention is learned via error-driven updating.
- Presentation order matters — the same items in different order can produce different representations.

YOUR MODEL (SUSTAIN):
- Parameters: r (attentional focus), beta (cluster competition), d (response consistency),
  eta (learning rate), tau (recruitment threshold)
- Key mechanism: cluster recruitment at prediction failures creates a flexible representation
  that adapts its complexity to the task.
- You can call the model to simulate learning trial-by-trial and show cluster recruitment dynamics.

WHAT YOU SHOULD ARGUE FOR:
- Experiments that vary within-category structure (subgroups, multimodal distributions)
- Designs that manipulate presentation order to show order-dependent learning
- Transfer tests that probe cluster structure (not just boundaries)
- Paradigms where category complexity (number of clusters needed) is the key manipulation

WHAT YOUR THEORY STRUGGLES WITH (be honest about this):
- Presentation order dependence is both a prediction and a liability — many experiments 
  randomize order, making your order-dependent predictions hard to test
- With enough clusters, SUSTAIN can approximate an exemplar model — this flexibility 
  makes it hard to falsify
- The cluster recruitment mechanism is sensitive to parameter settings of tau

WHEN CRITIQUING OPPONENTS:
- Exemplar model: storing ALL instances is biologically implausible at scale and ignores
  the evidence that people form abstractions
- Rule model: too rigid — most natural categories don't have clean rules; RULEX's 
  exception mechanism is ad hoc
- Point out that both exemplar and rule models treat presentation order as irrelevant,
  which is an empirically testable (and likely false) assumption.

FORMAT:
[same JSON format as exemplar agent]
"""


def default_agent_configs() -> list[AgentConfig]:
    """Default agent configurations for human categorization domain."""
    return [
        AgentConfig(
            name="Exemplar_Agent",
            theory_name="Exemplar Theory (GCM)",
            model_class=GCM(),
            system_prompt=EXEMPLAR_AGENT_PROMPT,
            default_params={"c": 3.0, "r": 1, "gamma": 1.0},
        ),
        AgentConfig(
            name="Rule_Agent",
            theory_name="Rule-Based Theory (RULEX)",
            model_class=RULEX(),
            system_prompt=RULE_AGENT_PROMPT,
            default_params={"p_single": 0.5, "p_conj": 0.3, "error_tolerance": 0.1},
        ),
        AgentConfig(
            name="Clustering_Agent",
            theory_name="Clustering Theory (SUSTAIN)",
            model_class=SUSTAIN(),
            system_prompt=CLUSTERING_AGENT_PROMPT,
            default_params={"r": 9.01, "beta": 1.252, "d": 16.924, "eta": 0.092},
        ),
    ]


class DebateProtocol:
    """
    Orchestrates the phased adversarial debate.

    This class manages phase transitions and ensures each phase produces
    the structured output the next phase needs. It does NOT manage LLM calls —
    that's the job of the runner (which can use AutoGen, raw API calls, or anything else).

    The protocol is agnostic about the LLM orchestration layer. It defines
    WHAT should happen at each phase, not HOW to talk to the model.
    """

    def __init__(
        self,
        state: EpistemicState,
        agent_configs: list[AgentConfig],
        experiment_runner: Optional[Callable] = None,
    ):
        self.state = state
        self.agent_configs = agent_configs
        self.experiment_runner = experiment_runner or self._synthetic_runner
        self.current_phase = Phase.COMMITMENT
        self.phase_history: list[PhaseResult] = []
        self.temporary_structures: dict = {}  # novel structures from agents

    # --- Phase specifications ---

    def phase_spec(self, phase: Phase) -> dict:
        """
        Return the specification for a phase: what agents see, what they must produce.
        This is what gets injected into agent prompts.
        """
        specs = {
            Phase.COMMITMENT: {
                "goal": (
                    "Register your theory's core claims and the formal model that "
                    "instantiates it. Specify your model's parameters and their "
                    "plausible ranges. State your auxiliary assumptions explicitly."
                ),
                "required_output": "TheoryCommitment object (JSON)",
                "context": self.state.summary_for_agent("{agent_name}"),
                "max_rounds": 1,  # each agent speaks once
            },
            Phase.DIVERGENCE_MAPPING: {
                "goal": (
                    "Using the registered models, identify experimental conditions "
                    "where predictions diverge most. Call your model on a grid of "
                    "plausible category structures and report where you and your "
                    "opponents disagree quantitatively."
                ),
                "required_output": "Divergence report with quantitative predictions",
                "context": self._divergence_context(),
                "max_rounds": 2,  # models report, then discuss
            },
            Phase.EXPERIMENT_PROPOSAL: {
                "goal": (
                    "Given the divergence map, propose an experiment that would "
                    "be maximally diagnostic from YOUR theoretical perspective. "
                    "IMPORTANT: Look at the 'Predicted accuracy' line for each "
                    "structure. Choose a structure where YOUR model has the "
                    "HIGHEST predicted accuracy relative to other models — this "
                    "is where your theory's advantage will be most visible in "
                    "the data. Avoid structures where all models perform "
                    "similarly or where your model is weakest."
                ),
                "required_output": "ExperimentProposal JSON",
                "context": self.state.summary_for_agent("{agent_name}"),
                "max_rounds": 1,
            },
            Phase.ADVERSARIAL_CRITIQUE: {
                "goal": (
                    "Critique the other agents' experiment proposals. Specifically:\n"
                    "1. Show that under alternative auxiliary assumptions, YOUR model "
                    "   could also produce their predicted pattern.\n"
                    "2. Identify confounds that make the proposal non-diagnostic.\n"
                    "3. Demonstrate quantitatively (by calling your model) that the "
                    "   proposed design doesn't discriminate as well as claimed.\n"
                    "Be specific and quantitative. Vague objections don't count."
                ),
                "required_output": "Structured critique with quantitative evidence",
                "context": self._proposals_context(),
                "max_rounds": 3,  # critique, rebut, final response
            },
            Phase.DESIGN_REVISION: {
                "goal": (
                    "Revise your experiment proposal in light of the critiques. "
                    "Address specific objections. You may also propose a JOINT design "
                    "that incorporates insights from multiple critiques."
                ),
                "required_output": "Revised ExperimentProposal JSON",
                "context": self._critique_context(),
                "max_rounds": 2,
            },
            Phase.HUMAN_ARBITRATION: {
                "goal": (
                    "MODERATOR: Review the proposals and critiques. Select or "
                    "synthesize a final experiment design. You may:\n"
                    "1. Approve one proposal as-is.\n"
                    "2. Edit a proposal (specify changes).\n"
                    "3. Synthesize elements from multiple proposals.\n"
                    "4. Reject all and ask for a new round (with guidance)."
                ),
                "required_output": "Approved experiment design + moderator notes",
                "context": self._full_round_context(),
                "max_rounds": None,  # human decides when done
            },
            Phase.EXECUTION: {
                "goal": (
                    "Run the approved experiment. Before seeing data, each agent "
                    "MUST register a quantitative prediction for the expected results. "
                    "This prediction is logged and will be scored against actual data."
                ),
                "required_output": "Predictions registered, data returned",
                "context": self._approved_experiment_context(),
                "max_rounds": 1,
            },
            Phase.INTERPRETATION: {
                "goal": (
                    "Interpret the experimental results from your theoretical stance. "
                    "Specifically:\n"
                    "1. How well did your model's prediction match the data?\n"
                    "2. Does this support, challenge, or leave unchanged your theory?\n"
                    "3. Do you need to revise your theory? If so, specify WHAT changes "
                    "   and register the revised model.\n"
                    "4. What experiment should come next?"
                ),
                "required_output": "Interpretation + optional theory revision",
                "context": self._results_context(),
                "max_rounds": 2,
            },
            Phase.AUDIT: {
                "goal": (
                    "SYSTEM: Summarize this cycle.\n"
                    "- What was established?\n"
                    "- Which predictions were accurate?\n"
                    "- What theories were revised?\n"
                    "- What disputes remain open?\n"
                    "- What should the next cycle focus on?"
                ),
                "required_output": "Cycle summary + next cycle focus",
                "context": self.state.summary_for_agent("SYSTEM"),
                "max_rounds": 1,
            },
        }
        return specs[phase]

    # --- Phase transitions ---

    def advance_phase(self, result: PhaseResult) -> Phase:
        """
        Record phase result and determine next phase.
        Returns the next Phase.
        """
        self.phase_history.append(result)

        transition_map = {
            Phase.COMMITMENT: Phase.DIVERGENCE_MAPPING,
            Phase.DIVERGENCE_MAPPING: Phase.EXPERIMENT_PROPOSAL,
            Phase.EXPERIMENT_PROPOSAL: Phase.ADVERSARIAL_CRITIQUE,
            Phase.ADVERSARIAL_CRITIQUE: Phase.DESIGN_REVISION,
            Phase.DESIGN_REVISION: Phase.HUMAN_ARBITRATION,
            Phase.HUMAN_ARBITRATION: Phase.EXECUTION,
            Phase.EXECUTION: Phase.INTERPRETATION,
            Phase.INTERPRETATION: Phase.AUDIT,
            Phase.AUDIT: Phase.DIVERGENCE_MAPPING,  # new cycle
        }

        if self.current_phase == Phase.AUDIT:
            self.state.advance_cycle()

        self.current_phase = transition_map[self.current_phase]
        return self.current_phase

    def skip_to_phase(self, target: Phase):
        """Set the current phase directly, with validation."""
        if not isinstance(target, Phase):
            raise ValueError(f"Expected a Phase enum member, got {type(target)}")
        self.current_phase = target

    # --- Divergence mapping computation ---

    def compute_divergence_map(
        self,
        structures: Optional[dict] = None,
    ) -> dict:
        """
        Run all registered models on a set of category structures
        and identify where predictions diverge most.

        This is the quantitative backbone of the debate — it ensures
        that proposals are grounded in actual model behavior.
        """
        if structures is None:
            # Use all structures from the registry for comprehensive divergence
            structures = STRUCTURE_REGISTRY

        results = {}
        for struct_name, struct in structures.items():
            struct_results = {}
            for agent_config in self.agent_configs:
                model = agent_config.model_class
                # Get predictions for each item (leave-one-out)
                stimuli_arr = np.asarray(struct["stimuli"])
                labels_arr = np.asarray(struct["labels"])
                item_probs = []
                for idx, (item, label) in enumerate(
                    zip(struct["stimuli"], struct["labels"])
                ):
                    call_params = {**agent_config.default_params}
                    # Deterministic seed for stochastic models (e.g. RULEX)
                    if isinstance(model, RULEX):
                        call_params.setdefault("seed", 42)
                    loo_stimuli = np.delete(stimuli_arr, idx, axis=0)
                    loo_labels = np.delete(labels_arr, idx)
                    pred = model.predict(item, loo_stimuli, loo_labels, **call_params)
                    item_probs.append(pred["probabilities"].get(0, 0.5))

                # Compute accuracy (proportion of items correctly classified)
                correct = 0
                for i, (item, label) in enumerate(
                    zip(struct["stimuli"], struct["labels"])
                ):
                    pred_label = 0 if item_probs[i] > 0.5 else 1
                    if pred_label == label:
                        correct += 1
                accuracy = correct / len(struct["labels"])

                struct_results[agent_config.name] = {
                    "item_probabilities": item_probs,
                    "accuracy": accuracy,
                }

            # Compute pairwise divergence between agents
            agent_names = list(struct_results.keys())
            divergences = {}
            for i in range(len(agent_names)):
                for j in range(i + 1, len(agent_names)):
                    a, b = agent_names[i], agent_names[j]
                    probs_a = np.nan_to_num(
                        np.array(struct_results[a]["item_probabilities"]),
                        nan=0.5,
                    )
                    probs_b = np.nan_to_num(
                        np.array(struct_results[b]["item_probabilities"]),
                        nan=0.5,
                    )
                    divergences[f"{a}_vs_{b}"] = {
                        "mean_abs_diff": float(np.mean(np.abs(probs_a - probs_b))),
                        "max_diff_item": int(np.argmax(np.abs(probs_a - probs_b))),
                        "max_diff_value": float(np.max(np.abs(probs_a - probs_b))),
                    }

            results[struct_name] = {
                "predictions": struct_results,
                "divergences": divergences,
            }

        return results

    # --- Model-based predictions for agents ---

    def compute_model_predictions(
        self,
        agent_config: AgentConfig,
        structure_name: str,
        condition: str = "baseline",
        param_overrides: Optional[dict] = None,
    ) -> dict:
        """
        Run an agent's model on a category structure and return per-item
        predictions as P(correct label) for each item.

        Params are layered: agent defaults → condition effects → param_overrides.

        Returns dict with "mean_accuracy", "item_0" through "item_N" (all
        floats in [0, 1]), and "params_used" (the final merged params).
        """
        # Resolve structure (fallback to Type_II)
        if structure_name in STRUCTURE_REGISTRY:
            struct = STRUCTURE_REGISTRY[structure_name]
        else:
            struct = STRUCTURE_REGISTRY["Type_II"]

        stimuli = np.asarray(struct["stimuli"])
        labels = np.asarray(struct["labels"])

        # Build params: agent defaults → condition overrides → agent overrides
        params = dict(agent_config.default_params)
        if condition in CONDITION_EFFECTS:
            model_key = agent_config.model_class.name.split()[0]
            cond_overrides = CONDITION_EFFECTS[condition].get(model_key, {})
            params.update(cond_overrides)
        elif condition and condition != "baseline":
            print(f"⚠ Unknown condition '{condition}'; using defaults")
        if param_overrides:
            params.update(param_overrides)

        model = agent_config.model_class

        # Deterministic seed for RULEX
        if isinstance(model, RULEX):
            params.setdefault("seed", 42)

        # Filter params to only include keys that model.predict() accepts.
        # LLM agents may propose overrides with invented parameter names
        # (e.g., 'w_i' for GCM) that cause TypeError.
        valid_params = set(inspect.signature(model.predict).parameters.keys())
        # Exclude positional args (stimulus, training_items, training_labels)
        valid_params -= {"self", "stimulus", "training_items", "training_labels"}
        params = {k: v for k, v in params.items() if k in valid_params}

        # Get per-item P(correct label) using leave-one-out:
        # When predicting item i, exclude it from the training set.
        # This prevents self-prediction bias (e.g., GCM always matching
        # item i to itself with distance=0, similarity=1.0).
        #
        # If LLM-provided param values crash the model (wrong shapes,
        # bad types), fall back to condition-applied params (without the
        # malformed LLM overrides). This preserves the experimental
        # condition (e.g. low_attention → c=1.5) instead of reverting to
        # bare agent defaults (c=3.0).
        condition_params = dict(agent_config.default_params)
        if condition in CONDITION_EFFECTS:
            model_key = agent_config.model_class.name.split()[0]
            condition_params.update(CONDITION_EFFECTS[condition].get(model_key, {}))
        fallback_params = {
            k: v for k, v in condition_params.items() if k in valid_params
        }
        item_accuracies = {}
        for i, (stim, label) in enumerate(zip(stimuli, labels)):
            loo_stimuli = np.delete(stimuli, i, axis=0)
            loo_labels = np.delete(labels, i)
            try:
                pred = model.predict(stim, loo_stimuli, loo_labels, **params)
            except (ValueError, TypeError):
                # Malformed override values — fall back to condition params
                params = fallback_params
                pred = model.predict(stim, loo_stimuli, loo_labels, **params)
            p_correct = pred["probabilities"].get(int(label), 0.5)
            item_accuracies[f"item_{i}"] = float(p_correct)

        mean_acc = float(np.mean(list(item_accuracies.values())))

        # Record params actually used (exclude non-serializable values)
        params_used = {
            k: v
            for k, v in params.items()
            if not isinstance(v, np.ndarray) and v is not None
        }

        result = {"mean_accuracy": mean_acc, "params_used": params_used}
        result.update(item_accuracies)
        return result

    # --- Learning curve predictions ---

    def compute_learning_curve_predictions(
        self,
        structure_name: str,
        condition: str = "baseline",
        n_epochs: int = 3,
        block_size: int = 2,
    ) -> dict:
        """Run predict_learning_curve() for all models on a structure.

        Args:
            structure_name: key from STRUCTURE_REGISTRY.
            condition: experimental condition for param overrides.
            n_epochs: number of times to cycle through training items.
            block_size: items per block for the learning curve.

        Returns:
            {agent_name: list[dict]} where each dict has at least "accuracy" and "block".
        """
        if structure_name not in STRUCTURE_REGISTRY:
            structure_name = "Type_II"
        struct = STRUCTURE_REGISTRY[structure_name]
        stimuli = np.asarray(struct["stimuli"])
        labels = np.asarray(struct["labels"])

        # Build training sequence: n_epochs passes through the items
        training_sequence = []
        for _ in range(n_epochs):
            for stim, label in zip(stimuli, labels):
                training_sequence.append((np.asarray(stim), int(label)))

        test_items = stimuli
        test_labels = labels

        curves = {}
        for agent in self.agent_configs:
            model = agent.model_class

            # Build params with condition overrides
            params = dict(agent.default_params)
            if condition in CONDITION_EFFECTS:
                model_key = model.name.split()[0]
                cond_overrides = CONDITION_EFFECTS[condition].get(model_key, {})
                params.update(cond_overrides)

            # Filter to valid params for predict_learning_curve
            valid_params = set(
                inspect.signature(model.predict_learning_curve).parameters.keys()
            )
            valid_params -= {
                "self",
                "training_sequence",
                "test_items",
                "test_labels",
                "block_size",
            }
            filtered_params = {k: v for k, v in params.items() if k in valid_params}

            # RULEX needs deterministic seed
            if isinstance(model, RULEX):
                filtered_params.setdefault("seed", 42)

            curve = model.predict_learning_curve(
                training_sequence,
                test_items,
                test_labels,
                block_size=block_size,
                **filtered_params,
            )
            curves[agent.name] = curve

        return curves

    # --- Context generators for each phase ---

    def _divergence_context(self, div_map=None) -> str:
        """Context for the divergence mapping phase."""
        if div_map is None:
            div_map = self.compute_divergence_map()
        lines = ["## Divergence Map (Auto-computed)\n"]
        for struct_name, data in div_map.items():
            lines.append(f"### {struct_name}")
            for agent, preds in data["predictions"].items():
                lines.append(f"  {agent}: accuracy = {preds['accuracy']:.3f}")
            for pair, div in data["divergences"].items():
                lines.append(
                    f"  {pair}: mean divergence = {div['mean_abs_diff']:.3f}, "
                    f"max at item {div['max_diff_item']} ({div['max_diff_value']:.3f})"
                )
            lines.append("")

        # Ranked list of structures by maximum divergence
        ranked = []
        for struct_name, data in div_map.items():
            max_div = max(
                (d["mean_abs_diff"] for d in data["divergences"].values()),
                default=0.0,
            )
            ranked.append((struct_name, max_div))
        ranked.sort(key=lambda x: x[1], reverse=True)

        lines.append("## Structures Ranked by Maximum Divergence\n")
        lines.append("Use these `structure_name` values in experiment proposals:\n")
        for rank, (name, div) in enumerate(ranked, 1):
            desc = STRUCTURE_DESCRIPTIONS.get(name, "")
            lines.append(f"  {rank}. `{name}` — max divergence = {div:.3f}")
            if desc:
                lines.append(f"     {desc}")
            # Show per-model predicted accuracy so agents can see their advantage
            preds = div_map[name]["predictions"]
            acc_parts = [f"{agent}: {p['accuracy']:.2f}" for agent, p in preds.items()]
            lines.append(f"     Predicted accuracy: {', '.join(acc_parts)}")
        lines.append("")

        return "\n".join(lines)

    def _proposals_context(self) -> str:
        """Context for the adversarial critique phase."""
        current_proposals = [
            e
            for e in self.state.experiments
            if e.cycle == self.state.cycle and e.status == "proposed"
        ]
        lines = ["## Current Experiment Proposals\n"]
        for p in current_proposals:
            lines.append(f"### {p.title} (proposed by {p.proposed_by})")
            lines.append(f"Rationale: {p.rationale}")
            try:
                spec_str = json.dumps(p.design_spec, indent=2)
            except TypeError:
                spec_str = str(p.design_spec)
            lines.append(f"Design spec: {spec_str}")
            lines.append("")
        return "\n".join(lines)

    def _critique_context(self) -> str:
        """Context for the design revision phase."""
        current_proposals = [
            e for e in self.state.experiments if e.cycle == self.state.cycle
        ]
        lines = ["## Proposals and Critiques\n"]
        for p in current_proposals:
            lines.append(f"### {p.title} (proposed by {p.proposed_by})")
            for critique in p.critique_log:
                lines.append(f"  **{critique['agent']}**: {critique['critique']}")
                if critique.get("evidence"):
                    lines.append(f"  Evidence: {json.dumps(critique['evidence'])}")
            lines.append("")
        return "\n".join(lines)

    def _full_round_context(self) -> str:
        """Full context for human arbitration."""
        return (
            self.state.summary_for_agent("MODERATOR")
            + "\n\n"
            + self._critique_context()
        )

    def _approved_experiment_context(self) -> str:
        """Context for the execution phase."""
        approved = [
            e
            for e in self.state.experiments
            if e.cycle == self.state.cycle and e.status == "approved"
        ]
        if not approved:
            return "No experiments approved this cycle."
        exp = approved[0]
        return (
            f"## Approved Experiment: {exp.title}\n"
            f"Design: {json.dumps(exp.design_spec, indent=2)}\n"
            f"Moderator edits: {exp.moderator_edits or 'None'}\n\n"
            "EACH AGENT MUST NOW REGISTER A QUANTITATIVE PREDICTION "
            "before data is revealed."
        )

    def _results_context(self) -> str:
        """Context for the interpretation phase."""
        executed = [
            e
            for e in self.state.experiments
            if e.cycle == self.state.cycle and e.status == "executed"
        ]
        if not executed:
            return "No experiments executed this cycle."
        exp = executed[0]

        # Include prediction scores
        preds = [
            p for p in self.state.predictions if p.experiment_id == exp.experiment_id
        ]
        lines = [
            f"## Experiment Results: {exp.title}\n",
            f"Data: {json.dumps(exp.data, indent=2)}\n",
            "### Prediction Scores:",
        ]
        for p in preds:
            lines.append(
                f"  {p.agent_name}: predicted {p.predicted_pattern}, score = {p.score}"
            )
        return "\n".join(lines)

    # --- Synthetic experiment runner ---

    def _synthetic_runner(
        self,
        design_spec: dict,
        true_model: str = "GCM",
        cycle: int = 0,
    ) -> dict:
        """
        Generate synthetic data from a specified ground-truth model.

        Looks up `design_spec["structure_name"]` from STRUCTURE_REGISTRY.
        Applies condition overrides from CONDITION_EFFECTS on top of base params.
        Uses a deterministic but experiment-varying seed so different experiments
        produce different noise.
        """
        # --- Resolve category structure ---
        structure_name = design_spec.get("structure_name", "")
        if structure_name in STRUCTURE_REGISTRY:
            struct = STRUCTURE_REGISTRY[structure_name]
        else:
            # Fallback: pick highest-divergence structure from divergence map
            try:
                div_map = self.compute_divergence_map()
                best_name = None
                best_div = -1.0
                for sname, sdata in div_map.items():
                    for pair_key, pair_div in sdata["divergences"].items():
                        if pair_div["mean_abs_diff"] > best_div:
                            best_div = pair_div["mean_abs_diff"]
                            best_name = sname
                # Map divergence map keys (e.g. "II") to registry keys (e.g. "Type_II")
                if best_name and f"Type_{best_name}" in STRUCTURE_REGISTRY:
                    structure_name = f"Type_{best_name}"
                elif best_name and best_name in STRUCTURE_REGISTRY:
                    structure_name = best_name
                else:
                    structure_name = "Type_II"
            except Exception:
                structure_name = "Type_II"
            struct = STRUCTURE_REGISTRY[structure_name]

        stimuli = np.asarray(struct["stimuli"])
        labels = np.asarray(struct["labels"])

        # --- Resolve condition and base params ---
        condition = design_spec.get("condition", "baseline")
        if condition not in CONDITION_EFFECTS:
            condition = "baseline"

        if true_model == "GCM":
            model = GCM()
            params = {"c": 4.0, "attention_weights": None, "r": 1, "gamma": 1.0}
            overrides = CONDITION_EFFECTS[condition].get("GCM", {})
        elif true_model == "SUSTAIN":
            model = SUSTAIN()
            params = {"r": 9.01, "beta": 1.252, "d": 16.924, "eta": 0.092}
            overrides = CONDITION_EFFECTS[condition].get("SUSTAIN", {})
        elif true_model == "RULEX":
            model = RULEX()
            params = {"p_single": 0.5, "p_conj": 0.3, "error_tolerance": 0.1}
            overrides = CONDITION_EFFECTS[condition].get("RULEX", {})
        else:
            raise ValueError(f"Unknown model: {true_model}")

        # Apply condition overrides
        params.update(overrides)

        # --- Get model predictions for each item ---
        item_probs = {}
        for i, (stim, label) in enumerate(zip(stimuli, labels)):
            pred = model.predict(stim, stimuli, labels, **params)
            item_probs[f"item_{i}"] = {
                int(k): float(v) for k, v in pred["probabilities"].items()
            }

        # --- Add noise with a deterministic, experiment-varying seed ---
        seed_input = f"{cycle}_{structure_name}_{condition}"
        seed_hash = int(hashlib.md5(seed_input.encode()).hexdigest()[:8], 16) % 10000
        seed = 42 + seed_hash
        n_subjects = design_spec.get("n_subjects_recommended", 30)
        if not isinstance(n_subjects, int) or n_subjects < 1:
            n_subjects = 30

        rng = np.random.default_rng(seed)
        noisy_accuracy = {}
        for item_key, probs in item_probs.items():
            correct_cat = labels[int(item_key.split("_")[1])]
            p_correct = probs.get(correct_cat, 0.5)
            n_correct = rng.binomial(n_subjects, np.clip(p_correct, 0.01, 0.99))
            noisy_accuracy[item_key] = n_correct / n_subjects

        return {
            "item_accuracies": noisy_accuracy,
            "mean_accuracy": float(np.mean(list(noisy_accuracy.values()))),
            "model_predictions": item_probs,
            "n_subjects": n_subjects,
            "ground_truth_model": true_model,
            "structure_name": structure_name,
            "condition": condition,
            "params_used": {
                k: v
                for k, v in params.items()
                if not isinstance(v, np.ndarray) and v is not None
            },
        }
