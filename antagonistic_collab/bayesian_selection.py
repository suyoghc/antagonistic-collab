"""
Bayesian Information-Gain Experiment Selection.

Replaces the heuristic diversity penalty (D17) with principled adaptive
design: maintain a posterior over models and select experiments that
maximize expected information gain (EIG).

References:
    Myung & Pitt (2009). Optimal experimental design for model discrimination.
    Cavagnaro et al. (2010). Adaptive design optimization in experiments with people.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import binom


@dataclass
class ModelPosterior:
    """Bayesian posterior over competing models.

    Stores log-probabilities for numerical stability.
    """

    log_probs: np.ndarray  # shape (n_models,)
    model_names: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)

    @property
    def probs(self) -> np.ndarray:
        """Normalized posterior probabilities."""
        shifted = self.log_probs - np.max(self.log_probs)
        p = np.exp(shifted)
        return p / p.sum()

    @property
    def entropy(self) -> float:
        """Shannon entropy of the posterior (nats)."""
        p = self.probs
        return -float(np.sum(p * np.log(p + 1e-30)))

    def update(self, log_likelihoods: np.ndarray):
        """Bayesian update: posterior ∝ prior × likelihood.

        Args:
            log_likelihoods: shape (n_models,) — log P(data | model_i)
        """
        log_likelihoods = np.asarray(log_likelihoods, dtype=np.float64)
        self.log_probs = self.log_probs + log_likelihoods
        # Re-center for numerical stability
        self.log_probs = self.log_probs - np.max(self.log_probs)

    def to_dict(self) -> dict:
        """Serialize for JSON storage."""
        return {
            "log_probs": self.log_probs.tolist(),
            "model_names": self.model_names,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ModelPosterior:
        """Reconstruct from serialized dict."""
        return cls(
            log_probs=np.array(d["log_probs"], dtype=np.float64),
            model_names=d.get("model_names", []),
            history=d.get("history", []),
        )

    @classmethod
    def uniform(cls, model_names: list[str]) -> ModelPosterior:
        """Create a uniform prior over n models."""
        n = len(model_names)
        return cls(
            log_probs=np.zeros(n, dtype=np.float64),
            model_names=model_names,
        )


def compute_log_likelihood(
    observed: np.ndarray,
    predicted: np.ndarray,
    n_subjects: int,
) -> float:
    """Item-level binomial log-likelihood.

    For each item, recover n_correct from observed proportion,
    clip predicted P(correct) to [0.01, 0.99], and sum
    binom.logpmf across items.

    Args:
        observed: shape (n_items,) — observed P(correct) per item
        predicted: shape (n_items,) — model-predicted P(correct) per item
        n_subjects: number of simulated subjects (trials per item)

    Returns:
        Total log-likelihood (float).
    """
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    # Clip predictions to avoid log(0)
    predicted = np.clip(predicted, 0.01, 0.99)

    # Recover integer counts from proportions
    n_correct = np.round(observed * n_subjects).astype(int)
    n_correct = np.clip(n_correct, 0, n_subjects)

    # Sum log-PMF across items
    ll = np.sum(binom.logpmf(n_correct, n_subjects, predicted))
    return float(ll)


def compute_eig(
    model_predictions: dict[str, np.ndarray],
    posterior: ModelPosterior,
    n_subjects: int = 20,
    n_sim: int = 200,
    seed: Optional[int] = 42,
) -> float:
    """Monte Carlo expected information gain for a candidate experiment.

    EIG = H(current posterior) - E[H(posterior | simulated data)]

    For each model as hypothetical ground truth (weighted by prior),
    simulate n_sim datasets via binomial sampling, compute posterior
    update for each, and average the resulting entropy.

    Args:
        model_predictions: {model_name: array of P(correct) per item}
            Must have same keys and order as posterior.model_names.
        posterior: current ModelPosterior
        n_subjects: number of simulated subjects per item
        n_sim: number of Monte Carlo simulations per model
        seed: random seed for reproducibility

    Returns:
        EIG in nats (non-negative float).
    """
    rng = np.random.default_rng(seed)

    h_current = posterior.entropy
    model_names = posterior.model_names
    n_models = len(model_names)
    priors = posterior.probs

    # Collect prediction arrays in order
    pred_arrays = []
    for name in model_names:
        p = np.asarray(model_predictions[name], dtype=np.float64)
        p = np.clip(p, 0.01, 0.99)
        pred_arrays.append(p)

    n_items = len(pred_arrays[0])

    # For each model as ground truth, simulate datasets and compute
    # expected posterior entropy
    expected_h = 0.0
    for gt_idx in range(n_models):
        if priors[gt_idx] < 1e-10:
            continue  # skip models with negligible prior

        gt_preds = pred_arrays[gt_idx]

        # Simulate n_sim datasets: shape (n_sim, n_items)
        sim_correct = rng.binomial(n_subjects, gt_preds, size=(n_sim, n_items))
        sim_observed = sim_correct / n_subjects

        # For each simulated dataset, compute log-likelihoods under all models
        # and get posterior entropy
        entropies = np.zeros(n_sim)
        for s in range(n_sim):
            obs = sim_observed[s]
            # Log-likelihoods under each model
            lls = np.zeros(n_models)
            for m_idx in range(n_models):
                lls[m_idx] = compute_log_likelihood(obs, pred_arrays[m_idx], n_subjects)

            # Posterior update (on a copy)
            new_lp = posterior.log_probs + lls
            new_lp = new_lp - np.max(new_lp)
            new_p = np.exp(new_lp)
            new_p = new_p / new_p.sum()
            entropies[s] = -np.sum(new_p * np.log(new_p + 1e-30))

        expected_h += priors[gt_idx] * np.mean(entropies)

    eig = h_current - expected_h
    return max(0.0, float(eig))  # EIG >= 0 by Jensen's inequality


def generate_full_candidate_pool(
    protocol,
    extra_structures: Optional[dict] = None,
) -> list[tuple[str, str]]:
    """All structure×condition pairs from STRUCTURE_REGISTRY × CONDITION_EFFECTS.

    Args:
        protocol: DebateProtocol instance (for access to registries).
        extra_structures: Optional dict of additional structures (e.g., from
            novel agent proposals) to include in the pool.

    Returns:
        List of (structure_name, condition) tuples.
    """
    from .debate_protocol import STRUCTURE_REGISTRY, CONDITION_EFFECTS

    structures = dict(STRUCTURE_REGISTRY)
    if extra_structures:
        structures.update(extra_structures)

    pool = []
    for struct_name in structures:
        for condition in CONDITION_EFFECTS:
            pool.append((struct_name, condition))
    return pool


def _pairwise_divergence(
    model_predictions: dict[str, np.ndarray],
    pair: tuple[str, str],
) -> float:
    """Mean absolute prediction divergence between two models."""
    a, b = pair
    if a not in model_predictions or b not in model_predictions:
        return 0.0
    return float(np.mean(np.abs(model_predictions[a] - model_predictions[b])))


def select_from_pool(
    protocol,
    posterior: ModelPosterior,
    pool: list[tuple[str, str]],
    n_subjects: int = 20,
    n_sim: int = 200,
    seed: Optional[int] = 42,
    focus_pair: Optional[tuple[str, str]] = None,
    pair_boost: float = 1.5,
    crux_boost_specs: Optional[list[dict]] = None,
) -> tuple[int, list[float]]:
    """Select the best (structure, condition) pair from the full pool by EIG.

    Args:
        protocol: DebateProtocol instance (for compute_model_predictions).
        posterior: current ModelPosterior.
        pool: list of (structure_name, condition) tuples.
        n_subjects: subjects per item for likelihood computation.
        n_sim: Monte Carlo simulations per model per candidate.
        seed: random seed.
        focus_pair: optional (model_name_a, model_name_b) to boost EIG
            for candidates where these two models diverge.
        pair_boost: multiplier for EIG when focus pair has high divergence.
        crux_boost_specs: optional list of dicts with keys "structure",
            "condition", "boost". Matching candidates get EIG multiplied
            by boost.

    Returns:
        (best_index, eig_scores) — index into pool, and EIG for each candidate.
    """
    eig_scores = []

    for i, (struct_name, condition) in enumerate(pool):
        # Get predictions from each model for this candidate
        model_predictions = {}
        for agent_config in protocol.agent_configs:
            preds = protocol.compute_model_predictions(
                agent_config, struct_name, condition
            )
            item_keys = sorted(
                [k for k in preds if k.startswith("item_")],
                key=lambda k: int(k.split("_")[1]),
            )
            pred_array = np.array([preds[k] for k in item_keys])
            model_predictions[agent_config.name] = pred_array

        eig = compute_eig(
            model_predictions,
            posterior,
            n_subjects=n_subjects,
            n_sim=n_sim,
            seed=seed + i if seed is not None else None,
        )

        # Apply focus pair boost
        if focus_pair is not None:
            divergence = _pairwise_divergence(model_predictions, focus_pair)
            if divergence > 0.05:  # Only boost if meaningfully divergent
                eig *= pair_boost

        # Apply crux boost
        if crux_boost_specs:
            for spec in crux_boost_specs:
                if (
                    spec.get("structure") == struct_name
                    and spec.get("condition") == condition
                ):
                    eig *= spec.get("boost", 1.0)

        eig_scores.append(eig)

    best_idx = int(np.argmax(eig_scores))
    return best_idx, eig_scores


def select_experiment(
    protocol,
    posterior: ModelPosterior,
    candidates: list,
    n_subjects: int = 20,
    n_sim: int = 200,
    seed: Optional[int] = 42,
) -> tuple[int, list[float]]:
    """Select the experiment that maximizes expected information gain.

    Args:
        protocol: DebateProtocol instance (for compute_model_predictions)
        posterior: current ModelPosterior
        candidates: list of ExperimentRecord proposals
        n_subjects: subjects per item for likelihood computation
        n_sim: Monte Carlo simulations per model per candidate
        seed: random seed

    Returns:
        (best_index, eig_scores) — index into candidates, and EIG for each.
    """
    eig_scores = []

    for i, proposal in enumerate(candidates):
        design = proposal.design_spec if isinstance(proposal.design_spec, dict) else {}
        struct_name = design.get("structure_name", "Type_II")
        condition = design.get("condition", "baseline")

        # Get predictions from each model for this candidate
        model_predictions = {}
        for agent_config in protocol.agent_configs:
            preds = protocol.compute_model_predictions(
                agent_config, struct_name, condition
            )
            # Extract item-level predictions as array
            item_keys = sorted(
                [k for k in preds if k.startswith("item_")],
                key=lambda k: int(k.split("_")[1]),
            )
            pred_array = np.array([preds[k] for k in item_keys])
            model_predictions[agent_config.name] = pred_array

        eig = compute_eig(
            model_predictions,
            posterior,
            n_subjects=n_subjects,
            n_sim=n_sim,
            seed=seed + i if seed is not None else None,
        )
        eig_scores.append(eig)

    best_idx = int(np.argmax(eig_scores))
    return best_idx, eig_scores


def update_posterior_from_experiment(
    posterior: ModelPosterior,
    protocol,
    data: dict,
    structure_name: str,
    condition: str,
    cycle: int,
    n_subjects: int = 20,
    learning_curves: Optional[dict] = None,
    curve_weight: float = 0.5,
) -> ModelPosterior:
    """Update the posterior after observing experimental results.

    Computes log-likelihoods of the observed data under each model
    and performs a Bayesian update. Optionally incorporates learning
    curve evidence.

    Args:
        posterior: current ModelPosterior
        protocol: DebateProtocol instance
        data: experimental results dict (with item_accuracies)
        structure_name: which category structure was tested
        condition: experimental condition
        cycle: current debate cycle
        n_subjects: assumed subjects per item
        learning_curves: optional {agent_name: curve_list} from
            compute_learning_curve_predictions(). If provided,
            curve-shape RMSE is added as additional evidence.
        curve_weight: weight of curve evidence relative to accuracy
            evidence (default 0.5 = half weight).

    Returns:
        The same posterior (updated in place), for convenience.
    """
    # Extract observed item accuracies
    item_accs = data.get("item_accuracies", {})
    if not item_accs:
        return posterior  # no item-level data to update from

    item_keys = sorted(item_accs.keys(), key=lambda k: int(k.split("_")[1]))
    observed = np.array([item_accs[k] for k in item_keys])

    # Compute log-likelihoods under each model
    log_likelihoods = np.zeros(len(posterior.model_names))
    for m_idx, agent_config in enumerate(protocol.agent_configs):
        preds = protocol.compute_model_predictions(
            agent_config, structure_name, condition
        )
        pred_keys = sorted(
            [k for k in preds if k.startswith("item_")],
            key=lambda k: int(k.split("_")[1]),
        )
        predicted = np.array([preds[k] for k in pred_keys])

        log_likelihoods[m_idx] = compute_log_likelihood(observed, predicted, n_subjects)

    # Add learning curve evidence if provided
    if learning_curves:
        # Simulate ground-truth curve from data's ground_truth_model
        gt_model = data.get("ground_truth_model", "")
        if gt_model:
            gt_curve = None
            # Find ground truth model's curve by matching model name
            for agent_config in protocol.agent_configs:
                model_key = agent_config.model_class.name.split()[0]
                if model_key == gt_model and agent_config.name in learning_curves:
                    gt_curve = learning_curves[agent_config.name]
                    break

            if gt_curve:
                gt_accs = np.array([b["accuracy"] for b in gt_curve])
                for m_idx, agent_config in enumerate(protocol.agent_configs):
                    if agent_config.name in learning_curves:
                        pred_curve = learning_curves[agent_config.name]
                        pred_accs = np.array([b["accuracy"] for b in pred_curve])
                        # Match lengths
                        min_len = min(len(gt_accs), len(pred_accs))
                        if min_len > 0:
                            curve_rmse = float(
                                np.sqrt(
                                    np.mean(
                                        (gt_accs[:min_len] - pred_accs[:min_len]) ** 2
                                    )
                                )
                            )
                            # Convert RMSE to log-likelihood-like score
                            # Lower RMSE = higher score
                            curve_ll = -curve_rmse * n_subjects * curve_weight
                            log_likelihoods[m_idx] += curve_ll

    # Record history before update
    history_entry = {
        "cycle": cycle,
        "structure": structure_name,
        "condition": condition,
        "log_likelihoods": log_likelihoods.tolist(),
        "prior_probs": posterior.probs.tolist(),
        "has_curve_evidence": learning_curves is not None,
    }

    posterior.update(log_likelihoods)

    history_entry["posterior_probs"] = posterior.probs.tolist()
    history_entry["entropy"] = posterior.entropy
    posterior.history.append(history_entry)

    return posterior


# ---------------------------------------------------------------------------
# Focus pair extraction
# ---------------------------------------------------------------------------


def extract_focus_pair_from_posterior(
    posterior: ModelPosterior,
) -> tuple[str, str]:
    """Identify the two models with closest posterior probabilities.

    These are the models the experiment should try to distinguish.
    """
    probs = posterior.probs
    names = posterior.model_names
    n = len(names)

    min_gap = float("inf")
    pair = (names[0], names[1]) if n >= 2 else (names[0], names[0])
    for i in range(n):
        for j in range(i + 1, n):
            gap = abs(probs[i] - probs[j])
            if gap < min_gap:
                min_gap = gap
                pair = (names[i], names[j])
    return pair


def extract_focus_pair_from_ledger(state) -> Optional[tuple[str, str]]:
    """Extract contested model pair from recent claims in the ledger.

    Looks for the two agents most frequently involved in falsified or
    contested (untested) claims against each other.
    """
    from collections import Counter

    agents_in_disputes = Counter()
    for claim in state.claim_ledger:
        if claim.status in ("falsified", "untested") and claim.claim_type == "critique":
            agents_in_disputes[claim.agent] += 1

    if len(agents_in_disputes) >= 2:
        top_two = agents_in_disputes.most_common(2)
        return (top_two[0][0], top_two[1][0])
    return None
