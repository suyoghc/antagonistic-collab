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
) -> ModelPosterior:
    """Update the posterior after observing experimental results.

    Computes log-likelihoods of the observed data under each model
    and performs a Bayesian update.

    Args:
        posterior: current ModelPosterior
        protocol: DebateProtocol instance
        data: experimental results dict (with item_accuracies)
        structure_name: which category structure was tested
        condition: experimental condition
        cycle: current debate cycle
        n_subjects: assumed subjects per item

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

    # Record history before update
    history_entry = {
        "cycle": cycle,
        "structure": structure_name,
        "condition": condition,
        "log_likelihoods": log_likelihoods.tolist(),
        "prior_probs": posterior.probs.tolist(),
    }

    posterior.update(log_likelihoods)

    history_entry["posterior_probs"] = posterior.probs.tolist()
    history_entry["entropy"] = posterior.entropy
    posterior.history.append(history_entry)

    return posterior
