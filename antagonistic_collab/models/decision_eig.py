"""Decision-domain EIG adapter.

Bridges decision model predictions (P(choose A) per gamble) into the
domain-agnostic compute_eig() and posterior update machinery used by
the categorization pipeline.

The key mapping:
    categorization "item"       → decision "gamble"
    P(correct category)         → P(choose A)
    category structure          → gamble group (diagnostic theme)
    agent + model class         → agent + decision model

References:
    Myung & Pitt (2009). Optimal experimental design for model discrimination.
    Cavagnaro et al. (2010). Adaptive design optimization.
"""

from __future__ import annotations

import numpy as np

from antagonistic_collab.bayesian_selection import (
    ModelPosterior,
    compute_eig,
    compute_log_likelihood,
)
from antagonistic_collab.models.decision_runner import (
    DECISION_AGENT_MAP,
    GT_DECISION_PARAMS,
    compute_decision_predictions,
)


# ── Gamble Groups ──
# Analogous to STRUCTURE_REGISTRY candidates in categorization.
# Each group is a set of thematically related gambles tested together.

GAMBLE_GROUPS = {
    "certainty_effect": [
        "certainty_effect_1",
        "certainty_effect_2",
    ],
    "common_ratio": [
        "common_ratio_high",
        "common_ratio_low",
    ],
    "fourfold_gain": [
        "fourfold_high_prob_gain",
        "fourfold_low_prob_gain",
    ],
    "fourfold_loss": [
        "fourfold_high_prob_loss",
        "fourfold_low_prob_loss",
    ],
    "loss_aversion": [
        "loss_aversion_symmetric",
        "loss_aversion_2to1",
        "loss_aversion_3to1",
    ],
    "risk_premium": [
        "risk_premium_moderate",
        "risk_premium_high_stakes",
    ],
    "ph_diagnostic": [
        "ph_minimum_decisive",
        "ph_probability_decisive",
        "ph_maximum_decisive",
    ],
}


def decision_predictions_for_eig(
    gamble_names: list[str],
    agent_params: dict[str, dict] | None = None,
) -> dict[str, np.ndarray]:
    """Compute decision model predictions in the format compute_eig() expects.

    For each agent (CPT_Agent, EU_Agent, PH_Agent), computes P(choose A) for
    every gamble in gamble_names and returns as a numpy array.

    Args:
        gamble_names: list of gamble names from the registry.
        agent_params: optional {agent_name: {param: value}} overrides.
            Uses GT_DECISION_PARAMS defaults for any agent not specified.

    Returns:
        {agent_name: np.ndarray of shape (n_gambles,)} — P(choose A) per gamble,
        ordered consistently with gamble_names.
    """
    if agent_params is None:
        agent_params = {}

    predictions = {}
    for agent_name, model_name in DECISION_AGENT_MAP.items():
        params = agent_params.get(agent_name, GT_DECISION_PARAMS[model_name])
        preds = []
        for gname in gamble_names:
            p = compute_decision_predictions(model_name, gname, params=params)
            preds.append(p[gname])
        predictions[agent_name] = np.array(preds, dtype=np.float64)

    return predictions


def select_decision_experiment(
    candidates: list[list[str]],
    posterior: ModelPosterior,
    agent_params: dict[str, dict] | None = None,
    n_subjects: int = 30,
    n_sim: int = 200,
    seed: int | None = 42,
    learning_rate: float = 1.0,
    selection_strategy: str = "thompson",
) -> tuple[int, list[float]]:
    """Select the most informative gamble group via EIG.

    Analogous to select_from_pool() in categorization.

    Args:
        candidates: list of gamble-name lists. Each element is a group
            of gambles to evaluate together (like a category structure).
        posterior: current ModelPosterior over decision agents.
        agent_params: optional parameter overrides per agent.
        n_subjects: simulated subjects per gamble.
        n_sim: Monte Carlo simulations per model.
        seed: random seed.
        learning_rate: likelihood tempering parameter.
        selection_strategy: "greedy" or "thompson".

    Returns:
        (best_index, eig_scores) — selected index and EIG for each candidate.
    """
    eig_scores = []
    for i, gamble_names in enumerate(candidates):
        preds = decision_predictions_for_eig(gamble_names, agent_params)
        eig = compute_eig(
            preds,
            posterior,
            n_subjects=n_subjects,
            n_sim=n_sim,
            seed=seed + i if seed is not None else None,
            learning_rate=learning_rate,
        )
        eig_scores.append(eig)

    if selection_strategy == "greedy":
        best_idx = int(np.argmax(eig_scores))
    elif selection_strategy == "thompson":
        rng = np.random.default_rng(seed)
        scores = np.array(eig_scores)
        if scores.sum() < 1e-10:
            best_idx = rng.integers(len(scores))
        else:
            probs = scores / scores.sum()
            best_idx = int(rng.choice(len(scores), p=probs))
    else:
        raise ValueError(f"Unknown selection strategy: {selection_strategy}")

    return best_idx, eig_scores


def update_decision_posterior(
    posterior: ModelPosterior,
    observed_choices: dict[str, float],
    gamble_names: list[str],
    agent_params: dict[str, dict] | None = None,
    n_subjects: int = 30,
    learning_rate: float = 1.0,
) -> ModelPosterior:
    """Bayesian posterior update from observed choice data.

    Analogous to update_posterior_from_experiment() in categorization.

    Args:
        posterior: current ModelPosterior (updated in place).
        observed_choices: {gamble_name: observed P(choose A)} from data.
        gamble_names: which gambles were observed (determines ordering).
        agent_params: optional parameter overrides.
        n_subjects: number of subjects per gamble.
        learning_rate: likelihood tempering parameter.

    Returns:
        The same posterior (updated in place).
    """
    # Build observed array in gamble_names order
    observed = np.array([observed_choices[g] for g in gamble_names], dtype=np.float64)

    # Compute log-likelihoods under each model
    preds = decision_predictions_for_eig(gamble_names, agent_params)
    log_likelihoods = np.zeros(len(posterior.model_names))

    for m_idx, agent_name in enumerate(posterior.model_names):
        predicted = preds[agent_name]
        log_likelihoods[m_idx] = compute_log_likelihood(observed, predicted, n_subjects)

    posterior.update(log_likelihoods, learning_rate=learning_rate)
    return posterior
