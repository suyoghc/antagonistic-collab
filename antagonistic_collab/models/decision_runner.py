"""
Decision-Making Experiment Runner — synthetic data generation and model scoring.

Analogous to _synthetic_runner() and compute_model_predictions() in the
categorization pipeline. Generates choice data from a ground-truth model
and scores all models against it.

The key interface difference from categorization:
- Categorization: predict P(correct category) for each item in a structure
- Decision-making: predict P(choose A) for each gamble in a set

We bridge this by treating each gamble in a registry as an "item" and
P(choose A) as the analogous "accuracy" measure.
"""

import hashlib

import numpy as np

from antagonistic_collab.models.expected_utility import ExpectedUtility
from antagonistic_collab.models.prospect_theory import CumulativeProspectTheory
from antagonistic_collab.models.priority_heuristic import PriorityHeuristic
from antagonistic_collab.models.gamble_structures import (
    GAMBLE_REGISTRY,
    FULL_GAMBLE_REGISTRY,
)


# ── Ground-truth parameters ──

GT_DECISION_PARAMS = {
    "CPT": {
        "alpha": 0.88,
        "beta": 0.88,
        "lambda_": 2.25,
        "gamma_pos": 0.61,
        "gamma_neg": 0.69,
        "temperature": 1.0,
    },
    "EU": {
        "r": 0.5,
        "temperature": 1.0,
    },
    "PH": {
        "outcome_threshold_frac": 0.1,
        "prob_threshold": 0.10,
        "phi": 0.5,
    },
}

# Misspecified params (calibrated to narrow gap without flipping winner)
MISSPEC_DECISION_PARAMS = {
    "CPT": {
        "alpha": 0.50,  # much more concave (less sensitivity to magnitude)
        "beta": 0.50,
        "lambda_": 1.2,  # weak loss aversion (vs true 2.25)
        "gamma_pos": 0.40,  # more distortion
        "gamma_neg": 0.40,
        "temperature": 1.0,
    },
    "EU": {
        "r": 0.1,  # nearly risk-neutral (vs true 0.5)
        "temperature": 1.0,
    },
    "PH": {
        "outcome_threshold_frac": 0.3,  # much wider threshold (less sensitive)
        "prob_threshold": 0.25,  # much wider
        "phi": 1.5,  # more noise
    },
}

# Agent name mapping
DECISION_AGENT_MAP = {
    "CPT_Agent": "CPT",
    "EU_Agent": "EU",
    "PH_Agent": "PH",
}

DECISION_EXPECTED_WINNER = {
    "CPT": "CPT_Agent",
    "EU": "EU_Agent",
    "PH": "PH_Agent",
}


def get_decision_model(model_name):
    """Instantiate a decision model by name."""
    if model_name == "CPT":
        return CumulativeProspectTheory()
    elif model_name == "EU":
        return ExpectedUtility()
    elif model_name == "PH":
        return PriorityHeuristic()
    else:
        raise ValueError(f"Unknown decision model: {model_name}")


def compute_decision_predictions(model_name, gamble_set_name, params=None):
    """Compute P(choose A) for each gamble in a set.

    Args:
        model_name: "CPT", "EU", or "PH"
        gamble_set_name: key in GAMBLE_REGISTRY or FULL_GAMBLE_REGISTRY,
            or "all_base" for all base registry gambles,
            or "all" for full registry including parametric.
        params: model parameters (uses defaults if None)

    Returns:
        dict of {gamble_name: p_choose_A} — analogous to item-level predictions
        in categorization.
    """
    model = get_decision_model(model_name)
    if params is None:
        params = GT_DECISION_PARAMS[model_name]

    # Determine which gambles to evaluate
    if gamble_set_name == "all_base":
        gambles = GAMBLE_REGISTRY
    elif gamble_set_name == "all":
        gambles = FULL_GAMBLE_REGISTRY
    elif gamble_set_name in FULL_GAMBLE_REGISTRY:
        gambles = {gamble_set_name: FULL_GAMBLE_REGISTRY[gamble_set_name]}
    else:
        raise ValueError(f"Unknown gamble set: {gamble_set_name}")

    predictions = {}
    for name, gamble in gambles.items():
        result = model.predict(gamble, **params)
        predictions[name] = result["p_choose_A"]

    return predictions


def generate_synthetic_choices(
    gamble_set_name, true_model, n_subjects=30, cycle=0, params=None
):
    """Generate synthetic choice data from a ground-truth decision model.

    Analogous to _synthetic_runner() in categorization. For each gamble,
    generates binomial choice data: n_subjects each independently choose
    A or B according to the model's P(choose A).

    Args:
        gamble_set_name: which gambles to run ("all_base" or specific name)
        true_model: "CPT", "EU", or "PH"
        n_subjects: number of simulated choosers per gamble
        cycle: used for deterministic seed variation
        params: ground-truth params (uses GT_DECISION_PARAMS if None)

    Returns:
        dict with:
            - item_accuracies: {gamble_name: observed P(choose A)}
            - mean_accuracy: mean across gambles
            - model_predictions: {gamble_name: true P(choose A)}
            - ground_truth_model: model name
    """
    if params is None:
        params = GT_DECISION_PARAMS[true_model]

    # Get true model predictions
    true_preds = compute_decision_predictions(true_model, gamble_set_name, params)

    # Generate noisy observations
    seed_input = f"{cycle}_{gamble_set_name}_{true_model}"
    seed_hash = int(hashlib.md5(seed_input.encode()).hexdigest()[:8], 16) % 10000
    rng = np.random.default_rng(42 + seed_hash)

    observed = {}
    for name, p_a in true_preds.items():
        p_clipped = np.clip(p_a, 0.01, 0.99)
        n_chose_a = rng.binomial(n_subjects, p_clipped)
        observed[name] = n_chose_a / n_subjects

    return {
        "item_accuracies": observed,
        "mean_accuracy": float(np.mean(list(observed.values()))),
        "model_predictions": true_preds,
        "ground_truth_model": true_model,
        "n_subjects": n_subjects,
        "gamble_set": gamble_set_name,
    }


def score_decision_models(observed, model_params=None):
    """Score all three models against observed choice data.

    Args:
        observed: dict of {gamble_name: observed P(choose A)}
        model_params: dict of {model_name: params}. Uses defaults if None.

    Returns:
        dict of {model_name: {"predictions": {...}, "rmse": float}}
    """
    if model_params is None:
        model_params = GT_DECISION_PARAMS

    results = {}
    for model_name in ["CPT", "EU", "PH"]:
        params = model_params.get(model_name, GT_DECISION_PARAMS[model_name])
        model = get_decision_model(model_name)

        predictions = {}
        errors = []
        for gamble_name, obs_p in observed.items():
            if gamble_name in FULL_GAMBLE_REGISTRY:
                gamble = FULL_GAMBLE_REGISTRY[gamble_name]
                pred = model.predict(gamble, **params)
                pred_p = pred["p_choose_A"]
                predictions[gamble_name] = pred_p
                errors.append((pred_p - obs_p) ** 2)

        rmse = float(np.sqrt(np.mean(errors))) if errors else 999.0

        results[model_name] = {
            "predictions": predictions,
            "rmse": rmse,
        }

    return results
