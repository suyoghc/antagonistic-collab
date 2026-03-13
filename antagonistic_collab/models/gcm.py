"""
Generalized Context Model (GCM) — Nosofsky (1986).

The exemplar model of categorization. Classification of a stimulus is based
on its summed similarity to all stored exemplars of each category, weighted
by dimensional attention.

Parameters:
    c (float): Sensitivity/specificity parameter. Higher = sharper similarity gradient.
    attention_weights (array): Per-dimension attention weights, sum to 1.
    r (int): Distance metric. 1 = city-block, 2 = Euclidean.
    gamma (float): Response scaling parameter (>= 1). Higher = more deterministic.
    bias (dict): Category response bias. Keys are category labels, values are bias weights.
"""

import numpy as np
from scipy.optimize import differential_evolution
from typing import Optional


class GCM:
    """Generalized Context Model for categorization."""

    name = "GCM (Exemplar Model)"
    description = (
        "Nosofsky (1986). Categorization by summed similarity to stored exemplars. "
        "Predicts sensitivity to individual training items, no information loss, "
        "good performance on non-linearly-separable categories."
    )
    core_claims = [
        "People store individual exemplars, not summaries or rules.",
        "Classification is based on similarity to all stored instances.",
        "Attention weights over dimensions are learned and flexibly deployed.",
        "No information is lost during category learning — all instances are retained.",
    ]

    def __init__(self):
        self.default_params = {
            "c": 3.0,
            "attention_weights": None,  # uniform by default
            "r": 1,  # city-block
            "gamma": 1.0,
            "bias": None,  # uniform by default
        }

    def _distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        attention_weights: np.ndarray,
        r: int = 1,
    ) -> float:
        """Weighted Minkowski distance between two stimuli."""
        if r <= 0:
            raise ValueError(
                f"Distance metric r must be positive (got r={r}). "
                "Use r=1 for city-block or r=2 for Euclidean."
            )
        diff = np.abs(x - y)
        if r == 1:
            return np.sum(attention_weights * diff)
        else:
            return np.sum(attention_weights * diff**r) ** (1.0 / r)

    def _similarity(self, distance: float, c: float) -> float:
        """Exponential similarity function: eta = exp(-c * d)."""
        return np.exp(-c * distance)

    def predict(
        self,
        stimulus: np.ndarray,
        training_items: np.ndarray,
        training_labels: np.ndarray,
        c: float = 3.0,
        attention_weights: Optional[np.ndarray] = None,
        r: int = 1,
        gamma: float = 1.0,
        bias: Optional[dict] = None,
    ) -> dict:
        """
        Predict category response probabilities for a single stimulus.

        Returns:
            dict with keys:
                "probabilities": {label: P(label|stimulus)}
                "similarities": {label: summed similarity to that category}
        """
        stimulus = np.asarray(stimulus, dtype=float)
        training_items = np.asarray(training_items, dtype=float)
        n_dims = stimulus.shape[0]

        if attention_weights is None:
            attention_weights = np.ones(n_dims) / n_dims
        attention_weights = np.asarray(attention_weights, dtype=float)

        categories = sorted(set(training_labels))
        if bias is None:
            bias = {cat: 1.0 for cat in categories}
        else:
            missing = [cat for cat in categories if cat not in bias]
            if missing:
                raise ValueError(
                    f"Bias dict is missing categories {missing}. "
                    f"Expected keys: {categories}"
                )

        # Compute summed similarity to each category
        cat_similarities = {}
        for cat in categories:
            cat_mask = training_labels == cat
            cat_items = training_items[cat_mask]
            total_sim = 0.0
            for item in cat_items:
                d = self._distance(stimulus, item, attention_weights, r)
                total_sim += self._similarity(d, c)
            cat_similarities[cat] = total_sim

        # Response rule (Luce choice)
        numerators = {
            cat: bias[cat] * (cat_similarities[cat] ** gamma) for cat in categories
        }
        total = sum(numerators.values())
        if total == 0:
            probs = {cat: 1.0 / len(categories) for cat in categories}
        else:
            probs = {cat: numerators[cat] / total for cat in categories}

        return {
            "probabilities": probs,
            "similarities": cat_similarities,
        }

    def predict_batch(
        self,
        test_items: np.ndarray,
        training_items: np.ndarray,
        training_labels: np.ndarray,
        **params,
    ) -> list[dict]:
        """Predict for multiple test items."""
        return [
            self.predict(item, training_items, training_labels, **params)
            for item in test_items
        ]

    def predict_learning_curve(
        self,
        training_sequence: list[tuple[np.ndarray, int]],
        test_items: np.ndarray,
        test_labels: np.ndarray,
        block_size: int = 1,
        **params,
    ) -> list[dict]:
        """
        Predict accuracy over the course of learning.

        training_sequence: list of (stimulus, label) pairs in presentation order.
        test_items: items to test at each block boundary.
        test_labels: correct labels for test items.
        block_size: number of training trials per block.

        Returns list of dicts, one per block:
            {"block": int, "accuracy": float, "item_probs": list}
        """
        stored_items = []
        stored_labels = []
        curve = []

        for block_idx in range(0, len(training_sequence), block_size):
            block = training_sequence[block_idx : block_idx + block_size]

            # Add block items to memory (exemplar storage is immediate & complete)
            for stim, label in block:
                stored_items.append(stim)
                stored_labels.append(label)

            if len(stored_items) == 0:
                continue

            if len(test_items) == 0:
                continue

            # Test current state
            train_arr = np.array(stored_items)
            label_arr = np.array(stored_labels)

            correct = 0
            item_probs = []
            for test_item, true_label in zip(test_items, test_labels):
                pred = self.predict(test_item, train_arr, label_arr, **params)
                item_probs.append(pred["probabilities"])
                if (
                    max(pred["probabilities"], key=pred["probabilities"].get)
                    == true_label
                ):
                    correct += 1

            curve.append(
                {
                    "block": block_idx // block_size,
                    "n_stored": len(stored_items),
                    "accuracy": correct / len(test_items),
                    "item_probabilities": item_probs,
                }
            )

        return curve

    def predict_generalization_gradient(
        self,
        probe_items: np.ndarray,
        training_items: np.ndarray,
        training_labels: np.ndarray,
        target_category: int = 0,
        **params,
    ) -> np.ndarray:
        """
        Predict P(target_category) for a range of probe items.
        Returns the generalization gradient — useful for visualizing
        the shape of the category boundary.
        """
        gradients = []
        for probe in probe_items:
            pred = self.predict(probe, training_items, training_labels, **params)
            gradients.append(pred["probabilities"].get(target_category, 0.0))
        return np.array(gradients)

    def fit(
        self,
        training_items: np.ndarray,
        training_labels: np.ndarray,
        response_data: np.ndarray,
        r: int = 1,
        gamma: float = 1.0,
        seed: Optional[int] = None,
    ) -> dict:
        """
        Fit GCM parameters (c and attention weights) to response data.

        response_data: array of shape (n_items,) with P(category 0) for each item.
            Typically averaged over participants.

        Returns:
            dict with "c", "attention_weights", "loss", "predictions"
        """
        n_dims = training_items.shape[1]

        def objective(params_flat):
            c = params_flat[0]
            raw_weights = params_flat[1:]
            # Softmax to ensure weights sum to 1
            exp_w = np.exp(raw_weights - np.max(raw_weights))
            attention_weights = exp_w / exp_w.sum()

            total_loss = 0.0
            for i, item in enumerate(training_items):
                pred = self.predict(
                    item,
                    training_items,
                    training_labels,
                    c=c,
                    attention_weights=attention_weights,
                    r=r,
                    gamma=gamma,
                )
                p_cat0 = pred["probabilities"].get(0, 0.5)
                # Negative log-likelihood
                p_cat0 = np.clip(p_cat0, 1e-10, 1 - 1e-10)
                target = response_data[i]
                total_loss -= target * np.log(p_cat0) + (1 - target) * np.log(
                    1 - p_cat0
                )

            return total_loss

        bounds = [(0.1, 20.0)] + [(-5.0, 5.0)] * n_dims
        result = differential_evolution(
            objective, bounds, seed=seed, maxiter=200, tol=1e-6
        )

        c_fit = result.x[0]
        raw_w = result.x[1:]
        exp_w = np.exp(raw_w - np.max(raw_w))
        attention_fit = exp_w / exp_w.sum()

        # Get predictions at best fit
        predictions = []
        for item in training_items:
            pred = self.predict(
                item,
                training_items,
                training_labels,
                c=c_fit,
                attention_weights=attention_fit,
                r=r,
                gamma=gamma,
            )
            predictions.append(pred["probabilities"].get(0, 0.5))

        return {
            "c": c_fit,
            "attention_weights": attention_fit.tolist(),
            "loss": result.fun,
            "predictions": predictions,
            "n_free_params": 1
            + n_dims,  # c + n_dims attention weights (minus 1 for sum constraint, but we count the raw params)
        }
