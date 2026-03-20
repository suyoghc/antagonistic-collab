"""
Expected Utility Theory — von Neumann & Morgenstern (1944).

The normative baseline for decision under risk. Decision-maker maximizes
probability-weighted utility with a concave (risk-averse) or convex
(risk-seeking) utility function.

Parameters:
    r (float): Risk aversion coefficient (CRRA power utility).
        r > 0: risk-averse, r = 0: risk-neutral, r < 0: risk-seeking.
    temperature (float): Response noise. Higher = more deterministic.
"""

import numpy as np


class ExpectedUtility:
    """Expected Utility model for risky choice."""

    name = "Expected Utility"
    description = (
        "von Neumann & Morgenstern (1944). Maximize probability-weighted "
        "utility. Predicts consistent risk attitudes, no loss aversion, "
        "no framing effects. The normative benchmark."
    )
    core_claims = [
        "People maximize expected utility of outcomes.",
        "Risk attitude is determined by utility function curvature.",
        "Probabilities enter linearly — no distortion.",
        "Gains and losses are treated symmetrically.",
        "The independence axiom holds.",
    ]

    def __init__(self):
        self.default_params = {
            "r": 0.5,
            "temperature": 1.0,
        }

    def _utility(self, x: float, r: float) -> float:
        """CRRA power utility: u(x) = x^(1-r) / (1-r) for gains.

        For losses and zero, uses sign-preserving formulation.
        """
        if r == 1.0:
            if x <= 0:
                return -1e10 if x < 0 else 0.0
            return np.log(x)

        if x >= 0:
            return x ** (1.0 - r) / (1.0 - r) if x > 0 else 0.0
        else:
            # Symmetric treatment for losses
            return -((-x) ** (1.0 - r) / (1.0 - r))

    def _expected_utility(self, outcomes, probs, r):
        """Compute EU of a gamble."""
        return sum(p * self._utility(x, r) for x, p in zip(outcomes, probs))

    def predict(self, gamble, **params):
        """Predict P(choose gamble A) for a gamble pair.

        Args:
            gamble: dict with keys outcomes_A, probs_A, outcomes_B, probs_B
            r: risk aversion coefficient
            temperature: softmax temperature (default 1.0)

        Returns:
            dict with p_choose_A and expected utilities
        """
        r = params.get("r", self.default_params["r"])
        temperature = params.get("temperature", self.default_params["temperature"])

        eu_a = self._expected_utility(gamble["outcomes_A"], gamble["probs_A"], r)
        eu_b = self._expected_utility(gamble["outcomes_B"], gamble["probs_B"], r)

        # Softmax choice rule
        diff = temperature * (eu_a - eu_b)
        diff = np.clip(diff, -500, 500)
        p_a = 1.0 / (1.0 + np.exp(-diff))

        return {
            "p_choose_A": float(p_a),
            "eu_A": float(eu_a),
            "eu_B": float(eu_b),
        }

    def predict_batch(self, gambles, **params):
        """Predict for multiple gamble pairs."""
        return [self.predict(g, **params) for g in gambles]
