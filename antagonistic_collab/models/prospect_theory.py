"""
Cumulative Prospect Theory — Tversky & Kahneman (1992).

The dominant descriptive model of decision under risk. Extends expected
utility with reference dependence, loss aversion, diminishing sensitivity,
and nonlinear probability weighting.

Parameters:
    alpha (float): Value function exponent for gains (0 < alpha <= 1).
    beta (float): Value function exponent for losses (0 < beta <= 1).
    lambda_ (float): Loss aversion coefficient (> 1 means loss-averse).
    gamma_pos (float): Probability weighting for gains (0 < gamma <= 1).
    gamma_neg (float): Probability weighting for losses (0 < gamma <= 1).
    temperature (float): Response noise.

References:
    Tversky, A. & Kahneman, D. (1992). Advances in prospect theory:
    Cumulative representation of uncertainty. J. Risk & Uncertainty, 5, 297-323.
"""

import numpy as np


class CumulativeProspectTheory:
    """Cumulative Prospect Theory for risky choice."""

    name = "Cumulative Prospect Theory"
    description = (
        "Tversky & Kahneman (1992). Reference-dependent evaluation with "
        "S-shaped value function, loss aversion, and inverse-S probability "
        "weighting. Predicts the fourfold pattern, Allais paradox, and "
        "framing effects."
    )
    core_claims = [
        "Outcomes are evaluated as gains/losses relative to a reference point.",
        "The value function is concave for gains, convex for losses (diminishing sensitivity).",
        "Losses loom larger than gains (loss aversion, lambda ~ 2.25).",
        "Small probabilities are overweighted, large probabilities underweighted.",
        "Decision weights are rank-dependent (cumulative weighting).",
    ]

    def __init__(self):
        self.default_params = {
            "alpha": 0.88,
            "beta": 0.88,
            "lambda_": 2.25,
            "gamma_pos": 0.61,
            "gamma_neg": 0.69,
            "temperature": 1.0,
        }

    def _value(self, x, alpha, beta, lambda_):
        """S-shaped value function: concave for gains, convex for losses."""
        if x >= 0:
            return x**alpha
        else:
            return -lambda_ * ((-x) ** beta)

    def _weight_function(self, p, gamma):
        """Tversky-Kahneman probability weighting function (inverse-S)."""
        if p <= 0:
            return 0.0
        if p >= 1:
            return 1.0
        return p**gamma / (p**gamma + (1 - p) ** gamma) ** (1.0 / gamma)

    def _cumulative_weights(self, outcomes, probs, gamma, is_gain):
        """Compute rank-dependent decision weights.

        For gains: sort descending, weight by cumulative from best to worst.
        For losses: sort ascending, weight by cumulative from worst to best.
        """
        n = len(outcomes)
        if n == 0:
            return []

        if is_gain:
            # Sort by outcome descending (best first)
            order = np.argsort(outcomes)[::-1]
        else:
            # Sort by outcome ascending (worst first)
            order = np.argsort(outcomes)

        sorted_probs = [probs[i] for i in order]
        weights = []
        cumulative = 0.0
        for i in range(n):
            new_cum = cumulative + sorted_probs[i]
            w_new = self._weight_function(new_cum, gamma)
            w_old = self._weight_function(cumulative, gamma)
            weights.append(w_new - w_old)
            cumulative = new_cum

        # Restore original order
        result = [0.0] * n
        for i, idx in enumerate(order):
            result[idx] = weights[i]

        return result

    def _prospect_value(
        self, outcomes, probs, alpha, beta, lambda_, gamma_pos, gamma_neg
    ):
        """Compute CPT value of a prospect (gains and losses separated)."""
        outcomes = list(outcomes)
        probs = list(probs)

        # Separate gains and losses
        gain_outcomes, gain_probs = [], []
        loss_outcomes, loss_probs = [], []

        for x, p in zip(outcomes, probs):
            if x >= 0:
                gain_outcomes.append(x)
                gain_probs.append(p)
            else:
                loss_outcomes.append(x)
                loss_probs.append(p)

        # Compute decision weights
        gain_weights = self._cumulative_weights(
            gain_outcomes, gain_probs, gamma_pos, is_gain=True
        )
        loss_weights = self._cumulative_weights(
            loss_outcomes, loss_probs, gamma_neg, is_gain=False
        )

        # Compute value
        v = 0.0
        for x, w in zip(gain_outcomes, gain_weights):
            v += w * self._value(x, alpha, beta, lambda_)
        for x, w in zip(loss_outcomes, loss_weights):
            v += w * self._value(x, alpha, beta, lambda_)

        return v

    def predict(self, gamble, **params):
        """Predict P(choose gamble A) for a gamble pair.

        Args:
            gamble: dict with keys outcomes_A, probs_A, outcomes_B, probs_B
            alpha, beta, lambda_, gamma_pos, gamma_neg: CPT parameters
            temperature: softmax temperature

        Returns:
            dict with p_choose_A and prospect values
        """
        alpha = params.get("alpha", self.default_params["alpha"])
        beta = params.get("beta", self.default_params["beta"])
        lambda_ = params.get("lambda_", self.default_params["lambda_"])
        gamma_pos = params.get("gamma_pos", self.default_params["gamma_pos"])
        gamma_neg = params.get("gamma_neg", self.default_params["gamma_neg"])
        temperature = params.get("temperature", self.default_params["temperature"])

        v_a = self._prospect_value(
            gamble["outcomes_A"],
            gamble["probs_A"],
            alpha,
            beta,
            lambda_,
            gamma_pos,
            gamma_neg,
        )
        v_b = self._prospect_value(
            gamble["outcomes_B"],
            gamble["probs_B"],
            alpha,
            beta,
            lambda_,
            gamma_pos,
            gamma_neg,
        )

        # Softmax choice rule
        diff = temperature * (v_a - v_b)
        diff = np.clip(diff, -500, 500)
        p_a = 1.0 / (1.0 + np.exp(-diff))

        return {
            "p_choose_A": float(p_a),
            "value_A": float(v_a),
            "value_B": float(v_b),
        }

    def predict_batch(self, gambles, **params):
        """Predict for multiple gamble pairs."""
        return [self.predict(g, **params) for g in gambles]
