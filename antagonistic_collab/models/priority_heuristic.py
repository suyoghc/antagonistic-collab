"""
Priority Heuristic — Brandstätter, Gigerenzer & Hertwig (2006).

A non-compensatory, lexicographic model of risky choice. Examines
reasons in a fixed priority order and stops at the first decisive
difference. No information integration, no parameter estimation.

For gains:
    1. Compare minimum gains. If difference >= 1/10 of max outcome, choose
       the gamble with the higher minimum. STOP.
    2. Compare probabilities of minimum gain. If difference >= 0.10, choose
       the gamble with the lower probability of minimum gain. STOP.
    3. Compare maximum gains. Choose the higher maximum. STOP.

For losses, the priority order reverses (maximum loss first).

Parameters:
    outcome_threshold (float): Aspiration level for outcome comparison.
        Default: 1/10 of the maximum outcome across both gambles.
    prob_threshold (float): Aspiration level for probability comparison.
        Default: 0.10.
    phi (float): Noise/temperature for probabilistic version. 0 = deterministic.

References:
    Brandstätter, E., Gigerenzer, G., & Hertwig, R. (2006). The priority
    heuristic: Making choices without trade-offs. Psychological Review,
    113(2), 409-432.
"""

import numpy as np


class PriorityHeuristic:
    """Priority Heuristic for risky choice."""

    name = "Priority Heuristic"
    description = (
        "Brandstätter, Gigerenzer & Hertwig (2006). Lexicographic, "
        "non-compensatory choice: examine reasons in priority order, "
        "stop at first decisive difference. No information integration. "
        "Predicts discrete choice patterns and ignored information."
    )
    core_claims = [
        "People do NOT integrate all information — they use one-reason stopping.",
        "Reasons are examined in a fixed priority order (minimum outcome first).",
        "A reason is decisive when the difference exceeds an aspiration threshold.",
        "Most information is never examined — search terminates early.",
        "No trade-offs: a large advantage on one dimension cannot compensate for a small disadvantage on another.",
    ]

    def __init__(self):
        self.default_params = {
            "outcome_threshold_frac": 0.1,  # 1/10 of max outcome
            "prob_threshold": 0.10,
            "phi": 0.5,  # noise for probabilistic version
        }

    def _get_min_max(self, outcomes, probs):
        """Extract minimum/maximum outcomes and their probabilities."""
        pairs = list(zip(outcomes, probs))
        if len(pairs) == 1:
            return {
                "min_outcome": pairs[0][0],
                "max_outcome": pairs[0][0],
                "prob_min": pairs[0][1],
                "prob_max": pairs[0][1],
            }
        sorted_pairs = sorted(pairs, key=lambda x: x[0])
        return {
            "min_outcome": sorted_pairs[0][0],
            "max_outcome": sorted_pairs[-1][0],
            "prob_min": sorted_pairs[0][1],
            "prob_max": sorted_pairs[-1][1],
        }

    def _is_gain_domain(self, gamble):
        """Check if all outcomes are non-negative."""
        all_outcomes = list(gamble["outcomes_A"]) + list(gamble["outcomes_B"])
        return all(x >= 0 for x in all_outcomes)

    def predict(self, gamble, **params):
        """Predict P(choose gamble A) using the Priority Heuristic.

        Args:
            gamble: dict with keys outcomes_A, probs_A, outcomes_B, probs_B
            outcome_threshold_frac: fraction of max outcome for aspiration (default 0.1)
            prob_threshold: probability difference threshold (default 0.10)
            phi: noise parameter (0 = deterministic, higher = more noise)

        Returns:
            dict with p_choose_A, reason (which step was decisive),
            and deterministic_choice
        """
        otf = params.get(
            "outcome_threshold_frac",
            self.default_params["outcome_threshold_frac"],
        )
        pt = params.get("prob_threshold", self.default_params["prob_threshold"])
        phi = params.get("phi", self.default_params["phi"])

        info_a = self._get_min_max(gamble["outcomes_A"], gamble["probs_A"])
        info_b = self._get_min_max(gamble["outcomes_B"], gamble["probs_B"])

        # Determine aspiration threshold for outcomes
        all_outcomes = list(gamble["outcomes_A"]) + list(gamble["outcomes_B"])
        max_outcome = max(abs(x) for x in all_outcomes) if all_outcomes else 1
        outcome_threshold = otf * max_outcome

        is_gain = self._is_gain_domain(gamble)

        if is_gain:
            # GAINS: priority = minimum gain → prob of minimum → maximum gain
            # Step 1: Compare minimum gains
            diff_min = info_a["min_outcome"] - info_b["min_outcome"]
            if abs(diff_min) >= outcome_threshold:
                # Choose gamble with higher minimum gain
                prefer_a = diff_min > 0
                reason = "minimum_gain"
            else:
                # Step 2: Compare probabilities of minimum gain
                diff_prob = info_a["prob_min"] - info_b["prob_min"]
                if abs(diff_prob) >= pt:
                    # Choose gamble with LOWER probability of minimum gain
                    prefer_a = diff_prob < 0
                    reason = "probability"
                else:
                    # Step 3: Compare maximum gains
                    diff_max = info_a["max_outcome"] - info_b["max_outcome"]
                    prefer_a = diff_max > 0
                    reason = "maximum_gain"
        else:
            # LOSSES: priority = maximum loss → prob of maximum loss → minimum loss
            # Step 1: Compare maximum losses (least negative = better)
            diff_max_loss = info_a["max_outcome"] - info_b["max_outcome"]
            if abs(diff_max_loss) >= outcome_threshold:
                # Choose gamble with less severe maximum loss (higher value)
                prefer_a = diff_max_loss > 0
                reason = "maximum_loss"
            else:
                # Step 2: Compare probabilities of maximum loss
                diff_prob = info_a["prob_max"] - info_b["prob_max"]
                if abs(diff_prob) >= pt:
                    prefer_a = diff_prob < 0
                    reason = "probability"
                else:
                    # Step 3: Compare minimum losses
                    diff_min_loss = info_a["min_outcome"] - info_b["min_outcome"]
                    prefer_a = diff_min_loss > 0
                    reason = "minimum_loss"

        # Probabilistic version: add noise around deterministic choice
        if phi <= 0:
            p_a = 1.0 if prefer_a else 0.0
        else:
            # Sigmoid with noise
            strength = 1.0  # deterministic preference strength
            if not prefer_a:
                strength = -1.0
            p_a = 1.0 / (1.0 + np.exp(-strength / phi))

        return {
            "p_choose_A": float(p_a),
            "reason": reason,
            "deterministic_choice": "A" if prefer_a else "B",
        }

    def predict_batch(self, gambles, **params):
        """Predict for multiple gamble pairs."""
        return [self.predict(g, **params) for g in gambles]
