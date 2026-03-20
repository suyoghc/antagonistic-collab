"""Tests for decision-making models: EU, CPT, Priority Heuristic.

Each model takes a gamble pair and returns P(choose gamble A).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from antagonistic_collab.models.expected_utility import ExpectedUtility
from antagonistic_collab.models.prospect_theory import CumulativeProspectTheory
from antagonistic_collab.models.priority_heuristic import PriorityHeuristic


# ── Test Gambles ──

# Simple: $100 for sure vs 50/50 $200 or $0
SAFE_VS_RISKY = {
    "outcomes_A": [100],
    "probs_A": [1.0],
    "outcomes_B": [200, 0],
    "probs_B": [0.5, 0.5],
}

# Allais common consequence (scaled to manageable magnitudes)
ALLAIS_S1 = {
    "outcomes_A": [100],
    "probs_A": [1.0],
    "outcomes_B": [500, 100, 0],
    "probs_B": [0.10, 0.89, 0.01],
}

ALLAIS_S2 = {
    "outcomes_A": [100, 0],
    "probs_A": [0.11, 0.89],
    "outcomes_B": [500, 0],
    "probs_B": [0.10, 0.90],
}

# Loss aversion: 50/50 win $100 / lose $100
MIXED_SYMMETRIC = {
    "outcomes_A": [0],
    "probs_A": [1.0],
    "outcomes_B": [100, -100],
    "probs_B": [0.5, 0.5],
}

# Fourfold: high prob gain
HIGH_PROB_GAIN = {
    "outcomes_A": [950],
    "probs_A": [1.0],
    "outcomes_B": [1000, 0],
    "probs_B": [0.95, 0.05],
}

# Fourfold: low prob gain (lottery)
LOW_PROB_GAIN = {
    "outcomes_A": [50],
    "probs_A": [1.0],
    "outcomes_B": [1000, 0],
    "probs_B": [0.05, 0.95],
}


def _gamble_dict(g):
    """Convert test gamble to model input format."""
    return g


# ── Expected Utility Tests ──


class TestExpectedUtility:
    def test_risk_neutral_indifferent_on_equal_ev(self):
        """Risk-neutral EU should be indifferent between safe and risky with same EV."""
        eu = ExpectedUtility()
        p = eu.predict(SAFE_VS_RISKY, r=0.0)  # r=0 is risk neutral
        assert abs(p["p_choose_A"] - 0.5) < 0.15

    def test_risk_averse_prefers_safe(self):
        """Risk-averse EU should prefer the safe option."""
        eu = ExpectedUtility()
        p = eu.predict(SAFE_VS_RISKY, r=0.5)
        assert p["p_choose_A"] > 0.5

    def test_no_loss_aversion(self):
        """EU treats gains and losses symmetrically — should accept fair mixed gamble."""
        eu = ExpectedUtility()
        p = eu.predict(MIXED_SYMMETRIC, r=0.0)
        # Risk-neutral: EV of gamble = 0, EV of safe = 0 → indifferent
        assert abs(p["p_choose_A"] - 0.5) < 0.15

    def test_allais_consistency(self):
        """EU predicts consistent choices in both Allais situations."""
        eu = ExpectedUtility()
        p1 = eu.predict(ALLAIS_S1, r=0.3)
        p2 = eu.predict(ALLAIS_S2, r=0.3)
        # If prefer A in S1, should prefer A in S2 (independence axiom)
        # Both should go same direction
        prefer_safe_s1 = p1["p_choose_A"] > 0.5
        prefer_safe_s2 = p2["p_choose_A"] > 0.5
        assert prefer_safe_s1 == prefer_safe_s2

    def test_returns_valid_probability(self):
        eu = ExpectedUtility()
        p = eu.predict(SAFE_VS_RISKY, r=0.3)
        assert 0.0 <= p["p_choose_A"] <= 1.0

    def test_predict_batch(self):
        eu = ExpectedUtility()
        gambles = [SAFE_VS_RISKY, ALLAIS_S1, HIGH_PROB_GAIN]
        results = eu.predict_batch(gambles, r=0.3)
        assert len(results) == 3
        assert all(0.0 <= r["p_choose_A"] <= 1.0 for r in results)


# ── Cumulative Prospect Theory Tests ──


class TestCPT:
    def test_loss_aversion(self):
        """CPT should reject symmetric mixed gamble due to loss aversion."""
        cpt = CumulativeProspectTheory()
        p = cpt.predict(MIXED_SYMMETRIC, lambda_=2.25)
        # Should prefer safe (A) because losses loom larger
        assert p["p_choose_A"] > 0.5

    def test_certainty_effect(self):
        """CPT should overweight certainty — prefer sure thing over risky with higher EV."""
        cpt = CumulativeProspectTheory()
        # $100 certain vs 80% of $150 (EV = $120 > $100, but certainty dominates)
        certainty_test = {
            "outcomes_A": [100],
            "probs_A": [1.0],
            "outcomes_B": [150, 0],
            "probs_B": [0.80, 0.20],
        }
        p = cpt.predict(certainty_test)
        # CPT underweights 0.80, making the risky option less attractive
        assert p["p_choose_A"] > 0.5  # prefer certainty despite lower EV

    def test_fourfold_risk_averse_high_prob_gain(self):
        """CPT: risk-averse for high-probability gains (underweight high probs)."""
        cpt = CumulativeProspectTheory()
        p = cpt.predict(HIGH_PROB_GAIN)
        assert p["p_choose_A"] > 0.5  # prefer safe $950 over 95% $1000

    def test_fourfold_risk_seeking_low_prob_gain(self):
        """CPT: risk-seeking for low-probability gains (overweight small probs)."""
        cpt = CumulativeProspectTheory()
        p = cpt.predict(LOW_PROB_GAIN)
        assert p["p_choose_A"] < 0.5  # prefer 5% at $1000 over safe $50

    def test_returns_valid_probability(self):
        cpt = CumulativeProspectTheory()
        p = cpt.predict(SAFE_VS_RISKY)
        assert 0.0 <= p["p_choose_A"] <= 1.0

    def test_predict_batch(self):
        cpt = CumulativeProspectTheory()
        gambles = [SAFE_VS_RISKY, ALLAIS_S1, MIXED_SYMMETRIC]
        results = cpt.predict_batch(gambles)
        assert len(results) == 3
        assert all(0.0 <= r["p_choose_A"] <= 1.0 for r in results)


# ── Priority Heuristic Tests ──


class TestPriorityHeuristic:
    def test_minimum_gain_decisive(self):
        """When minimum gains differ enough, PH stops at step 1."""
        ph = PriorityHeuristic()
        # A: $100 certain vs B: $200 or $0
        # Min gain A = 100, min gain B = 0. Diff = 100 > 1/10 * 200 = 20
        # PH chooses A (higher minimum)
        p = ph.predict(SAFE_VS_RISKY)
        assert p["p_choose_A"] > 0.5
        assert p["reason"] == "minimum_gain"

    def test_probability_decisive(self):
        """When minimum gains are similar, PH checks probabilities."""
        ph = PriorityHeuristic()
        gamble = {
            "outcomes_A": [100, 90],
            "probs_A": [0.5, 0.5],
            "outcomes_B": [100, 85],
            "probs_B": [0.3, 0.7],
        }
        # Min gains: A=90, B=85. Diff=5. Threshold=1/10*100=10. Not decisive.
        # Prob of min: A=0.5, B=0.7. Diff=0.2 >= 0.10. Decisive.
        # PH chooses A (lower probability of minimum gain)
        p = ph.predict(gamble)
        assert p["p_choose_A"] > 0.5

    def test_returns_valid_probability(self):
        ph = PriorityHeuristic()
        p = ph.predict(SAFE_VS_RISKY)
        assert 0.0 <= p["p_choose_A"] <= 1.0

    def test_reason_field_present(self):
        """PH should report which step was decisive."""
        ph = PriorityHeuristic()
        p = ph.predict(SAFE_VS_RISKY)
        assert p["reason"] in ("minimum_gain", "probability", "maximum_gain")

    def test_predict_batch(self):
        ph = PriorityHeuristic()
        gambles = [SAFE_VS_RISKY, HIGH_PROB_GAIN, LOW_PROB_GAIN]
        results = ph.predict_batch(gambles)
        assert len(results) == 3
        assert all(0.0 <= r["p_choose_A"] <= 1.0 for r in results)
