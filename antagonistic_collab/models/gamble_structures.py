"""
Gamble Structure Registry — diagnostic problems for decision-making.

Analogous to STRUCTURE_REGISTRY for categorization (Shepard types, 5-4, etc.).
Each entry is a gamble pair that discriminates between EU, CPT, and Priority
Heuristic in known ways.

Each structure is a dict with:
    - outcomes_A, probs_A: gamble A specification
    - outcomes_B, probs_B: gamble B specification
    - domain: "gain", "loss", or "mixed"
    - description: human-readable description
    - discriminates: which model pairs this problem distinguishes
"""

import numpy as np

# ── Core Diagnostic Problems ──

GAMBLE_REGISTRY = {
    # --- Certainty Effect (CE) ---
    # EU: indifferent (same EV). CPT: prefer certain (overweight certainty).
    # PH: prefer certain (higher minimum gain).
    "certainty_effect_1": {
        "outcomes_A": [100],
        "probs_A": [1.0],
        "outcomes_B": [150, 0],
        "probs_B": [0.80, 0.20],
        "domain": "gain",
        "description": "$100 certain vs 80% of $150 (EV=$120). Tests certainty effect.",
        "discriminates": ["EU_vs_CPT"],
    },
    "certainty_effect_2": {
        "outcomes_A": [50],
        "probs_A": [1.0],
        "outcomes_B": [80, 0],
        "probs_B": [0.75, 0.25],
        "domain": "gain",
        "description": "$50 certain vs 75% of $80 (EV=$60). Weaker certainty effect.",
        "discriminates": ["EU_vs_CPT"],
    },
    # --- Common Ratio Effect (CR) ---
    # EU: consistent across scaling. CPT: reversal due to nonlinear weighting.
    # PH: may reverse depending on aspiration thresholds.
    "common_ratio_high": {
        "outcomes_A": [100, 0],
        "probs_A": [0.80, 0.20],
        "outcomes_B": [150, 0],
        "probs_B": [0.60, 0.40],
        "domain": "gain",
        "description": "80% of $100 vs 60% of $150. High-probability version.",
        "discriminates": ["EU_vs_CPT", "EU_vs_PH"],
    },
    "common_ratio_low": {
        "outcomes_A": [100, 0],
        "probs_A": [0.20, 0.80],
        "outcomes_B": [150, 0],
        "probs_B": [0.15, 0.85],
        "domain": "gain",
        "description": "20% of $100 vs 15% of $150. Low-probability version.",
        "discriminates": ["EU_vs_CPT", "EU_vs_PH"],
    },
    # --- Fourfold Pattern ---
    # EU: uniform risk attitude. CPT: four-cell pattern (risk-averse high-p gains,
    # risk-seeking low-p gains, risk-seeking high-p losses, risk-averse low-p losses).
    # PH: partial fourfold via minimum-outcome priority.
    "fourfold_high_prob_gain": {
        "outcomes_A": [950],
        "probs_A": [1.0],
        "outcomes_B": [1000, 0],
        "probs_B": [0.95, 0.05],
        "domain": "gain",
        "description": "$950 certain vs 95% of $1000. Risk-averse for high-p gains.",
        "discriminates": ["EU_vs_CPT", "CPT_vs_PH"],
    },
    "fourfold_low_prob_gain": {
        "outcomes_A": [50],
        "probs_A": [1.0],
        "outcomes_B": [1000, 0],
        "probs_B": [0.05, 0.95],
        "domain": "gain",
        "description": "$50 certain vs 5% of $1000. Risk-seeking for low-p gains.",
        "discriminates": ["EU_vs_CPT", "CPT_vs_PH"],
    },
    "fourfold_high_prob_loss": {
        "outcomes_A": [-950],
        "probs_A": [1.0],
        "outcomes_B": [-1000, 0],
        "probs_B": [0.95, 0.05],
        "domain": "loss",
        "description": "Lose $950 certain vs 95% lose $1000. Risk-seeking for high-p losses.",
        "discriminates": ["EU_vs_CPT", "CPT_vs_PH"],
    },
    "fourfold_low_prob_loss": {
        "outcomes_A": [-50],
        "probs_A": [1.0],
        "outcomes_B": [-1000, 0],
        "probs_B": [0.05, 0.95],
        "domain": "loss",
        "description": "Lose $50 certain vs 5% lose $1000. Risk-averse for low-p losses.",
        "discriminates": ["EU_vs_CPT", "CPT_vs_PH"],
    },
    # --- Loss Aversion (LA) ---
    # EU: accept fair gambles. CPT: reject due to lambda > 1. PH: reject (minimum is loss).
    "loss_aversion_symmetric": {
        "outcomes_A": [0],
        "probs_A": [1.0],
        "outcomes_B": [100, -100],
        "probs_B": [0.5, 0.5],
        "domain": "mixed",
        "description": "$0 certain vs 50/50 win/lose $100. Tests loss aversion.",
        "discriminates": ["EU_vs_CPT", "EU_vs_PH"],
    },
    "loss_aversion_2to1": {
        "outcomes_A": [0],
        "probs_A": [1.0],
        "outcomes_B": [200, -100],
        "probs_B": [0.5, 0.5],
        "domain": "mixed",
        "description": "$0 vs 50/50 +$200/-$100. EV=$50. Tests if 2:1 ratio suffices.",
        "discriminates": ["EU_vs_CPT"],
    },
    "loss_aversion_3to1": {
        "outcomes_A": [0],
        "probs_A": [1.0],
        "outcomes_B": [300, -100],
        "probs_B": [0.5, 0.5],
        "domain": "mixed",
        "description": "$0 vs 50/50 +$300/-$100. EV=$100. Should overcome loss aversion.",
        "discriminates": ["EU_vs_CPT"],
    },
    # --- Risk Premium Elicitation ---
    # Different risk premiums under each model.
    "risk_premium_moderate": {
        "outcomes_A": [60],
        "probs_A": [1.0],
        "outcomes_B": [100, 0],
        "probs_B": [0.50, 0.50],
        "domain": "gain",
        "description": "$60 certain vs 50/50 $100/$0 (EV=$50). Tests risk premium.",
        "discriminates": ["EU_vs_CPT", "EU_vs_PH"],
    },
    "risk_premium_high_stakes": {
        "outcomes_A": [400],
        "probs_A": [1.0],
        "outcomes_B": [1000, 0],
        "probs_B": [0.50, 0.50],
        "domain": "gain",
        "description": "$400 certain vs 50/50 $1000/$0. Higher stakes risk premium.",
        "discriminates": ["EU_vs_CPT", "EU_vs_PH"],
    },
    # --- PH-Diagnostic Problems ---
    # Designed to distinguish PH from CPT/EU specifically.
    "ph_minimum_decisive": {
        "outcomes_A": [100, 50],
        "probs_A": [0.5, 0.5],
        "outcomes_B": [200, 0],
        "probs_B": [0.5, 0.5],
        "domain": "gain",
        "description": "50/50 $100/$50 vs 50/50 $200/$0. PH: minimum decisive (A). CPT/EU: may prefer B.",
        "discriminates": ["CPT_vs_PH", "EU_vs_PH"],
    },
    "ph_probability_decisive": {
        "outcomes_A": [100, 90],
        "probs_A": [0.5, 0.5],
        "outcomes_B": [105, 85],
        "probs_B": [0.3, 0.7],
        "domain": "gain",
        "description": "Similar minimums, different probabilities. PH: probability step decisive.",
        "discriminates": ["CPT_vs_PH", "EU_vs_PH"],
    },
    "ph_maximum_decisive": {
        "outcomes_A": [200, 50],
        "probs_A": [0.5, 0.5],
        "outcomes_B": [150, 55],
        "probs_B": [0.5, 0.5],
        "domain": "gain",
        "description": "Similar minimums and probs, different maximums. PH: maximum step.",
        "discriminates": ["CPT_vs_PH"],
    },
}


# ── Parametric Variants ──
# Analogous to sampled_ls_Xd and sampled_rpe structures in categorization.


def generate_parametric_gambles(n_samples=20, seed=42):
    """Generate parametric variants by varying outcome magnitudes and probabilities.

    Returns dict of gamble_name → gamble_spec, like PARAMETRIC_STRUCTURES
    in categorization.
    """
    rng = np.random.default_rng(seed)
    gambles = {}

    for i in range(n_samples):
        # Vary: outcome magnitude, probability, gain/loss/mixed
        magnitude = rng.choice([50, 100, 200, 500, 1000])
        p_high = round(rng.uniform(0.1, 0.9), 2)
        p_low = round(1.0 - p_high, 2)

        # Type 1: certain vs risky (gain)
        ce = round(magnitude * p_high * rng.uniform(0.7, 1.0))
        gambles[f"param_cert_gain_{i}"] = {
            "outcomes_A": [ce],
            "probs_A": [1.0],
            "outcomes_B": [magnitude, 0],
            "probs_B": [p_high, p_low],
            "domain": "gain",
            "description": f"${ce} certain vs {p_high * 100:.0f}% of ${magnitude}",
            "discriminates": ["EU_vs_CPT"],
        }

    for i in range(n_samples):
        # Type 2: mixed gambles (varying gain/loss ratio)
        gain = rng.choice([100, 200, 300, 400, 500])
        loss = rng.choice([50, 100, 150, 200])
        gambles[f"param_mixed_{i}"] = {
            "outcomes_A": [0],
            "probs_A": [1.0],
            "outcomes_B": [gain, -loss],
            "probs_B": [0.5, 0.5],
            "domain": "mixed",
            "description": f"$0 vs 50/50 +${gain}/-${loss}. Ratio={gain / loss:.1f}",
            "discriminates": ["EU_vs_CPT", "EU_vs_PH"],
        }

    for i in range(n_samples):
        # Type 3: two risky options (tests PH's lexicographic ordering)
        min_a = rng.choice([10, 20, 30, 40, 50])
        max_a = min_a + rng.choice([50, 100, 150, 200])
        min_b = min_a + rng.choice([-20, -10, 0, 10, 20])
        min_b = max(0, min_b)
        max_b = min_b + rng.choice([50, 100, 150, 200, 250])
        p_a = round(rng.uniform(0.3, 0.7), 2)
        p_b = round(rng.uniform(0.3, 0.7), 2)
        gambles[f"param_risky_pair_{i}"] = {
            "outcomes_A": [max_a, min_a],
            "probs_A": [p_a, round(1 - p_a, 2)],
            "outcomes_B": [max_b, min_b],
            "probs_B": [p_b, round(1 - p_b, 2)],
            "domain": "gain",
            "description": f"{p_a * 100:.0f}% ${max_a}/${min_a} vs {p_b * 100:.0f}% ${max_b}/${min_b}",
            "discriminates": ["CPT_vs_PH", "EU_vs_PH"],
        }

    return gambles


# Full registry: base + parametric
FULL_GAMBLE_REGISTRY = {**GAMBLE_REGISTRY, **generate_parametric_gambles()}
