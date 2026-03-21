"""Tests for decision-domain EIG adapter.

Verifies that decision model predictions can flow through the same
compute_eig() and posterior update machinery used for categorization.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from antagonistic_collab.models.decision_eig import (
    GAMBLE_GROUPS,
    decision_predictions_for_eig,
    select_decision_experiment,
    update_decision_posterior,
)
from antagonistic_collab.bayesian_selection import (
    ModelPosterior,
    compute_eig,
)


DECISION_AGENTS = ["CPT_Agent", "EU_Agent", "PH_Agent"]


class TestDecisionPredictionsForEIG:
    """Test that decision predictions are formatted correctly for compute_eig()."""

    def test_returns_dict_with_agent_names(self):
        """Output keys must match decision agent names."""
        preds = decision_predictions_for_eig(["certainty_effect_1"])
        assert set(preds.keys()) == set(DECISION_AGENTS)

    # Helper used throughout tests
    @staticmethod
    def _uniform_posterior():
        return ModelPosterior.uniform(DECISION_AGENTS)

    def test_returns_numpy_arrays(self):
        """Each value must be a numpy array."""
        preds = decision_predictions_for_eig(["certainty_effect_1"])
        for name, arr in preds.items():
            assert isinstance(arr, np.ndarray), f"{name} is not ndarray"

    def test_array_length_matches_gamble_count(self):
        """Array length must equal the number of gambles passed in."""
        gambles = ["certainty_effect_1", "certainty_effect_2", "common_ratio_high"]
        preds = decision_predictions_for_eig(gambles)
        for name, arr in preds.items():
            assert len(arr) == 3, f"{name} has wrong length: {len(arr)}"

    def test_predictions_are_valid_probabilities(self):
        """All predictions must be in [0, 1]."""
        preds = decision_predictions_for_eig(
            ["certainty_effect_1", "fourfold_low_prob_gain"]
        )
        for name, arr in preds.items():
            assert np.all(arr >= 0.0), f"{name} has prediction < 0"
            assert np.all(arr <= 1.0), f"{name} has prediction > 1"

    def test_consistent_array_lengths(self):
        """All model arrays must have the same length."""
        preds = decision_predictions_for_eig(
            ["certainty_effect_1", "common_ratio_high", "loss_aversion_symmetric"]
        )
        lengths = [len(arr) for arr in preds.values()]
        assert len(set(lengths)) == 1

    def test_custom_params_override_defaults(self):
        """Custom params should change predictions."""
        default_preds = decision_predictions_for_eig(["certainty_effect_1"])
        custom_params = {
            "CPT_Agent": {
                "alpha": 0.50,
                "beta": 0.50,
                "lambda_": 1.0,
                "gamma_pos": 0.40,
                "gamma_neg": 0.40,
                "temperature": 1.0,
            },
        }
        custom_preds = decision_predictions_for_eig(
            ["certainty_effect_1"], agent_params=custom_params
        )
        # CPT predictions should differ with different params
        assert not np.allclose(default_preds["CPT_Agent"], custom_preds["CPT_Agent"]), (
            "Custom params didn't change CPT predictions"
        )

    def test_models_disagree_on_diagnostic_gambles(self):
        """On diagnostic gambles, models should predict differently."""
        # Certainty effect: CPT/PH prefer certain, EU may not
        preds = decision_predictions_for_eig(["certainty_effect_1"])
        cpt_p = preds["CPT_Agent"][0]
        eu_p = preds["EU_Agent"][0]
        # These should meaningfully differ
        assert abs(cpt_p - eu_p) > 0.05, (
            f"CPT ({cpt_p:.3f}) and EU ({eu_p:.3f}) too similar on certainty effect"
        )


class TestEIGWithDecisionPredictions:
    """Test that compute_eig() works correctly with decision predictions."""

    def test_eig_nonnegative(self):
        """EIG must be non-negative."""
        posterior = ModelPosterior.uniform(DECISION_AGENTS)
        preds = decision_predictions_for_eig(
            ["certainty_effect_1", "certainty_effect_2"]
        )
        eig = compute_eig(preds, posterior, n_subjects=30, n_sim=100, seed=42)
        assert eig >= 0.0

    def test_eig_positive_for_discriminating_gambles(self):
        """Diagnostic gambles should produce positive EIG."""
        posterior = ModelPosterior.uniform(DECISION_AGENTS)
        # Use multiple gambles where models disagree
        preds = decision_predictions_for_eig(
            [
                "certainty_effect_1",
                "loss_aversion_symmetric",
                "fourfold_low_prob_gain",
                "ph_minimum_decisive",
            ]
        )
        eig = compute_eig(preds, posterior, n_subjects=30, n_sim=200, seed=42)
        assert eig > 0.0, f"EIG should be positive for diagnostic gambles, got {eig}"

    def test_more_gambles_gives_more_information(self):
        """A larger gamble group should generally have higher EIG."""
        posterior = ModelPosterior.uniform(DECISION_AGENTS)
        # Small group
        preds_small = decision_predictions_for_eig(["certainty_effect_1"])
        eig_small = compute_eig(
            preds_small, posterior, n_subjects=30, n_sim=200, seed=42
        )
        # Larger group (superset)
        preds_large = decision_predictions_for_eig(
            [
                "certainty_effect_1",
                "loss_aversion_symmetric",
                "fourfold_low_prob_gain",
                "ph_minimum_decisive",
            ]
        )
        eig_large = compute_eig(
            preds_large, posterior, n_subjects=30, n_sim=200, seed=42
        )
        assert eig_large > eig_small, (
            f"Larger group EIG ({eig_large:.4f}) should exceed "
            f"single gamble ({eig_small:.4f})"
        )


class TestGambleGroups:
    """Test that gamble groups are well-defined."""

    def test_groups_exist(self):
        """GAMBLE_GROUPS should define candidate experiment groups."""
        assert len(GAMBLE_GROUPS) >= 6, "Need at least 6 diagnostic groups"

    def test_groups_have_valid_gamble_names(self):
        """All gamble names in groups must exist in the registry."""
        from antagonistic_collab.models.gamble_structures import FULL_GAMBLE_REGISTRY

        for group_name, gamble_names in GAMBLE_GROUPS.items():
            for gname in gamble_names:
                assert gname in FULL_GAMBLE_REGISTRY, (
                    f"Gamble '{gname}' in group '{group_name}' not in registry"
                )

    def test_groups_are_nonempty(self):
        for group_name, gamble_names in GAMBLE_GROUPS.items():
            assert len(gamble_names) >= 1, f"Group '{group_name}' is empty"


class TestSelectDecisionExperiment:
    """Test EIG-based experiment selection over gamble groups."""

    def test_returns_valid_index(self):
        """Selected index must be within range of candidates."""
        posterior = ModelPosterior.uniform(DECISION_AGENTS)
        candidates = list(GAMBLE_GROUPS.values())
        idx, scores = select_decision_experiment(
            candidates, posterior, n_subjects=30, n_sim=50, seed=42
        )
        assert 0 <= idx < len(candidates)

    def test_returns_eig_scores_for_all_candidates(self):
        posterior = ModelPosterior.uniform(DECISION_AGENTS)
        candidates = list(GAMBLE_GROUPS.values())
        idx, scores = select_decision_experiment(
            candidates, posterior, n_subjects=30, n_sim=50, seed=42
        )
        assert len(scores) == len(candidates)

    def test_all_eig_scores_nonnegative(self):
        posterior = ModelPosterior.uniform(DECISION_AGENTS)
        candidates = list(GAMBLE_GROUPS.values())
        _, scores = select_decision_experiment(
            candidates, posterior, n_subjects=30, n_sim=50, seed=42
        )
        assert all(s >= 0.0 for s in scores), f"Negative EIG scores: {scores}"


class TestUpdateDecisionPosterior:
    """Test Bayesian posterior update from observed choice data."""

    def test_posterior_updates(self):
        """Posterior should change after observing data."""
        posterior = ModelPosterior.uniform(DECISION_AGENTS)
        initial_probs = posterior.probs.copy()

        gamble_names = ["certainty_effect_1", "certainty_effect_2"]
        # Simulate observing CPT-consistent choices (prefer safe option strongly)
        observed = {"certainty_effect_1": 0.85, "certainty_effect_2": 0.80}

        update_decision_posterior(posterior, observed, gamble_names, n_subjects=30)
        assert not np.allclose(posterior.probs, initial_probs), (
            "Posterior didn't change"
        )

    def test_correct_model_gains_probability(self):
        """Data generated from CPT should increase CPT_Agent's posterior."""
        posterior = ModelPosterior.uniform(DECISION_AGENTS)

        # Generate data from CPT ground truth
        from antagonistic_collab.models.decision_runner import (
            generate_synthetic_choices,
        )

        data = generate_synthetic_choices("all_base", "CPT", n_subjects=30, cycle=0)

        gamble_names = list(data["item_accuracies"].keys())
        update_decision_posterior(
            posterior, data["item_accuracies"], gamble_names, n_subjects=30
        )

        cpt_idx = DECISION_AGENTS.index("CPT_Agent")
        assert posterior.probs[cpt_idx] > 1.0 / 3, (
            f"CPT posterior ({posterior.probs[cpt_idx]:.3f}) should exceed prior (0.333)"
        )

    def test_multiple_updates_increase_confidence(self):
        """Repeated observations from same GT should increase confidence monotonically."""
        posterior = ModelPosterior.uniform(DECISION_AGENTS)

        prev_prob = 1.0 / 3
        for cycle in range(3):
            from antagonistic_collab.models.decision_runner import (
                generate_synthetic_choices,
            )

            data = generate_synthetic_choices(
                "all_base", "EU", n_subjects=30, cycle=cycle
            )
            gamble_names = list(data["item_accuracies"].keys())
            update_decision_posterior(
                posterior,
                data["item_accuracies"],
                gamble_names,
                n_subjects=30,
                learning_rate=0.5,
            )
            eu_idx = DECISION_AGENTS.index("EU_Agent")
            current_prob = posterior.probs[eu_idx]
            assert current_prob >= prev_prob - 0.01, (  # small tolerance for noise
                f"EU posterior should not decrease: {prev_prob:.3f} → {current_prob:.3f}"
            )
            prev_prob = current_prob

    def test_returns_posterior(self):
        """Should return the updated posterior for chaining."""
        posterior = ModelPosterior.uniform(DECISION_AGENTS)
        observed = {"certainty_effect_1": 0.7}
        result = update_decision_posterior(
            posterior, observed, ["certainty_effect_1"], n_subjects=30
        )
        assert result is posterior


class TestFullDecisionCycle:
    """Integration test: select → observe → update → converge."""

    @pytest.mark.parametrize("gt_model", ["CPT", "EU", "PH"])
    def test_correct_winner_after_cycles(self, gt_model):
        """After 5 cycles of EIG selection + updates, the correct model should lead."""
        from antagonistic_collab.models.decision_runner import (
            DECISION_EXPECTED_WINNER,
        )

        posterior = ModelPosterior.uniform(DECISION_AGENTS)
        candidates = list(GAMBLE_GROUPS.values())

        for cycle in range(5):
            # Select experiment
            idx, _ = select_decision_experiment(
                candidates,
                posterior,
                n_subjects=30,
                n_sim=100,
                seed=42 + cycle,
                selection_strategy="greedy",
            )

            # Generate data from GT
            selected_gambles = candidates[idx]
            # Generate per-gamble synthetic data
            from antagonistic_collab.models.decision_runner import (
                compute_decision_predictions,
            )

            true_preds = {}
            for gname in selected_gambles:
                p = compute_decision_predictions(gt_model, gname)
                true_preds[gname] = list(p.values())[0]

            rng = np.random.default_rng(42 + cycle * 100)
            observed = {}
            for gname, p_a in true_preds.items():
                p_clipped = np.clip(p_a, 0.01, 0.99)
                n_chose_a = rng.binomial(30, p_clipped)
                observed[gname] = n_chose_a / 30

            # Update posterior
            update_decision_posterior(
                posterior,
                observed,
                selected_gambles,
                n_subjects=30,
                learning_rate=0.5,
            )

        # Check winner
        winner_idx = np.argmax(posterior.probs)
        winner = DECISION_AGENTS[winner_idx]
        expected = DECISION_EXPECTED_WINNER[gt_model]
        assert winner == expected, (
            f"GT={gt_model}: expected {expected}, got {winner}. "
            f"Probs: {dict(zip(DECISION_AGENTS, posterior.probs))}"
        )
