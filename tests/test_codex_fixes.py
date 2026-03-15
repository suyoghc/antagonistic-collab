"""Tests for Codex review fixes: ground-truth leakage, LOO mismatch,
RULEX curves, novel structure resolution, and n_subjects threading.
"""

import numpy as np

from antagonistic_collab.debate_protocol import (
    DebateProtocol,
    STRUCTURE_REGISTRY,
)
from antagonistic_collab.epistemic_state import EpistemicState


def _make_protocol():
    """Helper: create a minimal protocol with default agents."""
    from antagonistic_collab.debate_protocol import default_agent_configs

    state = EpistemicState(domain="test")
    agents = default_agent_configs()
    return DebateProtocol(state=state, agent_configs=agents)


# =========================================================================
# Fix #1: Curve evidence must not leak ground-truth model
# =========================================================================


class TestCurveEvidenceNoLeakage:
    """update_posterior_from_experiment must not use ground_truth_model."""

    def test_curve_evidence_does_not_use_ground_truth_model(self):
        """Posterior update should not read data['ground_truth_model']
        to select a reference curve. That leaks the answer key."""
        from antagonistic_collab.bayesian_selection import (
            ModelPosterior,
            update_posterior_from_experiment,
        )

        protocol = _make_protocol()
        posterior = ModelPosterior.uniform([a.name for a in protocol.agent_configs])

        # Type_II has 8 items — match the structure
        data = {
            "item_accuracies": {f"item_{i}": 0.5 + 0.05 * i for i in range(8)},
            "ground_truth_model": "GCM",
            "n_subjects": 20,
        }

        # Fake learning curves — different per agent so gt selection matters
        agents = protocol.agent_configs
        fake_curves = {
            agents[0].name: [
                {"accuracy": 0.9, "block": 0},
                {"accuracy": 0.95, "block": 1},
            ],
            agents[1].name: [
                {"accuracy": 0.4, "block": 0},
                {"accuracy": 0.5, "block": 1},
            ],
            agents[2].name: [
                {"accuracy": 0.6, "block": 0},
                {"accuracy": 0.7, "block": 1},
            ],
        }

        # Update with curves — should NOT use ground_truth_model
        update_posterior_from_experiment(
            posterior,
            protocol,
            data,
            "Type_II",
            "baseline",
            cycle=0,
            learning_curves=fake_curves,
        )

        # Change ground_truth_model and re-run — result should be identical
        # if the code doesn't read it
        posterior2 = ModelPosterior.uniform([a.name for a in protocol.agent_configs])
        data2 = dict(data)
        data2["ground_truth_model"] = "RULEX"

        update_posterior_from_experiment(
            posterior2,
            protocol,
            data2,
            "Type_II",
            "baseline",
            cycle=0,
            learning_curves=fake_curves,
        )

        # Compare log_probs (not probs — probs saturate to ~1.0 hiding differences)
        np.testing.assert_array_almost_equal(
            posterior.log_probs,
            posterior2.log_probs,
            err_msg="Posterior log_probs should not depend on ground_truth_model",
        )


# =========================================================================
# Fix #2: Novel structures must resolve in compute_model_predictions
# =========================================================================


class TestNovelStructureResolution:
    """Novel structures in temporary_structures must not fall back to Type_II."""

    def test_novel_structure_used_in_predictions(self):
        """compute_model_predictions should use temporary_structures, not
        silently fall back to Type_II."""
        protocol = _make_protocol()

        # Register a novel structure with very different geometry
        novel = {
            "stimuli": [[0, 0], [0, 1], [1, 0], [1, 1]],
            "labels": [0, 1, 1, 0],  # XOR — very different from Type_II
        }
        protocol.temporary_structures["xor_test"] = novel

        agent = protocol.agent_configs[0]

        # Predictions on the novel structure
        preds_novel = protocol.compute_model_predictions(agent, "xor_test", "baseline")
        # Predictions on Type_II
        preds_type2 = protocol.compute_model_predictions(agent, "Type_II", "baseline")

        # They must differ (different structures have different items)
        # Novel has 4 items, Type_II has more
        novel_items = [k for k in preds_novel if k.startswith("item_")]
        type2_items = [k for k in preds_type2 if k.startswith("item_")]
        assert len(novel_items) == 4, (
            f"Novel structure should have 4 items, got {len(novel_items)}. "
            "Likely fell back to Type_II."
        )
        assert len(novel_items) != len(type2_items), (
            "Novel predictions should differ from Type_II"
        )


# =========================================================================
# Fix #3: Synthetic data must use LOO predictions (matching scoring)
# =========================================================================


class TestSyntheticDataLOO:
    """_synthetic_runner must use LOO predictions to match scoring path."""

    def test_synthetic_predictions_use_loo(self):
        """Synthetic data should use LOO — predict item i without item i
        in the training set. The model_predictions field should reflect this."""
        protocol = _make_protocol()

        data = protocol._synthetic_runner(
            design_spec={"structure_name": "five_four", "condition": "baseline"},
            true_model="GCM",
            cycle=0,
        )

        # Get LOO predictions via compute_model_predictions (the scoring path)
        from antagonistic_collab.debate_protocol import default_agent_configs

        gcm_agent = None
        for a in default_agent_configs():
            if "GCM" in a.model_class.name:
                gcm_agent = a
                break

        scoring_preds = protocol.compute_model_predictions(
            gcm_agent, "five_four", "baseline"
        )

        # The synthetic runner's model_predictions and the scoring path
        # should produce comparable values (both LOO).
        # Previously, synthetic used full-set predictions (higher) while
        # scoring used LOO (lower).
        synth_items = data["model_predictions"]
        for item_key in synth_items:
            idx = int(item_key.split("_")[1])
            correct_label = int(
                np.asarray(STRUCTURE_REGISTRY["five_four"]["labels"])[idx]
            )
            synth_p = synth_items[item_key].get(correct_label, 0.5)
            scoring_p = scoring_preds.get(item_key, 0.5)
            # With LOO in both paths, the gap should be small
            # (params differ slightly so exact match isn't expected)
            assert abs(synth_p - scoring_p) < 0.35, (
                f"{item_key}: synthetic P(correct)={synth_p:.3f} vs "
                f"scoring P(correct)={scoring_p:.3f} — gap too large, "
                f"likely not using LOO in synthetic runner"
            )


# =========================================================================
# Fix #4: RULEX learning curves must include exception retrieval
# =========================================================================


class TestRULEXCurveExceptions:
    """RULEX predict_learning_curve must use predict(), not _evaluate_rule()."""

    def test_rulex_curve_uses_predict(self):
        """On rule_plus_exception structures, RULEX's curve should reflect
        exception retrieval (via predict()), not just rule evaluation."""
        from antagonistic_collab.models.rulex import RULEX

        model = RULEX()
        struct = STRUCTURE_REGISTRY["rule_plus_exception_1exc"]
        stimuli = np.asarray(struct["stimuli"])
        labels = np.asarray(struct["labels"])

        training_seq = [
            (np.asarray(s), int(lab))
            for s, lab in zip(stimuli, labels)
            for _ in range(3)  # 3 epochs
        ]

        curve = model.predict_learning_curve(
            training_seq,
            stimuli,
            labels,
            block_size=len(stimuli),
            p_single=0.5,
            p_conj=0.3,
            error_tolerance=0.1,
            seed=42,
            p_exception=0.8,
        )

        # With exception retrieval, accuracy on the exception item should
        # be higher than pure rule evaluation. The curve's final accuracy
        # should be > what _evaluate_rule alone would give.
        # At minimum, the curve code should accept p_exception without error.
        assert len(curve) >= 1
        final_acc = curve[-1]["accuracy"]
        # With exceptions, accuracy should be meaningfully above 0.5
        assert final_acc > 0.5, (
            f"RULEX curve final accuracy {final_acc} suggests exceptions "
            "are not being used"
        )


# =========================================================================
# Fix #5: Posterior update must use actual n_subjects from data
# =========================================================================


class TestNSubjectsThreading:
    """update_posterior_from_experiment should use data's n_subjects."""

    def test_n_subjects_from_data_used(self):
        """Posterior update should use n_subjects from the data dict,
        not always fall back to the default of 20."""
        from antagonistic_collab.bayesian_selection import (
            ModelPosterior,
            update_posterior_from_experiment,
        )

        protocol = _make_protocol()

        # Type_II has 8 items — match the structure
        accs = {f"item_{i}": 0.5 + 0.05 * i for i in range(8)}

        data = {
            "item_accuracies": dict(accs),
            "n_subjects": 100,  # Much higher than default 20
        }

        # Update with n_subjects=100
        p100 = ModelPosterior.uniform([a.name for a in protocol.agent_configs])
        update_posterior_from_experiment(
            p100,
            protocol,
            data,
            "Type_II",
            "baseline",
            cycle=0,
        )

        # Update with default (should use 20 if we DON'T pass n_subjects)
        p20 = ModelPosterior.uniform([a.name for a in protocol.agent_configs])
        data_no_n = {
            "item_accuracies": dict(accs),
        }
        update_posterior_from_experiment(
            p20,
            protocol,
            data_no_n,
            "Type_II",
            "baseline",
            cycle=0,
        )

        # Higher n_subjects → stronger evidence → more extreme posterior
        # (further from uniform)
        assert p100.entropy < p20.entropy, (
            "n_subjects=100 should produce more extreme posterior than default"
        )
