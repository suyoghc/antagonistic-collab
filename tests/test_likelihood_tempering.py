"""Tests for M7: Likelihood Tempering (posterior collapse fix).

Verifies that likelihood tempering (learning_rate / tau parameter) correctly
slows posterior convergence to prevent EIG collapse in synthetic data settings.
"""

import numpy as np
import pytest

from antagonistic_collab.bayesian_selection import (
    ModelPosterior,
    compute_eig,
    update_posterior_from_experiment,
    select_from_pool,
)


class TestLikelihoodTempering:
    """Tests for the learning_rate (tau) parameter across bayesian_selection."""

    # ------------------------------------------------------------------
    # Tests 1-3: ModelPosterior.update() tempering
    # ------------------------------------------------------------------

    def test_tempered_update_slower_than_untempered(self):
        """tau < 1 keeps posterior closer to prior than tau = 1."""
        log_likelihoods = np.array([10.0, -5.0, -3.0])

        # Untempered update
        p_full = ModelPosterior.uniform(["A", "B", "C"])
        p_full.update(log_likelihoods, learning_rate=1.0)

        # Tempered update
        p_tempered = ModelPosterior.uniform(["A", "B", "C"])
        p_tempered.update(log_likelihoods, learning_rate=0.2)

        # Tempered posterior should have higher entropy (closer to uniform)
        assert p_tempered.entropy > p_full.entropy

        # Tempered max probability should be less extreme
        assert np.max(p_tempered.probs) < np.max(p_full.probs)

    def test_tempered_update_preserves_ordering(self):
        """Winner unchanged by tempering — just less confident."""
        log_likelihoods = np.array([10.0, -5.0, -3.0])

        p_full = ModelPosterior.uniform(["A", "B", "C"])
        p_full.update(log_likelihoods, learning_rate=1.0)

        p_tempered = ModelPosterior.uniform(["A", "B", "C"])
        p_tempered.update(log_likelihoods, learning_rate=0.2)

        # Same winner
        assert np.argmax(p_full.probs) == np.argmax(p_tempered.probs)
        # Same ranking
        full_rank = np.argsort(p_full.probs)
        tempered_rank = np.argsort(p_tempered.probs)
        np.testing.assert_array_equal(full_rank, tempered_rank)

    def test_tempering_at_one_matches_standard(self):
        """learning_rate=1.0 reproduces the original (untempered) behavior."""
        log_likelihoods = np.array([5.0, -2.0, 1.0])

        # With explicit learning_rate=1.0
        p_lr = ModelPosterior.uniform(["A", "B", "C"])
        p_lr.update(log_likelihoods, learning_rate=1.0)

        # Without learning_rate (default)
        p_default = ModelPosterior.uniform(["A", "B", "C"])
        p_default.update(log_likelihoods)

        np.testing.assert_array_almost_equal(p_lr.probs, p_default.probs)
        np.testing.assert_array_almost_equal(p_lr.log_probs, p_default.log_probs)

    # ------------------------------------------------------------------
    # Tests 4-5: compute_eig() tempering
    # ------------------------------------------------------------------

    def test_eig_changes_with_learning_rate(self):
        """EIG differs when learning_rate is applied vs not."""
        preds = {
            "A": np.array([0.9, 0.1, 0.5]),
            "B": np.array([0.1, 0.9, 0.5]),
        }
        posterior = ModelPosterior.uniform(["A", "B"])

        eig_full = compute_eig(
            preds, posterior, n_subjects=20, n_sim=100, seed=42, learning_rate=1.0
        )
        eig_tempered = compute_eig(
            preds, posterior, n_subjects=20, n_sim=100, seed=42, learning_rate=0.2
        )

        # Both should be positive
        assert eig_full > 0
        assert eig_tempered > 0
        # They should differ (tempering changes the expected posterior entropy)
        assert eig_full != eig_tempered

    def test_eig_nonzero_after_tempered_update(self):
        """Core property: tempering prevents EIG collapse to zero.

        Without tempering, after 2 strong updates the posterior collapses
        and EIG goes to zero. With tempering, EIG should remain positive.
        """
        preds = {
            "A": np.array([0.95, 0.05, 0.8]),
            "B": np.array([0.05, 0.95, 0.2]),
        }

        # Simulate 2 strong updates (data consistent with model A)
        log_lls = np.array([20.0, -20.0])  # Overwhelming evidence for A

        # Without tempering: posterior collapses
        p_full = ModelPosterior.uniform(["A", "B"])
        p_full.update(log_lls, learning_rate=1.0)
        p_full.update(log_lls, learning_rate=1.0)
        eig_collapsed = compute_eig(
            preds, p_full, n_subjects=20, n_sim=50, seed=42, learning_rate=1.0
        )

        # With tempering: posterior stays spread
        p_tempered = ModelPosterior.uniform(["A", "B"])
        p_tempered.update(log_lls, learning_rate=0.1)
        p_tempered.update(log_lls, learning_rate=0.1)
        eig_tempered = compute_eig(
            preds, p_tempered, n_subjects=20, n_sim=50, seed=42, learning_rate=0.1
        )

        # Tempered EIG should be substantially larger
        assert eig_tempered > eig_collapsed
        # The tempered version should still have meaningful EIG
        assert eig_tempered > 0.001

    # ------------------------------------------------------------------
    # Test 6: update_posterior_from_experiment() records learning_rate
    # ------------------------------------------------------------------

    def test_update_posterior_records_learning_rate(self):
        """History entry includes the learning_rate used."""
        posterior = ModelPosterior.uniform(["GCM_agent", "RULEX_agent"])

        # Need a minimal protocol mock
        class FakeConfig:
            def __init__(self, name):
                self.name = name

        class FakeProtocol:
            agent_configs = [FakeConfig("GCM_agent"), FakeConfig("RULEX_agent")]

            def compute_model_predictions(self, agent_config, struct, cond):
                if agent_config.name == "GCM_agent":
                    return {"item_0": 0.8, "item_1": 0.3}
                return {"item_0": 0.3, "item_1": 0.8}

        data = {"item_accuracies": {"item_0": 0.7, "item_1": 0.4}}

        update_posterior_from_experiment(
            posterior,
            FakeProtocol(),
            data,
            "Type_II",
            "baseline",
            cycle=0,
            learning_rate=0.2,
        )

        assert len(posterior.history) == 1
        assert posterior.history[0]["learning_rate"] == 0.2

    # ------------------------------------------------------------------
    # Test 7: select_from_pool() passes learning_rate
    # ------------------------------------------------------------------

    def test_select_from_pool_passes_learning_rate(self):
        """select_from_pool threads learning_rate to compute_eig.

        We verify by comparing results with different learning rates.
        """

        class FakeConfig:
            def __init__(self, name):
                self.name = name

        class FakeProtocol:
            agent_configs = [FakeConfig("A"), FakeConfig("B")]

            def compute_model_predictions(self, agent_config, struct, cond):
                if agent_config.name == "A":
                    return {"item_0": 0.9, "item_1": 0.1}
                return {"item_0": 0.1, "item_1": 0.9}

        posterior = ModelPosterior.uniform(["A", "B"])
        pool = [("struct1", "cond1")]

        _, eig_full = select_from_pool(
            FakeProtocol(),
            posterior,
            pool,
            n_subjects=20,
            n_sim=50,
            seed=42,
            learning_rate=1.0,
        )
        _, eig_tempered = select_from_pool(
            FakeProtocol(),
            posterior,
            pool,
            n_subjects=20,
            n_sim=50,
            seed=42,
            learning_rate=0.2,
        )

        # EIG scores should differ when learning_rate differs
        assert eig_full[0] != eig_tempered[0]

    # ------------------------------------------------------------------
    # Test 8: CLI --learning-rate flag
    # ------------------------------------------------------------------

    def test_cli_learning_rate_parsed(self):
        """--learning-rate and --no-tempering flags parsed correctly."""
        from antagonistic_collab.__main__ import _build_argparser

        parser = _build_argparser()

        # Default: tempering on at 0.2
        args = parser.parse_args([])
        assert args.learning_rate == 0.2
        assert args.no_tempering is False

        # Custom value
        args = parser.parse_args(["--learning-rate", "0.3"])
        assert args.learning_rate == 0.3

        # --no-tempering flag
        args = parser.parse_args(["--no-tempering"])
        assert args.no_tempering is True

    # ------------------------------------------------------------------
    # Test 9: Input validation
    # ------------------------------------------------------------------

    def test_cli_no_arbiter_parsed(self):
        """--no-arbiter flag is parsed correctly."""
        from antagonistic_collab.__main__ import _build_argparser

        parser = _build_argparser()

        # Default: ARBITER on
        args = parser.parse_args([])
        assert args.no_arbiter is False

        # --no-arbiter flag
        args = parser.parse_args(["--no-arbiter"])
        assert args.no_arbiter is True

    def test_learning_rate_validation(self):
        """Invalid learning_rate values are rejected."""
        posterior = ModelPosterior.uniform(["A", "B"])
        lls = np.array([1.0, -1.0])

        # Zero
        with pytest.raises(ValueError, match="learning_rate"):
            posterior.update(lls, learning_rate=0.0)

        # Negative
        with pytest.raises(ValueError, match="learning_rate"):
            posterior.update(lls, learning_rate=-0.5)

        # Greater than 1
        with pytest.raises(ValueError, match="learning_rate"):
            posterior.update(lls, learning_rate=1.5)


class TestConfig:
    """Tests for config file loading and CLI integration."""

    def test_default_config_loads(self):
        """Built-in default_config.yaml loads successfully."""
        from antagonistic_collab.config import load_config

        config = load_config()
        assert config["learning_rate"] == 0.2
        assert config["no_tempering"] is False
        assert config["no_arbiter"] is False
        assert config["mode"] == "full_pool"
        assert config["selection"] == "bayesian"
        assert config["cycles"] == 1

    def test_user_config_overrides_defaults(self, tmp_path):
        """User config file overrides built-in defaults."""
        from antagonistic_collab.config import load_config

        user_cfg = tmp_path / "config.yaml"
        user_cfg.write_text("learning_rate: 0.5\nno_arbiter: true\ncycles: 10\n")

        config = load_config(str(user_cfg))
        assert config["learning_rate"] == 0.5
        assert config["no_arbiter"] is True
        assert config["cycles"] == 10
        # Non-overridden defaults preserved
        assert config["mode"] == "full_pool"

    def test_apply_config_defaults_to_parser(self):
        """Config values become argparse defaults, CLI flags override."""
        from antagonistic_collab.__main__ import _build_argparser
        from antagonistic_collab.config import apply_config_defaults

        parser = _build_argparser()
        apply_config_defaults(parser, {"learning_rate": 0.3, "cycles": 5})

        # Config defaults applied
        args = parser.parse_args([])
        assert args.learning_rate == 0.3
        assert args.cycles == 5

        # CLI overrides config
        args = parser.parse_args(["--learning-rate", "0.1", "--cycles", "2"])
        assert args.learning_rate == 0.1
        assert args.cycles == 2

    def test_missing_config_file_raises(self):
        """Specifying a nonexistent config file raises FileNotFoundError."""
        from antagonistic_collab.config import load_config

        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")
