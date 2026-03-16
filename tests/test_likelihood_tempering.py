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
    select_experiment,
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

        # Default: tempering on at 0.005
        args = parser.parse_args([])
        assert args.learning_rate == 0.005
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
        assert config["learning_rate"] == 0.005
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


class TestPredictionClipping:
    """Tests for wider prediction clipping in compute_log_likelihood."""

    def test_predictions_clipped_to_0_05_0_95(self):
        """Extreme predictions (near 0 or 1) should be clipped to [0.05, 0.95].

        Previously clipped to [0.01, 0.99], which let near-binary model
        predictions (e.g., SUSTAIN at 0.001/0.999) generate catastrophic
        log-likelihoods that overwhelmed tempering.
        """
        from antagonistic_collab.bayesian_selection import compute_log_likelihood

        observed = np.array([0.5, 0.5])

        # With extreme predictions, the LL should be the same as clip boundary
        ll_extreme = compute_log_likelihood(observed, np.array([0.001, 0.999]), n_subjects=20)
        ll_at_clip = compute_log_likelihood(observed, np.array([0.05, 0.95]), n_subjects=20)

        # If clipping works, these should be equal
        np.testing.assert_almost_equal(
            ll_extreme, ll_at_clip,
            err_msg="Predictions outside [0.05, 0.95] should be clipped",
        )

    def test_posterior_noncollapse_with_default_tau(self):
        """With the calibrated default tau, posterior should not collapse
        after a single experiment on typical synthetic data.

        This is the core integration test for the tempering calibration.
        """
        from antagonistic_collab.debate_protocol import (
            DebateProtocol,
            default_agent_configs,
        )
        from antagonistic_collab.bayesian_selection import (
            ModelPosterior,
            update_posterior_from_experiment,
        )
        from antagonistic_collab.epistemic_state import EpistemicState

        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state=state, agent_configs=agents)

        # Generate synthetic data from GCM
        data = protocol._synthetic_runner(
            design_spec={"structure_name": "Type_II", "condition": "baseline"},
            true_model="GCM",
            cycle=0,
        )

        posterior = ModelPosterior.uniform([a.name for a in agents])
        update_posterior_from_experiment(
            posterior,
            protocol,
            data,
            "Type_II",
            "baseline",
            cycle=0,
            learning_rate=0.005,
        )

        # After 1 experiment with tau=0.005, entropy should remain meaningful
        # (not collapse to 0)
        assert posterior.entropy > 0.1, (
            f"Posterior entropy {posterior.entropy:.4f} collapsed after 1 experiment "
            f"with tau=0.005. Probs: {posterior.probs}"
        )


class TestSelectionStrategy:
    """Tests for configurable experiment selection strategy (greedy vs thompson)."""

    @pytest.fixture
    def protocol_and_pool(self):
        """Protocol with 3 candidates where EIG scores differ meaningfully."""

        class Cfg:
            def __init__(self, name):
                self.name = name

        class Proto:
            agent_configs = [Cfg("A"), Cfg("B")]

            def compute_model_predictions(self, agent_config, struct, cond):
                # Different structures produce different model divergence
                if struct == "high_div":
                    return {"item_0": 0.9, "item_1": 0.1} if agent_config.name == "A" else {"item_0": 0.1, "item_1": 0.9}
                elif struct == "med_div":
                    return {"item_0": 0.7, "item_1": 0.3} if agent_config.name == "A" else {"item_0": 0.3, "item_1": 0.7}
                else:
                    return {"item_0": 0.5, "item_1": 0.5} if agent_config.name == "A" else {"item_0": 0.5, "item_1": 0.5}

        pool = [("high_div", "base"), ("med_div", "base"), ("low_div", "base")]
        posterior = ModelPosterior.uniform(["A", "B"])
        return Proto(), posterior, pool

    # ------------------------------------------------------------------
    # Test 1: greedy always picks argmax
    # ------------------------------------------------------------------

    def test_greedy_picks_argmax(self, protocol_and_pool):
        """strategy='greedy' always selects the highest-EIG candidate."""
        proto, posterior, pool = protocol_and_pool
        idx, scores = select_from_pool(
            proto, posterior, pool,
            n_subjects=20, n_sim=50, seed=42,
            selection_strategy="greedy",
        )
        assert idx == int(np.argmax(scores))

    # ------------------------------------------------------------------
    # Test 2: thompson produces valid index
    # ------------------------------------------------------------------

    def test_thompson_returns_valid_index(self, protocol_and_pool):
        """strategy='thompson' returns a valid index into the pool."""
        proto, posterior, pool = protocol_and_pool
        idx, scores = select_from_pool(
            proto, posterior, pool,
            n_subjects=20, n_sim=50, seed=42,
            selection_strategy="thompson",
        )
        assert 0 <= idx < len(pool)

    # ------------------------------------------------------------------
    # Test 3: thompson explores — not always argmax over many seeds
    # ------------------------------------------------------------------

    def test_thompson_explores_over_seeds(self, protocol_and_pool):
        """Thompson sampling selects different candidates across seeds.

        Over 50 runs with different seeds, Thompson should select at least
        2 distinct candidates (not always the greedy choice).
        """
        proto, posterior, pool = protocol_and_pool
        selected = set()
        for s in range(50):
            idx, _ = select_from_pool(
                proto, posterior, pool,
                n_subjects=20, n_sim=50, seed=1000 + s,
                selection_strategy="thompson",
            )
            selected.add(idx)
        assert len(selected) >= 2, (
            f"Thompson sampling selected only {selected} across 50 seeds — no exploration"
        )

    # ------------------------------------------------------------------
    # Test 4: thompson favors high-EIG candidates
    # ------------------------------------------------------------------

    def test_thompson_favors_high_eig(self, protocol_and_pool):
        """Thompson sampling should select high-EIG candidates more often."""
        proto, posterior, pool = protocol_and_pool
        counts = {0: 0, 1: 0, 2: 0}
        for s in range(200):
            idx, _ = select_from_pool(
                proto, posterior, pool,
                n_subjects=20, n_sim=50, seed=2000 + s,
                selection_strategy="thompson",
            )
            counts[idx] += 1
        # high_div (idx 0) should be selected more often than low_div (idx 2)
        assert counts[0] > counts[2], (
            f"High-EIG candidate not favored: {counts}"
        )

    # ------------------------------------------------------------------
    # Test 5: greedy is deterministic across seeds
    # ------------------------------------------------------------------

    def test_greedy_deterministic(self, protocol_and_pool):
        """Greedy selection returns the same result regardless of seed."""
        proto, posterior, pool = protocol_and_pool
        results = set()
        for s in range(10):
            idx, _ = select_from_pool(
                proto, posterior, pool,
                n_subjects=20, n_sim=50, seed=42,
                selection_strategy="greedy",
            )
            results.add(idx)
        assert len(results) == 1

    # ------------------------------------------------------------------
    # Test 6: select_experiment also supports strategy
    # ------------------------------------------------------------------

    def test_select_experiment_supports_strategy(self):
        """select_experiment() accepts and uses selection_strategy."""
        from dataclasses import dataclass

        @dataclass
        class FakeProposal:
            design_spec: dict

        class Cfg:
            def __init__(self, name):
                self.name = name

        class Proto:
            agent_configs = [Cfg("A"), Cfg("B")]

            def compute_model_predictions(self, agent_config, struct, cond):
                if agent_config.name == "A":
                    return {"item_0": 0.9, "item_1": 0.1}
                return {"item_0": 0.1, "item_1": 0.9}

        candidates = [
            FakeProposal({"structure_name": "Type_I", "condition": "baseline"}),
            FakeProposal({"structure_name": "Type_II", "condition": "baseline"}),
        ]
        posterior = ModelPosterior.uniform(["A", "B"])

        # Should not raise
        idx, scores = select_experiment(
            Proto(), posterior, candidates,
            n_subjects=20, n_sim=50, seed=42,
            selection_strategy="thompson",
        )
        assert 0 <= idx < len(candidates)

    # ------------------------------------------------------------------
    # Test 7: invalid strategy raises ValueError
    # ------------------------------------------------------------------

    def test_invalid_strategy_raises(self, protocol_and_pool):
        """Unknown selection_strategy raises ValueError."""
        proto, posterior, pool = protocol_and_pool
        with pytest.raises(ValueError, match="selection_strategy"):
            select_from_pool(
                proto, posterior, pool,
                n_subjects=20, n_sim=50, seed=42,
                selection_strategy="unknown",
            )

    # ------------------------------------------------------------------
    # Test 8: thompson with all-zero EIG falls back to uniform
    # ------------------------------------------------------------------

    def test_thompson_uniform_when_all_eig_zero(self):
        """When all EIG scores are zero, Thompson samples uniformly."""

        class Cfg:
            def __init__(self, name):
                self.name = name

        class Proto:
            agent_configs = [Cfg("A"), Cfg("B")]

            def compute_model_predictions(self, agent_config, struct, cond):
                # Identical predictions → zero EIG
                return {"item_0": 0.5, "item_1": 0.5}

        pool = [("s1", "c1"), ("s2", "c2"), ("s3", "c3")]
        posterior = ModelPosterior.uniform(["A", "B"])

        selected = set()
        for s in range(100):
            idx, _ = select_from_pool(
                Proto(), posterior, pool,
                n_subjects=20, n_sim=50, seed=3000 + s,
                selection_strategy="thompson",
            )
            selected.add(idx)
        # With uniform fallback, should see multiple candidates
        assert len(selected) >= 2, (
            f"Zero-EIG Thompson didn't explore: selected only {selected}"
        )

    # ------------------------------------------------------------------
    # Test 9: config and CLI support selection_strategy
    # ------------------------------------------------------------------

    def test_config_selection_strategy(self):
        """default_config.yaml includes selection_strategy."""
        from antagonistic_collab.config import load_config

        config = load_config()
        assert "selection_strategy" in config
        assert config["selection_strategy"] in ("greedy", "thompson")

    def test_cli_selection_strategy_parsed(self):
        """--selection-strategy CLI flag is parsed correctly."""
        from antagonistic_collab.__main__ import _build_argparser

        parser = _build_argparser()

        # Default should be thompson
        args = parser.parse_args([])
        assert args.selection_strategy == "thompson"

        # Override to greedy
        args = parser.parse_args(["--selection-strategy", "greedy"])
        assert args.selection_strategy == "greedy"


class TestCruxDirectedThompson:
    """Tests for crux-directed mixture distribution in Thompson sampling.

    When active cruxes exist, _select_index should sample from a mixture:
    with probability crux_weight, pick uniformly from crux-matching candidates;
    otherwise sample from standard EIG-weighted Thompson.
    """

    def test_crux_weight_one_always_picks_crux(self):
        """With crux_weight=1.0, always selects from crux_indices."""
        from antagonistic_collab.bayesian_selection import _select_index

        scores = [0.1, 0.5, 0.3, 0.2]
        for seed in range(20):
            idx = _select_index(
                scores, "thompson", seed=seed,
                crux_indices=[2], crux_weight=1.0,
            )
            assert idx == 2

    def test_crux_weight_zero_ignores_cruxes(self):
        """With crux_weight=0.0, crux_indices have no effect."""
        from antagonistic_collab.bayesian_selection import _select_index

        scores = [0.1, 0.5, 0.3, 0.2]
        for seed in range(20):
            without = _select_index(scores, "thompson", seed=seed)
            with_crux = _select_index(
                scores, "thompson", seed=seed,
                crux_indices=[2], crux_weight=0.0,
            )
            assert without == with_crux

    def test_crux_weight_increases_crux_frequency(self):
        """crux_weight=0.5 should select crux candidates more often."""
        from antagonistic_collab.bayesian_selection import _select_index

        scores = [0.2, 0.2, 0.2, 0.2, 0.2]  # uniform EIG
        n_trials = 500

        count_without = sum(
            1 for s in range(n_trials)
            if _select_index(scores, "thompson", seed=s) == 0
        )
        count_with = sum(
            1 for s in range(n_trials)
            if _select_index(
                scores, "thompson", seed=s,
                crux_indices=[0], crux_weight=0.5,
            ) == 0
        )
        # Without cruxes: ~20% (1/5). With crux_weight=0.5: ~60% (0.5*1 + 0.5*0.2)
        assert count_with > count_without * 1.5, (
            f"Crux direction should increase crux candidate frequency: "
            f"{count_with} vs {count_without}"
        )

    def test_crux_weight_greedy_unchanged(self):
        """Crux direction only applies to Thompson; greedy still picks argmax."""
        from antagonistic_collab.bayesian_selection import _select_index

        scores = [0.1, 0.5, 0.3, 0.2]
        idx = _select_index(
            scores, "greedy",
            crux_indices=[0], crux_weight=1.0,
        )
        assert idx == 1  # argmax

    def test_crux_weight_empty_indices_unchanged(self):
        """Empty crux_indices behaves like no crux direction."""
        from antagonistic_collab.bayesian_selection import _select_index

        scores = [0.1, 0.5, 0.3, 0.2]
        for seed in range(20):
            without = _select_index(scores, "thompson", seed=seed)
            with_empty = _select_index(
                scores, "thompson", seed=seed,
                crux_indices=[], crux_weight=0.5,
            )
            assert without == with_empty

    def test_crux_weight_validation(self):
        """crux_weight outside [0, 1] should raise ValueError."""
        from antagonistic_collab.bayesian_selection import _select_index

        with pytest.raises(ValueError, match="crux_weight"):
            _select_index([0.1], "thompson", crux_weight=-0.1)
        with pytest.raises(ValueError, match="crux_weight"):
            _select_index([0.1], "thompson", crux_weight=1.5)

    def test_crux_weight_multiple_crux_indices(self):
        """With multiple crux_indices, samples uniformly among them."""
        from antagonistic_collab.bayesian_selection import _select_index

        scores = [0.01, 0.01, 0.01, 0.01]
        # With crux_weight=1.0 and crux_indices=[1,3], should only pick 1 or 3
        selected = set()
        for seed in range(50):
            idx = _select_index(
                scores, "thompson", seed=seed,
                crux_indices=[1, 3], crux_weight=1.0,
            )
            selected.add(idx)
        assert selected == {1, 3}

    def test_select_from_pool_crux_weight(self):
        """select_from_pool threads crux_weight to _select_index."""
        from antagonistic_collab.bayesian_selection import (
            select_from_pool,
            generate_full_candidate_pool,
        )
        from antagonistic_collab.debate_protocol import DebateProtocol, default_agent_configs
        from antagonistic_collab.epistemic_state import EpistemicState

        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state=state, agent_configs=agents)
        posterior = ModelPosterior.uniform([a.name for a in agents])
        pool = generate_full_candidate_pool(protocol)[:5]

        # With crux_weight=1.0 matching pool[0], should always pick index 0
        boost_specs = [{"structure": pool[0][0], "condition": pool[0][1], "boost": 2.0}]
        selections = set()
        for seed in range(10):
            idx, _ = select_from_pool(
                protocol, posterior, pool, n_sim=50, seed=seed,
                crux_boost_specs=boost_specs, crux_weight=1.0,
                selection_strategy="thompson",
            )
            selections.add(idx)
        assert selections == {0}, (
            f"crux_weight=1.0 should always select matching candidate, got {selections}"
        )

    def test_select_from_pool_crux_weight_default_zero(self):
        """Default crux_weight is 0.0, preserving backward compatibility."""
        from antagonistic_collab.bayesian_selection import select_from_pool
        import inspect

        sig = inspect.signature(select_from_pool)
        assert sig.parameters["crux_weight"].default == 0.0

    def test_config_crux_weight(self):
        """default_config.yaml includes crux_weight."""
        from antagonistic_collab.config import load_config

        config = load_config()
        assert "crux_weight" in config

    def test_cli_crux_weight_parsed(self):
        """--crux-weight CLI flag is parsed correctly."""
        from antagonistic_collab.__main__ import _build_argparser

        parser = _build_argparser()
        args = parser.parse_args(["--crux-weight", "0.5"])
        assert args.crux_weight == 0.5
