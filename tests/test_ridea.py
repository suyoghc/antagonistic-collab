"""Tests for R-IDeA scoring (standalone, separate from main codebase).

R-IDeA = Representativeness + Informativeness + De-amplification
(Tang, Sloman & Kaski, 2025)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from antagonistic_collab.bayesian_selection import ModelPosterior
from antagonistic_collab.ridea import (
    compute_representativeness,
    compute_deamplification,
    compute_ridea_scores,
    select_ridea,
)


# ── Fixtures ──


def _uniform_posterior():
    return ModelPosterior.uniform(["GCM", "SUSTAIN", "RULEX"])


def _skewed_posterior():
    """Posterior heavily favoring GCM."""
    p = ModelPosterior.uniform(["GCM", "SUSTAIN", "RULEX"])
    p.log_probs = np.array([0.0, -5.0, -5.0])
    return p


def _make_predictions_similar():
    """All models predict similarly — low informativeness."""
    return {
        "GCM": np.array([0.7, 0.7, 0.3, 0.3]),
        "SUSTAIN": np.array([0.65, 0.68, 0.32, 0.35]),
        "RULEX": np.array([0.72, 0.69, 0.28, 0.31]),
    }


def _make_predictions_divergent():
    """Models disagree strongly — high informativeness."""
    return {
        "GCM": np.array([0.9, 0.9, 0.1, 0.1]),
        "SUSTAIN": np.array([0.1, 0.5, 0.5, 0.9]),
        "RULEX": np.array([0.5, 0.1, 0.9, 0.5]),
    }


def _make_predictions_gcm_biased():
    """Experiment that mainly distinguishes GCM from the rest."""
    return {
        "GCM": np.array([0.9, 0.9, 0.9, 0.9]),
        "SUSTAIN": np.array([0.5, 0.5, 0.5, 0.5]),
        "RULEX": np.array([0.5, 0.5, 0.5, 0.5]),
    }


def _make_predictions_balanced():
    """Each pair of models disagrees on different items."""
    return {
        "GCM": np.array([0.9, 0.5, 0.1, 0.5]),
        "SUSTAIN": np.array([0.5, 0.9, 0.5, 0.1]),
        "RULEX": np.array([0.1, 0.5, 0.9, 0.5]),
    }


# ── Representativeness Tests ──


class TestRepresentativeness:
    def test_first_experiment_is_maximally_representative(self):
        """No prior experiments → representativeness should be maximal (1.0)."""
        preds = _make_predictions_divergent()
        rep = compute_representativeness(preds, previous_predictions=[])
        assert rep == 1.0

    def test_identical_experiment_is_minimally_representative(self):
        """Same predictions as a prior experiment → low representativeness."""
        preds = _make_predictions_divergent()
        rep = compute_representativeness(preds, previous_predictions=[preds])
        assert rep == 0.0

    def test_different_experiment_has_higher_representativeness(self):
        """Novel predictions → higher representativeness than repeated ones."""
        prev = [_make_predictions_similar()]
        rep_novel = compute_representativeness(_make_predictions_divergent(), prev)
        rep_repeat = compute_representativeness(_make_predictions_similar(), prev)
        assert rep_novel > rep_repeat

    def test_representativeness_is_non_negative(self):
        preds = _make_predictions_similar()
        prev = [_make_predictions_divergent(), _make_predictions_balanced()]
        rep = compute_representativeness(preds, prev)
        assert rep >= 0.0


# ── De-amplification Tests ──


class TestDeamplification:
    def test_balanced_experiment_has_higher_deamplification(self):
        """Experiment informative for all models > experiment biased toward one."""
        posterior = _uniform_posterior()
        deamp_balanced = compute_deamplification(
            _make_predictions_balanced(), posterior, n_sim=100, seed=42
        )
        deamp_biased = compute_deamplification(
            _make_predictions_gcm_biased(), posterior, n_sim=100, seed=42
        )
        assert deamp_balanced > deamp_biased

    def test_deamplification_is_non_negative(self):
        posterior = _uniform_posterior()
        deamp = compute_deamplification(
            _make_predictions_divergent(), posterior, n_sim=100, seed=42
        )
        assert deamp >= 0.0

    def test_similar_predictions_have_low_deamplification(self):
        """If models agree, no experiment can be informative for any of them."""
        posterior = _uniform_posterior()
        deamp = compute_deamplification(
            _make_predictions_similar(), posterior, n_sim=100, seed=42
        )
        # Low but not necessarily zero (Monte Carlo noise)
        assert deamp < 0.5


# ── R-IDeA Combined Scoring Tests ──


class TestRIDeAScoring:
    def test_ridea_scores_are_non_negative(self):
        posterior = _uniform_posterior()
        pool_preds = [
            _make_predictions_divergent(),
            _make_predictions_similar(),
            _make_predictions_balanced(),
        ]
        scores = compute_ridea_scores(
            pool_preds, posterior, previous_predictions=[], n_sim=50, seed=42
        )
        assert all(s >= 0 for s in scores)

    def test_ridea_returns_correct_number_of_scores(self):
        posterior = _uniform_posterior()
        pool_preds = [
            _make_predictions_divergent(),
            _make_predictions_similar(),
            _make_predictions_balanced(),
        ]
        scores = compute_ridea_scores(
            pool_preds, posterior, previous_predictions=[], n_sim=50, seed=42
        )
        assert len(scores) == 3

    def test_pure_eig_mode_matches_eig(self):
        """With alpha=0, beta=0, R-IDeA should rank same as EIG."""
        posterior = _uniform_posterior()
        pool_preds = [
            _make_predictions_divergent(),
            _make_predictions_similar(),
        ]
        # Pure informativeness
        scores = compute_ridea_scores(
            pool_preds,
            posterior,
            previous_predictions=[],
            alpha=0.0,
            beta=0.0,
            n_sim=100,
            seed=42,
        )
        # Divergent should score higher than similar
        assert scores[0] > scores[1]

    def test_representativeness_prefers_novel_over_repeated(self):
        """After running a similar experiment, representativeness should
        prefer a novel candidate over a repeat."""
        posterior = _uniform_posterior()
        pool_preds = [
            _make_predictions_similar(),  # close to previous → low rep
            _make_predictions_balanced(),  # different from previous → high rep
        ]
        prev = [_make_predictions_similar()]
        scores = compute_ridea_scores(
            pool_preds,
            posterior,
            previous_predictions=prev,
            alpha=0.5,
            beta=0.0,
            n_sim=50,
            seed=42,
        )
        # Balanced (novel) should outscore similar (repeat)
        assert scores[1] > scores[0]

    def test_deamplification_penalizes_biased_experiment(self):
        """With beta > 0, a balanced experiment should beat a biased one
        even if the biased one has higher EIG."""
        posterior = _uniform_posterior()
        pool_preds = [
            _make_predictions_gcm_biased(),  # high EIG for GCM pair, biased
            _make_predictions_balanced(),  # balanced across all pairs
        ]
        # With strong de-amplification weight
        scores = compute_ridea_scores(
            pool_preds,
            posterior,
            previous_predictions=[],
            alpha=0.0,
            beta=0.5,
            n_sim=100,
            seed=42,
        )
        # Balanced should score higher due to de-amplification
        assert scores[1] > scores[0]


# ── Selection Tests ──


class TestRIDeASelection:
    def test_select_returns_valid_index(self):
        posterior = _uniform_posterior()
        pool_preds = [
            _make_predictions_divergent(),
            _make_predictions_similar(),
            _make_predictions_balanced(),
        ]
        idx, scores = select_ridea(
            pool_preds,
            posterior,
            previous_predictions=[],
            strategy="greedy",
            n_sim=50,
            seed=42,
        )
        assert 0 <= idx < 3
        assert len(scores) == 3

    def test_greedy_selects_highest_score(self):
        posterior = _uniform_posterior()
        pool_preds = [
            _make_predictions_similar(),  # low score
            _make_predictions_divergent(),  # high score
            _make_predictions_similar(),  # low score
        ]
        idx, scores = select_ridea(
            pool_preds,
            posterior,
            previous_predictions=[],
            strategy="greedy",
            n_sim=50,
            seed=42,
        )
        assert idx == np.argmax(scores)

    def test_thompson_returns_valid_index(self):
        posterior = _uniform_posterior()
        pool_preds = [
            _make_predictions_divergent(),
            _make_predictions_similar(),
            _make_predictions_balanced(),
        ]
        idx, scores = select_ridea(
            pool_preds,
            posterior,
            previous_predictions=[],
            strategy="thompson",
            n_sim=50,
            seed=42,
        )
        assert 0 <= idx < 3
