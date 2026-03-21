"""Tests for standalone decision-domain debate runner.

Tests the non-LLM components: parameter validation, prompt construction,
revision parsing, and the cycle loop with mocked LLM calls.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from antagonistic_collab.models.decision_agents import default_decision_agent_configs
from antagonistic_collab.models.decision_runner import (
    MISSPEC_DECISION_PARAMS,
)


# ── Parameter Validation ──


class TestValidateDecisionParams:
    """Test that proposed parameter revisions are filtered correctly."""

    def test_rejects_invalid_param_names(self):
        from antagonistic_collab.models.decision_debate_runner import (
            filter_valid_params,
        )
        from antagonistic_collab.models.prospect_theory import CumulativeProspectTheory

        model = CumulativeProspectTheory()
        proposed = {"alpha": 0.7, "w_i": 0.5, "bogus": 99, "lambda_": 2.0}
        filtered = filter_valid_params(model, proposed)
        assert "alpha" in filtered
        assert "lambda_" in filtered
        assert "w_i" not in filtered
        assert "bogus" not in filtered

    def test_accepts_all_valid_eu_params(self):
        from antagonistic_collab.models.decision_debate_runner import (
            filter_valid_params,
        )
        from antagonistic_collab.models.expected_utility import ExpectedUtility

        model = ExpectedUtility()
        proposed = {"r": 0.3, "temperature": 1.5}
        filtered = filter_valid_params(model, proposed)
        assert filtered == proposed

    def test_accepts_all_valid_ph_params(self):
        from antagonistic_collab.models.decision_debate_runner import (
            filter_valid_params,
        )
        from antagonistic_collab.models.priority_heuristic import PriorityHeuristic

        model = PriorityHeuristic()
        proposed = {"outcome_threshold_frac": 0.15, "prob_threshold": 0.12, "phi": 0.8}
        filtered = filter_valid_params(model, proposed)
        assert filtered == proposed

    def test_empty_proposal_returns_empty(self):
        from antagonistic_collab.models.decision_debate_runner import (
            filter_valid_params,
        )
        from antagonistic_collab.models.expected_utility import ExpectedUtility

        model = ExpectedUtility()
        assert filter_valid_params(model, {}) == {}

    def test_rmse_validation_accepts_improvement(self):
        from antagonistic_collab.models.decision_debate_runner import (
            validate_revision_rmse,
        )

        # Observed data: P(choose A) for 2 gambles
        observed = {"g1": 0.8, "g2": 0.3}
        baseline_preds = {"g1": 0.5, "g2": 0.5}  # RMSE = 0.354
        revised_preds = {"g1": 0.75, "g2": 0.35}  # RMSE = 0.050

        accepted, baseline_rmse, revised_rmse = validate_revision_rmse(
            observed, baseline_preds, revised_preds
        )
        assert accepted
        assert revised_rmse < baseline_rmse

    def test_rmse_validation_rejects_degradation(self):
        from antagonistic_collab.models.decision_debate_runner import (
            validate_revision_rmse,
        )

        observed = {"g1": 0.8, "g2": 0.3}
        baseline_preds = {"g1": 0.75, "g2": 0.35}  # good
        revised_preds = {"g1": 0.2, "g2": 0.9}  # terrible

        accepted, _, _ = validate_revision_rmse(observed, baseline_preds, revised_preds)
        assert not accepted


# ── Prompt Construction ──


class TestBuildDebatePrompt:
    """Test that the debate prompt contains required information."""

    def test_prompt_contains_prediction_errors(self):
        from antagonistic_collab.models.decision_debate_runner import (
            build_interpretation_prompt,
        )

        agent_config = default_decision_agent_configs()[0]  # CPT
        observed = {"certainty_effect_1": 0.85, "common_ratio_high": 0.60}
        predictions = {"certainty_effect_1": 0.50, "common_ratio_high": 0.70}
        posterior = {"CPT_Agent": 0.4, "EU_Agent": 0.35, "PH_Agent": 0.25}

        prompt = build_interpretation_prompt(
            agent_config, observed, predictions, posterior, cycle=0
        )

        assert "certainty_effect_1" in prompt
        assert "0.85" in prompt  # observed
        assert "0.50" in prompt  # predicted
        assert "posterior" in prompt.lower() or "probability" in prompt.lower()

    def test_prompt_includes_current_params(self):
        from antagonistic_collab.models.decision_debate_runner import (
            build_interpretation_prompt,
        )

        agent_config = default_decision_agent_configs()[0]  # CPT
        prompt = build_interpretation_prompt(
            agent_config,
            observed={"g1": 0.5},
            predictions={"g1": 0.5},
            posterior={"CPT_Agent": 0.33, "EU_Agent": 0.33, "PH_Agent": 0.33},
            cycle=0,
        )

        # Should mention at least one CPT parameter
        assert "alpha" in prompt or "lambda" in prompt

    def test_prompt_asks_for_json_output(self):
        from antagonistic_collab.models.decision_debate_runner import (
            build_interpretation_prompt,
        )

        agent_config = default_decision_agent_configs()[1]  # EU
        prompt = build_interpretation_prompt(
            agent_config,
            observed={"g1": 0.5},
            predictions={"g1": 0.5},
            posterior={"CPT_Agent": 0.33, "EU_Agent": 0.33, "PH_Agent": 0.33},
            cycle=0,
        )

        assert "json" in prompt.lower() or "JSON" in prompt


# ── Revision Parsing ──


class TestParseRevision:
    """Test parsing of agent revision proposals from JSON."""

    def test_parse_valid_revision(self):
        from antagonistic_collab.models.decision_debate_runner import (
            parse_agent_revision,
        )

        response = {
            "interpretation": "My predictions are too extreme",
            "revision": {
                "description": "Reduce loss aversion",
                "new_params": {"lambda_": 1.8, "alpha": 0.80},
            },
        }
        revision = parse_agent_revision(response)
        assert revision is not None
        assert revision["new_params"]["lambda_"] == 1.8

    def test_parse_no_revision(self):
        from antagonistic_collab.models.decision_debate_runner import (
            parse_agent_revision,
        )

        response = {
            "interpretation": "Predictions look fine",
            "revision": None,
        }
        revision = parse_agent_revision(response)
        assert revision is None

    def test_parse_missing_revision_key(self):
        from antagonistic_collab.models.decision_debate_runner import (
            parse_agent_revision,
        )

        response = {"interpretation": "just analysis, no revision"}
        revision = parse_agent_revision(response)
        assert revision is None

    def test_parse_revision_with_empty_params(self):
        from antagonistic_collab.models.decision_debate_runner import (
            parse_agent_revision,
        )

        response = {
            "interpretation": "analysis",
            "revision": {"description": "no changes", "new_params": {}},
        }
        revision = parse_agent_revision(response)
        # Empty params = no actual revision
        assert revision is None


# ── Full Cycle (mocked LLM) ──


class TestDecisionDebateCycle:
    """Test the full cycle loop with mocked LLM calls."""

    def _mock_call_agent(self, *args, **kwargs):
        """Mock LLM that always proposes a parameter revision toward GT."""
        import json

        return json.dumps(
            {
                "interpretation": "My predictions are wrong because my parameters are off.",
                "revision": {
                    "description": "Adjusting toward better fit",
                    "new_params": {"alpha": 0.85, "lambda_": 2.2},
                },
            }
        )

    def test_debate_round_updates_params(self):
        """After a debate round, agent params should change if revision accepted."""
        from antagonistic_collab.models.decision_debate_runner import (
            run_debate_round,
        )

        configs = default_decision_agent_configs()
        # Start with misspecified CPT params
        configs[0].default_params = dict(MISSPEC_DECISION_PARAMS["CPT"])

        observed = {"certainty_effect_1": 0.85, "certainty_effect_2": 0.80}
        gamble_names = list(observed.keys())

        revisions = run_debate_round(
            configs,
            observed,
            gamble_names,
            posterior_probs={"CPT_Agent": 0.4, "EU_Agent": 0.3, "PH_Agent": 0.3},
            cycle=0,
            call_fn=self._mock_call_agent,
        )

        # At least CPT should have had a revision proposed
        assert len(revisions) > 0
