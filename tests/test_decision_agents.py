"""Tests for decision-domain agent configurations."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from antagonistic_collab.models.decision_agents import (
    default_decision_agent_configs,
    CPT_AGENT_PROMPT,
    EU_AGENT_PROMPT,
    PH_AGENT_PROMPT,
)
from antagonistic_collab.models.decision_runner import GT_DECISION_PARAMS


class TestDecisionAgentConfigs:
    def test_three_agents_returned(self):
        configs = default_decision_agent_configs()
        assert len(configs) == 3

    def test_agent_names(self):
        configs = default_decision_agent_configs()
        names = {c.name for c in configs}
        assert names == {"CPT_Agent", "EU_Agent", "PH_Agent"}

    def test_each_has_model_class(self):
        configs = default_decision_agent_configs()
        for c in configs:
            assert c.model_class is not None
            assert hasattr(c.model_class, "predict"), f"{c.name} model has no predict()"

    def test_each_has_system_prompt(self):
        configs = default_decision_agent_configs()
        for c in configs:
            assert len(c.system_prompt) > 100, f"{c.name} prompt too short"

    def test_each_has_default_params(self):
        configs = default_decision_agent_configs()
        for c in configs:
            assert len(c.default_params) > 0, f"{c.name} has no default params"

    def test_params_match_gt(self):
        """Default params should match GT_DECISION_PARAMS."""
        configs = default_decision_agent_configs()
        name_to_model = {"CPT_Agent": "CPT", "EU_Agent": "EU", "PH_Agent": "PH"}
        for c in configs:
            model_name = name_to_model[c.name]
            assert c.default_params == GT_DECISION_PARAMS[model_name], (
                f"{c.name} params don't match GT: {c.default_params}"
            )

    def test_model_can_predict_with_default_params(self):
        """Each model should produce valid predictions with its default params."""
        gamble = {
            "outcomes_A": [100],
            "probs_A": [1.0],
            "outcomes_B": [200, 0],
            "probs_B": [0.5, 0.5],
        }
        configs = default_decision_agent_configs()
        for c in configs:
            result = c.model_class.predict(gamble, **c.default_params)
            p = result["p_choose_A"]
            assert 0.0 <= p <= 1.0, f"{c.name}: invalid prediction {p}"


class TestPromptContent:
    """Verify prompts contain essential information for debate."""

    def test_cpt_prompt_mentions_loss_aversion(self):
        assert "loss aversion" in CPT_AGENT_PROMPT.lower()

    def test_cpt_prompt_mentions_probability_weighting(self):
        assert "probability weighting" in CPT_AGENT_PROMPT.lower()

    def test_cpt_prompt_has_weaknesses_section(self):
        assert "STRUGGLES WITH" in CPT_AGENT_PROMPT

    def test_eu_prompt_mentions_independence_axiom(self):
        assert "independence" in EU_AGENT_PROMPT.lower()

    def test_eu_prompt_mentions_parsimony(self):
        assert (
            "parsimony" in EU_AGENT_PROMPT.lower()
            or "parsimonious" in EU_AGENT_PROMPT.lower()
        )

    def test_eu_prompt_has_weaknesses_section(self):
        assert "STRUGGLES WITH" in EU_AGENT_PROMPT

    def test_ph_prompt_mentions_lexicographic(self):
        assert "lexicographic" in PH_AGENT_PROMPT.lower()

    def test_ph_prompt_mentions_aspiration(self):
        assert (
            "aspiration" in PH_AGENT_PROMPT.lower()
            or "threshold" in PH_AGENT_PROMPT.lower()
        )

    def test_ph_prompt_has_weaknesses_section(self):
        assert "STRUGGLES WITH" in PH_AGENT_PROMPT

    def test_each_prompt_has_critique_section(self):
        for prompt in [CPT_AGENT_PROMPT, EU_AGENT_PROMPT, PH_AGENT_PROMPT]:
            assert "CRITIQUING OPPONENTS" in prompt

    def test_each_prompt_has_format_section(self):
        for prompt in [CPT_AGENT_PROMPT, EU_AGENT_PROMPT, PH_AGENT_PROMPT]:
            assert "FORMAT" in prompt
