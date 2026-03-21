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

    def test_rmse_validated_against_all_observed(self):
        """RMSE gate should use accumulated observations, not just current cycle."""
        from antagonistic_collab.models.decision_debate_runner import (
            run_debate_round,
        )

        configs = default_decision_agent_configs()
        configs[0].default_params = dict(MISSPEC_DECISION_PARAMS["CPT"])

        # Current cycle: 2 gambles where the revision helps
        observed = {"certainty_effect_1": 0.85, "certainty_effect_2": 0.80}
        gamble_names = list(observed.keys())

        # Historical observations: 3 gambles where revision might NOT help
        all_observed = {
            "certainty_effect_1": 0.85,
            "certainty_effect_2": 0.80,
            "common_ratio_high": 0.60,
            "common_ratio_low": 0.45,
            "loss_aversion_symmetric": 0.30,
        }

        revisions_local = run_debate_round(
            configs,
            observed,
            gamble_names,
            posterior_probs={"CPT_Agent": 0.4, "EU_Agent": 0.3, "PH_Agent": 0.3},
            cycle=0,
            call_fn=self._mock_call_agent,
        )

        revisions_global = run_debate_round(
            configs,
            observed,
            gamble_names,
            posterior_probs={"CPT_Agent": 0.4, "EU_Agent": 0.3, "PH_Agent": 0.3},
            cycle=0,
            call_fn=self._mock_call_agent,
            all_observed=all_observed,
        )

        # With accumulated data, acceptance should be same or stricter
        local_accepted = sum(1 for r in revisions_local if r["accepted"])
        global_accepted = sum(1 for r in revisions_global if r["accepted"])
        assert global_accepted <= local_accepted

    def test_neutral_revisions_rejected_with_zero_tolerance(self):
        """Revisions that don't improve RMSE should be rejected with tolerance=0."""
        from antagonistic_collab.models.decision_debate_runner import (
            validate_revision_rmse,
        )

        observed = {"g1": 0.8, "g2": 0.3}
        baseline = {"g1": 0.75, "g2": 0.35}
        # Same predictions = no improvement
        accepted, _, _ = validate_revision_rmse(
            observed, baseline, baseline, tolerance=0.0
        )
        assert not accepted


# ── Interpretation Preservation (Phase 0) ──


class TestInterpretationPreservation:
    """Test that debate records include agent interpretations, even without revisions."""

    def _mock_no_revision(self, system, user):
        """Mock LLM that interprets but proposes no revision."""
        import json

        return json.dumps(
            {
                "interpretation": "The data shows clear risk aversion pattern.",
                "revision": None,
            }
        )

    def _mock_with_revision(self, system, user):
        """Mock LLM that interprets and proposes a revision."""
        import json

        return json.dumps(
            {
                "interpretation": "Loss aversion parameter is too high for these gambles.",
                "revision": {
                    "description": "Reduce loss aversion",
                    "new_params": {"alpha": 0.85, "lambda_": 2.2},
                },
            }
        )

    def test_interpretation_preserved_without_revision(self):
        """Agents that don't propose revisions should still have records with interpretation."""
        from antagonistic_collab.models.decision_debate_runner import (
            run_debate_round,
        )

        configs = default_decision_agent_configs()
        observed = {"certainty_effect_1": 0.85, "certainty_effect_2": 0.80}
        gamble_names = list(observed.keys())

        records = run_debate_round(
            configs,
            observed,
            gamble_names,
            posterior_probs={"CPT_Agent": 0.4, "EU_Agent": 0.3, "PH_Agent": 0.3},
            cycle=0,
            call_fn=self._mock_no_revision,
        )

        # Should have a record for each agent, even without revisions
        assert len(records) == 3
        for rec in records:
            assert "interpretation" in rec
            assert rec["interpretation"] != ""

    def test_interpretation_preserved_with_revision(self):
        """Agents that propose revisions should have interpretation in their record."""
        from antagonistic_collab.models.decision_debate_runner import (
            run_debate_round,
        )

        configs = default_decision_agent_configs()
        configs[0].default_params = dict(MISSPEC_DECISION_PARAMS["CPT"])

        observed = {"certainty_effect_1": 0.85, "certainty_effect_2": 0.80}
        gamble_names = list(observed.keys())

        records = run_debate_round(
            configs,
            observed,
            gamble_names,
            posterior_probs={"CPT_Agent": 0.4, "EU_Agent": 0.3, "PH_Agent": 0.3},
            cycle=0,
            call_fn=self._mock_with_revision,
        )

        # All agents should have records with interpretations
        assert len(records) == 3
        for rec in records:
            assert "interpretation" in rec
            assert "Loss aversion" in rec["interpretation"] or "loss aversion" in rec["interpretation"].lower()


# ── Crux Protocol (Phase 1) ──


class TestDecisionCruxProtocol:
    """Test crux identification, negotiation, and finalization for decision domain."""

    def _mock_crux_call(self, system, user):
        """Mock LLM that proposes cruxes targeting gamble groups."""
        import json

        return json.dumps(
            {
                "cruxes": [
                    {
                        "description": "Certainty effect gambles should distinguish CPT from EU",
                        "discriminating_experiment": "certainty_effect",
                        "resolution_criterion": "RMSE < 0.10 for winner",
                    }
                ]
            }
        )

    def _mock_crux_call_two(self, system, user):
        """Mock that proposes two cruxes."""
        import json

        return json.dumps(
            {
                "cruxes": [
                    {
                        "description": "Loss aversion distinguishes CPT",
                        "discriminating_experiment": "loss_aversion",
                        "resolution_criterion": "loss aversion index > 2",
                    },
                    {
                        "description": "PH diagnostic gambles test lexicographic rules",
                        "discriminating_experiment": "ph_diagnostic",
                        "resolution_criterion": "PH predicts correctly",
                    },
                ]
            }
        )

    def _mock_negotiation_accept(self, system, user):
        """Mock LLM that accepts all cruxes."""
        import json

        return json.dumps(
            {
                "responses": [
                    {"crux_id": "crux_001", "action": "accept", "reason": "Agreed"},
                ]
            }
        )

    def _mock_negotiation_counter(self, system, user):
        """Mock LLM that counter-proposes."""
        import json

        return json.dumps(
            {
                "responses": [
                    {
                        "crux_id": "crux_001",
                        "action": "counter",
                        "reason": "Better test exists",
                        "counter_crux": {
                            "description": "Fourfold gain pattern is more diagnostic",
                            "discriminating_experiment": "fourfold_gain",
                        },
                    },
                ]
            }
        )

    def test_crux_identification_returns_crux_objects(self):
        """Crux identification should return list of Crux dataclass objects."""
        from antagonistic_collab.models.decision_debate_runner import (
            run_decision_crux_identification,
        )
        from antagonistic_collab.epistemic_state import Crux

        configs = default_decision_agent_configs()
        cruxes = run_decision_crux_identification(
            configs, client=None, call_fn=self._mock_crux_call, cycle=1, crux_counter=0
        )

        assert len(cruxes) > 0
        for crux in cruxes:
            assert isinstance(crux, Crux)
            assert crux.proposer in [c.name for c in configs]

    def test_crux_identification_targets_gamble_groups(self):
        """Crux discriminating_experiment should reference valid gamble groups."""
        from antagonistic_collab.models.decision_debate_runner import (
            run_decision_crux_identification,
        )
        from antagonistic_collab.models.decision_eig import GAMBLE_GROUPS

        configs = default_decision_agent_configs()
        cruxes = run_decision_crux_identification(
            configs, client=None, call_fn=self._mock_crux_call, cycle=1, crux_counter=0
        )

        valid_groups = set(GAMBLE_GROUPS.keys())
        for crux in cruxes:
            if crux.discriminating_experiment:
                assert crux.discriminating_experiment in valid_groups, (
                    f"Crux targets '{crux.discriminating_experiment}' which is not a valid gamble group"
                )

    def test_crux_identification_assigns_unique_ids(self):
        """Each crux should have a unique ID."""
        from antagonistic_collab.models.decision_debate_runner import (
            run_decision_crux_identification,
        )

        configs = default_decision_agent_configs()
        cruxes = run_decision_crux_identification(
            configs, client=None, call_fn=self._mock_crux_call_two, cycle=1, crux_counter=0
        )

        ids = [c.id for c in cruxes]
        assert len(ids) == len(set(ids)), "Crux IDs are not unique"

    def test_crux_negotiation_updates_supporters(self):
        """Accepting a crux should add the agent to supporters."""
        from antagonistic_collab.models.decision_debate_runner import (
            run_decision_crux_identification,
            run_decision_crux_negotiation,
        )

        configs = default_decision_agent_configs()
        cruxes = run_decision_crux_identification(
            configs, client=None, call_fn=self._mock_crux_call, cycle=1, crux_counter=0
        )

        updated = run_decision_crux_negotiation(
            configs, cruxes, client=None, call_fn=self._mock_negotiation_accept, cycle=1
        )

        # At least one crux should have multiple supporters
        multi_support = [c for c in updated if len(c.supporters) > 1]
        assert len(multi_support) > 0

    def test_crux_negotiation_counter_adds_new_crux(self):
        """Counter-proposing should add a new crux to the list."""
        from antagonistic_collab.models.decision_debate_runner import (
            run_decision_crux_identification,
            run_decision_crux_negotiation,
        )

        configs = default_decision_agent_configs()
        cruxes = run_decision_crux_identification(
            configs, client=None, call_fn=self._mock_crux_call, cycle=1, crux_counter=0
        )
        initial_count = len(cruxes)

        updated = run_decision_crux_negotiation(
            configs, cruxes, client=None, call_fn=self._mock_negotiation_counter, cycle=1
        )

        assert len(updated) > initial_count

    def test_finalize_cruxes_accepts_with_enough_support(self):
        """Cruxes with >= min_supporters should be accepted."""
        from antagonistic_collab.models.decision_debate_runner import (
            finalize_decision_cruxes,
        )
        from antagonistic_collab.epistemic_state import Crux

        cruxes = [
            Crux(
                id="crux_001",
                proposer="CPT_Agent",
                description="Test crux",
                supporters=["CPT_Agent", "EU_Agent"],
                cycle_proposed=1,
            ),
            Crux(
                id="crux_002",
                proposer="PH_Agent",
                description="Lonely crux",
                supporters=["PH_Agent"],
                cycle_proposed=1,
            ),
        ]

        accepted = finalize_decision_cruxes(cruxes, min_supporters=2)
        assert len(accepted) == 1
        assert accepted[0].id == "crux_001"
        assert accepted[0].status == "accepted"
        assert cruxes[1].status == "rejected"

    def test_cruxes_to_boost_indices(self):
        """Accepted cruxes should map to valid gamble group indices."""
        from antagonistic_collab.models.decision_debate_runner import (
            decision_cruxes_to_boost_indices,
        )
        from antagonistic_collab.epistemic_state import Crux
        from antagonistic_collab.models.decision_eig import GAMBLE_GROUPS

        group_names = list(GAMBLE_GROUPS.keys())
        cruxes = [
            Crux(
                id="crux_001",
                proposer="CPT_Agent",
                description="Test crux",
                discriminating_experiment="certainty_effect",
                status="accepted",
                supporters=["CPT_Agent", "EU_Agent"],
                cycle_proposed=1,
            ),
            Crux(
                id="crux_002",
                proposer="PH_Agent",
                description="PH crux",
                discriminating_experiment="ph_diagnostic",
                status="accepted",
                supporters=["PH_Agent", "EU_Agent"],
                cycle_proposed=1,
            ),
        ]

        indices = decision_cruxes_to_boost_indices(cruxes, group_names)
        assert len(indices) == 2
        assert group_names[indices[0]] == "certainty_effect"
        assert group_names[indices[1]] == "ph_diagnostic"

    def test_cruxes_to_boost_indices_ignores_invalid_groups(self):
        """Cruxes targeting non-existent groups should be skipped."""
        from antagonistic_collab.models.decision_debate_runner import (
            decision_cruxes_to_boost_indices,
        )
        from antagonistic_collab.epistemic_state import Crux
        from antagonistic_collab.models.decision_eig import GAMBLE_GROUPS

        group_names = list(GAMBLE_GROUPS.keys())
        cruxes = [
            Crux(
                id="crux_001",
                proposer="CPT_Agent",
                description="Bad target",
                discriminating_experiment="nonexistent_group",
                status="accepted",
                supporters=["CPT_Agent", "EU_Agent"],
                cycle_proposed=1,
            ),
        ]

        indices = decision_cruxes_to_boost_indices(cruxes, group_names)
        assert len(indices) == 0


# ── Meta-Agents (Phase 3) ──


class TestDecisionMetaAgents:
    """Test meta-agent creation and arbiter round for decision domain."""

    def test_create_decision_meta_agents(self):
        """Should create Integrator and Critic MetaAgentConfig objects."""
        from antagonistic_collab.models.decision_debate_runner import (
            create_decision_meta_agents,
        )
        from antagonistic_collab.debate_protocol import MetaAgentConfig

        meta_agents = create_decision_meta_agents()
        assert len(meta_agents) == 2
        names = {ma.name for ma in meta_agents}
        assert "Integrator" in names
        assert "Critic" in names
        for ma in meta_agents:
            assert isinstance(ma, MetaAgentConfig)
            assert "CPT" in ma.system_prompt or "decision" in ma.system_prompt.lower()

    def test_meta_agent_prompts_reference_decision_models(self):
        """Meta-agent system prompts should reference CPT/EU/PH, not GCM/SUSTAIN/RULEX."""
        from antagonistic_collab.models.decision_debate_runner import (
            create_decision_meta_agents,
        )

        for ma in create_decision_meta_agents():
            # Should reference decision models
            assert any(m in ma.system_prompt for m in ["CPT", "EU", "PH", "Prospect"]), (
                f"{ma.name} prompt doesn't reference decision models"
            )
            # Should NOT reference categorization models
            assert "GCM" not in ma.system_prompt
            assert "SUSTAIN" not in ma.system_prompt
            assert "RULEX" not in ma.system_prompt

    def _mock_meta_agent_call(self, system, user):
        """Mock LLM call returning meta-agent JSON."""
        import json

        return json.dumps(
            {
                "interpretation": "CPT and EU both struggle with certainty effects, but for different reasons.",
                "confounds_flagged": ["Sample size may be too small to distinguish models"],
                "hypothesis": "Test with mixed gambles to expose loss aversion differences",
                "claims": [],
            }
        )

    def test_arbiter_round_returns_meta_agent_responses(self):
        """Arbiter round should return structured responses from meta-agents."""
        from antagonistic_collab.models.decision_debate_runner import (
            create_decision_meta_agents,
            run_decision_arbiter_round,
        )

        meta_agents = create_decision_meta_agents()
        debate_records = [
            {
                "agent_name": "CPT_Agent",
                "interpretation": "Loss aversion param is too high",
                "accepted": False,
                "has_revision": False,
            },
            {
                "agent_name": "EU_Agent",
                "interpretation": "Risk aversion parameter fits well",
                "accepted": False,
                "has_revision": False,
            },
            {
                "agent_name": "PH_Agent",
                "interpretation": "Lexicographic rules match the pattern",
                "accepted": False,
                "has_revision": False,
            },
        ]
        posterior = {"CPT_Agent": 0.4, "EU_Agent": 0.35, "PH_Agent": 0.25}

        responses = run_decision_arbiter_round(
            meta_agents=meta_agents,
            debate_records=debate_records,
            observed={"certainty_effect_1": 0.85},
            posterior=posterior,
            cycle=1,
            client=None,
            call_fn=self._mock_meta_agent_call,
        )

        assert len(responses) == 2  # Integrator + Critic
        for resp in responses:
            assert "agent" in resp
            assert "interpretation" in resp
            assert resp["meta_agent"] is True

    def test_arbiter_round_does_not_modify_params(self):
        """Meta-agents should never trigger parameter revisions."""
        from antagonistic_collab.models.decision_debate_runner import (
            create_decision_meta_agents,
            run_decision_arbiter_round,
        )

        meta_agents = create_decision_meta_agents()
        configs = default_decision_agent_configs()
        original_params = {c.name: dict(c.default_params) for c in configs}

        debate_records = [
            {
                "agent_name": c.name,
                "interpretation": "Some analysis",
                "accepted": False,
                "has_revision": False,
            }
            for c in configs
        ]

        run_decision_arbiter_round(
            meta_agents=meta_agents,
            debate_records=debate_records,
            observed={"certainty_effect_1": 0.85},
            posterior={"CPT_Agent": 0.4, "EU_Agent": 0.3, "PH_Agent": 0.3},
            cycle=1,
            client=None,
            call_fn=self._mock_meta_agent_call,
        )

        # Params should be unchanged
        for c in configs:
            assert c.default_params == original_params[c.name]

    def test_arbiter_round_includes_hypotheses_and_confounds(self):
        """Responses should include hypotheses and confounds from meta-agents."""
        from antagonistic_collab.models.decision_debate_runner import (
            create_decision_meta_agents,
            run_decision_arbiter_round,
        )

        meta_agents = create_decision_meta_agents()
        debate_records = [
            {
                "agent_name": "CPT_Agent",
                "interpretation": "Analysis of CPT predictions",
                "accepted": False,
                "has_revision": False,
            },
        ]

        responses = run_decision_arbiter_round(
            meta_agents=meta_agents,
            debate_records=debate_records,
            observed={"certainty_effect_1": 0.85},
            posterior={"CPT_Agent": 0.4, "EU_Agent": 0.3, "PH_Agent": 0.3},
            cycle=1,
            client=None,
            call_fn=self._mock_meta_agent_call,
        )

        for resp in responses:
            assert "hypothesis" in resp
            assert "confounds" in resp


# ── Arbiter Integration (Phase 4) ──


class TestDecisionArbiterIntegration:
    """Test the full arbiter-enabled debate loop with mocked LLM."""

    def _mock_call(self, system, user):
        """Mock LLM that handles all prompt types."""
        import json

        if "crux identification" in user.lower() or "crux" in user.lower() and "propose" in user.lower():
            return json.dumps(
                {
                    "cruxes": [
                        {
                            "description": "Certainty effect distinguishes CPT from EU",
                            "discriminating_experiment": "certainty_effect",
                            "resolution_criterion": "RMSE < 0.10",
                        }
                    ]
                }
            )
        elif "crux negotiation" in user.lower() or "accept|reject|counter" in user:
            return json.dumps(
                {
                    "responses": [
                        {
                            "crux_id": "crux_001",
                            "action": "accept",
                            "reason": "Agreed",
                        }
                    ]
                }
            )
        elif "meta-agent" in user.lower() or "meta" in system.lower():
            return json.dumps(
                {
                    "interpretation": "Theory synthesis across models",
                    "confounds_flagged": [],
                    "hypothesis": "Test loss aversion gambles next",
                    "claims": [],
                }
            )
        else:
            # Default: debate round response
            return json.dumps(
                {
                    "interpretation": "My predictions show moderate fit.",
                    "revision": None,
                }
            )

    def test_arbiter_mode_runs_without_error(self):
        """Full loop with enable_arbiter=True should complete without error."""
        from antagonistic_collab.models.decision_debate_runner import (
            run_decision_debate,
        )

        result = run_decision_debate(
            gt_model="EU",
            n_cycles=3,
            n_subjects=30,
            learning_rate=0.01,
            selection_strategy="thompson",
            agent_params={
                "CPT_Agent": {"alpha": 0.7, "beta": 0.7, "lambda_": 1.5, "gamma_pos": 0.5, "gamma_neg": 0.5, "temperature": 1.0},
                "EU_Agent": {"r": 0.3, "temperature": 1.0},
                "PH_Agent": {"outcome_threshold_frac": 0.15, "prob_threshold": 0.15, "phi": 0.5},
            },
            call_fn=self._mock_call,
            enable_debate=True,
            enable_arbiter=True,
            verbose=False,
        )

        assert "condition" in result
        assert result["condition"] == "arbiter"
        assert "cruxes" in result
        assert "meta_agent_responses" in result

    def test_arbiter_returns_cruxes_in_results(self):
        """Results should include crux data from each cycle."""
        from antagonistic_collab.models.decision_debate_runner import (
            run_decision_debate,
        )

        result = run_decision_debate(
            gt_model="PH",
            n_cycles=3,
            n_subjects=30,
            learning_rate=0.01,
            selection_strategy="thompson",
            agent_params={
                "CPT_Agent": {"alpha": 0.7, "beta": 0.7, "lambda_": 1.5, "gamma_pos": 0.5, "gamma_neg": 0.5, "temperature": 1.0},
                "EU_Agent": {"r": 0.3, "temperature": 1.0},
                "PH_Agent": {"outcome_threshold_frac": 0.15, "prob_threshold": 0.15, "phi": 0.5},
            },
            call_fn=self._mock_call,
            enable_debate=True,
            enable_arbiter=True,
            verbose=False,
        )

        # Cruxes should exist (possibly empty for cycle 0, populated for later)
        assert isinstance(result["cruxes"], list)
        # Meta-agent responses should exist
        assert isinstance(result["meta_agent_responses"], list)

    def test_arbiter_skips_cruxes_on_cycle_zero(self):
        """Crux identification should not run on cycle 0 (no data yet)."""
        from antagonistic_collab.models.decision_debate_runner import (
            run_decision_debate,
        )

        result = run_decision_debate(
            gt_model="CPT",
            n_cycles=1,  # Only cycle 0
            n_subjects=30,
            learning_rate=0.01,
            selection_strategy="thompson",
            agent_params={
                "CPT_Agent": {"alpha": 0.7, "beta": 0.7, "lambda_": 1.5, "gamma_pos": 0.5, "gamma_neg": 0.5, "temperature": 1.0},
                "EU_Agent": {"r": 0.3, "temperature": 1.0},
                "PH_Agent": {"outcome_threshold_frac": 0.15, "prob_threshold": 0.15, "phi": 0.5},
            },
            call_fn=self._mock_call,
            enable_debate=True,
            enable_arbiter=True,
            verbose=False,
        )

        # With only 1 cycle (cycle 0), no cruxes should be identified
        assert len(result["cruxes"]) == 0
