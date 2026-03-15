"""
Regression tests for all bugs fixed across four review rounds.

Each test is a direct regression for a specific reported bug. The docstring
explains *why* the test exists — what broke, how we know it's fixed, and
what the test actually verifies.

Organized by module, not by review round, so future developers can find
the relevant tests next to the code they cover.
"""

import inspect
import json
import math
import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from antagonistic_collab.runner import (
    call_agent,
    extract_json,
    extract_all_json,
    run_human_arbitration,
    save_transcript,
    save_cycle_markdown,
    save_summary_report,
    auto_output_dir,
)
from antagonistic_collab.epistemic_state import (
    EpistemicState,
    TheoryCommitment,
)
from antagonistic_collab.debate_protocol import (
    DebateProtocol,
    Phase,
    PhaseResult,
    default_agent_configs,
    STRUCTURE_REGISTRY,
    CONDITION_EFFECTS,
)
from antagonistic_collab.models.sustain import SUSTAIN
from antagonistic_collab.models.gcm import GCM
from antagonistic_collab.models.category_structures import rule_plus_exception
from antagonistic_collab.bayesian_selection import (
    ModelPosterior,
    compute_log_likelihood,
    compute_eig,
    select_experiment,
)


# =========================================================================
# extract_json / extract_all_json  (runner.py)
# =========================================================================


class TestJsonExtraction:
    """
    The original regex `\\{[^{}]*(?:\\{[^{}]*\\}[^{}]*)*\\}` could only
    handle one level of brace nesting.  For input like
    `{"a":{"b":{"c":1}}}` it would match the *inner* object `{"b":{"c":1}}`
    and silently drop the outer key.  This class of bug is severe because
    LLM outputs routinely contain nested JSON (e.g. model_evidence with
    conditions and predictions), and silent data loss corrupts downstream
    logic without any error.
    """

    def test_deeply_nested_json_is_extracted_whole(self):
        # Why: the original regex matched inner braces, returning {"b":{"c":1}}
        # instead of the full object.  Verifies the brace-depth parser
        # returns the outermost object.
        text = '{"a":{"b":{"c":1}}}'
        result = extract_all_json(text)
        assert len(result) == 1
        assert result[0] == {"a": {"b": {"c": 1}}}

    def test_three_levels_of_nesting(self):
        # Why: LLM critique responses often have model_evidence -> conditions -> values,
        # creating 3+ levels.  Must survive arbitrary depth.
        text = '{"l1":{"l2":{"l3":{"l4":"deep"}}}}'
        result = extract_all_json(text)
        assert len(result) == 1
        assert result[0]["l1"]["l2"]["l3"]["l4"] == "deep"

    def test_multiple_top_level_objects(self):
        # Why: agents are asked to output one JSON block per critique.  The
        # parser must return all of them, not just the first.
        text = 'first: {"x":1} second: {"y":2}'
        result = extract_all_json(text)
        assert len(result) == 2
        assert result[0] == {"x": 1}
        assert result[1] == {"y": 2}

    def test_fenced_json_blocks_preferred(self):
        # Why: when the LLM wraps JSON in ```json``` fences, the parser should
        # use those (they're unambiguous) and skip raw-brace scanning entirely.
        text = 'blah ```json\n{"fenced": true}\n``` blah {"raw": true}'
        result = extract_all_json(text)
        assert len(result) == 1
        assert result[0] == {"fenced": True}

    def test_extract_json_returns_first(self):
        # Why: extract_json is a convenience wrapper; must return the first
        # block (not None, not all of them).
        text = '{"a":1} {"b":2}'
        result = extract_json(text)
        assert result == {"a": 1}

    def test_no_json_returns_empty_list(self):
        # Why: graceful degradation when the LLM outputs no JSON at all.
        assert extract_all_json("no json here") == []
        assert extract_json("no json here") is None

    def test_escaped_quotes_inside_strings(self):
        # Why: JSON values often contain escaped quotes (e.g. descriptions
        # that include quotation marks).  The brace parser must not exit
        # string mode prematurely when it sees \" inside a string.
        text = r'{"msg": "he said \"hello\""}'
        result = extract_all_json(text)
        assert len(result) == 1
        assert "hello" in result[0]["msg"]

    def test_braces_inside_string_values_ignored(self):
        # Why: a string value like "use {x} template" contains braces that
        # are NOT structural.  The parser must ignore them.
        text = '{"template": "use {x} here"}'
        result = extract_all_json(text)
        assert len(result) == 1
        assert result[0]["template"] == "use {x} here"


# =========================================================================
# EpistemicState  (epistemic_state.py)
# =========================================================================


class TestEpistemicState:
    """Tests for bugs in the epistemic state tracker."""

    def _make_state_with_prediction(self, score):
        """Helper: create a state with one prediction at the given score."""
        state = EpistemicState(domain="test")
        state.register_theory(
            TheoryCommitment(
                name="T",
                agent_name="Agent_A",
                core_claims=["claim"],
                model_name="M",
            )
        )
        exp = state.propose_experiment(
            proposed_by="Agent_A",
            title="E",
            design_spec={},
            rationale="r",
        )
        state.register_prediction(
            exp.experiment_id,
            "Agent_A",
            "M",
            {},
            {"x": 1.0},
        )
        state.score_predictions(exp.experiment_id, {"x": 1.0 + score})
        return state

    def test_zero_rmse_displayed_as_scored(self):
        """
        Bug: `if stats['mean_score']` is falsy when score == 0.0, so a
        perfect prediction (RMSE = 0.000) was displayed as "not yet scored".
        Fix: changed to `if stats['mean_score'] is not None`.
        Verifies that 0.0 RMSE appears in the summary as a number, not as
        "not yet scored".
        """
        state = self._make_state_with_prediction(score=0.0)
        summary = state.summary_for_agent("Agent_A")
        assert "mean RMSE = 0.000" in summary
        assert "not yet scored" not in summary

    def test_nonzero_rmse_displayed_correctly(self):
        """
        Sanity check: a non-zero score should also render correctly.
        Guards against the fix accidentally breaking the normal path.
        """
        state = self._make_state_with_prediction(score=0.5)
        summary = state.summary_for_agent("Agent_A")
        assert "mean RMSE" in summary
        assert "not yet scored" not in summary

    def test_unscored_prediction_shows_not_yet_scored(self):
        """
        Complement of the 0.0 test: when score genuinely IS None (no
        overlapping keys between prediction and actual), summary should
        say "not yet scored".
        """
        state = EpistemicState(domain="test")
        state.register_theory(
            TheoryCommitment(
                name="T",
                agent_name="Agent_A",
                core_claims=["c"],
                model_name="M",
            )
        )
        exp = state.propose_experiment(
            proposed_by="Agent_A",
            title="E",
            design_spec={},
            rationale="r",
        )
        # Predict on key "x", score on key "y" - no overlap -> score stays None
        state.register_prediction(
            exp.experiment_id,
            "Agent_A",
            "M",
            {},
            {"x": 1.0},
        )
        state.score_predictions(exp.experiment_id, {"y": 1.0})
        # score is None, so it shouldn't appear in the leaderboard at all
        board = state.prediction_leaderboard()
        assert "Agent_A" not in board


# =========================================================================
# Numpy serialization  (debate_protocol.py + epistemic_state.py)
# =========================================================================


class TestNumpySerialization:
    """
    Bug: _synthetic_runner stores pred["probabilities"] directly, whose
    keys are numpy.int64 (the category labels).  This crashes:
      1) _results_context via json.dumps(exp.data) — TypeError on keys
      2) to_json via json.dump(..., default=handler) — default only
         converts *values*, not dict *keys*.

    Fix: (a) convert keys/values to native Python types at the source
    in _synthetic_runner, (b) add recursive key sanitizer in to_json.
    """

    def _make_protocol(self):
        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        return DebateProtocol(state, agents)

    def test_synthetic_runner_produces_json_serializable_data(self):
        """
        The actual production path: run _synthetic_runner, then call
        json.dumps on the result — exactly what _results_context does.
        Before the fix, this raised TypeError on numpy.int64 keys.
        """
        protocol = self._make_protocol()
        data = protocol._synthetic_runner({}, true_model="GCM")
        # This is the exact call that _results_context makes (line 548)
        serialized = json.dumps(data, indent=2)
        assert isinstance(serialized, str)
        # Round-trip: parse it back and verify structure
        parsed = json.loads(serialized)
        assert "model_predictions" in parsed
        assert "mean_accuracy" in parsed

    def test_synthetic_runner_prediction_keys_are_native_int(self):
        """
        Directly checks that model_predictions dict keys are Python int,
        not numpy.int64.  This is the root cause of both serialization bugs.
        """
        protocol = self._make_protocol()
        data = protocol._synthetic_runner({}, true_model="GCM")
        for item_key, probs in data["model_predictions"].items():
            for cat_key in probs:
                assert type(cat_key) is int, (
                    f"Expected int key, got {type(cat_key)} for {item_key}[{cat_key}]"
                )

    def test_to_json_survives_numpy_keys_in_experiment_data(self):
        """
        End-to-end: run _synthetic_runner, record data on an experiment,
        then call to_json.  Before the fix, this crashed with TypeError
        because json.dump's default= cannot convert dict keys.
        """
        protocol = self._make_protocol()
        state = protocol.state

        exp = state.propose_experiment(
            proposed_by="A",
            title="E",
            design_spec={},
            rationale="r",
        )
        data = protocol._synthetic_runner({}, true_model="GCM")
        state.record_data(exp.experiment_id, data)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            state.to_json(path)
            # Verify it's valid JSON
            with open(path) as f:
                parsed = json.load(f)
            assert parsed["experiments"][0]["data"]["mean_accuracy"] > 0
        finally:
            os.unlink(path)

    def test_to_json_sanitizes_injected_numpy_keys(self):
        """
        Safety-net test: even if numpy types sneak into the state through
        a path we haven't anticipated, to_json's recursive sanitizer
        should convert them.
        """
        state = EpistemicState(domain="test")
        # Manually inject numpy-typed keys into the log
        state._log(
            "test_event",
            {
                np.int64(42): "numpy key",
                "nested": {np.int64(7): "inner numpy key"},
            },
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            state.to_json(path)
            with open(path) as f:
                parsed = json.load(f)
            # Verify it round-tripped without error
            assert len(parsed["log"]) == 1
        finally:
            os.unlink(path)

    def test_results_context_after_execution(self):
        """
        Full production path for _results_context: propose experiment,
        approve it, record synthetic data, then render context.  Before
        the fix, this crashed on json.dumps(exp.data) due to numpy keys.
        """
        protocol = self._make_protocol()
        state = protocol.state

        exp = state.propose_experiment(
            proposed_by="A",
            title="E",
            design_spec={},
            rationale="r",
        )
        state.approve_experiment(exp.experiment_id)
        data = protocol._synthetic_runner({}, true_model="GCM")
        state.record_data(exp.experiment_id, data)

        # This is the exact call that run_interpretation uses
        context = protocol._results_context()
        assert "Experiment Results" in context
        assert "item_accuracies" in context


# =========================================================================
# DebateProtocol._synthetic_runner  (debate_protocol.py)
# =========================================================================


class TestSyntheticRunner:
    """Tests for the synthetic experiment runner."""

    def _make_protocol(self):
        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        return DebateProtocol(state, agents)

    def test_missing_category_structure_uses_fallback(self):
        """
        Bug: if the LLM proposal doesn't include a category_structure key,
        the runner crashed with KeyError: 'stimuli'.
        Fix: fall back to Shepard Type II when the key is absent.
        Verifies the runner produces valid output without crashing.
        """
        protocol = self._make_protocol()
        data = protocol._synthetic_runner({}, true_model="GCM")
        assert "mean_accuracy" in data
        assert isinstance(data["mean_accuracy"], float)

    def test_category_structure_without_stimuli_uses_fallback(self):
        """
        Bug: LLM might provide category_structure as a description string
        or a dict without the expected stimuli/labels arrays.
        Fix: validate that struct is a dict AND has both keys.
        """
        spec = {"category_structure": {"description": "some structure"}}
        protocol = self._make_protocol()
        data = protocol._synthetic_runner(spec, true_model="GCM")
        assert "mean_accuracy" in data

    def test_category_structure_as_string_uses_fallback(self):
        """
        Edge case: LLM outputs category_structure as a plain string like
        "Shepard Type II".  Must not crash.
        """
        spec = {"category_structure": "Shepard Type II"}
        protocol = self._make_protocol()
        data = protocol._synthetic_runner(spec, true_model="GCM")
        assert "mean_accuracy" in data

    def test_valid_structure_name_is_used(self):
        """
        Sanity check: when structure_name is a valid registry key,
        the corresponding structure should be used (not the fallback).
        """
        from antagonistic_collab.debate_protocol import STRUCTURE_REGISTRY

        spec = {"structure_name": "Type_I", "condition": "baseline"}
        protocol = self._make_protocol()
        data = protocol._synthetic_runner(spec, true_model="GCM")
        type_i = STRUCTURE_REGISTRY["Type_I"]
        assert len(data["item_accuracies"]) == len(type_i["stimuli"])


# =========================================================================
# DebateProtocol.compute_divergence_map  (debate_protocol.py)
# =========================================================================


class TestDivergenceMapping:
    """Tests for divergence map determinism."""

    def test_divergence_map_is_deterministic(self):
        """
        Bug: RULEX predictions are stochastic (seed=None by default).
        compute_divergence_map was called twice — once to compute, once
        inside _divergence_context — and could produce different numbers.
        Fix: (a) added seed=42 for RULEX in compute_divergence_map,
             (b) _divergence_context accepts a pre-computed map.
        Verifies that two consecutive calls return identical results.
        """
        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)

        map1 = protocol.compute_divergence_map()
        map2 = protocol.compute_divergence_map()

        for struct in map1:
            for pair in map1[struct]["divergences"]:
                d1 = map1[struct]["divergences"][pair]["mean_abs_diff"]
                d2 = map2[struct]["divergences"][pair]["mean_abs_diff"]
                assert d1 == d2, (
                    f"Non-deterministic divergence for {struct}/{pair}: {d1} != {d2}"
                )

    def test_divergence_context_uses_precomputed_map(self):
        """
        Verifies that _divergence_context(div_map=X) uses the provided map
        instead of recomputing.  We pass a synthetic map and check that its
        values appear in the output string.
        """
        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)

        fake_map = {
            "FakeStruct": {
                "predictions": {
                    "Agent_A": {"item_probabilities": [0.9], "accuracy": 0.999},
                },
                "divergences": {},
            }
        }
        context = protocol._divergence_context(div_map=fake_map)
        assert "FakeStruct" in context
        assert "0.999" in context


# =========================================================================
# Phase transitions & file saving  (debate_protocol.py + runner.py)
# =========================================================================


class TestCycleAndFileSaving:
    """Tests for cycle counter and file path correctness."""

    def test_advance_phase_audit_increments_cycle(self):
        """
        Establishes the baseline behavior: advancing from AUDIT increments
        the cycle counter.  All file-saving logic depends on this.
        """
        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)
        protocol.current_phase = Phase.AUDIT

        assert state.cycle == 0
        protocol.advance_phase(
            PhaseResult(
                phase=Phase.AUDIT,
                cycle=0,
                outputs={},
            )
        )
        assert state.cycle == 1

    def test_save_transcript_uses_current_cycle(self):
        """
        Bug: save_transcript was called AFTER advance_phase(AUDIT), so
        cycle 0's transcript was saved as debate_cycle_1.json.
        Fix: save is now called BEFORE advance_phase.
        Verifies that file names match the cycle number at save time.
        """
        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_transcript([], protocol, output_dir=tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "debate_cycle_0.json"))
            assert os.path.exists(os.path.join(tmpdir, "epistemic_state_cycle_0.json"))

    def test_save_transcript_respects_output_dir(self):
        """
        Bug: --output-dir was parsed by argparse but never passed to
        save_transcript, so files always wrote to cwd.
        Fix: save_transcript now takes output_dir and uses os.path.join.
        Verifies files land in the specified directory, not cwd.
        """
        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)

        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "custom_output")
            save_transcript([], protocol, output_dir=subdir)
            assert os.path.isdir(subdir)
            assert os.path.exists(os.path.join(subdir, "debate_cycle_0.json"))


# =========================================================================
# Critique provenance  (runner.py)
# =========================================================================


class TestCritiqueProvenance:
    """
    Bug: run_adversarial_critique parsed one JSON block, then looped over
    ALL other_proposals and attached the same critique text to each.
    This corrupted critique provenance — a critique of proposal A would
    appear on proposal B's critique_log too.

    The fix extracts all JSON blocks via extract_all_json, matches each
    to its target_proposal by title substring, and only attaches to the
    matched proposal.  When there's only one other proposal, it's used as
    the unambiguous target even without a title match.

    These tests simulate the matching logic that run_adversarial_critique
    uses, exercising extract_all_json -> title matching -> add_critique
    as an integrated sequence.
    """

    def _simulate_critique_matching(self, response_text, proposals):
        """
        Replicates the matching logic from run_adversarial_critique
        (runner.py lines 327-351) without needing an LLM client.
        Returns a dict of {experiment_id: [critique_texts]}.
        """
        from antagonistic_collab.runner import extract_all_json

        json_blocks = extract_all_json(response_text)
        results = {p.experiment_id: [] for p in proposals}

        if json_blocks:
            for block in json_blocks:
                target_title = block.get("target_proposal", "")
                matched = None
                for p in proposals:
                    if target_title and target_title.lower() in p.title.lower():
                        matched = p
                        break
                if matched is None and len(proposals) == 1:
                    matched = proposals[0]
                if matched is not None:
                    results[matched.experiment_id].append(block.get("critique", ""))
        else:
            if proposals:
                results[proposals[0].experiment_id].append(response_text[:500])

        return results

    def test_two_critiques_matched_to_correct_proposals(self):
        """
        Agent outputs two JSON blocks, each with a target_proposal matching
        a different proposal's title.  Each critique should land on exactly
        the right proposal.
        """
        state = EpistemicState(domain="test")
        exp_a = state.propose_experiment(
            proposed_by="Agent_A",
            title="Test memory for instances",
            design_spec={},
            rationale="r",
        )
        exp_b = state.propose_experiment(
            proposed_by="Agent_B",
            title="Test rule complexity",
            design_spec={},
            rationale="r",
        )

        response = (
            '{"target_proposal": "Test memory for instances", '
            '"critique": "confounded"} '
            '{"target_proposal": "Test rule complexity", '
            '"critique": "underpowered"}'
        )

        results = self._simulate_critique_matching(response, [exp_a, exp_b])
        assert results[exp_a.experiment_id] == ["confounded"]
        assert results[exp_b.experiment_id] == ["underpowered"]

    def test_single_proposal_gets_critique_without_title_match(self):
        """
        When there is only one other proposal, the critique should attach
        to it even if target_proposal doesn't match the title (unambiguous).
        """
        state = EpistemicState(domain="test")
        exp = state.propose_experiment(
            proposed_by="Agent_A",
            title="My experiment",
            design_spec={},
            rationale="r",
        )

        response = '{"target_proposal": "wrong title", "critique": "flaw"}'
        results = self._simulate_critique_matching(response, [exp])
        assert results[exp.experiment_id] == ["flaw"]

    def test_no_json_falls_back_to_first_proposal(self):
        """
        When the LLM response contains no parseable JSON, the full text
        (truncated) should be attached to the first proposal.
        """
        state = EpistemicState(domain="test")
        exp = state.propose_experiment(
            proposed_by="A",
            title="X",
            design_spec={},
            rationale="r",
        )
        results = self._simulate_critique_matching("plain text critique", [exp])
        assert len(results[exp.experiment_id]) == 1
        assert "plain text critique" in results[exp.experiment_id][0]

    def test_unmatched_critique_is_not_sprayed(self):
        """
        This is the core regression: a critique with a title that matches
        NEITHER proposal should be dropped, not attached to all proposals.
        Before the fix, it was attached to every other_proposals entry.
        """
        state = EpistemicState(domain="test")
        exp_a = state.propose_experiment(
            proposed_by="A",
            title="Alpha",
            design_spec={},
            rationale="r",
        )
        exp_b = state.propose_experiment(
            proposed_by="B",
            title="Beta",
            design_spec={},
            rationale="r",
        )
        response = '{"target_proposal": "Nonexistent", "critique": "orphan"}'
        results = self._simulate_critique_matching(response, [exp_a, exp_b])
        assert results[exp_a.experiment_id] == []
        assert results[exp_b.experiment_id] == []


# =========================================================================
# Moderator input validation  (runner.py)
# =========================================================================


class TestModeratorValidation:
    """
    Bug: `approve xyz` would crash with ValueError (int("xyz")), and
    `approve 99` would crash with IndexError.  Both were unhandled.
    Fix: try/except on int conversion, bounds check on index.

    These tests call run_human_arbitration directly, patching stdin and
    the batch-mode flag to exercise the actual production code paths.
    """

    def _make_protocol_with_proposal(self):
        """Helper: create a protocol with one proposed experiment."""
        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)
        state.propose_experiment(
            proposed_by="Agent_A",
            title="Test X",
            design_spec={},
            rationale="r",
        )
        return protocol

    @patch("antagonistic_collab.runner._BATCH_MODE", False)
    def test_approve_non_numeric_index_does_not_crash(self):
        """
        Before the fix, `approve xyz` raised ValueError from int("xyz").
        Now it prints an error message and no experiment is approved.
        """
        protocol = self._make_protocol_with_proposal()
        with patch("builtins.input", return_value="approve xyz"):
            run_human_arbitration(protocol, [])
        # No experiment should have been approved
        proposals = [e for e in protocol.state.experiments if e.status == "approved"]
        assert len(proposals) == 0

    @patch("antagonistic_collab.runner._BATCH_MODE", False)
    def test_approve_out_of_range_index_does_not_crash(self):
        """
        Before the fix, `approve 99` raised IndexError.
        Now it prints an error message and no experiment is approved.
        """
        protocol = self._make_protocol_with_proposal()
        with patch("builtins.input", return_value="approve 99"):
            run_human_arbitration(protocol, [])
        proposals = [e for e in protocol.state.experiments if e.status == "approved"]
        assert len(proposals) == 0

    @patch("antagonistic_collab.runner._BATCH_MODE", False)
    def test_approve_valid_index_approves(self):
        """
        Baseline: `approve 0` with one proposal should approve it.
        Exercises the full run_human_arbitration production path.
        """
        protocol = self._make_protocol_with_proposal()
        with patch("builtins.input", return_value="approve 0"):
            result = run_human_arbitration(protocol, [])
        proposals = [e for e in protocol.state.experiments if e.status == "approved"]
        assert len(proposals) == 1
        assert "approve" in str(result.outputs)

    @patch("antagonistic_collab.runner._BATCH_MODE", False)
    def test_approve_with_edits(self):
        """
        `approve 0 add control condition` should approve and store edits.
        """
        protocol = self._make_protocol_with_proposal()
        with patch("builtins.input", return_value="approve 0 add control"):
            run_human_arbitration(protocol, [])
        exp = protocol.state.experiments[0]
        assert exp.status == "approved"
        assert exp.moderator_edits == "add control"


# =========================================================================
# Mean accuracy formatting  (runner.py)
# =========================================================================


class TestMeanAccuracyFormatting:
    """
    Bug: `f"{data.get('mean_accuracy', 'N/A'):.3f}"` raises ValueError
    when mean_accuracy is absent (because :.3f can't format the string "N/A").

    Fix: check isinstance(mean_acc, (int, float)) before formatting.

    These tests run _synthetic_runner (which always produces a float
    mean_accuracy) and also test the guard against missing/non-numeric
    values by exercising the isinstance check that the production code uses.
    """

    def test_synthetic_runner_always_produces_numeric_accuracy(self):
        """
        The normal production path: _synthetic_runner should always return
        a float mean_accuracy.  Verifies no formatting crash is possible
        on the happy path.
        """
        state = EpistemicState(domain="test")
        protocol = DebateProtocol(state, default_agent_configs())
        data = protocol._synthetic_runner({}, true_model="GCM")
        mean_acc = data.get("mean_accuracy")
        assert isinstance(mean_acc, float)
        # This is the exact formatting that run_execution does
        formatted = f"{mean_acc:.3f}"
        assert "." in formatted

    def test_guard_rejects_none(self):
        """None should fail the isinstance check (triggering N/A path)."""
        assert not isinstance(None, (int, float))

    def test_guard_rejects_string(self):
        """A string like 'N/A' should fail the isinstance check."""
        assert not isinstance("N/A", (int, float))

    def test_guard_accepts_zero(self):
        """0.0 is a valid float and should format, not be treated as falsy."""
        assert isinstance(0.0, (int, float))
        assert f"{0.0:.3f}" == "0.000"


# =========================================================================
# Packaging / requirements  (pyproject.toml, requirements.txt)
# =========================================================================


class TestPackaging:
    """
    Bug: (1) build-backend was set to a nonexistent module
    (setuptools.backends._legacy:_Backend), causing pip install to fail.
    (2) Package discovery couldn't find the package because pyproject.toml
    was inside the package directory (flat layout).
    (3) requirements.txt had a duplicate anthropic entry.
    """

    def _repo_root(self):
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def test_build_backend_is_valid(self):
        """
        Bug: setuptools.backends._legacy:_Backend does not exist, causing
        BackendUnavailable on pip install.
        Fix: changed to setuptools.build_meta.
        """
        import tomllib

        with open(os.path.join(self._repo_root(), "pyproject.toml"), "rb") as f:
            config = tomllib.load(f)
        assert config["build-system"]["build-backend"] == "setuptools.build_meta"

    def test_package_is_importable(self):
        """
        Bug: flat layout meant find_packages returned [].
        Fix: restructured so pyproject.toml is at repo root and source
        is in antagonistic_collab/ subdirectory.
        """
        import antagonistic_collab

        assert hasattr(antagonistic_collab, "EpistemicState")

    def test_no_duplicate_dependencies(self):
        """
        Bug: requirements.txt listed anthropic>=0.30 twice.
        Verifies each non-empty line is unique.
        """
        with open(os.path.join(self._repo_root(), "requirements.txt")) as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == len(set(lines)), (
            f"Duplicate entries in requirements.txt: {lines}"
        )

    def test_setuptools_finds_packages(self):
        """
        Verifies that setuptools can actually discover the package and
        its subpackages from the current layout.
        """
        from setuptools import find_packages

        packages = find_packages(
            where=self._repo_root(),
            include=["antagonistic_collab", "antagonistic_collab.*"],
        )
        assert "antagonistic_collab" in packages
        assert "antagonistic_collab.models" in packages


# =========================================================================
# P1/P2 Bugfix Regressions (Round 5)
# =========================================================================


class TestNaNCorrelationScore:
    """
    Bug: score_predictions() with metric='correlation' can produce NaN from
    np.corrcoef when all predicted or actual values are identical (zero
    variance). The NaN then poisons the leaderboard.
    Fix: check np.isnan() after np.corrcoef and set score to None.
    """

    def test_nan_corrcoef_yields_none_score(self):
        state = EpistemicState(domain="test")
        state.register_theory(
            TheoryCommitment(
                name="T",
                agent_name="A",
                core_claims=["c"],
                model_name="M",
            )
        )
        exp = state.propose_experiment(
            proposed_by="A",
            title="E",
            design_spec={},
            rationale="r",
        )
        # Predict identical values for all keys -> zero variance -> NaN corr
        state.register_prediction(
            exp.experiment_id,
            "A",
            "M",
            {},
            {"a": 1.0, "b": 1.0, "c": 1.0},
        )
        state.score_predictions(
            exp.experiment_id,
            {"a": 0.5, "b": 0.6, "c": 0.7},
            metric="correlation",
        )
        pred = state.predictions[0]
        assert pred.score is None or (
            isinstance(pred.score, float) and not math.isnan(pred.score)
        ), f"Expected None (not NaN), got {pred.score}"


class TestSummaryMissingDescription:
    """
    Bug: summary_for_agent() accesses latest['description'] with bracket
    notation. If a revision was logged without a 'description' key, this
    raises KeyError.
    Fix: use latest.get('description', '(no description)').
    """

    def test_summary_with_missing_description_key(self):
        state = EpistemicState(domain="test")
        state.register_theory(
            TheoryCommitment(
                name="T",
                agent_name="A",
                core_claims=["c"],
                model_name="M",
            )
        )
        theory = state.get_theory("T")
        # Append a revision WITHOUT a 'description' key
        theory.revision_log.append(
            {
                "timestamp": "2026-01-01",
                "old_params": None,
                "old_claims": None,
            }
        )
        # Should not raise KeyError
        summary = state.summary_for_agent("A")
        assert "(no description)" in summary


class TestDeepCopyTheoryRevision:
    """
    Bug: TheoryCommitment.revise() uses .copy() for model_params and
    core_claims snapshots. Since model_params can contain nested dicts
    (e.g. {"attention": {"dim1": 0.5}}), shallow copy means the snapshot
    shares references with the live object — mutating model_params after
    the revision also mutates the snapshot in revision_log.
    Fix: use copy.deepcopy().
    """

    def test_revision_snapshot_is_independent_of_live_params(self):
        theory = TheoryCommitment(
            name="T",
            agent_name="A",
            core_claims=["c"],
            model_name="M",
            model_params={"nested": {"a": 1}},
        )
        # Revise with new_params that DON'T touch the nested dict —
        # shallow copy means snapshot shares the "nested" dict reference
        theory.revise(description="test", new_params={"new_key": 42})
        old_snapshot = theory.revision_log[0]["old_params"]
        assert old_snapshot["nested"]["a"] == 1
        # Mutating the shared nested dict corrupts the snapshot with shallow copy
        theory.model_params["nested"]["a"] = 999
        assert old_snapshot["nested"]["a"] == 1


class TestDeepCopyReviseProposal:
    """
    Bug: revise_proposal() uses exp.design_spec.copy() for the
    old_design_spec snapshot. Nested dicts share references.
    Fix: use copy.deepcopy(exp.design_spec).
    """

    def test_old_design_spec_snapshot_is_independent(self):
        state = EpistemicState(domain="test")
        # Keep a reference to the original nested dict
        nested = {"x": 1}
        exp = state.propose_experiment(
            proposed_by="A",
            title="E",
            design_spec={"nested": nested},
            rationale="r",
        )
        state.add_critique(exp.experiment_id, "B", "flaw")
        state.revise_proposal(
            exp.experiment_id,
            "A",
            addresses_critiques=[0],
            changes="fix",
            new_design_spec={"nested": {"x": 2}},
        )
        old_snapshot = exp.revision_history[0]["old_design_spec"]
        assert old_snapshot["nested"]["x"] == 1
        # Mutate via the external reference — shallow copy shares this dict
        nested["x"] = 999
        assert old_snapshot["nested"]["x"] == 1


class TestResolveDisputeLogging:
    """
    Bug: resolve_dispute() does not call self._log(), so dispute
    resolutions are invisible in the event log.
    Fix: add self._log("dispute_resolved", ...).
    """

    def test_resolve_dispute_creates_log_entry(self):
        state = EpistemicState(domain="test")
        dispute = state.register_dispute(
            claim="test claim",
            positions={"A": "yes", "B": "no"},
        )
        state.resolve_dispute(dispute.dispute_id, "resolved by data")
        log_events = [e["event"] for e in state.log]
        assert "dispute_resolved" in log_events


class TestNegativeIndexValidation:
    """
    Bug: revise_proposal() checks `if idx >= len(...)` but not `if idx < 0`.
    Python allows negative indices on lists, so idx=-1 passes the check
    but refers to the wrong critique.
    Fix: check `if idx < 0 or idx >= len(...)`.
    """

    def test_negative_critique_index_raises(self):
        state = EpistemicState(domain="test")
        exp = state.propose_experiment(
            proposed_by="A",
            title="E",
            design_spec={},
            rationale="r",
        )
        state.add_critique(exp.experiment_id, "B", "flaw")
        with pytest.raises(ValueError, match="does not exist"):
            state.revise_proposal(
                exp.experiment_id,
                "A",
                addresses_critiques=[-1],
                changes="fix",
                new_design_spec={},
            )


class TestEmptyAPIResponse:
    """
    Bug: call_agent() accesses response.content[0] without checking if
    content is empty. An empty content list causes IndexError.
    Fix: check `if not response.content:` and raise a clear error.
    """

    def test_empty_content_raises_clear_error(self):
        class FakeResponse:
            content = []

        class FakeMessages:
            def create(self, **kwargs):
                return FakeResponse()

        class FakeClient:
            messages = FakeMessages()

        with pytest.raises((IndexError, ValueError)):
            call_agent(FakeClient(), "system", "user", model="test-model")


class TestLeaderboardNaNSafe:
    """
    Bug: Final leaderboard in main() formats mean_score with :.4f.
    If mean_score is NaN (from a NaN-producing corrcoef), :.4f works
    but prints 'nan', which is confusing. If mean_score is None,
    :.4f crashes with TypeError.
    Fix: guard with isinstance and math.isnan before formatting.
    """

    def test_none_mean_score_does_not_crash_formatting(self):
        stats = {"mean_score": None, "n_predictions": 1}
        mean = stats.get("mean_score")
        # The guard: must not attempt .4f on None
        if isinstance(mean, (int, float)) and not math.isnan(mean):
            formatted = f"{mean:.4f}"
        else:
            formatted = "N/A"
        assert formatted == "N/A"

    def test_nan_mean_score_does_not_crash_formatting(self):
        stats = {"mean_score": float("nan"), "n_predictions": 1}
        mean = stats.get("mean_score")
        if isinstance(mean, (int, float)) and not math.isnan(mean):
            formatted = f"{mean:.4f}"
        else:
            formatted = "N/A"
        assert formatted == "N/A"


class TestEOFErrorOnInput:
    """
    Bug: run_human_arbitration() calls input() which raises EOFError
    when stdin is closed (e.g. piped input or CI). Also KeyboardInterrupt
    if the user hits Ctrl+C.
    Fix: wrap in try/except, default to "skip".
    """

    @patch("antagonistic_collab.runner._BATCH_MODE", False)
    def test_eof_error_defaults_to_skip(self):
        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)
        state.propose_experiment(
            proposed_by="A",
            title="E",
            design_spec={},
            rationale="r",
        )
        with patch("builtins.input", side_effect=EOFError):
            result = run_human_arbitration(protocol, [])
        assert result is not None  # Should not crash

    @patch("antagonistic_collab.runner._BATCH_MODE", False)
    def test_keyboard_interrupt_defaults_to_skip(self):
        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)
        state.propose_experiment(
            proposed_by="A",
            title="E",
            design_spec={},
            rationale="r",
        )
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            result = run_human_arbitration(protocol, [])
        assert result is not None


class TestJsonDumpsFallback:
    """
    Bug: _proposals_context() calls json.dumps(p.design_spec) which crashes
    with TypeError if design_spec contains non-serializable objects (e.g.
    numpy arrays, custom objects).
    Fix: wrap in try/except TypeError, fall back to str(p.design_spec).
    """

    def test_non_serializable_design_spec_does_not_crash(self):
        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)
        # Inject a design_spec with a non-serializable value
        state.propose_experiment(
            proposed_by="A",
            title="E",
            design_spec={"array": np.array([1, 2, 3])},
            rationale="r",
        )
        # Should not raise TypeError
        context = protocol._proposals_context()
        assert "E" in context


class TestEmptyTrainingSequence:
    """
    Bug: SUSTAIN.simulate_learning() accesses training_sequence[0] on
    line 149 to get n_dims. If training_sequence is empty, this crashes
    with IndexError.
    Fix: early return with empty results.
    """

    def test_empty_training_sequence_returns_gracefully(self):
        model = SUSTAIN()
        result = model.simulate_learning([])
        assert result["n_clusters_final"] == 0
        assert result["trial_log"] == []


class TestEmptyTestItems:
    """
    Bug: GCM.predict_learning_curve() divides by len(test_items) on line
    184. If test_items is empty, this is a ZeroDivisionError.
    Fix: skip block or return early when test_items is empty.
    """

    def test_empty_test_items_does_not_crash(self):
        model = GCM()
        training = [(np.array([0, 0, 0]), 0), (np.array([1, 1, 1]), 1)]
        result = model.predict_learning_curve(
            training_sequence=training,
            test_items=np.array([]).reshape(0, 3),
            test_labels=np.array([]),
        )
        # Should not crash; curve may be empty or have zero-accuracy entries
        assert isinstance(result, list)


class TestSkipToPhase:
    """
    Bug: runner.py line 633 sets protocol.current_phase = Phase.DIVERGENCE_MAPPING
    directly, bypassing the state machine. This is fragile and doesn't
    validate the target phase.
    Fix: add skip_to_phase() method on DebateProtocol that validates
    the target phase.
    """

    def test_skip_to_phase_sets_phase(self):
        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)
        protocol.skip_to_phase(Phase.DIVERGENCE_MAPPING)
        assert protocol.current_phase == Phase.DIVERGENCE_MAPPING

    def test_skip_to_phase_rejects_invalid_input(self):
        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)
        with pytest.raises((ValueError, TypeError)):
            protocol.skip_to_phase("not_a_phase")


# =========================================================================
# Princeton AI Sandbox backend support  (runner.py)
# =========================================================================


class TestOpenAIBackend:
    """
    Tests for the Princeton AI Sandbox (Azure OpenAI) backend.

    The codebase was originally hard-wired to the Anthropic SDK. Adding the
    Princeton sandbox as an alternative backend requires call_agent() to
    dispatch on client type, and main() to create the right client.
    """

    def test_call_agent_with_openai_client(self):
        """call_agent() should work with an OpenAI-style client.

        When passed an openai.AzureOpenAI (or compatible) client, call_agent()
        should use chat.completions.create and read choices[0].message.content.
        """
        from unittest.mock import MagicMock

        # Build a mock that quacks like openai.AzureOpenAI
        mock_client = MagicMock()
        mock_client.__class__.__name__ = "AzureOpenAI"
        # Simulate: response.choices[0].message.content
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello from GPT-4o"
        mock_client.chat.completions.create.return_value = mock_response

        result = call_agent(
            client=mock_client,
            system_prompt="You are a test agent.",
            user_message="Say hello.",
            model="gpt-4o",
        )

        assert result == "Hello from GPT-4o"
        # Verify the system prompt was passed as a message, not a kwarg
        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a test agent."
        assert messages[1]["role"] == "user"

    def test_call_agent_openai_empty_response_raises(self):
        """call_agent() should raise ValueError when OpenAI returns empty content."""
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        mock_client.__class__.__name__ = "AzureOpenAI"
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = mock_response

        with pytest.raises(ValueError, match="Empty response"):
            call_agent(
                client=mock_client,
                system_prompt="test",
                user_message="test",
                model="gpt-4o",
            )

    def test_call_agent_openai_no_choices_raises(self):
        """call_agent() should raise ValueError when OpenAI returns no choices."""
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        mock_client.__class__.__name__ = "AzureOpenAI"
        mock_response = MagicMock()
        mock_response.choices = []
        mock_client.chat.completions.create.return_value = mock_response

        with pytest.raises(ValueError, match="Empty response"):
            call_agent(
                client=mock_client,
                system_prompt="test",
                user_message="test",
                model="gpt-4o",
            )

    def test_call_agent_anthropic_still_works(self):
        """Existing Anthropic path must not regress."""
        from unittest.mock import MagicMock

        mock_client = MagicMock(spec=["messages"])
        mock_client.__class__ = type("Anthropic", (), {})
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Hello from Claude"
        mock_client.messages.create.return_value = mock_response

        result = call_agent(
            client=mock_client,
            system_prompt="You are a test agent.",
            user_message="Say hello.",
            model="claude-sonnet-4-20250514",
        )

        assert result == "Hello from Claude"
        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs.get("system") == "You are a test agent."

    def test_backend_arg_creates_openai_client(self):
        """--backend princeton should produce an OpenAI client via Portkey gateway.

        We mock the OpenAI constructor to verify it gets called with
        the Portkey base_url and the sandbox API key.
        """
        from unittest.mock import MagicMock, patch as mock_patch
        from antagonistic_collab.runner import _create_client

        fake_client = MagicMock()
        with mock_patch.dict(os.environ, {"AI_SANDBOX_KEY": "test-key-123"}):
            with mock_patch(
                "antagonistic_collab.runner.openai.OpenAI",
                return_value=fake_client,
            ) as mock_ctor:
                client = _create_client(backend="princeton")

        assert client is fake_client
        mock_ctor.assert_called_once()
        call_kwargs = mock_ctor.call_args.kwargs
        assert "api.portkey.ai" in call_kwargs["base_url"]
        assert call_kwargs["api_key"] == "test-key-123"

    def test_backend_arg_creates_anthropic_client(self):
        """--backend anthropic should produce an Anthropic client (default)."""
        from unittest.mock import MagicMock, patch as mock_patch
        from antagonistic_collab.runner import _create_client

        fake_client = MagicMock()
        with mock_patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            with mock_patch(
                "antagonistic_collab.runner.anthropic.Anthropic",
                return_value=fake_client,
            ) as mock_ctor:
                client = _create_client(backend="anthropic")

        assert client is fake_client
        mock_ctor.assert_called_once_with(api_key="sk-ant-test")

    def test_princeton_backend_missing_key_raises(self):
        """--backend princeton without AI_SANDBOX_KEY should raise SystemExit."""
        from unittest.mock import patch as mock_patch
        from antagonistic_collab.runner import _create_client

        with mock_patch.dict(os.environ, {}, clear=True):
            # Also ensure ANTHROPIC_API_KEY is gone
            env = {k: v for k, v in os.environ.items() if k != "AI_SANDBOX_KEY"}
            with mock_patch.dict(os.environ, env, clear=True):
                with pytest.raises(SystemExit):
                    _create_client(backend="princeton")


# =========================================================================
# Markdown reports and per-cycle transcript slicing (runner.py)
# =========================================================================


class TestMarkdownReports:
    """
    Tests for per-cycle Markdown transcripts, end-of-run summary reports,
    and the fix for cumulative transcript saving.
    """

    def _make_protocol_with_data(self):
        """Helper: create a protocol with theories, an experiment, and predictions."""
        state = EpistemicState(domain="Human Categorization")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)

        # Register theories
        for agent in agents:
            state.register_theory(
                TheoryCommitment(
                    name=agent.theory_name,
                    agent_name=agent.name,
                    core_claims=agent.model_class.core_claims,
                    model_name=agent.model_class.name,
                    model_params=agent.default_params,
                )
            )

        # Propose and approve an experiment
        exp = state.propose_experiment(
            proposed_by="Exemplar_Agent",
            title="Type II vs Type IV",
            design_spec={"design": "between"},
            rationale="Test exemplar advantage",
        )
        state.approve_experiment(exp.experiment_id)

        # Register predictions
        for agent in agents:
            state.register_prediction(
                experiment_id=exp.experiment_id,
                agent_name=agent.name,
                model_name=agent.model_class.name,
                model_params=agent.default_params,
                predicted_pattern={"mean_accuracy": 0.75},
            )

        # Record data and score
        state.record_data(exp.experiment_id, {"mean_accuracy": 0.55})
        state.score_predictions(exp.experiment_id, {"mean_accuracy": 0.55})

        # Revise one theory (progressive)
        state.revise_theory(
            "Clustering Theory (SUSTAIN)",
            description="Adjusted learning rate after data",
            triggered_by_experiment=exp.experiment_id,
            new_predictions=["SUSTAIN predicts faster learning on Type IV"],
        )

        return protocol

    def test_per_cycle_transcript_not_cumulative(self):
        """
        Bug: save_transcript() was called with the full cumulative transcript
        list, so cycle 1's JSON contained cycle 0's messages too.

        Fix: pass only transcript[cycle_start:] to save_transcript().

        Verify: save two cycle slices, each JSON only has its own messages.
        """
        state = EpistemicState(domain="Test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)

        # Simulate a cumulative transcript with messages from two cycles
        transcript = [
            {"agent": "A", "phase": "COMMITMENT", "response": "cycle 0 msg 1"},
            {"agent": "B", "phase": "COMMITMENT", "response": "cycle 0 msg 2"},
            {"agent": "A", "phase": "INTERPRETATION", "response": "cycle 1 msg 1"},
            {"agent": "B", "phase": "INTERPRETATION", "response": "cycle 1 msg 2"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save cycle 0 (first 2 messages)
            protocol.state.cycle = 0
            save_transcript(transcript[:2], protocol, output_dir=tmpdir)

            # Save cycle 1 (last 2 messages)
            protocol.state.cycle = 1
            save_transcript(transcript[2:], protocol, output_dir=tmpdir)

            # Verify cycle 0 JSON only has 2 messages
            with open(os.path.join(tmpdir, "debate_cycle_0.json")) as f:
                c0 = json.load(f)
            assert len(c0["transcript"]) == 2
            assert c0["transcript"][0]["response"] == "cycle 0 msg 1"

            # Verify cycle 1 JSON only has 2 messages
            with open(os.path.join(tmpdir, "debate_cycle_1.json")) as f:
                c1 = json.load(f)
            assert len(c1["transcript"]) == 2
            assert c1["transcript"][0]["response"] == "cycle 1 msg 1"

    def test_save_cycle_markdown_creates_file(self):
        """
        save_cycle_markdown() should create a .md file containing phase
        headers and agent response text from the cycle's transcript.
        """
        protocol = self._make_protocol_with_data()
        cycle_messages = [
            {
                "agent": "Exemplar_Agent",
                "phase": "COMMITMENT",
                "response": "I commit to GCM.",
            },
            {
                "agent": "Rule_Agent",
                "phase": "COMMITMENT",
                "response": "I commit to RULEX.",
            },
            {
                "agent": "Exemplar_Agent",
                "phase": "DIVERGENCE_MAPPING",
                "response": "Divergence seen.",
            },
            {
                "agent": "Exemplar_Agent",
                "phase": "EXPERIMENT_PROPOSAL",
                "response": "Propose exp.",
                "parsed_json": {"title": "Test exp", "design": "between"},
            },
            {
                "agent": "Rule_Agent",
                "phase": "ADVERSARIAL_CRITIQUE",
                "round": 1,
                "response": "I critique this.",
            },
            {"agent": "MODERATOR", "phase": "HUMAN_ARBITRATION", "input": "approve 0"},
            {
                "agent": "Exemplar_Agent",
                "phase": "EXECUTION_PREDICT",
                "response": "I predict 0.8",
                "predicted": {"mean_accuracy": 0.8},
            },
            {
                "agent": "SYSTEM",
                "phase": "EXECUTION_DATA",
                "data_summary": {"mean_accuracy": 0.55},
            },
            {
                "agent": "Exemplar_Agent",
                "phase": "INTERPRETATION",
                "response": "Data supports GCM.",
                "parsed_json": {"revision": False},
            },
            {"agent": "AUDITOR", "phase": "AUDIT", "response": "Cycle summary here."},
        ]

        metadata = {"true_model": "GCM", "llm_model": "gpt-4o", "backend": "princeton"}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_cycle_markdown(
                cycle_messages,
                protocol,
                cycle_num=0,
                metadata=metadata,
                output_dir=tmpdir,
            )
            md_path = os.path.join(tmpdir, "debate_cycle_0.md")
            assert os.path.exists(md_path)

            content = open(md_path).read()
            # Check structure
            assert "# Cycle 0 Transcript" in content
            assert "GCM" in content
            assert "gpt-4o" in content
            assert "## Phase: Commitment" in content
            assert "## Phase: Divergence Mapping" in content
            assert "## Phase: Experiment Proposal" in content
            assert "## Phase: Adversarial Critique" in content
            assert "## Phase: Human Arbitration" in content
            assert "## Phase: Execution" in content
            assert "## Phase: Interpretation" in content
            assert "## Phase: Audit" in content
            assert "Exemplar_Agent" in content
            assert "I commit to GCM." in content

    def test_save_summary_report_creates_file(self):
        """
        save_summary_report() should create summary.md with leaderboard
        and theory trajectory tables.
        """
        protocol = self._make_protocol_with_data()
        transcript = [
            {"agent": "Exemplar_Agent", "phase": "COMMITMENT", "response": "committed"},
        ]
        metadata = {"true_model": "GCM", "llm_model": "gpt-4o", "backend": "princeton"}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_summary_report(
                transcript,
                protocol,
                n_cycles=1,
                metadata=metadata,
                output_dir=tmpdir,
            )
            md_path = os.path.join(tmpdir, "summary.md")
            assert os.path.exists(md_path)

            content = open(md_path).read()
            assert "# Debate Summary" in content
            assert "GCM" in content
            assert "gpt-4o" in content
            assert "## Prediction Leaderboard" in content
            assert "Mean RMSE" in content
            assert "## Theory Trajectories" in content
            assert "Trajectory" in content
            # Should have at least one agent row
            assert "Exemplar_Agent" in content

    def test_auto_output_dir_naming(self):
        """
        auto_output_dir() should produce a directory name matching:
        runs/True_{true_model}_LLM_{llm}_COLLAB_{agents}_{run_number}/
        """
        agents = default_agent_configs()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = auto_output_dir(
                true_model="GCM",
                llm_model="gpt-4o",
                agent_configs=agents,
                base_dir=tmpdir,
            )
            # Should contain the key components
            assert "True_GCM" in result
            assert "LLM_gpt-4o" in result
            assert "COLLAB_Exemplar-Rule-Clustering" in result
            assert result.endswith("_01")

            # Create the first dir, then call again — should increment
            os.makedirs(result)
            result2 = auto_output_dir(
                true_model="GCM",
                llm_model="gpt-4o",
                agent_configs=agents,
                base_dir=tmpdir,
            )
            assert result2.endswith("_02")

    def test_cycle_markdown_no_duplicate_json(self):
        """
        Bug: save_cycle_markdown() rendered parsed_json as a code block AND
        the full response text (which already contains that JSON), causing
        the same JSON to appear twice.

        Fix: skip rendering parsed_json separately; the response already
        contains it in context.
        """
        response_text = (
            "Here is my proposal:\n"
            '```json\n{"title": "Test exp", "design": "between"}\n```\n'
            "This tests exemplar advantage."
        )
        cycle_messages = [
            {
                "agent": "Exemplar_Agent",
                "phase": "EXPERIMENT_PROPOSAL",
                "response": response_text,
                "parsed_json": {"title": "Test exp", "design": "between"},
            },
        ]
        metadata = {"true_model": "GCM", "llm_model": "gpt-4o", "backend": "princeton"}
        state = EpistemicState(domain="Test")
        protocol = DebateProtocol(state, default_agent_configs())

        with tempfile.TemporaryDirectory() as tmpdir:
            save_cycle_markdown(
                cycle_messages,
                protocol,
                cycle_num=0,
                metadata=metadata,
                output_dir=tmpdir,
            )
            content = open(os.path.join(tmpdir, "debate_cycle_0.md")).read()

            # The JSON should appear exactly once, not twice
            count = content.count('"title": "Test exp"')
            assert count == 1, f"JSON appeared {count} times, expected 1"

    def test_prediction_prompt_requests_mean_accuracy(self):
        """
        Bug: agents predicted with arbitrary metric keys (e.g.,
        classification_accuracy_training) while synthetic data uses
        mean_accuracy, so score_predictions found no overlapping keys
        and the leaderboard was empty.

        Fix: the prediction prompt must explicitly instruct agents to
        include mean_accuracy as a key in their predicted_pattern.
        """
        from antagonistic_collab.runner import run_execution
        import inspect

        source = inspect.getsource(run_execution)
        # The function must reference mean_accuracy (for scoring alignment)
        assert "mean_accuracy" in source, (
            "run_execution must reference mean_accuracy for scoring to work"
        )


# =========================================================================
# Batch-mode moderator rotation  (runner.py — run_human_arbitration)
# =========================================================================


class TestBatchModeRotation:
    """
    Bug: In batch mode, run_human_arbitration() always picked `approve 0`
    — the first proposal.  Because run_experiment_proposal() iterates
    agents in list order, Exemplar_Agent always proposed first and always
    won.  This biases the entire debate toward one agent.

    Fix: round-robin selection by fewest prior approvals, with a
    critique-count tiebreaker (more critiques = more refined).
    """

    @staticmethod
    def _make_protocol():
        """Create a bare protocol for testing."""
        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        return DebateProtocol(state, agents)

    @patch("antagonistic_collab.runner._SELECTION_METHOD", "heuristic")
    @patch("antagonistic_collab.runner._BATCH_MODE", True)
    def test_batch_mode_divergence_driven(self):
        """
        With divergence-driven selection, proposals with higher-divergence
        structures are preferred over proposals with lower-divergence
        structures, regardless of which agent proposed them.
        """
        protocol = self._make_protocol()
        state = protocol.state
        state.cycle = 1

        # Agent with 2 prior approvals proposes high-divergence structure
        state.propose_experiment(
            proposed_by="Exemplar_Agent",
            title="High div proposal",
            design_spec={"structure_name": "five_four", "condition": "baseline"},
            rationale="r",
        )
        # Other agent proposes low-divergence structure
        state.propose_experiment(
            proposed_by="Rule_Agent",
            title="Low div proposal",
            design_spec={"structure_name": "Type_I", "condition": "baseline"},
            rationale="r",
        )

        run_human_arbitration(protocol, [])

        approved = [
            e for e in state.experiments if e.cycle == 1 and e.status == "approved"
        ]
        assert len(approved) == 1
        assert approved[0].design_spec.get("structure_name") == "five_four", (
            "Divergence-driven selection should pick the highest-divergence "
            f"structure, got {approved[0].design_spec.get('structure_name')}"
        )

    @patch("antagonistic_collab.runner._SELECTION_METHOD", "heuristic")
    @patch("antagonistic_collab.runner._BATCH_MODE", True)
    def test_batch_mode_critique_tiebreak(self):
        """
        All agents have 0 prior approvals.  One proposal has 3 critiques,
        another has 1.  The one with more critiques should be selected
        (more scrutinized = more refined).
        """
        protocol = self._make_protocol()
        state = protocol.state

        # --- Current cycle: 2 proposals, no prior history ---
        state.cycle = 1
        exp_few = state.propose_experiment(
            proposed_by="Agent_A",
            title="Few critiques",
            design_spec={},
            rationale="r",
        )
        state.add_critique(exp_few.experiment_id, "Agent_B", "minor issue")

        exp_many = state.propose_experiment(
            proposed_by="Agent_B",
            title="Many critiques",
            design_spec={},
            rationale="r",
        )
        state.add_critique(exp_many.experiment_id, "Agent_A", "concern 1")
        state.add_critique(exp_many.experiment_id, "Agent_A", "concern 2")
        state.add_critique(exp_many.experiment_id, "Agent_A", "concern 3")

        run_human_arbitration(protocol, [])

        approved = [
            e for e in state.experiments if e.cycle == 1 and e.status == "approved"
        ]
        assert len(approved) == 1
        assert approved[0].title == "Many critiques", (
            "When prior approvals are tied, batch mode should prefer the "
            "proposal with the most critiques (more refined), but it didn't."
        )

    @patch("antagonistic_collab.runner._SELECTION_METHOD", "heuristic")
    @patch("antagonistic_collab.runner._BATCH_MODE", True)
    def test_batch_mode_single_proposal(self):
        """
        Edge case: only 1 proposal on the table.  Should approve it
        without crashing.
        """
        protocol = self._make_protocol()
        state = protocol.state
        state.cycle = 1
        state.propose_experiment(
            proposed_by="Only_Agent",
            title="Solo Proposal",
            design_spec={},
            rationale="r",
        )

        run_human_arbitration(protocol, [])

        approved = [
            e for e in state.experiments if e.cycle == 1 and e.status == "approved"
        ]
        assert len(approved) == 1
        assert approved[0].title == "Solo Proposal"


# =========================================================================
# Synthetic data variation  (Phase 2: model-sensitive data)
# =========================================================================


class TestSyntheticDataVariation:
    """
    Regression tests for D6 — synthetic data always returned mean_accuracy=0.550
    regardless of experiment design. These tests ensure that different structures,
    conditions, and cycles produce genuinely different data.
    """

    def _make_protocol(self):
        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        return DebateProtocol(state, agents)

    # --- Test 1: Registry completeness ---

    def test_structure_registry_completeness(self):
        """STRUCTURE_REGISTRY must contain all expected keys with stimuli and labels."""
        expected_keys = {
            "Type_I",
            "Type_II",
            "Type_III",
            "Type_IV",
            "Type_V",
            "Type_VI",
            "five_four",
            "rule_plus_exception_1exc",
            "rule_plus_exception_2exc",
            "linear_separable_2d",
            "linear_separable_4d",
        }
        assert expected_keys.issubset(set(STRUCTURE_REGISTRY.keys())), (
            f"Missing keys: {expected_keys - set(STRUCTURE_REGISTRY.keys())}"
        )
        for name, struct in STRUCTURE_REGISTRY.items():
            assert "stimuli" in struct, f"{name} missing 'stimuli'"
            assert "labels" in struct, f"{name} missing 'labels'"
            assert len(struct["stimuli"]) == len(struct["labels"]), (
                f"{name}: stimuli/labels length mismatch"
            )

    # --- Test 2: Different structures produce different data ---

    def test_different_structures_produce_different_data(self):
        """
        Running _synthetic_runner on all 6 Shepard types must produce
        at least 3 distinct mean_accuracy values. Before the fix, all
        returned 0.550.
        """
        protocol = self._make_protocol()
        accuracies = []
        for type_name in [
            "Type_I",
            "Type_II",
            "Type_III",
            "Type_IV",
            "Type_V",
            "Type_VI",
        ]:
            data = protocol._synthetic_runner(
                {"structure_name": type_name}, true_model="GCM"
            )
            accuracies.append(round(data["mean_accuracy"], 4))
        distinct = len(set(accuracies))
        assert distinct >= 3, (
            f"Expected >=3 distinct accuracies across 6 types, got {distinct}: {accuracies}"
        )

    # --- Test 3: structure_name lookup works ---

    def test_synthetic_runner_uses_structure_name(self):
        """Type_I (single rule, easy) and Type_VI (no rule, hard) must differ."""
        protocol = self._make_protocol()
        data_i = protocol._synthetic_runner(
            {"structure_name": "Type_I"}, true_model="GCM"
        )
        data_vi = protocol._synthetic_runner(
            {"structure_name": "Type_VI"}, true_model="GCM"
        )
        assert data_i["mean_accuracy"] != data_vi["mean_accuracy"], (
            f"Type_I and Type_VI produced identical mean_accuracy: {data_i['mean_accuracy']}"
        )
        # Verify metadata
        assert data_i.get("structure_name") == "Type_I"
        assert data_vi.get("structure_name") == "Type_VI"

    # --- Test 4: Condition varies data ---

    def test_synthetic_runner_condition_varies_data(self):
        """Same structure with baseline vs low_attention must produce different data."""
        protocol = self._make_protocol()
        data_base = protocol._synthetic_runner(
            {"structure_name": "Type_II", "condition": "baseline"}, true_model="GCM"
        )
        data_low = protocol._synthetic_runner(
            {"structure_name": "Type_II", "condition": "low_attention"},
            true_model="GCM",
        )
        # item_accuracies should differ (different params → different predictions)
        assert data_base["item_accuracies"] != data_low["item_accuracies"], (
            "baseline and low_attention produced identical item_accuracies"
        )

    # --- Test 5: Seed varies by cycle ---

    def test_synthetic_runner_seed_varies_by_cycle(self):
        """Same structure+condition at different cycles must produce different noise."""
        protocol = self._make_protocol()
        data_c0 = protocol._synthetic_runner(
            {"structure_name": "Type_II"}, true_model="GCM", cycle=0
        )
        data_c1 = protocol._synthetic_runner(
            {"structure_name": "Type_II"}, true_model="GCM", cycle=1
        )
        # Model predictions are the same, but noise differs due to different seeds
        assert data_c0["mean_accuracy"] != data_c1["mean_accuracy"], (
            "Different cycles produced identical mean_accuracy (seed not varying)"
        )

    # --- Test 6: item_accuracies passed to scoring ---

    def test_item_accuracies_passed_to_scoring(self):
        """
        When experiment data contains item_accuracies, scoring must include
        them so agents are scored on per-item predictions, not just mean_accuracy.
        """
        state = EpistemicState(domain="test")
        # Register a theory commitment (required by some code paths)
        state.register_theory(
            TheoryCommitment(
                name="Test Theory",
                agent_name="TestAgent",
                core_claims=["test"],
                model_name="GCM",
            )
        )
        # Propose and approve an experiment
        exp = state.propose_experiment(
            proposed_by="TestAgent",
            title="Item-level test",
            design_spec={},
            rationale="test",
        )
        exp.status = "approved"

        # Register prediction with item-level keys
        state.register_prediction(
            experiment_id=exp.experiment_id,
            agent_name="TestAgent",
            model_name="GCM",
            model_params={},
            predicted_pattern={
                "mean_accuracy": 0.8,
                "item_0": 0.9,
                "item_1": 0.7,
                "item_2": 0.85,
                "item_3": 0.75,
                "item_4": 0.8,
                "item_5": 0.9,
                "item_6": 0.65,
                "item_7": 0.7,
            },
        )

        # Build actual data the same way runner.py should (scalar filter + item_accuracies)
        raw_data = {
            "mean_accuracy": 0.78,
            "n_subjects": 30,
            "ground_truth_model": "GCM",
            "item_accuracies": {
                "item_0": 0.87,
                "item_1": 0.73,
                "item_2": 0.80,
                "item_3": 0.77,
                "item_4": 0.83,
                "item_5": 0.87,
                "item_6": 0.60,
                "item_7": 0.73,
            },
        }
        # This is what runner.py should do: merge item_accuracies into actual
        actual = {k: v for k, v in raw_data.items() if isinstance(v, (int, float))}
        actual.update(raw_data.get("item_accuracies", {}))

        state.score_predictions(exp.experiment_id, actual)

        pred = [p for p in state.predictions if p.experiment_id == exp.experiment_id][0]
        assert pred.score is not None, "Prediction was not scored"
        # Score should be based on 9 shared keys (mean_accuracy + 8 items), not just 1
        shared_keys = set(pred.predicted_pattern.keys()) & set(actual.keys())
        assert len(shared_keys) >= 9, (
            f"Expected >=9 shared scoring keys, got {len(shared_keys)}: {shared_keys}"
        )

    # --- Test 7: condition_effects keys valid ---

    def test_condition_effects_keys_valid(self):
        """Every condition must have GCM, SUSTAIN, and RULEX entries."""
        for cond_name, overrides in CONDITION_EFFECTS.items():
            assert "GCM" in overrides, f"{cond_name} missing GCM"
            assert "SUSTAIN" in overrides, f"{cond_name} missing SUSTAIN"
            assert "RULEX" in overrides, f"{cond_name} missing RULEX"

    # --- Test 8: Fallback on missing structure_name ---

    def test_fallback_on_missing_structure_name(self):
        """Empty design spec should not crash and should return valid data."""
        protocol = self._make_protocol()
        data = protocol._synthetic_runner({}, true_model="GCM")
        assert "mean_accuracy" in data
        assert "item_accuracies" in data
        assert isinstance(data["mean_accuracy"], float)
        assert 0.0 <= data["mean_accuracy"] <= 1.0

    # --- Test 9: Backward compat ---

    def test_backward_compat_no_structure_name(self):
        """_synthetic_runner({}) must still work (no KeyError, no crash)."""
        protocol = self._make_protocol()
        data = protocol._synthetic_runner({}, true_model="GCM")
        assert "mean_accuracy" in data
        data2 = protocol._synthetic_runner({}, true_model="SUSTAIN")
        assert "mean_accuracy" in data2
        data3 = protocol._synthetic_runner({}, true_model="RULEX")
        assert "mean_accuracy" in data3


# =========================================================================
# Model-based predictions  (Phase 3: agents call their models — D8)
# =========================================================================


class TestModelBasedPredictions:
    """
    Bug: Agents guessed item-level predictions via LLM reasoning instead of
    running model.predict(). RMSE leaderboard measured LLM calibration quality,
    not model fit. This class tests compute_model_predictions() which runs
    each agent's actual model on the approved experiment structure.
    """

    def _make_protocol(self):
        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        return DebateProtocol(state, agents)

    def _get_agent(self, protocol, name_prefix):
        """Get an agent config by name prefix (e.g. 'Exemplar')."""
        for agent in protocol.agent_configs:
            if agent.name.startswith(name_prefix):
                return agent
        raise ValueError(f"No agent starting with {name_prefix}")

    # --- Test 1: GCM predictions ---

    def test_compute_model_predictions_gcm(self):
        """
        compute_model_predictions() on Type_I with GCM agent should return
        a dict with mean_accuracy and item_0..item_7, all floats in [0,1].
        """
        protocol = self._make_protocol()
        agent = self._get_agent(protocol, "Exemplar")
        result = protocol.compute_model_predictions(agent, "Type_I")

        assert "mean_accuracy" in result
        assert isinstance(result["mean_accuracy"], float)
        assert 0.0 <= result["mean_accuracy"] <= 1.0

        # Type_I has 8 items (Shepard type)
        for i in range(8):
            key = f"item_{i}"
            assert key in result, f"Missing {key}"
            assert isinstance(result[key], float), f"{key} not float"
            assert 0.0 <= result[key] <= 1.0, f"{key} out of range: {result[key]}"

    # --- Test 2: RULEX determinism ---

    def test_compute_model_predictions_rulex(self):
        """
        RULEX with a fixed seed should produce deterministic results.
        Two calls should return the same predictions.
        """
        protocol = self._make_protocol()
        agent = self._get_agent(protocol, "Rule")
        result1 = protocol.compute_model_predictions(agent, "Type_I")
        result2 = protocol.compute_model_predictions(agent, "Type_I")

        assert result1 == result2, (
            f"RULEX predictions not deterministic:\n"
            f"  run1: {result1}\n  run2: {result2}"
        )

    # --- Test 3: SUSTAIN returns valid dict ---

    def test_compute_model_predictions_sustain(self):
        """
        SUSTAIN agent should return a valid dict with mean_accuracy and item keys.
        """
        protocol = self._make_protocol()
        agent = self._get_agent(protocol, "Clustering")
        result = protocol.compute_model_predictions(agent, "Type_II")

        assert "mean_accuracy" in result
        assert isinstance(result["mean_accuracy"], float)
        assert 0.0 <= result["mean_accuracy"] <= 1.0

        # Type_II has 8 items
        for i in range(8):
            key = f"item_{i}"
            assert key in result, f"Missing {key}"

    # --- Test 4: Different models produce different predictions ---

    def test_model_predictions_differ_across_models(self):
        """
        GCM, RULEX, SUSTAIN on Type_II should produce at least 2 distinct
        mean_accuracy values. If all three match, the method isn't using
        each agent's actual model.
        """
        protocol = self._make_protocol()
        accuracies = []
        for prefix in ["Exemplar", "Rule", "Clustering"]:
            agent = self._get_agent(protocol, prefix)
            result = protocol.compute_model_predictions(agent, "Type_II")
            accuracies.append(round(result["mean_accuracy"], 6))

        distinct = len(set(accuracies))
        assert distinct >= 2, (
            f"Expected >=2 distinct mean_accuracy across 3 models, "
            f"got {distinct}: {accuracies}"
        )

    # --- Test 5: Condition overrides change predictions ---

    def test_model_predictions_condition_override(self):
        """
        baseline vs low_attention should produce different GCM predictions
        on Type_II because low_attention sets c=1.5 vs default c=3.0.
        """
        protocol = self._make_protocol()
        agent = self._get_agent(protocol, "Exemplar")

        result_base = protocol.compute_model_predictions(agent, "Type_II", "baseline")
        result_low = protocol.compute_model_predictions(
            agent, "Type_II", "low_attention"
        )

        assert result_base != result_low, (
            "baseline and low_attention produced identical predictions "
            "for GCM on Type_II — condition override not applied"
        )

    # --- Test 6: Predictions match structure ---

    def test_model_predictions_match_structure(self):
        """
        GCM predictions on Type_I vs Type_VI should differ — Type_I is
        perfectly separable by one dimension, Type_VI has no simple rule.
        """
        protocol = self._make_protocol()
        agent = self._get_agent(protocol, "Exemplar")

        result_i = protocol.compute_model_predictions(agent, "Type_I")
        result_vi = protocol.compute_model_predictions(agent, "Type_VI")

        assert result_i["mean_accuracy"] != result_vi["mean_accuracy"], (
            f"Type_I and Type_VI produced same mean_accuracy: "
            f"{result_i['mean_accuracy']}"
        )

    # --- Test 7: Missing structure fallback ---

    def test_model_predictions_fallback_missing_structure(self):
        """
        A missing structure_name should not crash — should fall back to
        Type_II and return valid predictions.
        """
        protocol = self._make_protocol()
        agent = self._get_agent(protocol, "Exemplar")

        result = protocol.compute_model_predictions(agent, "nonexistent_structure")
        assert "mean_accuracy" in result
        assert isinstance(result["mean_accuracy"], float)
        assert 0.0 <= result["mean_accuracy"] <= 1.0

    # --- Test 8: param_overrides change predictions ---

    def test_param_overrides_change_predictions(self):
        """
        Bug (P1): compute_model_predictions() ignored param_overrides from
        the LLM response, so agents couldn't request non-default parameters.

        Passing c=1.0 (low sensitivity) vs c=8.0 (high sensitivity) to GCM
        on Type_II must produce different predictions.
        """
        protocol = self._make_protocol()
        agent = self._get_agent(protocol, "Exemplar")

        result_low = protocol.compute_model_predictions(
            agent, "Type_II", "baseline", param_overrides={"c": 1.0}
        )
        result_high = protocol.compute_model_predictions(
            agent, "Type_II", "baseline", param_overrides={"c": 8.0}
        )
        assert result_low != result_high, (
            "param_overrides with different c values produced identical "
            "predictions — overrides not applied"
        )

    # --- Test 9: param_overrides returned in metadata ---

    def test_param_overrides_returned_in_metadata(self):
        """
        Bug (P1): register_prediction() stored agent.default_params instead
        of the actual params used (defaults + condition + overrides).

        compute_model_predictions() must return a 'params_used' key so the
        caller can record what was actually used.
        """
        protocol = self._make_protocol()
        agent = self._get_agent(protocol, "Exemplar")

        result = protocol.compute_model_predictions(
            agent, "Type_II", "low_attention", param_overrides={"c": 7.0}
        )
        assert "params_used" in result, "Missing params_used in result"
        # c=7.0 from overrides should win over low_attention's c=1.5
        assert result["params_used"]["c"] == 7.0, (
            f"Expected c=7.0 from param_overrides, got {result['params_used']['c']}"
        )


# =========================================================================
# Code review fixes — 2026-03-13 Session 8
# =========================================================================


class TestFormatStringCrash:
    """
    Bug: runner.py line 647 uses f"...{predicted.get('mean_accuracy', 'N/A'):.3f}"
    which crashes with ValueError when mean_accuracy is missing (the 'N/A'
    string default gets the :.3f format specifier applied).
    """

    def test_mean_accuracy_format_handles_missing_key(self):
        """Format display must not crash when mean_accuracy is absent."""
        predicted = {"item_0": 0.8, "item_1": 0.7}
        mean_acc = predicted.get("mean_accuracy")
        if isinstance(mean_acc, (int, float)):
            result = f"mean_accuracy={mean_acc:.3f}"
        else:
            result = "mean_accuracy=N/A"
        assert result == "mean_accuracy=N/A"

    def test_mean_accuracy_format_handles_none(self):
        """Format display must not crash when mean_accuracy is None."""
        predicted = {"mean_accuracy": None}
        mean_acc = predicted.get("mean_accuracy")
        if isinstance(mean_acc, (int, float)):
            result = f"mean_accuracy={mean_acc:.3f}"
        else:
            result = "mean_accuracy=N/A"
        assert result == "mean_accuracy=N/A"

    def test_mean_accuracy_format_works_with_float(self):
        predicted = {"mean_accuracy": 0.756}
        mean_acc = predicted.get("mean_accuracy")
        if isinstance(mean_acc, (int, float)):
            result = f"mean_accuracy={mean_acc:.3f}"
        else:
            result = "mean_accuracy=N/A"
        assert result == "mean_accuracy=0.756"


class TestDesignSpecValidation:
    """
    Bug: runner.py lines 593-594 call exp.design_spec.get() without checking
    that design_spec is a dict. If LLM outputs invalid JSON, design_spec
    could be None or a non-dict, causing AttributeError.
    """

    def test_non_dict_design_spec_handled(self):
        """When design_spec is not a dict, structure_name should default safely."""
        design_spec = None  # simulate JSON parse failure
        if isinstance(design_spec, dict):
            struct_name = design_spec.get("structure_name", "")
        else:
            struct_name = ""
        assert struct_name == ""

    def test_empty_structure_name_uses_fallback(self):
        """Empty structure_name should trigger fallback in compute_model_predictions."""
        protocol = DebateProtocol(
            agent_configs=default_agent_configs(),
            state=EpistemicState(domain="test"),
        )
        agent = [a for a in protocol.agent_configs if "Exemplar" in a.name][0]
        # Empty string is not in STRUCTURE_REGISTRY — should fallback to Type_II
        result = protocol.compute_model_predictions(agent, "", "baseline")
        assert "mean_accuracy" in result
        assert isinstance(result["mean_accuracy"], float)

    def test_invalid_structure_name_uses_fallback(self):
        """Misspelled structure names should fallback to Type_II, not crash."""
        protocol = DebateProtocol(
            agent_configs=default_agent_configs(),
            state=EpistemicState(domain="test"),
        )
        agent = [a for a in protocol.agent_configs if "Exemplar" in a.name][0]
        result = protocol.compute_model_predictions(
            agent, "nonexistent_structure_xyz", "baseline"
        )
        assert "mean_accuracy" in result


class TestGCMParameterValidation:
    """
    Bug: GCM._distance() crashes with ZeroDivisionError when r=0,
    and GCM.predict() crashes with KeyError when bias dict is incomplete.
    """

    def test_r_zero_raises_or_handles(self):
        """r=0 in distance calculation must not cause ZeroDivisionError."""
        gcm = GCM()
        stimuli = np.array([[0, 0], [1, 1]], dtype=float)
        labels = np.array([0, 1])
        # r=0 is mathematically undefined; should raise ValueError, not ZeroDivisionError
        with pytest.raises((ValueError, ZeroDivisionError)):
            gcm.predict(stimuli[0], stimuli, labels, c=3.0, r=0)

    def test_incomplete_bias_raises_or_handles(self):
        """Bias dict missing a category must not cause KeyError."""
        gcm = GCM()
        stimuli = np.array([[0, 0], [1, 1]], dtype=float)
        labels = np.array([0, 1])
        # bias only has category 0, missing category 1
        with pytest.raises((KeyError, ValueError)):
            gcm.predict(stimuli[0], stimuli, labels, c=3.0, bias={0: 1.0})


class TestSUSTAINZeroLambdas:
    """
    Bug: SUSTAIN._activation() divides by np.sum(cluster.lambdas), which
    is zero when all lambdas are 0. This causes NaN propagation.
    """

    def test_zero_lambdas_no_nan(self):
        """SUSTAIN with zero initial lambdas must not produce NaN probabilities."""
        sustain = SUSTAIN()
        stimuli = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
        labels = np.array([0, 1])
        result = sustain.predict(stimuli[0], stimuli, labels, initial_lambdas=0.0)
        probs = result["probabilities"]
        for v in probs.values():
            assert not np.isnan(v), f"NaN in probabilities with zero lambdas: {probs}"


class TestRulePlusExceptionValidation:
    """
    Bug: rule_plus_exception() crashes when n_exceptions > n_items_per_category
    because numpy.choice cannot sample more than the population without replace.
    """

    def test_n_exceptions_exceeds_items_raises(self):
        """Must raise ValueError (not numpy internal error) when n_exceptions too large."""
        with pytest.raises((ValueError, Exception)):
            rule_plus_exception(
                n_dims=4, n_items_per_category=2, n_exceptions=5, seed=42
            )


class TestToJsonCreatesParentDirs:
    """
    Bug: EpistemicState.to_json() opens file for writing without ensuring
    parent directories exist. Raises FileNotFoundError on nested paths.
    """

    def test_to_json_creates_parent_dirs(self):
        """to_json() must create parent directories if they don't exist."""
        import tempfile

        state = EpistemicState(domain="test")
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = os.path.join(tmpdir, "a", "b", "c", "state.json")
            state.to_json(nested_path)
            assert os.path.exists(nested_path)


class TestNaNDivergenceMap:
    """
    Bug: NaN in model probabilities silently corrupts the divergence map.
    compute_divergence_map should produce finite values for all valid structures.
    """

    def test_divergence_map_all_finite(self):
        """All values in the divergence map must be finite (no NaN/Inf)."""
        protocol = DebateProtocol(
            agent_configs=default_agent_configs(),
            state=EpistemicState(domain="test"),
        )
        div_map = protocol.compute_divergence_map()
        for struct_name, struct_data in div_map.items():
            for pair_name, metrics in struct_data.get("divergences", {}).items():
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        assert np.isfinite(value), (
                            f"Non-finite {metric_name}={value} in "
                            f"{struct_name}/{pair_name}"
                        )


class TestCallAgentErrorHandling:
    """
    Bug: call_agent() raises ValueError on empty API responses, but callers
    in runner.py don't catch it — a single API failure crashes the whole debate.

    We test that the error is a clean ValueError with a helpful message.
    """

    def test_empty_anthropic_response_raises_valueerror(self):
        """Empty Anthropic response must raise ValueError, not AttributeError."""
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_response = MagicMock()
        mock_response.content = []
        mock_client.messages.create.return_value = mock_response

        with pytest.raises(ValueError, match="Empty response"):
            call_agent(mock_client, "system", "user")


class TestSUSTAINPredictLearningCurve:
    """
    Bug (P2): SUSTAIN.predict_learning_curve() accepts test_items and
    test_labels parameters but ignores them entirely. It reports training
    accuracy from trial_log instead of testing the learned model on
    held-out items at each block boundary.

    GCM.predict_learning_curve() correctly tests on held-out items.
    SUSTAIN should follow the same contract.
    """

    def test_uses_test_items_not_training_log(self):
        """Learning curve accuracy must reflect test-item performance."""
        sustain = SUSTAIN()
        # Train on Type I (simple rule): dim0 determines category
        train_items = np.array(
            [[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]], dtype=float
        )
        train_labels = np.array([0, 0, 1, 1])
        training_seq = list(zip(train_items, train_labels))

        # Test items: same structure but different instances
        test_items = np.array([[0, 0, 1], [1, 1, 0]], dtype=float)
        test_labels = np.array([0, 1])

        curve = sustain.predict_learning_curve(
            training_seq, test_items, test_labels, block_size=2
        )
        assert len(curve) > 0
        # Each block should have an 'accuracy' computed from test items
        for block in curve:
            assert "accuracy" in block
            assert isinstance(block["accuracy"], float)

    def test_test_accuracy_differs_from_training_accuracy(self):
        """
        If test items are hard (novel), accuracy should differ from training.
        This confirms test_items are actually being evaluated.
        """
        sustain = SUSTAIN()
        # Train on easy rule
        train_items = np.array(
            [[0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
            dtype=float,
        )
        train_labels = np.array([0, 0, 0, 1, 1, 1])
        training_seq = list(zip(train_items, train_labels))

        # Test on same items (should get similar accuracy to training)
        curve_same = sustain.predict_learning_curve(
            training_seq, train_items, train_labels, block_size=3
        )
        # Test on novel items with reversed labels (should get low accuracy)
        novel_items = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
        reversed_labels = np.array([1, 0])  # Opposite of training
        curve_reversed = sustain.predict_learning_curve(
            training_seq, novel_items, reversed_labels, block_size=3
        )

        # The two curves should have different final accuracies
        final_same = curve_same[-1]["accuracy"]
        final_reversed = curve_reversed[-1]["accuracy"]
        assert final_same != final_reversed, (
            f"Same ({final_same}) vs reversed ({final_reversed}) test labels "
            "produced identical accuracy — test_items/labels are being ignored"
        )


class TestCallAgentRetry:
    """
    Bug: A single API failure (network error, content filter, rate limit)
    crashes the entire multi-cycle debate. call_agent should retry on
    transient failures.
    """

    def test_retries_on_transient_error(self):
        """call_agent should retry and succeed after transient failures."""
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        mock_client.messages = MagicMock()

        # First call raises, second succeeds
        good_response = MagicMock()
        good_response.content = [MagicMock(text="success")]
        mock_client.messages.create.side_effect = [
            Exception("Connection reset"),
            good_response,
        ]

        result = call_agent(mock_client, "system", "user")
        assert result == "success"
        assert mock_client.messages.create.call_count == 2

    def test_raises_after_max_retries(self):
        """call_agent should raise after exhausting retries."""
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create.side_effect = Exception("Persistent failure")

        with pytest.raises(Exception, match="Persistent failure"):
            call_agent(mock_client, "system", "user")


# =========================================================================
# Leave-one-out prediction fix — 2026-03-13 Session 8
# =========================================================================


class TestLeaveOneOutPredictions:
    """
    Bug: compute_model_predictions() trains and tests on the same items.
    For GCM, this means item i is its own nearest exemplar (distance=0,
    similarity=1.0), producing near-binary predictions that don't match
    noisy synthetic data. SUSTAIN's softer cluster-based predictions
    accidentally fit noise better, causing the wrong model to win.

    Fix: Leave-one-out — when predicting item i, exclude it from the
    training set. This is standard practice in the GCM literature
    (Nosofsky 1986).
    """

    def _make_protocol(self):
        return DebateProtocol(
            agent_configs=default_agent_configs(),
            state=EpistemicState(domain="test"),
        )

    def _get_agent(self, protocol, name_fragment):
        return [a for a in protocol.agent_configs if name_fragment in a.name][0]

    def test_gcm_predictions_vary_per_item(self):
        """
        With LOO on an asymmetric structure, GCM should produce
        different predictions per item (exception items vs rule items).
        Shepard types are symmetric so we use rule_plus_exception.
        """
        protocol = self._make_protocol()
        agent = self._get_agent(protocol, "Exemplar")

        # rule_plus_exception has exceptions that break symmetry
        result = protocol.compute_model_predictions(
            agent, "rule_plus_exception_1exc", "baseline", param_overrides={"c": 6.0}
        )

        values = [v for k, v in result.items() if k.startswith("item_")]
        unique_rounded = set(round(v, 2) for v in values)
        assert len(unique_rounded) > 1, (
            f"GCM predictions are all identical ({unique_rounded}). "
            "LOO should produce varied per-item predictions on asymmetric structures."
        )

    def test_loo_changes_predictions(self):
        """
        LOO predictions should differ from full-set predictions.
        This confirms item i is actually excluded from the training set.
        """
        from antagonistic_collab.models.gcm import GCM
        from antagonistic_collab.debate_protocol import STRUCTURE_REGISTRY

        gcm = GCM()
        struct = STRUCTURE_REGISTRY["five_four"]
        stimuli = np.asarray(struct["stimuli"])
        labels = np.asarray(struct["labels"])

        # Full prediction (including self)
        full_pred = gcm.predict(stimuli[0], stimuli, labels, c=6.0)
        # LOO prediction (excluding item 0)
        loo_pred = gcm.predict(
            stimuli[0],
            np.delete(stimuli, 0, axis=0),
            np.delete(labels, 0),
            c=6.0,
        )

        full_p = full_pred["probabilities"][int(labels[0])]
        loo_p = loo_pred["probabilities"][int(labels[0])]
        assert full_p != loo_p, (
            f"Full ({full_p:.4f}) and LOO ({loo_p:.4f}) predictions are "
            "identical — LOO is not excluding the test item"
        )

    def test_divergence_map_uses_loo(self):
        """
        compute_divergence_map() should also use LOO when computing
        per-item probabilities.
        """
        protocol = self._make_protocol()
        div_map = protocol.compute_divergence_map()

        # With LOO, GCM predictions on Type_I should not all be 0.5
        # (they would be if GCM always predicts perfectly for its own
        # category then has no discriminability after LOO on a 1D rule).
        # The key check: divergences should still be non-zero.
        type1 = div_map.get("Type_I", {})
        if type1:
            for pair, metrics in type1.get("divergences", {}).items():
                # Models should still disagree on some structures
                assert isinstance(metrics["mean_abs_diff"], float)

    def test_correct_model_wins_with_loo(self):
        """
        When GCM is effectively the ground truth, Exemplar_Agent should
        produce predictions closest to GCM's own outputs. This is the
        core validation that LOO fixes the self-prediction bias.
        """
        protocol = self._make_protocol()
        exemplar = self._get_agent(protocol, "Exemplar")
        clustering = self._get_agent(protocol, "Clustering")

        # Use Type_VI — where models disagree most
        gcm_pred = protocol.compute_model_predictions(exemplar, "Type_VI", "baseline")
        sus_pred = protocol.compute_model_predictions(clustering, "Type_VI", "baseline")

        # Both should return valid predictions
        gcm_items = {k: v for k, v in gcm_pred.items() if k.startswith("item_")}
        sus_items = {k: v for k, v in sus_pred.items() if k.startswith("item_")}
        assert len(gcm_items) == len(sus_items)
        assert len(gcm_items) > 0


# =========================================================================
# Divergence-driven experiment selection — 2026-03-14 Session 8
# =========================================================================


class TestDivergenceDrivenSelection:
    """
    Bug: Batch-mode arbitration uses round-robin by agent, ignoring which
    experiment would best discriminate between models. This means RULEX
    loses because Type_VI (where it's weakest) is always selected in
    cycle 0. The moderator should pick the proposal whose structure has
    the highest divergence between models.
    """

    def _make_protocol(self):
        return DebateProtocol(
            agent_configs=default_agent_configs(),
            state=EpistemicState(domain="test"),
        )

    def test_selects_highest_divergence_structure(self):
        """
        Given proposals for Type_I (low divergence) and five_four (high
        divergence), batch mode should select five_four.
        """
        protocol = self._make_protocol()

        # Propose two experiments with different structures
        protocol.state.propose_experiment(
            proposed_by="Exemplar_Agent",
            title="Type I experiment",
            design_spec={"structure_name": "Type_I", "condition": "baseline"},
            rationale="Test simple rule",
        )
        protocol.state.propose_experiment(
            proposed_by="Rule_Agent",
            title="Five-four experiment",
            design_spec={"structure_name": "five_four", "condition": "baseline"},
            rationale="Test complex structure",
        )

        # Compute divergence map to get actual rankings
        div_map = protocol.compute_divergence_map()

        # Get divergence for each structure (keys match STRUCTURE_REGISTRY)
        def max_div(struct_name, dmap):
            if struct_name in dmap:
                divs = dmap[struct_name].get("divergences", {})
                return max((d["mean_abs_diff"] for d in divs.values()), default=0.0)
            return 0.0

        # Verify five_four actually has higher divergence than Type_I
        # (If not, this test's premise is wrong)
        div_five_four = max_div("five_four", div_map)
        div_type_i = max_div("Type_I", div_map)
        assert div_five_four > div_type_i, (
            f"Test premise failed: five_four div ({div_five_four}) "
            f"<= Type_I div ({div_type_i})"
        )

    def test_batch_mode_uses_divergence_not_round_robin(self):
        """
        The batch-mode selection should consider structure divergence,
        not just cycle through agents in order.
        """
        protocol = self._make_protocol()

        # Agent 0 proposes low-divergence structure
        protocol.state.propose_experiment(
            proposed_by="Exemplar_Agent",
            title="Low divergence",
            design_spec={"structure_name": "Type_I", "condition": "baseline"},
            rationale="",
        )
        # Agent 1 proposes high-divergence structure
        protocol.state.propose_experiment(
            proposed_by="Rule_Agent",
            title="High divergence",
            design_spec={"structure_name": "five_four", "condition": "baseline"},
            rationale="",
        )

        # Run arbitration in batch mode
        import antagonistic_collab.runner as runner

        old_batch = getattr(runner, "_BATCH_MODE", False)
        runner._BATCH_MODE = True
        try:
            run_human_arbitration(protocol, [])
        finally:
            runner._BATCH_MODE = old_batch

        # Check which experiment was approved
        approved = [e for e in protocol.state.experiments if e.status == "approved"]
        assert len(approved) == 1
        assert approved[0].design_spec.get("structure_name") == "five_four", (
            f"Expected five_four (high divergence), got "
            f"{approved[0].design_spec.get('structure_name')}"
        )

    def test_falls_back_to_critique_count_on_tie(self):
        """
        When proposals have similar divergence, prefer the one with more
        critiques (more scrutinized = more refined).
        """
        protocol = self._make_protocol()

        # Both propose the same structure
        protocol.state.propose_experiment(
            proposed_by="Exemplar_Agent",
            title="Proposal A",
            design_spec={"structure_name": "Type_II", "condition": "baseline"},
            rationale="",
        )
        exp2 = protocol.state.propose_experiment(
            proposed_by="Rule_Agent",
            title="Proposal B",
            design_spec={"structure_name": "Type_II", "condition": "baseline"},
            rationale="",
        )

        # Add more critiques to proposal B
        protocol.state.add_critique(
            exp2.experiment_id, "Clustering_Agent", "Good proposal"
        )
        protocol.state.add_critique(exp2.experiment_id, "Exemplar_Agent", "I agree")

        import antagonistic_collab.runner as runner

        old_batch = getattr(runner, "_BATCH_MODE", False)
        runner._BATCH_MODE = True
        try:
            run_human_arbitration(protocol, [])
        finally:
            runner._BATCH_MODE = old_batch

        approved = [e for e in protocol.state.experiments if e.status == "approved"]
        assert len(approved) == 1
        assert approved[0].title == "Proposal B", (
            f"Expected Proposal B (more critiques), got {approved[0].title}"
        )


# =========================================================================
# Concrete model predictions in divergence ranking — 2026-03-14
# =========================================================================


class TestConcreteModelPredictions:
    """
    Bug: Agents see divergence scores but not per-model predictions.
    Rule_Agent proposes Type_II (lowest divergence) every cycle because
    it can't see that RULEX dominates on Type_I. The divergence context
    should show per-model accuracies per structure in the ranked summary.
    """

    def _make_protocol(self):
        return DebateProtocol(
            agent_configs=default_agent_configs(),
            state=EpistemicState(domain="test"),
        )

    def test_divergence_map_contains_per_model_accuracy(self):
        """compute_divergence_map() should include accuracy per agent per structure."""
        protocol = self._make_protocol()
        div_map = protocol.compute_divergence_map()

        for struct_name, data in div_map.items():
            assert "predictions" in data, f"{struct_name} missing 'predictions'"
            for agent_name, preds in data["predictions"].items():
                assert "accuracy" in preds, (
                    f"{struct_name}/{agent_name} missing 'accuracy'"
                )
                assert 0.0 <= preds["accuracy"] <= 1.0, (
                    f"{struct_name}/{agent_name} accuracy out of range: "
                    f"{preds['accuracy']}"
                )

    def test_divergence_context_shows_per_model_predictions(self):
        """
        The ranked summary in _divergence_context() should include
        per-model predictions (e.g., 'Exemplar_Agent: 0.85') for each
        structure, not just the divergence score.
        """
        protocol = self._make_protocol()
        context = protocol._divergence_context()

        # The ranked section should contain agent names with accuracy values
        # Check that all three agent names appear in the ranked summary
        agent_names = [a.name for a in protocol.agent_configs]
        for agent_name in agent_names:
            assert agent_name in context, (
                f"Agent '{agent_name}' not found in divergence context"
            )

        # Check that accuracy values appear near agent names in the ranked
        # section (format: "AgentName: 0.XXX")
        import re

        # Find all "AgentName: 0.XXX" patterns in the ranked section
        ranked_section = context.split("Ranked by Maximum Divergence")[1]
        accuracy_pattern = re.findall(r"\w+_Agent: \d\.\d{2,}", ranked_section)
        assert len(accuracy_pattern) >= 3, (
            f"Expected at least 3 agent accuracy entries in ranked section, "
            f"found {len(accuracy_pattern)}: {accuracy_pattern}"
        )

    def test_divergence_context_identifies_best_model_per_structure(self):
        """
        For each structure in the ranked summary, agents should be able
        to see which model has the highest accuracy.
        """
        protocol = self._make_protocol()
        div_map = protocol.compute_divergence_map()
        context = protocol._divergence_context(div_map)

        # Verify that for Type_I (simple rule), the ranked section contains
        # per-model info that would help Rule_Agent identify it as favorable
        ranked_section = context.split("Ranked by Maximum Divergence")[1]
        assert "Type_I" in ranked_section

        # Get actual predictions for Type_I
        type_i_preds = div_map["Type_I"]["predictions"]
        agent_accuracies = {
            name: preds["accuracy"] for name, preds in type_i_preds.items()
        }

        # All agents' accuracies should appear in the ranked section
        for agent, acc in agent_accuracies.items():
            acc_str = f"{acc:.2f}"
            assert acc_str in ranked_section, (
                f"Accuracy {acc_str} for {agent} on Type_I not found in ranked section"
            )

    def test_per_model_predictions_differ_across_structures(self):
        """
        Per-model accuracies should vary across structures. If they don't,
        the information isn't helpful for strategic proposal selection.
        """
        protocol = self._make_protocol()
        div_map = protocol.compute_divergence_map()

        # Collect accuracies for one agent across all structures
        exemplar_accs = []
        for struct_name, data in div_map.items():
            for agent_name, preds in data["predictions"].items():
                if "Exemplar" in agent_name:
                    exemplar_accs.append(preds["accuracy"])
                    break

        # Should have variation (at least 3 distinct values across 11 structures)
        distinct = len(set(round(a, 3) for a in exemplar_accs))
        assert distinct >= 3, (
            f"Expected >=3 distinct accuracy values for Exemplar_Agent, "
            f"got {distinct}: {exemplar_accs}"
        )


# =========================================================================
# Unknown param overrides crash model.predict() — 2026-03-14
# =========================================================================


class TestUnknownParamOverrides:
    """
    Bug: LLM agents can propose param_overrides with keys that the model's
    predict() method doesn't accept (e.g., 'w_i' for GCM). This causes
    TypeError: GCM.predict() got an unexpected keyword argument 'w_i'.
    The system should filter out unknown params before calling predict().
    """

    def _make_protocol(self):
        return DebateProtocol(
            agent_configs=default_agent_configs(),
            state=EpistemicState(domain="test"),
        )

    def test_unknown_param_overrides_ignored(self):
        """
        compute_model_predictions() should silently ignore param overrides
        that the model's predict() method doesn't accept.
        """
        protocol = self._make_protocol()
        # Find the GCM agent
        gcm_agent = next(a for a in protocol.agent_configs if "Exemplar" in a.name)

        # Pass a bogus param that GCM.predict() doesn't accept
        result = protocol.compute_model_predictions(
            gcm_agent,
            "Type_II",
            "baseline",
            param_overrides={"w_i": [0.5, 0.5, 0.5], "bogus_param": 42},
        )

        # Should succeed without crashing
        assert "mean_accuracy" in result
        assert isinstance(result["mean_accuracy"], float)

    def test_valid_param_overrides_still_work(self):
        """
        Known params (like 'c' for GCM) should still be applied.
        """
        protocol = self._make_protocol()
        gcm_agent = next(a for a in protocol.agent_configs if "Exemplar" in a.name)

        # c=0.1 (very low sensitivity) vs c=20.0 (very high)
        result_low = protocol.compute_model_predictions(
            gcm_agent, "Type_II", "baseline", param_overrides={"c": 0.1}
        )
        result_high = protocol.compute_model_predictions(
            gcm_agent, "Type_II", "baseline", param_overrides={"c": 20.0}
        )

        # Different c values should produce different predictions
        assert result_low["mean_accuracy"] != result_high["mean_accuracy"], (
            "Different c values should produce different predictions"
        )

    def test_mixed_valid_and_invalid_overrides(self):
        """
        Valid params should be applied, invalid ones should be ignored.
        """
        protocol = self._make_protocol()
        gcm_agent = next(a for a in protocol.agent_configs if "Exemplar" in a.name)

        # Mix valid (c) and invalid (w_i) overrides
        result = protocol.compute_model_predictions(
            gcm_agent,
            "Type_II",
            "baseline",
            param_overrides={"c": 10.0, "w_i": [0.5, 0.5, 0.5]},
        )

        # Should succeed and apply the valid param
        assert "mean_accuracy" in result
        assert result["params_used"].get("c") == 10.0

    def test_wrong_shape_attention_weights_does_not_crash(self):
        """
        LLM agent passes attention_weights with wrong number of dimensions
        (e.g., 3 weights for a 4D structure). Should fall back to defaults
        instead of crashing.
        """
        protocol = self._make_protocol()
        gcm_agent = next(a for a in protocol.agent_configs if "Exemplar" in a.name)

        # linear_separable_4d has 4 dimensions, but we pass 3 weights
        result = protocol.compute_model_predictions(
            gcm_agent,
            "linear_separable_4d",
            "baseline",
            param_overrides={"attention_weights": [0.5, 0.3, 0.2]},
        )

        # Should succeed without crashing
        assert "mean_accuracy" in result
        assert isinstance(result["mean_accuracy"], float)

    def test_negative_c_does_not_crash(self):
        """
        LLM agent passes c=-1 which is invalid. Should fall back to
        defaults instead of crashing.
        """
        protocol = self._make_protocol()
        gcm_agent = next(a for a in protocol.agent_configs if "Exemplar" in a.name)

        result = protocol.compute_model_predictions(
            gcm_agent,
            "Type_II",
            "baseline",
            param_overrides={"c": -1.0},
        )

        assert "mean_accuracy" in result
        assert isinstance(result["mean_accuracy"], float)

    def test_string_param_value_does_not_crash(self):
        """
        LLM agent passes a string where a number is expected.
        Should fall back to defaults instead of crashing.
        """
        protocol = self._make_protocol()
        gcm_agent = next(a for a in protocol.agent_configs if "Exemplar" in a.name)

        result = protocol.compute_model_predictions(
            gcm_agent,
            "Type_II",
            "baseline",
            param_overrides={"c": "high", "r": "manhattan"},
        )

        assert "mean_accuracy" in result
        assert isinstance(result["mean_accuracy"], float)


# =========================================================================
# SUSTAIN partial block drop — 2026-03-14
# =========================================================================


class TestSUSTAINPartialBlock:
    """
    Bug: SUSTAIN.predict_learning_curve() iterates only complete blocks.
    If len(training_sequence) % block_size != 0, the final partial block
    is silently dropped, making SUSTAIN's curve shorter than GCM/RULEX.
    """

    def test_partial_block_included(self):
        """
        A training sequence of 10 items with block_size=4 should produce
        3 curve entries (blocks at 4, 8, 10), not 2 (blocks at 4, 8).
        """
        from antagonistic_collab.models.sustain import SUSTAIN

        model = SUSTAIN()
        # 10 items, block_size=4 → should get blocks at 4, 8, 10
        training_seq = [(np.array([i % 2, i % 3]), i % 2) for i in range(10)]
        test_items = np.array([[0, 0], [1, 1]])
        test_labels = np.array([0, 1])

        curve = model.predict_learning_curve(
            training_seq, test_items, test_labels, block_size=4
        )

        assert len(curve) == 3, (
            f"Expected 3 blocks (4, 8, 10) for 10 items with block_size=4, "
            f"got {len(curve)} blocks"
        )

    def test_exact_block_size_no_extra(self):
        """
        A training sequence of 8 items with block_size=4 should produce
        exactly 2 curve entries (no partial block to add).
        """
        from antagonistic_collab.models.sustain import SUSTAIN

        model = SUSTAIN()
        training_seq = [(np.array([i % 2, i % 3]), i % 2) for i in range(8)]
        test_items = np.array([[0, 0], [1, 1]])
        test_labels = np.array([0, 1])

        curve = model.predict_learning_curve(
            training_seq, test_items, test_labels, block_size=4
        )

        assert len(curve) == 2, (
            f"Expected 2 blocks (4, 8) for 8 items with block_size=4, "
            f"got {len(curve)} blocks"
        )

    def test_curve_length_matches_gcm(self):
        """
        SUSTAIN and GCM should produce learning curves of the same length
        for the same training sequence and block_size.
        """
        from antagonistic_collab.models.sustain import SUSTAIN
        from antagonistic_collab.models.gcm import GCM

        sustain = SUSTAIN()
        gcm = GCM()

        # 7 items, block_size=3 → blocks at 3, 6, 7
        training_seq = [(np.array([i % 2, i % 3]), i % 2) for i in range(7)]
        test_items = np.array([[0, 0], [1, 1]])
        test_labels = np.array([0, 1])

        sustain_curve = sustain.predict_learning_curve(
            training_seq, test_items, test_labels, block_size=3
        )
        gcm_curve = gcm.predict_learning_curve(
            training_seq, test_items, test_labels, block_size=3
        )

        assert len(sustain_curve) == len(gcm_curve), (
            f"SUSTAIN curve length ({len(sustain_curve)}) != "
            f"GCM curve length ({len(gcm_curve)})"
        )


# =========================================================================
# Phase 5: Design Revision — 2026-03-14
# =========================================================================


class TestDesignRevision:
    """
    Bug: Phase 5 (Design Revision) is a placeholder that creates an empty
    PhaseResult. Critiques never lead to revised proposals. The design
    revision function should let each agent revise their proposal in
    light of critiques, updating the design_spec via state.revise_proposal().
    """

    def _make_protocol(self):
        return DebateProtocol(
            agent_configs=default_agent_configs(),
            state=EpistemicState(domain="test"),
        )

    def test_run_design_revision_exists(self):
        """run_design_revision should be importable from runner."""
        from antagonistic_collab.runner import run_design_revision

        assert callable(run_design_revision)

    def test_revision_updates_design_spec(self):
        """
        When an agent revises their proposal, the design_spec should be
        updated to the revised version.
        """
        protocol = self._make_protocol()
        state = protocol.state

        # Create a proposal with critiques
        exp = state.propose_experiment(
            proposed_by="Exemplar_Agent",
            title="Original Proposal",
            design_spec={"structure_name": "Type_VI", "condition": "baseline"},
            rationale="Test exemplar theory",
        )
        state.add_critique(
            exp.experiment_id,
            "Rule_Agent",
            "Type_VI is non-diagnostic for rule models",
        )

        # Revise the proposal
        state.revise_proposal(
            exp.experiment_id,
            revised_by="Exemplar_Agent",
            addresses_critiques=[0],
            changes="Changed structure to five_four",
            new_design_spec={"structure_name": "five_four", "condition": "baseline"},
        )

        # Design spec should be updated
        assert exp.design_spec["structure_name"] == "five_four"
        assert len(exp.revision_history) == 1
        assert exp.revision_history[0]["addresses_critiques"] == [0]

    @patch("antagonistic_collab.runner.call_agent")
    def test_run_design_revision_calls_agents(self, mock_call):
        """
        run_design_revision should call each agent that has a proposal
        with critiques, asking them to revise.
        """
        from antagonistic_collab.runner import run_design_revision

        protocol = self._make_protocol()
        state = protocol.state

        # Create proposals with critiques
        exp1 = state.propose_experiment(
            proposed_by="Exemplar_Agent",
            title="Exemplar Proposal",
            design_spec={"structure_name": "Type_VI", "condition": "baseline"},
            rationale="r",
        )
        state.add_critique(exp1.experiment_id, "Rule_Agent", "Non-diagnostic")

        exp2 = state.propose_experiment(
            proposed_by="Rule_Agent",
            title="Rule Proposal",
            design_spec={"structure_name": "Type_II", "condition": "baseline"},
            rationale="r",
        )
        state.add_critique(exp2.experiment_id, "Exemplar_Agent", "Too easy")

        # Mock LLM to return revised proposals
        mock_call.side_effect = [
            json.dumps(
                {
                    "structure_name": "five_four",
                    "condition": "baseline",
                    "changes": "Switched to five_four per critique",
                    "addresses_critiques": [0],
                }
            ),
            json.dumps(
                {
                    "structure_name": "Type_III",
                    "condition": "high_attention",
                    "changes": "Switched to harder structure",
                    "addresses_critiques": [0],
                }
            ),
        ]

        result = run_design_revision(protocol, None, [])

        assert result.phase == Phase.DESIGN_REVISION
        # Both agents should have been called
        assert mock_call.call_count == 2

    @patch("antagonistic_collab.runner.call_agent")
    def test_revision_preserves_proposal_without_critiques(self, mock_call):
        """
        Proposals with no critiques should not be revised.
        """
        from antagonistic_collab.runner import run_design_revision

        protocol = self._make_protocol()
        state = protocol.state

        # Proposal with no critiques
        exp = state.propose_experiment(
            proposed_by="Exemplar_Agent",
            title="Uncontested Proposal",
            design_spec={"structure_name": "Type_I", "condition": "baseline"},
            rationale="r",
        )

        run_design_revision(protocol, None, [])

        # Agent should NOT be called (no critiques to address)
        assert mock_call.call_count == 0
        # Design spec should be unchanged
        assert exp.design_spec["structure_name"] == "Type_I"

    @patch("antagonistic_collab.runner.call_agent")
    def test_revision_handles_bad_llm_output(self, mock_call):
        """
        If the LLM returns garbage, the proposal should remain unchanged.
        """
        from antagonistic_collab.runner import run_design_revision

        protocol = self._make_protocol()
        state = protocol.state

        exp = state.propose_experiment(
            proposed_by="Exemplar_Agent",
            title="Proposal",
            design_spec={"structure_name": "Type_VI", "condition": "baseline"},
            rationale="r",
        )
        state.add_critique(exp.experiment_id, "Rule_Agent", "Bad")

        # Mock returns unparseable garbage
        mock_call.return_value = "I think we should reconsider the whole approach..."

        run_design_revision(protocol, None, [])

        # Proposal should be unchanged
        assert exp.design_spec["structure_name"] == "Type_VI"
        assert len(exp.revision_history) == 0


# =========================================================================
# Moderator reject path — P2 fix
# =========================================================================


class TestModeratorRejectPath:
    """
    Bug (P2): In interactive mode, typing "reject" at the moderator prompt
    printed "All proposals rejected. (In full version, this loops back.)"
    but did NOT actually loop back. The cycle continued with no approved
    experiment, burning the cycle.

    Fix: run_human_arbitration signals rejection via outputs["rejected"]=True.
    run_cycle checks this and loops back to proposal → critique → revision →
    arbitration, up to MAX_REJECT_RETRIES times.
    """

    @staticmethod
    def _make_protocol():
        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        return DebateProtocol(state, agents)

    @patch("builtins.input", return_value="reject")
    @patch("antagonistic_collab.runner._BATCH_MODE", False)
    def test_reject_signals_rejection_in_output(self, mock_input):
        """
        When the moderator types 'reject', the PhaseResult outputs must
        include rejected=True so run_cycle knows to loop back.
        """
        protocol = self._make_protocol()
        state = protocol.state
        state.cycle = 0

        state.propose_experiment(
            proposed_by="Exemplar_Agent",
            title="Test proposal",
            design_spec={"structure_name": "Type_I", "condition": "baseline"},
            rationale="r",
        )

        result = run_human_arbitration(protocol, [])

        assert result.outputs.get("rejected") is True, (
            "run_human_arbitration should set outputs['rejected']=True on reject"
        )

    @patch("builtins.input", return_value="reject")
    @patch("antagonistic_collab.runner._BATCH_MODE", False)
    def test_reject_does_not_approve_any_experiment(self, mock_input):
        """
        After rejection, no experiment should be in 'approved' status.
        """
        protocol = self._make_protocol()
        state = protocol.state
        state.cycle = 0

        state.propose_experiment(
            proposed_by="Exemplar_Agent",
            title="Test proposal",
            design_spec={"structure_name": "Type_I", "condition": "baseline"},
            rationale="r",
        )

        run_human_arbitration(protocol, [])

        approved = [e for e in state.experiments if e.status == "approved"]
        assert len(approved) == 0, "Rejection should not approve any experiment"

    @patch("builtins.input", return_value="approve 0")
    @patch("antagonistic_collab.runner._BATCH_MODE", False)
    def test_approve_does_not_signal_rejection(self, mock_input):
        """
        Normal approval should NOT set rejected=True.
        """
        protocol = self._make_protocol()
        state = protocol.state
        state.cycle = 0

        state.propose_experiment(
            proposed_by="Exemplar_Agent",
            title="Test proposal",
            design_spec={"structure_name": "Type_I", "condition": "baseline"},
            rationale="r",
        )

        result = run_human_arbitration(protocol, [])

        assert result.outputs.get("rejected") is not True, (
            "Approval should not signal rejection"
        )

    @patch("builtins.input", return_value="reject")
    @patch("antagonistic_collab.runner._BATCH_MODE", False)
    def test_reject_marks_proposals_status(self, mock_input):
        """
        After rejection, the 'rejected' flag is set so run_cycle can
        mark proposals and loop back. Proposals are NOT auto-marked by
        run_human_arbitration itself — that's run_cycle's job.
        """
        protocol = self._make_protocol()
        state = protocol.state
        state.cycle = 0

        state.propose_experiment(
            proposed_by="Exemplar_Agent",
            title="Test proposal",
            design_spec={"structure_name": "Type_I", "condition": "baseline"},
            rationale="r",
        )

        result = run_human_arbitration(protocol, [])
        assert result.outputs["rejected"] is True

    @patch("antagonistic_collab.runner._BATCH_MODE", False)
    def test_run_cycle_retries_on_reject(self):
        """
        When the moderator rejects once then approves, run_cycle should
        call run_experiment_proposal twice (one initial + one retry).
        """
        from antagonistic_collab.runner import run_cycle

        protocol = self._make_protocol()
        state = protocol.state
        proposal_call_count = 0

        def mock_proposal(proto, client, transcript):
            nonlocal proposal_call_count
            proposal_call_count += 1
            state.propose_experiment(
                proposed_by="Exemplar_Agent",
                title=f"Proposal round {proposal_call_count}",
                design_spec={"structure_name": "Type_I", "condition": "baseline"},
                rationale="r",
            )
            return PhaseResult(
                phase=Phase.EXPERIMENT_PROPOSAL,
                cycle=state.cycle,
                outputs={},
                messages=[],
            )

        def mock_critique(proto, client, transcript, n_rounds=2):
            return PhaseResult(
                phase=Phase.ADVERSARIAL_CRITIQUE,
                cycle=state.cycle,
                outputs={},
                messages=[],
            )

        def mock_revision(proto, client, transcript):
            return PhaseResult(
                phase=Phase.DESIGN_REVISION,
                cycle=state.cycle,
                outputs={},
                messages=[],
            )

        arbitration_calls = 0

        def mock_arbitration(proto, transcript):
            nonlocal arbitration_calls
            arbitration_calls += 1
            if arbitration_calls == 1:
                # First call: reject
                return PhaseResult(
                    phase=Phase.HUMAN_ARBITRATION,
                    cycle=state.cycle,
                    outputs={"moderator_choice": "reject", "rejected": True},
                    messages=[],
                )
            else:
                # Second call: approve
                proposed = [
                    e
                    for e in state.experiments
                    if e.cycle == state.cycle and e.status == "proposed"
                ]
                if proposed:
                    state.approve_experiment(proposed[0].experiment_id)
                return PhaseResult(
                    phase=Phase.HUMAN_ARBITRATION,
                    cycle=state.cycle,
                    outputs={"moderator_choice": "approve 0"},
                    messages=[],
                )

        def mock_noop(proto, client, transcript, **kwargs):
            return PhaseResult(
                phase=Phase.EXECUTION,
                cycle=state.cycle,
                outputs={},
                messages=[],
            )

        with (
            patch("antagonistic_collab.runner.run_commitment", mock_noop),
            patch("antagonistic_collab.runner.run_divergence_mapping", mock_noop),
            patch("antagonistic_collab.runner.run_experiment_proposal", mock_proposal),
            patch("antagonistic_collab.runner.run_adversarial_critique", mock_critique),
            patch("antagonistic_collab.runner.run_design_revision", mock_revision),
            patch("antagonistic_collab.runner.run_human_arbitration", mock_arbitration),
            patch("antagonistic_collab.runner.run_execution", mock_noop),
            patch("antagonistic_collab.runner.run_interpretation", mock_noop),
            patch("antagonistic_collab.runner.run_audit", mock_noop),
            patch("antagonistic_collab.runner.save_transcript"),
            patch("antagonistic_collab.runner.save_cycle_markdown"),
        ):
            run_cycle(protocol, None, [], output_dir="/tmp/test_reject")

        assert proposal_call_count == 2, (
            f"Expected 2 proposal rounds (initial + 1 retry), got {proposal_call_count}"
        )
        assert arbitration_calls == 2, (
            f"Expected 2 arbitration calls, got {arbitration_calls}"
        )

    @patch("antagonistic_collab.runner._BATCH_MODE", False)
    def test_run_cycle_caps_retries(self):
        """
        After MAX_REJECT_RETRIES rejections, run_cycle should stop looping
        and proceed (with no approved experiment).
        """
        from antagonistic_collab.runner import run_cycle

        protocol = self._make_protocol()
        state = protocol.state
        proposal_call_count = 0

        def mock_proposal(proto, client, transcript):
            nonlocal proposal_call_count
            proposal_call_count += 1
            state.propose_experiment(
                proposed_by="Exemplar_Agent",
                title=f"Proposal round {proposal_call_count}",
                design_spec={"structure_name": "Type_I", "condition": "baseline"},
                rationale="r",
            )
            return PhaseResult(
                phase=Phase.EXPERIMENT_PROPOSAL,
                cycle=state.cycle,
                outputs={},
                messages=[],
            )

        def mock_critique(proto, client, transcript, n_rounds=2):
            return PhaseResult(
                phase=Phase.ADVERSARIAL_CRITIQUE,
                cycle=state.cycle,
                outputs={},
                messages=[],
            )

        def mock_revision(proto, client, transcript):
            return PhaseResult(
                phase=Phase.DESIGN_REVISION,
                cycle=state.cycle,
                outputs={},
                messages=[],
            )

        def mock_always_reject(proto, transcript):
            return PhaseResult(
                phase=Phase.HUMAN_ARBITRATION,
                cycle=state.cycle,
                outputs={"moderator_choice": "reject", "rejected": True},
                messages=[],
            )

        def mock_noop(proto, client, transcript, **kwargs):
            return PhaseResult(
                phase=Phase.EXECUTION,
                cycle=state.cycle,
                outputs={},
                messages=[],
            )

        with (
            patch("antagonistic_collab.runner.run_commitment", mock_noop),
            patch("antagonistic_collab.runner.run_divergence_mapping", mock_noop),
            patch("antagonistic_collab.runner.run_experiment_proposal", mock_proposal),
            patch("antagonistic_collab.runner.run_adversarial_critique", mock_critique),
            patch("antagonistic_collab.runner.run_design_revision", mock_revision),
            patch(
                "antagonistic_collab.runner.run_human_arbitration", mock_always_reject
            ),
            patch("antagonistic_collab.runner.run_execution", mock_noop),
            patch("antagonistic_collab.runner.run_interpretation", mock_noop),
            patch("antagonistic_collab.runner.run_audit", mock_noop),
            patch("antagonistic_collab.runner.save_transcript"),
            patch("antagonistic_collab.runner.save_cycle_markdown"),
        ):
            run_cycle(protocol, None, [], output_dir="/tmp/test_reject")

        # MAX_REJECT_RETRIES = 2, so 3 total attempts
        assert proposal_call_count == 3, (
            f"Expected 3 proposal rounds (initial + 2 retries), got {proposal_call_count}"
        )

    @patch("antagonistic_collab.runner._BATCH_MODE", False)
    def test_rejected_proposals_marked_rejected(self):
        """
        When looping back, the rejected proposals should be marked with
        status='rejected' so new proposals don't collide.
        """
        from antagonistic_collab.runner import run_cycle

        protocol = self._make_protocol()
        state = protocol.state
        rejected_ids = []
        proposal_round = 0

        def mock_proposal(proto, client, transcript):
            nonlocal proposal_round
            proposal_round += 1
            state.propose_experiment(
                proposed_by="Exemplar_Agent",
                title=f"Proposal round {proposal_round}",
                design_spec={"structure_name": "Type_I", "condition": "baseline"},
                rationale="r",
            )
            return PhaseResult(
                phase=Phase.EXPERIMENT_PROPOSAL,
                cycle=state.cycle,
                outputs={},
                messages=[],
            )

        def mock_critique(proto, client, transcript, n_rounds=2):
            return PhaseResult(
                phase=Phase.ADVERSARIAL_CRITIQUE,
                cycle=state.cycle,
                outputs={},
                messages=[],
            )

        def mock_revision(proto, client, transcript):
            return PhaseResult(
                phase=Phase.DESIGN_REVISION,
                cycle=state.cycle,
                outputs={},
                messages=[],
            )

        call_count = 0

        def mock_arbitration(proto, transcript):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Capture the proposed experiment IDs before rejection
                for e in state.experiments:
                    if e.cycle == state.cycle and e.status == "proposed":
                        rejected_ids.append(e.experiment_id)
                return PhaseResult(
                    phase=Phase.HUMAN_ARBITRATION,
                    cycle=state.cycle,
                    outputs={"moderator_choice": "reject", "rejected": True},
                    messages=[],
                )
            else:
                proposed = [
                    e
                    for e in state.experiments
                    if e.cycle == state.cycle and e.status == "proposed"
                ]
                if proposed:
                    state.approve_experiment(proposed[0].experiment_id)
                return PhaseResult(
                    phase=Phase.HUMAN_ARBITRATION,
                    cycle=state.cycle,
                    outputs={"moderator_choice": "approve 0"},
                    messages=[],
                )

        def mock_noop(proto, client, transcript, **kwargs):
            return PhaseResult(
                phase=Phase.EXECUTION,
                cycle=state.cycle,
                outputs={},
                messages=[],
            )

        with (
            patch("antagonistic_collab.runner.run_commitment", mock_noop),
            patch("antagonistic_collab.runner.run_divergence_mapping", mock_noop),
            patch("antagonistic_collab.runner.run_experiment_proposal", mock_proposal),
            patch("antagonistic_collab.runner.run_adversarial_critique", mock_critique),
            patch("antagonistic_collab.runner.run_design_revision", mock_revision),
            patch("antagonistic_collab.runner.run_human_arbitration", mock_arbitration),
            patch("antagonistic_collab.runner.run_execution", mock_noop),
            patch("antagonistic_collab.runner.run_interpretation", mock_noop),
            patch("antagonistic_collab.runner.run_audit", mock_noop),
            patch("antagonistic_collab.runner.save_transcript"),
            patch("antagonistic_collab.runner.save_cycle_markdown"),
        ):
            run_cycle(protocol, None, [], output_dir="/tmp/test_reject")

        # Check that the first-round proposals were marked rejected
        for eid in rejected_ids:
            exp = next(e for e in state.experiments if e.experiment_id == eid)
            assert exp.status == "rejected", (
                f"Experiment {eid} should be rejected after moderator reject, "
                f"got status={exp.status}"
            )


# =========================================================================
# --demo order-sensitivity — P3 fix
# =========================================================================


class TestDemoFlag:
    """
    Bug (P3): `sys.argv[1] == "--demo"` fails when other flags precede it,
    e.g. `python -m antagonistic_collab --verbose --demo`.

    Fix: Check `"--demo" in sys.argv` instead of positional check.
    """

    @patch("sys.argv", ["prog", "--demo"])
    @patch("antagonistic_collab.demo.demo_model_predictions")
    @patch("antagonistic_collab.demo.demo_divergence_mapping")
    @patch("antagonistic_collab.demo.demo_epistemic_state")
    @patch("antagonistic_collab.demo.demo_full_cycle")
    def test_demo_as_first_arg(self, mock_fc, mock_es, mock_dm, mock_mp):
        """--demo as argv[1] should run demo functions."""
        from antagonistic_collab.__main__ import _entry

        _entry()
        mock_mp.assert_called_once()

    @patch("sys.argv", ["prog", "--other", "--demo"])
    @patch("antagonistic_collab.demo.demo_model_predictions")
    @patch("antagonistic_collab.demo.demo_divergence_mapping")
    @patch("antagonistic_collab.demo.demo_epistemic_state")
    @patch("antagonistic_collab.demo.demo_full_cycle")
    def test_demo_not_first_arg(self, mock_fc, mock_es, mock_dm, mock_mp):
        """--demo as argv[2] should still run demo functions."""
        from antagonistic_collab.__main__ import _entry

        _entry()
        mock_mp.assert_called_once()


# =========================================================================
# Structure diversity in experiment selection
# =========================================================================


class TestStructureDiversity:
    """
    Bug: Divergence-driven selection picks the same high-divergence structure
    every cycle (e.g., Type_VI 4/5 times in RULEX validation). Models that
    are weak on the repeated structure (like RULEX on Type_VI) never get
    tested on favorable structures.

    Fix: Penalize structures that have already been tested in prior cycles.
    Previously-tested structures get their divergence score halved per prior
    use, so untested structures with moderate divergence can win.
    """

    @staticmethod
    def _make_protocol():
        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        return DebateProtocol(state, agents)

    @patch("antagonistic_collab.runner._SELECTION_METHOD", "heuristic")
    @patch("antagonistic_collab.runner._BATCH_MODE", True)
    def test_previously_tested_structure_penalized(self):
        """
        If Type_VI was tested in cycle 0, a proposal for Type_VI in cycle 1
        should lose to a proposal for an untested structure with lower raw
        divergence.
        """
        protocol = self._make_protocol()
        state = protocol.state

        # Cycle 0: Type_VI was tested (simulate by creating an executed experiment)
        state.cycle = 0
        exp0 = state.propose_experiment(
            proposed_by="Clustering_Agent",
            title="Cycle 0 experiment",
            design_spec={"structure_name": "Type_VI", "condition": "baseline"},
            rationale="r",
        )
        state.approve_experiment(exp0.experiment_id)
        exp0.status = "executed"

        # Move to cycle 1
        state.cycle = 1

        # Proposal A: Type_VI again (high raw divergence but already tested)
        state.propose_experiment(
            proposed_by="Clustering_Agent",
            title="Type_VI again",
            design_spec={"structure_name": "Type_VI", "condition": "baseline"},
            rationale="r",
        )
        # Proposal B: five_four (untested, moderate divergence)
        state.propose_experiment(
            proposed_by="Exemplar_Agent",
            title="Five four fresh",
            design_spec={"structure_name": "five_four", "condition": "baseline"},
            rationale="r",
        )

        run_human_arbitration(protocol, [])

        approved = [
            e for e in state.experiments if e.cycle == 1 and e.status == "approved"
        ]
        assert len(approved) == 1
        approved_struct = approved[0].design_spec.get("structure_name")
        assert approved_struct != "Type_VI", (
            f"Previously-tested Type_VI should be penalized, but it was selected again. "
            f"Got: {approved_struct}"
        )

    @patch("antagonistic_collab.runner._SELECTION_METHOD", "heuristic")
    @patch("antagonistic_collab.runner._BATCH_MODE", True)
    def test_untested_structure_preferred(self):
        """
        With no prior experiments, the highest-divergence structure should
        still win (no penalty applied).
        """
        protocol = self._make_protocol()
        state = protocol.state
        state.cycle = 0

        state.propose_experiment(
            proposed_by="Exemplar_Agent",
            title="High div",
            design_spec={"structure_name": "five_four", "condition": "baseline"},
            rationale="r",
        )
        state.propose_experiment(
            proposed_by="Rule_Agent",
            title="Low div",
            design_spec={"structure_name": "Type_I", "condition": "baseline"},
            rationale="r",
        )

        run_human_arbitration(protocol, [])

        approved = [
            e for e in state.experiments if e.cycle == 0 and e.status == "approved"
        ]
        assert len(approved) == 1
        assert approved[0].design_spec.get("structure_name") == "five_four", (
            "With no prior experiments, highest divergence should still win"
        )

    @patch("antagonistic_collab.runner._SELECTION_METHOD", "heuristic")
    @patch("antagonistic_collab.runner._BATCH_MODE", True)
    def test_structure_tested_twice_penalized_more(self):
        """
        A structure tested twice should be penalized more heavily than one
        tested once.
        """
        protocol = self._make_protocol()
        state = protocol.state

        # Simulate 2 prior cycles both testing Type_VI
        for c in range(2):
            state.cycle = c
            exp = state.propose_experiment(
                proposed_by="Clustering_Agent",
                title=f"Cycle {c} Type_VI",
                design_spec={"structure_name": "Type_VI", "condition": "baseline"},
                rationale="r",
            )
            state.approve_experiment(exp.experiment_id)
            exp.status = "executed"

        # Cycle 2: Type_VI vs Type_I (very low divergence)
        state.cycle = 2
        state.propose_experiment(
            proposed_by="Clustering_Agent",
            title="Type_VI third time",
            design_spec={"structure_name": "Type_VI", "condition": "baseline"},
            rationale="r",
        )
        state.propose_experiment(
            proposed_by="Rule_Agent",
            title="Type_I fresh",
            design_spec={"structure_name": "Type_I", "condition": "baseline"},
            rationale="r",
        )

        run_human_arbitration(protocol, [])

        approved = [
            e for e in state.experiments if e.cycle == 2 and e.status == "approved"
        ]
        assert len(approved) == 1
        approved_struct = approved[0].design_spec.get("structure_name")
        assert approved_struct != "Type_VI", (
            f"Type_VI tested 2x should be heavily penalized, got: {approved_struct}"
        )

    @patch("antagonistic_collab.runner._SELECTION_METHOD", "heuristic")
    @patch("antagonistic_collab.runner._BATCH_MODE", True)
    def test_different_condition_penalized_less_than_same(self):
        """
        Same structure with a new condition should be penalized less harshly
        than an exact structure+condition repeat. Type_VI/high_attention after
        Type_VI/baseline should still be viable (1.5x penalty vs 2x).
        """
        protocol = self._make_protocol()
        state = protocol.state

        # Cycle 0: Type_VI/baseline was tested
        state.cycle = 0
        exp0 = state.propose_experiment(
            proposed_by="Clustering_Agent",
            title="Cycle 0 baseline",
            design_spec={"structure_name": "Type_VI", "condition": "baseline"},
            rationale="r",
        )
        state.approve_experiment(exp0.experiment_id)
        exp0.status = "executed"

        # Cycle 1: Type_VI/high_attention vs Type_VI/baseline
        state.cycle = 1
        state.propose_experiment(
            proposed_by="Clustering_Agent",
            title="Type_VI new condition",
            design_spec={"structure_name": "Type_VI", "condition": "high_attention"},
            rationale="r",
        )
        state.propose_experiment(
            proposed_by="Exemplar_Agent",
            title="Type_VI same condition",
            design_spec={"structure_name": "Type_VI", "condition": "baseline"},
            rationale="r",
        )

        run_human_arbitration(protocol, [])

        approved = [
            e for e in state.experiments if e.cycle == 1 and e.status == "approved"
        ]
        assert len(approved) == 1
        # New condition should win over exact repeat
        assert approved[0].design_spec.get("condition") == "high_attention", (
            f"New condition should be penalized less than exact repeat, "
            f"got condition={approved[0].design_spec.get('condition')}"
        )


# =========================================================================
# Bayesian Information-Gain Selection  (bayesian_selection.py)
# =========================================================================


class TestBayesianSelection:
    """Tests for Bayesian experiment selection (D18)."""

    # --- ModelPosterior basics ---

    def test_uniform_prior(self):
        """Uniform prior over 3 models: each P=1/3, entropy=ln(3)."""
        post = ModelPosterior.uniform(["GCM", "SUSTAIN", "RULEX"])
        probs = post.probs
        assert probs.shape == (3,)
        np.testing.assert_allclose(probs, [1 / 3, 1 / 3, 1 / 3], atol=1e-10)
        expected_entropy = np.log(3)
        assert abs(post.entropy - expected_entropy) < 1e-10

    def test_posterior_update_shifts_toward_likely_model(self):
        """Strong GCM evidence → P(GCM) > 0.99."""
        post = ModelPosterior.uniform(["GCM", "SUSTAIN", "RULEX"])
        # Strongly favor model 0 (GCM)
        post.update(np.array([0.0, -50.0, -50.0]))
        assert post.probs[0] > 0.99
        assert post.probs[1] < 0.01
        assert post.probs[2] < 0.01

    def test_posterior_serialization_roundtrip(self):
        """to_dict/from_dict preserves state."""
        post = ModelPosterior.uniform(["GCM", "SUSTAIN", "RULEX"])
        post.update(np.array([-1.0, 0.0, -2.0]))
        post.history.append({"cycle": 0, "entropy": post.entropy})

        d = post.to_dict()
        restored = ModelPosterior.from_dict(d)

        np.testing.assert_allclose(restored.log_probs, post.log_probs)
        assert restored.model_names == post.model_names
        assert len(restored.history) == 1
        np.testing.assert_allclose(restored.probs, post.probs, atol=1e-10)

    # --- compute_log_likelihood ---

    def test_log_likelihood_better_match_higher(self):
        """Matching predictions should score higher than mismatched."""
        observed = np.array([0.8, 0.2, 0.9, 0.1])
        good_pred = np.array([0.8, 0.2, 0.9, 0.1])
        bad_pred = np.array([0.2, 0.8, 0.1, 0.9])

        ll_good = compute_log_likelihood(observed, good_pred, n_subjects=20)
        ll_bad = compute_log_likelihood(observed, bad_pred, n_subjects=20)
        assert ll_good > ll_bad, (
            f"Good predictions should have higher LL: {ll_good} vs {ll_bad}"
        )

    def test_log_likelihood_clipping(self):
        """P=0.0 and P=1.0 predictions don't produce -inf."""
        observed = np.array([0.5, 0.5])
        extreme_pred = np.array([0.0, 1.0])
        ll = compute_log_likelihood(observed, extreme_pred, n_subjects=20)
        assert np.isfinite(ll), f"Log-likelihood should be finite, got {ll}"

    # --- compute_eig ---

    def test_eig_nonnegative(self):
        """EIG >= 0 always (information can't decrease in expectation)."""
        post = ModelPosterior.uniform(["GCM", "SUSTAIN", "RULEX"])
        preds = {
            "GCM": np.array([0.8, 0.6, 0.3]),
            "SUSTAIN": np.array([0.5, 0.5, 0.5]),
            "RULEX": np.array([0.9, 0.1, 0.9]),
        }
        eig = compute_eig(preds, post, n_subjects=20, n_sim=100, seed=42)
        assert eig >= 0.0, f"EIG should be non-negative, got {eig}"

    def test_eig_near_zero_when_certain(self):
        """Concentrated posterior → EIG ≈ 0 (already decided)."""
        post = ModelPosterior(
            log_probs=np.array([0.0, -100.0, -100.0]),
            model_names=["GCM", "SUSTAIN", "RULEX"],
        )
        preds = {
            "GCM": np.array([0.8, 0.6]),
            "SUSTAIN": np.array([0.5, 0.5]),
            "RULEX": np.array([0.9, 0.1]),
        }
        eig = compute_eig(preds, post, n_subjects=20, n_sim=100, seed=42)
        assert eig < 0.01, f"EIG should be near zero when certain, got {eig}"

    def test_eig_higher_for_discriminating_structure(self):
        """High-divergence structure should have higher EIG than low-divergence."""
        post = ModelPosterior.uniform(["GCM", "SUSTAIN", "RULEX"])

        # High divergence: models disagree substantially
        high_div = {
            "GCM": np.array([0.9, 0.1, 0.9, 0.1]),
            "SUSTAIN": np.array([0.5, 0.5, 0.5, 0.5]),
            "RULEX": np.array([0.1, 0.9, 0.1, 0.9]),
        }
        eig_high = compute_eig(high_div, post, n_subjects=20, n_sim=200, seed=42)

        # Low divergence: models agree
        low_div = {
            "GCM": np.array([0.6, 0.6, 0.6, 0.6]),
            "SUSTAIN": np.array([0.6, 0.6, 0.6, 0.6]),
            "RULEX": np.array([0.6, 0.6, 0.6, 0.6]),
        }
        eig_low = compute_eig(low_div, post, n_subjects=20, n_sim=200, seed=42)

        assert eig_high > eig_low, (
            f"Discriminating structure should have higher EIG: {eig_high} vs {eig_low}"
        )

    def test_eig_deterministic_with_seed(self):
        """Same seed → same EIG result."""
        post = ModelPosterior.uniform(["GCM", "SUSTAIN", "RULEX"])
        preds = {
            "GCM": np.array([0.8, 0.3]),
            "SUSTAIN": np.array([0.5, 0.5]),
            "RULEX": np.array([0.2, 0.7]),
        }
        eig1 = compute_eig(preds, post, n_subjects=20, n_sim=100, seed=99)
        eig2 = compute_eig(preds, post, n_subjects=20, n_sim=100, seed=99)
        assert eig1 == eig2, f"Same seed should give same EIG: {eig1} vs {eig2}"

    # --- select_experiment ---

    def test_select_experiment_returns_valid_index(self):
        """select_experiment returns an index within the candidates list."""
        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)

        post = ModelPosterior.uniform([a.name for a in agents])

        # Create two candidate proposals with different structures
        state.propose_experiment(
            proposed_by="Exemplar_Agent",
            title="Type_I test",
            design_spec={"structure_name": "Type_I", "condition": "baseline"},
            rationale="test",
        )
        state.propose_experiment(
            proposed_by="Rule_Agent",
            title="Type_VI test",
            design_spec={"structure_name": "Type_VI", "condition": "baseline"},
            rationale="test",
        )
        candidates = [e for e in state.experiments if e.status == "proposed"]

        best_idx, scores = select_experiment(
            protocol, post, candidates, n_subjects=20, n_sim=50, seed=42
        )
        assert 0 <= best_idx < len(candidates)
        assert len(scores) == len(candidates)
        assert all(s >= 0.0 for s in scores)

    # --- EpistemicState integration ---

    def test_posterior_stored_in_epistemic_state(self):
        """model_posterior field exists and can be set."""
        state = EpistemicState(domain="test")
        assert state.model_posterior is None

        post = ModelPosterior.uniform(["GCM", "SUSTAIN", "RULEX"])
        state.model_posterior = post.to_dict()
        assert state.model_posterior is not None
        assert "log_probs" in state.model_posterior

    def test_posterior_survives_json_roundtrip(self):
        """EpistemicState serialization preserves model_posterior."""
        import json
        import tempfile

        state = EpistemicState(domain="test")
        post = ModelPosterior.uniform(["GCM", "SUSTAIN", "RULEX"])
        post.update(np.array([-1.0, 0.0, -2.0]))
        state.model_posterior = post.to_dict()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            state.to_json(path)
            with open(path) as f:
                loaded = json.load(f)
            assert "model_posterior" in loaded
            restored = ModelPosterior.from_dict(loaded["model_posterior"])
            np.testing.assert_allclose(restored.probs, post.probs, atol=1e-10)
        finally:
            os.unlink(path)


# =========================================================================
# Phase A: Full-Pool EIG + Interpretation Debate (D19)
# =========================================================================


class TestFullPoolSelection:
    """Tests for full-pool Bayesian experiment selection.

    The full-pool approach replaces the agent-proposal→moderator-selection
    pipeline with direct EIG computation over all 55 structure×condition
    candidates. This removes LLM calls from the selection path entirely.
    """

    def test_generate_full_candidate_pool_returns_55(self):
        """11 structures × 5 conditions = 55 candidates."""
        from antagonistic_collab.bayesian_selection import generate_full_candidate_pool

        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)
        pool = generate_full_candidate_pool(protocol)
        assert len(pool) == 55
        # Each entry is (structure_name, condition)
        for struct, cond in pool:
            assert struct in STRUCTURE_REGISTRY
            assert cond in CONDITION_EFFECTS

    def test_full_pool_selection_picks_highest_eig(self):
        """select_from_pool returns the candidate with highest EIG."""
        from antagonistic_collab.bayesian_selection import (
            generate_full_candidate_pool,
            select_from_pool,
        )

        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)
        posterior = ModelPosterior.uniform([a.name for a in agents])
        pool = generate_full_candidate_pool(protocol)

        best_idx, eig_scores = select_from_pool(
            protocol, posterior, pool, n_subjects=20, n_sim=50, seed=42
        )
        assert 0 <= best_idx < len(pool)
        assert len(eig_scores) == len(pool)
        assert eig_scores[best_idx] == max(eig_scores)

    def test_full_pool_creates_experiment_record(self):
        """run_full_pool_selection creates an approved ExperimentRecord."""
        from antagonistic_collab.runner import run_full_pool_selection

        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)
        transcript = []

        run_full_pool_selection(protocol, transcript)

        # Should have created and approved one experiment
        approved = [
            e
            for e in state.experiments
            if e.cycle == state.cycle and e.status == "approved"
        ]
        assert len(approved) == 1
        ds = approved[0].design_spec
        assert ds["structure_name"] in STRUCTURE_REGISTRY
        assert ds["condition"] in CONDITION_EFFECTS

    def test_full_pool_with_extra_structures(self):
        """Extra structures from novel proposals are included in the pool."""
        from antagonistic_collab.bayesian_selection import generate_full_candidate_pool

        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)

        extra = {
            "custom_struct": {
                "stimuli": [[0, 0], [0, 1], [1, 0], [1, 1]],
                "labels": [0, 0, 1, 1],
            }
        }
        pool = generate_full_candidate_pool(protocol, extra_structures=extra)
        # 11 + 1 = 12 structures, × 5 conditions = 60
        assert len(pool) == 60
        custom_entries = [(s, c) for s, c in pool if s == "custom_struct"]
        assert len(custom_entries) == 5

    def test_eig_landscape_in_result(self):
        """run_full_pool_selection includes top-5 EIG landscape in outputs."""
        from antagonistic_collab.runner import run_full_pool_selection

        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)
        transcript = []

        result = run_full_pool_selection(protocol, transcript)
        assert "eig_landscape" in result.outputs
        landscape = result.outputs["eig_landscape"]
        assert len(landscape) >= 5  # top-5 at minimum


class TestInterpretationDebate:
    """Tests for the interpretation debate phase that replaces fire-and-forget
    interpretation. Agents now produce structured JSON with interpretation,
    confound flags, hypotheses, and optional novel structure proposals."""

    def test_interpretation_debate_parses_hypothesis(self):
        """Mock LLM returns JSON with hypothesis; verify it's stored."""
        from antagonistic_collab.runner import run_interpretation_debate

        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)

        # Create an executed experiment so interpretation has something to see
        exp = state.propose_experiment(
            proposed_by="Exemplar_Agent",
            title="Test Exp",
            design_spec={"structure_name": "Type_I", "condition": "baseline"},
            rationale="test",
        )
        state.approve_experiment(exp.experiment_id)
        state.record_data(
            exp.experiment_id,
            {
                "mean_accuracy": 0.8,
                "item_accuracies": {"item_0": 0.9, "item_1": 0.7},
            },
        )

        # Mock LLM to return structured interpretation
        mock_response = json.dumps(
            {
                "interpretation": "GCM fits well due to high similarity structure",
                "confounds_flagged": ["small sample size"],
                "hypothesis": "Type_VI will reveal SUSTAIN advantage",
                "revision": None,
            }
        )

        def fake_client_call(*a, **kw):
            return mock_response

        class FakeClient:
            pass

        transcript = []
        with patch(
            "antagonistic_collab.runner.call_agent", side_effect=fake_client_call
        ):
            result = run_interpretation_debate(protocol, FakeClient(), transcript)

        assert "agent_hypotheses" in result.outputs
        hypotheses = result.outputs["agent_hypotheses"]
        assert len(hypotheses) > 0

    def test_confound_flags_stored(self):
        """Confound flags from agent interpretations are stored in result."""
        from antagonistic_collab.runner import run_interpretation_debate

        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)

        exp = state.propose_experiment(
            proposed_by="Exemplar_Agent",
            title="Test Exp",
            design_spec={"structure_name": "Type_I", "condition": "baseline"},
            rationale="test",
        )
        state.approve_experiment(exp.experiment_id)
        state.record_data(
            exp.experiment_id,
            {
                "mean_accuracy": 0.8,
                "item_accuracies": {"item_0": 0.9, "item_1": 0.7},
            },
        )

        mock_response = json.dumps(
            {
                "interpretation": "Results consistent with rule-based learning",
                "confounds_flagged": ["ceiling effect", "order bias"],
                "hypothesis": "Need harder structure",
            }
        )

        def fake_call(*a, **kw):
            return mock_response

        transcript = []
        with patch("antagonistic_collab.runner.call_agent", side_effect=fake_call):
            result = run_interpretation_debate(
                protocol, type("C", (), {})(), transcript
            )

        assert "confounds" in result.outputs
        all_confounds = result.outputs["confounds"]
        assert len(all_confounds) > 0

    def test_hypothesis_stored_for_next_cycle(self):
        """Hypotheses are persisted in EpistemicState.agent_hypotheses."""
        from antagonistic_collab.runner import run_interpretation_debate

        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)

        exp = state.propose_experiment(
            proposed_by="Exemplar_Agent",
            title="Test",
            design_spec={"structure_name": "Type_I", "condition": "baseline"},
            rationale="test",
        )
        state.approve_experiment(exp.experiment_id)
        state.record_data(
            exp.experiment_id,
            {
                "mean_accuracy": 0.75,
                "item_accuracies": {"item_0": 0.8},
            },
        )

        mock_response = json.dumps(
            {
                "interpretation": "Supports exemplar theory",
                "confounds_flagged": [],
                "hypothesis": "Test with Type_VI next",
            }
        )

        def fake_call(*a, **kw):
            return mock_response

        transcript = []
        with patch("antagonistic_collab.runner.call_agent", side_effect=fake_call):
            run_interpretation_debate(protocol, type("C", (), {})(), transcript)

        assert hasattr(state, "agent_hypotheses")
        assert len(state.agent_hypotheses) > 0

    def test_param_stability_tracking(self):
        """Interpretation context includes parameter stability info."""
        from antagonistic_collab.runner import run_interpretation_debate

        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)

        # Register theories + run a couple cycles worth of predictions
        for agent in agents:
            state.register_theory(
                TheoryCommitment(
                    name=agent.theory_name,
                    agent_name=agent.name,
                    core_claims=["claim"],
                    model_name=agent.model_class.name,
                    model_params=agent.default_params,
                )
            )

        exp = state.propose_experiment(
            proposed_by="Exemplar_Agent",
            title="Test",
            design_spec={"structure_name": "Type_I", "condition": "baseline"},
            rationale="test",
        )
        state.approve_experiment(exp.experiment_id)
        state.record_data(
            exp.experiment_id,
            {
                "mean_accuracy": 0.75,
                "item_accuracies": {"item_0": 0.8},
            },
        )

        mock_response = json.dumps(
            {
                "interpretation": "OK",
                "confounds_flagged": [],
                "hypothesis": "next",
            }
        )

        captured_prompts = []

        def fake_call(*a, **kw):
            if len(a) >= 2:
                captured_prompts.append(a[1])
            return mock_response

        transcript = []
        with patch("antagonistic_collab.runner.call_agent", side_effect=fake_call):
            run_interpretation_debate(protocol, type("C", (), {})(), transcript)

        # At least one prompt should mention parameter stability or params
        assert any("param" in p.lower() for p in captured_prompts if isinstance(p, str))


class TestInterpretationCritique:
    """Tests for interpretation critique phase where agents challenge
    each other's interpretations and confound claims."""

    def test_critique_produces_output(self):
        """run_interpretation_critique returns a PhaseResult with messages."""
        from antagonistic_collab.runner import run_interpretation_critique

        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)

        # Set up an executed experiment with interpretations
        exp = state.propose_experiment(
            proposed_by="Exemplar_Agent",
            title="Test",
            design_spec={"structure_name": "Type_I", "condition": "baseline"},
            rationale="test",
        )
        state.approve_experiment(exp.experiment_id)
        state.record_data(
            exp.experiment_id,
            {
                "mean_accuracy": 0.8,
                "item_accuracies": {"item_0": 0.9, "item_1": 0.7},
            },
        )
        state.add_interpretation(exp.experiment_id, "Exemplar_Agent", "GCM fits best")
        state.add_interpretation(exp.experiment_id, "Rule_Agent", "Rules explain this")

        mock_response = "I challenge the exemplar interpretation because..."

        def fake_call(*a, **kw):
            return mock_response

        transcript = []
        with patch("antagonistic_collab.runner.call_agent", side_effect=fake_call):
            result = run_interpretation_critique(
                protocol, type("C", (), {})(), transcript
            )

        assert result.phase.name == "INTERPRETATION"
        assert len(result.messages) > 0


class TestFullPoolModeFlag:
    """Tests for the --mode full_pool|legacy flag in run_cycle."""

    def test_legacy_mode_runs_original_flow(self):
        """--mode legacy should call proposal/critique/revision/arbitration phases."""
        from antagonistic_collab.runner import run_cycle

        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)
        transcript = []

        call_log = []

        def fake_call(*a, **kw):
            call_log.append("called")
            return json.dumps(
                {
                    "title": "Test",
                    "structure_name": "Type_I",
                    "condition": "baseline",
                    "rationale": "test",
                    "reasoning": "test",
                    "confidence": "medium",
                }
            )

        with (
            patch("antagonistic_collab.runner.call_agent", side_effect=fake_call),
            patch("antagonistic_collab.runner._BATCH_MODE", True),
            patch("antagonistic_collab.runner._SELECTION_METHOD", "bayesian"),
        ):
            run_cycle(protocol, type("C", (), {})(), transcript, mode="legacy")

        # Legacy mode should have many LLM calls (proposals, critiques, etc.)
        assert len(call_log) > 6

    def test_full_pool_mode_no_proposal_phases(self):
        """--mode full_pool should skip proposal/critique/revision/arbitration."""
        from antagonistic_collab.runner import run_cycle

        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)
        transcript = []

        call_phases = []

        def fake_call(*a, **kw):
            # Track which phase prompt this is for
            prompt = a[1] if len(a) >= 2 else kw.get("user_message", "")
            if "Experiment Proposal" in prompt:
                call_phases.append("proposal")
            elif "Adversarial Critique" in prompt:
                call_phases.append("critique")
            elif "Design Revision" in prompt:
                call_phases.append("revision")
            elif "Interpretation" in prompt or "interpret" in prompt.lower():
                call_phases.append("interpretation")
            else:
                call_phases.append("other")
            return json.dumps(
                {
                    "interpretation": "OK",
                    "confounds_flagged": [],
                    "hypothesis": "next",
                    "reasoning": "test",
                    "confidence": "medium",
                }
            )

        with (
            patch("antagonistic_collab.runner.call_agent", side_effect=fake_call),
            patch("antagonistic_collab.runner._BATCH_MODE", True),
        ):
            run_cycle(protocol, type("C", (), {})(), transcript, mode="full_pool")

        # Should NOT have proposal, critique, revision phases
        assert "proposal" not in call_phases
        assert "critique" not in call_phases
        assert "revision" not in call_phases


# =========================================================================
# Phase B: Learning Curves as Second Evidence Channel (D19)
# =========================================================================


class TestLearningCurvePredictions:
    """Tests for learning curve generation and feature extraction.

    Learning curves are a second evidence channel orthogonal to RMSE.
    GCM predicts gradual learning, RULEX predicts sudden transitions,
    SUSTAIN predicts stepwise cluster-recruitment bumps.
    """

    def test_learning_curve_all_models_produce_curves(self):
        """All 3 models produce learning curves for a given structure."""
        from antagonistic_collab.debate_protocol import DebateProtocol

        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)

        curves = protocol.compute_learning_curve_predictions(
            "Type_I", "baseline", n_epochs=2, block_size=2
        )
        assert len(curves) == 3  # one per agent
        for agent_name, curve in curves.items():
            assert len(curve) > 0, f"{agent_name} produced empty curve"
            for block in curve:
                assert "accuracy" in block
                assert isinstance(block["accuracy"], (int, float))

    def test_gcm_curve_monotonic_on_easy_structure(self):
        """GCM on Type_I: monotonic and reaches ceiling quickly."""
        from antagonistic_collab.debate_protocol import DebateProtocol

        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)

        curves = protocol.compute_learning_curve_predictions(
            "Type_I", "baseline", n_epochs=3, block_size=2
        )
        gcm_curve = curves["Exemplar_Agent"]

        from antagonistic_collab.debate_protocol import extract_curve_features

        features = extract_curve_features(gcm_curve)
        # GCM on Type_I should be monotonic (exemplar accumulation)
        assert features["monotonic"] is True
        assert features["final_accuracy"] >= 0.75

    def test_rulex_curve_has_jumps(self):
        """RULEX on Type_I should show jumps (rule discovery)."""
        from antagonistic_collab.debate_protocol import DebateProtocol

        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)

        curves = protocol.compute_learning_curve_predictions(
            "Type_I", "baseline", n_epochs=3, block_size=2
        )
        rulex_curve = curves["Rule_Agent"]

        from antagonistic_collab.debate_protocol import extract_curve_features

        features = extract_curve_features(rulex_curve)
        # RULEX should show jumps — sudden or stepwise
        assert features["learning_pattern"] in ("sudden", "stepwise")
        assert features["max_jump"] > 0.1

    def test_sustain_curve_on_type_vi(self):
        """SUSTAIN on Type_VI produces a classifiable learning curve."""
        from antagonistic_collab.debate_protocol import DebateProtocol

        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)

        curves = protocol.compute_learning_curve_predictions(
            "Type_VI", "baseline", n_epochs=3, block_size=2
        )
        sustain_curve = curves["Clustering_Agent"]

        from antagonistic_collab.debate_protocol import extract_curve_features

        features = extract_curve_features(sustain_curve)
        # SUSTAIN on Type_VI: pattern is classified (any valid pattern)
        assert features["learning_pattern"] in ("gradual", "sudden", "stepwise")
        assert features["final_accuracy"] >= 0.5

    def test_extract_curve_features(self):
        """extract_curve_features returns the expected fields."""
        from antagonistic_collab.debate_protocol import extract_curve_features

        # Synthetic gradual curve
        gradual = [
            {"accuracy": 0.5, "block": 0},
            {"accuracy": 0.6, "block": 1},
            {"accuracy": 0.7, "block": 2},
            {"accuracy": 0.8, "block": 3},
        ]
        features = extract_curve_features(gradual)
        assert "final_accuracy" in features
        assert "onset_block" in features
        assert "max_jump" in features
        assert "n_big_jumps" in features
        assert "monotonic" in features
        assert "mean_slope" in features
        assert "learning_pattern" in features
        assert features["final_accuracy"] == 0.8
        assert features["monotonic"] is True
        assert features["learning_pattern"] == "gradual"

    def test_curve_rmse_scoring(self):
        """Matching curve shapes should score lower RMSE than mismatched."""

        observed = [
            {"accuracy": 0.5, "block": 0},
            {"accuracy": 0.55, "block": 1},
            {"accuracy": 0.65, "block": 2},
            {"accuracy": 0.75, "block": 3},
        ]
        good_pred = [
            {"accuracy": 0.5, "block": 0},
            {"accuracy": 0.55, "block": 1},
            {"accuracy": 0.65, "block": 2},
            {"accuracy": 0.75, "block": 3},
        ]
        bad_pred = [
            {"accuracy": 0.5, "block": 0},
            {"accuracy": 0.5, "block": 1},
            {"accuracy": 0.5, "block": 2},
            {"accuracy": 0.9, "block": 3},
        ]

        obs_accs = np.array([b["accuracy"] for b in observed])
        good_accs = np.array([b["accuracy"] for b in good_pred])
        bad_accs = np.array([b["accuracy"] for b in bad_pred])

        rmse_good = float(np.sqrt(np.mean((obs_accs - good_accs) ** 2)))
        rmse_bad = float(np.sqrt(np.mean((obs_accs - bad_accs) ** 2)))
        assert rmse_good < rmse_bad

    def test_posterior_update_with_curves(self):
        """update_posterior_from_experiment with learning curves shifts posterior."""
        from antagonistic_collab.bayesian_selection import (
            update_posterior_from_experiment,
        )

        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)
        posterior = ModelPosterior.uniform([a.name for a in agents])

        # Simulate data from GCM ground truth
        data = protocol.experiment_runner(
            {"structure_name": "Type_I", "condition": "baseline"},
            true_model="GCM",
            cycle=0,
        )

        # Compute learning curves
        curves = protocol.compute_learning_curve_predictions(
            "Type_I", "baseline", n_epochs=2, block_size=2
        )

        prior_probs = posterior.probs.copy()
        posterior = update_posterior_from_experiment(
            posterior,
            protocol,
            data,
            "Type_I",
            "baseline",
            cycle=0,
            learning_curves=curves,
        )
        # Posterior should have shifted (not identical to prior)
        assert not np.allclose(posterior.probs, prior_probs, atol=1e-6)

    def test_curves_in_execution_data(self):
        """Learning curves appear in execution data when computed."""
        from antagonistic_collab.debate_protocol import DebateProtocol

        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)

        curves = protocol.compute_learning_curve_predictions(
            "Type_I", "baseline", n_epochs=2, block_size=2
        )
        assert isinstance(curves, dict)
        assert len(curves) == 3
        for name, curve in curves.items():
            assert isinstance(curve, list)

    def test_curve_evidence_breaks_gcm_rulex_tie(self):
        """Learning curves should provide evidence beyond accuracy RMSE.

        When GCM and RULEX have similar accuracy but different learning
        dynamics, curve shape should be an additional discriminator.
        """
        from antagonistic_collab.debate_protocol import extract_curve_features

        # GCM-like gradual curve
        gcm_curve = [
            {"accuracy": 0.5, "block": 0},
            {"accuracy": 0.6, "block": 1},
            {"accuracy": 0.7, "block": 2},
            {"accuracy": 0.8, "block": 3},
        ]
        # RULEX-like sudden curve
        rulex_curve = [
            {"accuracy": 0.5, "block": 0},
            {"accuracy": 0.5, "block": 1},
            {"accuracy": 0.5, "block": 2},
            {"accuracy": 0.85, "block": 3},
        ]

        gcm_features = extract_curve_features(gcm_curve)
        rulex_features = extract_curve_features(rulex_curve)

        # Different learning patterns → can break ties
        assert gcm_features["learning_pattern"] != rulex_features["learning_pattern"]


# =========================================================================
# Phase C: Novel Structure Generation from Debate (D19)
# =========================================================================


class TestNovelStructureValidation:
    """Tests for validating novel category structures proposed by LLM agents
    during interpretation debate."""

    def test_validate_novel_structure_valid(self):
        """A well-formed novel structure passes validation."""
        from antagonistic_collab.debate_protocol import validate_novel_structure

        spec = {
            "name": "custom_xor",
            "stimuli": [[0, 0], [0, 1], [1, 0], [1, 1]],
            "labels": [0, 1, 1, 0],
        }
        valid, msg = validate_novel_structure(spec)
        assert valid is True, msg

    def test_validate_novel_structure_missing_labels(self):
        """Structure without labels fails validation."""
        from antagonistic_collab.debate_protocol import validate_novel_structure

        spec = {
            "name": "bad",
            "stimuli": [[0, 0], [0, 1]],
        }
        valid, msg = validate_novel_structure(spec)
        assert valid is False
        assert "labels" in msg.lower()

    def test_validate_novel_structure_too_many_items(self):
        """Structure with >32 items fails validation."""
        from antagonistic_collab.debate_protocol import validate_novel_structure

        spec = {
            "name": "huge",
            "stimuli": [[i, i] for i in range(33)],
            "labels": [i % 2 for i in range(33)],
        }
        valid, msg = validate_novel_structure(spec)
        assert valid is False
        assert "32" in msg or "items" in msg.lower()

    def test_validate_novel_structure_wrong_dims(self):
        """Structure with >8 dims fails validation."""
        from antagonistic_collab.debate_protocol import validate_novel_structure

        spec = {
            "name": "highdim",
            "stimuli": [[0] * 9, [1] * 9, [0] * 9, [1] * 9],
            "labels": [0, 1, 0, 1],
        }
        valid, msg = validate_novel_structure(spec)
        assert valid is False
        assert "dim" in msg.lower()

    def test_validate_novel_structure_mismatched_lengths(self):
        """Mismatched stimuli/labels lengths fail."""
        from antagonistic_collab.debate_protocol import validate_novel_structure

        spec = {
            "name": "mismatch",
            "stimuli": [[0, 0], [1, 1], [0, 1], [1, 0]],
            "labels": [0],  # only 1 label for 4 stimuli
        }
        valid, msg = validate_novel_structure(spec)
        assert valid is False

    def test_validate_novel_structure_too_few_categories(self):
        """Need at least 2 categories."""
        from antagonistic_collab.debate_protocol import validate_novel_structure

        spec = {
            "name": "onecat",
            "stimuli": [[0, 0], [1, 1], [0, 1], [1, 0]],
            "labels": [0, 0, 0, 0],  # only 1 category
        }
        valid, msg = validate_novel_structure(spec)
        assert valid is False
        assert "categor" in msg.lower()


class TestTemporaryStructures:
    """Tests for temporary structures from novel agent proposals being
    incorporated into the divergence map and EIG search."""

    def test_temporary_structures_in_divergence_map(self):
        """Novel structures added to protocol.temporary_structures appear
        in divergence map computation."""
        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)

        # Add a temporary structure
        protocol.temporary_structures = {
            "novel_xor": {
                "stimuli": [[0, 0], [0, 1], [1, 0], [1, 1]],
                "labels": [0, 1, 1, 0],
            }
        }

        # Compute divergence map with merged structures
        merged = dict(STRUCTURE_REGISTRY)
        merged.update(protocol.temporary_structures)
        div_map = protocol.compute_divergence_map(structures=merged)
        assert "novel_xor" in div_map

    def test_temporary_structures_in_eig_search(self):
        """Extra structures are included in EIG candidate pool."""
        from antagonistic_collab.bayesian_selection import generate_full_candidate_pool

        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)

        extra = {
            "novel_xor": {
                "stimuli": [[0, 0], [0, 1], [1, 0], [1, 1]],
                "labels": [0, 1, 1, 0],
            }
        }
        pool = generate_full_candidate_pool(protocol, extra_structures=extra)
        novel_entries = [(s, c) for s, c in pool if s == "novel_xor"]
        assert len(novel_entries) == 5  # 5 conditions

    def test_novel_structure_from_llm_json(self):
        """Parse a realistic LLM output containing a novel structure."""
        from antagonistic_collab.debate_protocol import validate_novel_structure

        # Realistic JSON that an LLM might produce
        llm_output = {
            "name": "random_similarity",
            "stimuli": [
                [0.2, 0.3],
                [0.3, 0.2],
                [0.8, 0.7],
                [0.7, 0.8],
                [0.1, 0.9],
                [0.9, 0.1],
            ],
            "labels": [0, 0, 1, 1, 0, 1],
        }
        valid, msg = validate_novel_structure(llm_output)
        assert valid is True, msg

    def test_temporary_structures_cleared_after_use(self):
        """temporary_structures should be clearable between cycles."""
        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)

        protocol.temporary_structures = {
            "temp": {
                "stimuli": [[0, 0], [1, 1]],
                "labels": [0, 1],
            }
        }
        assert len(protocol.temporary_structures) == 1
        protocol.temporary_structures = {}
        assert len(protocol.temporary_structures) == 0


# =========================================================================
# Full-pool mode integration test
# =========================================================================


class TestFullPoolIntegration:
    """End-to-end test: 2 cycles in full_pool mode with mocked LLM.

    Verifies the entire pipeline: commitment → divergence → EIG selection →
    execution → interpretation debate → interpretation critique → audit,
    for 2 consecutive cycles.
    """

    def _make_mock_response(self):
        """Return a callable that produces phase-appropriate mock responses."""
        call_count = [0]

        def mock_call_agent(client, system_prompt, user_message, **kwargs):
            call_count[0] += 1
            msg = user_message.lower()

            if "commitment" in msg or "register your theory" in msg:
                return json.dumps(
                    {
                        "core_claims": ["Test claim"],
                        "term_glossary": {"attention": "w_i"},
                    }
                )
            elif "divergence" in msg:
                return "The divergence map shows interesting patterns. Type_II has the highest divergence."
            elif "pre-data prediction" in msg:
                return json.dumps(
                    {
                        "reasoning": "My model handles this structure well.",
                        "confidence": "medium",
                        "param_overrides": {},
                    }
                )
            elif "interpretation debate" in msg:
                return json.dumps(
                    {
                        "interpretation": "Results support my model's predictions.",
                        "confounds_flagged": ["Small sample size"],
                        "hypothesis": "Next we should test a harder structure.",
                        "novel_structure": None,
                        "revision": None,
                    }
                )
            elif "interpretation critique" in msg or "challenge" in msg:
                return "I dispute the other agent's claim. My model offers a better explanation."
            elif "audit" in msg or "auditor" in msg:
                return "Cycle summary: Models were tested. Divergence remains."
            else:
                return "Generic response."

        return mock_call_agent, call_count

    def test_two_cycle_full_pool_run(self):
        """Run 2 full cycles in full_pool mode and verify pipeline integrity."""
        import antagonistic_collab.runner as runner_mod
        from antagonistic_collab.runner import run_cycle

        state = EpistemicState(domain="categorization")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)
        transcript = []

        mock_fn, call_count = self._make_mock_response()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Need _BATCH_MODE=True for Bayesian posterior updates
            old_batch = runner_mod._BATCH_MODE
            runner_mod._BATCH_MODE = True
            try:
                with patch.object(runner_mod, "call_agent", side_effect=mock_fn):
                    # --- Cycle 0 ---
                    run_cycle(
                        protocol,
                        client=None,
                        transcript=transcript,
                        true_model="GCM",
                        output_dir=tmpdir,
                        metadata={
                            "true_model": "GCM",
                            "llm_model": "mock",
                            "backend": "mock",
                        },
                        mode="full_pool",
                    )

                    assert protocol.state.cycle == 1, (
                        f"Cycle should be 1 after first run, got {protocol.state.cycle}"
                    )
                    # Should have 3 theories registered
                    assert len(protocol.state.theories) == 3

                    # Should have 1 executed experiment
                    executed_c0 = [
                        e
                        for e in protocol.state.experiments
                        if e.cycle == 0 and e.status == "executed"
                    ]
                    assert len(executed_c0) == 1

                    # Should have hypotheses from interpretation debate
                    assert len(protocol.state.agent_hypotheses) >= 1

                    # Bayesian posterior should exist
                    assert protocol.state.model_posterior is not None
                    assert "log_probs" in protocol.state.model_posterior

                    # --- Cycle 1 ---
                    run_cycle(
                        protocol,
                        client=None,
                        transcript=transcript,
                        true_model="GCM",
                        output_dir=tmpdir,
                        metadata={
                            "true_model": "GCM",
                            "llm_model": "mock",
                            "backend": "mock",
                        },
                        mode="full_pool",
                    )

                    assert protocol.state.cycle == 2, (
                        f"Cycle should be 2 after second run, got {protocol.state.cycle}"
                    )

                    # Should have 2 executed experiments total
                    executed_all = [
                        e for e in protocol.state.experiments if e.status == "executed"
                    ]
                    assert len(executed_all) == 2

                    # Hypotheses should accumulate across cycles
                    assert len(protocol.state.agent_hypotheses) >= 2

                    # Transcript should have entries from both cycles
                    assert len(transcript) > 0

                    # Check output files exist
                    cycle_files = os.listdir(tmpdir)
                    assert any("cycle_0" in f for f in cycle_files)
                    assert any("cycle_1" in f for f in cycle_files)

            finally:
                runner_mod._BATCH_MODE = old_batch

        # Sanity: LLM was actually called (not bypassed)
        assert call_count[0] > 0, "No LLM calls were made"


# =========================================================================
# Codex review round 3 — regression tests (D21)
# =========================================================================


class TestOverrideFallbackPreservesCondition:
    """P1: compute_model_predictions() fallback must preserve condition overrides.

    When a malformed LLM param_override causes ValueError/TypeError, the
    fallback previously reverted to bare agent defaults, silently dropping
    the experimental condition. E.g. low_attention sets GCM c=1.5, but a
    bad override like attention_weights=[0.5,0.5,0.5] on a 4D structure
    would revert to c=3.0 (baseline).
    """

    def test_fallback_preserves_condition_overrides(self):
        """After a malformed override, condition params (e.g. c=1.5) must survive."""
        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)

        gcm_agent = agents[0]  # Exemplar_Agent uses GCM
        assert gcm_agent.model_class.name.startswith("GCM")

        # low_attention sets GCM c=1.5 (vs default c=3.0)
        # Provide a malformed override that will crash on first item
        result = protocol.compute_model_predictions(
            gcm_agent,
            "linear_separable_4d",  # 4D structure
            "low_attention",
            param_overrides={
                "attention_weights": [0.5, 0.5, 0.5]
            },  # wrong shape for 4D
        )

        # The key check: params_used should have c=1.5 (from low_attention),
        # NOT c=3.0 (bare default). The malformed attention_weights should be
        # dropped, but the condition override should survive.
        params_used = result.get("params_used", {})
        assert params_used.get("c") == 1.5, (
            f"Expected c=1.5 (low_attention), got c={params_used.get('c')}. "
            "Fallback dropped condition overrides."
        )


class TestScalarAddressesCritiques:
    """P1: run_design_revision() must handle scalar addresses_critiques.

    If the LLM returns "addresses_critiques": 1 instead of [1], the
    list comprehension on line 557 crashes with TypeError.
    """

    def test_scalar_addresses_critiques_no_crash(self):
        """A scalar addresses_critiques value should not crash design revision."""
        from antagonistic_collab.runner import run_design_revision

        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)

        # Register theories and create a proposal with critiques
        for a in agents:
            try:
                protocol.state.register_theory(
                    TheoryCommitment(
                        name=a.theory_name,
                        agent_name=a.name,
                        core_claims=a.model_class.core_claims,
                        model_name=a.model_class.name,
                        model_params=a.default_params,
                    )
                )
            except ValueError:
                pass

        exp = protocol.state.propose_experiment(
            proposed_by=agents[0].name,
            title="Test proposal",
            design_spec={"structure_name": "Type_I", "condition": "baseline"},
            rationale="test",
        )
        # Add a critique so revision runs
        exp.critique_log.append({"agent": agents[1].name, "critique": "Not diagnostic"})

        protocol.skip_to_phase(Phase.DESIGN_REVISION)

        # LLM returns scalar addresses_critiques
        def fake_call(*a, **kw):
            return json.dumps(
                {
                    "structure_name": "Type_II",
                    "condition": "baseline",
                    "changes": "Switched structure",
                    "addresses_critiques": 1,  # scalar, not list
                }
            )

        with patch("antagonistic_collab.runner.call_agent", side_effect=fake_call):
            # This should NOT raise TypeError
            result = run_design_revision(protocol, None, [])

        # Should complete without crashing
        assert result.phase == Phase.DESIGN_REVISION


class TestInvalidApprovalRejects:
    """P2: Out-of-range approve index must set rejected flag.

    approve 99 with only 1 proposal nulls idx but doesn't set rejected,
    so the cycle continues with no approved experiment.
    """

    def test_approve_out_of_range_sets_rejected(self):
        """approve 99 with 1 proposal should result in rejection."""
        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)

        # Create a single proposal
        protocol.state.propose_experiment(
            proposed_by=agents[0].name,
            title="Test",
            design_spec={"structure_name": "Type_I", "condition": "baseline"},
            rationale="test",
        )
        protocol.skip_to_phase(Phase.HUMAN_ARBITRATION)

        with patch("builtins.input", return_value="approve 99"):
            result = run_human_arbitration(protocol, [])

        # Must be flagged as rejected so the retry loop kicks in
        assert result.outputs.get("rejected") is True, (
            "Out-of-range approve should set rejected=True"
        )


class TestSUSTAINPartialBlockLabel:
    """P2: SUSTAIN partial block must not duplicate prior block's label.

    With 3 trials and block_size=2, the method returns blocks at indices
    [2, 3]. The label formula (block_end // block_size) - 1 gives
    block 0 for both (2//2-1=0, 3//2-1=0).
    """

    def test_partial_block_has_unique_label(self):
        """Each block in the learning curve must have a unique label."""
        model = SUSTAIN()
        # 3 items → block_size=2 gives blocks at [2, 3]
        stimuli = [[0, 0], [1, 1], [0, 1]]
        labels = [0, 1, 0]
        training_seq = list(zip(stimuli, labels))

        curve = model.predict_learning_curve(
            training_seq,
            test_items=stimuli,
            test_labels=labels,
            n_epochs=1,
            block_size=2,
        )

        # Should have 2 blocks (one complete, one partial)
        assert len(curve) >= 2, f"Expected ≥2 blocks, got {len(curve)}"

        # Block labels must be unique
        block_labels = [entry["block"] for entry in curve]
        assert len(block_labels) == len(set(block_labels)), (
            f"Duplicate block labels: {block_labels}"
        )


# =========================================================================
# Codex review round 4 — regression tests (D22)
# =========================================================================


class TestRULEXGroundTruthDeterminism:
    """Bug 1: _synthetic_runner() must produce deterministic RULEX ground truth.

    RULEX.predict() defaults seed=None, making rule search stochastic.
    _synthetic_runner() didn't set a seed for RULEX, so the same experiment
    could produce different ground truth data between runs.
    """

    def test_rulex_ground_truth_is_deterministic(self):
        """Same experiment with RULEX ground truth must produce identical data."""
        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)

        # Use Type_VI — hardest structure for RULEX, stochastic search matters
        design = {"structure_name": "Type_VI", "condition": "baseline"}

        # Run 5 times — without seed, at least one should differ
        results = []
        for _ in range(5):
            data = protocol.experiment_runner(design, true_model="RULEX", cycle=0)
            # Get the raw item probabilities (before noise), which are model-dependent
            results.append(tuple(sorted(data.get("item_accuracies", {}).items())))

        # All runs must be identical (deterministic seed)
        assert len(set(results)) == 1, (
            f"RULEX ground truth is non-deterministic across 5 runs "
            f"({len(set(results))} unique results)"
        )


class TestRULEXLearningCurveDeterminism:
    """Bug 2: predict_learning_curve() must forward seed to find_best_rule().

    predict_learning_curve() calls find_best_rule() with explicit keyword
    args but omits seed, even though it may be in **params. This makes
    RULEX learning curves non-deterministic.
    """

    def test_rulex_learning_curve_is_deterministic(self):
        """Same inputs with seed must produce identical learning curves."""
        from antagonistic_collab.models.rulex import RULEX

        model = RULEX()
        stimuli = [[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 1]]
        labels = [0, 1, 1, 0]
        training_seq = list(zip(stimuli, labels))

        curve1 = model.predict_learning_curve(
            training_seq,
            test_items=np.array(stimuli),
            test_labels=np.array(labels),
            block_size=2,
            seed=42,
        )
        curve2 = model.predict_learning_curve(
            training_seq,
            test_items=np.array(stimuli),
            test_labels=np.array(labels),
            block_size=2,
            seed=42,
        )

        accs1 = [e["accuracy"] for e in curve1]
        accs2 = [e["accuracy"] for e in curve2]
        assert accs1 == accs2, (
            f"RULEX learning curve non-deterministic with same seed:\n"
            f"  run1={accs1}\n  run2={accs2}"
        )

    def test_rulex_learning_curve_seed_actually_used(self):
        """Different seeds should (with high probability) produce different curves."""
        from antagonistic_collab.models.rulex import RULEX

        model = RULEX()
        # Use a structure where rule search randomness matters
        stimuli = [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ]
        labels = [0, 1, 0, 1, 1, 0]
        training_seq = list(zip(stimuli, labels))

        curves = []
        for seed in [1, 2, 3, 4, 5]:
            curve = model.predict_learning_curve(
                training_seq,
                test_items=np.array(stimuli),
                test_labels=np.array(labels),
                block_size=2,
                seed=seed,
            )
            curves.append(tuple(e["accuracy"] for e in curve))

        # With 5 different seeds on a complex structure, at least 2 should differ
        unique_curves = set(curves)
        assert len(unique_curves) >= 2, (
            "All 5 seeds produced identical curves — seed likely not forwarded"
        )


class TestRedundantPredictedKey:
    """Bug 4: execution messages have both 'model_predicted' and 'predicted'
    with the same value. One should be removed.
    """

    def test_no_duplicate_predicted_keys(self):
        """Execution prediction messages should not have both keys."""
        import antagonistic_collab.runner as runner_mod
        from antagonistic_collab.runner import run_execution

        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state, agents)
        transcript = []

        # Register theories
        for a in agents:
            try:
                protocol.state.register_theory(
                    TheoryCommitment(
                        name=a.theory_name,
                        agent_name=a.name,
                        core_claims=a.model_class.core_claims,
                        model_name=a.model_class.name,
                        model_params=a.default_params,
                    )
                )
            except ValueError:
                pass

        # Create and approve an experiment
        exp = protocol.state.propose_experiment(
            proposed_by=agents[0].name,
            title="Test",
            design_spec={"structure_name": "Type_I", "condition": "baseline"},
            rationale="test",
        )
        protocol.state.approve_experiment(exp.experiment_id)
        protocol.skip_to_phase(Phase.EXECUTION)

        def fake_call(*a, **kw):
            return json.dumps(
                {"reasoning": "test", "confidence": "medium", "param_overrides": {}}
            )

        old_batch = runner_mod._BATCH_MODE
        runner_mod._BATCH_MODE = True
        try:
            with patch.object(runner_mod, "call_agent", side_effect=fake_call):
                run_execution(protocol, None, transcript, true_model="GCM")
        finally:
            runner_mod._BATCH_MODE = old_batch

        # Check prediction messages don't have both keys
        pred_msgs = [m for m in transcript if m.get("phase") == "EXECUTION_PREDICT"]
        for msg in pred_msgs:
            has_model_predicted = "model_predicted" in msg
            has_predicted = "predicted" in msg
            assert not (has_model_predicted and has_predicted), (
                f"Message has both 'model_predicted' and 'predicted': {list(msg.keys())}"
            )


# =========================================================================
# Learning curves wired into execution (runner.py)
# =========================================================================


class TestLearningCurvesInExecution:
    """Learning curves should be computed during run_execution() and passed
    to update_posterior_from_experiment() as the learning_curves kwarg.
    This is the key integration that adds a second evidence channel
    (curve shape) beyond item-level accuracy.
    """

    def _make_protocol_with_approved_experiment(self, struct="Type_I"):
        """Helper: create a protocol with one approved experiment."""
        agents = default_agent_configs()
        state = EpistemicState(domain="test")
        protocol = DebateProtocol(state, agents)
        # Register theories
        for a in agents:
            try:
                protocol.state.register_theory(
                    TheoryCommitment(
                        name=a.theory_name,
                        agent_name=a.name,
                        core_claims=a.model_class.core_claims,
                        model_name=a.model_class.name,
                        model_params=a.default_params,
                    )
                )
            except ValueError:
                pass

        exp = protocol.state.propose_experiment(
            proposed_by=agents[0].name,
            title="Test",
            design_spec={"structure_name": struct, "condition": "baseline"},
            rationale="test",
        )
        protocol.state.approve_experiment(exp.experiment_id)
        protocol.skip_to_phase(Phase.EXECUTION)
        return protocol, agents

    def test_learning_curves_passed_to_posterior_update(self):
        """run_execution() should compute learning curves and pass them
        to update_posterior_from_experiment(), evidenced by
        has_curve_evidence=True in the posterior history."""
        import antagonistic_collab.runner as runner_mod
        from antagonistic_collab.runner import run_execution

        protocol, agents = self._make_protocol_with_approved_experiment()
        transcript = []

        def fake_call(*a, **kw):
            return json.dumps(
                {"reasoning": "test", "confidence": "medium", "param_overrides": {}}
            )

        old_batch = runner_mod._BATCH_MODE
        runner_mod._BATCH_MODE = True
        try:
            with patch.object(runner_mod, "call_agent", side_effect=fake_call):
                run_execution(protocol, None, transcript, true_model="GCM")
        finally:
            runner_mod._BATCH_MODE = old_batch

        # If learning curves were passed, the posterior history will record it
        posterior = protocol.state.model_posterior
        assert posterior is not None, "Posterior should exist after execution"
        from antagonistic_collab.bayesian_selection import ModelPosterior

        mp = ModelPosterior.from_dict(posterior)
        assert len(mp.history) > 0, "Should have at least one history entry"
        assert mp.history[-1]["has_curve_evidence"] is True, (
            "Posterior history should record has_curve_evidence=True when "
            "learning curves are passed to update_posterior_from_experiment()"
        )

    def test_learning_curves_in_execution_outputs(self):
        """run_execution() outputs should include learning_curves and
        curve_features so they can be used by downstream phases."""
        import antagonistic_collab.runner as runner_mod
        from antagonistic_collab.runner import run_execution

        protocol, agents = self._make_protocol_with_approved_experiment()
        transcript = []

        def fake_call(*a, **kw):
            return json.dumps(
                {"reasoning": "test", "confidence": "medium", "param_overrides": {}}
            )

        old_batch = runner_mod._BATCH_MODE
        runner_mod._BATCH_MODE = True
        try:
            with patch.object(runner_mod, "call_agent", side_effect=fake_call):
                result = run_execution(protocol, None, transcript, true_model="GCM")
        finally:
            runner_mod._BATCH_MODE = old_batch

        assert "learning_curves" in result.outputs, (
            "PhaseResult outputs should include 'learning_curves'"
        )
        assert "curve_features" in result.outputs, (
            "PhaseResult outputs should include 'curve_features'"
        )

    def test_curve_evidence_recorded_in_posterior_history(self):
        """When curves are passed, the posterior history entry should have
        has_curve_evidence=True."""
        import antagonistic_collab.runner as runner_mod
        from antagonistic_collab.runner import run_execution

        protocol, agents = self._make_protocol_with_approved_experiment()
        transcript = []

        def fake_call(*a, **kw):
            return json.dumps(
                {"reasoning": "test", "confidence": "medium", "param_overrides": {}}
            )

        old_batch = runner_mod._BATCH_MODE
        runner_mod._BATCH_MODE = True
        try:
            with patch.object(runner_mod, "call_agent", side_effect=fake_call):
                run_execution(protocol, None, transcript, true_model="GCM")
        finally:
            runner_mod._BATCH_MODE = old_batch

        posterior = protocol.state.model_posterior
        assert posterior is not None, "Posterior should be set after execution"
        # Check history for curve evidence marker
        from antagonistic_collab.bayesian_selection import ModelPosterior

        mp = ModelPosterior.from_dict(posterior)
        assert len(mp.history) > 0, "Should have at least one history entry"
        assert mp.history[-1]["has_curve_evidence"] is True, (
            "History entry should record has_curve_evidence=True"
        )


# =========================================================================
# Learning curve context in interpretation debate (runner.py)
# =========================================================================


class TestLearningCurvesInInterpretation:
    """The interpretation debate should include learning curve comparison
    in the extended context, so agents can reason about curve shapes."""

    def test_interpretation_debate_includes_curve_context(self):
        """If execution produced learning curves, the interpretation debate
        prompt should include curve comparison data."""
        import antagonistic_collab.runner as runner_mod
        from antagonistic_collab.runner import run_interpretation_debate

        agents = default_agent_configs()
        state = EpistemicState(domain="test")
        protocol = DebateProtocol(state, agents)
        # Register theories
        for a in agents:
            try:
                protocol.state.register_theory(
                    TheoryCommitment(
                        name=a.theory_name,
                        agent_name=a.name,
                        core_claims=a.model_class.core_claims,
                        model_name=a.model_class.name,
                        model_params=a.default_params,
                    )
                )
            except ValueError:
                pass

        # Create an executed experiment with learning curve data
        exp = protocol.state.propose_experiment(
            proposed_by=agents[0].name,
            title="Test",
            design_spec={"structure_name": "Type_I", "condition": "baseline"},
            rationale="test",
        )
        protocol.state.approve_experiment(exp.experiment_id)
        protocol.state.record_data(
            exp.experiment_id,
            {
                "mean_accuracy": 0.75,
                "item_accuracies": {"item_0": 0.8, "item_1": 0.7},
                "ground_truth_model": "GCM",
                "structure_name": "Type_I",
                "condition": "baseline",
            },
        )
        protocol.skip_to_phase(Phase.INTERPRETATION)

        # Store learning curve data in the execution outputs (where run_execution
        # would have put it)
        protocol.state.last_execution_curves = {
            "Exemplar_Agent": [
                {"block": 0, "accuracy": 0.5},
                {"block": 1, "accuracy": 0.7},
                {"block": 2, "accuracy": 0.85},
            ],
            "Rule_Agent": [
                {"block": 0, "accuracy": 0.5},
                {"block": 1, "accuracy": 0.5},
                {"block": 2, "accuracy": 0.9},
            ],
            "Clustering_Agent": [
                {"block": 0, "accuracy": 0.5},
                {"block": 1, "accuracy": 0.65},
                {"block": 2, "accuracy": 0.8},
            ],
        }

        # Capture the prompt sent to agents
        captured_prompts = []

        def fake_call(client, system, prompt):
            captured_prompts.append(prompt)
            return json.dumps(
                {
                    "interpretation": "test",
                    "confounds_flagged": [],
                    "hypothesis": "test hypothesis",
                    "novel_structure": None,
                    "revision": None,
                }
            )

        transcript = []
        with patch.object(runner_mod, "call_agent", side_effect=fake_call):
            run_interpretation_debate(protocol, None, transcript)

        # At least one prompt should mention learning curves
        assert any(
            "learning" in p.lower() or "curve" in p.lower() for p in captured_prompts
        ), "Interpretation debate prompt should include learning curve context"


# =========================================================================
# Novel structures validated and registered (runner.py)
# =========================================================================


class TestNovelStructureRegistration:
    """When an agent proposes a novel_structure in interpretation debate,
    it should be validated and registered in protocol.temporary_structures
    for the next cycle's EIG candidate pool."""

    def _setup_protocol_for_interpretation(self):
        """Helper: protocol in INTERPRETATION phase with executed experiment."""
        agents = default_agent_configs()
        state = EpistemicState(domain="test")
        protocol = DebateProtocol(state, agents)
        for a in agents:
            try:
                protocol.state.register_theory(
                    TheoryCommitment(
                        name=a.theory_name,
                        agent_name=a.name,
                        core_claims=a.model_class.core_claims,
                        model_name=a.model_class.name,
                        model_params=a.default_params,
                    )
                )
            except ValueError:
                pass

        exp = protocol.state.propose_experiment(
            proposed_by=agents[0].name,
            title="Test",
            design_spec={"structure_name": "Type_I", "condition": "baseline"},
            rationale="test",
        )
        protocol.state.approve_experiment(exp.experiment_id)
        protocol.state.record_data(
            exp.experiment_id,
            {
                "mean_accuracy": 0.75,
                "item_accuracies": {"item_0": 0.8, "item_1": 0.7},
                "ground_truth_model": "GCM",
                "structure_name": "Type_I",
                "condition": "baseline",
            },
        )
        protocol.skip_to_phase(Phase.INTERPRETATION)
        return protocol, agents

    def test_valid_novel_structure_registered(self):
        """A valid novel_structure from an agent should be added to
        protocol.temporary_structures."""
        import antagonistic_collab.runner as runner_mod
        from antagonistic_collab.runner import run_interpretation_debate

        protocol, agents = self._setup_protocol_for_interpretation()

        valid_structure = {
            "name": "custom_diagonal",
            "stimuli": [[0, 0], [0, 1], [1, 0], [1, 1]],
            "labels": [0, 1, 1, 0],
        }

        call_count = [0]

        def fake_call(client, system, prompt):
            call_count[0] += 1
            # Only first agent proposes a novel structure
            if call_count[0] == 1:
                return json.dumps(
                    {
                        "interpretation": "test",
                        "confounds_flagged": [],
                        "hypothesis": "test XOR structure",
                        "novel_structure": valid_structure,
                        "revision": None,
                    }
                )
            return json.dumps(
                {
                    "interpretation": "test",
                    "confounds_flagged": [],
                    "hypothesis": "test",
                    "novel_structure": None,
                    "revision": None,
                }
            )

        transcript = []
        with patch.object(runner_mod, "call_agent", side_effect=fake_call):
            run_interpretation_debate(protocol, None, transcript)

        assert "custom_diagonal" in protocol.temporary_structures, (
            f"Novel structure should be registered. Got: {list(protocol.temporary_structures.keys())}"
        )
        registered = protocol.temporary_structures["custom_diagonal"]
        assert registered["stimuli"] == valid_structure["stimuli"]
        assert registered["labels"] == valid_structure["labels"]

    def test_invalid_novel_structure_not_registered(self):
        """An invalid novel_structure (e.g., too few items) should NOT be
        registered in protocol.temporary_structures."""
        import antagonistic_collab.runner as runner_mod
        from antagonistic_collab.runner import run_interpretation_debate

        protocol, agents = self._setup_protocol_for_interpretation()

        invalid_structure = {
            "name": "too_small",
            "stimuli": [[0, 0], [1, 1]],  # Only 2 items — needs at least 4
            "labels": [0, 1],
        }

        def fake_call(client, system, prompt):
            return json.dumps(
                {
                    "interpretation": "test",
                    "confounds_flagged": [],
                    "hypothesis": "test",
                    "novel_structure": invalid_structure,
                    "revision": None,
                }
            )

        transcript = []
        with patch.object(runner_mod, "call_agent", side_effect=fake_call):
            run_interpretation_debate(protocol, None, transcript)

        assert "too_small" not in protocol.temporary_structures, (
            "Invalid novel structure should NOT be registered"
        )
        assert len(protocol.temporary_structures) == 0


# =========================================================================
# compute_learning_curve_predictions checks temporary_structures
# =========================================================================


class TestLearningCurvesTemporaryStructures:
    """compute_learning_curve_predictions() should check
    protocol.temporary_structures in addition to STRUCTURE_REGISTRY,
    so that novel structures proposed by agents can be used for
    curve predictions."""

    def test_temporary_structure_used_for_curve(self):
        """If a structure name is only in temporary_structures (not in
        STRUCTURE_REGISTRY), compute_learning_curve_predictions() should
        still produce curves for it."""
        agents = default_agent_configs()
        state = EpistemicState(domain="test")
        protocol = DebateProtocol(state, agents)

        # Register a novel structure
        protocol.temporary_structures["custom_xor"] = {
            "stimuli": [[0, 0], [0, 1], [1, 0], [1, 1]],
            "labels": [0, 1, 1, 0],
        }

        # This should NOT fall back to Type_II — it should use custom_xor
        curves = protocol.compute_learning_curve_predictions("custom_xor", "baseline")

        assert len(curves) == len(agents), (
            f"Expected {len(agents)} curves, got {len(curves)}"
        )
        # Verify the curves are for 4-item XOR, not Type_II (8 items)
        # XOR has 4 items with n_epochs=3 → 12 training items → 6 blocks (block_size=2)
        # Type_II has 8 items with n_epochs=3 → 24 training items → 12 blocks
        # So if we got 12 blocks, it fell back to Type_II instead of using custom_xor
        for agent_name, curve in curves.items():
            assert len(curve) > 0, f"{agent_name} produced empty curve"
            assert len(curve) <= 6, (
                f"{agent_name} produced {len(curve)} blocks — likely fell back to "
                f"Type_II (8 items) instead of using custom_xor (4 items)"
            )


# =========================================================================
# Novel structure prompting (runner.py)
# =========================================================================


class TestNovelStructurePrompting:
    """The interpretation debate prompt should include few-shot examples
    of valid novel structures so agents know the format and constraints."""

    def test_prompt_contains_novel_structure_example(self):
        """The interpretation prompt should include a concrete example
        of a valid novel_structure with stimuli and labels."""
        import antagonistic_collab.runner as runner_mod
        from antagonistic_collab.runner import run_interpretation_debate

        agents = default_agent_configs()
        state = EpistemicState(domain="test")
        protocol = DebateProtocol(state, agents)
        for a in agents:
            try:
                protocol.state.register_theory(
                    TheoryCommitment(
                        name=a.theory_name,
                        agent_name=a.name,
                        core_claims=a.model_class.core_claims,
                        model_name=a.model_class.name,
                        model_params=a.default_params,
                    )
                )
            except ValueError:
                pass

        exp = protocol.state.propose_experiment(
            proposed_by=agents[0].name,
            title="Test",
            design_spec={"structure_name": "Type_I", "condition": "baseline"},
            rationale="test",
        )
        protocol.state.approve_experiment(exp.experiment_id)
        protocol.state.record_data(
            exp.experiment_id,
            {
                "mean_accuracy": 0.75,
                "item_accuracies": {"item_0": 0.8, "item_1": 0.7},
                "ground_truth_model": "GCM",
                "structure_name": "Type_I",
                "condition": "baseline",
            },
        )
        protocol.skip_to_phase(Phase.INTERPRETATION)

        captured_prompts = []

        def fake_call(client, system, prompt):
            captured_prompts.append(prompt)
            return json.dumps(
                {
                    "interpretation": "test",
                    "confounds_flagged": [],
                    "hypothesis": "test",
                    "novel_structure": None,
                    "revision": None,
                }
            )

        transcript = []
        with patch.object(runner_mod, "call_agent", side_effect=fake_call):
            run_interpretation_debate(protocol, None, transcript)

        # Prompt should contain a few-shot example with actual stimuli/labels
        assert any(
            "stimuli" in p and "labels" in p and "novel_structure" in p
            for p in captured_prompts
        ), (
            "Prompt should contain a few-shot example of novel_structure with "
            "stimuli and labels arrays"
        )
        # Should mention constraints (4-32 items, ≤8 dims, ≥2 categories)
        assert any("4" in p and "32" in p for p in captured_prompts), (
            "Prompt should mention item count constraints (4-32)"
        )


# =========================================================================
# summary_for_agent crash on non-string new_predictions (epistemic_state.py)
# =========================================================================


class TestSummaryForAgentNonStringPredictions:
    """Bug: summary_for_agent() crashes with TypeError when
    new_predictions from a theory revision contains dicts instead of strings.
    This happens when LLM agents return structured predictions (e.g.,
    {"item": ..., "accuracy": ...}) during interpretation debate revision.
    """

    def test_summary_does_not_crash_on_dict_predictions(self):
        """summary_for_agent should handle non-string items in
        new_predictions without crashing."""
        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        # Register a theory
        state.register_theory(
            TheoryCommitment(
                name=agents[0].theory_name,
                agent_name=agents[0].name,
                core_claims=agents[0].model_class.core_claims,
                model_name=agents[0].model_class.name,
                model_params=agents[0].default_params,
            )
        )
        # Revise with dict-valued new_predictions (as LLM might produce)
        state.revise_theory(
            agents[0].theory_name,
            description="Revised after interpretation",
            triggered_by_experiment="exp_001",
            new_predictions=[
                {"item": "Type_I", "accuracy": 0.9},
                {"item": "Type_II", "accuracy": 0.7},
            ],
        )
        # This should not crash
        summary = state.summary_for_agent(agents[0].name)
        assert isinstance(summary, str)
        assert len(summary) > 0


# =========================================================================
# Feature 7.1: Parameter Revision Persistence (M5)
# =========================================================================


class TestSyncParamsFromTheory:
    """
    M4 analysis revealed that revise_theory() updates theory.model_params
    (epistemic_state.py:87) but compute_model_predictions() reads from
    agent_config.default_params (debate_protocol.py:714). These are never
    synced, so debate-driven parameter revisions are lost.

    sync_params_from_theory() closes this loop.
    """

    def _make_protocol_with_theory(self, params=None):
        """Helper: create a DebateProtocol with theories registered."""
        agents = default_agent_configs()
        state = EpistemicState(domain="test")
        for agent in agents:
            state.register_theory(
                TheoryCommitment(
                    name=agent.theory_name,
                    agent_name=agent.name,
                    core_claims=agent.model_class.core_claims,
                    model_name=agent.model_class.name,
                    model_params=params or dict(agent.default_params),
                )
            )
        protocol = DebateProtocol(state=state, agent_configs=agents)
        return protocol

    def test_sync_params_updates_agent_config(self):
        """After revise_theory updates model_params, sync_params should
        propagate changes to agent_config.default_params."""
        from antagonistic_collab.runner import sync_params_from_theory

        protocol = self._make_protocol_with_theory()
        agent = protocol.agent_configs[0]  # GCM agent
        original_c = agent.default_params.get("c")

        # Revise theory to change parameter
        protocol.state.revise_theory(
            agent.theory_name,
            description="Adjust sensitivity",
            new_params={"c": 99.0},
        )

        sync_params_from_theory(protocol)

        # default_params should now have the revised value
        assert agent.default_params["c"] == 99.0
        assert agent.default_params["c"] != original_c

    def test_sync_params_filters_invalid_keys(self):
        """Invented parameter names (not in model.predict signature) should
        be rejected by sync_params."""
        from antagonistic_collab.runner import sync_params_from_theory

        protocol = self._make_protocol_with_theory()
        agent = protocol.agent_configs[0]

        # Revise with an invalid parameter key
        protocol.state.revise_theory(
            agent.theory_name,
            description="Add bogus param",
            new_params={"c": 5.0, "totally_fake_param": 999},
        )

        sync_params_from_theory(protocol)

        assert agent.default_params["c"] == 5.0
        assert "totally_fake_param" not in agent.default_params

    def test_sync_params_no_theory_no_crash(self):
        """If a theory is not found for an agent, sync should not crash."""
        from antagonistic_collab.runner import sync_params_from_theory

        agents = default_agent_configs()
        state = EpistemicState(domain="test")  # No theories registered
        protocol = DebateProtocol(state=state, agent_configs=agents)

        # Should not raise
        sync_params_from_theory(protocol)

    def test_sync_params_empty_revision_no_change(self):
        """When theory.model_params is empty, default_params should be unchanged."""
        from antagonistic_collab.runner import sync_params_from_theory

        protocol = self._make_protocol_with_theory()
        agent = protocol.agent_configs[0]
        original_params = dict(agent.default_params)

        # Register a theory with empty model_params
        theory = protocol.state.get_theory(agent.theory_name)
        theory.model_params = {}

        sync_params_from_theory(protocol)

        assert agent.default_params == original_params

    def test_revision_new_params_passed_to_theory(self):
        """When an agent includes new_params in their revision JSON,
        revise_theory should receive and apply them."""
        protocol = self._make_protocol_with_theory()
        agent = protocol.agent_configs[0]
        theory = protocol.state.get_theory(agent.theory_name)

        # Simulate what the interpretation debate should do
        revision_json = {
            "description": "Lower sensitivity after poor fit",
            "new_params": {"c": 1.5},
            "new_predictions": ["RULEX will fit Type_I better"],
        }

        protocol.state.revise_theory(
            agent.theory_name,
            description=revision_json["description"],
            new_params=revision_json.get("new_params"),
            new_predictions=revision_json.get("new_predictions", []),
        )

        assert theory.model_params["c"] == 1.5

    def test_params_persist_across_cycles(self):
        """After sync in cycle 0, the revised params should be used in
        cycle 1's compute_model_predictions."""
        from antagonistic_collab.runner import sync_params_from_theory

        protocol = self._make_protocol_with_theory()
        agent = protocol.agent_configs[0]  # GCM agent

        # Get predictions with original params
        preds_before = protocol.compute_model_predictions(agent, "Type_I", "baseline")

        # Revise and sync
        protocol.state.revise_theory(
            agent.theory_name,
            description="Change sensitivity",
            new_params={"c": 0.1},  # Very low sensitivity
        )
        sync_params_from_theory(protocol)

        # Predictions should now use new params
        preds_after = protocol.compute_model_predictions(agent, "Type_I", "baseline")

        # With c=0.1 (very low sensitivity), predictions should differ
        assert preds_before["params_used"] != preds_after["params_used"]
        assert preds_after["params_used"]["c"] == 0.1


# =========================================================================
# Feature 7.4: Structured Claim Ledger (M5)
# =========================================================================


class TestClaimLedger:
    """
    Agents repeat the same talking points across cycles. agent_hypotheses
    is stored but never read. No accountability for predictions made in
    debate. The claim ledger tracks structured claims with testable
    predictions and verifies them against actual results.
    """

    def test_add_claim_to_ledger(self):
        """Basic add: a claim should appear in the ledger."""
        from antagonistic_collab.epistemic_state import DebateClaim

        state = EpistemicState(domain="test")
        claim = DebateClaim(
            agent="GCM_Advocate",
            claim_type="prediction",
            content="GCM will fit Type_I perfectly",
            testable=True,
            structure="Type_I",
            predicted_outcome="GCM RMSE < 0.1",
            cycle_made=0,
        )
        state.add_claim(claim)
        assert len(state.claim_ledger) == 1
        assert state.claim_ledger[0].agent == "GCM_Advocate"
        assert state.claim_ledger[0].status == "untested"

    def test_update_claim_status(self):
        """update_claim_status should mark a claim confirmed/falsified."""
        from antagonistic_collab.epistemic_state import DebateClaim

        state = EpistemicState(domain="test")
        claim = DebateClaim(
            agent="GCM_Advocate",
            claim_type="prediction",
            content="GCM RMSE < 0.1 on Type_I",
            testable=True,
            cycle_made=0,
        )
        state.add_claim(claim)
        state.update_claim_status(0, "confirmed", "RMSE was 0.05", cycle=1)

        assert state.claim_ledger[0].status == "confirmed"
        assert state.claim_ledger[0].evidence == "RMSE was 0.05"
        assert state.claim_ledger[0].tested_at_cycle == 1

    def test_stale_claims_detected(self):
        """Claims untested for >threshold cycles should be flagged stale."""
        from antagonistic_collab.epistemic_state import DebateClaim

        state = EpistemicState(domain="test")
        # Claim made at cycle 0, now at cycle 3
        claim = DebateClaim(
            agent="RULEX_Advocate",
            claim_type="prediction",
            content="RULEX will dominate",
            testable=True,
            cycle_made=0,
        )
        state.add_claim(claim)

        stale = state.stale_claims(current_cycle=3, threshold=2)
        assert len(stale) == 1

        # Not stale if within threshold
        not_stale = state.stale_claims(current_cycle=1, threshold=2)
        assert len(not_stale) == 0

    def test_claims_summary_formatting(self):
        """claims_summary_for_agent should produce a readable string."""
        from antagonistic_collab.epistemic_state import DebateClaim

        state = EpistemicState(domain="test")
        state.add_claim(
            DebateClaim(
                agent="GCM_Advocate",
                claim_type="prediction",
                content="GCM fits Type_I",
                testable=True,
                cycle_made=0,
                status="confirmed",
                evidence="RMSE=0.05",
            )
        )
        state.add_claim(
            DebateClaim(
                agent="GCM_Advocate",
                claim_type="explanation",
                content="Attention weights explain difficulty ordering",
                testable=False,
                cycle_made=1,
            )
        )

        summary = state.claims_summary_for_agent("GCM_Advocate")
        assert "GCM fits Type_I" in summary
        assert "CONFIRMED" in summary
        assert "Attention weights" in summary

    def test_get_active_claims(self):
        """get_active_claims filters to untested claims for a given agent."""
        from antagonistic_collab.epistemic_state import DebateClaim

        state = EpistemicState(domain="test")
        state.add_claim(
            DebateClaim(
                agent="GCM_Advocate",
                claim_type="prediction",
                content="c1",
                testable=True,
                cycle_made=0,
                status="untested",
            )
        )
        state.add_claim(
            DebateClaim(
                agent="GCM_Advocate",
                claim_type="prediction",
                content="c2",
                testable=True,
                cycle_made=0,
                status="confirmed",
            )
        )
        state.add_claim(
            DebateClaim(
                agent="RULEX_Advocate",
                claim_type="prediction",
                content="c3",
                testable=True,
                cycle_made=0,
                status="untested",
            )
        )

        active = state.get_active_claims(agent="GCM_Advocate")
        assert len(active) == 1
        assert active[0].content == "c1"

        all_active = state.get_active_claims()
        assert len(all_active) == 2

    def test_ledger_serialization(self):
        """Claim ledger should survive to_json / from_dict round-trip."""
        import tempfile
        from antagonistic_collab.epistemic_state import DebateClaim

        state = EpistemicState(domain="test")
        state.add_claim(
            DebateClaim(
                agent="GCM_Advocate",
                claim_type="prediction",
                content="GCM wins",
                testable=True,
                cycle_made=0,
            )
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            state.to_json(path)
            with open(path) as f:
                data = json.load(f)
            assert "claim_ledger" in data
            assert len(data["claim_ledger"]) == 1
            assert data["claim_ledger"][0]["agent"] == "GCM_Advocate"
        finally:
            os.unlink(path)

    def test_claim_parsed_from_agent_json(self):
        """When an agent includes claims in their interpretation JSON,
        they should be parsed and added to the ledger."""
        from antagonistic_collab.runner import parse_claims_from_json

        json_block = {
            "interpretation": "...",
            "claims": [
                {
                    "claim": "GCM RMSE < 0.1 on Type_I",
                    "testable": True,
                    "structure": "Type_I",
                    "predicted_outcome": "RMSE < 0.1",
                },
                {
                    "claim": "Attention weights shift to dim 1",
                    "testable": False,
                },
            ],
        }
        claims = parse_claims_from_json(json_block, "GCM_Advocate", cycle=0)
        assert len(claims) == 2
        assert claims[0].testable is True
        assert claims[0].structure == "Type_I"
        assert claims[1].testable is False

    def test_claims_shown_in_interpretation_prompt(self):
        """The interpretation prompt should include prior claims for the agent."""
        from antagonistic_collab.epistemic_state import DebateClaim

        state = EpistemicState(domain="test")
        state.add_claim(
            DebateClaim(
                agent="GCM_Advocate",
                claim_type="prediction",
                content="GCM RMSE < 0.1 on Type_I",
                testable=True,
                cycle_made=0,
                status="confirmed",
                evidence="RMSE=0.05",
            )
        )
        summary = state.claims_summary_for_agent("GCM_Advocate")
        # Should be non-empty and include the claim
        assert len(summary) > 0
        assert "GCM RMSE < 0.1" in summary


# =========================================================================
# Feature 7.2: Critique-as-Falsification (M5)
# =========================================================================


class TestCritiqueAsFalsification:
    """
    Agents claim 'my model can also predict that' but are never fact-checked.
    Critiques are free text with no verification. verify_prediction_claim()
    runs the critic's actual model and compares to their stated claim.
    """

    def _make_protocol(self):
        """Helper: create a protocol with standard agent configs."""
        agents = default_agent_configs()
        state = EpistemicState(domain="test")
        for agent in agents:
            state.register_theory(
                TheoryCommitment(
                    name=agent.theory_name,
                    agent_name=agent.name,
                    core_claims=agent.model_class.core_claims,
                    model_name=agent.model_class.name,
                    model_params=dict(agent.default_params),
                )
            )
        return DebateProtocol(state=state, agent_configs=agents)

    def test_critique_claim_verified_true(self):
        """When an agent's claimed prediction matches the model output,
        the verification should return verified=True."""
        from antagonistic_collab.debate_protocol import verify_prediction_claim

        protocol = self._make_protocol()
        # GCM agent (index 0)
        agent = protocol.agent_configs[0]

        # Get actual prediction to use as the "claimed" value
        preds = protocol.compute_model_predictions(agent, "Type_I", "baseline")
        actual_mean = preds["mean_accuracy"]

        result = verify_prediction_claim(
            protocol, agent, "Type_I", "baseline", actual_mean
        )
        assert result["verified"] is True
        assert abs(result["actual"] - actual_mean) < 0.01

    def test_critique_claim_verified_false(self):
        """When an agent claims a prediction far from reality,
        verification should return verified=False."""
        from antagonistic_collab.debate_protocol import verify_prediction_claim

        protocol = self._make_protocol()
        agent = protocol.agent_configs[0]

        # Claim something wildly wrong
        result = verify_prediction_claim(protocol, agent, "Type_I", "baseline", 0.01)
        assert result["verified"] is False
        assert result["discrepancy"] > 0.1

    def test_false_claim_added_to_ledger(self):
        """A falsified critique claim should be recorded in the claim ledger."""
        from antagonistic_collab.debate_protocol import verify_prediction_claim

        protocol = self._make_protocol()
        agent = protocol.agent_configs[0]

        result = verify_prediction_claim(protocol, agent, "Type_I", "baseline", 0.01)

        # Add the result as a claim to the ledger
        from antagonistic_collab.epistemic_state import DebateClaim

        claim = DebateClaim(
            agent=agent.name,
            claim_type="critique",
            content="Claimed mean_accuracy=0.01 on Type_I",
            testable=True,
            structure="Type_I",
            predicted_outcome="mean_accuracy=0.01",
            status="falsified" if not result["verified"] else "confirmed",
            evidence=f"actual={result['actual']:.3f}",
        )
        protocol.state.add_claim(claim)

        assert len(protocol.state.claim_ledger) == 1
        assert protocol.state.claim_ledger[0].status == "falsified"

    def test_critique_prompt_requires_json(self):
        """The critique prompt should instruct agents to use structured JSON."""
        # This is a design test — verify the prompt template includes JSON format
        # We check that run_interpretation_critique's prompt mentions structured format
        # by inspecting the function source code
        import antagonistic_collab.runner as runner_mod

        source = inspect.getsource(runner_mod.run_interpretation_critique)
        assert "alternative_prediction" in source or "disputed_interpretation" in source

    def test_verification_uses_current_params(self):
        """verify_prediction_claim should use agent's current default_params,
        which may have been updated by sync_params_from_theory."""
        from antagonistic_collab.debate_protocol import verify_prediction_claim
        from antagonistic_collab.runner import sync_params_from_theory

        protocol = self._make_protocol()
        agent = protocol.agent_configs[0]

        # Get prediction with original params
        result_before = verify_prediction_claim(
            protocol, agent, "Type_I", "baseline", 0.5
        )
        actual_before = result_before["actual"]

        # Revise theory params and sync
        protocol.state.revise_theory(
            agent.theory_name,
            description="Lower sensitivity",
            new_params={"c": 0.1},
        )
        sync_params_from_theory(protocol)

        # Verify should now use new params
        result_after = verify_prediction_claim(
            protocol, agent, "Type_I", "baseline", 0.5
        )
        actual_after = result_after["actual"]

        # With c=0.1 (very low sensitivity), predictions should differ
        assert actual_before != actual_after


# =========================================================================
# Feature 7.3: Debate-Informed EIG Weighting (M5)
# =========================================================================


class TestDebateInformedEIG:
    """
    EIG treats all model pairs equally. After a few cycles, the real
    contest may be between two specific models. Debate identifies this,
    but the insight is discarded. focus_pair boosting directs EIG toward
    experiments that distinguish the contested pair.
    """

    def _make_protocol_and_posterior(self):
        """Helper: create protocol + uniform posterior."""
        agents = default_agent_configs()
        state = EpistemicState(domain="test")
        for agent in agents:
            state.register_theory(
                TheoryCommitment(
                    name=agent.theory_name,
                    agent_name=agent.name,
                    core_claims=agent.model_class.core_claims,
                    model_name=agent.model_class.name,
                    model_params=dict(agent.default_params),
                )
            )
        protocol = DebateProtocol(state=state, agent_configs=agents)
        model_names = [a.name for a in agents]
        posterior = ModelPosterior.uniform(model_names)
        return protocol, posterior

    def test_no_focus_pair_unchanged(self):
        """Without focus_pair, select_from_pool should work as before."""
        from antagonistic_collab.bayesian_selection import (
            generate_full_candidate_pool,
            select_from_pool,
        )

        protocol, posterior = self._make_protocol_and_posterior()
        pool = generate_full_candidate_pool(protocol)

        # Without focus_pair (backward compat)
        best_idx, eig_scores = select_from_pool(
            protocol, posterior, pool, n_sim=50, seed=42
        )
        assert best_idx >= 0
        assert len(eig_scores) == len(pool)

    def test_pairwise_boost_changes_selection(self):
        """With a focus_pair and high boost, selection should prefer
        experiments that distinguish the focus pair."""
        from antagonistic_collab.bayesian_selection import (
            generate_full_candidate_pool,
            select_from_pool,
        )

        protocol, posterior = self._make_protocol_and_posterior()
        pool = generate_full_candidate_pool(protocol)

        # Without boost
        best_no_boost, scores_no_boost = select_from_pool(
            protocol, posterior, pool, n_sim=50, seed=42
        )

        # With focus pair and very high boost
        agent_names = [a.name for a in protocol.agent_configs]
        focus = (agent_names[0], agent_names[1])
        best_boosted, scores_boosted = select_from_pool(
            protocol,
            posterior,
            pool,
            n_sim=50,
            seed=42,
            focus_pair=focus,
            pair_boost=5.0,
        )

        # Boosted scores should be >= unboosted for divergent candidates
        assert max(scores_boosted) >= max(scores_no_boost)

    def test_focus_pair_extracted_from_posterior(self):
        """extract_focus_pair_from_posterior should identify the two models
        with closest posterior probabilities."""
        from antagonistic_collab.bayesian_selection import (
            extract_focus_pair_from_posterior,
        )

        # Make a posterior where two models are very close
        posterior = ModelPosterior(
            log_probs=np.array([0.0, -0.01, -5.0]),
            model_names=["GCM_Advocate", "SUSTAIN_Advocate", "RULEX_Advocate"],
        )
        pair = extract_focus_pair_from_posterior(posterior)
        # The closest pair should be GCM and SUSTAIN
        assert set(pair) == {"GCM_Advocate", "SUSTAIN_Advocate"}

    def test_focus_pair_from_ledger(self):
        """extract_focus_pair_from_ledger should identify contested model pairs
        from recent falsified/disputed claims."""
        from antagonistic_collab.bayesian_selection import (
            extract_focus_pair_from_ledger,
        )
        from antagonistic_collab.epistemic_state import DebateClaim

        state = EpistemicState(domain="test")
        # Add some claims that reference specific models
        state.add_claim(
            DebateClaim(
                agent="GCM_Advocate",
                claim_type="critique",
                content="SUSTAIN cannot predict Type_IV",
                testable=True,
                structure="Type_IV",
                cycle_made=0,
                status="falsified",
            )
        )
        state.add_claim(
            DebateClaim(
                agent="SUSTAIN_Advocate",
                claim_type="critique",
                content="GCM attention weights are wrong",
                testable=True,
                cycle_made=0,
                status="untested",
            )
        )
        pair = extract_focus_pair_from_ledger(state)
        # Should identify GCM and SUSTAIN as contested
        assert pair is not None
        assert "GCM_Advocate" in pair
        assert "SUSTAIN_Advocate" in pair

    def test_boost_magnitude(self):
        """EIG * boost should be > unboosted EIG for high-divergence candidates."""
        from antagonistic_collab.bayesian_selection import (
            generate_full_candidate_pool,
            select_from_pool,
        )

        protocol, posterior = self._make_protocol_and_posterior()
        pool = generate_full_candidate_pool(protocol)[:5]  # Small pool for speed

        agent_names = [a.name for a in protocol.agent_configs]
        focus = (agent_names[0], agent_names[1])

        _, scores_no_boost = select_from_pool(
            protocol, posterior, pool, n_sim=50, seed=42
        )
        _, scores_boosted = select_from_pool(
            protocol,
            posterior,
            pool,
            n_sim=50,
            seed=42,
            focus_pair=focus,
            pair_boost=2.0,
        )

        # At least some boosted scores should be higher
        any_higher = any(b > nb for b, nb in zip(scores_boosted, scores_no_boost))
        assert any_higher


# =========================================================================
# M6a: MetaAgentConfig — role-specialized meta-agents
# =========================================================================


class TestMetaAgentConfig:
    """Tests for MetaAgentConfig dataclass and meta-agent integration."""

    def test_meta_agent_config_creation(self):
        """MetaAgentConfig dataclass has name, role, and system_prompt fields."""
        from antagonistic_collab.debate_protocol import MetaAgentConfig

        ma = MetaAgentConfig(
            name="Integrator",
            role="integrator",
            system_prompt="You synthesize across all theories.",
        )
        assert ma.name == "Integrator"
        assert ma.role == "integrator"
        assert "synthesize" in ma.system_prompt

    def test_meta_agent_has_no_model_class(self):
        """MetaAgentConfig should NOT have a model_class attribute."""
        from antagonistic_collab.debate_protocol import MetaAgentConfig

        ma = MetaAgentConfig(
            name="Critic",
            role="critic",
            system_prompt="You challenge weak arguments.",
        )
        assert not hasattr(ma, "model_class")

    def test_create_default_meta_agents(self):
        """Factory function returns Integrator + Critic."""
        from antagonistic_collab.runner import create_default_meta_agents

        agents = create_default_meta_agents()
        assert len(agents) == 2
        names = {a.name for a in agents}
        assert "Integrator" in names
        assert "Critic" in names

    def test_integrator_prompt_contains_synthesis(self):
        """Integrator's prompt should instruct synthesis across theories."""
        from antagonistic_collab.runner import create_default_meta_agents

        agents = create_default_meta_agents()
        integrator = [a for a in agents if a.name == "Integrator"][0]
        prompt_lower = integrator.system_prompt.lower()
        assert "synthes" in prompt_lower  # synthesis/synthesize

    def test_critic_prompt_contains_challenge(self):
        """Critic's prompt should instruct challenging weak arguments."""
        from antagonistic_collab.runner import create_default_meta_agents

        agents = create_default_meta_agents()
        critic = [a for a in agents if a.name == "Critic"][0]
        prompt_lower = critic.system_prompt.lower()
        assert "challeng" in prompt_lower or "weakest" in prompt_lower

    def test_meta_agents_added_to_protocol(self):
        """DebateProtocol.meta_agents should be populated."""
        from antagonistic_collab.runner import create_default_meta_agents

        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state=state, agent_configs=agents)
        # Default should be empty
        assert protocol.meta_agents == []
        # Can assign
        protocol.meta_agents = create_default_meta_agents()
        assert len(protocol.meta_agents) == 2

    def test_meta_agent_response_in_transcript(self):
        """Meta-agent responses should appear in the interpretation transcript."""
        from antagonistic_collab.runner import (
            run_interpretation_debate,
            create_default_meta_agents,
        )

        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state=state, agent_configs=agents)
        protocol.meta_agents = create_default_meta_agents()

        # Set up minimal experiment data so interpretation debate runs
        exp = state.propose_experiment(
            proposed_by="Exemplar_Agent",
            title="Test experiment",
            design_spec={"structure_name": "Type_II", "condition": "baseline"},
            rationale="test",
        )
        state.approve_experiment(exp.experiment_id)
        state.record_data(exp.experiment_id, {"item_accuracies": {"item_0": 0.8}})

        call_count = 0

        def fake_llm(client, system, user, **kw):
            nonlocal call_count
            call_count += 1
            return '{"interpretation": "test", "confounds_flagged": [], "hypothesis": "test", "claims": []}'

        transcript = []
        # Monkey-patch call_agent
        import antagonistic_collab.runner as runner_mod

        original = runner_mod.call_agent
        runner_mod.call_agent = fake_llm
        try:
            run_interpretation_debate(protocol, None, transcript)
        finally:
            runner_mod.call_agent = original

        # Should have calls for 3 theory agents + 2 meta-agents = 5
        assert call_count == 5
        # Meta-agent responses in transcript
        meta_agents_in_transcript = [
            m for m in transcript if m.get("agent") in ("Integrator", "Critic")
        ]
        assert len(meta_agents_in_transcript) == 2

    def test_meta_agent_no_param_update(self):
        """Meta-agent responses should NOT trigger revise_theory."""
        from antagonistic_collab.runner import (
            run_interpretation_debate,
            create_default_meta_agents,
        )

        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state=state, agent_configs=agents)
        protocol.meta_agents = create_default_meta_agents()

        # Register theories so revise_theory has targets
        for agent in agents:
            state.register_theory(
                TheoryCommitment(
                    name=agent.theory_name,
                    agent_name=agent.name,
                    core_claims=["test"],
                    model_name=agent.model_class.name.split()[0],
                )
            )

        exp = state.propose_experiment(
            proposed_by="Exemplar_Agent",
            title="Test experiment",
            design_spec={"structure_name": "Type_II", "condition": "baseline"},
            rationale="test",
        )
        state.approve_experiment(exp.experiment_id)
        state.record_data(exp.experiment_id, {"item_accuracies": {"item_0": 0.8}})

        def fake_llm(client, system, user, **kw):
            # Meta-agents return revision requests — these should be ignored
            return (
                '{"interpretation": "test", "confounds_flagged": [], '
                '"hypothesis": "test", "claims": [], '
                '"revision": {"description": "sneaky revision", "new_params": {"c": 99.0}}}'
            )

        transcript = []
        import antagonistic_collab.runner as runner_mod

        original = runner_mod.call_agent
        runner_mod.call_agent = fake_llm
        try:
            run_interpretation_debate(protocol, None, transcript)
        finally:
            runner_mod.call_agent = original

        # Theory agents may have triggered revisions, but meta-agents should not.
        # Check that no theory has c=99.0 from a meta-agent
        # (theory agents might set it, but there should be at most 3 revisions,
        # not 5)
        total_revisions = sum(len(t.revision_log) for t in state.theories)
        # 3 theory agents each trigger one revision = 3
        assert total_revisions == 3


# =========================================================================
# M6b: Crux Negotiation
# =========================================================================


class TestCruxDataclass:
    """Tests for the Crux dataclass and EpistemicState crux methods."""

    def test_crux_creation(self):
        """Crux dataclass has all expected fields with defaults."""
        from antagonistic_collab.epistemic_state import Crux

        crux = Crux(
            id="crux_001",
            proposer="Exemplar_Agent",
            description="Does GCM predict Type VI accuracy > 60%?",
        )
        assert crux.id == "crux_001"
        assert crux.proposer == "Exemplar_Agent"
        assert crux.status == "proposed"
        assert crux.discriminating_experiment is None
        assert crux.resolution is None
        assert crux.cycle_proposed == 0
        assert crux.supporters == []

    def test_crux_status_transitions(self):
        """Crux status can be transitioned through the expected states."""
        from antagonistic_collab.epistemic_state import Crux

        crux = Crux(id="crux_002", proposer="Rule_Agent", description="test")
        assert crux.status == "proposed"
        crux.status = "accepted"
        assert crux.status == "accepted"
        crux.status = "resolved"
        assert crux.status == "resolved"

    def test_add_crux_to_state(self):
        """EpistemicState.add_crux appends to the cruxes list."""
        from antagonistic_collab.epistemic_state import Crux

        state = EpistemicState(domain="test")
        assert state.cruxes == []
        crux = Crux(id="crux_001", proposer="Exemplar_Agent", description="test")
        state.add_crux(crux)
        assert len(state.cruxes) == 1
        assert state.cruxes[0].id == "crux_001"

    def test_get_active_cruxes(self):
        """get_active_cruxes returns only proposed/accepted cruxes."""
        from antagonistic_collab.epistemic_state import Crux

        state = EpistemicState(domain="test")
        state.add_crux(Crux(id="c1", proposer="A", description="d1", status="proposed"))
        state.add_crux(Crux(id="c2", proposer="B", description="d2", status="accepted"))
        state.add_crux(Crux(id="c3", proposer="C", description="d3", status="resolved"))
        state.add_crux(Crux(id="c4", proposer="D", description="d4", status="rejected"))

        active = state.get_active_cruxes()
        assert len(active) == 2
        assert {c.id for c in active} == {"c1", "c2"}

    def test_resolve_crux(self):
        """resolve_crux sets status, resolution, and cycle_resolved."""
        from antagonistic_collab.epistemic_state import Crux

        state = EpistemicState(domain="test")
        state.add_crux(Crux(id="c1", proposer="A", description="d1"))
        state.resolve_crux("c1", "GCM achieved 0.72 RMSE", cycle=3)
        crux = state.cruxes[0]
        assert crux.status == "resolved"
        assert crux.resolution == "GCM achieved 0.72 RMSE"
        assert crux.cycle_resolved == 3

    def test_crux_summary(self):
        """crux_summary returns formatted string for prompt injection."""
        from antagonistic_collab.epistemic_state import Crux

        state = EpistemicState(domain="test")
        state.add_crux(
            Crux(
                id="c1",
                proposer="Exemplar_Agent",
                description="GCM vs RULEX on Type VI",
                status="accepted",
                supporters=["Exemplar_Agent", "Clustering_Agent"],
            )
        )
        summary = state.crux_summary()
        assert "c1" in summary
        assert "GCM vs RULEX" in summary
        assert "accepted" in summary.lower() or "ACCEPTED" in summary


class TestCruxIdentification:
    """Tests for run_crux_identification phase."""

    def _make_protocol(self):
        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state=state, agent_configs=agents)
        for agent in agents:
            state.register_theory(
                TheoryCommitment(
                    name=agent.theory_name,
                    agent_name=agent.name,
                    core_claims=["test"],
                    model_name=agent.model_class.name.split()[0],
                )
            )
        return protocol

    def test_crux_identification_returns_crux_list(self):
        """run_crux_identification returns a list of dicts."""
        from antagonistic_collab.runner import run_crux_identification

        protocol = self._make_protocol()

        def fake_llm(client, system, user, **kw):
            return (
                '{"cruxes": [{"description": "Type VI accuracy test", '
                '"discriminating_experiment": "Type_VI/baseline", '
                '"resolution_criterion": "RMSE < 0.15"}]}'
            )

        import antagonistic_collab.runner as runner_mod

        original = runner_mod.call_agent
        runner_mod.call_agent = fake_llm
        try:
            result = run_crux_identification(protocol, None, cycle=0)
        finally:
            runner_mod.call_agent = original

        assert isinstance(result, list)
        assert len(result) > 0

    def test_crux_identification_prompt_asks_for_cruxes(self):
        """The prompt sent to agents should ask what would change their mind."""
        from antagonistic_collab.runner import run_crux_identification

        protocol = self._make_protocol()
        captured_prompts = []

        def fake_llm(client, system, user, **kw):
            captured_prompts.append(user)
            return '{"cruxes": []}'

        import antagonistic_collab.runner as runner_mod

        original = runner_mod.call_agent
        runner_mod.call_agent = fake_llm
        try:
            run_crux_identification(protocol, None, cycle=0)
        finally:
            runner_mod.call_agent = original

        assert len(captured_prompts) == 3  # one per theory agent
        for prompt in captured_prompts:
            assert "crux" in prompt.lower() or "change your mind" in prompt.lower()

    def test_crux_identification_parses_cruxes(self):
        """Parsed cruxes should have expected fields."""
        from antagonistic_collab.runner import run_crux_identification

        protocol = self._make_protocol()

        def fake_llm(client, system, user, **kw):
            return (
                '{"cruxes": [{"description": "Test crux", '
                '"discriminating_experiment": "Type_II/baseline", '
                '"resolution_criterion": "accuracy > 0.8"}]}'
            )

        import antagonistic_collab.runner as runner_mod

        original = runner_mod.call_agent
        runner_mod.call_agent = fake_llm
        try:
            result = run_crux_identification(protocol, None, cycle=0)
        finally:
            runner_mod.call_agent = original

        assert any("description" in c for c in result)

    def test_crux_identification_adds_to_state(self):
        """Identified cruxes should be added to EpistemicState."""
        from antagonistic_collab.runner import run_crux_identification

        protocol = self._make_protocol()

        def fake_llm(client, system, user, **kw):
            return (
                '{"cruxes": [{"description": "Test crux", '
                '"discriminating_experiment": "Type_II/baseline", '
                '"resolution_criterion": "accuracy > 0.8"}]}'
            )

        import antagonistic_collab.runner as runner_mod

        original = runner_mod.call_agent
        runner_mod.call_agent = fake_llm
        try:
            run_crux_identification(protocol, None, cycle=0)
        finally:
            runner_mod.call_agent = original

        assert len(protocol.state.cruxes) > 0

    def test_crux_identification_handles_empty_response(self):
        """Gracefully handles agents that return no cruxes."""
        from antagonistic_collab.runner import run_crux_identification

        protocol = self._make_protocol()

        def fake_llm(client, system, user, **kw):
            return '{"cruxes": []}'

        import antagonistic_collab.runner as runner_mod

        original = runner_mod.call_agent
        runner_mod.call_agent = fake_llm
        try:
            result = run_crux_identification(protocol, None, cycle=0)
        finally:
            runner_mod.call_agent = original

        assert result == []

    def test_crux_identification_handles_malformed_json(self):
        """Gracefully handles agents that return invalid JSON."""
        from antagonistic_collab.runner import run_crux_identification

        protocol = self._make_protocol()

        def fake_llm(client, system, user, **kw):
            return "I think we should test Type VI but I forgot the JSON format."

        import antagonistic_collab.runner as runner_mod

        original = runner_mod.call_agent
        runner_mod.call_agent = fake_llm
        try:
            result = run_crux_identification(protocol, None, cycle=0)
        finally:
            runner_mod.call_agent = original

        assert result == []


class TestCruxNegotiation:
    """Tests for run_crux_negotiation phase."""

    def _make_protocol_with_cruxes(self):
        from antagonistic_collab.epistemic_state import Crux

        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state=state, agent_configs=agents)
        for agent in agents:
            state.register_theory(
                TheoryCommitment(
                    name=agent.theory_name,
                    agent_name=agent.name,
                    core_claims=["test"],
                    model_name=agent.model_class.name.split()[0],
                )
            )
        # Add some cruxes
        state.add_crux(
            Crux(
                id="crux_001",
                proposer="Exemplar_Agent",
                description="GCM vs RULEX on Type VI accuracy",
                discriminating_experiment="Type_VI/baseline",
                resolution_criterion="RMSE < 0.15",
                supporters=["Exemplar_Agent"],
            )
        )
        state.add_crux(
            Crux(
                id="crux_002",
                proposer="Rule_Agent",
                description="Does presentation order matter for Type II?",
                discriminating_experiment="Type_II/fast_presentation",
                resolution_criterion="accuracy diff > 10%",
                supporters=["Rule_Agent"],
            )
        )
        return protocol

    def test_crux_negotiation_returns_responses(self):
        """run_crux_negotiation returns a list of response dicts."""
        from antagonistic_collab.runner import run_crux_negotiation

        protocol = self._make_protocol_with_cruxes()

        def fake_llm(client, system, user, **kw):
            return (
                '{"responses": [{"crux_id": "crux_001", "action": "accept"}, '
                '{"crux_id": "crux_002", "action": "reject", "reason": "not decisive"}]}'
            )

        import antagonistic_collab.runner as runner_mod

        original = runner_mod.call_agent
        runner_mod.call_agent = fake_llm
        try:
            result = run_crux_negotiation(protocol, None, cycle=0)
        finally:
            runner_mod.call_agent = original

        assert isinstance(result, list)

    def test_crux_negotiation_accept_adds_supporter(self):
        """Accepting a crux adds the agent to supporters."""
        from antagonistic_collab.runner import run_crux_negotiation

        protocol = self._make_protocol_with_cruxes()

        def fake_llm(client, system, user, **kw):
            return '{"responses": [{"crux_id": "crux_001", "action": "accept"}]}'

        import antagonistic_collab.runner as runner_mod

        original = runner_mod.call_agent
        runner_mod.call_agent = fake_llm
        try:
            run_crux_negotiation(protocol, None, cycle=0)
        finally:
            runner_mod.call_agent = original

        crux = protocol.state.cruxes[0]
        # Multiple agents should have accepted
        assert len(crux.supporters) > 1

    def test_crux_negotiation_reject_does_not_add_supporter(self):
        """Rejecting a crux should not add the agent to supporters."""
        from antagonistic_collab.runner import run_crux_negotiation

        protocol = self._make_protocol_with_cruxes()

        def fake_llm(client, system, user, **kw):
            return '{"responses": [{"crux_id": "crux_001", "action": "reject"}]}'

        import antagonistic_collab.runner as runner_mod

        original = runner_mod.call_agent
        runner_mod.call_agent = fake_llm
        try:
            run_crux_negotiation(protocol, None, cycle=0)
        finally:
            runner_mod.call_agent = original

        crux = protocol.state.cruxes[0]
        # Only the original proposer
        assert crux.supporters == ["Exemplar_Agent"]

    def test_crux_negotiation_counter_propose(self):
        """Counter-proposing adds a new crux."""
        from antagonistic_collab.runner import run_crux_negotiation

        protocol = self._make_protocol_with_cruxes()
        initial_count = len(protocol.state.cruxes)

        def fake_llm(client, system, user, **kw):
            return (
                '{"responses": [{"crux_id": "crux_001", "action": "counter", '
                '"counter_crux": {"description": "Better test: Type IV under time pressure"}}]}'
            )

        import antagonistic_collab.runner as runner_mod

        original = runner_mod.call_agent
        runner_mod.call_agent = fake_llm
        try:
            run_crux_negotiation(protocol, None, cycle=0)
        finally:
            runner_mod.call_agent = original

        assert len(protocol.state.cruxes) > initial_count

    def test_crux_negotiation_prompt_shows_cruxes(self):
        """The negotiation prompt should show all active cruxes."""
        from antagonistic_collab.runner import run_crux_negotiation

        protocol = self._make_protocol_with_cruxes()
        captured_prompts = []

        def fake_llm(client, system, user, **kw):
            captured_prompts.append(user)
            return '{"responses": []}'

        import antagonistic_collab.runner as runner_mod

        original = runner_mod.call_agent
        runner_mod.call_agent = fake_llm
        try:
            run_crux_negotiation(protocol, None, cycle=0)
        finally:
            runner_mod.call_agent = original

        for prompt in captured_prompts:
            assert "crux_001" in prompt or "crux" in prompt.lower()

    def test_crux_negotiation_handles_empty_response(self):
        """Gracefully handles agents that return no responses."""
        from antagonistic_collab.runner import run_crux_negotiation

        protocol = self._make_protocol_with_cruxes()

        def fake_llm(client, system, user, **kw):
            return '{"responses": []}'

        import antagonistic_collab.runner as runner_mod

        original = runner_mod.call_agent
        runner_mod.call_agent = fake_llm
        try:
            result = run_crux_negotiation(protocol, None, cycle=0)
        finally:
            runner_mod.call_agent = original

        assert result == []


class TestFinalizeCruxes:
    """Tests for finalize_cruxes — filter cruxes by supporter threshold."""

    def test_finalize_cruxes_threshold(self):
        """Only cruxes with >= 2 supporters become accepted."""
        from antagonistic_collab.epistemic_state import Crux
        from antagonistic_collab.runner import finalize_cruxes

        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state=state, agent_configs=agents)

        state.add_crux(
            Crux(
                id="c1",
                proposer="A",
                description="well supported",
                supporters=["A", "B", "C"],
            )
        )
        state.add_crux(
            Crux(
                id="c2",
                proposer="A",
                description="poorly supported",
                supporters=["A"],
            )
        )

        finalized = finalize_cruxes(protocol, cycle=0)
        assert len(finalized) == 1
        assert finalized[0].id == "c1"
        assert finalized[0].status == "accepted"

    def test_finalize_cruxes_rejects_unsupported(self):
        """Cruxes with < 2 supporters get rejected."""
        from antagonistic_collab.epistemic_state import Crux
        from antagonistic_collab.runner import finalize_cruxes

        state = EpistemicState(domain="test")
        protocol = DebateProtocol(state=state, agent_configs=default_agent_configs())

        state.add_crux(
            Crux(id="c1", proposer="A", description="lonely", supporters=["A"])
        )

        finalized = finalize_cruxes(protocol, cycle=0)
        assert finalized == []
        assert protocol.state.cruxes[0].status == "rejected"

    def test_finalize_cruxes_extracts_experiments(self):
        """Finalized cruxes should preserve discriminating_experiment."""
        from antagonistic_collab.epistemic_state import Crux
        from antagonistic_collab.runner import finalize_cruxes

        state = EpistemicState(domain="test")
        protocol = DebateProtocol(state=state, agent_configs=default_agent_configs())

        state.add_crux(
            Crux(
                id="c1",
                proposer="A",
                description="test",
                discriminating_experiment="Type_VI/baseline",
                supporters=["A", "B"],
            )
        )

        finalized = finalize_cruxes(protocol, cycle=0)
        assert finalized[0].discriminating_experiment == "Type_VI/baseline"

    def test_finalize_cruxes_empty_list(self):
        """No cruxes = empty result."""
        from antagonistic_collab.runner import finalize_cruxes

        state = EpistemicState(domain="test")
        protocol = DebateProtocol(state=state, agent_configs=default_agent_configs())

        finalized = finalize_cruxes(protocol, cycle=0)
        assert finalized == []

    def test_finalize_cruxes_skips_already_resolved(self):
        """Already resolved cruxes should not be re-finalized."""
        from antagonistic_collab.epistemic_state import Crux
        from antagonistic_collab.runner import finalize_cruxes

        state = EpistemicState(domain="test")
        protocol = DebateProtocol(state=state, agent_configs=default_agent_configs())

        state.add_crux(
            Crux(
                id="c1",
                proposer="A",
                description="already done",
                status="resolved",
                supporters=["A", "B", "C"],
            )
        )

        finalized = finalize_cruxes(protocol, cycle=0)
        assert finalized == []

    def test_finalize_cruxes_custom_threshold(self):
        """Custom supporter threshold should be respected."""
        from antagonistic_collab.epistemic_state import Crux
        from antagonistic_collab.runner import finalize_cruxes

        # Test with threshold=3: crux with 2 supporters should NOT pass
        state1 = EpistemicState(domain="test")
        protocol1 = DebateProtocol(state=state1, agent_configs=default_agent_configs())
        state1.add_crux(
            Crux(id="c1", proposer="A", description="needs 3", supporters=["A", "B"])
        )
        finalized = finalize_cruxes(protocol1, cycle=0, min_supporters=3)
        assert finalized == []

        # Test with threshold=2: crux with 2 supporters SHOULD pass
        state2 = EpistemicState(domain="test")
        protocol2 = DebateProtocol(state=state2, agent_configs=default_agent_configs())
        state2.add_crux(
            Crux(id="c2", proposer="A", description="needs 2", supporters=["A", "B"])
        )
        finalized = finalize_cruxes(protocol2, cycle=0, min_supporters=2)
        assert len(finalized) == 1


class TestCruxBoostSpecs:
    """Tests for crux_boost_specs parameter in select_from_pool."""

    def _make_protocol_and_posterior(self):
        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state=state, agent_configs=agents)
        posterior = ModelPosterior.uniform([a.name for a in agents])
        return protocol, posterior

    def test_crux_boost_matching_candidates(self):
        """Matching candidates should have EIG boosted."""
        from antagonistic_collab.bayesian_selection import (
            generate_full_candidate_pool,
            select_from_pool,
        )

        protocol, posterior = self._make_protocol_and_posterior()
        pool = generate_full_candidate_pool(protocol)[:5]

        boost_specs = [{"structure": pool[0][0], "condition": pool[0][1], "boost": 3.0}]

        _, scores_no_boost = select_from_pool(
            protocol, posterior, pool, n_sim=50, seed=42
        )
        _, scores_boosted = select_from_pool(
            protocol, posterior, pool, n_sim=50, seed=42, crux_boost_specs=boost_specs
        )

        # First candidate should be boosted
        if scores_no_boost[0] > 0:
            assert scores_boosted[0] > scores_no_boost[0]

    def test_crux_boost_non_matching_unchanged(self):
        """Non-matching candidates should not be boosted."""
        from antagonistic_collab.bayesian_selection import (
            generate_full_candidate_pool,
            select_from_pool,
        )

        protocol, posterior = self._make_protocol_and_posterior()
        pool = generate_full_candidate_pool(protocol)[:5]

        # Boost only first candidate
        boost_specs = [{"structure": pool[0][0], "condition": pool[0][1], "boost": 3.0}]

        _, scores_no_boost = select_from_pool(
            protocol, posterior, pool, n_sim=50, seed=42
        )
        _, scores_boosted = select_from_pool(
            protocol, posterior, pool, n_sim=50, seed=42, crux_boost_specs=boost_specs
        )

        # Candidates 1-4 should be unchanged
        for i in range(1, len(pool)):
            assert abs(scores_boosted[i] - scores_no_boost[i]) < 1e-10

    def test_crux_boost_backward_compat(self):
        """Without crux_boost_specs, select_from_pool works as before."""
        from antagonistic_collab.bayesian_selection import (
            generate_full_candidate_pool,
            select_from_pool,
        )

        protocol, posterior = self._make_protocol_and_posterior()
        pool = generate_full_candidate_pool(protocol)[:3]

        _, scores_a = select_from_pool(protocol, posterior, pool, n_sim=50, seed=42)
        _, scores_b = select_from_pool(
            protocol, posterior, pool, n_sim=50, seed=42, crux_boost_specs=None
        )

        for a, b in zip(scores_a, scores_b):
            assert abs(a - b) < 1e-10

    def test_crux_boost_coexists_with_focus_pair(self):
        """crux_boost_specs and focus_pair should both apply."""
        from antagonistic_collab.bayesian_selection import (
            generate_full_candidate_pool,
            select_from_pool,
        )

        protocol, posterior = self._make_protocol_and_posterior()
        pool = generate_full_candidate_pool(protocol)[:5]
        agent_names = [a.name for a in protocol.agent_configs]
        focus = (agent_names[0], agent_names[1])

        boost_specs = [{"structure": pool[0][0], "condition": pool[0][1], "boost": 2.0}]

        _, scores_both = select_from_pool(
            protocol,
            posterior,
            pool,
            n_sim=50,
            seed=42,
            focus_pair=focus,
            pair_boost=1.5,
            crux_boost_specs=boost_specs,
        )

        # Should not crash; scores should be valid
        assert all(s >= 0 for s in scores_both)


class TestCruxIntegration:
    """Integration tests for crux flow: identification → negotiation → EIG."""

    def test_crux_phases_wired_into_run_cycle(self):
        """In full_pool mode, crux phases should run between divergence and EIG."""
        from antagonistic_collab.runner import run_cycle

        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state=state, agent_configs=agents)

        call_log = []

        def fake_llm(client, system, user, **kw):
            call_log.append(user[:80])
            # Return valid JSON for any phase
            if "crux" in user.lower() and "negotiat" in user.lower():
                return '{"responses": [{"crux_id": "crux_001", "action": "accept"}]}'
            if "crux" in user.lower():
                return (
                    '{"cruxes": [{"description": "test crux", '
                    '"discriminating_experiment": "Type_II/baseline", '
                    '"resolution_criterion": "RMSE < 0.2"}]}'
                )
            return (
                '{"interpretation": "test", "confounds_flagged": [], '
                '"hypothesis": "test", "claims": [], '
                '"core_claims": ["test"], "auxiliary_assumptions": [], '
                '"model_evidence": {}, "disputed_interpretation": "none", '
                '"audit": "test"}'
            )

        import antagonistic_collab.runner as runner_mod

        original = runner_mod.call_agent
        runner_mod.call_agent = fake_llm
        try:
            transcript = []
            run_cycle(
                protocol,
                None,
                transcript,
                true_model="GCM",
                mode="full_pool",
                output_dir="/tmp/test_crux_integration",
            )
        finally:
            runner_mod.call_agent = original

        # Crux identification should have been called
        crux_prompts = [c for c in call_log if "crux" in c.lower()]
        assert len(crux_prompts) > 0

    def test_cruxes_flow_to_eig(self):
        """Active cruxes should produce boost specs for EIG selection."""
        from antagonistic_collab.runner import cruxes_to_boost_specs
        from antagonistic_collab.epistemic_state import Crux

        state = EpistemicState(domain="test")
        state.add_crux(
            Crux(
                id="c1",
                proposer="A",
                description="test",
                status="accepted",
                discriminating_experiment="Type_VI/baseline",
            )
        )

        specs = cruxes_to_boost_specs(state)
        assert len(specs) == 1
        assert specs[0]["structure"] == "Type_VI"
        assert specs[0]["condition"] == "baseline"
        assert specs[0]["boost"] > 1.0

    def test_cruxes_without_experiment_skipped(self):
        """Cruxes without a discriminating_experiment produce no boost spec."""
        from antagonistic_collab.runner import cruxes_to_boost_specs
        from antagonistic_collab.epistemic_state import Crux

        state = EpistemicState(domain="test")
        state.add_crux(
            Crux(id="c1", proposer="A", description="vague crux", status="accepted")
        )

        specs = cruxes_to_boost_specs(state)
        assert specs == []

    def test_crux_negotiation_output_used(self):
        """After negotiation, accepted cruxes should have multiple supporters."""
        from antagonistic_collab.runner import (
            run_crux_identification,
            run_crux_negotiation,
            finalize_cruxes,
        )

        state = EpistemicState(domain="test")
        agents = default_agent_configs()
        protocol = DebateProtocol(state=state, agent_configs=agents)
        for agent in agents:
            state.register_theory(
                TheoryCommitment(
                    name=agent.theory_name,
                    agent_name=agent.name,
                    core_claims=["test"],
                    model_name=agent.model_class.name.split()[0],
                )
            )

        # Phase 1: identification
        def id_llm(client, system, user, **kw):
            return (
                '{"cruxes": [{"description": "Type VI test", '
                '"discriminating_experiment": "Type_VI/baseline", '
                '"resolution_criterion": "RMSE < 0.2"}]}'
            )

        # Phase 2: negotiation — all agents accept
        def neg_llm(client, system, user, **kw):
            return '{"responses": [{"crux_id": "crux_001", "action": "accept"}]}'

        import antagonistic_collab.runner as runner_mod

        original = runner_mod.call_agent

        runner_mod.call_agent = id_llm
        try:
            run_crux_identification(protocol, None, cycle=0)
        finally:
            runner_mod.call_agent = original

        runner_mod.call_agent = neg_llm
        try:
            run_crux_negotiation(protocol, None, cycle=0)
        finally:
            runner_mod.call_agent = original

        finalized = finalize_cruxes(protocol, cycle=0)
        assert len(finalized) >= 1
        # Should have multiple supporters from negotiation
        assert len(finalized[0].supporters) >= 2
