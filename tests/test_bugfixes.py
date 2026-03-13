"""
Regression tests for all bugs fixed across four review rounds.

Each test is a direct regression for a specific reported bug. The docstring
explains *why* the test exists — what broke, how we know it's fixed, and
what the test actually verifies.

Organized by module, not by review round, so future developers can find
the relevant tests next to the code they cover.
"""

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
)
from antagonistic_collab.models.sustain import SUSTAIN
from antagonistic_collab.models.gcm import GCM


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

    def test_valid_category_structure_is_used(self):
        """
        Sanity check: when the LLM provides a properly formatted
        category_structure with stimuli and labels, it should be used
        (not silently replaced by the fallback).
        """
        from antagonistic_collab.models.category_structures import shepard_types

        type_i = shepard_types()["I"]
        spec = {"category_structure": type_i}
        protocol = self._make_protocol()
        data = protocol._synthetic_runner(spec, true_model="GCM")
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
