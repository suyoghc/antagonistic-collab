"""Experiment framework for multi-condition ablation studies.

Loads YAML experiment configs, expands condition × ground_truth grids,
runs each condition by setting module globals, and collects results into
a comparison table.

Usage:
    python -m antagonistic_collab --experiment experiments/debate_ablation.yaml
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import numpy as np


VALID_SELECTION_STRATEGIES = {"thompson", "greedy"}
VALID_GROUND_TRUTHS = {"GCM", "SUSTAIN", "RULEX"}
VALID_DESIGN_SPACES = {"base", "richer", "continuous", "open"}


@dataclass
class ExperimentCondition:
    """Validated parameter set for one experiment run."""

    name: str
    true_model: str
    cycles: int = 5
    mode: str = "full_pool"
    design_space: str = "continuous"
    n_continuous_samples: int = 50
    selection_strategy: str = "thompson"
    learning_rate: float = 0.005
    arbiter: bool = True
    crux_weight: float = 0.3
    claim_responsive: bool = True
    no_debate: bool = False
    backend: str = "princeton"
    model: str = "gpt-4o"

    def __post_init__(self):
        if self.selection_strategy not in VALID_SELECTION_STRATEGIES:
            raise ValueError(
                f"selection_strategy must be one of {VALID_SELECTION_STRATEGIES}, "
                f"got '{self.selection_strategy}'"
            )
        if self.true_model not in VALID_GROUND_TRUTHS:
            raise ValueError(
                f"true_model must be one of {VALID_GROUND_TRUTHS}, "
                f"got '{self.true_model}'"
            )
        if self.design_space not in VALID_DESIGN_SPACES:
            raise ValueError(
                f"design_space must be one of {VALID_DESIGN_SPACES}, "
                f"got '{self.design_space}'"
            )


def load_experiment(yaml_path: str) -> list[ExperimentCondition]:
    """Parse a YAML experiment config and expand into a flat list of conditions.

    Each condition in the YAML is crossed with each ground truth, producing
    ``len(ground_truths) × len(conditions)`` ExperimentCondition objects.
    """
    import yaml

    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    defaults = raw.get("defaults", {})
    ground_truths = raw.get("ground_truths", ["GCM"])
    conditions = raw.get("conditions", {})

    result = []
    for cond_name, cond_overrides in conditions.items():
        for gt in ground_truths:
            # Merge: defaults → condition overrides → ground truth
            params = dict(defaults)
            if cond_overrides:
                params.update(cond_overrides)
            params["true_model"] = gt
            params["name"] = f"{cond_name}_{gt}"

            # Reject unknown keys — silent discard would hide typos
            valid_fields = {
                f.name for f in ExperimentCondition.__dataclass_fields__.values()
            }
            unknown = set(params.keys()) - valid_fields
            if unknown:
                raise ValueError(
                    f"Unknown keys in condition '{cond_name}': {sorted(unknown)}. "
                    f"Valid keys: {sorted(valid_fields)}"
                )

            result.append(ExperimentCondition(**params))

    return result


def run_condition(
    condition: ExperimentCondition, output_dir: str | None = None
) -> dict:
    """Run a single experiment condition and return analysis results.

    Sets module globals, creates client/state/protocol, runs the cycle loop,
    and returns a dict with winner, RMSE, gap, posterior, timing, etc.
    """
    import antagonistic_collab.runner as runner_mod
    from .debate_protocol import DebateProtocol, default_agent_configs
    from .epistemic_state import EpistemicState

    # Save original globals
    saved = {
        "_BATCH_MODE": runner_mod._BATCH_MODE,
        "_LLM_MODEL": runner_mod._LLM_MODEL,
        "_SELECTION_METHOD": runner_mod._SELECTION_METHOD,
        "_SELECTION_STRATEGY": runner_mod._SELECTION_STRATEGY,
        "_LEARNING_RATE": runner_mod._LEARNING_RATE,
        "_ARBITER": runner_mod._ARBITER,
        "_CRUX_WEIGHT": runner_mod._CRUX_WEIGHT,
        "_CLAIM_RESPONSIVE": runner_mod._CLAIM_RESPONSIVE,
        "_DESIGN_SPACE": runner_mod._DESIGN_SPACE,
        "_N_CONTINUOUS_SAMPLES": runner_mod._N_CONTINUOUS_SAMPLES,
        "_NO_DEBATE": runner_mod._NO_DEBATE,
    }

    try:
        # Set globals from condition
        runner_mod._BATCH_MODE = True
        runner_mod._LLM_MODEL = condition.model
        runner_mod._SELECTION_METHOD = "bayesian"
        runner_mod._SELECTION_STRATEGY = condition.selection_strategy
        runner_mod._LEARNING_RATE = condition.learning_rate
        runner_mod._ARBITER = condition.arbiter
        runner_mod._CRUX_WEIGHT = condition.crux_weight
        runner_mod._CLAIM_RESPONSIVE = condition.claim_responsive
        runner_mod._DESIGN_SPACE = condition.design_space
        runner_mod._N_CONTINUOUS_SAMPLES = condition.n_continuous_samples
        runner_mod._NO_DEBATE = condition.no_debate

        # Create client (None for no-debate, real client otherwise)
        client = None
        if not condition.no_debate:
            client = runner_mod._create_client(backend=condition.backend)

        # Set up protocol
        state = EpistemicState(domain="Human Categorization")
        agents = default_agent_configs()
        meta_agents = (
            runner_mod.create_default_meta_agents()
            if condition.arbiter and not condition.no_debate
            else None
        )
        protocol = DebateProtocol(state, agents, meta_agents=meta_agents)
        transcript = []

        if output_dir is None:
            output_dir = f"runs/{condition.name}"
        os.makedirs(output_dir, exist_ok=True)

        metadata = {
            "condition": condition.name,
            "true_model": condition.true_model,
            "llm_model": condition.model,
            "backend": condition.backend,
            "no_debate": condition.no_debate,
            "selection_strategy": condition.selection_strategy,
        }

        start_time = time.time()

        for cycle in range(condition.cycles):
            runner_mod.run_cycle(
                protocol,
                client,
                transcript,
                true_model=condition.true_model,
                output_dir=output_dir,
                metadata=metadata,
                mode=condition.mode,
            )

        total_time = time.time() - start_time

        # --- Analysis (extracted from validate_m12_live.py) ---
        board = state.prediction_leaderboard()
        winner = None
        winner_rmse = 999.0
        gap_pct = 0.0

        if board:
            sorted_board = sorted(
                board.items(), key=lambda x: x[1].get("mean_score", 999)
            )
            winner = sorted_board[0][0]
            winner_rmse = sorted_board[0][1].get("mean_score", 999.0)
            if len(sorted_board) > 1:
                runner_up_rmse = sorted_board[1][1].get("mean_score", 999.0)
                gap_pct = (
                    ((runner_up_rmse - winner_rmse) / runner_up_rmse * 100)
                    if runner_up_rmse > 0
                    else 0.0
                )

        # Posterior
        posterior_probs = {}
        posterior_entropy = None
        if state.model_posterior and "log_probs" in state.model_posterior:
            lp = np.array(state.model_posterior["log_probs"])
            probs = np.exp(lp - np.max(lp))
            probs = probs / probs.sum()
            model_names = state.model_posterior.get("model_names", [])
            posterior_probs = {
                model_names[i] if i < len(model_names) else f"model_{i}": float(
                    probs[i]
                )
                for i in range(len(probs))
            }
            posterior_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

        analysis = {
            "condition": condition.name,
            "true_model": condition.true_model,
            "no_debate": condition.no_debate,
            "selection_strategy": condition.selection_strategy,
            "cycles": condition.cycles,
            "total_time_s": total_time,
            "winner": winner,
            "winner_rmse": float(winner_rmse) if winner_rmse < 999 else None,
            "gap_pct": float(gap_pct),
            "posterior": posterior_probs,
            "posterior_entropy": posterior_entropy,
            "n_experiments": len(state.experiments),
            "leaderboard": {
                a: {"rmse": s.get("mean_score"), "n": s.get("n_predictions")}
                for a, s in board.items()
            }
            if board
            else {},
        }

        # Save per-condition analysis
        with open(os.path.join(output_dir, "analysis.json"), "w") as f:
            json.dump(analysis, f, indent=2, default=str)

        return analysis

    finally:
        # Restore globals
        for k, v in saved.items():
            setattr(runner_mod, k, v)


def run_experiment(yaml_path: str) -> dict:
    """Load an experiment config, run all conditions, and print comparison table.

    Returns a dict mapping condition names to their analysis results.
    """
    conditions = load_experiment(yaml_path)
    exp_name = os.path.splitext(os.path.basename(yaml_path))[0]
    base_dir = f"runs/{exp_name}"
    os.makedirs(base_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'=' * 70}")
    print(f"  Conditions: {len(conditions)}")
    print(f"  Ground truths: {sorted({c.true_model for c in conditions})}")
    cond_names = sorted({c.name.rsplit("_", 1)[0] for c in conditions})
    print(f"  Condition groups: {cond_names}")
    print()

    results = {}
    for i, cond in enumerate(conditions):
        cond_dir = os.path.join(base_dir, cond.name)
        print(f"\n[{i + 1}/{len(conditions)}] Running: {cond.name}")
        print(
            f"  true_model={cond.true_model}, strategy={cond.selection_strategy}, "
            f"no_debate={cond.no_debate}"
        )

        try:
            results[cond.name] = run_condition(cond, output_dir=cond_dir)
            winner = results[cond.name].get("winner", "?")
            rmse = results[cond.name].get("winner_rmse")
            rmse_str = f"{rmse:.3f}" if rmse else "?"
            print(f"  Result: winner={winner}, RMSE={rmse_str}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()
            results[cond.name] = {
                "error": str(e),
                "condition": cond.name,
                "true_model": cond.true_model,
            }

    _print_comparison_table(results, title=exp_name.upper())

    # Save combined results
    summary_path = os.path.join(base_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to: {summary_path}")

    return results


def _print_comparison_table(results: dict, title: str = "SUMMARY") -> None:
    """Print a condition × ground_truth comparison table from results dict."""
    # Infer ground truths and condition groups from result keys
    ground_truths = sorted(
        {r.get("true_model", k.rsplit("_", 1)[-1]) for k, r in results.items()}
        & {"GCM", "SUSTAIN", "RULEX"}
    )
    cond_groups = sorted(
        {
            k.rsplit("_", 1)[0]
            for k in results.keys()
            if k.rsplit("_", 1)[-1] in ground_truths
        }
    )

    print(f"\n\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")

    # Header
    header = f"{'Condition':<28}"
    for gt in ground_truths:
        header += f"  {'Winner':<16} {'RMSE':<7} {'Gap%':<6}"
    print(header)
    print("-" * len(header))

    for cg in cond_groups:
        row = f"{cg:<28}"
        for gt in ground_truths:
            key = f"{cg}_{gt}"
            r = results.get(key, {})
            if not r or "error" in r:
                row += f"  {'ERROR':<16} {'?':<7} {'?':<6}"
            else:
                w = r.get("winner", "?")
                if len(w) > 14:
                    w = w[:14] + ".."
                rmse = r.get("winner_rmse")
                rmse_str = f"{rmse:.3f}" if rmse else "?"
                gap = r.get("gap_pct", 0)
                row += f"  {w:<16} {rmse_str:<7} {gap:<6.1f}"
        print(row)


def merge_summaries(*paths: str, output: str | None = None) -> dict:
    """Merge multiple summary.json files and print a unified comparison table.

    Args:
        *paths: Paths to summary.json files (or directories containing one).
        output: Optional path to write merged summary.json. If None, writes
                to the parent directory of the first path.

    Returns:
        Merged results dict.
    """
    merged = {}
    for path in paths:
        # Accept either a file or a directory containing summary.json
        if os.path.isdir(path):
            path = os.path.join(path, "summary.json")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        with open(path) as f:
            data = json.load(f)
        n_before = len(merged)
        merged.update(data)
        print(
            f"  Loaded {len(data)} conditions from {path} ({len(merged) - n_before} new)"
        )

    if not merged:
        print("No results to merge.")
        return {}

    _print_comparison_table(merged, title="MERGED ABLATION SUMMARY")

    # Save merged output
    if output is None:
        first_dir = (
            os.path.dirname(paths[0]) if not os.path.isdir(paths[0]) else paths[0]
        )
        output = os.path.join(first_dir, "summary_merged.json")
    with open(output, "w") as f:
        json.dump(merged, f, indent=2, default=str)
    print(f"\nMerged results saved to: {output}")

    return merged
