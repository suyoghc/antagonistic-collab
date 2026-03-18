"""
M16 Live Validation — Open Design Space.

Tests whether agents can propose diagnostic experiments via debate when no
pre-registered structures exist. Only agent-proposed structures enter the
EIG pool. EIG still scores and selects — the question is whether
debate-proposed structures are more diagnostic than registry structures.

Three conditions per ground truth:
  - closed_no_debate: computation only, full registry (_DESIGN_SPACE="continuous",
    _NO_DEBATE=True) — M14 baseline
  - closed_debate: standard debate, full registry (_DESIGN_SPACE="continuous",
    _NO_DEBATE=False, _ARBITER=False)
  - open_debate: agents propose all structures, EIG scores only those
    (_DESIGN_SPACE="open", _NO_DEBATE=False, _ARBITER=False)

No-debate open is logically impossible (no agents = no structures).

Usage:
    python scripts/validation/validate_m16_live.py                  # all 9 runs
    python scripts/validation/validate_m16_live.py GCM              # one GT, 3 conditions
    python scripts/validation/validate_m16_live.py --open-only      # 3 runs (open condition)
    python scripts/validation/validate_m16_live.py --closed-only    # 3 runs (closed no-debate)
    python scripts/validation/validate_m16_live.py --debate-only    # 3 runs (closed debate)
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from antagonistic_collab.epistemic_state import EpistemicState
from antagonistic_collab.debate_protocol import (
    DebateProtocol,
    default_agent_configs,
)
from antagonistic_collab.runner import (
    run_cycle,
    _create_client,
)
import antagonistic_collab.runner as runner_mod


# Agent name → model name mapping
AGENT_MODEL_MAP = {
    "Exemplar_Agent": "GCM",
    "Rule_Agent": "RULEX",
    "Clustering_Agent": "SUSTAIN",
}

EXPECTED_WINNER = {
    "GCM": "Exemplar_Agent",
    "SUSTAIN": "Clustering_Agent",
    "RULEX": "Rule_Agent",
}


def run_condition(
    client, true_model: str, design_space: str, debate: bool, n_cycles: int = 5
):
    """Run one condition.

    Args:
        design_space: "continuous" (registry) or "open" (agent-proposed only).
        debate: If False, skip all LLM phases (computational pipeline only).
    """
    if not debate:
        condition_label = "closed_no_debate"
    elif design_space == "open":
        condition_label = "open_debate"
    else:
        condition_label = "closed_debate"

    tag = f"m16_{condition_label}_{true_model}"
    output_dir = f"runs/{tag}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"M16 [{condition_label.upper()}] — Ground Truth: {true_model}")
    print(f"{'=' * 70}")
    print(f"  Cycles: {n_cycles}")
    print(f"  Design space: {design_space}")
    print(f"  _NO_DEBATE: {not debate}")
    print()

    # Fresh state and agents for each condition
    state = EpistemicState(domain="Human Categorization")
    agents = default_agent_configs()

    # Set modes
    runner_mod._NO_DEBATE = not debate
    runner_mod._ARBITER = False  # No arbiter in any M16 condition
    runner_mod._DESIGN_SPACE = design_space

    protocol = DebateProtocol(state, agents)
    transcript = []

    metadata = {
        "true_model": true_model,
        "llm_model": "gpt-4o",
        "backend": "princeton",
        "milestone": "M16",
        "condition": condition_label,
        "design_space": design_space,
        "arbiter": False,
    }

    # Track M16-specific metrics
    structures_per_cycle = []  # n_structures in pool each cycle
    start_time = time.time()

    for cycle in range(n_cycles):
        cycle_start = time.time()
        print(f"\n{'=' * 50}")
        print(f"  Cycle {cycle} ({time.time() - start_time:.0f}s elapsed)")
        print(f"{'=' * 50}")

        # Snapshot temporary structures before cycle
        pre_temp = set(getattr(protocol, "temporary_structures", {}).keys())

        run_cycle(
            protocol,
            client,
            transcript,
            true_model=true_model,
            critique_rounds=1,
            output_dir=output_dir,
            metadata=metadata,
            mode="full_pool",
        )

        # Count new proposals this cycle
        post_temp = set(getattr(protocol, "temporary_structures", {}).keys())
        new_this_cycle = post_temp - pre_temp
        structures_per_cycle.append(
            {
                "cycle": cycle,
                "total_structures": len(post_temp),
                "new_this_cycle": len(new_this_cycle),
                "new_names": sorted(new_this_cycle),
            }
        )

        elapsed = time.time() - cycle_start
        print(f"\n  Cycle {cycle} completed in {elapsed:.0f}s")
        if design_space == "open":
            print(f"  Structures in pool: {len(post_temp)}")

    total_time = time.time() - start_time

    # ── Analysis ──

    print(f"\n\n{'=' * 70}")
    print(f"M16 RESULTS [{condition_label.upper()}] — {true_model} ({total_time:.0f}s)")
    print(f"{'=' * 70}")

    # Prediction leaderboard
    board = state.prediction_leaderboard()
    winner = None
    winner_rmse = 999
    gap_pct = 0
    if board:
        print("\n### Prediction Leaderboard (RMSE - lower is better)")
        sorted_board = sorted(board.items(), key=lambda x: x[1].get("mean_score", 999))
        for rank, (agent, stats) in enumerate(sorted_board):
            mean = stats.get("mean_score")
            n = stats.get("n_predictions", 0)
            marker = " <-- WINNER" if rank == 0 else ""
            if mean is not None:
                print(
                    f"  {rank + 1}. {agent}: RMSE={mean:.4f} ({n} predictions){marker}"
                )

        winner = sorted_board[0][0]
        winner_rmse = sorted_board[0][1].get("mean_score", 999)
        if len(sorted_board) > 1:
            runner_up_rmse = sorted_board[1][1].get("mean_score", 999)
            gap_pct = (
                ((runner_up_rmse - winner_rmse) / runner_up_rmse * 100)
                if runner_up_rmse > 0
                else 0
            )
        print(f"\n  Winner: {winner} (gap: {gap_pct:.1f}%)")

    # Bayesian posterior
    if state.model_posterior and "log_probs" in state.model_posterior:
        lp = np.array(state.model_posterior["log_probs"])
        probs = np.exp(lp - np.max(lp))
        probs = probs / probs.sum()
        model_names = state.model_posterior.get("model_names", [])
        print("\n### Final Bayesian Posterior")
        for i, p in enumerate(probs):
            name = model_names[i] if i < len(model_names) else f"model_{i}"
            marker = " <-- MOST PROBABLE" if p == max(probs) else ""
            print(f"  P({name}) = {p:.4f}{marker}")

    # Experiments
    print("\n### Experiments")
    selected_structs = []
    for exp in state.experiments:
        ds = exp.design_spec or {}
        s = ds.get("structure_name", "?")
        c = ds.get("condition", "?")
        selected_structs.append(s)
        print(f"  Cycle {exp.cycle}: {s} / {c}")

    # Structure proposal metrics (open mode)
    if design_space == "open":
        print("\n### Structure Proposals")
        total_structs = len(getattr(protocol, "temporary_structures", {}))
        print(f"  Total structures proposed: {total_structs}")
        for cycle_info in structures_per_cycle:
            c = cycle_info["cycle"]
            n = cycle_info["new_this_cycle"]
            names = (
                ", ".join(cycle_info["new_names"])
                if cycle_info["new_names"]
                else "(none)"
            )
            print(f"  Cycle {c}: +{n} new ({names})")

        # Which agent's structures were selected by EIG?
        all_proposed = set(getattr(protocol, "temporary_structures", {}).keys())
        selected_proposed = [s for s in selected_structs if s in all_proposed]
        print(
            f"\n  Proposed structures selected by EIG: {len(selected_proposed)}/{n_cycles}"
        )

    # Save analysis JSON
    analysis = {
        "ground_truth": true_model,
        "milestone": "M16",
        "condition": condition_label,
        "design_space": design_space,
        "n_cycles": n_cycles,
        "total_time_s": total_time,
        "winner": winner,
        "correct": winner == EXPECTED_WINNER.get(true_model),
        "winner_rmse": float(winner_rmse) if winner_rmse < 999 else None,
        "gap_pct": float(gap_pct),
        "selected_structures": selected_structs,
        "structures_per_cycle": structures_per_cycle,
        "total_proposed_structures": len(getattr(protocol, "temporary_structures", {})),
        "leaderboard": {
            a: {"rmse": s.get("mean_score"), "n": s.get("n_predictions")}
            for a, s in board.items()
        }
        if board
        else {},
    }
    analysis_path = os.path.join(output_dir, f"{tag}_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    return analysis


def setup_client():
    """Create Princeton client, loading .env if needed."""
    if not os.environ.get("AI_SANDBOX_KEY"):
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, val = line.partition("=")
                        os.environ[key.strip()] = val.strip()

    if not os.environ.get("AI_SANDBOX_KEY"):
        print("ERROR: AI_SANDBOX_KEY environment variable is not set.")
        print("Set it before running: export AI_SANDBOX_KEY=your_key_here")
        sys.exit(1)

    return _create_client(backend="princeton")


def set_common_globals():
    """Set shared globals for M16 validation."""
    runner_mod._LLM_MODEL = "gpt-4o"
    runner_mod._BATCH_MODE = True
    runner_mod._SELECTION_METHOD = "bayesian"
    runner_mod._SELECTION_STRATEGY = "thompson"
    runner_mod._LEARNING_RATE = 0.005
    runner_mod._ARBITER = False
    runner_mod._CLAIM_RESPONSIVE = True
    runner_mod._DESIGN_SPACE = "continuous"
    runner_mod._N_CONTINUOUS_SAMPLES = 50
    runner_mod._NORMALIZE_CLAIMS = True
    runner_mod._FUZZY_STRUCTURE_MATCH = True
    runner_mod._CRUX_WEIGHT = 0.0


def print_comparison_table(results, conditions_run):
    """Print three-way comparison across ground truths."""
    cond_labels = [c[0] for c in conditions_run]

    print(f"\n\n{'=' * 100}")
    print("M16 COMPARISON: OPEN vs CLOSED DESIGN SPACE")
    print(f"{'=' * 100}")

    header = (
        f"{'GT':<10} {'Condition':<18} {'Winner':<20} {'OK?':<5} "
        f"{'RMSE':<8} {'Gap%':<8} {'#Structs':<10}"
    )
    print(header)
    print("-" * 100)

    for model in ["GCM", "SUSTAIN", "RULEX"]:
        for cond in cond_labels:
            key = f"{model}_{cond}"
            r = results.get(key, {})
            if not r:
                continue
            if "error" in r:
                print(f"{model:<10} {cond:<18} ERROR: {r['error'][:40]}")
                continue
            w = r.get("winner", "?")
            correct = "Yes" if r.get("correct") else "No"
            rmse = r.get("winner_rmse")
            rmse_str = f"{rmse:.3f}" if rmse else "?"
            gap = r.get("gap_pct", 0)
            n_structs = r.get("total_proposed_structures", "-")
            print(
                f"{model:<10} {cond:<18} {w:<20} {correct:<5} "
                f"{rmse_str:<8} {gap:<8.1f} {str(n_structs):<10}"
            )
        print()

    # Summary: pairwise advantages
    print("\n### Gap Advantage (pp over closed_no_debate baseline)")
    for model in ["GCM", "SUSTAIN", "RULEX"]:
        nd = results.get(f"{model}_closed_no_debate", {})
        nd_gap = nd.get("gap_pct", 0)
        parts = [f"closed_no_debate={nd_gap:.1f}%"]
        for cond in ["closed_debate", "open_debate"]:
            r = results.get(f"{model}_{cond}", {})
            if r and "error" not in r:
                g = r.get("gap_pct", 0)
                diff = g - nd_gap
                parts.append(f"{cond}={g:.1f}% ({diff:+.1f}pp)")
        print(f"  {model}: {', '.join(parts)}")


if __name__ == "__main__":
    set_common_globals()

    models = ["GCM", "SUSTAIN", "RULEX"]

    # All three conditions by default:
    #   (label, design_space, debate)
    ALL_CONDITIONS = [
        ("closed_no_debate", "continuous", False),
        ("closed_debate", "continuous", True),
        ("open_debate", "open", True),
    ]
    selected_conditions = None  # None = all

    # Parse CLI args
    for arg in sys.argv[1:]:
        if arg in models:
            models = [arg]
        elif arg == "--open-only":
            selected_conditions = [("open_debate", "open", True)]
        elif arg == "--closed-only":
            selected_conditions = [("closed_no_debate", "continuous", False)]
        elif arg == "--debate-only":
            selected_conditions = [("closed_debate", "continuous", True)]

    conditions = selected_conditions or ALL_CONDITIONS

    # Only create LLM client if a debate condition is requested
    needs_llm = any(debate for _, _, debate in conditions)
    client = setup_client() if needs_llm else None

    results = {}
    for true_model in models:
        for cond_label, ds, debate_flag in conditions:
            key = f"{true_model}_{cond_label}"
            try:
                results[key] = run_condition(
                    client,
                    true_model,
                    design_space=ds,
                    debate=debate_flag,
                    n_cycles=5,
                )
            except Exception as e:
                print(f"\nERROR running {key}: {e}")
                import traceback

                traceback.print_exc()
                results[key] = {"error": str(e)}

    print_comparison_table(results, conditions)

    # Save combined results
    combined_path = "runs/m16_summary.json"
    os.makedirs("runs", exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to: {combined_path}")
