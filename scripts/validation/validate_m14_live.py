"""
M14 Live Validation + Ablation — Real LLM calls.

Runs 5-cycle debates with M14 features (claim-directed selection, validated
param revisions, claim auto-resolution). Tests 3 ground truths.

Ablation mode (--ablation): disables claim-directed selection by setting
crux_weight=0, keeping param validation and claim resolution active.
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
    PARAMETRIC_CONDITIONS,
    STRUCTURE_REGISTRY,
    CONDITION_EFFECTS,
)
from antagonistic_collab.runner import (
    run_cycle,
    create_default_meta_agents,
    _create_client,
)
import antagonistic_collab.runner as runner_mod


def run_validation(client, true_model: str, n_cycles: int = 5, ablation: bool = False):
    """Run a full M14 validation with real LLM calls."""

    tag = "m14_ablation" if ablation else "m14_val"
    output_dir = f"runs/{tag}_{true_model}"
    os.makedirs(output_dir, exist_ok=True)

    mode_label = "ABLATION (crux_weight=0)" if ablation else "FULL"

    print(f"\n{'=' * 70}")
    print(f"M14 VALIDATION [{mode_label}] — Ground Truth: {true_model}")
    print(f"{'=' * 70}")
    print(f"  Cycles: {n_cycles}")
    print(f"  crux_weight: {runner_mod._CRUX_WEIGHT}")
    print(f"  claim_responsive: {runner_mod._CLAIM_RESPONSIVE}")
    print(f"  design_space: {runner_mod._DESIGN_SPACE}")
    print(f"  Output: {output_dir}")
    print()

    state = EpistemicState(domain="Human Categorization")
    agents = default_agent_configs()
    protocol = DebateProtocol(state, agents, meta_agents=create_default_meta_agents())
    transcript = []

    metadata = {
        "true_model": true_model,
        "llm_model": "gpt-4o",
        "backend": "princeton",
        "milestone": "M14",
        "mode": "ablation" if ablation else "full",
        "crux_weight": runner_mod._CRUX_WEIGHT,
    }

    start_time = time.time()

    for cycle in range(n_cycles):
        cycle_start = time.time()
        print(f"\n{'=' * 50}")
        print(f"  Starting Cycle {cycle} ({time.time() - start_time:.0f}s elapsed)")
        print(f"{'=' * 50}")

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

        elapsed = time.time() - cycle_start
        print(f"\n  Cycle {cycle} completed in {elapsed:.0f}s")

    total_time = time.time() - start_time

    # --- Analysis ---
    print(f"\n\n{'=' * 70}")
    print(f"M14 RESULTS [{mode_label}] — {true_model} ({total_time:.0f}s)")
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
                print(f"  {rank + 1}. {agent}: RMSE={mean:.4f} ({n} predictions){marker}")

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

    # Claim ledger analysis (M14-specific)
    ledger = state.claim_ledger
    n_total = len(ledger)
    n_testable = sum(1 for c in ledger if c.testable)
    n_confirmed = sum(1 for c in ledger if c.status == "confirmed")
    n_falsified = sum(1 for c in ledger if c.status == "falsified")
    n_untested = sum(1 for c in ledger if c.status == "untested")

    print(f"\n### Claim Ledger (M14)")
    print(f"  Total: {n_total}, Testable: {n_testable}")
    print(f"  Confirmed: {n_confirmed}, Falsified: {n_falsified}, Untested: {n_untested}")
    print(f"  Auto-resolved: {n_confirmed + n_falsified}")

    # Param validation analysis
    print(f"\n### Parameter Validation (M14)")
    param_msgs = [
        m for m in transcript
        if isinstance(m, dict) and "param_validation" in str(m.get("phase", ""))
    ]
    print(f"  (Check log for 'Params accepted/REJECTED' lines)")

    # Experiments
    print(f"\n### Experiments")
    selected_structs = []
    selected_conds = []
    for exp in state.experiments:
        ds = exp.design_spec or {}
        s = ds.get("structure_name", "?")
        c = ds.get("condition", "?")
        selected_structs.append(s)
        selected_conds.append(c)
        print(f"  Cycle {exp.cycle}: {s} / {c}")

    # Bayesian posterior
    if state.model_posterior and "log_probs" in state.model_posterior:
        lp = np.array(state.model_posterior["log_probs"])
        probs = np.exp(lp - np.max(lp))
        probs = probs / probs.sum()
        model_names = state.model_posterior.get("model_names", [])
        print(f"\n### Final Bayesian Posterior")
        for i, p in enumerate(probs):
            name = model_names[i] if i < len(model_names) else f"model_{i}"
            marker = " <-- MOST PROBABLE" if p == max(probs) else ""
            print(f"  P({name}) = {p:.4f}{marker}")

    # Crux analysis
    total_cruxes = len(state.cruxes)
    accepted = [c for c in state.cruxes if c.status == "accepted"]
    print(f"\n### Cruxes: {total_cruxes} total, {len(accepted)} accepted")

    # Save analysis
    expected = {
        "GCM": "Exemplar_Agent",
        "SUSTAIN": "Clustering_Agent",
        "RULEX": "Rule_Agent",
    }
    analysis = {
        "ground_truth": true_model,
        "milestone": "M14",
        "mode": "ablation" if ablation else "full",
        "crux_weight": runner_mod._CRUX_WEIGHT,
        "n_cycles": n_cycles,
        "total_time_s": total_time,
        "winner": winner,
        "correct": winner == expected.get(true_model),
        "winner_rmse": float(winner_rmse) if winner_rmse < 999 else None,
        "gap_pct": float(gap_pct),
        "n_claims": n_total,
        "n_testable": n_testable,
        "n_confirmed": n_confirmed,
        "n_falsified": n_falsified,
        "n_untested": n_untested,
        "selected_structures": selected_structs,
        "selected_conditions": selected_conds,
        "total_cruxes": total_cruxes,
        "accepted_cruxes": len(accepted),
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
    """Set shared globals for M14 validation."""
    runner_mod._LLM_MODEL = "gpt-4o"
    runner_mod._BATCH_MODE = True
    runner_mod._SELECTION_METHOD = "bayesian"
    runner_mod._SELECTION_STRATEGY = "thompson"
    runner_mod._LEARNING_RATE = 0.005
    runner_mod._ARBITER = True
    runner_mod._CLAIM_RESPONSIVE = True
    runner_mod._DESIGN_SPACE = "continuous"
    runner_mod._N_CONTINUOUS_SAMPLES = 50
    runner_mod._NORMALIZE_CLAIMS = True
    runner_mod._FUZZY_STRUCTURE_MATCH = True


def print_summary_table(results, tag):
    """Print a comparison table across ground truths."""
    expected = {
        "GCM": "Exemplar_Agent",
        "SUSTAIN": "Clustering_Agent",
        "RULEX": "Rule_Agent",
    }
    print(f"\n\n{'=' * 70}")
    print(f"M14 {tag} SUMMARY")
    print(f"{'=' * 70}")
    print(
        f"{'GT':<10} {'Winner':<20} {'OK?':<5} {'RMSE':<8} {'Gap%':<8} "
        f"{'Claims':<8} {'Resolved':<10}"
    )
    print("-" * 79)

    for model in ["GCM", "SUSTAIN", "RULEX"]:
        r = results.get(model, {})
        if "error" in r:
            print(f"{model:<10} ERROR: {r['error'][:50]}")
            continue
        w = r.get("winner", "?")
        correct = "Yes" if w == expected.get(model) else "No"
        rmse = r.get("winner_rmse")
        rmse_str = f"{rmse:.3f}" if rmse else "?"
        gap = r.get("gap_pct", 0)
        nc = r.get("n_claims", 0)
        resolved = r.get("n_confirmed", 0) + r.get("n_falsified", 0)
        print(
            f"{model:<10} {w:<20} {correct:<5} {rmse_str:<8} {gap:<8.1f} "
            f"{nc:<8} {resolved:<10}"
        )


if __name__ == "__main__":
    ablation = "--ablation" in sys.argv

    set_common_globals()

    if ablation:
        runner_mod._CRUX_WEIGHT = 0.0  # Disable crux+claim boosting
    else:
        runner_mod._CRUX_WEIGHT = 0.3

    client = setup_client()

    models = ["GCM", "SUSTAIN", "RULEX"]

    # Allow running a single model via CLI arg
    for arg in sys.argv[1:]:
        if arg in models:
            models = [arg]
            break

    tag = "ABLATION" if ablation else "VALIDATION"
    results = {}
    for true_model in models:
        try:
            results[true_model] = run_validation(
                client, true_model, n_cycles=5, ablation=ablation
            )
        except Exception as e:
            print(f"\nERROR running {true_model}: {e}")
            import traceback
            traceback.print_exc()
            results[true_model] = {"error": str(e)}

    print_summary_table(results, tag)

    # Save combined results
    suffix = "ablation" if ablation else "validation"
    combined_path = f"runs/m14_{suffix}_summary.json"
    os.makedirs("runs", exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to: {combined_path}")
