"""
R-IDeA vs EIG Head-to-Head Validation.

Runs the same closed_no_debate condition (pure computation, no LLM calls)
with both EIG and R-IDeA scoring. Compares gap%, RMSE, and per-GT bias
patterns to test whether R-IDeA reduces model-type variance.

Usage:
    python scripts/validation/validate_ridea.py              # all 3 GTs
    python scripts/validation/validate_ridea.py GCM          # one GT
    python scripts/validation/validate_ridea.py --misspec    # with wrong params
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
from antagonistic_collab.bayesian_selection import (
    ModelPosterior,
    compute_eig,
    generate_full_candidate_pool,
    compute_log_likelihood,
)
from antagonistic_collab.ridea import (
    compute_ridea_scores,
)
import antagonistic_collab.runner as runner_mod

# ── Misspecification params (from M15) ──
import copy

GT_PARAMS = {
    "GCM": {"c": 4.0, "r": 1, "gamma": 1.0},
    "SUSTAIN": {"r": 9.01, "beta": 1.252, "d": 16.924, "eta": 0.092},
    "RULEX": {"p_single": 0.5, "p_conj": 0.3, "error_tolerance": 0.1},
}

MISSPEC_PARAMS = {
    "GCM": {"c": 0.5, "r": 1, "gamma": 1.0},
    "SUSTAIN": {"r": 3.0, "beta": 1.252, "d": 16.924, "eta": 0.15},
    "RULEX": {"p_single": 0.3, "p_conj": 0.3, "error_tolerance": 0.25},
}

EXPECTED_WINNER = {
    "GCM": "Exemplar_Agent",
    "SUSTAIN": "Clustering_Agent",
    "RULEX": "Rule_Agent",
}


def get_pool_predictions(protocol, pool):
    """Extract model predictions for every candidate in the pool.

    Returns list of dicts, one per candidate:
        [{"GCM": array, "SUSTAIN": array, "RULEX": array}, ...]
    """
    pool_preds = []
    for struct_name, condition in pool:
        model_predictions = {}
        for agent_config in protocol.agent_configs:
            preds = protocol.compute_model_predictions(
                agent_config, struct_name, condition
            )
            item_keys = sorted(
                [k for k in preds if k.startswith("item_")],
                key=lambda k: int(k.split("_")[1]),
            )
            pred_array = np.array([preds[k] for k in item_keys])
            model_predictions[agent_config.name] = pred_array
        pool_preds.append(model_predictions)
    return pool_preds


def run_ridea_condition(
    true_model: str,
    oed_type: str = "ridea",
    misspec: bool = False,
    alpha: float = 0.3,
    beta: float = 0.3,
    n_cycles: int = 5,
):
    """Run one GT with R-IDeA or EIG scoring (no debate, no LLM calls).

    This is a standalone computation loop that bypasses the debate phases
    entirely. Each cycle:
      1. Generate candidate pool
      2. Score all candidates (EIG or R-IDeA)
      3. Select via Thompson sampling
      4. Execute synthetic experiment
      5. Update posterior
    """
    misspec_tag = "_misspec" if misspec else ""
    tag = f"ridea_{oed_type}{misspec_tag}_{true_model}"
    output_dir = f"runs/{tag}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"R-IDeA VALIDATION [{oed_type.upper()}] — GT: {true_model}")
    print(f"{'=' * 70}")
    print(f"  OED type: {oed_type}")
    print(f"  Misspecified: {misspec}")
    if oed_type == "ridea":
        print(f"  alpha (rep): {alpha}, beta (deamp): {beta}")
    print(f"  Cycles: {n_cycles}")
    print()

    # Fresh state and agents
    state = EpistemicState(domain="Human Categorization")
    agents = default_agent_configs()

    # Apply misspecification if requested
    if misspec:
        target_agent = EXPECTED_WINNER[true_model]
        for agent in agents:
            if agent.name == target_agent:
                agent.default_params = copy.deepcopy(MISSPEC_PARAMS[true_model])
                print(f"  Misspecified {agent.name}: {agent.default_params}")

    # Set computation-only mode
    runner_mod._NO_DEBATE = True
    runner_mod._ARBITER = False
    runner_mod._DESIGN_SPACE = "continuous"
    runner_mod._N_CONTINUOUS_SAMPLES = 50
    runner_mod._CRUX_WEIGHT = 0.0

    protocol = DebateProtocol(state, agents, meta_agents=[])

    # Initialize posterior
    model_names = [a.name for a in agents]
    posterior = ModelPosterior.uniform(model_names)
    learning_rate = 0.005

    # Track experiment history for representativeness
    previous_predictions = []
    selected_structures = []
    # Manual leaderboard (agent_name → list of RMSE scores)
    leaderboard = {name: [] for name in model_names}
    start_time = time.time()

    for cycle in range(n_cycles):
        cycle_start = time.time()
        print(f"\n--- Cycle {cycle} ({time.time() - start_time:.0f}s) ---")

        # 1. Generate candidate pool
        pool = generate_full_candidate_pool(
            protocol,
            design_space="continuous",
            n_continuous_samples=50,
            continuous_seed=42 + cycle,
        )
        print(f"  Pool: {len(pool)} candidates")

        # 2. Get predictions for all candidates
        pool_preds = get_pool_predictions(protocol, pool)

        # 3. Score and select
        if oed_type == "eig":
            # Score with EIG only (for comparison)
            eig_scores = []
            for i, preds in enumerate(pool_preds):
                eig = compute_eig(
                    preds, posterior, n_subjects=20, n_sim=200,
                    seed=42 + i, learning_rate=learning_rate,
                )
                eig_scores.append(eig)

            # Thompson sampling
            arr = np.array(eig_scores)
            total = arr.sum()
            rng = np.random.default_rng(42 + cycle)
            if total > 0:
                weights = arr / total
                best_idx = int(rng.choice(len(eig_scores), p=weights))
            else:
                best_idx = int(rng.integers(len(eig_scores)))
            scores = eig_scores

        elif oed_type == "ridea":
            # Score with R-IDeA
            scores = compute_ridea_scores(
                pool_preds, posterior, previous_predictions,
                alpha=alpha, beta=beta,
                n_subjects=20, n_sim=200,
                seed=42 + cycle, learning_rate=learning_rate,
            )

            # Thompson sampling
            arr = np.array(scores)
            total = arr.sum()
            rng = np.random.default_rng(42 + cycle)
            if total > 0:
                weights = arr / total
                best_idx = int(rng.choice(len(scores), p=weights))
            else:
                best_idx = int(rng.integers(len(scores)))

        else:
            raise ValueError(f"Unknown OED type: {oed_type}")

        selected_struct, selected_cond = pool[best_idx]
        selected_structures.append(selected_struct)
        print(f"  Selected: {selected_struct} / {selected_cond} (score={scores[best_idx]:.4f})")

        # Print top 5 for comparison
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        for rank, (idx, sc) in enumerate(ranked[:5]):
            s, c = pool[idx]
            marker = " <-- SELECTED" if idx == best_idx else ""
            print(f"    {rank+1}. {s} / {c} — {sc:.4f}{marker}")

        # Track predictions for representativeness in future cycles
        previous_predictions.append(pool_preds[best_idx])

        # 4. Execute synthetic experiment via protocol's runner
        design_spec = {
            "structure_name": selected_struct,
            "condition": selected_cond,
        }
        result = protocol._synthetic_runner(
            design_spec, true_model=true_model, cycle=cycle
        )
        item_acc = result.get("item_accuracies", {})
        mean_acc = result.get("mean_accuracy", 0)
        print(f"  Result: mean_accuracy={mean_acc:.3f}")

        # Convert item_accuracies dict to sorted array
        item_keys = sorted(item_acc.keys(), key=lambda k: int(k.split("_")[1]))
        obs_array = np.array([item_acc[k] for k in item_keys])

        # 5. Score predictions against data
        selected_preds = pool_preds[best_idx]
        for agent_name, pred_array in selected_preds.items():
            trimmed = pred_array[: len(obs_array)]
            obs_trimmed = obs_array[: len(trimmed)]
            rmse = float(np.sqrt(np.mean((trimmed - obs_trimmed) ** 2)))
            leaderboard[agent_name].append(rmse)

        # 6. Update posterior
        log_lls = np.zeros(len(model_names))
        for m_idx, name in enumerate(model_names):
            pred = selected_preds[name]
            pred_trimmed = pred[: len(obs_array)]
            obs_trimmed = obs_array[: len(pred_trimmed)]
            log_lls[m_idx] = compute_log_likelihood(
                obs_trimmed, pred_trimmed, n_subjects=20
            )
        posterior.update(log_lls, learning_rate=learning_rate)

        probs = posterior.probs
        print(f"  Posterior: {', '.join(f'P({n})={p:.3f}' for n, p in zip(model_names, probs))}")
        print(f"  Entropy: {posterior.entropy:.4f}")
        print(f"  Cycle time: {time.time() - cycle_start:.0f}s")

    total_time = time.time() - start_time

    # ── Analysis ──

    print(f"\n\n{'=' * 70}")
    print(f"RESULTS [{oed_type.upper()}] — {true_model} ({total_time:.0f}s)")
    print(f"{'=' * 70}")

    # Compute leaderboard from tracked scores
    board = {
        name: {"mean_score": np.mean(scores) if scores else 999, "n": len(scores)}
        for name, scores in leaderboard.items()
    }
    winner = None
    winner_rmse = 999
    gap_pct = 0
    if board:
        sorted_board = sorted(board.items(), key=lambda x: x[1]["mean_score"])
        for rank, (agent, stats) in enumerate(sorted_board):
            mean = stats["mean_score"]
            marker = " <-- WINNER" if rank == 0 else ""
            print(f"  {rank+1}. {agent}: RMSE={mean:.4f} ({stats['n']} predictions){marker}")

        winner = sorted_board[0][0]
        winner_rmse = sorted_board[0][1]["mean_score"]
        if len(sorted_board) > 1:
            runner_up_rmse = sorted_board[1][1]["mean_score"]
            gap_pct = (
                ((runner_up_rmse - winner_rmse) / runner_up_rmse * 100)
                if runner_up_rmse > 0 else 0
            )

    print(f"\n  Winner: {winner} (gap: {gap_pct:.1f}%)")
    print(f"  Correct: {winner == EXPECTED_WINNER.get(true_model)}")
    print(f"  Structures selected: {selected_structures}")

    analysis = {
        "ground_truth": true_model,
        "oed_type": oed_type,
        "misspec": misspec,
        "alpha": alpha if oed_type == "ridea" else None,
        "beta": beta if oed_type == "ridea" else None,
        "n_cycles": n_cycles,
        "total_time_s": total_time,
        "winner": winner,
        "correct": winner == EXPECTED_WINNER.get(true_model),
        "winner_rmse": float(winner_rmse) if winner_rmse < 999 else None,
        "gap_pct": float(gap_pct),
        "selected_structures": selected_structures,
    }

    analysis_path = os.path.join(output_dir, f"{tag}_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    return analysis


def print_comparison(results):
    """Print EIG vs R-IDeA comparison table."""
    print(f"\n\n{'=' * 80}")
    print("EIG vs R-IDeA COMPARISON")
    print(f"{'=' * 80}")
    print(f"{'GT':<10} {'OED':<10} {'Winner':<20} {'OK?':<5} {'RMSE':<8} {'Gap%':<8}")
    print("-" * 80)

    for model in ["GCM", "SUSTAIN", "RULEX"]:
        for oed in ["eig", "ridea"]:
            key = f"{model}_{oed}"
            r = results.get(key, {})
            if not r:
                continue
            if "error" in r:
                print(f"{model:<10} {oed:<10} ERROR: {r['error'][:40]}")
                continue
            w = r.get("winner", "?")
            correct = "Yes" if r.get("correct") else "No"
            rmse = r.get("winner_rmse")
            rmse_str = f"{rmse:.3f}" if rmse else "?"
            gap = r.get("gap_pct", 0)
            print(f"{model:<10} {oed:<10} {w:<20} {correct:<5} {rmse_str:<8} {gap:<8.1f}")
        print()

    # Bias summary
    print("\n### Model-type variance (lower = more fair)")
    for oed in ["eig", "ridea"]:
        gaps = []
        for model in ["GCM", "SUSTAIN", "RULEX"]:
            key = f"{model}_{oed}"
            r = results.get(key, {})
            if r and "error" not in r:
                gaps.append(r.get("gap_pct", 0))
        if gaps:
            mean_gap = np.mean(gaps)
            std_gap = np.std(gaps)
            min_gap = np.min(gaps)
            print(f"  {oed}: mean={mean_gap:.1f}%, std={std_gap:.1f}%, min={min_gap:.1f}%")


if __name__ == "__main__":
    runner_mod._LLM_MODEL = "gpt-4o"
    runner_mod._BATCH_MODE = True
    runner_mod._SELECTION_METHOD = "bayesian"
    runner_mod._SELECTION_STRATEGY = "thompson"
    runner_mod._LEARNING_RATE = 0.005
    runner_mod._NORMALIZE_CLAIMS = True
    runner_mod._FUZZY_STRUCTURE_MATCH = True

    models = ["GCM", "SUSTAIN", "RULEX"]
    misspec = False

    for arg in sys.argv[1:]:
        if arg in models:
            models = [arg]
        elif arg == "--misspec":
            misspec = True

    results = {}
    for true_model in models:
        for oed_type in ["eig", "ridea"]:
            key = f"{true_model}_{oed_type}"
            try:
                results[key] = run_ridea_condition(
                    true_model,
                    oed_type=oed_type,
                    misspec=misspec,
                    n_cycles=5,
                )
            except Exception as e:
                print(f"\nERROR running {key}: {e}")
                import traceback
                traceback.print_exc()
                results[key] = {"error": str(e)}

    print_comparison(results)

    # Save combined results
    tag = "ridea_misspec" if misspec else "ridea_correctspec"
    combined_path = f"runs/{tag}_comparison.json"
    os.makedirs("runs", exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {combined_path}")
