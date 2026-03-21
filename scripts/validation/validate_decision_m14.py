"""
Decision-Domain M14 Validation — Computational-only baseline.

Runs the same EIG → observe → posterior update loop as categorization M14,
but for decision-making models (EU, CPT, Priority Heuristic). No LLM calls —
this tests whether the computational pipeline alone can identify the correct
decision model from synthetic choice data.

Conditions:
  no_debate:  EIG selection over gamble groups → synthetic data → posterior update
  (debate and arbiter conditions require agent configs — added separately)

Usage:
    python scripts/validation/validate_decision_m14.py
    python scripts/validation/validate_decision_m14.py CPT    # single model
    python scripts/validation/validate_decision_m14.py --n-cycles 10
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from antagonistic_collab.bayesian_selection import ModelPosterior
from antagonistic_collab.models.decision_eig import (
    GAMBLE_GROUPS,
    decision_predictions_for_eig,
    select_decision_experiment,
    update_decision_posterior,
)
from antagonistic_collab.models.decision_runner import (
    DECISION_EXPECTED_WINNER,
    GT_DECISION_PARAMS,
    MISSPEC_DECISION_PARAMS,
    compute_decision_predictions,
)

DECISION_AGENTS = ["CPT_Agent", "EU_Agent", "PH_Agent"]


def generate_observed_choices(gt_model, gamble_names, n_subjects, cycle, params=None):
    """Generate synthetic choice data from a GT decision model.

    Returns {gamble_name: observed P(choose A)}.
    """
    if params is None:
        params = GT_DECISION_PARAMS[gt_model]

    import hashlib

    seed_input = f"{cycle}_{gt_model}_{'_'.join(sorted(gamble_names))}"
    seed_hash = int(hashlib.md5(seed_input.encode()).hexdigest()[:8], 16) % 10000
    rng = np.random.default_rng(42 + seed_hash)

    observed = {}
    for gname in gamble_names:
        preds = compute_decision_predictions(gt_model, gname, params=params)
        p_a = preds[gname]
        p_clipped = np.clip(p_a, 0.01, 0.99)
        n_chose_a = rng.binomial(n_subjects, p_clipped)
        observed[gname] = n_chose_a / n_subjects

    return observed


def run_no_debate(
    gt_model,
    n_cycles=5,
    n_subjects=30,
    learning_rate=0.5,
    selection_strategy="thompson",
    agent_params=None,
    verbose=True,
):
    """Run computation-only decision model identification.

    Args:
        gt_model: "CPT", "EU", or "PH"
        n_cycles: number of EIG selection cycles
        n_subjects: simulated subjects per gamble
        learning_rate: likelihood tempering parameter
        selection_strategy: "greedy" or "thompson"
        agent_params: optional {agent_name: params} overrides (for misspec)
        verbose: print progress

    Returns:
        dict with results: winner, correct, rmse, gap, posterior history, etc.
    """
    posterior = ModelPosterior.uniform(DECISION_AGENTS)
    candidates = list(GAMBLE_GROUPS.values())
    group_names = list(GAMBLE_GROUPS.keys())

    history = []
    all_gambles_tested = []

    if verbose:
        print(f"\n  Prior: {dict(zip(DECISION_AGENTS, posterior.probs))}")

    for cycle in range(n_cycles):
        # Select experiment via EIG
        idx, eig_scores = select_decision_experiment(
            candidates,
            posterior,
            agent_params=agent_params,
            n_subjects=n_subjects,
            n_sim=200,
            seed=42 + cycle,
            learning_rate=learning_rate,
            selection_strategy=selection_strategy,
        )

        selected_gambles = candidates[idx]
        selected_group = group_names[idx]
        all_gambles_tested.extend(selected_gambles)

        # Generate synthetic observations from GT
        observed = generate_observed_choices(
            gt_model, selected_gambles, n_subjects, cycle
        )

        # Compute per-model RMSE on this experiment
        preds = decision_predictions_for_eig(selected_gambles, agent_params)
        obs_arr = np.array([observed[g] for g in selected_gambles])
        cycle_rmses = {}
        for agent_name in DECISION_AGENTS:
            pred_arr = preds[agent_name]
            rmse = float(np.sqrt(np.mean((pred_arr - obs_arr) ** 2)))
            cycle_rmses[agent_name] = rmse

        # Update posterior
        update_decision_posterior(
            posterior,
            observed,
            selected_gambles,
            agent_params=agent_params,
            n_subjects=n_subjects,
            learning_rate=learning_rate,
        )

        cycle_entry = {
            "cycle": cycle,
            "group": selected_group,
            "gambles": selected_gambles,
            "eig_scores": {g: float(s) for g, s in zip(group_names, eig_scores)},
            "best_eig": float(eig_scores[idx]),
            "rmses": cycle_rmses,
            "posterior": dict(
                zip(DECISION_AGENTS, [float(p) for p in posterior.probs])
            ),
            "entropy": float(posterior.entropy),
        }
        history.append(cycle_entry)

        if verbose:
            leader = DECISION_AGENTS[np.argmax(posterior.probs)]
            print(
                f"  Cycle {cycle}: {selected_group} "
                f"(EIG={eig_scores[idx]:.4f}) → "
                f"leader={leader} "
                f"[{', '.join(f'{a}={p:.3f}' for a, p in zip(DECISION_AGENTS, posterior.probs))}]"
            )

    # Final analysis
    winner_idx = np.argmax(posterior.probs)
    winner = DECISION_AGENTS[winner_idx]
    expected = DECISION_EXPECTED_WINNER[gt_model]
    correct = winner == expected

    # Compute gap: (runner_up_prob - winner_prob) / runner_up_prob * 100
    sorted_probs = sorted(enumerate(posterior.probs), key=lambda x: -x[1])
    winner_prob = sorted_probs[0][1]
    runner_up_prob = sorted_probs[1][1] if len(sorted_probs) > 1 else 0

    # Gap as percentage: how much better winner is than runner-up
    # Using RMSE-based gap is more standard, but posterior gap is cleaner
    # for the computational-only condition
    gap_pct = (
        float((winner_prob - runner_up_prob) / winner_prob * 100)
        if winner_prob > 0
        else 0
    )

    # Overall RMSE per model across all tested gambles
    all_preds = decision_predictions_for_eig(all_gambles_tested, agent_params)
    all_observed = {}
    for cycle in range(n_cycles):
        selected_gambles = history[cycle]["gambles"]
        obs = generate_observed_choices(gt_model, selected_gambles, n_subjects, cycle)
        all_observed.update(obs)

    overall_rmses = {}
    for agent_name in DECISION_AGENTS:
        pred_arr = np.array(
            [all_preds[agent_name][i] for i in range(len(all_gambles_tested))]
        )
        obs_arr = np.array([all_observed[g] for g in all_gambles_tested])
        overall_rmses[agent_name] = float(np.sqrt(np.mean((pred_arr - obs_arr) ** 2)))

    return {
        "ground_truth": gt_model,
        "condition": "no_debate",
        "n_cycles": n_cycles,
        "winner": winner,
        "expected": expected,
        "correct": correct,
        "winner_rmse": overall_rmses[winner],
        "gap_pct": gap_pct,
        "final_posterior": dict(
            zip(DECISION_AGENTS, [float(p) for p in posterior.probs])
        ),
        "final_entropy": float(posterior.entropy),
        "overall_rmses": overall_rmses,
        "gambles_tested": all_gambles_tested,
        "n_unique_gambles": len(set(all_gambles_tested)),
        "groups_selected": [h["group"] for h in history],
        "history": history,
    }


def print_summary_table(results):
    """Print a comparison table across ground truths."""
    print(f"\n{'=' * 80}")
    print("DECISION-DOMAIN M14 — NO-DEBATE BASELINE")
    print(f"{'=' * 80}")
    print(
        f"{'GT':<6} {'Condition':<12} {'Winner':<12} {'OK?':<5} "
        f"{'RMSE':<8} {'Gap%':<8} {'Entropy':<8} {'Groups Selected'}"
    )
    print("-" * 80)

    for model in ["CPT", "EU", "PH"]:
        r = results.get(model)
        if r is None:
            continue
        if "error" in r:
            print(f"{model:<6} {'no_debate':<12} ERROR: {r['error'][:40]}")
            continue

        w = r["winner"].replace("_Agent", "")
        correct = "Yes" if r["correct"] else "**No**"
        rmse = f"{r['winner_rmse']:.3f}"
        gap = f"{r['gap_pct']:.1f}"
        entropy = f"{r['final_entropy']:.3f}"
        groups = ", ".join(r["groups_selected"])
        print(
            f"{model:<6} {'no_debate':<12} {w:<12} {correct:<5} "
            f"{rmse:<8} {gap:<8} {entropy:<8} {groups}"
        )

    n_correct = sum(
        1 for r in results.values() if isinstance(r, dict) and r.get("correct")
    )
    print(f"\nCorrect: {n_correct}/{len(results)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Decision-domain M14 validation")
    parser.add_argument(
        "model",
        nargs="?",
        choices=["CPT", "EU", "PH"],
        help="Run a single ground-truth model",
    )
    parser.add_argument(
        "--n-cycles",
        type=int,
        default=5,
        help="Number of EIG selection cycles (default: 5)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Likelihood tempering rate (default: 0.1)",
    )
    parser.add_argument(
        "--strategy",
        choices=["greedy", "thompson"],
        default="thompson",
        help="Selection strategy (default: thompson)",
    )
    parser.add_argument(
        "--misspec", action="store_true", help="Use misspecified parameters for agents"
    )
    args = parser.parse_args()

    models = [args.model] if args.model else ["CPT", "EU", "PH"]

    # Set agent params (misspecified or default)
    agent_params = None
    if args.misspec:
        agent_params = {
            f"{m}_Agent": MISSPEC_DECISION_PARAMS[m] for m in ["CPT", "EU", "PH"]
        }
        print("Running with MISSPECIFIED parameters")

    output_dir = "runs/decision_m14_no_debate"
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    start = time.time()

    for gt_model in models:
        print(f"\n{'=' * 60}")
        print(f"Ground Truth: {gt_model}")
        print(f"{'=' * 60}")

        try:
            results[gt_model] = run_no_debate(
                gt_model,
                n_cycles=args.n_cycles,
                learning_rate=args.learning_rate,
                selection_strategy=args.strategy,
                agent_params=agent_params,
            )
        except Exception as e:
            print(f"\nERROR running {gt_model}: {e}")
            import traceback

            traceback.print_exc()
            results[gt_model] = {"error": str(e)}

    elapsed = time.time() - start
    print_summary_table(results)
    print(f"\nTotal time: {elapsed:.1f}s")

    # Save results
    tag = "misspec" if args.misspec else "correct"
    results_path = os.path.join(output_dir, f"decision_m14_{tag}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {results_path}")
