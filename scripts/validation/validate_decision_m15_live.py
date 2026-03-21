"""
Decision-Domain M15 Live Validation — Debate under Misspecification.

Tests whether LLM debate can recover from parameter misspecification in the
decision-making domain. All agents start with calibrated wrong params.
Ground truth uses correct params for synthetic data generation.

Three conditions per ground truth:
  - No-debate: params stay fixed, EIG + posterior update only
  - Debate: agents see prediction errors, diagnose + propose param revisions
  - Arbiter: debate + crux protocol + meta-agents (Integrator + Critic)

This is the NeurIPS experiment: if the same pattern replicates across both
categorization and decision-making domains, the implicit-prior finding is
domain-general.

Usage:
    python scripts/validation/validate_decision_m15_live.py
    python scripts/validation/validate_decision_m15_live.py CPT
    python scripts/validation/validate_decision_m15_live.py --no-debate-only
    python scripts/validation/validate_decision_m15_live.py --debate-only
    python scripts/validation/validate_decision_m15_live.py --arbiter
    python scripts/validation/validate_decision_m15_live.py --arbiter-only
    python scripts/validation/validate_decision_m15_live.py --backend princeton
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from antagonistic_collab.models.decision_debate_runner import (
    run_decision_debate,
)
from antagonistic_collab.models.decision_runner import (
    DECISION_AGENT_MAP,
    DECISION_EXPECTED_WINNER,
    GT_DECISION_PARAMS,
    MISSPEC_DECISION_PARAMS,
)
import antagonistic_collab.runner as runner_mod
from antagonistic_collab.runner import _create_client, call_agent

BACKEND_MODELS = {
    "princeton": "gpt-4o",
    "anthropic": "claude-sonnet-4-20250514",
}


def make_call_fn(client):
    """Wrap call_agent into the (system, user) -> str interface."""

    def _call(system_prompt, user_message):
        return call_agent(client, system_prompt, user_message)

    return _call


def param_distance(params_a, params_b):
    """Mean relative distance between two param dicts (shared keys only)."""
    shared = set(params_a.keys()) & set(params_b.keys())
    if not shared:
        return float("inf")
    dists = []
    for k in shared:
        a, b = float(params_a[k]), float(params_b[k])
        if b != 0:
            dists.append(abs(a - b) / abs(b))
        else:
            dists.append(abs(a))
    return float(np.mean(dists))


def run_condition(
    gt_model,
    enable_debate,
    client=None,
    n_cycles=5,
    learning_rate=0.01,
    enable_arbiter=False,
    verbose=True,
):
    """Run one condition (no_debate, debate, or arbiter) under misspecification."""
    if enable_arbiter:
        condition = "arbiter"
    elif enable_debate:
        condition = "debate"
    else:
        condition = "no_debate"

    print(f"\n{'=' * 70}")
    print(f"DECISION M15 [{condition.upper()}] — Ground Truth: {gt_model}")
    print(f"{'=' * 70}")

    # Build misspecified agent params
    agent_params = {
        f"{m}_Agent": dict(MISSPEC_DECISION_PARAMS[m]) for m in ["CPT", "EU", "PH"]
    }

    # Record initial params for recovery analysis
    target_agent = DECISION_EXPECTED_WINNER[gt_model]
    model_name = DECISION_AGENT_MAP[target_agent]
    initial_params = dict(agent_params[target_agent])
    gt_params = GT_DECISION_PARAMS[model_name]

    print(f"  Target agent: {target_agent}")
    print(f"  Misspec params: {initial_params}")
    print(f"  GT params:      {gt_params}")
    print(f"  Initial distance to GT: {param_distance(initial_params, gt_params):.4f}")
    print()

    call_fn = make_call_fn(client) if client else None

    start = time.time()
    result = run_decision_debate(
        gt_model=gt_model,
        n_cycles=n_cycles,
        learning_rate=learning_rate,
        selection_strategy="thompson",
        agent_params=agent_params,
        call_fn=call_fn,
        enable_debate=enable_debate,
        enable_arbiter=enable_arbiter,
        verbose=verbose,
    )
    elapsed = time.time() - start

    # Compute recovery percentage
    final_params = result["param_recovery"][target_agent]["final_params"]
    initial_dist = param_distance(initial_params, gt_params)
    final_dist = param_distance(final_params, gt_params)
    recovery_pct = (
        (initial_dist - final_dist) / initial_dist * 100 if initial_dist > 0 else 0
    )

    result["elapsed_seconds"] = elapsed
    result["initial_params"] = initial_params
    result["final_params"] = final_params
    result["initial_distance"] = initial_dist
    result["final_distance"] = final_dist
    result["recovery_pct"] = recovery_pct

    # Print results
    print(f"\n{'=' * 70}")
    print(f"RESULTS [{condition.upper()}] — {gt_model} ({elapsed:.0f}s)")
    print(f"{'=' * 70}")
    print(f"  Winner: {result['winner']} (expected: {result['expected']})")
    print(f"  Correct: {result['correct']}")
    print(f"  Final posterior: {result['final_posterior']}")
    print(f"  Final entropy: {result['final_entropy']:.4f}")

    if enable_debate:
        print(f"\n  Revisions proposed: {result['n_revisions_proposed']}")
        print(f"  Revisions accepted: {result['n_revisions_accepted']}")
        print(f"\n  Parameter Recovery ({target_agent}):")
        print(f"    Initial: {initial_params}")
        print(f"    Final:   {final_params}")
        print(f"    GT:      {gt_params}")
        print(f"    Distance: {initial_dist:.4f} → {final_dist:.4f}")
        print(f"    Recovery: {recovery_pct:.1f}%")

    return result


def print_summary_table(results):
    """Print a comparison table across all runs."""
    print(f"\n\n{'=' * 90}")
    print("DECISION-DOMAIN M15 — MISSPECIFICATION + DEBATE")
    print(f"{'=' * 90}")
    print(
        f"{'GT':<6} {'Condition':<12} {'Winner':<12} {'OK?':<5} "
        f"{'Entropy':<8} {'Recovery':<10} {'Rev Prop':<10} {'Rev Acc':<10}"
    )
    print("-" * 90)

    for key, r in sorted(results.items()):
        if "error" in r:
            print(f"{key}: ERROR — {r['error'][:50]}")
            continue

        gt = r["ground_truth"]
        cond = r["condition"]
        winner = r["winner"].replace("_Agent", "")
        correct = "Yes" if r["correct"] else "**No**"
        entropy = f"{r['final_entropy']:.3f}"
        recovery = f"{r.get('recovery_pct', 0):.1f}%"
        n_prop = str(r.get("n_revisions_proposed", 0))
        n_acc = str(r.get("n_revisions_accepted", 0))

        print(
            f"{gt:<6} {cond:<12} {winner:<12} {correct:<5} "
            f"{entropy:<8} {recovery:<10} {n_prop:<10} {n_acc:<10}"
        )

    n_correct = sum(
        1 for r in results.values() if isinstance(r, dict) and r.get("correct", False)
    )
    total = sum(1 for r in results.values() if isinstance(r, dict) and "correct" in r)
    print(f"\nCorrect: {n_correct}/{total}")

    # Debate/arbiter advantage
    print("\n### Condition Comparison (vs no_debate)")
    for gt in ["CPT", "EU", "PH"]:
        nd_key = f"{gt}_no_debate"
        d_key = f"{gt}_debate"
        a_key = f"{gt}_arbiter"
        parts = []
        if nd_key in results:
            nd = results[nd_key]
            nd_ok = "correct" if nd.get("correct") else "WRONG"
            parts.append(f"no_debate={nd_ok}")
        if d_key in results:
            d = results[d_key]
            d_ok = "correct" if d.get("correct") else "WRONG"
            recovery = d.get("recovery_pct", 0)
            parts.append(f"debate={d_ok} ({recovery:.1f}% recovery)")
        if a_key in results:
            a = results[a_key]
            a_ok = "correct" if a.get("correct") else "WRONG"
            a_recovery = a.get("recovery_pct", 0)
            parts.append(f"arbiter={a_ok} ({a_recovery:.1f}% recovery)")
        if parts:
            print(f"  {gt}: {', '.join(parts)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Decision-domain M15: debate under misspecification"
    )
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
        help="Number of cycles (default: 5)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Likelihood tempering rate (default: 0.01)",
    )
    parser.add_argument(
        "--no-debate-only",
        action="store_true",
        help="Run only no-debate conditions",
    )
    parser.add_argument(
        "--debate-only",
        action="store_true",
        help="Run only debate conditions",
    )
    parser.add_argument(
        "--arbiter",
        action="store_true",
        help="Also run arbiter conditions (debate + crux + meta-agents)",
    )
    parser.add_argument(
        "--arbiter-only",
        action="store_true",
        help="Run only arbiter conditions",
    )
    parser.add_argument(
        "--backend",
        choices=["anthropic", "princeton"],
        default="princeton",
        help="LLM backend (default: princeton)",
    )
    args = parser.parse_args()

    models = [args.model] if args.model else ["CPT", "EU", "PH"]

    # Determine which conditions to run
    if args.arbiter_only:
        run_no_debate = False
        run_debate = False
        run_arbiter = True
    else:
        run_no_debate = not args.debate_only
        run_debate = not args.no_debate_only
        run_arbiter = args.arbiter

    # Create LLM client if running debate or arbiter conditions
    client = None
    if run_debate or run_arbiter:
        # Set the model name for the chosen backend — runner.call_agent() uses
        # this global when no model= kwarg is passed.
        runner_mod._LLM_MODEL = BACKEND_MODELS[args.backend]
        try:
            client = _create_client(args.backend)
            print(
                f"LLM client created (backend: {args.backend}, "
                f"model: {runner_mod._LLM_MODEL})"
            )
        except SystemExit:
            print("ERROR: Could not create LLM client. Check API keys.")
            print("Run with --no-debate-only to skip debate conditions.")
            sys.exit(1)

    output_dir = "runs/decision_m15"
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    total_start = time.time()

    for gt_model in models:
        if run_no_debate:
            key = f"{gt_model}_no_debate"
            try:
                results[key] = run_condition(
                    gt_model,
                    enable_debate=False,
                    n_cycles=args.n_cycles,
                    learning_rate=args.learning_rate,
                )
            except Exception as e:
                import traceback

                traceback.print_exc()
                results[key] = {"error": str(e)}

        if run_debate:
            key = f"{gt_model}_debate"
            try:
                results[key] = run_condition(
                    gt_model,
                    enable_debate=True,
                    client=client,
                    n_cycles=args.n_cycles,
                    learning_rate=args.learning_rate,
                )
            except Exception as e:
                import traceback

                traceback.print_exc()
                results[key] = {"error": str(e)}

        if run_arbiter:
            key = f"{gt_model}_arbiter"
            try:
                results[key] = run_condition(
                    gt_model,
                    enable_debate=True,
                    enable_arbiter=True,
                    client=client,
                    n_cycles=args.n_cycles,
                    learning_rate=args.learning_rate,
                )
            except Exception as e:
                import traceback

                traceback.print_exc()
                results[key] = {"error": str(e)}

    total_elapsed = time.time() - total_start
    print_summary_table(results)
    print(f"\nTotal time: {total_elapsed:.0f}s")

    # Save results
    results_path = os.path.join(output_dir, "decision_m15_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {results_path}")
