"""
M17 Live Validation — Misspecification + Open Design Space.

Combines M15 (parameter misspecification) with M16 (open design space).
All agents start with gap-calibrated wrong params AND must propose all
experiment structures via debate (no curated registry).

Scientific question: Do debate's parameter recovery (M15's win) and
structure proposal (M16's RULEX win) compose or interfere?

Five conditions per ground truth (matching M16 factorial):
  - closed_no_debate: misspecified + curated registry, no debate
  - closed_debate: misspecified + curated registry, debate
  - closed_arbiter: misspecified + curated registry, debate + arbiter
  - open_debate: misspecified + agent-proposed, debate
  - open_arbiter: misspecified + agent-proposed, debate + arbiter

The first three replicate M15 conditions. The last two are new.

Usage:
    python scripts/validation/validate_m17_live.py                  # all 15 runs
    python scripts/validation/validate_m17_live.py GCM              # one GT, 5 conditions
    python scripts/validation/validate_m17_live.py --open-only      # 6 runs (new conditions)
    python scripts/validation/validate_m17_live.py --closed-only    # 3 runs (no-debate baseline)
    python scripts/validation/validate_m17_live.py --debate-only    # 3 runs (closed debate)
    python scripts/validation/validate_m17_live.py --arbiter-only   # 6 runs (both arbiter)
    python scripts/validation/validate_m17_live.py --new-only       # 6 runs (open conditions only)
"""

import copy
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
    create_default_meta_agents,
    param_distance,
)
import antagonistic_collab.runner as runner_mod


# ── Calibrated misspecification settings (from M15 Phase 1b sweep) ──────

# Ground-truth params (used by _synthetic_runner)
GT_PARAMS = {
    "GCM": {"c": 4.0, "r": 1, "gamma": 1.0},
    "SUSTAIN": {"r": 9.01, "beta": 1.252, "d": 16.924, "eta": 0.092},
    "RULEX": {"p_single": 0.5, "p_conj": 0.3, "error_tolerance": 0.1},
}

# Misspecified params — narrowest-gap settings from Phase 1b
MISSPEC_PARAMS = {
    "GCM": {"c": 0.5, "r": 1, "gamma": 1.0},  # gap: 61% → 28%
    "SUSTAIN": {"r": 3.0, "beta": 1.252, "d": 16.924, "eta": 0.15},  # gap: 65% → 29%
    "RULEX": {"p_single": 0.3, "p_conj": 0.3, "error_tolerance": 0.25},  # gap: 82% → 16%
}

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


def patch_agent_params(agents, misspec_model: str):
    """Overwrite default_params for the misspecified agent only."""
    target_agent = EXPECTED_WINNER[misspec_model]
    for agent in agents:
        if agent.name == target_agent:
            agent.default_params = copy.deepcopy(MISSPEC_PARAMS[misspec_model])
            print(f"  Misspecified {agent.name}: {agent.default_params}")
        else:
            print(f"  {agent.name} (competitor): {agent.default_params}")


def run_condition(
    client,
    true_model: str,
    design_space: str,
    debate: bool,
    arbiter: bool = False,
    n_cycles: int = 5,
):
    """Run one condition with misspecified params.

    Args:
        design_space: "continuous" (registry) or "open" (agent-proposed only).
        debate: If False, skip all LLM phases (computational pipeline only).
        arbiter: If True, enable cruxes, meta-agents, and claim-directed selection.
    """
    if not debate:
        condition_label = "closed_no_debate"
    elif design_space == "open":
        condition_label = "open_arbiter" if arbiter else "open_debate"
    else:
        condition_label = "closed_arbiter" if arbiter else "closed_debate"

    tag = f"m17_{condition_label}_{true_model}"
    output_dir = f"runs/{tag}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"M17 [{condition_label.upper()}] — Ground Truth: {true_model}")
    print(f"{'=' * 70}")
    print(f"  Cycles: {n_cycles}")
    print(f"  Misspecified model: {EXPECTED_WINNER[true_model]}")
    print(f"  Misspec params: {MISSPEC_PARAMS[true_model]}")
    print(f"  GT params: {GT_PARAMS[true_model]}")
    print(f"  Design space: {design_space}")
    print(f"  _NO_DEBATE: {not debate}")
    print(f"  _ARBITER: {arbiter and debate}")
    print()

    # Fresh state and agents for each condition
    state = EpistemicState(domain="Human Categorization")
    agents = default_agent_configs()

    # Patch the correct model's agent with wrong params
    print("  Agent params:")
    patch_agent_params(agents, true_model)
    print()

    # Record initial params for tracking recovery
    target_agent_name = EXPECTED_WINNER[true_model]
    initial_params = None
    for agent in agents:
        if agent.name == target_agent_name:
            initial_params = copy.deepcopy(agent.default_params)

    # Set modes
    runner_mod._NO_DEBATE = not debate
    runner_mod._ARBITER = arbiter and debate
    runner_mod._DESIGN_SPACE = design_space
    # Restore crux weight when arbiter is active
    runner_mod._CRUX_WEIGHT = 0.3 if (arbiter and debate) else 0.0

    # Only create meta-agents when arbiter is active
    meta = create_default_meta_agents() if (arbiter and debate) else []
    protocol = DebateProtocol(state, agents, meta_agents=meta)
    transcript = []

    metadata = {
        "true_model": true_model,
        "llm_model": "gpt-4o",
        "backend": "princeton",
        "milestone": "M17",
        "condition": condition_label,
        "design_space": design_space,
        "arbiter": arbiter and debate,
        "misspec_params": MISSPEC_PARAMS[true_model],
        "gt_params": GT_PARAMS[true_model],
    }

    # Track param revisions and structure proposals per cycle
    param_history = []
    structures_per_cycle = []
    start_time = time.time()

    for cycle in range(n_cycles):
        cycle_start = time.time()
        print(f"\n{'=' * 50}")
        print(f"  Cycle {cycle} ({time.time() - start_time:.0f}s elapsed)")
        print(f"{'=' * 50}")

        # Snapshot params and structures before cycle
        pre_params = {}
        for agent in agents:
            pre_params[agent.name] = copy.deepcopy(agent.default_params)
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

        # Detect param revisions
        cycle_revisions = {}
        for agent in agents:
            post = agent.default_params
            pre = pre_params[agent.name]
            changed = post != pre
            cycle_revisions[agent.name] = {
                "pre": pre,
                "post": copy.deepcopy(post),
                "changed": changed,
            }
            if changed:
                print(f"  >> {agent.name} params revised: {pre} → {post}")
        param_history.append(cycle_revisions)

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
    print(f"M17 RESULTS [{condition_label.upper()}] — {true_model} ({total_time:.0f}s)")
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

    # Parameter recovery analysis
    print("\n### Parameter Recovery")
    target = EXPECTED_WINNER[true_model]
    final_params = None
    for agent in agents:
        if agent.name == target:
            final_params = copy.deepcopy(agent.default_params)

    n_revisions = sum(
        1 for h in param_history if h.get(target, {}).get("changed", False)
    )
    print(f"  Initial params: {initial_params}")
    print(f"  Final params:   {final_params}")
    print(f"  GT params:      {GT_PARAMS[true_model]}")
    print(f"  Cycles with revision: {n_revisions}/{n_cycles}")

    # Compute param distance to GT
    gt = GT_PARAMS[true_model]
    if initial_params and final_params:
        initial_dist = param_distance(initial_params, gt)
        final_dist = param_distance(final_params, gt)
        print(f"  Distance to GT: {initial_dist:.4f} (initial) → {final_dist:.4f} (final)")
        recovery_pct = (
            ((initial_dist - final_dist) / initial_dist * 100)
            if initial_dist > 0
            else 0
        )
        print(f"  Recovery: {recovery_pct:.1f}%")
    else:
        initial_dist = final_dist = recovery_pct = 0

    # Claim ledger (debate conditions only)
    ledger = state.claim_ledger
    n_claims = len(ledger)
    n_confirmed = sum(1 for c in ledger if c.status == "confirmed")
    n_falsified = sum(1 for c in ledger if c.status == "falsified")

    if debate:
        print("\n### Claim Ledger")
        print(f"  Total: {n_claims}")
        print(f"  Confirmed: {n_confirmed}, Falsified: {n_falsified}")

    # Crux analysis (arbiter condition only)
    n_cruxes = len(state.cruxes) if hasattr(state, "cruxes") else 0
    n_accepted_cruxes = (
        sum(1 for c in state.cruxes if c.status == "accepted")
        if n_cruxes > 0
        else 0
    )
    if arbiter and debate:
        print("\n### Cruxes")
        print(f"  Total: {n_cruxes}, Accepted: {n_accepted_cruxes}")

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
        "milestone": "M17",
        "condition": condition_label,
        "design_space": design_space,
        "arbiter": arbiter and debate,
        "n_cycles": n_cycles,
        "total_time_s": total_time,
        "winner": winner,
        "correct": winner == EXPECTED_WINNER.get(true_model),
        "winner_rmse": float(winner_rmse) if winner_rmse < 999 else None,
        "gap_pct": float(gap_pct),
        "misspec_params": MISSPEC_PARAMS[true_model],
        "gt_params": GT_PARAMS[true_model],
        "initial_params": initial_params,
        "final_params": final_params,
        "initial_dist_to_gt": float(initial_dist),
        "final_dist_to_gt": float(final_dist),
        "recovery_pct": float(recovery_pct),
        "n_param_revisions": n_revisions,
        "n_claims": n_claims,
        "n_confirmed": n_confirmed,
        "n_falsified": n_falsified,
        "n_cruxes": n_cruxes,
        "n_accepted_cruxes": n_accepted_cruxes,
        "selected_structures": selected_structs,
        "structures_per_cycle": structures_per_cycle,
        "total_proposed_structures": len(getattr(protocol, "temporary_structures", {})),
        "param_history": [
            {agent: {"changed": v["changed"]} for agent, v in h.items()}
            for h in param_history
        ],
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
        for base in [
            os.path.dirname(__file__),
            os.path.join(os.path.dirname(__file__), "..", ".."),
        ]:
            env_path = os.path.join(base, ".env")
            if os.path.exists(env_path):
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, _, val = line.partition("=")
                            os.environ[key.strip()] = val.strip()
                break

    if not os.environ.get("AI_SANDBOX_KEY"):
        print("ERROR: AI_SANDBOX_KEY environment variable is not set.")
        print("Set it before running: export AI_SANDBOX_KEY=your_key_here")
        sys.exit(1)

    return _create_client(backend="princeton")


def set_common_globals():
    """Set shared globals for M17 validation."""
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
    runner_mod._CRUX_WEIGHT = 0.0  # Overridden per-condition when arbiter=True


def print_comparison_table(results, conditions_run):
    """Print five-way comparison across ground truths."""
    cond_labels = [c[0] for c in conditions_run]

    print(f"\n\n{'=' * 110}")
    print("M17 COMPARISON: MISSPECIFICATION + OPEN DESIGN SPACE")
    print(f"{'=' * 110}")

    header = (
        f"{'GT':<10} {'Condition':<18} {'Winner':<20} {'OK?':<5} "
        f"{'RMSE':<8} {'Gap%':<8} {'#Structs':<10} {'Recovery%':<10}"
    )
    print(header)
    print("-" * 110)

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
            recov = r.get("recovery_pct", 0)
            print(
                f"{model:<10} {cond:<18} {w:<20} {correct:<5} "
                f"{rmse_str:<8} {gap:<8.1f} {str(n_structs):<10} {recov:<10.1f}"
            )
        print()

    # Summary: pairwise advantages
    all_conds = [
        "closed_debate",
        "closed_arbiter",
        "open_debate",
        "open_arbiter",
    ]
    print("\n### Gap Advantage (pp over closed_no_debate baseline)")
    for model in ["GCM", "SUSTAIN", "RULEX"]:
        nd = results.get(f"{model}_closed_no_debate", {})
        nd_gap = nd.get("gap_pct", 0)
        parts = [f"closed_no_debate={nd_gap:.1f}%"]
        for cond in all_conds:
            r = results.get(f"{model}_{cond}", {})
            if r and "error" not in r:
                g = r.get("gap_pct", 0)
                diff = g - nd_gap
                parts.append(f"{cond}={g:.1f}% ({diff:+.1f}pp)")
        print(f"  {model}: {', '.join(parts)}")

    # M15 comparison
    print("\n### M15 vs M17 comparison (misspec closed vs misspec open)")
    print("  Load M15 results from runs/m15_phase2_summary.json for comparison.")


if __name__ == "__main__":
    set_common_globals()

    models = ["GCM", "SUSTAIN", "RULEX"]

    # All five conditions by default:
    #   (label, design_space, debate, arbiter)
    ALL_CONDITIONS = [
        ("closed_no_debate", "continuous", False, False),
        ("closed_debate", "continuous", True, False),
        ("closed_arbiter", "continuous", True, True),
        ("open_debate", "open", True, False),
        ("open_arbiter", "open", True, True),
    ]
    selected_conditions = None  # None = all

    # Parse CLI args
    for arg in sys.argv[1:]:
        if arg in models:
            models = [arg]
        elif arg == "--open-only" or arg == "--new-only":
            selected_conditions = [
                ("open_debate", "open", True, False),
                ("open_arbiter", "open", True, True),
            ]
        elif arg == "--closed-only":
            selected_conditions = [("closed_no_debate", "continuous", False, False)]
        elif arg == "--debate-only":
            selected_conditions = [("closed_debate", "continuous", True, False)]
        elif arg == "--arbiter-only":
            selected_conditions = [
                ("closed_arbiter", "continuous", True, True),
                ("open_arbiter", "open", True, True),
            ]

    conditions = selected_conditions or ALL_CONDITIONS

    # Only create LLM client if a debate condition is requested
    needs_llm = any(debate for _, _, debate, _ in conditions)
    client = setup_client() if needs_llm else None

    results = {}
    for true_model in models:
        for cond_label, ds, debate_flag, arbiter_flag in conditions:
            key = f"{true_model}_{cond_label}"
            try:
                results[key] = run_condition(
                    client,
                    true_model,
                    design_space=ds,
                    debate=debate_flag,
                    arbiter=arbiter_flag,
                    n_cycles=5,
                )
            except Exception as e:
                print(f"\nERROR running {key}: {e}")
                import traceback

                traceback.print_exc()
                results[key] = {"error": str(e)}

    print_comparison_table(results, conditions)

    # Save combined results
    combined_path = "runs/m17_summary.json"
    os.makedirs("runs", exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to: {combined_path}")
