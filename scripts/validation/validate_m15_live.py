"""
M15 Live Validation — Phase 2: Three-way comparison under Misspecification.

Tests whether LLM debate recovers from deliberate parameter misspecification.
All agents start with gap-calibrated wrong params (from Phase 1b sweep).
Ground truth uses correct params in _synthetic_runner().

Three conditions per ground truth:
  - No-debate: params stay fixed, computational pipeline only (_NO_DEBATE=True)
  - Debate (no arbiter): agents debate + propose param revisions, but no
    cruxes, meta-agents, or claim-directed selection (_ARBITER=False)
  - Debate (with arbiter): full debate + cruxes + meta-agents + claim-directed
    selection + claim auto-resolution (_ARBITER=True)

Measures: RMSE, gap, param revisions accepted/rejected, cycles to recovery.

Usage:
    python scripts/validation/validate_m15_live.py                  # all 9 runs
    python scripts/validation/validate_m15_live.py GCM              # one GT, 3 conditions
    python scripts/validation/validate_m15_live.py --no-debate-only # 3 runs
    python scripts/validation/validate_m15_live.py --debate-only    # 3 runs (no arbiter)
    python scripts/validation/validate_m15_live.py --arbiter-only   # 3 runs (with arbiter)
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


# ── Calibrated misspecification settings (from Phase 1b sweep) ──────────

# Ground-truth params (used by _synthetic_runner)
GT_PARAMS = {
    "GCM": {"c": 4.0, "r": 1, "gamma": 1.0},
    "SUSTAIN": {"r": 9.01, "beta": 1.252, "d": 16.924, "eta": 0.092},
    "RULEX": {"p_single": 0.5, "p_conj": 0.3, "error_tolerance": 0.1},
}

# Misspecified params — narrowest-gap settings from Phase 1b
MISSPEC_PARAMS = {
    "GCM": {"c": 0.5, "r": 1, "gamma": 1.0},           # gap: 61% → 28%
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
    """Overwrite default_params for the misspecified agent only.

    The correct model's agent gets wrong params. Competitors keep defaults.
    """
    target_agent = EXPECTED_WINNER[misspec_model]
    for agent in agents:
        if agent.name == target_agent:
            agent.default_params = copy.deepcopy(MISSPEC_PARAMS[misspec_model])
            print(f"  Misspecified {agent.name}: {agent.default_params}")
        else:
            model_name = AGENT_MODEL_MAP[agent.name]
            print(f"  {agent.name} (competitor): {agent.default_params}")


def run_condition(client, true_model: str, debate: bool, arbiter: bool = True,
                  n_cycles: int = 5):
    """Run one condition with misspecified params.

    Args:
        debate: If False, skip all LLM phases (computational pipeline only).
        arbiter: If True (and debate=True), enable cruxes, meta-agents, and
                 claim-directed selection. Ignored when debate=False.
    """
    if not debate:
        condition_label = "no_debate"
    elif arbiter:
        condition_label = "arbiter"
    else:
        condition_label = "debate"

    tag = f"m15_{condition_label}_{true_model}"
    output_dir = f"runs/{tag}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"M15 [{condition_label.upper()}] — Ground Truth: {true_model}")
    print(f"{'=' * 70}")
    print(f"  Cycles: {n_cycles}")
    print(f"  Misspecified model: {EXPECTED_WINNER[true_model]}")
    print(f"  Misspec params: {MISSPEC_PARAMS[true_model]}")
    print(f"  GT params: {GT_PARAMS[true_model]}")
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

    # Set debate/arbiter mode
    runner_mod._NO_DEBATE = not debate
    runner_mod._ARBITER = arbiter and debate

    # Only create meta-agents when arbiter is active
    meta = create_default_meta_agents() if (arbiter and debate) else []
    protocol = DebateProtocol(state, agents, meta_agents=meta)
    transcript = []

    metadata = {
        "true_model": true_model,
        "llm_model": "gpt-4o",
        "backend": "princeton",
        "milestone": "M15",
        "condition": condition_label,
        "arbiter": arbiter and debate,
        "misspec_params": MISSPEC_PARAMS[true_model],
        "gt_params": GT_PARAMS[true_model],
    }

    # Track param revisions per cycle
    param_history = []  # list of dicts per cycle
    start_time = time.time()

    for cycle in range(n_cycles):
        cycle_start = time.time()
        print(f"\n{'=' * 50}")
        print(f"  Cycle {cycle} ({time.time() - start_time:.0f}s elapsed)")
        print(f"{'=' * 50}")

        # Snapshot params before cycle
        pre_params = {}
        for agent in agents:
            pre_params[agent.name] = copy.deepcopy(agent.default_params)

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

        # Snapshot params after cycle, detect revisions
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

        elapsed = time.time() - cycle_start
        print(f"\n  Cycle {cycle} completed in {elapsed:.0f}s")

    total_time = time.time() - start_time

    # ── Analysis ──

    print(f"\n\n{'=' * 70}")
    print(f"M15 RESULTS [{condition_label.upper()}] — {true_model} ({total_time:.0f}s)")
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

    # Parameter recovery analysis
    print(f"\n### Parameter Recovery")
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
        initial_dist = _param_distance(initial_params, gt)
        final_dist = _param_distance(final_params, gt)
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
        print(f"\n### Claim Ledger")
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
        print(f"\n### Cruxes")
        print(f"  Total: {n_cruxes}, Accepted: {n_accepted_cruxes}")

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

    # Experiments
    print(f"\n### Experiments")
    selected_structs = []
    for exp in state.experiments:
        ds = exp.design_spec or {}
        s = ds.get("structure_name", "?")
        c = ds.get("condition", "?")
        selected_structs.append(s)
        print(f"  Cycle {exp.cycle}: {s} / {c}")

    # Save analysis JSON
    analysis = {
        "ground_truth": true_model,
        "milestone": "M15",
        "condition": condition_label,
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


def _param_distance(params_a: dict, params_b: dict) -> float:
    """Normalized Euclidean distance between two param dicts.

    Only compares shared keys. Normalizes each dimension by the GT value
    to make distances comparable across params with different scales.
    """
    shared = set(params_a.keys()) & set(params_b.keys())
    if not shared:
        return float("inf")
    sq_diffs = []
    for k in shared:
        a, b = float(params_a[k]), float(params_b[k])
        # Normalize by GT magnitude (avoid div-by-zero)
        scale = max(abs(b), 1e-6)
        sq_diffs.append(((a - b) / scale) ** 2)
    return (sum(sq_diffs) / len(sq_diffs)) ** 0.5


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
    """Set shared globals for M15 validation."""
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
    runner_mod._CRUX_WEIGHT = 0.3


def print_comparison_table(results, conditions_run):
    """Print three-way comparison across ground truths."""
    cond_labels = [c[0] for c in conditions_run]

    print(f"\n\n{'=' * 100}")
    print("M15 COMPARISON: THREE CONDITIONS UNDER MISSPECIFICATION")
    print(f"{'=' * 100}")

    header = (
        f"{'GT':<10} {'Condition':<14} {'Winner':<20} {'OK?':<5} "
        f"{'RMSE':<8} {'Gap%':<8} {'Revisions':<10} {'Recovery%':<10} {'Cruxes':<8}"
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
                print(f"{model:<10} {cond:<14} ERROR: {r['error'][:40]}")
                continue
            w = r.get("winner", "?")
            correct = "Yes" if r.get("correct") else "No"
            rmse = r.get("winner_rmse")
            rmse_str = f"{rmse:.3f}" if rmse else "?"
            gap = r.get("gap_pct", 0)
            revs = r.get("n_param_revisions", 0)
            recov = r.get("recovery_pct", 0)
            cruxes = r.get("n_accepted_cruxes", 0)
            print(
                f"{model:<10} {cond:<14} {w:<20} {correct:<5} "
                f"{rmse_str:<8} {gap:<8.1f} {revs:<10} {recov:<10.1f} {cruxes:<8}"
            )
        print()  # blank line between GTs

    # Summary: pairwise advantages
    print("\n### Gap Advantage (pp over no-debate baseline)")
    for model in ["GCM", "SUSTAIN", "RULEX"]:
        nd = results.get(f"{model}_no_debate", {})
        nd_gap = nd.get("gap_pct", 0)
        parts = [f"no_debate={nd_gap:.1f}%"]
        for cond in ["debate", "arbiter"]:
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
    #   (label, debate, arbiter)
    ALL_CONDITIONS = [
        ("no_debate", False, False),
        ("debate",    True,  False),
        ("arbiter",   True,  True),
    ]
    selected_conditions = None  # None = all

    # Parse CLI args
    for arg in sys.argv[1:]:
        if arg in models:
            models = [arg]
        elif arg == "--no-debate-only":
            selected_conditions = [("no_debate", False, False)]
        elif arg == "--debate-only":
            selected_conditions = [("debate", True, False)]
        elif arg == "--arbiter-only":
            selected_conditions = [("arbiter", True, True)]

    conditions = selected_conditions or ALL_CONDITIONS

    # Only create LLM client if a debate condition is requested
    needs_llm = any(debate for _, debate, _ in conditions)
    client = setup_client() if needs_llm else None

    results = {}
    for true_model in models:
        for cond_label, debate_flag, arbiter_flag in conditions:
            key = f"{true_model}_{cond_label}"
            try:
                results[key] = run_condition(
                    client, true_model,
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
    combined_path = "runs/m15_phase2_summary.json"
    os.makedirs("runs", exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to: {combined_path}")
