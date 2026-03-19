"""
R-IDeA + Debate Validation — Under Misspecification.

Tests whether R-IDeA experiment selection + debate parameter recovery
outperforms either alone. Monkeypatches the selection function to use
R-IDeA scoring while keeping all debate phases intact.

Conditions (3 GTs × 3 OED types = 9 runs):
  - EIG + debate (M15 replication)
  - R-IDeA + debate (new)
  - EIG + no-debate (M15 baseline replication)

Usage:
    python scripts/validation/validate_ridea_debate.py              # all 9
    python scripts/validation/validate_ridea_debate.py GCM          # one GT
    python scripts/validation/validate_ridea_debate.py --ridea-only # 3 runs
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
from antagonistic_collab.bayesian_selection import (
    ModelPosterior,
    compute_eig,
)
from antagonistic_collab.ridea import (
    compute_ridea_scores,
)
import antagonistic_collab.runner as runner_mod
import antagonistic_collab.bayesian_selection as bsel

# ── Misspecification params (from M15) ──

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

# ── R-IDeA monkeypatch ──

# Store previous predictions for representativeness across cycles
_ridea_previous_preds = []
_ridea_active = False

_original_select_from_pool = bsel.select_from_pool


def _ridea_select_from_pool(
    protocol, posterior, pool, n_subjects=20, n_sim=200, seed=42,
    focus_pair=None, pair_boost=1.5, crux_boost_specs=None,
    learning_rate=1.0, selection_strategy="thompson", crux_weight=0.0,
):
    """Drop-in replacement for select_from_pool using R-IDeA scoring."""
    global _ridea_previous_preds

    # Extract predictions for all candidates (same as original)
    pool_preds = []
    for i, (struct_name, condition) in enumerate(pool):
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

    # Score with R-IDeA
    scores = compute_ridea_scores(
        pool_preds, posterior, _ridea_previous_preds,
        alpha=0.3, beta=0.3,
        n_subjects=n_subjects, n_sim=n_sim,
        seed=seed, learning_rate=learning_rate,
    )

    # Thompson sampling
    rng = np.random.default_rng(seed)
    arr = np.array(scores, dtype=np.float64)
    total = arr.sum()
    if total > 0:
        weights = arr / total
        best_idx = int(rng.choice(len(scores), p=weights))
    else:
        best_idx = int(rng.integers(len(scores)))

    # Track for representativeness in future cycles
    _ridea_previous_preds.append(pool_preds[best_idx])

    return best_idx, scores


def _activate_ridea():
    """Monkeypatch select_from_pool to use R-IDeA."""
    global _ridea_active, _ridea_previous_preds
    _ridea_previous_preds = []
    _ridea_active = True
    bsel.select_from_pool = _ridea_select_from_pool


def _deactivate_ridea():
    """Restore original select_from_pool."""
    global _ridea_active, _ridea_previous_preds
    _ridea_previous_preds = []
    _ridea_active = False
    bsel.select_from_pool = _original_select_from_pool


# ── Run conditions ──


def run_condition(
    client, true_model: str, debate: bool, use_ridea: bool = False,
    n_cycles: int = 5,
):
    """Run one condition with misspecified params."""
    if not debate:
        cond_label = "eig_no_debate"
    elif use_ridea:
        cond_label = "ridea_debate"
    else:
        cond_label = "eig_debate"

    tag = f"ridea_cmp_{cond_label}_{true_model}"
    output_dir = f"runs/{tag}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"R-IDeA+DEBATE [{cond_label.upper()}] — GT: {true_model}")
    print(f"{'=' * 70}")
    print(f"  Debate: {debate}, R-IDeA: {use_ridea}")
    print(f"  Misspec params: {MISSPEC_PARAMS[true_model]}")
    print()

    # Fresh state and agents
    state = EpistemicState(domain="Human Categorization")
    agents = default_agent_configs()

    # Patch with wrong params
    target_agent = EXPECTED_WINNER[true_model]
    for agent in agents:
        if agent.name == target_agent:
            agent.default_params = copy.deepcopy(MISSPEC_PARAMS[true_model])
            print(f"  Misspecified {agent.name}: {agent.default_params}")

    initial_params = None
    for agent in agents:
        if agent.name == target_agent:
            initial_params = copy.deepcopy(agent.default_params)

    # Set modes
    runner_mod._NO_DEBATE = not debate
    runner_mod._ARBITER = False
    runner_mod._DESIGN_SPACE = "continuous"
    runner_mod._N_CONTINUOUS_SAMPLES = 50
    runner_mod._CRUX_WEIGHT = 0.0

    protocol = DebateProtocol(state, agents, meta_agents=[])
    transcript = []

    metadata = {
        "true_model": true_model,
        "llm_model": "gpt-4o",
        "backend": "princeton",
        "milestone": "R-IDeA",
        "condition": cond_label,
        "misspec_params": MISSPEC_PARAMS[true_model],
        "gt_params": GT_PARAMS[true_model],
    }

    # Activate R-IDeA if requested
    if use_ridea:
        _activate_ridea()
    else:
        _deactivate_ridea()

    param_history = []
    start_time = time.time()

    for cycle in range(n_cycles):
        cycle_start = time.time()
        print(f"\n{'=' * 50}")
        print(f"  Cycle {cycle} ({time.time() - start_time:.0f}s elapsed)")
        print(f"{'=' * 50}")

        # Snapshot params
        pre_params = {}
        for agent in agents:
            pre_params[agent.name] = copy.deepcopy(agent.default_params)

        run_cycle(
            protocol, client, transcript,
            true_model=true_model, critique_rounds=1,
            output_dir=output_dir, metadata=metadata,
            mode="full_pool",
        )

        # Detect revisions
        for agent in agents:
            if agent.default_params != pre_params[agent.name]:
                print(f"  >> {agent.name} params revised")
                param_history.append(agent.name)

        print(f"  Cycle {cycle} completed in {time.time() - cycle_start:.0f}s")

    # Deactivate monkeypatch
    _deactivate_ridea()

    total_time = time.time() - start_time

    # ── Analysis ──
    print(f"\n\n{'=' * 70}")
    print(f"RESULTS [{cond_label.upper()}] — {true_model} ({total_time:.0f}s)")
    print(f"{'=' * 70}")

    board = state.prediction_leaderboard()
    winner = None
    winner_rmse = 999
    gap_pct = 0
    if board:
        sorted_board = sorted(board.items(), key=lambda x: x[1].get("mean_score", 999))
        for rank, (agent, stats) in enumerate(sorted_board):
            mean = stats.get("mean_score")
            marker = " <-- WINNER" if rank == 0 else ""
            if mean is not None:
                print(f"  {rank+1}. {agent}: RMSE={mean:.4f}{marker}")

        winner = sorted_board[0][0]
        winner_rmse = sorted_board[0][1].get("mean_score", 999)
        if len(sorted_board) > 1:
            runner_up_rmse = sorted_board[1][1].get("mean_score", 999)
            gap_pct = (
                ((runner_up_rmse - winner_rmse) / runner_up_rmse * 100)
                if runner_up_rmse > 0 else 0
            )

    # Param recovery
    final_params = None
    for agent in agents:
        if agent.name == target_agent:
            final_params = copy.deepcopy(agent.default_params)

    gt = GT_PARAMS[true_model]
    initial_dist = param_distance(initial_params, gt) if initial_params else 0
    final_dist = param_distance(final_params, gt) if final_params else 0
    recovery_pct = (
        ((initial_dist - final_dist) / initial_dist * 100)
        if initial_dist > 0 else 0
    )

    print(f"\n  Winner: {winner} (gap: {gap_pct:.1f}%)")
    print(f"  Correct: {winner == EXPECTED_WINNER.get(true_model)}")
    print(f"  Param recovery: {recovery_pct:.1f}%")

    analysis = {
        "ground_truth": true_model,
        "condition": cond_label,
        "debate": debate,
        "use_ridea": use_ridea,
        "n_cycles": n_cycles,
        "total_time_s": total_time,
        "winner": winner,
        "correct": winner == EXPECTED_WINNER.get(true_model),
        "winner_rmse": float(winner_rmse) if winner_rmse < 999 else None,
        "gap_pct": float(gap_pct),
        "recovery_pct": float(recovery_pct),
    }

    analysis_path = os.path.join(output_dir, f"{tag}_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    return analysis


def setup_client():
    """Create Princeton client."""
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
        print("ERROR: AI_SANDBOX_KEY not set.")
        sys.exit(1)

    return _create_client(backend="princeton")


if __name__ == "__main__":
    runner_mod._LLM_MODEL = "gpt-4o"
    runner_mod._BATCH_MODE = True
    runner_mod._SELECTION_METHOD = "bayesian"
    runner_mod._SELECTION_STRATEGY = "thompson"
    runner_mod._LEARNING_RATE = 0.005
    runner_mod._CLAIM_RESPONSIVE = True
    runner_mod._NORMALIZE_CLAIMS = True
    runner_mod._FUZZY_STRUCTURE_MATCH = True

    models = ["GCM", "SUSTAIN", "RULEX"]
    ridea_only = False

    for arg in sys.argv[1:]:
        if arg in models:
            models = [arg]
        elif arg == "--ridea-only":
            ridea_only = True

    # Conditions: (label, debate, use_ridea)
    if ridea_only:
        conditions = [("ridea_debate", True, True)]
    else:
        conditions = [
            ("eig_no_debate", False, False),
            ("eig_debate", True, False),
            ("ridea_debate", True, True),
        ]

    needs_llm = any(d for _, d, _ in conditions)
    client = setup_client() if needs_llm else None

    results = {}
    for true_model in models:
        for cond_label, debate_flag, ridea_flag in conditions:
            key = f"{true_model}_{cond_label}"
            try:
                results[key] = run_condition(
                    client, true_model,
                    debate=debate_flag, use_ridea=ridea_flag,
                    n_cycles=5,
                )
            except Exception as e:
                print(f"\nERROR running {key}: {e}")
                import traceback
                traceback.print_exc()
                results[key] = {"error": str(e)}

    # Comparison table
    print(f"\n\n{'=' * 90}")
    print("EIG vs R-IDeA + DEBATE COMPARISON (under misspecification)")
    print(f"{'=' * 90}")
    print(f"{'GT':<10} {'Condition':<18} {'Winner':<20} {'OK?':<5} {'Gap%':<8} {'Recovery%':<10}")
    print("-" * 90)
    for model in ["GCM", "SUSTAIN", "RULEX"]:
        for cond in ["eig_no_debate", "eig_debate", "ridea_debate"]:
            key = f"{model}_{cond}"
            r = results.get(key, {})
            if not r or "error" in r:
                err = r.get("error", "not run")[:40] if r else "not run"
                print(f"{model:<10} {cond:<18} ERROR: {err}")
                continue
            w = r.get("winner", "?")
            ok = "Yes" if r.get("correct") else "No"
            gap = r.get("gap_pct", 0)
            rec = r.get("recovery_pct", 0)
            print(f"{model:<10} {cond:<18} {w:<20} {ok:<5} {gap:<8.1f} {rec:<10.1f}")
        print()

    # Save
    combined_path = "runs/ridea_debate_comparison.json"
    os.makedirs("runs", exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {combined_path}")
