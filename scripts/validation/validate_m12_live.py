"""
M12 Continuous Design Space Live Validation — Real LLM calls.

Runs 5-cycle debates with continuous design space (M12) using GPT-4o via
Princeton AI Sandbox. Tests 3 ground truths: GCM, SUSTAIN, RULEX.

Key M12 metrics:
- Does each cycle sample different structures? (key behavioral difference from M11)
- Are sampled structures selected by EIG?
- What parameter ranges (separation, dims) does EIG prefer?
- Are results still correct with the continuous search space?
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


def run_validation(client, true_model: str, n_cycles: int = 5):
    """Run a full M12 validation with real LLM calls."""

    output_dir = f"runs/m12_val_{true_model}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"M12 CONTINUOUS DESIGN SPACE VALIDATION — Ground Truth: {true_model}")
    print(f"{'=' * 70}")
    print("  LLM: gpt-4o via Princeton AI Sandbox")
    print("  Features: all M6-M11 + continuous design space (M12)")
    print(f"  Cycles: {n_cycles}")
    print(f"  design_space: {runner_mod._DESIGN_SPACE}")
    print(f"  n_continuous_samples: {runner_mod._N_CONTINUOUS_SAMPLES}")
    n_structs = len(STRUCTURE_REGISTRY) + runner_mod._N_CONTINUOUS_SAMPLES
    n_conds = len(CONDITION_EFFECTS) + len(PARAMETRIC_CONDITIONS)
    print(
        f"  Pool size per cycle: ~{n_structs} structures x {n_conds} conditions = ~{n_structs * n_conds} candidates"
    )
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
        "milestone": "M12",
        "design_space": "continuous",
        "n_continuous_samples": runner_mod._N_CONTINUOUS_SAMPLES,
        "claim_responsive": True,
    }

    start_time = time.time()

    # Track sampled structures per cycle
    sampled_per_cycle = []

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

        # Record which structures were sampled this cycle
        sampled_names = set(protocol.sampled_structures.keys())
        sampled_per_cycle.append(sampled_names)
        print(f"  Sampled {len(sampled_names)} structures this cycle")

        elapsed = time.time() - cycle_start
        print(f"\n  Cycle {cycle} completed in {elapsed:.0f}s")

    total_time = time.time() - start_time

    # --- Analysis ---
    print(f"\n\n{'=' * 70}")
    print(f"M12 RESULTS — Ground Truth: {true_model} ({total_time:.0f}s total)")
    print(f"{'=' * 70}")

    # Prediction leaderboard
    board = protocol.state.prediction_leaderboard()
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

    # --- M12-specific: Continuous design space analysis ---
    print("\n### Design Space Analysis (M12 — Continuous)")

    # Track which structures and conditions were selected
    experiments = state.experiments
    selected_structs = []
    selected_conds = []
    for exp in experiments:
        ds = exp.design_spec or {}
        s = ds.get("structure_name", "?")
        c = ds.get("condition", "?")
        selected_structs.append(s)
        selected_conds.append(c)

    print(f"  Experiments run: {len(experiments)}")
    for i, (s, c) in enumerate(zip(selected_structs, selected_conds)):
        is_sampled = s.startswith("sampled_")
        is_base = s in STRUCTURE_REGISTRY
        is_parametric_c = c in PARAMETRIC_CONDITIONS
        tags = []
        if is_sampled:
            tags.append("SAMPLED")
        elif is_base:
            tags.append("BASE")
        else:
            tags.append("PARAMETRIC")
        if is_parametric_c:
            tags.append("INTERP-COND")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        print(f"    Cycle {i}: {s} / {c}{tag_str}")

    n_sampled = sum(1 for s in selected_structs if s.startswith("sampled_"))
    n_base = sum(1 for s in selected_structs if s in STRUCTURE_REGISTRY)
    n_parametric_cond = sum(1 for c in selected_conds if c in PARAMETRIC_CONDITIONS)
    print(f"\n  Sampled structures selected: {n_sampled}/{len(experiments)}")
    print(f"  Base structures selected: {n_base}/{len(experiments)}")
    print(f"  Parametric conditions selected: {n_parametric_cond}/{len(experiments)}")
    print(f"  Unique structures: {len(set(selected_structs))}")
    print(f"  Unique conditions: {len(set(selected_conds))}")

    # Key M12 metric: do different cycles have different sampled structures?
    print("\n### Cycle Diversity (key M12 metric)")
    if len(sampled_per_cycle) >= 2:
        for i in range(1, len(sampled_per_cycle)):
            overlap = sampled_per_cycle[i] & sampled_per_cycle[i - 1]
            pct = len(overlap) / max(len(sampled_per_cycle[i]), 1) * 100
            print(
                f"  Cycle {i - 1}→{i} overlap: {len(overlap)} structures ({pct:.0f}%)"
            )
    else:
        print("  (only 1 cycle — no diversity comparison)")

    # Parameter distribution analysis
    print("\n### Parameter Distribution of Selected Sampled Structures")
    for s in selected_structs:
        if s.startswith("sampled_ls_"):
            # Parse: sampled_ls_{n_dims}d_sep{sep}_{idx}
            parts = s.replace("sampled_ls_", "").split("d_sep")
            if len(parts) == 2:
                dims = parts[0]
                sep_idx = parts[1].rsplit("_", 1)
                sep = sep_idx[0] if sep_idx else "?"
                print(f"    LS: dims={dims}, separation={sep}")
        elif s.startswith("sampled_rpe_"):
            parts = s.replace("sampled_rpe_", "").split("d_")
            if len(parts) == 2:
                dims = parts[0]
                exc_idx = parts[1].replace("exc", "").rsplit("_", 1)
                exc = exc_idx[0] if exc_idx else "?"
                print(f"    RPE: dims={dims}, exceptions={exc}")

    # Claim-responsive analysis (from M10)
    print("\n### Claim-Responsive Analysis")
    claim_ledger = protocol.state.claim_ledger
    falsified = [c for c in claim_ledger if c.status == "falsified"]
    print(f"  Claims: {len(claim_ledger)} total | {len(falsified)} falsified")

    interp_messages = [
        m for m in transcript if m.get("phase") == "INTERPRETATION_DEBATE"
    ]
    n_with_fr = 0
    n_theory_interps = 0
    for m in interp_messages:
        parsed = m.get("parsed_json", {})
        agent_name = m.get("agent", "")
        if agent_name in ("Integrator", "Critic"):
            continue
        n_theory_interps += 1
        if parsed.get("falsified_response"):
            n_with_fr += 1
    fr_rate = (n_with_fr / n_theory_interps * 100) if n_theory_interps > 0 else 0
    print(f"  FR rate: {fr_rate:.0f}% ({n_with_fr}/{n_theory_interps})")

    # Crux analysis
    print("\n### Crux Summary")
    total_cruxes = len(protocol.state.cruxes)
    accepted = [c for c in protocol.state.cruxes if c.status == "accepted"]
    print(f"  Total: {total_cruxes}, Accepted: {len(accepted)}")

    # Bayesian posterior
    if protocol.state.model_posterior and "log_probs" in protocol.state.model_posterior:
        lp = np.array(protocol.state.model_posterior["log_probs"])
        probs = np.exp(lp - np.max(lp))
        probs = probs / probs.sum()
        model_names = protocol.state.model_posterior.get("model_names", [])
        print("\n### Final Bayesian Posterior")
        for i, p in enumerate(probs):
            name = model_names[i] if i < len(model_names) else f"model_{i}"
            marker = " <-- MOST PROBABLE" if p == max(probs) else ""
            print(f"  P({name}) = {p:.4f}{marker}")

    # Save analysis
    analysis = {
        "ground_truth": true_model,
        "milestone": "M12",
        "n_cycles": n_cycles,
        "total_time_s": total_time,
        "winner": winner,
        "winner_rmse": float(winner_rmse) if winner_rmse < 999 else None,
        "gap_pct": float(gap_pct),
        "design_space": "continuous",
        "n_continuous_samples": runner_mod._N_CONTINUOUS_SAMPLES,
        "pool_size_approx": n_structs * n_conds,
        "selected_structures": selected_structs,
        "selected_conditions": selected_conds,
        "n_sampled_selected": n_sampled,
        "n_base_selected": n_base,
        "n_parametric_cond_selected": n_parametric_cond,
        "unique_structures": len(set(selected_structs)),
        "unique_conditions": len(set(selected_conds)),
        "total_claims": len(claim_ledger),
        "falsified_claims": len(falsified),
        "fr_rate": fr_rate,
        "total_cruxes": total_cruxes,
        "accepted_cruxes": len(accepted),
        "leaderboard": {
            a: {"rmse": s.get("mean_score"), "n": s.get("n_predictions")}
            for a, s in board.items()
        }
        if board
        else {},
    }
    with open(os.path.join(output_dir, "m12_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    return analysis


if __name__ == "__main__":
    # Set globals for M12 validation
    runner_mod._LLM_MODEL = "gpt-4o"
    runner_mod._BATCH_MODE = True
    runner_mod._SELECTION_METHOD = "bayesian"
    runner_mod._SELECTION_STRATEGY = "thompson"
    runner_mod._LEARNING_RATE = 0.005
    runner_mod._ARBITER = True
    runner_mod._CRUX_WEIGHT = 0.3
    runner_mod._CLAIM_RESPONSIVE = True
    runner_mod._DESIGN_SPACE = "continuous"  # M12 feature
    runner_mod._N_CONTINUOUS_SAMPLES = 50

    # Create Princeton client
    if not os.environ.get("AI_SANDBOX_KEY"):
        # Try loading from .env
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

    client = _create_client(backend="princeton")

    models = ["GCM", "SUSTAIN", "RULEX"]

    # Allow running a single model via CLI arg
    if len(sys.argv) > 1 and sys.argv[1] in models:
        models = [sys.argv[1]]

    results = {}
    for true_model in models:
        try:
            results[true_model] = run_validation(client, true_model, n_cycles=5)
        except Exception as e:
            print(f"\nERROR running {true_model}: {e}")
            import traceback

            traceback.print_exc()
            results[true_model] = {"error": str(e)}

    # Summary table
    print(f"\n\n{'=' * 70}")
    print("M12 VALIDATION SUMMARY")
    print(f"{'=' * 70}")
    print(
        f"{'GT':<10} {'Winner':<20} {'OK?':<5} {'RMSE':<8} {'Gap%':<8} "
        f"{'Sampled':<8} {'Base':<8} {'FR%':<6}"
    )
    print("-" * 83)

    expected = {
        "GCM": "Exemplar_Agent",
        "SUSTAIN": "Clustering_Agent",
        "RULEX": "Rule_Agent",
    }
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
        ns = r.get("n_sampled_selected", 0)
        nb = r.get("n_base_selected", 0)
        fr = r.get("fr_rate", 0)
        print(
            f"{model:<10} {w:<20} {correct:<5} {rmse_str:<8} {gap:<8.1f} "
            f"{ns:<8} {nb:<8} {fr:<6.0f}%"
        )

    # Save combined results
    combined_path = "runs/m12_validation_summary.json"
    os.makedirs("runs", exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to: {combined_path}")
