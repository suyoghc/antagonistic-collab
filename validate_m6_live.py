"""
M6 ARBITER Live Validation — Real LLM calls via Princeton AI Sandbox.

Runs 5-cycle debates with all M6 features enabled using GPT-4o.
Tests 3 ground truths: GCM, SUSTAIN, RULEX.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from antagonistic_collab.epistemic_state import EpistemicState, TheoryCommitment
from antagonistic_collab.debate_protocol import DebateProtocol, default_agent_configs
from antagonistic_collab.runner import (
    run_cycle,
    create_default_meta_agents,
    generate_preregistration,
    _create_client,
    _DEFAULT_MODELS,
)
import antagonistic_collab.runner as runner_mod


def run_validation(client, true_model: str, n_cycles: int = 5):
    """Run a full M6 validation with real LLM calls."""

    output_dir = f"runs/m6_live_{true_model}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"M6 ARBITER LIVE VALIDATION — Ground Truth: {true_model}")
    print(f"{'=' * 70}")
    print(f"  LLM: gpt-4o via Princeton AI Sandbox")
    print(f"  Features: meta-agents, crux negotiation, conflict map, pre-registration")
    print(f"  Cycles: {n_cycles}")
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
        "m6_features": "meta-agents, crux_negotiation, conflict_map, preregistration",
    }

    start_time = time.time()

    for cycle in range(n_cycles):
        cycle_start = time.time()
        print(f"\n{'=' * 50}")
        print(f"  Starting Cycle {cycle} ({time.time() - start_time:.0f}s elapsed)")
        print(f"{'=' * 50}")

        # Generate pre-registration before execution (after cycle 0)
        if cycle > 0:
            prereg = generate_preregistration(protocol, cycle=protocol.state.cycle)
            prereg_path = os.path.join(output_dir, f"preregistration_cycle_{protocol.state.cycle}.json")
            with open(prereg_path, "w") as f:
                json.dump(prereg, f, indent=2, default=str)

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
    print(f"M6 LIVE RESULTS — Ground Truth: {true_model} ({total_time:.0f}s total)")
    print(f"{'=' * 70}")

    # Prediction leaderboard
    board = protocol.state.prediction_leaderboard()
    winner = None
    winner_rmse = 999
    gap_pct = 0
    if board:
        print("\n### Prediction Leaderboard (RMSE — lower is better)")
        sorted_board = sorted(board.items(), key=lambda x: x[1].get("mean_score", 999))
        for rank, (agent, stats) in enumerate(sorted_board):
            mean = stats.get("mean_score")
            n = stats.get("n_predictions", 0)
            marker = " ← WINNER" if rank == 0 else ""
            if mean is not None:
                print(f"  {rank+1}. {agent}: RMSE={mean:.4f} ({n} predictions){marker}")

        winner = sorted_board[0][0]
        winner_rmse = sorted_board[0][1].get("mean_score", 999)
        if len(sorted_board) > 1:
            runner_up_rmse = sorted_board[1][1].get("mean_score", 999)
            gap_pct = ((runner_up_rmse - winner_rmse) / runner_up_rmse * 100) if runner_up_rmse > 0 else 0
        print(f"\n  Winner: {winner} (gap: {gap_pct:.1f}%)")

    # Crux analysis
    print(f"\n### Crux Negotiation Summary")
    total_cruxes = len(protocol.state.cruxes)
    accepted = [c for c in protocol.state.cruxes if c.status == "accepted"]
    rejected = [c for c in protocol.state.cruxes if c.status == "rejected"]
    print(f"  Total proposed: {total_cruxes}, Accepted: {len(accepted)}, Rejected: {len(rejected)}")
    if accepted:
        print(f"  Accepted cruxes:")
        for c in accepted[:5]:
            print(f"    - {c.id}: {c.description[:80]}")
            print(f"      Exp: {c.discriminating_experiment}, Supporters: {len(c.supporters)}")

    # Conflict map
    conflict_map = protocol.state.conflict_map_summary()
    if conflict_map:
        # Truncate for readability
        lines = conflict_map.split("\n")
        print(f"\n### Conflict Map ({len(lines)} lines)")
        for line in lines[:20]:
            print(line)
        if len(lines) > 20:
            print(f"  ... ({len(lines) - 20} more lines)")

    # Claim ledger
    print(f"\n### Claim Ledger")
    total_claims = len(protocol.state.claim_ledger)
    confirmed = sum(1 for c in protocol.state.claim_ledger if c.status == "confirmed")
    falsified = sum(1 for c in protocol.state.claim_ledger if c.status == "falsified")
    untested = sum(1 for c in protocol.state.claim_ledger if c.status == "untested")
    print(f"  Total: {total_claims} | Confirmed: {confirmed} | Falsified: {falsified} | Untested: {untested}")

    # Meta-agent analysis
    meta_messages = [m for m in transcript if m.get("meta_agent")]
    print(f"\n### Meta-Agent Activity ({len(meta_messages)} responses)")
    for m in meta_messages[:6]:
        parsed = m.get("parsed_json", {})
        interp = parsed.get("interpretation", m.get("response", ""))[:150]
        print(f"  [{m['agent']}]: {interp}...")

    # Theory trajectories
    print(f"\n### Theory Trajectories")
    for theory in protocol.state.active_theories():
        try:
            traj = protocol.state.theory_trajectory(theory.name)
            print(f"  {theory.name}: {traj['trajectory']} "
                  f"({traj['n_revisions']} rev, {traj['n_progressive']} prog, {traj['n_degenerative']} degen)")
        except Exception as e:
            print(f"  {theory.name}: {e}")

    # Bayesian posterior
    if protocol.state.model_posterior and "log_probs" in protocol.state.model_posterior:
        lp = np.array(protocol.state.model_posterior["log_probs"])
        probs = np.exp(lp - np.max(lp))
        probs = probs / probs.sum()
        model_names = protocol.state.model_posterior.get("model_names", [])
        print(f"\n### Final Bayesian Posterior")
        for i, p in enumerate(probs):
            name = model_names[i] if i < len(model_names) else f"model_{i}"
            marker = " ← MOST PROBABLE" if p == max(probs) else ""
            print(f"  P({name}) = {p:.4f}{marker}")

    # Experiment selections
    eig_selections = [m for m in transcript if m.get("phase") == "FULL_POOL_SELECTION"]
    if eig_selections:
        print(f"\n### Experiment Selection History")
        for sel in eig_selections:
            s = sel.get("selected", {})
            eig = sel.get("eig", 0)
            print(f"  {s.get('structure', '?')} / {s.get('condition', '?')} (EIG={eig:.4f})")

    # Save analysis
    analysis = {
        "ground_truth": true_model,
        "n_cycles": n_cycles,
        "total_time_s": total_time,
        "winner": winner,
        "winner_rmse": float(winner_rmse) if winner_rmse < 999 else None,
        "gap_pct": float(gap_pct),
        "total_cruxes": total_cruxes,
        "accepted_cruxes": len(accepted),
        "total_claims": total_claims,
        "confirmed_claims": confirmed,
        "falsified_claims": falsified,
        "meta_agent_responses": len(meta_messages),
        "leaderboard": {a: {"rmse": s.get("mean_score"), "n": s.get("n_predictions")}
                        for a, s in board.items()} if board else {},
    }
    with open(os.path.join(output_dir, "m6_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    return analysis


if __name__ == "__main__":
    # Set globals
    runner_mod._LLM_MODEL = "gpt-4o"
    runner_mod._BATCH_MODE = True
    runner_mod._SELECTION_METHOD = "bayesian"

    # Create Princeton client
    os.environ.setdefault("AI_SANDBOX_KEY", "T7ldjeA6ugVWCbNmyAXLUuuj5aza")
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
            print(f"\n  ERROR on {true_model}: {e}")
            import traceback
            traceback.print_exc()
            results[true_model] = {"error": str(e)}

    if len(results) > 1:
        print(f"\n\n{'=' * 70}")
        print("CROSS-MODEL SUMMARY")
        print(f"{'=' * 70}")
        print(f"\n{'Model':<10} {'Winner':<22} {'RMSE':>8} {'Gap%':>8} {'Cruxes':>8} {'Claims':>8} {'Meta':>6} {'Time':>6}")
        print("-" * 78)
        for model, r in results.items():
            if "error" in r:
                print(f"{model:<10} ERROR: {r['error'][:50]}")
            else:
                print(f"{model:<10} {r.get('winner','?'):<22} {r.get('winner_rmse',0):8.4f} "
                      f"{r.get('gap_pct',0):7.1f}% {r.get('accepted_cruxes',0):8} "
                      f"{r.get('total_claims',0):8} {r.get('meta_agent_responses',0):6} "
                      f"{r.get('total_time_s',0):5.0f}s")

        all_correct = all(
            (r.get("winner") == "Exemplar_Agent" and m == "GCM") or
            (r.get("winner") == "Rule_Agent" and m == "RULEX") or
            (r.get("winner") == "Clustering_Agent" and m == "SUSTAIN")
            for m, r in results.items() if "error" not in r
        )
        print(f"\nAll correct winners: {'YES' if all_correct else 'NO'}")
