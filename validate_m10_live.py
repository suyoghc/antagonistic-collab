"""
M10 Claim-Responsive Debate Live Validation — Real LLM calls.

Runs 5-cycle debates with claim-responsive debate enabled using GPT-4o.
Tests 3 ground truths: GCM, SUSTAIN, RULEX.

Key M10 metrics:
- Do agents produce "falsified_response" JSON fields?
- Do agents acknowledge/revise/abandon falsified claims?
- Does debate quality improve compared to prior milestones?
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from antagonistic_collab.epistemic_state import EpistemicState
from antagonistic_collab.debate_protocol import DebateProtocol, default_agent_configs
from antagonistic_collab.runner import (
    run_cycle,
    create_default_meta_agents,
    _create_client,
)
import antagonistic_collab.runner as runner_mod


def run_validation(client, true_model: str, n_cycles: int = 5):
    """Run a full M10 validation with real LLM calls."""

    output_dir = f"runs/m10_val_{true_model}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"M10 CLAIM-RESPONSIVE VALIDATION — Ground Truth: {true_model}")
    print(f"{'=' * 70}")
    print("  LLM: gpt-4o via Princeton AI Sandbox")
    print("  Features: all M6-M9 + claim-responsive debate (M10)")
    print(f"  Cycles: {n_cycles}")
    print(f"  claim_responsive: {runner_mod._CLAIM_RESPONSIVE}")
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
        "milestone": "M10",
        "claim_responsive": True,
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
    print(f"M10 RESULTS — Ground Truth: {true_model} ({total_time:.0f}s total)")
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

    # --- M10-specific: Claim-responsive analysis ---
    print("\n### Claim-Responsive Debate Analysis (M10)")

    # Count falsified claims per agent
    claim_ledger = protocol.state.claim_ledger
    total_claims = len(claim_ledger)
    falsified = [c for c in claim_ledger if c.status == "falsified"]
    confirmed = [c for c in claim_ledger if c.status == "confirmed"]
    untested = [c for c in claim_ledger if c.status == "untested"]
    print(f"  Claims: {total_claims} total | {len(falsified)} falsified | {len(confirmed)} confirmed | {len(untested)} untested")

    # Per-agent falsification
    agent_names = [a.name for a in agents]
    for agent_name in agent_names:
        agent_falsified = [c for c in falsified if c.agent == agent_name]
        agent_confirmed = [c for c in confirmed if c.agent == agent_name]
        print(f"  {agent_name}: {len(agent_falsified)} falsified, {len(agent_confirmed)} confirmed")
        for c in agent_falsified:
            print(f"    - \"{c.content}\" → {c.evidence}")

    # Check transcript for falsified_response fields
    interp_messages = [m for m in transcript if m.get("phase") == "INTERPRETATION_DEBATE"]
    n_with_falsified_response = 0
    n_theory_agent_interps = 0
    falsified_responses_log = []

    for m in interp_messages:
        parsed = m.get("parsed_json", {})
        agent_name = m.get("agent", "")
        # Skip meta-agents
        if agent_name in ("Integrator", "Critic"):
            continue
        n_theory_agent_interps += 1
        fr = parsed.get("falsified_response")
        if fr:
            n_with_falsified_response += 1
            falsified_responses_log.append({
                "agent": agent_name,
                "cycle": m.get("cycle", "?"),
                "responses": fr,
            })

    print(f"\n  Theory agent interpretations: {n_theory_agent_interps}")
    print(f"  With falsified_response field: {n_with_falsified_response}")
    if n_theory_agent_interps > 0:
        rate = n_with_falsified_response / n_theory_agent_interps * 100
        print(f"  Response rate: {rate:.0f}%")

    if falsified_responses_log:
        print("\n  Falsified claim responses:")
        for entry in falsified_responses_log:
            print(f"    [{entry['agent']}] cycle {entry['cycle']}:")
            responses = entry["responses"]
            if isinstance(responses, list):
                for r in responses:
                    if isinstance(r, dict):
                        action = r.get("action", "?")
                        claim = r.get("claim", "?")[:60]
                        reasoning = r.get("reasoning", "")[:80]
                        print(f"      {action}: \"{claim}\" — {reasoning}")
                    else:
                        print(f"      {r}")
            else:
                print(f"      {str(responses)[:200]}")

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
            marker = " ← MOST PROBABLE" if p == max(probs) else ""
            print(f"  P({name}) = {p:.4f}{marker}")

    # Theory trajectories
    print("\n### Theory Trajectories")
    for theory in protocol.state.active_theories():
        try:
            traj = protocol.state.theory_trajectory(theory.name)
            print(f"  {theory.name}: {traj['trajectory']} "
                  f"({traj['n_revisions']} rev, {traj['n_progressive']} prog, {traj['n_degenerative']} degen)")
        except Exception as e:
            print(f"  {theory.name}: {e}")

    # Save analysis
    analysis = {
        "ground_truth": true_model,
        "milestone": "M10",
        "n_cycles": n_cycles,
        "total_time_s": total_time,
        "winner": winner,
        "winner_rmse": float(winner_rmse) if winner_rmse < 999 else None,
        "gap_pct": float(gap_pct),
        "claim_responsive": True,
        "total_claims": total_claims,
        "falsified_claims": len(falsified),
        "confirmed_claims": len(confirmed),
        "untested_claims": len(untested),
        "theory_agent_interpretations": n_theory_agent_interps,
        "with_falsified_response": n_with_falsified_response,
        "falsified_response_rate": (n_with_falsified_response / n_theory_agent_interps * 100)
            if n_theory_agent_interps > 0 else 0,
        "falsified_responses": falsified_responses_log,
        "total_cruxes": total_cruxes,
        "accepted_cruxes": len(accepted),
        "leaderboard": {a: {"rmse": s.get("mean_score"), "n": s.get("n_predictions")}
                        for a, s in board.items()} if board else {},
    }
    with open(os.path.join(output_dir, "m10_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    return analysis


if __name__ == "__main__":
    # Set globals for M10 validation
    runner_mod._LLM_MODEL = "gpt-4o"
    runner_mod._BATCH_MODE = True
    runner_mod._SELECTION_METHOD = "bayesian"
    runner_mod._SELECTION_STRATEGY = "thompson"
    runner_mod._LEARNING_RATE = 0.005
    runner_mod._ARBITER = True
    runner_mod._CRUX_WEIGHT = 0.3
    runner_mod._CLAIM_RESPONSIVE = True  # M10 feature

    # Create Princeton client — requires AI_SANDBOX_KEY env var
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
    print("M10 VALIDATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Ground Truth':<15} {'Winner':<20} {'Correct?':<10} {'RMSE':<10} {'Gap%':<10} {'Falsified':<10} {'FR rate':<10}")
    print("-" * 85)

    expected = {"GCM": "Exemplar_Agent", "SUSTAIN": "Clustering_Agent", "RULEX": "Rule_Agent"}
    for model in ["GCM", "SUSTAIN", "RULEX"]:
        r = results.get(model, {})
        if "error" in r:
            print(f"{model:<15} ERROR: {r['error'][:50]}")
            continue
        w = r.get("winner", "?")
        correct = "Yes" if w == expected.get(model) else "No"
        rmse = r.get("winner_rmse")
        rmse_str = f"{rmse:.3f}" if rmse else "?"
        gap = r.get("gap_pct", 0)
        falsified = r.get("falsified_claims", 0)
        fr_rate = r.get("falsified_response_rate", 0)
        print(f"{model:<15} {w:<20} {correct:<10} {rmse_str:<10} {gap:<10.1f} {falsified:<10} {fr_rate:<10.0f}%")

    # Save combined results
    combined_path = "runs/m10_validation_summary.json"
    os.makedirs("runs", exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to: {combined_path}")
