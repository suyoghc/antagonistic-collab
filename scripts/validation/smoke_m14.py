"""
M14 Smoke Test — Verify claim→computation feedback loop fires end-to-end.

Runs 2-cycle debate with GCM ground truth. Checks that:
1. Claim-directed specs appear in EIG selection output
2. Parameter validation gate fires (accept/reject printed)
3. Claims are auto-resolved after execution

This is NOT a correctness test — just verifying the new code paths fire.
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
    STRUCTURE_REGISTRY,
    CONDITION_EFFECTS,
    PARAMETRIC_CONDITIONS,
)
from antagonistic_collab.runner import (
    run_cycle,
    create_default_meta_agents,
    _create_client,
)
import antagonistic_collab.runner as runner_mod


def run_smoke_test(client, true_model: str = "GCM", n_cycles: int = 2):
    """Run M14 smoke test with real LLM calls."""

    output_dir = f"runs/m14_smoke_{true_model}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"M14 SMOKE TEST — Ground Truth: {true_model}")
    print(f"{'=' * 70}")
    print(f"  Cycles: {n_cycles}")
    print(f"  Checking: claim-directed selection, param validation, claim resolution")
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
        "milestone": "M14_smoke",
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

    # --- M14-specific analysis ---
    print(f"\n\n{'=' * 70}")
    print(f"M14 SMOKE TEST RESULTS — {true_model} ({total_time:.0f}s)")
    print(f"{'=' * 70}")

    # 1. Claim ledger
    ledger = state.claim_ledger
    n_total = len(ledger)
    n_untested = sum(1 for c in ledger if c.status == "untested")
    n_confirmed = sum(1 for c in ledger if c.status == "confirmed")
    n_falsified = sum(1 for c in ledger if c.status == "falsified")
    n_testable = sum(1 for c in ledger if c.testable)

    print(f"\n### Claim Ledger")
    print(f"  Total claims: {n_total}")
    print(f"  Testable: {n_testable}")
    print(f"  Untested: {n_untested}")
    print(f"  Confirmed: {n_confirmed}")
    print(f"  Falsified: {n_falsified}")
    print(f"  Auto-resolved: {n_confirmed + n_falsified}")

    if ledger:
        print(f"\n  Individual claims:")
        for i, c in enumerate(ledger):
            struct_str = f"{c.structure}/{c.condition}" if c.structure else "(no target)"
            print(
                f"    {i}. [{c.status.upper()}] {c.agent}: {c.content[:60]}... "
                f"→ {struct_str}"
            )
            if c.evidence:
                print(f"       Evidence: {c.evidence}")

    # 2. Check if claim-directed selection fired
    claim_directed_msgs = [
        m for m in transcript
        if m.get("phase") == "FULL_POOL_SELECTION"
    ]
    print(f"\n### Claim-Directed Selection")
    print(f"  Selection messages: {len(claim_directed_msgs)}")

    # 3. Experiments and correctness
    print(f"\n### Experiments")
    for exp in state.experiments:
        ds = exp.design_spec or {}
        s = ds.get("structure_name", "?")
        c = ds.get("condition", "?")
        print(f"  Cycle {exp.cycle}: {s} / {c} (status={exp.status})")

    # 4. Leaderboard
    board = state.prediction_leaderboard()
    if board:
        print(f"\n### Prediction Leaderboard (RMSE)")
        sorted_board = sorted(board.items(), key=lambda x: x[1].get("mean_score", 999))
        for rank, (agent, stats) in enumerate(sorted_board):
            mean = stats.get("mean_score")
            n = stats.get("n_predictions", 0)
            marker = " <-- WINNER" if rank == 0 else ""
            if mean is not None:
                print(f"  {rank + 1}. {agent}: RMSE={mean:.4f} ({n} preds){marker}")

        winner = sorted_board[0][0]
        expected = {
            "GCM": "Exemplar_Agent",
            "SUSTAIN": "Clustering_Agent",
            "RULEX": "Rule_Agent",
        }
        correct = winner == expected.get(true_model)
        print(f"\n  Winner: {winner} — {'CORRECT' if correct else 'WRONG'}")

    # 5. Posterior
    if state.model_posterior and "log_probs" in state.model_posterior:
        lp = np.array(state.model_posterior["log_probs"])
        probs = np.exp(lp - np.max(lp))
        probs = probs / probs.sum()
        model_names = state.model_posterior.get("model_names", [])
        print(f"\n### Bayesian Posterior")
        for i, p in enumerate(probs):
            name = model_names[i] if i < len(model_names) else f"model_{i}"
            print(f"  P({name}) = {p:.4f}")

    # --- Verdict ---
    print(f"\n{'=' * 70}")
    checks = {
        "claims_exist": n_total > 0,
        "testable_claims_exist": n_testable > 0,
        "auto_resolved": (n_confirmed + n_falsified) > 0,
        "experiments_ran": len(state.experiments) >= n_cycles,
    }
    all_ok = all(checks.values())
    for name, ok in checks.items():
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"\n  SMOKE TEST: {'PASS' if all_ok else 'FAIL'}")
    print(f"{'=' * 70}")

    # Save
    analysis = {
        "ground_truth": true_model,
        "n_cycles": n_cycles,
        "total_time_s": total_time,
        "n_claims": n_total,
        "n_testable": n_testable,
        "n_confirmed": n_confirmed,
        "n_falsified": n_falsified,
        "n_untested": n_untested,
        "checks": checks,
        "pass": all_ok,
    }
    with open(os.path.join(output_dir, "m14_smoke.json"), "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    return analysis


if __name__ == "__main__":
    # Set globals for M14 smoke test
    runner_mod._LLM_MODEL = "gpt-4o"
    runner_mod._BATCH_MODE = True
    runner_mod._SELECTION_METHOD = "bayesian"
    runner_mod._SELECTION_STRATEGY = "thompson"
    runner_mod._LEARNING_RATE = 0.005
    runner_mod._ARBITER = True
    runner_mod._CRUX_WEIGHT = 0.3
    runner_mod._CLAIM_RESPONSIVE = True
    runner_mod._DESIGN_SPACE = "continuous"
    runner_mod._N_CONTINUOUS_SAMPLES = 50

    # Create Princeton client
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

    client = _create_client(backend="princeton")

    run_smoke_test(client, true_model="GCM", n_cycles=2)
