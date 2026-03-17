"""
M6 ARBITER Validation — Run 5-cycle debates with all M6 features enabled.

Uses a structured mock LLM that produces realistic, theory-appropriate
responses. This exercises the full M6 machinery:
- Meta-agents (Integrator, Critic) in interpretation debate
- Crux identification, negotiation, finalization
- Crux-based EIG boosting
- Conflict map injection
- Pre-registration output
- All existing M5 features (param revision, claim ledger, etc.)

No API key needed — computation layer is real, LLM layer is mocked.
"""

import json
import os
import sys

import numpy as np

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from antagonistic_collab.epistemic_state import EpistemicState
from antagonistic_collab.debate_protocol import DebateProtocol, default_agent_configs
from antagonistic_collab.runner import (
    run_cycle,
    create_default_meta_agents,
    generate_preregistration,
)
import antagonistic_collab.runner as runner_mod


# ---------------------------------------------------------------------------
# Mock LLM that produces structured, theory-appropriate responses
# ---------------------------------------------------------------------------

CYCLE_CRUX_PROPOSALS = {
    "Exemplar_Agent": [
        {"description": "GCM predicts smooth generalization gradients on Type VI that RULEX cannot match",
         "discriminating_experiment": "Type_VI/baseline",
         "resolution_criterion": "RMSE difference > 0.10 between GCM and RULEX on Type VI"},
        {"description": "Exemplar storage predicts item-specific effects on five_four structure",
         "discriminating_experiment": "five_four/baseline",
         "resolution_criterion": "GCM item-level correlation > 0.80"},
    ],
    "Rule_Agent": [
        {"description": "Type I learning should show sudden rule discovery, not gradual improvement",
         "discriminating_experiment": "Type_I/baseline",
         "resolution_criterion": "Learning curve shows discrete jump (max_jump > 0.15)"},
        {"description": "Verbal load should disrupt rule-based but not exemplar-based learning",
         "discriminating_experiment": "Type_II/high_noise",
         "resolution_criterion": "RULEX accuracy drops > 15% under noise; GCM drops < 5%"},
    ],
    "Clustering_Agent": [
        {"description": "SUSTAIN predicts order effects that neither GCM nor RULEX can explain",
         "discriminating_experiment": "Type_IV/fast_presentation",
         "resolution_criterion": "SUSTAIN RMSE < 0.20 on Type IV; others > 0.30"},
        {"description": "Cluster recruitment creates qualitatively different learning curves",
         "discriminating_experiment": "Type_II/baseline",
         "resolution_criterion": "SUSTAIN shows stepwise learning; GCM shows gradual"},
    ],
}


def mock_llm(client, system_prompt, user_message, **kwargs):
    """Structured mock LLM that produces phase-appropriate responses."""

    user_lower = user_message.lower()

    # --- Crux Identification ---
    if "crux identification" in user_lower:
        # Match on theory keywords that appear in real system prompts
        # (EXEMPLAR_AGENT_PROMPT contains "EXEMPLAR theory", etc.)
        prompt_lower = system_prompt.lower()
        agent_prompt_markers = {
            "Exemplar_Agent": "exemplar",
            "Rule_Agent": "rule",
            "Clustering_Agent": "clustering",
        }
        for agent_name, cruxes in CYCLE_CRUX_PROPOSALS.items():
            marker = agent_prompt_markers.get(agent_name, "")
            if marker and marker in prompt_lower:
                return json.dumps({"cruxes": cruxes})
        return '{"cruxes": []}'

    # --- Crux Negotiation ---
    if "crux negotiation" in user_lower:
        # Each agent accepts cruxes from others, rejects their own (already supporter)
        responses = []
        for crux_id_num in range(1, 8):
            crux_id = f"crux_{crux_id_num:03d}"
            if crux_id in user_message:
                responses.append({"crux_id": crux_id, "action": "accept"})
        return json.dumps({"responses": responses})

    # --- Commitment Phase ---
    if "commitment" in user_lower or "register your theory" in user_lower:
        if "exemplar" in system_prompt.lower():
            return json.dumps({
                "core_claims": [
                    "People store individual exemplars in memory",
                    "Classification based on summed similarity to stored instances",
                    "Attention weights learned through experience",
                ],
                "auxiliary_assumptions": ["Exponential decay similarity function"],
                "model_evidence": {},
            })
        elif "rule" in system_prompt.lower():
            return json.dumps({
                "core_claims": [
                    "People search for verbalizable classification rules",
                    "Rule search is hierarchical: simple rules first",
                    "Exceptions stored for rule-violating items",
                ],
                "auxiliary_assumptions": ["Stochastic rule search"],
                "model_evidence": {},
            })
        else:
            return json.dumps({
                "core_claims": [
                    "Learners form flexible clusters driven by surprisal",
                    "New clusters recruited when predictions fail",
                    "Representation complexity adapts to task structure",
                ],
                "auxiliary_assumptions": ["Presentation order matters"],
                "model_evidence": {},
            })

    # --- Divergence Mapping ---
    if "divergence" in user_lower:
        return json.dumps({
            "divergence_report": "Models diverge most on Type VI and five_four structures",
            "model_evidence": {},
        })

    # --- Interpretation Debate ---
    if "interpretation" in user_lower and "debate" in user_lower:
        if "integrator" in system_prompt.lower() or "synthesiz" in system_prompt.lower():
            return json.dumps({
                "interpretation": "All three models capture aspects of human categorization. "
                    "GCM and SUSTAIN converge on similarity-based mechanisms but diverge on "
                    "representation granularity. The key unresolved question is whether "
                    "the cluster recruitment mechanism adds explanatory power beyond exemplar storage.",
                "confounds_flagged": ["Parameter sensitivity varies across models"],
                "hypothesis": "Test Type IV structure where cluster number is diagnostic",
                "claims": [],
            })
        elif "critic" in system_prompt.lower() or "challeng" in system_prompt.lower() or "weakest" in system_prompt.lower():
            return json.dumps({
                "interpretation": "The weakest argument this cycle comes from the model with "
                    "highest RMSE. Its advocate is explaining away poor predictions rather than "
                    "acknowledging the model's limitations on this structure. The claimed parameter "
                    "revision is degenerative — it accommodates the data without generating new predictions.",
                "confounds_flagged": ["Post-hoc parameter fitting may inflate apparent accuracy"],
                "hypothesis": "Test on a novel structure to distinguish genuine prediction from overfitting",
                "claims": [],
            })
        else:
            # Theory agent interpretation
            if "exemplar" in system_prompt.lower():
                return json.dumps({
                    "interpretation": "GCM's predictions align well with observed accuracy patterns. "
                        "The smooth generalization gradient is consistent with exemplar-based similarity.",
                    "confounds_flagged": [],
                    "hypothesis": "Test five_four structure where exemplar effects are strongest",
                    "claims": [
                        {"claim": "GCM predicts >80% accuracy on Type I", "testable": True,
                         "structure": "Type_I", "predicted_outcome": "accuracy>0.80", "claim_type": "prediction"},
                        {"claim": "GCM captures item-specific effects", "testable": False, "claim_type": "explanation"},
                    ],
                    "revision": {"description": "Increase sensitivity to match observed learning speed",
                                 "new_params": {"c": 3.5}, "new_predictions": ["Faster convergence on Type II"]},
                })
            elif "rule" in system_prompt.lower():
                return json.dumps({
                    "interpretation": "RULEX correctly predicts the Type I advantage. "
                        "The discrete rule discovery mechanism explains sudden learning transitions.",
                    "confounds_flagged": [],
                    "hypothesis": "Test Type II under verbal load to show rule-dependency",
                    "claims": [
                        {"claim": "RULEX predicts >90% on Type I", "testable": True,
                         "structure": "Type_I", "predicted_outcome": "accuracy>0.90", "claim_type": "prediction"},
                        {"claim": "Rule discovery is discrete, not gradual", "testable": True,
                         "structure": "Type_I", "predicted_outcome": "learning_curve_jump>0.15", "claim_type": "prediction"},
                    ],
                    "revision": None,
                })
            else:
                return json.dumps({
                    "interpretation": "SUSTAIN's cluster recruitment mechanism provides the most "
                        "flexible account. Prediction accuracy is competitive with GCM.",
                    "confounds_flagged": ["Presentation order not controlled in this design"],
                    "hypothesis": "Test order effects with Type IV structure",
                    "claims": [
                        {"claim": "SUSTAIN predicts adaptive complexity on Type IV", "testable": True,
                         "structure": "Type_IV", "predicted_outcome": "SUSTAIN_RMSE<0.25", "claim_type": "prediction"},
                    ],
                    "revision": {"description": "Lower recruitment threshold for faster adaptation",
                                 "new_params": {"tau": 0.8}, "new_predictions": ["More clusters on Type VI"]},
                })

    # --- Interpretation Critique ---
    if "critique" in user_lower and "interpretation" in user_lower:
        return json.dumps({
            "disputed_interpretation": "The opponent's claim of high accuracy is unverified",
            "alternative_prediction": {"structure": "Type_VI", "condition": "baseline", "my_model_predicts": 0.65},
            "distinguishing_experiment": "Type VI under baseline conditions",
        })

    # --- Audit ---
    if "audit" in user_lower:
        return (
            "AUDIT SUMMARY:\n"
            "1. This cycle tested a structure selected by Bayesian EIG\n"
            "2. Predictions varied across models — key discriminations identified\n"
            "3. Theory revisions proposed by 2 agents (progressive)\n"
            "4. Active cruxes focus next cycle on decisive questions\n"
            "5. Next cycle should target the highest-EIG crux-aligned experiment\n"
            "6. CONVERGENCE CHECK: Agents maintain distinct positions; adversarial pressure healthy"
        )

    # --- Default fallback ---
    return json.dumps({
        "response": "Acknowledged",
        "model_evidence": {},
        "core_claims": ["test"],
        "auxiliary_assumptions": [],
    })


# ---------------------------------------------------------------------------
# Run validation
# ---------------------------------------------------------------------------

def run_validation(true_model: str, n_cycles: int = 5, output_dir: str = None):
    """Run a full M6 validation with specified ground truth."""

    if output_dir is None:
        output_dir = f"runs/m6_validation_{true_model}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"M6 ARBITER VALIDATION — Ground Truth: {true_model}")
    print(f"{'=' * 70}")
    print("  Features: meta-agents, crux negotiation, conflict map, pre-registration")
    print(f"  Cycles: {n_cycles}")
    print(f"  Output: {output_dir}")
    print()

    # Initialize
    state = EpistemicState(domain="Human Categorization")
    agents = default_agent_configs()
    protocol = DebateProtocol(state, agents, meta_agents=create_default_meta_agents())
    transcript = []

    # Patch call_agent with mock
    original_call = runner_mod.call_agent
    original_batch_mode = runner_mod._BATCH_MODE
    runner_mod.call_agent = mock_llm

    # Enable batch mode
    runner_mod._BATCH_MODE = True

    metadata = {
        "true_model": true_model,
        "llm_model": "mock_structured",
        "backend": "mock",
        "m6_features": "meta-agents, crux_negotiation, conflict_map, preregistration",
    }

    try:
        for cycle in range(n_cycles):
            print(f"\n{'=' * 50}")
            print(f"  Starting Cycle {cycle}")
            print(f"{'=' * 50}")

            # Generate pre-registration before execution
            if cycle > 0:
                prereg = generate_preregistration(protocol, cycle=protocol.state.cycle)
                prereg_path = os.path.join(output_dir, f"preregistration_cycle_{protocol.state.cycle}.json")
                with open(prereg_path, "w") as f:
                    json.dump(prereg, f, indent=2, default=str)
                print(f"  Pre-registration saved to {prereg_path}")

            run_cycle(
                protocol,
                None,  # no real client needed
                transcript,
                true_model=true_model,
                critique_rounds=1,
                output_dir=output_dir,
                metadata=metadata,
                mode="full_pool",
            )

    finally:
        runner_mod.call_agent = original_call
        runner_mod._BATCH_MODE = original_batch_mode

    # --- Analysis ---
    print(f"\n\n{'=' * 70}")
    print(f"M6 VALIDATION RESULTS — Ground Truth: {true_model}")
    print(f"{'=' * 70}")

    # Prediction leaderboard
    board = protocol.state.prediction_leaderboard()
    if board:
        print("\n### Prediction Leaderboard (RMSE — lower is better)")
        sorted_board = sorted(board.items(), key=lambda x: x[1].get("mean_score", 999))
        for rank, (agent, stats) in enumerate(sorted_board):
            mean = stats.get("mean_score")
            n = stats.get("n_predictions", 0)
            marker = " ← WINNER" if rank == 0 else ""
            if mean is not None:
                print(f"  {rank+1}. {agent}: RMSE={mean:.4f} ({n} predictions){marker}")
            else:
                print(f"  {rank+1}. {agent}: not scored ({n} predictions)")

        winner = sorted_board[0][0]
        winner_rmse = sorted_board[0][1].get("mean_score", 999)
        runner_up_rmse = sorted_board[1][1].get("mean_score", 999) if len(sorted_board) > 1 else 999
        gap_pct = ((runner_up_rmse - winner_rmse) / runner_up_rmse * 100) if runner_up_rmse > 0 else 0
        print(f"\n  Winner: {winner} (gap: {gap_pct:.1f}%)")

    # Crux analysis
    print("\n### Crux Negotiation Summary")
    total_cruxes = len(protocol.state.cruxes)
    accepted = [c for c in protocol.state.cruxes if c.status == "accepted"]
    rejected = [c for c in protocol.state.cruxes if c.status == "rejected"]
    resolved = [c for c in protocol.state.cruxes if c.status == "resolved"]
    print(f"  Total cruxes proposed: {total_cruxes}")
    print(f"  Accepted: {len(accepted)}")
    print(f"  Rejected: {len(rejected)}")
    print(f"  Resolved: {len(resolved)}")

    if accepted:
        print("\n  Accepted cruxes:")
        for c in accepted:
            print(f"    - {c.id}: {c.description}")
            print(f"      Experiment: {c.discriminating_experiment}, Supporters: {c.supporters}")

    # Conflict map
    conflict_map = protocol.state.conflict_map_summary()
    if conflict_map:
        print("\n### Conflict Map")
        print(conflict_map)

    # Claim ledger stats
    print("\n### Claim Ledger")
    total_claims = len(protocol.state.claim_ledger)
    confirmed = sum(1 for c in protocol.state.claim_ledger if c.status == "confirmed")
    falsified = sum(1 for c in protocol.state.claim_ledger if c.status == "falsified")
    untested = sum(1 for c in protocol.state.claim_ledger if c.status == "untested")
    print(f"  Total claims: {total_claims}")
    print(f"  Confirmed: {confirmed}, Falsified: {falsified}, Untested: {untested}")

    # Meta-agent analysis
    meta_messages = [m for m in transcript if m.get("meta_agent")]
    print("\n### Meta-Agent Activity")
    print(f"  Total meta-agent responses: {len(meta_messages)}")
    for m in meta_messages[:4]:
        parsed = m.get("parsed_json", {})
        interp = parsed.get("interpretation", "")[:120]
        print(f"    {m['agent']}: {interp}...")

    # Theory trajectory
    print("\n### Theory Trajectories")
    for theory in protocol.state.active_theories():
        try:
            traj = protocol.state.theory_trajectory(theory.name)
            print(f"  {theory.name}: {traj['trajectory']} "
                  f"({traj['n_revisions']} revisions, "
                  f"{traj['n_progressive']} progressive, "
                  f"{traj['n_degenerative']} degenerative)")
        except Exception as e:
            print(f"  {theory.name}: error computing trajectory: {e}")

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

    # EIG selection history
    eig_selections = [m for m in transcript if m.get("phase") == "FULL_POOL_SELECTION"]
    if eig_selections:
        print("\n### Experiment Selection History")
        for sel in eig_selections:
            s = sel.get("selected", {})
            eig = sel.get("eig", 0)
            print(f"  Cycle: {s.get('structure', '?')} / {s.get('condition', '?')} (EIG={eig:.4f})")

    # Save analysis
    analysis = {
        "ground_truth": true_model,
        "n_cycles": n_cycles,
        "winner": winner if board else None,
        "winner_rmse": float(winner_rmse) if board else None,
        "gap_pct": float(gap_pct) if board else None,
        "total_cruxes": total_cruxes,
        "accepted_cruxes": len(accepted),
        "meta_agent_responses": len(meta_messages),
        "total_claims": total_claims,
        "confirmed_claims": confirmed,
        "falsified_claims": falsified,
        "leaderboard": {a: {"rmse": s.get("mean_score"), "n": s.get("n_predictions")} for a, s in board.items()} if board else {},
    }
    analysis_path = os.path.join(output_dir, "m6_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\n  Analysis saved to {analysis_path}")

    return analysis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = {}
    for true_model in ["GCM", "SUSTAIN", "RULEX"]:
        results[true_model] = run_validation(true_model, n_cycles=5)

    print(f"\n\n{'=' * 70}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n{'Model':<10} {'Winner':<22} {'RMSE':>8} {'Gap%':>8} {'Cruxes':>8} {'Claims':>8} {'Meta':>6}")
    print("-" * 70)
    for model, r in results.items():
        print(f"{model:<10} {r['winner']:<22} {r['winner_rmse']:8.4f} {r['gap_pct']:7.1f}% {r['accepted_cruxes']:8} {r['total_claims']:8} {r['meta_agent_responses']:6}")

    all_correct = all(
        (r["winner"] == "Exemplar_Agent" and m == "GCM") or
        (r["winner"] == "Rule_Agent" and m == "RULEX") or
        (r["winner"] == "Clustering_Agent" and m == "SUSTAIN")
        for m, r in results.items()
    )
    print(f"\nAll correct winners: {'YES' if all_correct else 'NO'}")
