"""
Demo: Antagonistic Collaboration in Human Categorization.

This script demonstrates the formal layer working WITHOUT any LLM calls.
It shows:
1. Models generating quantitative predictions on standard category structures
2. Divergence mapping — where do the models disagree most?
3. The epistemic state tracker accumulating knowledge
4. A simulated debate cycle

Run this to verify the formal layer works before connecting LLM agents.
"""

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from antagonistic_collab.models import (
    GCM,
    SUSTAIN,
    RULEX,
    shepard_types,
)
from antagonistic_collab.epistemic_state import (
    EpistemicState,
    TheoryCommitment,
)
from antagonistic_collab.debate_protocol import (
    DebateProtocol,
    Phase,
    default_agent_configs,
)


def demo_model_predictions():
    """Show that each model makes different predictions on the same structures."""
    print("=" * 70)
    print("PHASE 1: MODEL PREDICTIONS ON STANDARD STRUCTURES")
    print("=" * 70)

    gcm = GCM()
    sustain = SUSTAIN()
    rulex = RULEX()

    structures = shepard_types()

    print("\nShepard et al. (1961) Six Types — Model Accuracy\n")
    print(f"{'Type':<10} {'GCM':>10} {'SUSTAIN':>10} {'RULEX':>10}")
    print("-" * 42)

    for type_name in ["I", "II", "III", "IV", "V", "VI"]:
        struct = structures[type_name]
        stim = struct["stimuli"]
        labels = struct["labels"]

        # GCM predictions
        gcm_correct = 0
        for item, label in zip(stim, labels):
            pred = gcm.predict(item, stim, labels, c=3.0)
            pred_label = max(pred["probabilities"], key=pred["probabilities"].get)
            if pred_label == label:
                gcm_correct += 1
        gcm_acc = gcm_correct / len(labels)

        # SUSTAIN predictions (simulate learning)
        sequence = list(zip(stim, labels))
        # Run 5 epochs
        full_seq = sequence * 5
        sus_result = sustain.simulate_learning(full_seq)
        sus_log = sus_result["trial_log"]
        # Accuracy on final epoch
        final_epoch = sus_log[-len(sequence) :]
        sus_acc = sum(1 for t in final_epoch if t["correct"]) / len(final_epoch)

        # RULEX predictions
        rule_result = rulex.find_best_rule(stim, labels, seed=42)
        rulex_acc = rule_result["accuracy"]

        print(
            f"Type {type_name:<5} {gcm_acc:>10.3f} {sus_acc:>10.3f} {rulex_acc:>10.3f}"
        )

    print("\n--- Key diagnostic patterns ---")
    print("• GCM predicts relatively flat performance across types (similarity doesn't")
    print("  care about rule structure)")
    print("• RULEX predicts strong Type I advantage and Type VI difficulty")
    print("• SUSTAIN predicts intermediate — depends on cluster recruitment dynamics")
    print()


def demo_divergence_mapping():
    """Show the automatic divergence mapping between models."""
    print("=" * 70)
    print("PHASE 2: DIVERGENCE MAPPING")
    print("=" * 70)

    state = EpistemicState(domain="Human Categorization")
    agents = default_agent_configs()
    protocol = DebateProtocol(state, agents)

    div_map = protocol.compute_divergence_map()

    print("\nWhere do models disagree most?\n")

    # Find the structure with maximum divergence
    max_div = 0
    max_struct = ""
    max_pair = ""

    for struct_name, data in div_map.items():
        for pair, div in data["divergences"].items():
            print(
                f"  {struct_name:>15} | {pair:>35} | "
                f"mean div = {div['mean_abs_diff']:.3f} | "
                f"max at item {div['max_diff_item']} ({div['max_diff_value']:.3f})"
            )
            if div["mean_abs_diff"] > max_div:
                max_div = div["mean_abs_diff"]
                max_struct = struct_name
                max_pair = pair

    print(
        f"\n>>> Maximum divergence: {max_struct}, {max_pair} "
        f"(mean diff = {max_div:.3f})"
    )
    print(
        ">>> This is where an adversarial agent should focus its experiment proposal.\n"
    )


def demo_epistemic_state():
    """Show the epistemic state tracker in action across a simulated cycle."""
    print("=" * 70)
    print("PHASE 3: EPISTEMIC STATE TRACKING")
    print("=" * 70)

    state = EpistemicState(domain="Human Categorization")

    # Register theories — now with term glossaries
    state.register_theory(
        TheoryCommitment(
            name="Exemplar Theory",
            agent_name="Exemplar_Agent",
            core_claims=GCM.core_claims,
            model_name="GCM",
            model_params={"c": 3.0, "r": 1},
            auxiliary_assumptions=[
                "All training instances are stored with equal fidelity",
                "Similarity is computed in a psychological space with city-block metric",
            ],
            term_glossary={
                "attention": "w_i: dimensional weight parameters, sum to 1",
                "similarity": "exp(-c * weighted_distance), exponential decay",
                "learning": "immediate storage of each training instance",
            },
        )
    )

    state.register_theory(
        TheoryCommitment(
            name="Rule-Based Theory",
            agent_name="Rule_Agent",
            core_claims=RULEX.core_claims,
            model_name="RULEX",
            model_params={"p_single": 0.5, "p_conj": 0.3},
            auxiliary_assumptions=[
                "Rule search is exhaustive within the hypothesis space",
                "Verbal working memory is necessary for rule maintenance",
            ],
            term_glossary={
                "attention": "rule search priority: which dimensions are tested first",
                "learning": "discrete rule discovery event + exception memorization",
            },
        )
    )

    state.register_theory(
        TheoryCommitment(
            name="Clustering Theory",
            agent_name="Clustering_Agent",
            core_claims=SUSTAIN.core_claims,
            model_name="SUSTAIN",
            model_params={"r": 9.01, "beta": 1.252},
            auxiliary_assumptions=[
                "Cluster recruitment threshold is stable within an individual",
                "Presentation order affects cluster structure",
            ],
            term_glossary={
                "attention": "lambda_i: receptive field tuning per dimension",
                "similarity": "dimension-weighted activation, sharpened by r parameter",
                "learning": "error-driven cluster recruitment and weight update",
            },
        )
    )

    print(f"\nRegistered {len(state.theories)} theories.")
    print("\n--- Term glossary comparison: 'attention' ---")
    for t in state.theories:
        print(f"  {t.name}: {t.term_glossary.get('attention', 'not defined')}")
    print()

    # Propose an experiment
    exp = state.propose_experiment(
        proposed_by="Exemplar_Agent",
        title="5-4 Structure with Transfer Test",
        design_spec={
            "category_structure": "Medin & Schaffer (1978) 5-4",
            "training": "Supervised classification, 10 blocks of 9 items",
            "test": "Old items + novel transfer items at varying similarity",
            "conditions": ["standard training", "interleaved training"],
            "DVs": ["accuracy", "response_time", "generalization_gradient"],
        },
        rationale=(
            "The 5-4 structure has no valid prototype for either category. "
            "This dissociates exemplar models (which predict good performance) "
            "from prototype models."
        ),
    )

    # Add a critique — now returns an index
    critique_idx = state.add_critique(
        exp.experiment_id,
        agent_name="Clustering_Agent",
        critique=(
            "The 5-4 structure is well-known to favor exemplar models — this is "
            "not a novel test. More importantly, the proposed design ignores "
            "presentation order effects."
        ),
        quantitative_evidence={
            "sustain_random_order_acc": 0.78,
            "sustain_blocked_order_acc": 0.91,
            "gcm_order_independent": True,
        },
    )
    print(f"Critique added at index {critique_idx}")

    # Revise the proposal, linking to the specific critique
    state.revise_proposal(
        exp.experiment_id,
        revised_by="Exemplar_Agent",
        addresses_critiques=[critique_idx],
        changes="Added blocked-order condition to test SUSTAIN's order prediction",
        new_design_spec={
            **exp.design_spec,
            "conditions": ["standard", "interleaved", "blocked-by-substructure"],
        },
    )
    print(f"Proposal revised, addressing critique {critique_idx}")
    print(f"  Revision chain: {len(exp.revision_history)} revision(s) logged")

    # Register predictions
    state.register_prediction(
        experiment_id=exp.experiment_id,
        agent_name="Exemplar_Agent",
        model_name="GCM",
        model_params={"c": 3.5},
        predicted_pattern={
            "training_accuracy_final": 0.92,
            "transfer_near": 0.85,
            "transfer_far": 0.65,
            "order_effect": 0.02,
        },
    )

    state.register_prediction(
        experiment_id=exp.experiment_id,
        agent_name="Clustering_Agent",
        model_name="SUSTAIN",
        model_params={"r": 9.01, "beta": 1.252},
        predicted_pattern={
            "training_accuracy_final": 0.88,
            "transfer_near": 0.82,
            "transfer_far": 0.55,
            "order_effect": 0.15,
        },
    )

    # Simulate execution
    state.approve_experiment(
        exp.experiment_id, moderator_edits="Approved with blocked-order addition"
    )
    state.record_data(
        exp.experiment_id,
        {
            "training_accuracy_final": 0.89,
            "transfer_near": 0.81,
            "transfer_far": 0.58,
            "order_effect": 0.11,
        },
    )

    # Score predictions
    state.score_predictions(
        exp.experiment_id,
        {
            "training_accuracy_final": 0.89,
            "transfer_near": 0.81,
            "transfer_far": 0.58,
            "order_effect": 0.11,
        },
    )

    # Theory revision — progressive (generates new predictions)
    state.revise_theory(
        "Exemplar Theory",
        description="Increased sensitivity parameter to account for sharper transfer gradient",
        new_params={"c": 5.0},
        triggered_by_experiment=exp.experiment_id,
        new_predictions=[
            "Near-boundary items should now show < 65% accuracy",
            "Generalization gradient slope > 0.3 per similarity unit",
        ],
    )
    r = state.theories[0].revision_log[0]
    print(
        f"\nExemplar Theory revised: {r['revision_type']} "
        f"({len(r['new_predictions'])} new predictions)"
    )

    # Theory revision — degenerative (no new predictions)
    state.revise_theory(
        "Clustering Theory",
        description="Adjusted beta to accommodate observed order effect size",
        new_params={"beta": 1.5},
        triggered_by_experiment=exp.experiment_id,
        # No new_predictions → degenerative
    )
    r2 = state.theories[2].revision_log[0]
    print(f"Clustering Theory revised: {r2['revision_type']}")

    # Theory trajectories
    print("\n--- Theory Trajectories ---")
    for t in state.theories:
        traj = state.theory_trajectory(t.name)
        print(
            f"  {t.name}: {traj['trajectory']} "
            f"({traj['n_progressive']} progressive, {traj['n_degenerative']} degenerative)"
        )

    # Show summary
    print("\n--- Agent Summary: Exemplar_Agent ---")
    print(state.summary_for_agent("Exemplar_Agent"))

    # Show leaderboard
    board = state.prediction_leaderboard()
    print("--- Prediction Leaderboard ---")
    for agent, stats in sorted(
        board.items(), key=lambda x: x[1].get("mean_score", 999)
    ):
        print(
            f"  {agent}: RMSE = {stats['mean_score']:.4f} "
            f"({stats['n_predictions']} predictions)"
        )
    print()

    # Register a dispute
    state.register_dispute(
        claim="Presentation order has a meaningful effect on category representations",
        positions={
            "Exemplar_Agent": "Order is irrelevant — all exemplars are stored regardless of order.",
            "Clustering_Agent": "Order determines which clusters are recruited, changing representations.",
            "Rule_Agent": "Order may affect which rule is discovered first, but not the final rule.",
        },
    )

    print(f"Open disputes: {len(state.open_disputes())}")
    for d in state.open_disputes():
        print(f"  '{d.claim}'")
        for agent, pos in d.positions.items():
            print(f"    {agent}: {pos[:80]}...")

    # Save state
    out_path = os.path.join(os.path.dirname(__file__), "demo_state.json")
    state.to_json(out_path)
    print(f"\nEpistemic state saved to {out_path}")


def demo_full_cycle():
    """Demonstrate a complete debate cycle (without LLM calls)."""
    print("\n" + "=" * 70)
    print("PHASE 4: FULL CYCLE WALKTHROUGH")
    print("=" * 70)

    state = EpistemicState(domain="Human Categorization")
    agents = default_agent_configs()
    protocol = DebateProtocol(state, agents)

    # Walk through phases
    for phase in Phase:
        spec = protocol.phase_spec(phase)
        print(f"\n--- {phase.name} ---")
        print(f"  Goal: {spec['goal'][:120]}...")
        print(f"  Required output: {spec['required_output']}")
        if spec.get("max_rounds"):
            print(f"  Max rounds: {spec['max_rounds']}")

    print("\n\nThe protocol defines 9 phases per cycle.")
    print("After AUDIT, the cycle loops back to DIVERGENCE_MAPPING with updated state.")
    print("Each cycle should narrow the space of live disputes and accumulate")
    print("established facts in the epistemic state tracker.")


if __name__ == "__main__":
    demo_model_predictions()
    demo_divergence_mapping()
    demo_epistemic_state()
    demo_full_cycle()
