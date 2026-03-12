"""
Minimal LLM Runner for Antagonistic Collaboration.

No AutoGen, no frameworks. Raw API calls to Claude, a loop over phases,
and you as the human moderator typing into a terminal.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m antagonistic_collab.runner

The runner walks through the 9-phase debate protocol. At each phase it:
1. Shows you what's happening
2. Calls the LLM for each agent (or prompts you at human phases)
3. Parses structured output
4. Updates the epistemic state
5. Advances to the next phase

The full transcript is saved to debate_transcript.json after each phase.
"""

import os
import sys
import json
import re
from typing import Optional

# Try to import anthropic; give clear error if missing
try:
    import anthropic
except ImportError:
    print("Install the anthropic SDK: pip install anthropic")
    sys.exit(1)

from .epistemic_state import EpistemicState, TheoryCommitment
from .debate_protocol import (
    DebateProtocol, Phase, PhaseResult, default_agent_configs,
)


# ---------------------------------------------------------------------------
# Core LLM call
# ---------------------------------------------------------------------------

def call_agent(
    client: anthropic.Anthropic,
    system_prompt: str,
    user_message: str,
    model: str = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> str:
    """Single LLM call. Returns the text response."""
    response = client.messages.create(
        model=model or _LLM_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def extract_json(text: str) -> Optional[dict]:
    """Extract the first JSON block from LLM output."""
    results = extract_all_json(text)
    return results[0] if results else None


def extract_all_json(text: str) -> list[dict]:
    """Extract all JSON blocks from LLM output."""
    results = []
    # Try ```json ... ``` blocks first
    for match in re.finditer(r'```json\s*\n?(.*?)\n?\s*```', text, re.DOTALL):
        try:
            results.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            pass
    if results:
        return results
    # Find top-level JSON objects by brace-depth counting
    i = 0
    while i < len(text):
        if text[i] == '{':
            depth = 0
            in_string = False
            escape_next = False
            for j in range(i, len(text)):
                ch = text[j]
                if escape_next:
                    escape_next = False
                    continue
                if in_string:
                    if ch == '\\':
                        escape_next = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            results.append(json.loads(text[i:j + 1]))
                        except json.JSONDecodeError:
                            pass
                        i = j + 1
                        break
            else:
                i += 1
        else:
            i += 1
    return results


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def run_commitment(protocol: DebateProtocol, client: anthropic.Anthropic,
                   transcript: list) -> PhaseResult:
    """Phase 1: Each agent registers its theory."""
    spec = protocol.phase_spec(Phase.COMMITMENT)
    messages = []
    
    print("\n" + "=" * 70)
    print("PHASE: COMMITMENT — Agents register theories")
    print("=" * 70)
    
    for agent in protocol.agent_configs:
        context = spec["context"].replace("{agent_name}", agent.name)
        prompt = (
            f"PHASE: Theory Commitment\n\n"
            f"GOAL: {spec['goal']}\n\n"
            f"CURRENT STATE:\n{context}\n\n"
            f"Register your theory now. Include:\n"
            f"1. Your core theoretical claims\n"
            f"2. Your model's key parameters and their plausible ranges\n"
            f"3. Your auxiliary assumptions\n"
            f"4. A term glossary: map your key theoretical terms to specific "
            f"model parameters (e.g., 'attention' → 'w_i, dimensional weights')\n\n"
            f"Output a JSON block with these fields."
        )
        
        print(f"\n--- {agent.name} registering theory ---")
        response = call_agent(client, agent.system_prompt, prompt)
        print(response[:500] + "..." if len(response) > 500 else response)
        
        # Register the theory (use the agent config's known info)
        glossary = {}
        json_block = extract_json(response)
        if json_block:
            glossary = json_block.get("term_glossary", json_block.get("glossary", {}))
        
        try:
            protocol.state.register_theory(TheoryCommitment(
                name=agent.theory_name,
                agent_name=agent.name,
                core_claims=agent.model_class.core_claims,
                model_name=agent.model_class.name,
                model_params=agent.default_params,
                term_glossary=glossary,
            ))
        except ValueError:
            pass  # Already registered (e.g., re-running)
        
        messages.append({
            "agent": agent.name, "phase": "COMMITMENT",
            "response": response, "parsed_json": json_block,
        })
    
    transcript.extend(messages)
    return PhaseResult(phase=Phase.COMMITMENT, cycle=protocol.state.cycle,
                       outputs={"theories_registered": len(protocol.state.theories)},
                       messages=messages)


def run_divergence_mapping(protocol: DebateProtocol, client: anthropic.Anthropic,
                           transcript: list) -> PhaseResult:
    """Phase 2: Compute and discuss where models disagree."""
    print("\n" + "=" * 70)
    print("PHASE: DIVERGENCE MAPPING — Where do models disagree?")
    print("=" * 70)
    
    div_map = protocol.compute_divergence_map()
    div_context = protocol._divergence_context(div_map=div_map)
    print(div_context)
    
    messages = []
    # Each agent interprets the divergence map
    for agent in protocol.agent_configs:
        prompt = (
            f"PHASE: Divergence Mapping\n\n"
            f"The following divergence map was computed by running all registered "
            f"models on standard category structures:\n\n{div_context}\n\n"
            f"From YOUR theoretical perspective ({agent.theory_name}):\n"
            f"1. Which regions of maximum divergence are most interesting and why?\n"
            f"2. Where does your model perform best relative to competitors?\n"
            f"3. What category structures should we focus on in the next phase?"
        )
        
        print(f"\n--- {agent.name} interprets divergence ---")
        response = call_agent(client, agent.system_prompt, prompt)
        print(response[:600] + "..." if len(response) > 600 else response)
        messages.append({
            "agent": agent.name, "phase": "DIVERGENCE_MAPPING",
            "response": response,
        })
    
    transcript.extend(messages)
    return PhaseResult(phase=Phase.DIVERGENCE_MAPPING, cycle=protocol.state.cycle,
                       outputs={"divergence_map": _serialize_div_map(div_map)},
                       messages=messages)


def run_experiment_proposal(protocol: DebateProtocol, client: anthropic.Anthropic,
                            transcript: list) -> PhaseResult:
    """Phase 3: Each agent proposes an experiment."""
    spec = protocol.phase_spec(Phase.EXPERIMENT_PROPOSAL)
    messages = []
    
    print("\n" + "=" * 70)
    print("PHASE: EXPERIMENT PROPOSAL — Each agent proposes a design")
    print("=" * 70)
    
    for agent in protocol.agent_configs:
        context = protocol.state.summary_for_agent(agent.name)
        prompt = (
            f"PHASE: Experiment Proposal\n\n"
            f"GOAL: {spec['goal']}\n\n"
            f"CURRENT STATE:\n{context}\n\n"
            f"Propose an experiment. You MUST output a JSON block with:\n"
            f'{{"title": "...", "design": "between|within|mixed", '
            f'"category_structure": {{...}}, "conditions": [...], '
            f'"dependent_variables": [...], "n_subjects_recommended": N, '
            f'"prediction_if_supports_me": "...", '
            f'"prediction_if_challenges_me": "...", "rationale": "..."}}'
        )
        
        print(f"\n--- {agent.name} proposes ---")
        response = call_agent(client, agent.system_prompt, prompt)
        print(response[:800] + "..." if len(response) > 800 else response)
        
        json_block = extract_json(response) or {}
        
        # Register in epistemic state
        exp = protocol.state.propose_experiment(
            proposed_by=agent.name,
            title=json_block.get("title", f"Proposal by {agent.name}"),
            design_spec=json_block,
            rationale=json_block.get("rationale", response[:200]),
        )
        
        messages.append({
            "agent": agent.name, "phase": "EXPERIMENT_PROPOSAL",
            "response": response, "parsed_json": json_block,
            "experiment_id": exp.experiment_id,
        })
    
    transcript.extend(messages)
    return PhaseResult(phase=Phase.EXPERIMENT_PROPOSAL, cycle=protocol.state.cycle,
                       outputs={"proposals": [m.get("experiment_id") for m in messages]},
                       messages=messages)


def run_adversarial_critique(protocol: DebateProtocol, client: anthropic.Anthropic,
                             transcript: list, n_rounds: int = 2) -> PhaseResult:
    """Phase 4: Agents critique each other's proposals."""
    spec = protocol.phase_spec(Phase.ADVERSARIAL_CRITIQUE)
    proposals_context = protocol._proposals_context()
    messages = []
    
    print("\n" + "=" * 70)
    print("PHASE: ADVERSARIAL CRITIQUE — Agents attack proposals")
    print("=" * 70)
    
    current_proposals = [
        e for e in protocol.state.experiments
        if e.cycle == protocol.state.cycle and e.status == "proposed"
    ]
    
    for round_num in range(n_rounds):
        print(f"\n--- Critique round {round_num + 1} ---")
        
        for agent in protocol.agent_configs:
            # Each agent critiques proposals by OTHER agents
            other_proposals = [p for p in current_proposals if p.proposed_by != agent.name]
            if not other_proposals:
                continue
            
            # Build critique context including any prior critiques this round
            critique_history = ""
            for p in current_proposals:
                if p.critique_log:
                    critique_history += f"\nPrior critiques of '{p.title}':\n"
                    for c in p.critique_log:
                        critique_history += f"  {c['agent']}: {c['critique'][:200]}\n"
            
            prompt = (
                f"PHASE: Adversarial Critique (round {round_num + 1})\n\n"
                f"GOAL: {spec['goal']}\n\n"
                f"PROPOSALS TO CRITIQUE:\n{proposals_context}\n\n"
                f"{critique_history}\n\n"
                f"Critique the proposals by other agents. For each critique you MUST:\n"
                f"1. State which proposal you are critiquing (by title)\n"
                f"2. Make a specific, falsifiable objection\n"
                f"3. Back it up: describe what your model predicts under the "
                f"proposed conditions and why that undermines the proposal's "
                f"diagnostic value\n\n"
                f"Output a JSON block for each critique:\n"
                f'{{"target_proposal": "...", "critique": "...", '
                f'"model_evidence": {{"model_called": "...", "conditions": {{...}}, '
                f'"prediction": {{...}}, "interpretation": "..."}}}}'
            )
            
            print(f"\n  {agent.name} critiques:")
            response = call_agent(client, agent.system_prompt, prompt)
            print(response[:600] + "..." if len(response) > 600 else response)
            
            # Register critiques — match each to its targeted proposal
            json_blocks = extract_all_json(response)
            if json_blocks:
                for block in json_blocks:
                    target_title = block.get("target_proposal", "")
                    matched = None
                    for p in other_proposals:
                        if target_title and target_title.lower() in p.title.lower():
                            matched = p
                            break
                    if matched is None and len(other_proposals) == 1:
                        # Only one other proposal — unambiguous target
                        matched = other_proposals[0]
                    if matched is not None:
                        protocol.state.add_critique(
                            matched.experiment_id, agent.name,
                            block.get("critique", response[:500]),
                            quantitative_evidence=block.get("model_evidence"),
                        )
            else:
                # No JSON parsed — attach full response to first other proposal
                if other_proposals:
                    protocol.state.add_critique(
                        other_proposals[0].experiment_id, agent.name,
                        response[:500],
                    )
            
            messages.append({
                "agent": agent.name, "phase": "ADVERSARIAL_CRITIQUE",
                "round": round_num + 1, "response": response,
                "parsed_json": json_blocks,
            })
    
    transcript.extend(messages)
    return PhaseResult(phase=Phase.ADVERSARIAL_CRITIQUE, cycle=protocol.state.cycle,
                       outputs={"n_critiques": sum(len(p.critique_log) for p in current_proposals)},
                       messages=messages)


def run_human_arbitration(protocol: DebateProtocol, transcript: list) -> PhaseResult:
    """Phase 6: Human moderator reviews and decides."""
    print("\n" + "=" * 70)
    print("PHASE: HUMAN ARBITRATION — Your turn")
    print("=" * 70)
    
    # Show full context
    context = protocol._full_round_context()
    print(context)
    
    current_proposals = [
        e for e in protocol.state.experiments
        if e.cycle == protocol.state.cycle and e.status == "proposed"
    ]
    
    print("\nProposals on the table:")
    for i, p in enumerate(current_proposals):
        n_critiques = len(p.critique_log)
        print(f"  [{i}] {p.title} (by {p.proposed_by}, {n_critiques} critiques)")
    
    if _BATCH_MODE:
        # Auto-approve: pick proposal with most critiques addressed
        # (heuristic: more critique = more refined)
        choice = "approve 0"
        print(f"\n[BATCH MODE] Auto-selecting: {choice}")
    else:
        print("\nOptions:")
        print("  approve <N>          — Approve proposal N as-is")
        print("  approve <N> <edits>  — Approve with moderator edits")
        print("  reject               — Reject all, ask for new round")
        print("  skip                 — Skip to execution with first proposal")
        
        choice = input("\nModerator> ").strip()
    
    messages = [{"agent": "MODERATOR", "phase": "HUMAN_ARBITRATION", "input": choice}]
    
    if choice.startswith("approve"):
        parts = choice.split(maxsplit=2)
        try:
            idx = int(parts[1]) if len(parts) > 1 else 0
        except ValueError:
            print(f"\n✗ Invalid index: '{parts[1]}'. Expected a number.")
            idx = None
        if idx is not None and not (0 <= idx < len(current_proposals)):
            print(f"\n✗ Index {idx} out of range (0–{len(current_proposals) - 1}).")
            idx = None
        if idx is not None:
            edits = parts[2] if len(parts) > 2 else None
            selected = current_proposals[idx]
            protocol.state.approve_experiment(selected.experiment_id, moderator_edits=edits)
            print(f"\n✓ Approved: {selected.title}" + (f" (edits: {edits})" if edits else ""))
            messages[0]["approved"] = selected.experiment_id
    elif choice == "skip":
        if current_proposals:
            protocol.state.approve_experiment(current_proposals[0].experiment_id)
            print(f"\n✓ Auto-approved: {current_proposals[0].title}")
    else:
        print("\n✗ All proposals rejected. (In full version, this loops back.)")
    
    transcript.extend(messages)
    return PhaseResult(phase=Phase.HUMAN_ARBITRATION, cycle=protocol.state.cycle,
                       outputs={"moderator_choice": choice}, messages=messages)


def run_execution(protocol: DebateProtocol, client: anthropic.Anthropic,
                  transcript: list, true_model: str = "GCM") -> PhaseResult:
    """Phase 7: Register predictions, then run experiment."""
    protocol.phase_spec(Phase.EXECUTION)
    messages = []
    
    print("\n" + "=" * 70)
    print("PHASE: EXECUTION — Predictions then data")
    print("=" * 70)
    
    approved = [e for e in protocol.state.experiments
                if e.cycle == protocol.state.cycle and e.status == "approved"]
    if not approved:
        print("No approved experiments. Skipping.")
        return PhaseResult(phase=Phase.EXECUTION, cycle=protocol.state.cycle,
                           outputs={}, messages=[])
    
    exp = approved[0]
    context = protocol._approved_experiment_context()
    
    # Each agent registers predictions BEFORE seeing data
    print("\n--- Agents register predictions ---")
    for agent in protocol.agent_configs:
        prompt = (
            f"PHASE: Pre-data Prediction Registration\n\n"
            f"{context}\n\n"
            f"You MUST register a quantitative prediction for the approved "
            f"experiment BEFORE seeing any data. This prediction will be "
            f"scored against the actual results.\n\n"
            f"Output a JSON block:\n"
            f'{{"predicted_pattern": {{"metric_name": value, ...}}, '
            f'"confidence": "high|medium|low", '
            f'"reasoning": "..."}}'
        )
        
        print(f"\n  {agent.name} predicts:")
        response = call_agent(client, agent.system_prompt, prompt)
        print(response[:400] + "..." if len(response) > 400 else response)
        
        json_block = extract_json(response) or {}
        predicted = json_block.get("predicted_pattern", {})
        
        protocol.state.register_prediction(
            experiment_id=exp.experiment_id,
            agent_name=agent.name,
            model_name=agent.model_class.name,
            model_params=agent.default_params,
            predicted_pattern=predicted,
        )
        
        messages.append({
            "agent": agent.name, "phase": "EXECUTION_PREDICT",
            "response": response, "predicted": predicted,
        })
    
    # Run the experiment (synthetic)
    print(f"\n--- Running experiment (synthetic, true_model={true_model}) ---")
    data = protocol.experiment_runner(exp.design_spec, true_model=true_model)
    protocol.state.record_data(exp.experiment_id, data)
    
    mean_acc = data.get('mean_accuracy')
    if isinstance(mean_acc, (int, float)):
        print(f"Results: mean_accuracy={mean_acc:.3f}")
    else:
        print("Results: mean_accuracy=N/A")
    
    # Score predictions
    actual = {k: v for k, v in data.items() if isinstance(v, (int, float))}
    if actual:
        protocol.state.score_predictions(exp.experiment_id, actual)
        board = protocol.state.prediction_leaderboard()
        print("\nPrediction scores:")
        for agent_name, stats in board.items():
            print(f"  {agent_name}: RMSE = {stats['mean_score']:.4f}")
    
    messages.append({
        "agent": "SYSTEM", "phase": "EXECUTION_DATA",
        "data_summary": {k: v for k, v in data.items() if k != "model_predictions"},
    })
    
    transcript.extend(messages)
    return PhaseResult(phase=Phase.EXECUTION, cycle=protocol.state.cycle,
                       outputs={"data": data}, messages=messages)


def run_interpretation(protocol: DebateProtocol, client: anthropic.Anthropic,
                       transcript: list) -> PhaseResult:
    """Phase 8: Agents interpret results."""
    spec = protocol.phase_spec(Phase.INTERPRETATION)
    results_context = protocol._results_context()
    messages = []
    
    print("\n" + "=" * 70)
    print("PHASE: INTERPRETATION — Agents respond to data")
    print("=" * 70)
    
    executed = [e for e in protocol.state.experiments
                if e.cycle == protocol.state.cycle and e.status == "executed"]
    if not executed:
        return PhaseResult(phase=Phase.INTERPRETATION, cycle=protocol.state.cycle,
                           outputs={}, messages=[])
    
    exp = executed[0]
    
    for agent in protocol.agent_configs:
        prompt = (
            f"PHASE: Interpretation\n\n"
            f"GOAL: {spec['goal']}\n\n"
            f"RESULTS:\n{results_context}\n\n"
            f"Respond with:\n"
            f"1. How well your prediction matched the data\n"
            f"2. Whether this supports, challenges, or is neutral for your theory\n"
            f"3. If you want to revise your theory, specify:\n"
            f'   {{"revision": true, "description": "...", '
            f'"new_predictions": ["testable prediction 1", ...]}}\n'
            f"   If the revision does NOT generate new predictions, it is "
            f"   classified as degenerative (Lakatos). Be explicit.\n"
            f"4. What experiment should come next"
        )
        
        print(f"\n--- {agent.name} interprets ---")
        response = call_agent(client, agent.system_prompt, prompt)
        print(response[:600] + "..." if len(response) > 600 else response)
        
        protocol.state.add_interpretation(exp.experiment_id, agent.name, response)
        
        # Check for theory revision
        json_block = extract_json(response)
        if json_block and json_block.get("revision"):
            protocol.state.revise_theory(
                agent.theory_name,
                description=json_block.get("description", "Post-data revision"),
                triggered_by_experiment=exp.experiment_id,
                new_predictions=json_block.get("new_predictions", []),
            )
            rev_type = "progressive" if json_block.get("new_predictions") else "degenerative"
            print(f"  → Theory revised ({rev_type})")
        
        messages.append({
            "agent": agent.name, "phase": "INTERPRETATION",
            "response": response, "parsed_json": json_block,
        })
    
    transcript.extend(messages)
    return PhaseResult(phase=Phase.INTERPRETATION, cycle=protocol.state.cycle,
                       outputs={}, messages=messages)


def run_audit(protocol: DebateProtocol, client: anthropic.Anthropic,
              transcript: list) -> PhaseResult:
    """Phase 9: Summarize what was learned."""
    print("\n" + "=" * 70)
    print("PHASE: AUDIT — Cycle summary")
    print("=" * 70)
    
    summary = protocol.state.summary_for_agent("SYSTEM")
    
    prompt = (
        f"PHASE: Audit\n\n"
        f"You are the system auditor. Summarize this cycle:\n\n"
        f"{summary}\n\n"
        f"Produce a structured summary:\n"
        f"1. What was established this cycle?\n"
        f"2. Which predictions were accurate?\n"
        f"3. Were any theories revised? Progressive or degenerative?\n"
        f"4. What disputes remain open?\n"
        f"5. What should the next cycle focus on?\n"
        f"6. CONVERGENCE CHECK: Are agents' proposals becoming too similar? "
        f"Is adversarial pressure collapsing?"
    )
    
    response = call_agent(
        client,
        "You are a neutral scientific auditor. Be concise and specific.",
        prompt,
    )
    print(response)
    
    messages = [{"agent": "AUDITOR", "phase": "AUDIT", "response": response}]
    transcript.extend(messages)
    
    return PhaseResult(phase=Phase.AUDIT, cycle=protocol.state.cycle,
                       outputs={"audit": response}, messages=messages)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_cycle(
    protocol: DebateProtocol,
    client: anthropic.Anthropic,
    transcript: list,
    true_model: str = "GCM",
    critique_rounds: int = 2,
    output_dir: str = ".",
):
    """Run one full debate cycle through all 9 phases."""
    
    print(f"\n{'#' * 70}")
    print(f"# CYCLE {protocol.state.cycle}")
    print(f"{'#' * 70}")
    
    # Phase 1: Commitment (only on first cycle)
    if protocol.state.cycle == 0:
        result = run_commitment(protocol, client, transcript)
        protocol.advance_phase(result)
    else:
        protocol.current_phase = Phase.DIVERGENCE_MAPPING
    
    # Phase 2: Divergence mapping
    result = run_divergence_mapping(protocol, client, transcript)
    protocol.advance_phase(result)
    
    # Phase 3: Experiment proposal
    result = run_experiment_proposal(protocol, client, transcript)
    protocol.advance_phase(result)
    
    # Phase 4: Adversarial critique
    result = run_adversarial_critique(protocol, client, transcript, n_rounds=critique_rounds)
    protocol.advance_phase(result)
    
    # Phase 5: Design revision (simplified — agents revise in critique rounds)
    protocol.advance_phase(PhaseResult(
        phase=Phase.DESIGN_REVISION, cycle=protocol.state.cycle, outputs={}))
    
    # Phase 6: Human arbitration
    result = run_human_arbitration(protocol, transcript)
    protocol.advance_phase(result)
    
    # Phase 7: Execution
    result = run_execution(protocol, client, transcript, true_model=true_model)
    protocol.advance_phase(result)
    
    # Phase 8: Interpretation
    result = run_interpretation(protocol, client, transcript)
    protocol.advance_phase(result)
    
    # Phase 9: Audit
    result = run_audit(protocol, client, transcript)

    # Save transcript BEFORE advancing (advance_phase increments the cycle counter)
    save_transcript(transcript, protocol, output_dir=output_dir)
    protocol.advance_phase(result)


def save_transcript(transcript: list, protocol: DebateProtocol, output_dir: str = "."):
    """Save full transcript and epistemic state."""
    os.makedirs(output_dir, exist_ok=True)
    output = {
        "cycle": protocol.state.cycle,
        "transcript": transcript,
    }
    path = os.path.join(output_dir, f"debate_cycle_{protocol.state.cycle}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nTranscript saved to {path}")

    state_path = os.path.join(output_dir, f"epistemic_state_cycle_{protocol.state.cycle}.json")
    protocol.state.to_json(state_path)
    print(f"Epistemic state saved to {state_path}")


def _serialize_div_map(div_map: dict) -> dict:
    """Make divergence map JSON-serializable."""
    result = {}
    for k, v in div_map.items():
        result[k] = {}
        for k2, v2 in v.items():
            if isinstance(v2, dict):
                result[k][k2] = {
                    k3: (v3.tolist() if hasattr(v3, 'tolist') else v3)
                    for k3, v3 in v2.items()
                }
            else:
                result[k][k2] = v2
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run antagonistic collaboration debate")
    parser.add_argument("--cycles", type=int, default=1, help="Number of debate cycles")
    parser.add_argument("--true-model", choices=["GCM", "SUSTAIN", "RULEX"], default="GCM",
                        help="Ground truth model for synthetic data")
    parser.add_argument("--batch", action="store_true",
                        help="Non-interactive mode: auto-approve first proposal (for Della/SLURM)")
    parser.add_argument("--model", default="claude-sonnet-4-20250514",
                        help="LLM model to use")
    parser.add_argument("--output-dir", default=".", help="Directory for output files")
    parser.add_argument("--critique-rounds", type=int, default=2,
                        help="Number of adversarial critique rounds per cycle")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY environment variable.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Initialize
    state = EpistemicState(domain="Human Categorization")
    agents = default_agent_configs()
    protocol = DebateProtocol(state, agents)
    transcript = []
    
    print("=" * 70)
    print("ANTAGONISTIC COLLABORATION: Human Categorization")
    print("=" * 70)
    print(f"Agents: {', '.join(a.name for a in agents)}")
    print(f"Models: {', '.join(a.model_class.name for a in agents)}")
    print(f"Mode: {'batch' if args.batch else 'interactive'}")
    print(f"LLM: {args.model}")
    print(f"Cycles: {args.cycles}, True model: {args.true_model}\n")
    
    # Patch call_agent to use the specified model
    global _LLM_MODEL
    _LLM_MODEL = args.model
    
    # Patch human arbitration for batch mode
    if args.batch:
        global _BATCH_MODE
        _BATCH_MODE = True
    
    output_dir = args.output_dir

    for cycle in range(args.cycles):
        run_cycle(protocol, client, transcript,
                  true_model=args.true_model,
                  critique_rounds=args.critique_rounds,
                  output_dir=output_dir)
    
    print("\n" + "=" * 70)
    print("DEBATE COMPLETE")
    print("=" * 70)
    
    # Final leaderboard
    board = protocol.state.prediction_leaderboard()
    if board:
        print("\nFinal Prediction Leaderboard:")
        for agent, stats in sorted(board.items(), key=lambda x: x[1].get("mean_score", 999)):
            print(f"  {agent}: mean RMSE = {stats['mean_score']:.4f} "
                  f"({stats['n_predictions']} predictions)")
    
    # Theory trajectories
    for t in protocol.state.active_theories():
        try:
            traj = protocol.state.theory_trajectory(t.name)
            print(f"\n{t.name}: trajectory = {traj['trajectory']} "
                  f"({traj['n_revisions']} revisions, "
                  f"{traj['n_progressive']} progressive)")
        except (ValueError, KeyError):
            pass
    
    if args.cycles > 0:
        last_cycle = protocol.state.cycle - 1
        print(f"\nFull transcript: {os.path.join(output_dir, f'debate_cycle_{last_cycle}.json')}")
        print(f"Epistemic state: {os.path.join(output_dir, f'epistemic_state_cycle_{last_cycle}.json')}")
    else:
        print("\nNo cycles were run.")


# Module-level flags for batch mode and model selection
_BATCH_MODE = False
_LLM_MODEL = "claude-sonnet-4-20250514"


if __name__ == "__main__":
    main()
