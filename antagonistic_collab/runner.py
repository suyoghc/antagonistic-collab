"""
Minimal LLM Runner for Antagonistic Collaboration.

No AutoGen, no frameworks. Raw API calls to Claude, a loop over phases,
and you as the human moderator typing into a terminal.

Usage:
    # Anthropic (default)
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m antagonistic_collab.runner

    # Princeton AI Sandbox (via Portkey gateway)
    export AI_SANDBOX_KEY=<your Portkey API key>
    python -m antagonistic_collab.runner --backend princeton --model gpt-4o

The runner walks through the 9-phase debate protocol. At each phase it:
1. Shows you what's happening
2. Calls the LLM for each agent (or prompts you at human phases)
3. Parses structured output
4. Updates the epistemic state
5. Advances to the next phase

The full transcript is saved to debate_transcript.json after each phase.
"""

import math
import os
import sys
import json
import re
import time
from typing import Optional

import numpy as np

# Lazy imports for LLM backends — at least one must be available.
try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]

try:
    import openai
except ImportError:
    openai = None  # type: ignore[assignment]

from .epistemic_state import EpistemicState, TheoryCommitment
from .debate_protocol import (
    DebateProtocol,
    Phase,
    PhaseResult,
    default_agent_configs,
    STRUCTURE_REGISTRY,
    STRUCTURE_DESCRIPTIONS,
    CONDITION_EFFECTS,
    extract_curve_features,
    validate_novel_structure,
)


# ---------------------------------------------------------------------------
# Core LLM call
# ---------------------------------------------------------------------------


def _is_openai_client(client) -> bool:
    """Return True if *client* quacks like an OpenAI / Azure OpenAI client."""
    return type(client).__name__ in ("OpenAI", "AzureOpenAI")


def call_agent(
    client,
    system_prompt: str,
    user_message: str,
    model: str = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    max_retries: int = 3,
) -> str:
    """Single LLM call with retry logic. Returns the text response.

    Dispatches automatically based on client type:
    - anthropic.Anthropic  → client.messages.create (system= kwarg)
    - openai.AzureOpenAI   → client.chat.completions.create (system message)

    Retries up to max_retries times on transient errors (network, rate limit).
    Empty-response errors (content filter) are NOT retried.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            if _is_openai_client(client):
                response = client.chat.completions.create(
                    model=model or _LLM_MODEL,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                )
                if not response.choices or not response.choices[0].message.content:
                    raise ValueError(
                        "Empty response from API — no choices returned. "
                        "This may indicate a content filter or an API error."
                    )
                return response.choices[0].message.content
            else:
                # Anthropic path (original)
                response = client.messages.create(
                    model=model or _LLM_MODEL,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )
                if not response.content:
                    raise ValueError(
                        "Empty response from API — no content blocks returned. "
                        "This may indicate a content filter or an API error."
                    )
                return response.content[0].text
        except ValueError:
            raise  # Don't retry empty responses (content filter)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = 2**attempt  # 1s, 2s, 4s
                print(f"  ⚠ API error (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)
    raise last_error


def extract_json(text: str) -> Optional[dict]:
    """Extract the first JSON block from LLM output."""
    results = extract_all_json(text)
    return results[0] if results else None


def extract_all_json(text: str) -> list[dict]:
    """Extract all JSON blocks from LLM output."""
    results = []
    # Try ```json ... ``` blocks first
    for match in re.finditer(r"```json\s*\n?(.*?)\n?\s*```", text, re.DOTALL):
        try:
            results.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            pass
    if results:
        return results
    # Find top-level JSON objects by brace-depth counting
    i = 0
    while i < len(text):
        if text[i] == "{":
            depth = 0
            in_string = False
            escape_next = False
            for j in range(i, len(text)):
                ch = text[j]
                if escape_next:
                    escape_next = False
                    continue
                if in_string:
                    if ch == "\\":
                        escape_next = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            results.append(json.loads(text[i : j + 1]))
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


def run_commitment(protocol: DebateProtocol, client, transcript: list) -> PhaseResult:
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
            protocol.state.register_theory(
                TheoryCommitment(
                    name=agent.theory_name,
                    agent_name=agent.name,
                    core_claims=agent.model_class.core_claims,
                    model_name=agent.model_class.name,
                    model_params=agent.default_params,
                    term_glossary=glossary,
                )
            )
        except ValueError:
            pass  # Already registered (e.g., re-running)

        messages.append(
            {
                "agent": agent.name,
                "phase": "COMMITMENT",
                "response": response,
                "parsed_json": json_block,
            }
        )

    transcript.extend(messages)
    return PhaseResult(
        phase=Phase.COMMITMENT,
        cycle=protocol.state.cycle,
        outputs={"theories_registered": len(protocol.state.theories)},
        messages=messages,
    )


def run_divergence_mapping(
    protocol: DebateProtocol, client, transcript: list
) -> PhaseResult:
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
        messages.append(
            {
                "agent": agent.name,
                "phase": "DIVERGENCE_MAPPING",
                "response": response,
            }
        )

    transcript.extend(messages)
    return PhaseResult(
        phase=Phase.DIVERGENCE_MAPPING,
        cycle=protocol.state.cycle,
        outputs={"divergence_map": _serialize_div_map(div_map)},
        messages=messages,
    )


def run_experiment_proposal(
    protocol: DebateProtocol, client, transcript: list
) -> PhaseResult:
    """Phase 3: Each agent proposes an experiment."""
    spec = protocol.phase_spec(Phase.EXPERIMENT_PROPOSAL)
    messages = []

    print("\n" + "=" * 70)
    print("PHASE: EXPERIMENT PROPOSAL — Each agent proposes a design")
    print("=" * 70)

    # Build structure menu for the prompt
    struct_menu = "\n".join(
        f"  - {name}: {STRUCTURE_DESCRIPTIONS.get(name, '')}"
        for name in STRUCTURE_REGISTRY
    )
    cond_menu = "\n".join(f"  - {name}" for name in CONDITION_EFFECTS)

    for agent in protocol.agent_configs:
        context = protocol.state.summary_for_agent(agent.name)
        prompt = (
            f"PHASE: Experiment Proposal\n\n"
            f"GOAL: {spec['goal']}\n\n"
            f"CURRENT STATE:\n{context}\n\n"
            f"AVAILABLE STRUCTURES (pick one by exact name):\n{struct_menu}\n\n"
            f"AVAILABLE CONDITIONS (pick one):\n{cond_menu}\n\n"
            f"CRITICAL: You MUST pick a structure_name from the list above. "
            f"Do NOT invent new structures — the experiment runner can only "
            f"execute structures from this menu.\n\n"
            f"Propose an experiment. You MUST output a JSON block with:\n"
            f'{{"title": "...", "design": "between|within|mixed", '
            f'"structure_name": "<exact name from menu>", '
            f'"condition": "<condition from menu>", '
            f'"n_subjects_recommended": N, '
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

        messages.append(
            {
                "agent": agent.name,
                "phase": "EXPERIMENT_PROPOSAL",
                "response": response,
                "parsed_json": json_block,
                "experiment_id": exp.experiment_id,
            }
        )

    transcript.extend(messages)
    return PhaseResult(
        phase=Phase.EXPERIMENT_PROPOSAL,
        cycle=protocol.state.cycle,
        outputs={"proposals": [m.get("experiment_id") for m in messages]},
        messages=messages,
    )


def run_adversarial_critique(
    protocol: DebateProtocol,
    client,
    transcript: list,
    n_rounds: int = 2,
) -> PhaseResult:
    """Phase 4: Agents critique each other's proposals."""
    spec = protocol.phase_spec(Phase.ADVERSARIAL_CRITIQUE)
    proposals_context = protocol._proposals_context()
    messages = []

    print("\n" + "=" * 70)
    print("PHASE: ADVERSARIAL CRITIQUE — Agents attack proposals")
    print("=" * 70)

    current_proposals = [
        e
        for e in protocol.state.experiments
        if e.cycle == protocol.state.cycle and e.status == "proposed"
    ]

    for round_num in range(n_rounds):
        print(f"\n--- Critique round {round_num + 1} ---")

        for agent in protocol.agent_configs:
            # Each agent critiques proposals by OTHER agents
            other_proposals = [
                p for p in current_proposals if p.proposed_by != agent.name
            ]
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
                            matched.experiment_id,
                            agent.name,
                            block.get("critique", response[:500]),
                            quantitative_evidence=block.get("model_evidence"),
                        )
            else:
                # No JSON parsed — attach full response to first other proposal
                if other_proposals:
                    protocol.state.add_critique(
                        other_proposals[0].experiment_id,
                        agent.name,
                        response[:500],
                    )

            messages.append(
                {
                    "agent": agent.name,
                    "phase": "ADVERSARIAL_CRITIQUE",
                    "round": round_num + 1,
                    "response": response,
                    "parsed_json": json_blocks,
                }
            )

    transcript.extend(messages)
    return PhaseResult(
        phase=Phase.ADVERSARIAL_CRITIQUE,
        cycle=protocol.state.cycle,
        outputs={"n_critiques": sum(len(p.critique_log) for p in current_proposals)},
        messages=messages,
    )


def run_design_revision(
    protocol: DebateProtocol, client, transcript: list
) -> PhaseResult:
    """Phase 5: Each agent revises their proposal in light of critiques."""
    spec = protocol.phase_spec(Phase.DESIGN_REVISION)
    messages = []

    print("\n" + "=" * 70)
    print("PHASE: DESIGN REVISION — Agents revise proposals")
    print("=" * 70)

    current_proposals = [
        e
        for e in protocol.state.experiments
        if e.cycle == protocol.state.cycle and e.status == "proposed"
    ]

    for proposal in current_proposals:
        if not proposal.critique_log:
            print(f"\n  {proposal.proposed_by}: no critiques, skipping revision")
            continue

        # Find the agent config for the proposer
        agent = next(
            (a for a in protocol.agent_configs if a.name == proposal.proposed_by),
            None,
        )
        if agent is None:
            continue

        # Build critique summary for this proposal
        critique_text = "\n".join(
            f"  - {c['agent']}: {c['critique']}" for c in proposal.critique_log
        )

        # Build structure menu
        struct_menu = "\n".join(
            f"  - {name}: {STRUCTURE_DESCRIPTIONS.get(name, '')}"
            for name in STRUCTURE_REGISTRY
        )

        prompt = (
            f"PHASE: Design Revision\n\n"
            f"GOAL: {spec['goal']}\n\n"
            f"YOUR ORIGINAL PROPOSAL:\n"
            f"  Title: {proposal.title}\n"
            f"  Design: {json.dumps(proposal.design_spec)}\n\n"
            f"CRITIQUES RECEIVED ({len(proposal.critique_log)}):\n"
            f"{critique_text}\n\n"
            f"AVAILABLE STRUCTURES:\n{struct_menu}\n\n"
            f"Revise your proposal to address the critiques. You MUST output "
            f"a JSON block with:\n"
            f'{{"structure_name": "<exact name from menu>", '
            f'"condition": "<condition>", '
            f'"changes": "<what you changed and why>", '
            f'"addresses_critiques": [<indices of critiques addressed, 0-based>]}}\n\n'
            f"If you believe your original proposal is strong despite the "
            f"critiques, you may keep the same structure_name but must still "
            f"explain why the critiques don't warrant changes."
        )

        print(f"\n  {agent.name} revises '{proposal.title}':")
        response = call_agent(client, agent.system_prompt, prompt)
        print(response[:600] + "..." if len(response) > 600 else response)

        # Parse the revision
        revision_json = extract_json(response) or {}
        new_structure = revision_json.get("structure_name", "")
        addresses = revision_json.get("addresses_critiques", [])
        # LLM may return a scalar (e.g. 1) instead of a list — coerce
        if not isinstance(addresses, list):
            addresses = [addresses]
        changes = revision_json.get("changes", "")

        if new_structure and addresses and new_structure in STRUCTURE_REGISTRY:
            # Validate critique indices
            valid_indices = [
                i
                for i in addresses
                if isinstance(i, int) and 0 <= i < len(proposal.critique_log)
            ]
            if not valid_indices:
                valid_indices = list(range(len(proposal.critique_log)))

            new_spec = dict(proposal.design_spec)
            new_spec["structure_name"] = new_structure
            if "condition" in revision_json:
                new_spec["condition"] = revision_json["condition"]

            protocol.state.revise_proposal(
                proposal.experiment_id,
                revised_by=agent.name,
                addresses_critiques=valid_indices,
                changes=changes or f"Changed to {new_structure}",
                new_design_spec=new_spec,
            )
            print(f"  → Revised: {proposal.design_spec.get('structure_name')}")
        else:
            print("  → No valid revision parsed, keeping original")

        messages.append(
            {
                "agent": agent.name,
                "phase": "DESIGN_REVISION",
                "response": response,
                "parsed_json": revision_json,
                "experiment_id": proposal.experiment_id,
            }
        )

    transcript.extend(messages)
    return PhaseResult(
        phase=Phase.DESIGN_REVISION,
        cycle=protocol.state.cycle,
        outputs={
            "n_revisions": sum(1 for p in current_proposals if p.revision_history)
        },
        messages=messages,
    )


def run_human_arbitration(protocol: DebateProtocol, transcript: list) -> PhaseResult:
    """Phase 6: Human moderator reviews and decides."""
    print("\n" + "=" * 70)
    print("PHASE: HUMAN ARBITRATION — Your turn")
    print("=" * 70)

    # Show full context
    context = protocol._full_round_context()
    print(context)

    current_proposals = [
        e
        for e in protocol.state.experiments
        if e.cycle == protocol.state.cycle and e.status == "proposed"
    ]

    print("\nProposals on the table:")
    for i, p in enumerate(current_proposals):
        n_critiques = len(p.critique_log)
        print(f"  [{i}] {p.title} (by {p.proposed_by}, {n_critiques} critiques)")

    if _BATCH_MODE:
        # Experiment selection strategy: Bayesian EIG or heuristic fallback
        selection_method = _SELECTION_METHOD

        if selection_method == "bayesian":
            # --- Bayesian information-gain selection ---
            from .bayesian_selection import ModelPosterior, select_experiment

            model_names = [a.name for a in protocol.agent_configs]
            if (
                protocol.state.model_posterior
                and "log_probs" in protocol.state.model_posterior
            ):
                posterior = ModelPosterior.from_dict(protocol.state.model_posterior)
            else:
                posterior = ModelPosterior.uniform(model_names)

            best, eig_scores = select_experiment(
                protocol,
                posterior,
                current_proposals,
                n_subjects=20,
                n_sim=200,
                seed=42,
            )
            best_struct = ""
            if isinstance(current_proposals[best].design_spec, dict):
                best_struct = current_proposals[best].design_spec.get(
                    "structure_name", ""
                )
            choice = f"approve {best}"

            # Print EIG ranking for transparency
            print("\n  EIG ranking:")
            ranked_by_eig = sorted(
                range(len(current_proposals)),
                key=lambda i: -eig_scores[i],
            )
            for rank, idx in enumerate(ranked_by_eig):
                ds = (
                    current_proposals[idx].design_spec
                    if isinstance(current_proposals[idx].design_spec, dict)
                    else {}
                )
                sn = ds.get("structure_name", "?")
                marker = " ← SELECTED" if idx == best else ""
                print(
                    f"    {rank + 1}. [{idx}] {sn} — EIG={eig_scores[idx]:.4f}{marker}"
                )

            print(
                f"\n[BATCH MODE] Auto-selecting: {choice} "
                f"(bayesian EIG: {best_struct} EIG={eig_scores[best]:.4f})"
            )

        else:
            # --- Heuristic fallback: divergence-driven with diversity penalty ---
            div_map = protocol.compute_divergence_map()

            struct_counts: dict[str, int] = {}
            pair_counts: dict[tuple[str, str], int] = {}
            for exp in protocol.state.experiments:
                if (
                    exp.status in ("executed", "approved")
                    and exp.cycle < protocol.state.cycle
                ):
                    ds = exp.design_spec if isinstance(exp.design_spec, dict) else {}
                    sn = ds.get("structure_name", "")
                    cond = ds.get("condition", "baseline")
                    if sn:
                        struct_counts[sn] = struct_counts.get(sn, 0) + 1
                        pair_counts[(sn, cond)] = pair_counts.get((sn, cond), 0) + 1

            def _effective_divergence(proposal) -> float:
                design = (
                    proposal.design_spec
                    if isinstance(proposal.design_spec, dict)
                    else {}
                )
                struct_name = design.get("structure_name", "")
                condition = design.get("condition", "baseline")
                if struct_name in div_map:
                    divs = div_map[struct_name].get("divergences", {})
                    raw_div = max(
                        (d["mean_abs_diff"] for d in divs.values()), default=0.0
                    )
                else:
                    raw_div = 0.0
                n_exact = pair_counts.get((struct_name, condition), 0)
                n_struct = struct_counts.get(struct_name, 0)
                n_other_cond = n_struct - n_exact
                penalty = (2**n_exact) * (1.5**n_other_cond)
                return raw_div / penalty

            ranked = sorted(
                range(len(current_proposals)),
                key=lambda i: (
                    -_effective_divergence(current_proposals[i]),
                    -len(current_proposals[i].critique_log),
                ),
            )
            best = ranked[0]
            best_struct = ""
            if isinstance(current_proposals[best].design_spec, dict):
                best_struct = current_proposals[best].design_spec.get(
                    "structure_name", ""
                )
            best_div = _effective_divergence(current_proposals[best])
            n_prior = struct_counts.get(best_struct, 0)
            choice = f"approve {best}"
            print(
                f"\n[BATCH MODE] Auto-selecting: {choice} "
                f"(heuristic: {best_struct} eff_div={best_div:.3f}"
                f"{f', struct tested {n_prior}x before' if n_prior else ''})"
            )
    else:
        print("\nOptions:")
        print("  approve <N>          — Approve proposal N as-is")
        print("  approve <N> <edits>  — Approve with moderator edits")
        print("  reject               — Reject all, ask for new round")
        print("  skip                 — Skip to execution with first proposal")

        try:
            choice = input("\nModerator> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[No input available — defaulting to skip]")
            choice = "skip"

    messages = [{"agent": "MODERATOR", "phase": "HUMAN_ARBITRATION", "input": choice}]
    rejected = False

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
            protocol.state.approve_experiment(
                selected.experiment_id, moderator_edits=edits
            )
            print(
                f"\n✓ Approved: {selected.title}"
                + (f" (edits: {edits})" if edits else "")
            )
            messages[0]["approved"] = selected.experiment_id
        else:
            # Invalid or out-of-range index — treat as rejection
            rejected = True
    elif choice == "skip":
        if current_proposals:
            protocol.state.approve_experiment(current_proposals[0].experiment_id)
            print(f"\n✓ Auto-approved: {current_proposals[0].title}")
    else:
        print("\n✗ All proposals rejected. Will loop back for new proposals.")
        rejected = True

    transcript.extend(messages)
    outputs = {"moderator_choice": choice}
    if rejected:
        outputs["rejected"] = True
    return PhaseResult(
        phase=Phase.HUMAN_ARBITRATION,
        cycle=protocol.state.cycle,
        outputs=outputs,
        messages=messages,
    )


def run_execution(
    protocol: DebateProtocol,
    client,
    transcript: list,
    true_model: str = "GCM",
) -> PhaseResult:
    """Phase 7: Register predictions, then run experiment."""
    protocol.phase_spec(Phase.EXECUTION)
    messages = []

    print("\n" + "=" * 70)
    print("PHASE: EXECUTION — Predictions then data")
    print("=" * 70)

    approved = [
        e
        for e in protocol.state.experiments
        if e.cycle == protocol.state.cycle and e.status == "approved"
    ]
    if not approved:
        print("No approved experiments. Skipping.")
        return PhaseResult(
            phase=Phase.EXECUTION, cycle=protocol.state.cycle, outputs={}, messages=[]
        )

    exp = approved[0]
    context = protocol._approved_experiment_context()

    # Resolve structure name and condition for model predictions
    design = exp.design_spec if isinstance(exp.design_spec, dict) else {}
    struct_name = design.get("structure_name", "")
    condition = design.get("condition", "baseline")

    # Each agent registers predictions BEFORE seeing data
    # The system runs each agent's model automatically; the LLM provides
    # reasoning and confidence only.
    print("\n--- Agents register predictions (model-computed) ---")
    for agent in protocol.agent_configs:
        prompt = (
            f"PHASE: Pre-data Prediction Registration\n\n"
            f"{context}\n\n"
            f"The system will automatically run your model ({agent.model_class.name}) "
            f"on the approved experiment structure to generate quantitative predictions. "
            f"You do NOT need to write out item-level numbers.\n\n"
            f"Your job is to provide:\n"
            f"  1. 'reasoning': Explain WHY your model predicts the pattern it does "
            f"for this structure and condition. What mechanisms drive the prediction?\n"
            f"  2. 'confidence': high / medium / low — how confident are you that "
            f"your model will fit this data well?\n"
            f"  3. 'param_overrides' (optional): Non-default parameters you want "
            f"to use (e.g., higher attention weight). If omitted, defaults are used.\n\n"
            f"Output a JSON block:\n"
            f'{{"reasoning": "...", '
            f'"confidence": "high|medium|low", '
            f'"param_overrides": {{}}}}'
        )

        print(f"\n  {agent.name} predicts:")
        response = call_agent(client, agent.system_prompt, prompt)
        print(response[:400] + "..." if len(response) > 400 else response)

        json_block = extract_json(response) or {}
        llm_reasoning = json_block.get("reasoning", "")
        llm_confidence = json_block.get("confidence", "medium")
        llm_param_overrides = json_block.get("param_overrides") or {}
        # Sanitize: only accept dict of scalar values
        if not isinstance(llm_param_overrides, dict):
            llm_param_overrides = {}

        # Compute predictions by running the agent's actual model
        predicted = protocol.compute_model_predictions(
            agent, struct_name, condition, param_overrides=llm_param_overrides
        )
        params_used = predicted.pop("params_used", agent.default_params)

        protocol.state.register_prediction(
            experiment_id=exp.experiment_id,
            agent_name=agent.name,
            model_name=agent.model_class.name,
            model_params=params_used,
            predicted_pattern=predicted,
        )

        mean_acc = predicted.get("mean_accuracy")
        if isinstance(mean_acc, (int, float)):
            print(f"    Model-computed: mean_accuracy={mean_acc:.3f}")
        else:
            print("    Model-computed: mean_accuracy=N/A")

        messages.append(
            {
                "agent": agent.name,
                "phase": "EXECUTION_PREDICT",
                "response": response,
                "llm_reasoning": llm_reasoning,
                "llm_confidence": llm_confidence,
                "predicted": predicted,
            }
        )

    # Run the experiment (synthetic)
    print(f"\n--- Running experiment (synthetic, true_model={true_model}) ---")
    data = protocol.experiment_runner(
        exp.design_spec, true_model=true_model, cycle=protocol.state.cycle
    )
    protocol.state.record_data(exp.experiment_id, data)

    mean_acc = data.get("mean_accuracy")
    if isinstance(mean_acc, (int, float)):
        print(f"Results: mean_accuracy={mean_acc:.3f}")
    else:
        print("Results: mean_accuracy=N/A")

    # Score predictions — include item-level accuracies for fine-grained scoring
    actual = {k: v for k, v in data.items() if isinstance(v, (int, float))}
    actual.update(data.get("item_accuracies", {}))
    if actual:
        protocol.state.score_predictions(exp.experiment_id, actual)
        board = protocol.state.prediction_leaderboard()
        print("\nPrediction scores:")
        for agent_name, stats in board.items():
            mean = stats.get("mean_score")
            if isinstance(mean, (int, float)) and not math.isnan(mean):
                print(f"  {agent_name}: RMSE = {mean:.4f}")
            else:
                print(f"  {agent_name}: not yet scored")

    # --- Learning curve computation ---
    learning_curves = None
    curve_features = {}
    if struct_name:
        learning_curves = protocol.compute_learning_curve_predictions(
            struct_name, condition
        )
        print("\n--- Learning curve comparison ---")
        for agent_name, curve in learning_curves.items():
            features = extract_curve_features(curve)
            curve_features[agent_name] = features
            print(
                f"  {agent_name}: pattern={features['learning_pattern']}, "
                f"final={features['final_accuracy']:.3f}, "
                f"max_jump={features['max_jump']:.3f}"
            )

        # Store curves on state for interpretation debate
        protocol.state.last_execution_curves = learning_curves

    # --- Bayesian posterior update ---
    if _BATCH_MODE and struct_name:
        from .bayesian_selection import ModelPosterior, update_posterior_from_experiment

        model_names = [a.name for a in protocol.agent_configs]
        if (
            protocol.state.model_posterior
            and "log_probs" in protocol.state.model_posterior
        ):
            posterior = ModelPosterior.from_dict(protocol.state.model_posterior)
        else:
            posterior = ModelPosterior.uniform(model_names)

        posterior = update_posterior_from_experiment(
            posterior,
            protocol,
            data,
            struct_name,
            condition,
            protocol.state.cycle,
            learning_curves=learning_curves,
        )
        protocol.state.model_posterior = posterior.to_dict()

        print("\n  Bayesian posterior update:")
        for i, name in enumerate(posterior.model_names):
            print(f"    P({name}) = {posterior.probs[i]:.4f}")
        print(
            f"    Entropy = {posterior.entropy:.4f} (max = {np.log(len(model_names)):.4f})"
        )

    messages.append(
        {
            "agent": "SYSTEM",
            "phase": "EXECUTION_DATA",
            "data_summary": {k: v for k, v in data.items() if k != "model_predictions"},
        }
    )

    transcript.extend(messages)
    return PhaseResult(
        phase=Phase.EXECUTION,
        cycle=protocol.state.cycle,
        outputs={
            "data": data,
            "learning_curves": learning_curves,
            "curve_features": curve_features,
        },
        messages=messages,
    )


def run_interpretation(
    protocol: DebateProtocol, client, transcript: list
) -> PhaseResult:
    """Phase 8: Agents interpret results."""
    spec = protocol.phase_spec(Phase.INTERPRETATION)
    results_context = protocol._results_context()
    messages = []

    print("\n" + "=" * 70)
    print("PHASE: INTERPRETATION — Agents respond to data")
    print("=" * 70)

    executed = [
        e
        for e in protocol.state.experiments
        if e.cycle == protocol.state.cycle and e.status == "executed"
    ]
    if not executed:
        return PhaseResult(
            phase=Phase.INTERPRETATION,
            cycle=protocol.state.cycle,
            outputs={},
            messages=[],
        )

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
            rev_type = (
                "progressive" if json_block.get("new_predictions") else "degenerative"
            )
            print(f"  → Theory revised ({rev_type})")

        messages.append(
            {
                "agent": agent.name,
                "phase": "INTERPRETATION",
                "response": response,
                "parsed_json": json_block,
            }
        )

    transcript.extend(messages)
    return PhaseResult(
        phase=Phase.INTERPRETATION,
        cycle=protocol.state.cycle,
        outputs={},
        messages=messages,
    )


def run_full_pool_selection(
    protocol: DebateProtocol,
    transcript: list,
) -> PhaseResult:
    """Full-pool Bayesian selection: compute EIG over all 55+ candidates.

    Replaces phases 3-6 (proposal, critique, revision, arbitration) in
    full_pool mode. No LLM calls — pure computation.
    """
    from .bayesian_selection import (
        ModelPosterior,
        generate_full_candidate_pool,
        select_from_pool,
    )

    print("\n" + "=" * 70)
    print("PHASE: FULL-POOL BAYESIAN SELECTION")
    print("=" * 70)

    model_names = [a.name for a in protocol.agent_configs]
    if protocol.state.model_posterior and "log_probs" in protocol.state.model_posterior:
        posterior = ModelPosterior.from_dict(protocol.state.model_posterior)
    else:
        posterior = ModelPosterior.uniform(model_names)

    # Collect extra structures from prior cycle's novel proposals
    extra = getattr(protocol, "temporary_structures", None) or {}

    pool = generate_full_candidate_pool(protocol, extra_structures=extra or None)
    print(f"  Evaluating {len(pool)} candidates...")

    best_idx, eig_scores = select_from_pool(
        protocol, posterior, pool, n_subjects=20, n_sim=200, seed=42
    )

    best_struct, best_cond = pool[best_idx]

    # Build top-5 EIG landscape
    ranked = sorted(range(len(pool)), key=lambda i: -eig_scores[i])
    eig_landscape = []
    for rank, idx in enumerate(ranked[:10]):
        s, c = pool[idx]
        entry = {
            "rank": rank + 1,
            "structure": s,
            "condition": c,
            "eig": eig_scores[idx],
        }
        eig_landscape.append(entry)
        marker = " ← SELECTED" if idx == best_idx else ""
        print(f"    {rank + 1}. {s} / {c} — EIG={eig_scores[idx]:.4f}{marker}")

    # Create and approve the winning experiment
    exp = protocol.state.propose_experiment(
        proposed_by="SYSTEM (full-pool EIG)",
        title=f"EIG-selected: {best_struct} / {best_cond}",
        design_spec={"structure_name": best_struct, "condition": best_cond},
        rationale=f"Highest EIG ({eig_scores[best_idx]:.4f}) among {len(pool)} candidates",
    )
    protocol.state.approve_experiment(exp.experiment_id)
    print(
        f"\n  ✓ Selected: {best_struct} / {best_cond} (EIG={eig_scores[best_idx]:.4f})"
    )

    messages = [
        {
            "agent": "SYSTEM",
            "phase": "FULL_POOL_SELECTION",
            "selected": {"structure": best_struct, "condition": best_cond},
            "eig": eig_scores[best_idx],
        }
    ]
    transcript.extend(messages)

    return PhaseResult(
        phase=Phase.HUMAN_ARBITRATION,  # reuse phase enum for compatibility
        cycle=protocol.state.cycle,
        outputs={
            "selected_structure": best_struct,
            "selected_condition": best_cond,
            "eig": eig_scores[best_idx],
            "eig_landscape": eig_landscape,
        },
        messages=messages,
    )


def run_interpretation_debate(
    protocol: DebateProtocol,
    client,
    transcript: list,
) -> PhaseResult:
    """Interpretation debate: agents interpret results and generate hypotheses.

    Replaces the simple run_interpretation() in full_pool mode.
    Agents see: results, posterior, EIG landscape, parameter stability.
    They produce: interpretation, confound flags, hypothesis, optional novel structure.
    """
    messages = []
    agent_hypotheses = []
    all_confounds = []

    print("\n" + "=" * 70)
    print("PHASE: INTERPRETATION DEBATE — Agents interpret & hypothesize")
    print("=" * 70)

    executed = [
        e
        for e in protocol.state.experiments
        if e.cycle == protocol.state.cycle and e.status == "executed"
    ]
    if not executed:
        return PhaseResult(
            phase=Phase.INTERPRETATION,
            cycle=protocol.state.cycle,
            outputs={"agent_hypotheses": [], "confounds": []},
            messages=[],
        )

    exp = executed[0]
    results_context = protocol._results_context()

    # Build extended context: posterior, EIG landscape, param stability
    extra_context_lines = []

    # Bayesian posterior
    if protocol.state.model_posterior and "log_probs" in protocol.state.model_posterior:
        import numpy as _np

        lp = _np.array(protocol.state.model_posterior["log_probs"])
        probs = _np.exp(lp - _np.max(lp))
        probs = probs / probs.sum()
        model_names = protocol.state.model_posterior.get("model_names", [])
        extra_context_lines.append("\n### Bayesian Posterior")
        for i, p in enumerate(probs):
            name = model_names[i] if i < len(model_names) else f"model_{i}"
            extra_context_lines.append(f"  P({name}) = {p:.4f}")

    # Parameter stability across cycles
    extra_context_lines.append("\n### Parameter Stability")
    for theory in protocol.state.active_theories():
        params = theory.model_params
        n_revisions = len(theory.revision_log)
        extra_context_lines.append(
            f"  {theory.agent_name}: params={params}, revisions={n_revisions}"
        )

    # Learning curve comparison (if available from execution)
    curves = getattr(protocol.state, "last_execution_curves", None)
    if curves:
        extra_context_lines.append("\n### Learning Curve Comparison")
        for agent_name, curve in curves.items():
            features = extract_curve_features(curve)
            extra_context_lines.append(
                f"  {agent_name}: pattern={features['learning_pattern']}, "
                f"final_accuracy={features['final_accuracy']:.3f}, "
                f"max_jump={features['max_jump']:.3f}, "
                f"onset_block={features['onset_block']}"
            )

    extended_context = results_context + "\n".join(extra_context_lines)

    for agent in protocol.agent_configs:
        prompt = (
            f"PHASE: Interpretation Debate\n\n"
            f"You are interpreting experimental results. Analyze the data and "
            f"produce a structured response.\n\n"
            f"RESULTS AND CONTEXT:\n{extended_context}\n\n"
            f"You MUST output a JSON block with:\n"
            f'{{"interpretation": "your analysis of what the results show", '
            f'"confounds_flagged": ["list any confounds or methodological issues"], '
            f'"hypothesis": "what experiment should come next and why", '
            f'"novel_structure": null or {{"name": "...", "stimuli": [[...]], "labels": [...]}} '
            f"if you want to propose a new category structure, "
            f'"revision": null or {{"description": "...", "new_predictions": [...]}} '
            f"if you want to revise your theory}}"
        )

        print(f"\n--- {agent.name} interprets ---")
        response = call_agent(client, agent.system_prompt, prompt)
        print(response[:600] + "..." if len(response) > 600 else response)

        protocol.state.add_interpretation(exp.experiment_id, agent.name, response)

        # Parse structured response
        json_block = extract_json(response) or {}

        hypothesis = json_block.get("hypothesis", "")
        confounds = json_block.get("confounds_flagged", [])
        novel_structure = json_block.get("novel_structure")

        if hypothesis:
            hyp_entry = {
                "agent": agent.name,
                "hypothesis": hypothesis,
                "cycle": protocol.state.cycle,
            }
            if novel_structure and isinstance(novel_structure, dict):
                hyp_entry["novel_structure"] = novel_structure
            agent_hypotheses.append(hyp_entry)

        # Validate and register novel structures
        if novel_structure and isinstance(novel_structure, dict):
            struct_name = novel_structure.get("name", "")
            is_valid, reason = validate_novel_structure(novel_structure)
            if is_valid and struct_name:
                protocol.temporary_structures[struct_name] = {
                    "stimuli": novel_structure["stimuli"],
                    "labels": novel_structure["labels"],
                }
                print(f"    Registered novel structure: {struct_name}")
            elif struct_name:
                print(f"    Rejected novel structure '{struct_name}': {reason}")

        if confounds:
            for c in confounds:
                all_confounds.append({"agent": agent.name, "confound": c})

        # Handle theory revision
        if json_block.get("revision") and isinstance(json_block["revision"], dict):
            rev = json_block["revision"]
            protocol.state.revise_theory(
                agent.theory_name,
                description=rev.get("description", "Post-data revision"),
                triggered_by_experiment=exp.experiment_id,
                new_predictions=rev.get("new_predictions", []),
            )

        messages.append(
            {
                "agent": agent.name,
                "phase": "INTERPRETATION_DEBATE",
                "response": response,
                "parsed_json": json_block,
            }
        )

    # Store hypotheses in epistemic state for next cycle
    protocol.state.agent_hypotheses.extend(agent_hypotheses)

    transcript.extend(messages)
    return PhaseResult(
        phase=Phase.INTERPRETATION,
        cycle=protocol.state.cycle,
        outputs={
            "agent_hypotheses": agent_hypotheses,
            "confounds": all_confounds,
        },
        messages=messages,
    )


def run_interpretation_critique(
    protocol: DebateProtocol,
    client,
    transcript: list,
) -> PhaseResult:
    """Interpretation critique: agents challenge each other's interpretations.

    Each agent sees all other agents' interpretations and must challenge at
    least one claim or confound flag.
    """
    messages = []

    print("\n" + "=" * 70)
    print("PHASE: INTERPRETATION CRITIQUE — Agents challenge interpretations")
    print("=" * 70)

    executed = [
        e
        for e in protocol.state.experiments
        if e.cycle == protocol.state.cycle and e.status == "executed"
    ]
    if not executed:
        return PhaseResult(
            phase=Phase.INTERPRETATION,
            cycle=protocol.state.cycle,
            outputs={},
            messages=[],
        )

    exp = executed[0]
    interpretations = exp.interpretations

    for agent in protocol.agent_configs:
        # Build context from other agents' interpretations
        other_interps = {
            name: interp
            for name, interp in interpretations.items()
            if name != agent.name
        }
        if not other_interps:
            continue

        interp_text = "\n".join(
            f"  {name}: {interp[:500]}" for name, interp in other_interps.items()
        )

        prompt = (
            f"PHASE: Interpretation Critique\n\n"
            f"Other agents' interpretations of the latest results:\n"
            f"{interp_text}\n\n"
            f"Challenge at least one interpretation. Be specific:\n"
            f"1. Which interpretation do you dispute and why?\n"
            f"2. What alternative explanation does your theory offer?\n"
            f"3. What evidence would distinguish your view from theirs?"
        )

        print(f"\n--- {agent.name} critiques interpretations ---")
        response = call_agent(client, agent.system_prompt, prompt)
        print(response[:600] + "..." if len(response) > 600 else response)

        messages.append(
            {
                "agent": agent.name,
                "phase": "INTERPRETATION_CRITIQUE",
                "response": response,
            }
        )

    transcript.extend(messages)
    return PhaseResult(
        phase=Phase.INTERPRETATION,
        cycle=protocol.state.cycle,
        outputs={"n_critiques": len(messages)},
        messages=messages,
    )


def run_audit(protocol: DebateProtocol, client, transcript: list) -> PhaseResult:
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

    return PhaseResult(
        phase=Phase.AUDIT,
        cycle=protocol.state.cycle,
        outputs={"audit": response},
        messages=messages,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_cycle(
    protocol: DebateProtocol,
    client,
    transcript: list,
    true_model: str = "GCM",
    critique_rounds: int = 2,
    output_dir: str = ".",
    metadata: Optional[dict] = None,
    mode: str = "legacy",
):
    """Run one full debate cycle.

    Args:
        mode: "full_pool" uses EIG over all candidates + interpretation debate.
              "legacy" (default) uses the original 9-phase flow with LLM proposals.
    """
    cycle_start = len(transcript)

    print(f"\n{'#' * 70}")
    print(f"# CYCLE {protocol.state.cycle} (mode={mode})")
    print(f"{'#' * 70}")

    # Phase 1: Commitment (only on first cycle)
    if protocol.state.cycle == 0:
        result = run_commitment(protocol, client, transcript)
        protocol.advance_phase(result)
    else:
        protocol.skip_to_phase(Phase.DIVERGENCE_MAPPING)

    # Phase 2: Divergence mapping
    result = run_divergence_mapping(protocol, client, transcript)
    protocol.advance_phase(result)

    if mode == "full_pool":
        # Full-pool mode: EIG selection replaces phases 3-6.
        # Skip the state machine forward to HUMAN_ARBITRATION so that
        # advance_phase correctly transitions to EXECUTION.
        protocol.skip_to_phase(Phase.HUMAN_ARBITRATION)
        result = run_full_pool_selection(protocol, transcript)
        protocol.advance_phase(result)
    else:
        # Legacy mode: Phases 3–6 may loop if the moderator rejects all proposals
        MAX_REJECT_RETRIES = 2  # up to 3 total attempts (initial + 2 retries)
        for attempt in range(1 + MAX_REJECT_RETRIES):
            if attempt > 0:
                print(
                    f"\n[RETRY {attempt}/{MAX_REJECT_RETRIES}] "
                    "Moderator rejected all proposals. Requesting new round."
                )
                for exp in protocol.state.experiments:
                    if exp.cycle == protocol.state.cycle and exp.status == "proposed":
                        exp.status = "rejected"
                protocol.skip_to_phase(Phase.EXPERIMENT_PROPOSAL)

            # Phase 3: Experiment proposal
            result = run_experiment_proposal(protocol, client, transcript)
            protocol.advance_phase(result)

            # Phase 4: Adversarial critique
            result = run_adversarial_critique(
                protocol, client, transcript, n_rounds=critique_rounds
            )
            protocol.advance_phase(result)

            # Phase 5: Design revision
            result = run_design_revision(protocol, client, transcript)
            protocol.advance_phase(result)

            # Phase 6: Human arbitration
            result = run_human_arbitration(protocol, transcript)
            protocol.advance_phase(result)

            if not result.outputs.get("rejected"):
                break
        else:
            print(
                f"\n[WARNING] All {1 + MAX_REJECT_RETRIES} proposal rounds rejected. "
                "Proceeding with no approved experiment."
            )

    # Phase 7: Execution
    result = run_execution(protocol, client, transcript, true_model=true_model)
    protocol.advance_phase(result)

    if mode == "full_pool":
        # Interpretation debate + critique replaces simple interpretation
        result = run_interpretation_debate(protocol, client, transcript)
        protocol.advance_phase(result)

        result = run_interpretation_critique(protocol, client, transcript)
        # Don't advance_phase here — we still need audit
    else:
        # Phase 8: Interpretation (legacy)
        result = run_interpretation(protocol, client, transcript)
        protocol.advance_phase(result)

    # Phase 9: Audit
    result = run_audit(protocol, client, transcript)

    # Save transcript BEFORE advancing (advance_phase increments the cycle counter)
    cycle_messages = transcript[cycle_start:]
    save_transcript(cycle_messages, protocol, output_dir=output_dir)
    save_cycle_markdown(
        cycle_messages,
        protocol,
        cycle_num=protocol.state.cycle,
        metadata=metadata or {},
        output_dir=output_dir,
    )
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

    state_path = os.path.join(
        output_dir, f"epistemic_state_cycle_{protocol.state.cycle}.json"
    )
    protocol.state.to_json(state_path)
    print(f"Epistemic state saved to {state_path}")


def save_cycle_markdown(
    cycle_messages: list,
    protocol: DebateProtocol,
    cycle_num: int,
    metadata: dict,
    output_dir: str = ".",
):
    """Write a human-readable Markdown transcript for a single cycle."""
    os.makedirs(output_dir, exist_ok=True)

    true_model = metadata.get("true_model", "unknown")
    llm_model = metadata.get("llm_model", "unknown")
    backend = metadata.get("backend", "unknown")

    lines = [
        f"# Cycle {cycle_num} Transcript",
        "",
        f"**True model**: {true_model} | **LLM**: {llm_model} | **Backend**: {backend}",
        "",
    ]

    # Map internal phase names to display names
    phase_display = {
        "COMMITMENT": "Commitment",
        "DIVERGENCE_MAPPING": "Divergence Mapping",
        "EXPERIMENT_PROPOSAL": "Experiment Proposal",
        "ADVERSARIAL_CRITIQUE": "Adversarial Critique",
        "HUMAN_ARBITRATION": "Human Arbitration",
        "EXECUTION_PREDICT": "Execution",
        "EXECUTION_DATA": "Execution",
        "INTERPRETATION": "Interpretation",
        "AUDIT": "Audit",
    }

    current_phase = None
    current_round = None

    for msg in cycle_messages:
        phase = msg.get("phase", "")
        display_phase = phase_display.get(phase, phase)
        agent = msg.get("agent", "")

        # Emit phase header on change (merge EXECUTION_PREDICT and EXECUTION_DATA)
        if display_phase != current_phase:
            current_phase = display_phase
            current_round = None
            lines.append(f"## Phase: {current_phase}")
            lines.append("")

        # Adversarial critique: show round sub-headers
        if phase == "ADVERSARIAL_CRITIQUE":
            round_num = msg.get("round")
            if round_num and round_num != current_round:
                current_round = round_num
                lines.append(f"### Round {round_num}")
                lines.append("")
            lines.append(f"#### {agent} critiques:")
            lines.append(msg.get("response", ""))
            lines.append("")
            continue

        # Human arbitration
        if phase == "HUMAN_ARBITRATION":
            choice = msg.get("input", "")
            lines.append(f"Moderator choice: {choice}")
            if msg.get("approved"):
                lines.append(f"Approved experiment: {msg['approved']}")
            lines.append("")
            continue

        # Execution predictions sub-section
        if phase == "EXECUTION_PREDICT":
            if not any("### Predictions" in line for line in lines):
                lines.append("### Predictions")
                lines.append("")
            lines.append(f"#### {agent}")
            predicted = msg.get("predicted", {})
            if predicted:
                lines.append(f"Predicted: {predicted}")
            lines.append(msg.get("response", ""))
            lines.append("")
            continue

        # Execution data sub-section
        if phase == "EXECUTION_DATA":
            lines.append("### Results")
            data_summary = msg.get("data_summary", {})
            for k, v in data_summary.items():
                if isinstance(v, (int, float)):
                    lines.append(
                        f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}"
                    )
            lines.append("")
            continue

        # Default: agent header + response
        lines.append(f"### {agent}")
        lines.append(msg.get("response", ""))
        lines.append("")

    path = os.path.join(output_dir, f"debate_cycle_{cycle_num}.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Markdown transcript saved to {path}")


def save_summary_report(
    transcript: list,
    protocol: DebateProtocol,
    n_cycles: int,
    metadata: dict,
    output_dir: str = ".",
):
    """Write an end-of-run summary.md with leaderboard, trajectories, and per-cycle highlights."""
    os.makedirs(output_dir, exist_ok=True)

    true_model = metadata.get("true_model", "unknown")
    llm_model = metadata.get("llm_model", "unknown")
    backend = metadata.get("backend", "unknown")
    agent_names = ", ".join(a.name for a in protocol.agent_configs)

    lines = [
        "# Debate Summary",
        "",
        f"**Domain**: {protocol.state.domain}",
        f"**True model**: {true_model}",
        f"**LLM**: {llm_model} ({backend})",
        f"**Cycles**: {n_cycles}",
        f"**Agents**: {agent_names}",
        "",
    ]

    # Prediction leaderboard
    lines.append("## Prediction Leaderboard")
    lines.append("")
    board = protocol.state.prediction_leaderboard()
    if board:
        lines.append("| Agent | Mean RMSE | N Predictions |")
        lines.append("|-------|-----------|---------------|")
        for agent, stats in sorted(
            board.items(), key=lambda x: x[1].get("mean_score", 999)
        ):
            mean = stats.get("mean_score")
            mean_str = (
                f"{mean:.4f}"
                if isinstance(mean, (int, float)) and not math.isnan(mean)
                else "N/A"
            )
            lines.append(f"| {agent} | {mean_str} | {stats['n_predictions']} |")
    else:
        lines.append("No predictions recorded.")
    lines.append("")

    # Theory trajectories
    lines.append("## Theory Trajectories")
    lines.append("")
    lines.append(
        "| Theory | Status | Revisions | Progressive | Degenerative | Trajectory |"
    )
    lines.append(
        "|--------|--------|-----------|-------------|--------------|------------|"
    )
    for t in protocol.state.active_theories():
        try:
            traj = protocol.state.theory_trajectory(t.name)
            lines.append(
                f"| {t.name} | {t.status} | {traj['n_revisions']} | "
                f"{traj['n_progressive']} | {traj['n_degenerative']} | "
                f"{traj['trajectory']} |"
            )
        except (ValueError, KeyError):
            lines.append(f"| {t.name} | {t.status} | - | - | - | - |")
    lines.append("")

    # Per-cycle summary
    lines.append("## Per-Cycle Summary")
    lines.append("")
    for cycle in range(n_cycles):
        lines.append(f"### Cycle {cycle}")
        # Find experiments for this cycle
        cycle_exps = [e for e in protocol.state.experiments if e.cycle == cycle]
        for exp in cycle_exps:
            if exp.status in ("approved", "executed"):
                lines.append(
                    f"- **Experiment approved**: {exp.title} (proposed by {exp.proposed_by})"
                )
                if exp.data:
                    mean_acc = exp.data.get("mean_accuracy")
                    if isinstance(mean_acc, (int, float)):
                        lines.append(f"- **Mean accuracy**: {mean_acc:.3f}")
        # Theory revisions this cycle
        for t in protocol.state.active_theories():
            for rev in t.revision_log:
                triggered = rev.get("triggered_by", "")
                if any(triggered == exp.experiment_id for exp in cycle_exps):
                    rev_type = rev.get("revision_type", "unknown")
                    lines.append(
                        f"- **Theory revision**: {t.name} revised ({rev_type})"
                    )
        lines.append("")

    # Experiments conducted
    executed_exps = [e for e in protocol.state.experiments if e.status == "executed"]
    if executed_exps:
        lines.append("## Experiments Conducted")
        lines.append("")
        for exp in executed_exps:
            lines.append(f"### {exp.title} (Cycle {exp.cycle})")
            lines.append(f"- Proposed by: {exp.proposed_by}")
            lines.append(f"- Critiques: {len(exp.critique_log)}")
            if exp.data:
                for k, v in exp.data.items():
                    if isinstance(v, (int, float)):
                        lines.append(
                            f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}"
                        )
            lines.append("")

    path = os.path.join(output_dir, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Summary report saved to {path}")


def auto_output_dir(
    true_model: str,
    llm_model: str,
    agent_configs: list,
    base_dir: str = ".",
) -> str:
    """Generate an auto-incremented output directory path.

    Format: {base_dir}/runs/True_{true_model}_LLM_{llm_model}_COLLAB_{agents}_{NN}/
    """
    # Build agent shortnames: "Exemplar_Agent" -> "Exemplar", etc.
    short_names = []
    for a in agent_configs:
        # Take everything before "_Agent" if that suffix exists
        name = a.name
        if name.endswith("_Agent"):
            name = name[: -len("_Agent")]
        short_names.append(name)
    agents_str = "-".join(short_names)

    prefix = f"True_{true_model}_LLM_{llm_model}_COLLAB_{agents_str}"
    runs_dir = os.path.join(base_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    # Find next run number
    existing = []
    if os.path.isdir(runs_dir):
        for d in os.listdir(runs_dir):
            if d.startswith(prefix + "_"):
                suffix = d[len(prefix) + 1 :]
                try:
                    existing.append(int(suffix))
                except ValueError:
                    pass
    next_num = max(existing, default=0) + 1
    dir_name = f"{prefix}_{next_num:02d}"
    return os.path.join(runs_dir, dir_name)


def _serialize_div_map(div_map: dict) -> dict:
    """Make divergence map JSON-serializable."""
    result = {}
    for k, v in div_map.items():
        result[k] = {}
        for k2, v2 in v.items():
            if isinstance(v2, dict):
                result[k][k2] = {
                    k3: (v3.tolist() if hasattr(v3, "tolist") else v3)
                    for k3, v3 in v2.items()
                }
            else:
                result[k][k2] = v2
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _create_client(backend: str = "anthropic"):
    """Create the LLM client for the chosen backend.

    Args:
        backend: ``"anthropic"`` or ``"princeton"``.

    Returns:
        An ``anthropic.Anthropic`` or ``openai.OpenAI`` instance.

    Raises:
        SystemExit: if the required SDK is not installed or the env var is
            missing.
    """
    if backend == "princeton":
        if openai is None:
            print("Install the openai SDK: pip install openai")
            sys.exit(1)
        api_key = os.environ.get("AI_SANDBOX_KEY")
        if not api_key:
            print("Set AI_SANDBOX_KEY environment variable.")
            print("  export AI_SANDBOX_KEY=<your Princeton AI Sandbox key>")
            sys.exit(1)
        return openai.OpenAI(
            api_key=api_key,
            base_url="https://api.portkey.ai/v1",
        )
    else:
        if anthropic is None:
            print("Install the anthropic SDK: pip install anthropic")
            sys.exit(1)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Set ANTHROPIC_API_KEY environment variable.")
            print("  export ANTHROPIC_API_KEY=sk-ant-...")
            sys.exit(1)
        return anthropic.Anthropic(api_key=api_key)


_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "princeton": "gpt-4o",
}


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run antagonistic collaboration debate"
    )
    parser.add_argument("--cycles", type=int, default=1, help="Number of debate cycles")
    parser.add_argument(
        "--true-model",
        choices=["GCM", "SUSTAIN", "RULEX"],
        default="GCM",
        help="Ground truth model for synthetic data",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Non-interactive mode: auto-approve first proposal (for Della/SLURM)",
    )
    parser.add_argument(
        "--backend",
        choices=["anthropic", "princeton"],
        default="anthropic",
        help="LLM backend: anthropic (default) or princeton (Azure OpenAI sandbox)",
    )
    parser.add_argument("--model", default=None, help="LLM model to use")
    parser.add_argument("--output-dir", default=".", help="Directory for output files")
    parser.add_argument(
        "--critique-rounds",
        type=int,
        default=2,
        help="Number of adversarial critique rounds per cycle",
    )
    parser.add_argument(
        "--selection",
        choices=["bayesian", "heuristic"],
        default="bayesian",
        help="Experiment selection method: bayesian (EIG) or heuristic (diversity penalty)",
    )
    parser.add_argument(
        "--mode",
        choices=["full_pool", "legacy"],
        default="legacy",
        help="Phase flow: full_pool (EIG + interpretation debate) or legacy (9-phase)",
    )
    args = parser.parse_args()

    # Resolve model default based on backend
    if args.model is None:
        args.model = _DEFAULT_MODELS[args.backend]

    client = _create_client(backend=args.backend)

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
    print(f"Backend: {args.backend}")
    print(f"LLM: {args.model}")
    print(
        f"Cycles: {args.cycles}, True model: {args.true_model}, "
        f"Selection: {args.selection}, Mode: {args.mode}\n"
    )

    # Patch call_agent to use the specified model
    global _LLM_MODEL
    _LLM_MODEL = args.model

    # Patch human arbitration for batch mode
    if args.batch:
        global _BATCH_MODE
        _BATCH_MODE = True

    global _SELECTION_METHOD
    _SELECTION_METHOD = args.selection

    # Auto-generate output directory if not explicitly set
    if args.output_dir == ".":
        output_dir = auto_output_dir(
            true_model=args.true_model,
            llm_model=args.model,
            agent_configs=agents,
        )
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    metadata = {
        "true_model": args.true_model,
        "llm_model": args.model,
        "backend": args.backend,
    }

    for cycle in range(args.cycles):
        run_cycle(
            protocol,
            client,
            transcript,
            true_model=args.true_model,
            critique_rounds=args.critique_rounds,
            output_dir=output_dir,
            metadata=metadata,
            mode=args.mode,
        )

    # End-of-run summary report
    if args.cycles > 0:
        save_summary_report(
            transcript,
            protocol,
            n_cycles=args.cycles,
            metadata=metadata,
            output_dir=output_dir,
        )

    print("\n" + "=" * 70)
    print("DEBATE COMPLETE")
    print("=" * 70)

    # Final leaderboard
    board = protocol.state.prediction_leaderboard()
    if board:
        print("\nFinal Prediction Leaderboard:")
        for agent, stats in sorted(
            board.items(), key=lambda x: x[1].get("mean_score", 999)
        ):
            mean = stats.get("mean_score")
            if isinstance(mean, (int, float)) and not math.isnan(mean):
                print(
                    f"  {agent}: mean RMSE = {mean:.4f} "
                    f"({stats['n_predictions']} predictions)"
                )
            else:
                print(
                    f"  {agent}: {stats['n_predictions']} predictions, not yet scored"
                )

    # Theory trajectories
    for t in protocol.state.active_theories():
        try:
            traj = protocol.state.theory_trajectory(t.name)
            print(
                f"\n{t.name}: trajectory = {traj['trajectory']} "
                f"({traj['n_revisions']} revisions, "
                f"{traj['n_progressive']} progressive)"
            )
        except (ValueError, KeyError):
            pass

    if args.cycles > 0:
        last_cycle = protocol.state.cycle - 1
        print(
            f"\nFull transcript: {os.path.join(output_dir, f'debate_cycle_{last_cycle}.json')}"
        )
        print(
            f"Epistemic state: {os.path.join(output_dir, f'epistemic_state_cycle_{last_cycle}.json')}"
        )
    else:
        print("\nNo cycles were run.")


# Module-level flags for batch mode and model selection
_BATCH_MODE = False
_LLM_MODEL = "claude-sonnet-4-20250514"
_SELECTION_METHOD = "bayesian"  # "bayesian" or "heuristic"


if __name__ == "__main__":
    main()
