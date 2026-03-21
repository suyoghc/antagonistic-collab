"""Standalone decision-domain debate runner.

Orchestrates LLM debate for decision-making model identification without
depending on DebateProtocol (which is tightly coupled to categorization).

Reuses domain-agnostic pieces:
    - compute_eig(), ModelPosterior from bayesian_selection.py
    - decision_predictions_for_eig() from decision_eig.py
    - generate_synthetic_choices() from decision_runner.py
    - Agent configs from decision_agents.py

New functionality:
    - LLM debate round: show prediction errors, get parameter revisions
    - Parameter validation via inspect.signature
    - Cycle loop: EIG select → data → score → debate → param update → repeat

Architecture decision D49: standalone runner (Option C) to avoid risk to
the 47/48 categorization pipeline.
"""

from __future__ import annotations

import json
import re
from typing import Callable

import numpy as np

from antagonistic_collab.bayesian_selection import ModelPosterior
from antagonistic_collab.models.decision_eig import (
    GAMBLE_GROUPS,
    select_decision_experiment,
    update_decision_posterior,
)
from antagonistic_collab.models.decision_runner import (
    DECISION_AGENT_MAP,
    DECISION_EXPECTED_WINNER,
    GT_DECISION_PARAMS,
    compute_decision_predictions,
)

DECISION_AGENTS = ["CPT_Agent", "EU_Agent", "PH_Agent"]


# ── Parameter Validation ──


def filter_valid_params(model, proposed: dict) -> dict:
    """Filter proposed parameters to only those accepted by the model.

    Decision models use a `params` dict (not keyword args), so we validate
    against model.default_params keys rather than inspect.signature.

    Args:
        model: A decision model instance (CPT, EU, or PH) with default_params.
        proposed: Dict of {param_name: value} from LLM agent.

    Returns:
        Filtered dict with only valid parameter names.
    """
    valid_names = set(model.default_params.keys())
    return {k: v for k, v in proposed.items() if k in valid_names}


def validate_revision_rmse(
    observed: dict[str, float],
    baseline_preds: dict[str, float],
    revised_preds: dict[str, float],
    tolerance: float = 0.0,
) -> tuple[bool, float, float]:
    """Check whether revised predictions improve RMSE over baseline.

    Args:
        observed: {gamble_name: observed P(choose A)}
        baseline_preds: {gamble_name: baseline prediction}
        revised_preds: {gamble_name: revised prediction}
        tolerance: accept if revised_rmse < baseline_rmse - tolerance.
            Default 0.0 requires strict improvement.

    Returns:
        (accepted, baseline_rmse, revised_rmse)
    """
    gambles = list(observed.keys())
    obs = np.array([observed[g] for g in gambles])
    base = np.array([baseline_preds[g] for g in gambles])
    rev = np.array([revised_preds[g] for g in gambles])

    baseline_rmse = float(np.sqrt(np.mean((base - obs) ** 2)))
    revised_rmse = float(np.sqrt(np.mean((rev - obs) ** 2)))

    return revised_rmse < baseline_rmse - tolerance, baseline_rmse, revised_rmse


# ── Prompt Construction ──


def build_interpretation_prompt(
    agent_config,
    observed: dict[str, float],
    predictions: dict[str, float],
    posterior: dict[str, float],
    cycle: int,
    all_agent_preds: dict[str, dict[str, float]] | None = None,
) -> str:
    """Build the debate prompt showing prediction errors and asking for diagnosis.

    Args:
        agent_config: DecisionAgentConfig for this agent.
        observed: {gamble_name: observed P(choose A)}
        predictions: {gamble_name: this agent's prediction}
        posterior: {agent_name: posterior probability}
        cycle: current cycle number.
        all_agent_preds: optional {agent_name: {gamble: pred}} for comparison.

    Returns:
        User message string for the LLM call.
    """
    lines = [f"## Cycle {cycle} Results\n"]

    # Prediction errors table
    lines.append("### Your Predictions vs Observed Data\n")
    lines.append("| Gamble | Observed P(A) | Your Prediction | Error |")
    lines.append("|---|---|---|---|")
    for gname in sorted(observed.keys()):
        obs = observed[gname]
        pred = predictions.get(gname, "N/A")
        if isinstance(pred, (int, float)):
            err = abs(obs - pred)
            lines.append(f"| {gname} | {obs:.3f} | {pred:.3f} | {err:.3f} |")
        else:
            lines.append(f"| {gname} | {obs:.3f} | {pred} | — |")

    # Show other models' predictions if available
    if all_agent_preds:
        lines.append("\n### All Models' Predictions\n")
        lines.append(
            "| Gamble | Observed | " + " | ".join(all_agent_preds.keys()) + " |"
        )
        lines.append("|---|---|" + "|".join(["---"] * len(all_agent_preds)) + "|")
        for gname in sorted(observed.keys()):
            obs = observed[gname]
            preds = " | ".join(
                f"{all_agent_preds[a].get(gname, 0):.3f}" for a in all_agent_preds
            )
            lines.append(f"| {gname} | {obs:.3f} | {preds} |")

    # Posterior
    lines.append("\n### Current Model Posterior Probabilities\n")
    for agent_name, prob in sorted(posterior.items(), key=lambda x: -x[1]):
        marker = " ← YOU" if agent_name == agent_config.name else ""
        lines.append(f"- {agent_name}: {prob:.3f}{marker}")

    # Current parameters
    lines.append("\n### Your Current Parameters\n")
    for k, v in sorted(agent_config.default_params.items()):
        lines.append(f"- {k} = {v}")

    # Instructions
    lines.append(
        "\n### Instructions\n"
        "Analyze the results above. Diagnose why your predictions differ from "
        "the observed data. Consider whether your current parameter values are "
        "appropriate.\n\n"
        "Respond with a JSON block:\n"
        "```json\n"
        "{\n"
        '  "interpretation": "Your analysis of the results",\n'
        '  "revision": null OR {\n'
        '    "description": "What you are changing and why",\n'
        '    "new_params": {"param_name": new_value, ...}\n'
        "  }\n"
        "}\n"
        "```\n\n"
        "Set revision to null if your predictions are acceptable. "
        "Only propose parameter changes if you can diagnose a specific error."
    )

    return "\n".join(lines)


# ── Response Parsing ──


def extract_json_from_text(text: str) -> dict | None:
    """Extract first JSON block from LLM response text."""
    # Try ```json ... ``` blocks first
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try raw {...}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def parse_agent_revision(response: dict) -> dict | None:
    """Extract revision proposal from parsed agent response.

    Returns None if no revision proposed or params are empty.
    """
    revision = response.get("revision")
    if revision is None:
        return None
    if not isinstance(revision, dict):
        return None
    new_params = revision.get("new_params", {})
    if not new_params:
        return None
    return revision


# ── Debate Round ──


def run_debate_round(
    agent_configs: list,
    observed: dict[str, float],
    gamble_names: list[str],
    posterior_probs: dict[str, float],
    cycle: int,
    call_fn: Callable | None = None,
    client=None,
    agent_params: dict[str, dict] | None = None,
    all_observed: dict[str, float] | None = None,
) -> list[dict]:
    """Run one debate round: show errors to each agent, collect revisions.

    Args:
        agent_configs: list of DecisionAgentConfig.
        observed: {gamble_name: observed P(choose A)} for current cycle.
        gamble_names: gambles tested this cycle.
        posterior_probs: {agent_name: probability}.
        cycle: current cycle number.
        call_fn: LLM call function (for testing). Signature: (system, user) -> str.
            If None, uses client with call_agent from runner.
        client: LLM client (used if call_fn is None).
        agent_params: optional {agent_name: params} for predictions.
        all_observed: accumulated observations across all cycles. If provided,
            RMSE validation uses this full set instead of just current cycle.
            This prevents gaming local RMSE on 2-3 gambles.

    Returns:
        List of revision dicts: [{agent_name, accepted, old_params, new_params, rmse_before, rmse_after}]
    """
    if call_fn is None and client is None:
        raise ValueError("Must provide either call_fn or client")

    # Compute all agents' predictions for context
    all_preds = {}
    for config in agent_configs:
        model_name = DECISION_AGENT_MAP[config.name]
        params = (agent_params or {}).get(config.name, config.default_params)
        preds = {}
        for gname in gamble_names:
            p = compute_decision_predictions(model_name, gname, params=params)
            preds[gname] = p[gname]
        all_preds[config.name] = preds

    records = []

    for config in agent_configs:
        my_preds = all_preds[config.name]

        prompt = build_interpretation_prompt(
            config,
            observed,
            my_preds,
            posterior_probs,
            cycle,
            all_agent_preds=all_preds,
        )

        # Call LLM
        if call_fn is not None:
            raw = call_fn(config.system_prompt, prompt)
        else:
            from antagonistic_collab.runner import call_agent

            raw = call_agent(client, config.system_prompt, prompt)

        # Parse response
        parsed = extract_json_from_text(raw) if isinstance(raw, str) else raw
        if parsed is None:
            continue

        interpretation = ""
        if isinstance(parsed, dict):
            interpretation = parsed.get("interpretation", "")
            if not isinstance(interpretation, str):
                interpretation = str(interpretation)

        revision = parse_agent_revision(parsed)

        # Always create a record with interpretation, even without revision
        if revision is None:
            records.append({
                "agent_name": config.name,
                "interpretation": interpretation,
                "accepted": False,
                "has_revision": False,
            })
            continue

        # Validate params via inspect.signature
        new_params = filter_valid_params(config.model_class, revision["new_params"])
        if not new_params:
            records.append({
                "agent_name": config.name,
                "interpretation": interpretation,
                "accepted": False,
                "has_revision": False,
            })
            continue

        # Validate via RMSE against accumulated observations (if available)
        model_name = DECISION_AGENT_MAP[config.name]
        merged_params = {**config.default_params, **new_params}
        rmse_obs = all_observed if all_observed else observed

        # Compute baseline and revised predictions over the validation set
        baseline_preds_full = {}
        revised_preds_full = {}
        current_p = (agent_params or {}).get(config.name, config.default_params)
        for gname in rmse_obs:
            bp = compute_decision_predictions(model_name, gname, params=current_p)
            baseline_preds_full[gname] = bp[gname]
            rp = compute_decision_predictions(model_name, gname, params=merged_params)
            revised_preds_full[gname] = rp[gname]

        accepted, rmse_before, rmse_after = validate_revision_rmse(
            rmse_obs, baseline_preds_full, revised_preds_full
        )

        revision_record = {
            "agent_name": config.name,
            "interpretation": interpretation,
            "accepted": accepted,
            "has_revision": True,
            "old_params": dict(config.default_params),
            "proposed_params": new_params,
            "rmse_before": rmse_before,
            "rmse_after": rmse_after,
            "description": revision.get("description", ""),
        }

        if accepted:
            config.default_params.update(new_params)
            revision_record["new_params"] = dict(config.default_params)

        records.append(revision_record)

    return records


# ── Crux Protocol ──


def run_decision_crux_identification(
    configs: list,
    client,
    call_fn: Callable | None = None,
    cycle: int = 1,
    crux_counter: int = 0,
) -> list:
    """Each decision agent proposes 1-2 cruxes targeting gamble groups.

    Args:
        configs: list of DecisionAgentConfig.
        client: LLM client (used if call_fn is None).
        call_fn: mock LLM function. Signature: (system, user) -> str.
        cycle: current cycle number.
        crux_counter: starting counter for crux IDs.

    Returns:
        List of Crux objects.
    """
    from antagonistic_collab.epistemic_state import Crux

    group_names = list(GAMBLE_GROUPS.keys())
    all_cruxes = []

    for config in configs:
        prompt = (
            "PHASE: Crux Identification\n\n"
            "A 'crux' is a decisive question: a gamble group whose outcome "
            "would genuinely change your mind about which theory is correct.\n\n"
            "Propose 1-2 cruxes. For each, specify:\n"
            "- What gamble group would be decisive\n"
            "- What outcome would change your mind\n"
            "- Why this is a genuine crux\n\n"
            "AVAILABLE GAMBLE GROUPS:\n"
            f"  {', '.join(group_names)}\n\n"
            "Use EXACTLY one of the group names above.\n\n"
            "Output a JSON block:\n"
            '{"cruxes": [{"description": "...", '
            '"discriminating_experiment": "gamble_group_name", '
            '"resolution_criterion": "what outcome would be decisive"}]}\n'
        )

        if call_fn is not None:
            raw = call_fn(config.system_prompt, prompt)
        else:
            from antagonistic_collab.runner import call_agent

            raw = call_agent(client, config.system_prompt, prompt)

        parsed = extract_json_from_text(raw) if isinstance(raw, str) else raw
        if parsed is None:
            continue

        cruxes_raw = parsed.get("cruxes", [])
        if not isinstance(cruxes_raw, list):
            cruxes_raw = []

        for c in cruxes_raw:
            if not isinstance(c, dict) or not c.get("description"):
                continue
            crux_counter += 1
            crux_id = f"crux_{crux_counter:03d}"
            crux = Crux(
                id=crux_id,
                proposer=config.name,
                description=c["description"],
                discriminating_experiment=c.get("discriminating_experiment"),
                resolution_criterion=c.get("resolution_criterion"),
                cycle_proposed=cycle,
                supporters=[config.name],
            )
            all_cruxes.append(crux)

    return all_cruxes


def run_decision_crux_negotiation(
    configs: list,
    cruxes: list,
    client,
    call_fn: Callable | None = None,
    cycle: int = 1,
) -> list:
    """Agents respond to cruxes: accept, reject, or counter-propose.

    Args:
        configs: list of DecisionAgentConfig.
        cruxes: list of Crux objects from identification.
        client: LLM client (used if call_fn is None).
        call_fn: mock LLM function.
        cycle: current cycle number.

    Returns:
        Updated list of Crux objects (may include counter-proposals).
    """
    from antagonistic_collab.epistemic_state import Crux

    if not cruxes:
        return cruxes

    crux_text = "\n".join(
        f"  {c.id}: {c.description} (targets: {c.discriminating_experiment}, "
        f"proposed by: {c.proposer})"
        for c in cruxes
    )

    crux_counter = max(
        int(c.id.replace("crux_", "")) for c in cruxes
    )

    for config in configs:
        prompt = (
            "PHASE: Crux Negotiation\n\n"
            "Review the proposed cruxes below. For each, decide:\n"
            "- 'accept': this is a genuine crux that would change your mind too\n"
            "- 'reject': this is not decisive (explain why)\n"
            "- 'counter': propose a better crux instead\n\n"
            f"PROPOSED CRUXES:\n{crux_text}\n\n"
            "Output a JSON block:\n"
            '{"responses": [{"crux_id": "...", "action": "accept|reject|counter", '
            '"reason": "...", '
            '"counter_crux": {"description": "...", '
            '"discriminating_experiment": "gamble_group_name"}}]}\n'
        )

        if call_fn is not None:
            raw = call_fn(config.system_prompt, prompt)
        else:
            from antagonistic_collab.runner import call_agent

            raw = call_agent(client, config.system_prompt, prompt)

        parsed = extract_json_from_text(raw) if isinstance(raw, str) else raw
        if parsed is None:
            continue

        responses_raw = parsed.get("responses", [])
        if not isinstance(responses_raw, list):
            responses_raw = []

        for r in responses_raw:
            if not isinstance(r, dict):
                continue
            crux_id = r.get("crux_id", "")
            action = r.get("action", "")

            matching = [c for c in cruxes if c.id == crux_id]
            if not matching:
                continue

            crux = matching[0]

            if action == "accept":
                if config.name not in crux.supporters:
                    crux.supporters.append(config.name)
            elif action == "counter":
                counter = r.get("counter_crux", {})
                if isinstance(counter, dict) and counter.get("description"):
                    crux_counter += 1
                    new_id = f"crux_{crux_counter:03d}"
                    new_crux = Crux(
                        id=new_id,
                        proposer=config.name,
                        description=counter["description"],
                        discriminating_experiment=counter.get(
                            "discriminating_experiment"
                        ),
                        resolution_criterion=counter.get("resolution_criterion"),
                        cycle_proposed=cycle,
                        supporters=[config.name],
                    )
                    cruxes.append(new_crux)

    return cruxes


def finalize_decision_cruxes(
    cruxes: list,
    min_supporters: int = 2,
) -> list:
    """Accept cruxes with enough support, reject others.

    Only operates on 'proposed' cruxes (not already accepted/resolved/rejected).

    Args:
        cruxes: list of Crux objects.
        min_supporters: minimum supporters to accept.

    Returns:
        List of newly accepted Crux objects.
    """
    accepted = []
    for crux in cruxes:
        if crux.status != "proposed":
            continue
        if len(crux.supporters) >= min_supporters:
            crux.status = "accepted"
            accepted.append(crux)
        else:
            crux.status = "rejected"
    return accepted


def decision_cruxes_to_boost_indices(
    cruxes: list,
    group_names: list[str],
) -> list[int]:
    """Map accepted cruxes to gamble group indices for EIG boosting.

    Args:
        cruxes: list of Crux objects (should be accepted).
        group_names: ordered list of gamble group names.

    Returns:
        List of indices into the candidates/group_names list.
    """
    group_set = set(group_names)
    indices = []
    for crux in cruxes:
        if crux.status != "accepted":
            continue
        target = crux.discriminating_experiment
        if target and target in group_set:
            idx = group_names.index(target)
            if idx not in indices:
                indices.append(idx)
    return indices


# ── Full Debate Loop ──


def run_decision_debate(
    gt_model: str,
    n_cycles: int = 5,
    n_subjects: int = 30,
    learning_rate: float = 0.01,
    selection_strategy: str = "greedy",
    agent_params: dict[str, dict] | None = None,
    client=None,
    call_fn: Callable | None = None,
    enable_debate: bool = True,
    verbose: bool = True,
) -> dict:
    """Run the full decision debate loop.

    Args:
        gt_model: "CPT", "EU", or "PH"
        n_cycles: number of cycles
        n_subjects: simulated subjects per gamble
        learning_rate: likelihood tempering
        selection_strategy: "greedy" or "thompson"
        agent_params: optional {agent_name: params} for misspecification
        client: LLM client for debate rounds
        call_fn: mock LLM function (for testing)
        enable_debate: if False, skip debate round (computational only)
        verbose: print progress

    Returns:
        Results dict with winner, correctness, posterior history, revisions, etc.
    """
    from antagonistic_collab.models.decision_agents import (
        default_decision_agent_configs,
    )

    configs = default_decision_agent_configs()

    # Apply misspecified params if provided
    if agent_params:
        for config in configs:
            if config.name in agent_params:
                config.default_params = dict(agent_params[config.name])

    posterior = ModelPosterior.uniform(DECISION_AGENTS)
    candidates = list(GAMBLE_GROUPS.values())
    group_names = list(GAMBLE_GROUPS.keys())

    history = []
    all_revisions = []
    accumulated_observed = {}  # all observations across cycles for RMSE validation

    for cycle in range(n_cycles):
        # Build current agent_params dict from configs (may have been revised)
        current_params = {c.name: dict(c.default_params) for c in configs}

        # 1. EIG selection
        idx, eig_scores = select_decision_experiment(
            candidates,
            posterior,
            agent_params=current_params,
            n_subjects=n_subjects,
            n_sim=200,
            seed=42 + cycle,
            learning_rate=learning_rate,
            selection_strategy=selection_strategy,
        )

        selected_gambles = candidates[idx]
        selected_group = group_names[idx]

        # 2. Generate synthetic observations from GT (using TRUE params)
        import hashlib

        seed_input = f"{cycle}_{gt_model}_{'_'.join(sorted(selected_gambles))}"
        seed_hash = int(hashlib.md5(seed_input.encode()).hexdigest()[:8], 16) % 10000
        rng = np.random.default_rng(42 + seed_hash)

        observed = {}
        for gname in selected_gambles:
            preds = compute_decision_predictions(
                gt_model, gname, params=GT_DECISION_PARAMS[gt_model]
            )
            p_a = preds[gname]
            p_clipped = np.clip(p_a, 0.01, 0.99)
            observed[gname] = rng.binomial(n_subjects, p_clipped) / n_subjects

        # Accumulate observations for RMSE validation
        accumulated_observed.update(observed)

        # 3. Update posterior
        update_decision_posterior(
            posterior,
            observed,
            selected_gambles,
            agent_params=current_params,
            n_subjects=n_subjects,
            learning_rate=learning_rate,
        )

        # 4. Debate round (if enabled)
        cycle_revisions = []
        if enable_debate and (call_fn is not None or client is not None):
            cycle_revisions = run_debate_round(
                configs,
                observed,
                selected_gambles,
                posterior_probs=dict(zip(DECISION_AGENTS, posterior.probs.tolist())),
                cycle=cycle,
                call_fn=call_fn,
                client=client,
                agent_params=current_params,
                all_observed=accumulated_observed,
            )
            all_revisions.extend(cycle_revisions)

        # Record
        cycle_entry = {
            "cycle": cycle,
            "group": selected_group,
            "gambles": selected_gambles,
            "best_eig": float(eig_scores[idx]),
            "posterior": dict(
                zip(DECISION_AGENTS, [float(p) for p in posterior.probs])
            ),
            "entropy": float(posterior.entropy),
            "revisions": cycle_revisions,
            "agent_params": {c.name: dict(c.default_params) for c in configs},
        }
        history.append(cycle_entry)

        if verbose:
            leader = DECISION_AGENTS[np.argmax(posterior.probs)]
            rev_summary = (
                f", {len(cycle_revisions)} revisions"
                f" ({sum(1 for r in cycle_revisions if r['accepted'])} accepted)"
                if cycle_revisions
                else ""
            )
            print(
                f"  Cycle {cycle}: {selected_group} "
                f"(EIG={eig_scores[idx]:.4f}) → "
                f"leader={leader}{rev_summary}"
            )

    # Final analysis
    winner_idx = np.argmax(posterior.probs)
    winner = DECISION_AGENTS[winner_idx]
    expected = DECISION_EXPECTED_WINNER[gt_model]

    # Param recovery: how close are final params to GT?
    param_recovery = {}
    for config in configs:
        model_name = DECISION_AGENT_MAP[config.name]
        gt_params = GT_DECISION_PARAMS[model_name]
        final_params = config.default_params
        distances = []
        for k in gt_params:
            if k in final_params:
                gt_val = gt_params[k]
                final_val = final_params[k]
                if gt_val != 0:
                    distances.append(abs(final_val - gt_val) / abs(gt_val))
                else:
                    distances.append(abs(final_val))
        param_recovery[config.name] = {
            "mean_relative_distance": float(np.mean(distances)) if distances else 1.0,
            "final_params": dict(final_params),
            "gt_params": dict(gt_params),
        }

    return {
        "ground_truth": gt_model,
        "condition": "debate" if enable_debate else "no_debate",
        "n_cycles": n_cycles,
        "winner": winner,
        "expected": expected,
        "correct": winner == expected,
        "final_posterior": dict(
            zip(DECISION_AGENTS, [float(p) for p in posterior.probs])
        ),
        "final_entropy": float(posterior.entropy),
        "history": history,
        "revisions": all_revisions,
        "n_revisions_proposed": len(all_revisions),
        "n_revisions_accepted": sum(1 for r in all_revisions if r["accepted"]),
        "param_recovery": param_recovery,
    }
