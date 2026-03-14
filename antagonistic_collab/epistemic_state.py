"""
Epistemic State Tracker.

This is the structured, cumulative record of what the adversarial debate
has established across cycles. It replaces a flat chat history with a
semantically organized knowledge structure.

The key insight: a "shared scratchpad" that's just a message log is not
cumulative — it's just long. The epistemic state explicitly tracks:
- What each theory claims and what models instantiate it
- What experiments have been run and what they showed
- What facts all agents agree on
- What disputes remain live
- A prediction registry (who predicted what, and were they right?)
"""

import copy
import json
import hashlib
import os
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime
import numpy as np


@dataclass
class TheoryCommitment:
    """A theory registered by an agent."""

    name: str
    agent_name: str
    core_claims: list[str]
    model_name: str  # e.g., "GCM", "SUSTAIN", "RULEX"
    model_params: dict = field(default_factory=dict)
    auxiliary_assumptions: list[str] = field(default_factory=list)
    term_glossary: dict = field(default_factory=dict)
    """Maps natural language terms to specific model parameters/operations.
    e.g., {"attention": "w_i (dimensional weight, sum to 1)",
           "learning": "incremental exemplar storage after each trial",
           "similarity": "exp(-c * weighted_distance)"}
    This resolves terminology confusion: when two agents both say
    'attention shifts', the glossary pins each usage to a different
    model parameter, making the distinction explicit."""
    status: str = "active"  # active | modified | abandoned
    revision_log: list[dict] = field(default_factory=list)
    registered_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def revise(
        self,
        description: str,
        new_params: Optional[dict] = None,
        new_claims: Optional[list[str]] = None,
        new_auxiliaries: Optional[list[str]] = None,
        triggered_by_experiment: Optional[str] = None,
        new_predictions: Optional[list[str]] = None,
    ):
        """
        Register a theory revision. All modifications are logged.
        No stealth revisions — the prediction registry makes backtracking visible.

        The critical addition: new_predictions. A revision that generates
        new testable predictions is progressive (Lakatos); a revision that
        only accommodates the current failure without new predictions is
        degenerative. The system logs both, but the distinction is visible
        in the audit phase and the prediction leaderboard.

        Args:
            description: What changed and why.
            new_params: Updated model parameters.
            new_claims: Replacement core claims.
            new_auxiliaries: Replacement auxiliary assumptions.
            triggered_by_experiment: experiment_id that prompted this revision.
            new_predictions: Testable predictions the revised theory makes that
                the old version did NOT make. Empty list = degenerative revision.
        """
        revision = {
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "old_params": copy.deepcopy(self.model_params) if new_params else None,
            "old_claims": copy.deepcopy(self.core_claims) if new_claims else None,
            "triggered_by": triggered_by_experiment,
            "new_predictions": new_predictions or [],
            "revision_type": "progressive" if new_predictions else "degenerative",
        }
        if new_params:
            self.model_params.update(new_params)
        if new_claims:
            self.core_claims = new_claims
        if new_auxiliaries:
            self.auxiliary_assumptions = new_auxiliaries
        self.status = "modified"
        self.revision_log.append(revision)


@dataclass
class ExperimentRecord:
    """Record of a proposed and (possibly) executed experiment."""

    experiment_id: str
    cycle: int
    proposed_by: str  # agent name
    title: str
    design_spec: dict  # structured experiment specification
    rationale: str
    critique_log: list[dict] = field(
        default_factory=list
    )  # critiques from other agents
    revision_history: list[dict] = field(default_factory=list)
    """Links design revisions to the critiques that motivated them.
    Each entry: {
        "revised_by": agent_name,
        "addresses_critiques": [critique_indices],  # which critiques this responds to
        "changes": str,  # what changed in the design
        "old_design_spec": dict,  # snapshot before revision
        "new_design_spec": dict,  # snapshot after revision
        "timestamp": str,
    }
    This closes the provenance chain: proposal → critique → revision,
    fully traceable. A reader can follow exactly why each design
    element exists."""
    status: str = "proposed"  # proposed | approved | executed | rejected
    moderator_edits: Optional[str] = None
    data: Optional[dict] = None
    interpretations: dict = field(default_factory=dict)  # {agent_name: interpretation}


@dataclass
class Prediction:
    """A registered prediction: agent X's model predicts Y for experiment Z."""

    prediction_id: str
    experiment_id: str
    agent_name: str
    model_name: str
    model_params: dict
    predicted_pattern: dict  # e.g., {"accuracy_type_I": 0.95, "accuracy_type_VI": 0.62}
    actual_pattern: Optional[dict] = None
    score: Optional[float] = None  # quantified accuracy of prediction
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Dispute:
    """A live dispute between agents."""

    dispute_id: str
    claim: str
    positions: dict  # {agent_name: position_statement}
    proposed_resolution: Optional[str] = None
    status: str = "open"  # open | resolved | tabled
    resolution_experiment: Optional[str] = None  # experiment_id that could resolve it


@dataclass
class ModelClaim:
    """A structured claim about what a model predicts.

    This is the anti-straw-manning mechanism. When an agent makes a claim
    about what ANY model predicts (its own or an opponent's), that claim
    must be backed by an actual model run. The claim records the model
    called, the parameters used, the conditions tested, and the output.

    This makes it impossible to misrepresent an opponent's predictions:
    the model output is right there in the record.
    """

    claimant: str  # agent making the claim
    target_model: str  # which model was called
    params_used: dict  # parameters for the model call
    conditions: dict  # experimental conditions / stimulus spec
    model_output: dict  # actual output from calling the model
    interpretation: str  # agent's natural language interpretation
    verified: bool = False  # set to True after system verifies the call
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class EpistemicState:
    """
    The full epistemic state of the adversarial collaboration.

    This object is passed to agents at the start of each phase so they
    know what has been established, what's disputed, and how their
    past predictions have fared.
    """

    domain: str  # e.g., "human categorization"
    theories: list[TheoryCommitment] = field(default_factory=list)
    experiments: list[ExperimentRecord] = field(default_factory=list)
    predictions: list[Prediction] = field(default_factory=list)
    established_facts: list[str] = field(default_factory=list)
    disputes: list[Dispute] = field(default_factory=list)
    cycle: int = 0
    log: list[dict] = field(default_factory=list)  # free-form event log
    model_posterior: Optional[dict] = None  # Bayesian posterior over models
    agent_hypotheses: list[dict] = field(
        default_factory=list
    )  # from interpretation debate

    # --- Theory management ---

    def register_theory(self, theory: TheoryCommitment):
        """Register a new theory. Checks for duplicates."""
        existing = [t for t in self.theories if t.name == theory.name]
        if existing:
            raise ValueError(
                f"Theory '{theory.name}' already registered by {existing[0].agent_name}. "
                "Use revise_theory() to modify."
            )
        self.theories.append(theory)
        self._log(
            "theory_registered", {"theory": theory.name, "agent": theory.agent_name}
        )

    def revise_theory(self, theory_name: str, **kwargs):
        """Revise an existing theory. Logs the change."""
        theory = self.get_theory(theory_name)
        if theory is None:
            raise ValueError(f"Theory '{theory_name}' not found.")
        theory.revise(**kwargs)
        self._log(
            "theory_revised",
            {"theory": theory_name, "description": kwargs.get("description", "")},
        )

    def get_theory(self, name: str) -> Optional[TheoryCommitment]:
        matches = [t for t in self.theories if t.name == name]
        return matches[0] if matches else None

    def active_theories(self) -> list[TheoryCommitment]:
        return [t for t in self.theories if t.status in ("active", "modified")]

    # --- Experiment management ---

    def propose_experiment(
        self, proposed_by: str, title: str, design_spec: dict, rationale: str
    ) -> ExperimentRecord:
        """Register a new experiment proposal."""
        exp_id = f"exp_{self.cycle}_{hashlib.md5(title.encode()).hexdigest()[:8]}"
        record = ExperimentRecord(
            experiment_id=exp_id,
            cycle=self.cycle,
            proposed_by=proposed_by,
            title=title,
            design_spec=design_spec,
            rationale=rationale,
        )
        self.experiments.append(record)
        self._log(
            "experiment_proposed", {"id": exp_id, "by": proposed_by, "title": title}
        )
        return record

    def add_critique(
        self,
        experiment_id: str,
        agent_name: str,
        critique: str,
        quantitative_evidence: Optional[dict] = None,
        model_claims: Optional[list] = None,
    ):
        """Add a critique to an experiment proposal. Returns the critique index."""
        exp = self._get_experiment(experiment_id)
        critique_index = len(exp.critique_log)
        exp.critique_log.append(
            {
                "index": critique_index,
                "agent": agent_name,
                "critique": critique,
                "evidence": quantitative_evidence,
                "model_claims": model_claims or [],  # list of ModelClaim dicts
                "timestamp": datetime.now().isoformat(),
            }
        )
        return critique_index

    def revise_proposal(
        self,
        experiment_id: str,
        revised_by: str,
        addresses_critiques: list[int],
        changes: str,
        new_design_spec: dict,
    ):
        """
        Revise a proposal, explicitly linking the revision to the
        critiques it addresses. This closes the provenance chain:
        proposal → critique → revision is fully traceable.

        Args:
            experiment_id: The experiment being revised.
            revised_by: Agent making the revision.
            addresses_critiques: Indices into critique_log that this
                revision responds to. Cannot be empty — every revision
                must be motivated by at least one critique.
            changes: Natural language description of what changed.
            new_design_spec: The updated design specification.
        """
        if not addresses_critiques:
            raise ValueError(
                "Revisions must address at least one critique. "
                "Unmotivated revisions break the provenance chain."
            )
        exp = self._get_experiment(experiment_id)
        # Validate critique indices exist
        for idx in addresses_critiques:
            if idx < 0 or idx >= len(exp.critique_log):
                raise ValueError(
                    f"Critique index {idx} does not exist. "
                    f"Experiment has {len(exp.critique_log)} critiques."
                )
        exp.revision_history.append(
            {
                "revised_by": revised_by,
                "addresses_critiques": addresses_critiques,
                "changes": changes,
                "old_design_spec": copy.deepcopy(exp.design_spec),
                "new_design_spec": new_design_spec,
                "timestamp": datetime.now().isoformat(),
            }
        )
        exp.design_spec = new_design_spec
        self._log(
            "proposal_revised",
            {
                "id": experiment_id,
                "by": revised_by,
                "addressed_critiques": addresses_critiques,
            },
        )

    def approve_experiment(
        self, experiment_id: str, moderator_edits: Optional[str] = None
    ):
        """Moderator approves an experiment for execution."""
        exp = self._get_experiment(experiment_id)
        exp.status = "approved"
        exp.moderator_edits = moderator_edits
        self._log("experiment_approved", {"id": experiment_id})

    def record_data(self, experiment_id: str, data: dict):
        """Record experimental data."""
        exp = self._get_experiment(experiment_id)
        exp.status = "executed"
        exp.data = data
        self._log("data_recorded", {"id": experiment_id})

    def add_interpretation(
        self, experiment_id: str, agent_name: str, interpretation: str
    ):
        """Agent interprets experimental results."""
        exp = self._get_experiment(experiment_id)
        exp.interpretations[agent_name] = interpretation

    def _get_experiment(self, experiment_id: str) -> ExperimentRecord:
        matches = [e for e in self.experiments if e.experiment_id == experiment_id]
        if not matches:
            raise ValueError(f"Experiment '{experiment_id}' not found.")
        return matches[0]

    # --- Prediction registry ---

    def register_prediction(
        self,
        experiment_id: str,
        agent_name: str,
        model_name: str,
        model_params: dict,
        predicted_pattern: dict,
    ) -> Prediction:
        """
        Register a quantitative prediction. This is the accountability mechanism:
        agents commit to specific predictions BEFORE seeing data.
        """
        pred_id = f"pred_{agent_name}_{experiment_id}"
        pred = Prediction(
            prediction_id=pred_id,
            experiment_id=experiment_id,
            agent_name=agent_name,
            model_name=model_name,
            model_params=model_params,
            predicted_pattern=predicted_pattern,
        )
        self.predictions.append(pred)
        self._log(
            "prediction_registered",
            {"id": pred_id, "agent": agent_name, "experiment": experiment_id},
        )
        return pred

    def score_predictions(
        self, experiment_id: str, actual_pattern: dict, metric: str = "rmse"
    ):
        """
        Score all predictions for an experiment against actual data.
        Makes model performance transparent and cumulative.
        """
        relevant = [p for p in self.predictions if p.experiment_id == experiment_id]
        for pred in relevant:
            pred.actual_pattern = actual_pattern
            # Compute score on overlapping keys
            shared_keys = set(pred.predicted_pattern.keys()) & set(
                actual_pattern.keys()
            )
            if not shared_keys:
                continue
            predicted_vals = [pred.predicted_pattern[k] for k in shared_keys]
            actual_vals = [actual_pattern[k] for k in shared_keys]
            if metric == "rmse":
                pred.score = float(
                    np.sqrt(
                        np.mean((np.array(predicted_vals) - np.array(actual_vals)) ** 2)
                    )
                )
            elif metric == "correlation":
                if len(predicted_vals) > 2:
                    with np.errstate(invalid="ignore"):
                        corr = float(np.corrcoef(predicted_vals, actual_vals)[0, 1])
                    pred.score = None if np.isnan(corr) else corr
                else:
                    pred.score = None

        self._log(
            "predictions_scored",
            {
                "experiment": experiment_id,
                "scores": {p.agent_name: p.score for p in relevant},
            },
        )

    def prediction_leaderboard(self) -> dict:
        """
        Cumulative prediction accuracy across all experiments.
        Returns {agent_name: {n_predictions, mean_score, scores}}.
        """
        from collections import defaultdict

        board = defaultdict(lambda: {"scores": [], "n_predictions": 0})
        for pred in self.predictions:
            if pred.score is not None:
                board[pred.agent_name]["scores"].append(pred.score)
                board[pred.agent_name]["n_predictions"] += 1
        for agent in board:
            scores = board[agent]["scores"]
            board[agent]["mean_score"] = float(np.mean(scores)) if scores else None
        return dict(board)

    def theory_trajectory(self, theory_name: str) -> dict:
        """
        Compute the Lakatosian trajectory of a theory: is it progressing
        or degenerating?

        A theory is PROGRESSIVE if its revisions generate new testable
        predictions that are subsequently confirmed (or at least tested).
        A theory is DEGENERATIVE if it is repeatedly patched to accommodate
        data without generating new predictions.

        Returns:
            dict with:
                "n_revisions": total number of revisions
                "n_progressive": revisions that generated new predictions
                "n_degenerative": revisions that did not
                "new_predictions_made": list of all new predictions from revisions
                "new_predictions_tested": how many were subsequently tested
                "new_predictions_confirmed": how many of those tests succeeded
                "trajectory": "progressive" | "degenerative" | "stable" | "insufficient_data"
                "prediction_trend": list of RMSE scores over time (improving = progressive)
        """
        theory = self.get_theory(theory_name)
        if theory is None:
            raise ValueError(f"Theory '{theory_name}' not found.")

        if not theory.revision_log:
            # No revisions — check prediction trend
            agent_preds = [
                p
                for p in self.predictions
                if p.agent_name == theory.agent_name and p.score is not None
            ]
            return {
                "n_revisions": 0,
                "n_progressive": 0,
                "n_degenerative": 0,
                "new_predictions_made": [],
                "new_predictions_tested": 0,
                "new_predictions_confirmed": 0,
                "trajectory": "stable",
                "prediction_trend": [p.score for p in agent_preds],
            }

        n_progressive = sum(
            1 for r in theory.revision_log if r.get("revision_type") == "progressive"
        )
        n_degenerative = sum(
            1 for r in theory.revision_log if r.get("revision_type") == "degenerative"
        )

        # Collect all new predictions from progressive revisions
        all_new_predictions = []
        for r in theory.revision_log:
            all_new_predictions.extend(r.get("new_predictions", []))

        # Check prediction accuracy trend over time
        agent_preds = sorted(
            [
                p
                for p in self.predictions
                if p.agent_name == theory.agent_name and p.score is not None
            ],
            key=lambda p: p.timestamp,
        )
        prediction_trend = [p.score for p in agent_preds]

        # Determine trajectory
        if len(theory.revision_log) < 2:
            trajectory = "insufficient_data"
        elif n_progressive > n_degenerative:
            trajectory = "progressive"
        elif n_degenerative > n_progressive:
            trajectory = "degenerative"
        else:
            # Tied on revision counts — use prediction trend
            if (
                len(prediction_trend) >= 2
                and prediction_trend[-1] < prediction_trend[0]
            ):
                trajectory = "progressive"  # RMSE improving
            elif (
                len(prediction_trend) >= 2
                and prediction_trend[-1] > prediction_trend[0]
            ):
                trajectory = "degenerative"  # RMSE worsening
            else:
                trajectory = "insufficient_data"

        return {
            "n_revisions": len(theory.revision_log),
            "n_progressive": n_progressive,
            "n_degenerative": n_degenerative,
            "new_predictions_made": all_new_predictions,
            "new_predictions_tested": 0,  # TODO: cross-reference with prediction registry
            "new_predictions_confirmed": 0,  # TODO: same
            "trajectory": trajectory,
            "prediction_trend": prediction_trend,
        }

    # --- Dispute tracking ---

    def register_dispute(self, claim: str, positions: dict) -> Dispute:
        """Register a live dispute between agents."""
        d_id = f"disp_{hashlib.md5(claim.encode()).hexdigest()[:8]}"
        dispute = Dispute(dispute_id=d_id, claim=claim, positions=positions)
        self.disputes.append(dispute)
        return dispute

    def resolve_dispute(
        self, dispute_id: str, resolution: str, experiment_id: Optional[str] = None
    ):
        matches = [d for d in self.disputes if d.dispute_id == dispute_id]
        if matches:
            matches[0].status = "resolved"
            matches[0].proposed_resolution = resolution
            matches[0].resolution_experiment = experiment_id
            self._log(
                "dispute_resolved",
                {
                    "id": dispute_id,
                    "resolution": resolution,
                },
            )

    def open_disputes(self) -> list[Dispute]:
        return [d for d in self.disputes if d.status == "open"]

    # --- State summary (for agent context windows) ---

    def summary_for_agent(self, agent_name: str) -> str:
        """
        Generate a natural-language summary of the epistemic state
        for injection into an agent's context. This is what makes the
        debate cumulative — agents see the structured history, not
        just a chat log.
        """
        lines = [f"## Epistemic State — Cycle {self.cycle}"]
        lines.append(f"Domain: {self.domain}\n")

        # Active theories
        lines.append("### Active Theories")
        for t in self.active_theories():
            marker = " (YOUR THEORY)" if t.agent_name == agent_name else ""
            lines.append(f"**{t.name}**{marker} — {t.model_name}")
            lines.append(f"  Status: {t.status}")
            if t.revision_log:
                n_prog = sum(
                    1 for r in t.revision_log if r.get("revision_type") == "progressive"
                )
                n_degen = sum(
                    1
                    for r in t.revision_log
                    if r.get("revision_type") == "degenerative"
                )
                lines.append(
                    f"  Revisions: {len(t.revision_log)} ({n_prog} progressive, {n_degen} degenerative)"
                )
                latest = t.revision_log[-1]
                lines.append(
                    f"  Latest revision: {latest.get('description', '(no description)')}"
                )
                if latest.get("new_predictions"):
                    preds = [str(p) for p in latest["new_predictions"][:2]]
                    lines.append(f"  New predictions from revision: {'; '.join(preds)}")
            if t.term_glossary:
                key_terms = list(t.term_glossary.keys())[:4]
                lines.append(
                    f"  Key terms: {', '.join(key_terms)} (see glossary for operational definitions)"
                )
            lines.append(f"  Core claims: {'; '.join(t.core_claims[:3])}")
            lines.append("")

        # Theory trajectories (if enough data)
        if self.cycle >= 2:
            lines.append("### Theory Trajectories (Lakatos)")
            for t in self.active_theories():
                try:
                    traj = self.theory_trajectory(t.name)
                    if traj["trajectory"] != "insufficient_data":
                        marker = " (YOUR THEORY)" if t.agent_name == agent_name else ""
                        lines.append(
                            f"  {t.name}{marker}: **{traj['trajectory']}** "
                            f"({traj['n_revisions']} revisions, "
                            f"{traj['n_progressive']} progressive)"
                        )
                except (ValueError, KeyError):
                    pass
            lines.append("")

        # Prediction leaderboard
        board = self.prediction_leaderboard()
        if board:
            lines.append("### Prediction Track Record")
            for agent, stats in sorted(
                board.items(), key=lambda x: x[1].get("mean_score", 999)
            ):
                marker = " (you)" if agent == agent_name else ""
                if stats["mean_score"] is not None:
                    lines.append(
                        f"  {agent}{marker}: {stats['n_predictions']} predictions, "
                        f"mean RMSE = {stats['mean_score']:.3f}"
                    )
                else:
                    lines.append(
                        f"  {agent}{marker}: {stats['n_predictions']} predictions, not yet scored"
                    )
            lines.append("")

        # Bayesian model posterior (if available)
        if self.model_posterior and "log_probs" in self.model_posterior:
            import numpy as _np

            lp = _np.array(self.model_posterior["log_probs"])
            probs = _np.exp(lp - _np.max(lp))
            probs = probs / probs.sum()
            entropy = -float(_np.sum(probs * _np.log(probs + 1e-30)))
            model_names = self.model_posterior.get("model_names", [])
            lines.append("### Bayesian Model Posterior")
            for i, p in enumerate(probs):
                name = model_names[i] if i < len(model_names) else f"model_{i}"
                lines.append(f"  P({name}) = {p:.4f}")
            lines.append(f"  Entropy = {entropy:.4f} (max = {_np.log(len(probs)):.4f})")
            lines.append("")

        # Established facts
        if self.established_facts:
            lines.append("### Established Facts (all agents agree)")
            for fact in self.established_facts:
                lines.append(f"  - {fact}")
            lines.append("")

        # Open disputes
        open_d = self.open_disputes()
        if open_d:
            lines.append("### Open Disputes")
            for d in open_d:
                lines.append(f"  **{d.claim}**")
                for agent, pos in d.positions.items():
                    lines.append(f"    {agent}: {pos}")
            lines.append("")

        # Recent experiments
        recent = [e for e in self.experiments if e.cycle >= self.cycle - 2]
        if recent:
            lines.append("### Recent Experiments")
            for e in recent:
                lines.append(f"  **{e.title}** (cycle {e.cycle}, status: {e.status})")
                if e.data:
                    lines.append(f"  Data available: {list(e.data.keys())}")
                if agent_name in e.interpretations:
                    lines.append(
                        f"  Your interpretation: {e.interpretations[agent_name][:100]}..."
                    )
            lines.append("")

        return "\n".join(lines)

    def advance_cycle(self):
        """Move to the next cycle."""
        self.cycle += 1
        self._log("cycle_advanced", {"new_cycle": self.cycle})

    # --- Serialization ---

    def to_dict(self) -> dict:
        """Serialize to dict for saving/loading."""
        return asdict(self)

    def to_json(self, path: str):
        """Save to JSON file."""

        def _sanitize(obj):
            """Recursively convert numpy types in both keys and values."""
            if isinstance(obj, dict):
                return {_sanitize_scalar(k): _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize(v) for v in obj]
            return _sanitize_scalar(obj)

        def _sanitize_scalar(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(_sanitize(self.to_dict()), f, indent=2, default=str)

    def _log(self, event_type: str, details: dict):
        self.log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "cycle": self.cycle,
                "event": event_type,
                **details,
            }
        )
