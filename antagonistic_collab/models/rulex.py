"""
RULEX — Nosofsky, Palmeri & McKinley (1994).

A rule-based model that assumes learners search for simple rules first,
then store exceptions to those rules as exemplars. The search proceeds
hierarchically: single-dimension rules first, then conjunctive rules,
then exception memorization.

Key predictions:
- Strong advantage for rule-defined categories (Shepard Type I)
- Difficulty with categories that have no simple rule (Type VI)
- All-or-nothing learning for individual items (rule discovery is discrete)
- Interaction with verbal working memory (rules are verbally mediated)

Parameters:
    p_single (float): Probability of testing a single-dimension rule per search step.
    p_conj (float): Probability of testing a conjunctive rule per search step.
    p_exception (float): Probability of memorizing an exception per attempt.
    max_search_steps (int): Maximum rule search attempts.
    error_tolerance (float): Proportion of errors tolerated before rule is rejected.
"""

import numpy as np
from itertools import combinations
from typing import Optional


class RULEX:
    """RULEX model for categorization."""

    name = "RULEX (Rule-Plus-Exception)"
    description = (
        "Nosofsky, Palmeri & McKinley (1994). Categorization by rule search "
        "followed by exception memorization. Predicts advantage for simple rules, "
        "difficulty with non-rule-based categories, discrete learning transitions."
    )
    core_claims = [
        "Learners first search for simple verbalizable rules.",
        "Rule search is hierarchical: single-dimension rules tried before conjunctions.",
        "Items that violate the best rule are stored as exceptions.",
        "Learning shows discrete transitions (rule discovery), not gradual accumulation.",
    ]

    def __init__(self):
        self.default_params = {
            "p_single": 0.5,
            "p_conj": 0.3,
            "p_exception": 0.8,
            "max_search_steps": 50,
            "error_tolerance": 0.1,
        }

    def _evaluate_rule(
        self,
        rule: dict,
        stimuli: np.ndarray,
        labels: np.ndarray,
    ) -> dict:
        """
        Evaluate a candidate rule on the training data.

        rule format:
            {"type": "single", "dim": int, "threshold": float, "direction": int}
            {"type": "conjunction", "dims": list[int], "thresholds": list[float], "directions": list[int]}

        direction: 0 means below threshold -> cat 0, 1 means above threshold -> cat 0
        """
        n = len(labels)
        predictions = np.zeros(n, dtype=int)

        if rule["type"] == "single":
            dim = rule["dim"]
            for i, stim in enumerate(stimuli):
                if rule["direction"] == 0:
                    predictions[i] = 0 if stim[dim] <= rule["threshold"] else 1
                else:
                    predictions[i] = 0 if stim[dim] > rule["threshold"] else 1

        elif rule["type"] == "conjunction":
            for i, stim in enumerate(stimuli):
                # Both conditions must be met for category 0
                cat0 = True
                for dim, thresh, direc in zip(
                    rule["dims"], rule["thresholds"], rule["directions"]
                ):
                    if direc == 0:
                        cat0 = cat0 and (stim[dim] <= thresh)
                    else:
                        cat0 = cat0 and (stim[dim] > thresh)
                predictions[i] = 0 if cat0 else 1

        correct = predictions == labels
        accuracy = correct.mean()
        errors = np.where(~correct)[0]

        return {
            "accuracy": accuracy,
            "predictions": predictions,
            "error_indices": errors.tolist(),
            "n_errors": len(errors),
        }

    def _generate_single_rules(self, stimuli: np.ndarray) -> list[dict]:
        """Generate all possible single-dimension rules."""
        n_dims = stimuli.shape[1]
        rules = []
        for dim in range(n_dims):
            unique_vals = sorted(set(stimuli[:, dim]))
            # For binary dimensions, threshold at 0.5
            if len(unique_vals) == 2:
                thresholds = [(unique_vals[0] + unique_vals[1]) / 2]
            else:
                # Place thresholds between consecutive unique values
                thresholds = [
                    (unique_vals[i] + unique_vals[i + 1]) / 2
                    for i in range(len(unique_vals) - 1)
                ]
            for thresh in thresholds:
                for direction in [0, 1]:
                    rules.append(
                        {
                            "type": "single",
                            "dim": dim,
                            "threshold": thresh,
                            "direction": direction,
                        }
                    )
        return rules

    def _generate_conjunction_rules(self, stimuli: np.ndarray) -> list[dict]:
        """Generate all possible 2-dimension conjunctive rules."""
        n_dims = stimuli.shape[1]
        rules = []
        for dims in combinations(range(n_dims), 2):
            # For each pair of dimensions, try all threshold/direction combos
            for d0_dir in [0, 1]:
                for d1_dir in [0, 1]:
                    unique_0 = sorted(set(stimuli[:, dims[0]]))
                    unique_1 = sorted(set(stimuli[:, dims[1]]))
                    t0 = (
                        (unique_0[0] + unique_0[-1]) / 2
                        if len(unique_0) == 2
                        else np.median(unique_0)
                    )
                    t1 = (
                        (unique_1[0] + unique_1[-1]) / 2
                        if len(unique_1) == 2
                        else np.median(unique_1)
                    )
                    rules.append(
                        {
                            "type": "conjunction",
                            "dims": list(dims),
                            "thresholds": [t0, t1],
                            "directions": [d0_dir, d1_dir],
                        }
                    )
        return rules

    def find_best_rule(
        self,
        stimuli: np.ndarray,
        labels: np.ndarray,
        p_single: float = 0.5,
        p_conj: float = 0.3,
        max_search_steps: int = 50,
        error_tolerance: float = 0.1,
        seed: Optional[int] = None,
    ) -> dict:
        """
        Simulate RULEX's hierarchical rule search.

        Returns best rule found and its performance.
        """
        rng = np.random.default_rng(seed)

        single_rules = self._generate_single_rules(stimuli)
        conj_rules = self._generate_conjunction_rules(stimuli)

        best_rule = None
        best_accuracy = 0.0
        search_log = []

        for step in range(max_search_steps):
            r = rng.random()
            if r < p_single and single_rules:
                # Try a single-dimension rule
                idx = rng.integers(len(single_rules))
                candidate = single_rules[idx]
            elif r < p_single + p_conj and conj_rules:
                # Try a conjunctive rule
                idx = rng.integers(len(conj_rules))
                candidate = conj_rules[idx]
            else:
                continue

            eval_result = self._evaluate_rule(candidate, stimuli, labels)
            search_log.append(
                {
                    "step": step,
                    "rule": candidate,
                    "accuracy": eval_result["accuracy"],
                }
            )

            if eval_result["accuracy"] > best_accuracy:
                best_accuracy = eval_result["accuracy"]
                best_rule = candidate
                best_eval = eval_result

            # Accept rule if good enough
            if best_accuracy >= (1.0 - error_tolerance):
                break

        return {
            "rule": best_rule,
            "accuracy": best_accuracy,
            "eval": best_eval if best_rule else None,
            "search_steps": len(search_log),
            "search_log": search_log,
        }

    def predict(
        self,
        stimulus: np.ndarray,
        training_items: np.ndarray,
        training_labels: np.ndarray,
        p_single: float = 0.5,
        p_conj: float = 0.3,
        p_exception: float = 0.8,
        max_search_steps: int = 50,
        error_tolerance: float = 0.1,
        seed: Optional[int] = None,
    ) -> dict:
        """
        Predict category membership for a stimulus.

        1. Find best rule via stochastic search
        2. Apply rule to stimulus
        3. If stimulus matches a stored exception, override rule
        """
        stimulus = np.asarray(stimulus, dtype=float)

        # Find the best rule
        rule_result = self.find_best_rule(
            training_items,
            training_labels,
            p_single=p_single,
            p_conj=p_conj,
            max_search_steps=max_search_steps,
            error_tolerance=error_tolerance,
            seed=seed,
        )

        if rule_result["rule"] is None:
            # No rule found — guess
            categories = sorted(set(training_labels))
            return {
                "probabilities": {c: 1.0 / len(categories) for c in categories},
                "rule_used": None,
                "exception_match": False,
            }

        # Apply rule to test stimulus
        rule_eval = self._evaluate_rule(
            rule_result["rule"],
            stimulus.reshape(1, -1),
            np.array([0]),  # dummy label
        )
        rule_prediction = rule_eval["predictions"][0]

        # Check exception storage
        exception_match = False
        rng = np.random.default_rng(seed)
        if rule_result["eval"] is not None:
            exception_indices = rule_result["eval"]["error_indices"]
            for exc_idx in exception_indices:
                exc_item = training_items[exc_idx]
                if np.allclose(stimulus, exc_item):
                    # Stimulus matches a stored exception
                    if rng.random() < p_exception:
                        rule_prediction = training_labels[exc_idx]
                        exception_match = True
                    break

        # Convert to probabilities (RULEX is relatively deterministic)
        categories = sorted(set(training_labels))
        probs = {}
        for cat in categories:
            if cat == rule_prediction:
                probs[cat] = 0.9  # high confidence from rule
            else:
                probs[cat] = 0.1
        # Normalize
        total = sum(probs.values())
        probs = {cat: p / total for cat, p in probs.items()}

        return {
            "probabilities": probs,
            "rule_used": rule_result["rule"],
            "exception_match": exception_match,
            "rule_accuracy": rule_result["accuracy"],
            "search_steps": rule_result["search_steps"],
        }

    def predict_learning_curve(
        self,
        training_sequence: list[tuple[np.ndarray, int]],
        test_items: np.ndarray,
        test_labels: np.ndarray,
        block_size: int = 1,
        **params,
    ) -> list[dict]:
        """
        Simulate learning curve.

        RULEX predicts discrete transitions — poor performance until
        rule discovery, then a sudden jump. This is qualitatively
        different from GCM's gradual curve and SUSTAIN's step-wise.
        """
        p = {**self.default_params, **params}
        curve = []

        items_seen = []
        labels_seen = []

        for block_start in range(0, len(training_sequence), block_size):
            block = training_sequence[block_start : block_start + block_size]
            for stim, label in block:
                items_seen.append(stim)
                labels_seen.append(label)

            if len(items_seen) < 2:
                curve.append(
                    {
                        "block": block_start // block_size,
                        "accuracy": 0.5,
                        "rule_found": False,
                    }
                )
                continue

            train_arr = np.array(items_seen)
            label_arr = np.array(labels_seen)

            # Find best rule given items seen so far
            rule_result = self.find_best_rule(
                train_arr,
                label_arr,
                p_single=p["p_single"],
                p_conj=p["p_conj"],
                max_search_steps=p["max_search_steps"],
                error_tolerance=p["error_tolerance"],
                seed=p.get("seed"),
            )

            if rule_result["rule"] is None:
                acc = 0.5
                rule_found = False
            else:
                # Test on test items
                eval_result = self._evaluate_rule(
                    rule_result["rule"], test_items, test_labels
                )
                acc = eval_result["accuracy"]
                rule_found = True

            curve.append(
                {
                    "block": block_start // block_size,
                    "accuracy": acc,
                    "rule_found": rule_found,
                    "rule_accuracy_on_training": rule_result["accuracy"]
                    if rule_result["rule"]
                    else None,
                }
            )

        return curve
