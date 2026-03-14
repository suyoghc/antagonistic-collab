"""
SUSTAIN (Supervised and Unsupervised STratified Adaptive Incremental Network)
Love, Medin & Gureckis (2004).

A clustering model: the learner maintains a set of clusters (recruited
adaptively) and classifies stimuli based on similarity to existing clusters.
New clusters are recruited when prediction fails — "surprising" items
trigger the creation of a new cluster centered on that item.

Key predictions:
- Sensitivity to within-category structure (not just boundaries)
- Cluster recruitment at exceptions and surprising items
- Intermediate between exemplar and prototype predictions
- Unsupervised learning creates different clusters than supervised

Parameters:
    r (float): Attentional focus parameter. Higher = sharper receptive fields.
    beta (float): Cluster competition parameter. Higher = more winner-take-all.
    d (float): Decision consistency / response determinism.
    eta (float): Learning rate for cluster-to-category association weights.
    tau (float): Threshold for cluster recruitment. Lower = recruit more easily.
    initial_lambdas (float): Initial dimensional attention strengths.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class Cluster:
    """A single SUSTAIN cluster."""

    position: np.ndarray  # centroid in stimulus space
    lambdas: np.ndarray  # per-dimension receptive field tuning
    associations: dict  # {category_label: weight}
    n_updates: int = 0


class SUSTAIN:
    """SUSTAIN model for categorization."""

    name = "SUSTAIN (Clustering Model)"
    description = (
        "Love, Medin & Gureckis (2004). Categorization by adaptive clustering. "
        "Recruits new clusters for surprising items. Predicts sensitivity to "
        "within-category distribution, flexible number of representations, "
        "and different learning dynamics than pure exemplar or prototype models."
    )
    core_claims = [
        "Learners form a flexible number of clusters (neither one prototype nor N exemplars).",
        "New clusters are recruited when current clusters fail to predict correctly.",
        "Attention is learned — dimensions diagnostic of category membership get more weight.",
        "The number of clusters recruited depends on category structure complexity.",
    ]

    def __init__(self):
        self.default_params = {
            "r": 9.01,
            "beta": 1.252,
            "d": 16.924,
            "eta": 0.092,
            "tau": 0.0,  # always recruit on error (standard setting)
            "initial_lambdas": 1.0,
        }

    def _activation(self, stimulus: np.ndarray, cluster: Cluster, r: float) -> float:
        """
        Compute activation of a cluster given a stimulus.
        Uses a dimension-weighted similarity function.
        """
        # Per-dimension similarity
        dim_sim = np.exp(-cluster.lambdas * np.abs(stimulus - cluster.position))
        # Aggregate with attentional focus
        lambda_sum = np.sum(cluster.lambdas)
        if lambda_sum == 0:
            act = float(np.mean(dim_sim))
        else:
            act = np.sum(cluster.lambdas * dim_sim) / lambda_sum
        return act**r

    def _output(
        self, activations: np.ndarray, cluster_list: list[Cluster], beta: float
    ) -> dict:
        """
        Compute category output given cluster activations.
        Lateral inhibition (softmax over clusters), then sum associations.
        """
        if len(activations) == 0:
            return {}

        # Cluster competition (softmax)
        scaled = beta * activations
        scaled -= scaled.max()  # numerical stability
        exp_act = np.exp(scaled)
        if exp_act.sum() == 0:
            cluster_weights = np.ones_like(exp_act) / len(exp_act)
        else:
            cluster_weights = exp_act / exp_act.sum()

        # Sum weighted associations across clusters
        categories = set()
        for c in cluster_list:
            categories.update(c.associations.keys())

        output = {}
        for cat in categories:
            output[cat] = sum(
                cluster_weights[i] * cluster_list[i].associations.get(cat, 0.0)
                for i in range(len(cluster_list))
            )
        return output

    def _recruit_cluster(
        self,
        stimulus: np.ndarray,
        label: int,
        initial_lambdas: float,
        categories: set,
    ) -> Cluster:
        """Create a new cluster centered on the stimulus."""
        n_dims = len(stimulus)
        associations = {cat: 0.0 for cat in categories}
        associations[label] = 1.0  # initial association to correct category
        return Cluster(
            position=stimulus.copy(),
            lambdas=np.full(n_dims, initial_lambdas),
            associations=associations,
            n_updates=1,
        )

    def simulate_learning(
        self,
        training_sequence: list[tuple[np.ndarray, int]],
        r: float = 9.01,
        beta: float = 1.252,
        d: float = 16.924,
        eta: float = 0.092,
        tau: float = 0.0,
        initial_lambdas: float = 1.0,
    ) -> dict:
        """
        Simulate SUSTAIN learning on a training sequence.

        Returns:
            dict with:
                "clusters": final list of Cluster objects
                "trial_log": list of per-trial info (correct, recruited, n_clusters)
                "lambdas": final attention weights
        """
        if not training_sequence:
            return {
                "clusters": [],
                "trial_log": [],
                "lambdas": np.array([]),
                "n_clusters_final": 0,
            }

        clusters: list[Cluster] = []
        categories = set(label for _, label in training_sequence)
        n_dims = len(training_sequence[0][0])

        # Global attention strengths (learned)
        lambdas = np.full(n_dims, initial_lambdas)

        trial_log = []

        for trial_idx, (stimulus, label) in enumerate(training_sequence):
            stimulus = np.asarray(stimulus, dtype=float)

            if len(clusters) == 0:
                # First trial: create first cluster
                new_cluster = self._recruit_cluster(
                    stimulus, label, initial_lambdas, categories
                )
                new_cluster.lambdas = lambdas.copy()
                clusters.append(new_cluster)
                trial_log.append(
                    {
                        "trial": trial_idx,
                        "correct": True,  # trivially correct with one cluster
                        "recruited": True,
                        "n_clusters": 1,
                        "winning_cluster": 0,
                    }
                )
                continue

            # Compute activations for all existing clusters
            activations = np.array([self._activation(stimulus, c, r) for c in clusters])

            # Find winning cluster
            winner_idx = np.argmax(activations)
            winner = clusters[winner_idx]

            # Compute category output
            cat_output = self._output(activations, clusters, beta)

            # Apply response rule (softmax over category outputs)
            if cat_output:
                max_out = max(cat_output.values())
                exp_outs = {
                    cat: np.exp(d * (val - max_out)) for cat, val in cat_output.items()
                }
                total = sum(exp_outs.values())
                probs = {cat: exp_outs[cat] / total for cat, val in cat_output.items()}
            else:
                probs = {cat: 1.0 / len(categories) for cat in categories}

            predicted_label = max(probs, key=probs.get)
            correct = predicted_label == label

            # Determine whether to recruit new cluster
            recruited = False
            if not correct:
                # Humbled: prediction was wrong
                if probs.get(label, 0.0) < tau or tau == 0.0:
                    # Recruit new cluster
                    new_cluster = self._recruit_cluster(
                        stimulus, label, initial_lambdas, categories
                    )
                    new_cluster.lambdas = lambdas.copy()
                    clusters.append(new_cluster)
                    recruited = True
                    winner_idx = len(clusters) - 1
                    winner = clusters[winner_idx]

            # Learn: update winning cluster's position, associations, attention
            if not recruited:
                # Move winner toward stimulus
                winner.position += eta * (stimulus - winner.position)

                # Update associations via delta rule
                for cat in categories:
                    target = 1.0 if cat == label else 0.0
                    error = target - winner.associations.get(cat, 0.0)
                    winner.associations[cat] = (
                        winner.associations.get(cat, 0.0) + eta * error
                    )

                winner.n_updates += 1

            # Update attention (dimension-wise error-driven)
            # Increase attention to dimensions that reduce error
            for c_idx, c in enumerate(clusters):
                for dim in range(n_dims):
                    dim_sim = np.exp(
                        -c.lambdas[dim] * abs(stimulus[dim] - c.position[dim])
                    )
                    # Gradient: increase lambda for dimensions that help discriminate
                    c.lambdas[dim] += (
                        eta * (1.0 - dim_sim) * activations[c_idx]
                        if c_idx < len(activations)
                        else 0
                    )
                    c.lambdas[dim] = max(c.lambdas[dim], 0.01)  # floor

            lambdas = clusters[winner_idx].lambdas.copy()

            trial_log.append(
                {
                    "trial": trial_idx,
                    "correct": correct,
                    "recruited": recruited,
                    "n_clusters": len(clusters),
                    "winning_cluster": winner_idx,
                    "response_probs": probs,
                }
            )

        return {
            "clusters": clusters,
            "trial_log": trial_log,
            "lambdas": lambdas,
            "n_clusters_final": len(clusters),
        }

    def predict(
        self,
        stimulus: np.ndarray,
        training_items: np.ndarray,
        training_labels: np.ndarray,
        **params,
    ) -> dict:
        """
        Predict category probabilities for a stimulus after training.
        Trains on the provided items, then classifies the stimulus.
        """
        # Build training sequence (random order, single pass)
        sequence = list(zip(training_items, training_labels))

        # Use provided params or defaults
        p = {**self.default_params, **params}

        result = self.simulate_learning(
            sequence,
            r=p["r"],
            beta=p["beta"],
            d=p["d"],
            eta=p["eta"],
            tau=p["tau"],
            initial_lambdas=p["initial_lambdas"],
        )

        clusters = result["clusters"]
        if not clusters:
            categories = sorted(set(training_labels))
            return {"probabilities": {c: 1.0 / len(categories) for c in categories}}

        # Classify the test stimulus
        activations = np.array(
            [self._activation(np.asarray(stimulus), c, p["r"]) for c in clusters]
        )
        cat_output = self._output(activations, clusters, p["beta"])

        if cat_output:
            max_out = max(cat_output.values())
            exp_outs = {
                cat: np.exp(p["d"] * (val - max_out)) for cat, val in cat_output.items()
            }
            total = sum(exp_outs.values())
            probs = {cat: exp_outs[cat] / total for cat in cat_output}
        else:
            categories = sorted(set(training_labels))
            probs = {c: 1.0 / len(categories) for c in categories}

        return {
            "probabilities": probs,
            "n_clusters": len(clusters),
            "cluster_positions": [c.position.tolist() for c in clusters],
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
        Predict learning curve by simulating incremental learning.
        Unlike GCM, SUSTAIN is inherently sequential — order matters.

        At each block boundary, the model is tested on test_items/test_labels
        to produce held-out accuracy (matching GCM's contract).
        """
        p = {**self.default_params, **params}
        sim_params = {
            k: p[k] for k in ["r", "beta", "d", "eta", "tau", "initial_lambdas"]
        }

        test_items = np.asarray(test_items)
        test_labels = np.asarray(test_labels)

        curve = []
        block_ends = list(range(block_size, len(training_sequence) + 1, block_size))
        # Include the final partial block if the sequence isn't evenly divisible
        if len(training_sequence) % block_size != 0:
            block_ends.append(len(training_sequence))
        for block_end in block_ends:
            # Train on items up to this block
            partial_seq = training_sequence[:block_end]
            result = self.simulate_learning(partial_seq, **sim_params)
            clusters = result["clusters"]
            n_clusters = len(clusters)

            # Test on held-out items
            if len(test_items) == 0 or not clusters:
                continue

            correct = 0
            for test_item, true_label in zip(test_items, test_labels):
                activations = np.array(
                    [
                        self._activation(np.asarray(test_item), c, p["r"])
                        for c in clusters
                    ]
                )
                cat_output = self._output(activations, clusters, p["beta"])
                if cat_output:
                    pred_label = max(cat_output, key=cat_output.get)
                    if pred_label == true_label:
                        correct += 1

            curve.append(
                {
                    "block": (block_end // block_size) - 1,
                    "accuracy": float(correct / len(test_items)),
                    "n_clusters": n_clusters,
                }
            )

        return curve
