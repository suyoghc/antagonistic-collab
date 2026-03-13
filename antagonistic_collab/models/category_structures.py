"""
Standard category structures used in the categorization literature.

Each structure is returned as a dict:
    {
        "stimuli": np.ndarray of shape (n_items, n_dims),
        "labels": np.ndarray of shape (n_items,),  # 0 or 1
        "dim_names": list[str],
        "name": str,
        "description": str,
    }
"""

import numpy as np
from typing import Optional


def shepard_types() -> dict[str, dict]:
    """
    Shepard, Hovland & Jenkins (1961) six category structure types.
    3 binary dimensions, 8 stimuli split into two categories of 4.

    Type I:   Single dimension rule (easiest)
    Type II:  XOR on two dimensions
    Type III: Single dimension + exception
    Type IV:  Biconditional-like
    Type V:   Complex with 2 exceptions
    Type VI:  No simple rule (hardest for rules, fine for exemplars)

    Classic ordering: I < II < III ≈ IV ≈ V < VI
    But the *relative* ordering of II vs III-V is theory-diagnostic.
    """
    # All 8 stimuli in 3 binary dimensions
    stimuli = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=float,
    )

    type_labels = {
        "I": np.array([0, 0, 0, 0, 1, 1, 1, 1]),  # dim 1 rule
        "II": np.array([0, 1, 1, 0, 1, 0, 0, 1]),  # XOR on dims 1,2
        "III": np.array([0, 0, 0, 1, 1, 1, 1, 0]),  # dim 1 + exception
        "IV": np.array([0, 0, 1, 1, 1, 0, 0, 1]),  # family resemblance variant
        "V": np.array([0, 0, 1, 0, 1, 1, 0, 1]),  # complex
        "VI": np.array([0, 1, 1, 0, 0, 1, 1, 0]),  # parity (no simple rule)
    }

    structures = {}
    for type_name, labels in type_labels.items():
        structures[type_name] = {
            "stimuli": stimuli.copy(),
            "labels": labels.copy(),
            "dim_names": ["D1", "D2", "D3"],
            "name": f"Shepard Type {type_name}",
            "description": f"Shepard et al. (1961) Type {type_name} category structure.",
        }
    return structures


def five_four_structure() -> dict:
    """
    Medin & Schaffer (1978) 5-4 category structure.
    4 binary dimensions. Category A has 5 members, B has 4.
    Neither category has a prototype that is a member.

    This structure famously favors exemplar models over prototype models:
    the prototype of each category is NOT a category member, so a prototype
    model predicts classification of category members poorly.
    """
    stimuli = np.array(
        [
            # Category A (5 items)
            [1, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 1],
            # Category B (4 items)
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        dtype=float,
    )

    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])

    return {
        "stimuli": stimuli,
        "labels": labels,
        "dim_names": ["D1", "D2", "D3", "D4"],
        "name": "5-4 Structure",
        "description": (
            "Medin & Schaffer (1978). 5 items in category A, 4 in B. "
            "Neither prototype is a category member. Favors exemplar models."
        ),
    }


def rule_plus_exception(
    n_dims: int = 4,
    n_items_per_category: int = 8,
    n_exceptions: int = 1,
    seed: Optional[int] = None,
) -> dict:
    """
    Generate a rule-plus-exception structure.
    One dimension defines the rule; n_exceptions items violate it.

    Diagnostic: Rule models handle the rule, struggle with exceptions.
    Exemplar models handle exceptions via similarity to stored instances.
    SUSTAIN recruits extra clusters for exceptions.
    """
    rng = np.random.default_rng(seed)
    n_total = n_items_per_category * 2

    # Generate random binary features for non-rule dimensions
    stimuli = rng.integers(0, 2, size=(n_total, n_dims)).astype(float)

    # Dimension 0 is the rule: dim0=0 -> cat A, dim0=1 -> cat B
    stimuli[:n_items_per_category, 0] = 0
    stimuli[n_items_per_category:, 0] = 1
    labels = np.array([0] * n_items_per_category + [1] * n_items_per_category)

    # Introduce exceptions: flip the label for n_exceptions items per category
    exc_a = rng.choice(n_items_per_category, n_exceptions, replace=False)
    exc_b = rng.choice(
        range(n_items_per_category, n_total), n_exceptions, replace=False
    )
    labels[exc_a] = 1
    labels[exc_b] = 0

    return {
        "stimuli": stimuli,
        "labels": labels,
        "dim_names": [f"D{i + 1}" for i in range(n_dims)],
        "name": f"Rule-plus-exception ({n_exceptions} exceptions)",
        "description": (
            f"Rule on D1 with {n_exceptions} exception(s) per category. "
            "Diagnostic for rule vs. exemplar vs. clustering accounts."
        ),
        "rule_dimension": 0,
        "exception_indices": np.concatenate([exc_a, exc_b]).tolist(),
    }


def linear_separable(
    n_dims: int = 2,
    n_items_per_category: int = 10,
    separation: float = 2.0,
    seed: Optional[int] = None,
) -> dict:
    """
    Generate a linearly separable category structure in continuous space.

    Two Gaussian clusters separated along the first dimension.
    Useful for testing boundary placement and generalization gradients.
    """
    rng = np.random.default_rng(seed)

    cat_a = rng.normal(
        loc=[-separation / 2] + [0] * (n_dims - 1),
        scale=1.0,
        size=(n_items_per_category, n_dims),
    )
    cat_b = rng.normal(
        loc=[separation / 2] + [0] * (n_dims - 1),
        scale=1.0,
        size=(n_items_per_category, n_dims),
    )

    stimuli = np.vstack([cat_a, cat_b])
    labels = np.array([0] * n_items_per_category + [1] * n_items_per_category)

    return {
        "stimuli": stimuli,
        "labels": labels,
        "dim_names": [f"D{i + 1}" for i in range(n_dims)],
        "name": f"Linear separable ({n_dims}D, sep={separation})",
        "description": (
            f"Two Gaussian clusters in {n_dims}D, separation={separation}. "
            "Linearly separable. Good for testing decision boundaries."
        ),
    }


def make_structure(
    stimuli: np.ndarray,
    labels: np.ndarray,
    name: str = "Custom",
    description: str = "",
    dim_names: Optional[list[str]] = None,
) -> dict:
    """Convenience wrapper for creating a structure dict from arrays."""
    stimuli = np.asarray(stimuli, dtype=float)
    labels = np.asarray(labels)
    if dim_names is None:
        dim_names = [f"D{i + 1}" for i in range(stimuli.shape[1])]
    return {
        "stimuli": stimuli,
        "labels": labels,
        "dim_names": dim_names,
        "name": name,
        "description": description,
    }
