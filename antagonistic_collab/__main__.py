"""
Entry point for: python -m antagonistic_collab

Without arguments, runs the interactive debate.
With --demo, runs the formal layer demo (no API key needed).
"""

import argparse
import sys


def _build_argparser() -> argparse.ArgumentParser:
    """Build argument parser for CLI. Separate function for testability."""
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
        help="Experiment selection: bayesian (EIG) or heuristic (diversity penalty)",
    )
    parser.add_argument(
        "--mode",
        choices=["full_pool", "legacy"],
        default="legacy",
        help="Phase flow: full_pool (EIG + interpretation debate) or legacy (9-phase)",
    )
    parser.add_argument(
        "--hitl-checkpoints",
        action="store_true",
        default=False,
        help="Enable human-in-the-loop checkpoints (default: off)",
    )
    return parser


def _entry():
    if "--demo" in sys.argv:
        from .demo import (
            demo_model_predictions,
            demo_divergence_mapping,
            demo_epistemic_state,
            demo_full_cycle,
        )

        demo_model_predictions()
        demo_divergence_mapping()
        demo_epistemic_state()
        demo_full_cycle()
    else:
        from .runner import main

        main()


if __name__ == "__main__":
    _entry()
