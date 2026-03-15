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
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file. Defaults are loaded from default_config.yaml.",
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
        "--selection-strategy",
        choices=["thompson", "greedy"],
        default="thompson",
        help="EIG selection strategy: thompson (sample proportional, default) or greedy (argmax)",
    )
    parser.add_argument(
        "--hitl-checkpoints",
        action="store_true",
        default=False,
        help="Enable human-in-the-loop checkpoints (default: off)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.005,
        help="Likelihood tempering rate (0, 1]. Lower values slow posterior convergence. Default 0.005.",
    )
    parser.add_argument(
        "--no-tempering",
        action="store_true",
        default=False,
        help="Disable likelihood tempering (sets learning-rate to 1.0).",
    )
    parser.add_argument(
        "--no-arbiter",
        action="store_true",
        default=False,
        help="Disable ARBITER features (crux negotiation, meta-agents, conflict map).",
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
