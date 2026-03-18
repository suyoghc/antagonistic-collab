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
    parser.add_argument(
        "--crux-weight",
        type=float,
        default=0.3,
        help="Probability of crux-directed selection in Thompson sampling [0,1]. Default 0.3.",
    )
    parser.add_argument(
        "--no-claim-responsive",
        action="store_true",
        default=False,
        help="Disable claim-responsive debate (agents won't be directed to address falsified claims).",
    )
    parser.add_argument(
        "--design-space",
        choices=["base", "richer", "continuous", "open"],
        default="continuous",
        help="Design space: base (55), richer (168 fixed grid), continuous (sampled each cycle, default), open (agent-proposed only, M16).",
    )
    parser.add_argument(
        "--n-continuous-samples",
        type=int,
        default=50,
        help="Number of structures sampled per cycle in continuous mode (default: 50).",
    )
    parser.add_argument(
        "--no-debate",
        action="store_true",
        default=False,
        help="Skip all LLM phases; run only computational pipeline (EIG + models + posterior).",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Path to YAML experiment config. Overrides all other flags and runs a multi-condition experiment.",
    )
    # Deprecated alias — kept for backward compatibility
    parser.add_argument(
        "--no-richer-design-space",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,  # hidden from help; maps to design_space=base
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
    elif "--experiment" in sys.argv:
        # Extract experiment path from argv
        idx = sys.argv.index("--experiment")
        if idx + 1 >= len(sys.argv):
            print("ERROR: --experiment requires a YAML config path")
            sys.exit(1)
        yaml_path = sys.argv[idx + 1]
        from .experiment import run_experiment

        run_experiment(yaml_path)
    elif "--merge" in sys.argv:
        # Merge summary JSONs: --merge path1 path2 ... [--output merged.json]
        idx = sys.argv.index("--merge")
        remaining = sys.argv[idx + 1 :]
        output = None
        if "--output" in remaining:
            oi = remaining.index("--output")
            output = remaining[oi + 1]
            remaining = remaining[:oi] + remaining[oi + 2 :]
        if not remaining:
            print("ERROR: --merge requires at least one summary.json path or directory")
            sys.exit(1)
        from .experiment import merge_summaries

        merge_summaries(*remaining, output=output)
    else:
        from .runner import main

        main()


if __name__ == "__main__":
    _entry()
