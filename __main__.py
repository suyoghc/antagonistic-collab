"""
Entry point for: python -m antagonistic_collab

Without arguments, runs the interactive debate.
With --demo, runs the formal layer demo (no API key needed).
"""

import sys


def _entry():
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
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
