"""
Antagonistic Collaboration Framework for Automated Scientific Debate.

A dual-layer architecture for adversarial multi-agent scientific discourse:
- Semantic layer: LLM agents argue in natural language (judgment, interpretation)
- Formal layer: Executable models generate quantitative predictions (accountability)

The epistemic state tracker maintains a cumulative, structured record of
what the debate has established — replacing flat chat histories with
organized scientific knowledge.
"""

from .epistemic_state import EpistemicState, TheoryCommitment, ModelClaim
from .debate_protocol import DebateProtocol, Phase, default_agent_configs

__all__ = [
    "EpistemicState", "TheoryCommitment", "ModelClaim",
    "DebateProtocol", "Phase", "default_agent_configs",
]
