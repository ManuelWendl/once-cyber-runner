"""PyTorch DreamerV3 (ported from efficient-exploration-for-cyberrunner-pmlr).

Plain DreamerV3 (no OPAX/exploration bonuses) with the CyberRunner
mirror-symmetry replay augmentation from arXiv 2312.09906.
"""

from .agent import Dreamer
from .buffer import Buffer

__all__ = ["Dreamer", "Buffer"]
