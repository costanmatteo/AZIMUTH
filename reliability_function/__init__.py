"""
Reliability Function - Mathematical computation of process chain reliability F.

This module provides the mathematical formula for computing reliability F
from process chain trajectories. Used as ground truth for training
CasualiT surrogate and as an option in controller optimization.
"""

from .src.compute_reliability import compute_reliability, ReliabilityFunction
from .configs.process_targets import PROCESS_CONFIGS

__all__ = ['compute_reliability', 'ReliabilityFunction', 'PROCESS_CONFIGS']
