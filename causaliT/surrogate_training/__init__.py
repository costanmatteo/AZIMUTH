"""
CasualiT Surrogate Training Package.

This package provides tools for training a TransformerForecaster to predict
reliability F from process chain trajectories.

Usage:
    python -m causaliT.surrogate_training.train_surrogate [options]

Or import components:
    from causaliT.surrogate_training.data_generator import TrajectoryDataGenerator
    from causaliT.surrogate_training.train_surrogate import SurrogateTrainer
"""

from causaliT.surrogate_training.configs.surrogate_config import SURROGATE_CONFIG

__all__ = ['SURROGATE_CONFIG']
