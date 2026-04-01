"""
CasualiT Surrogate Training Package.

This package provides tools for training a TransformerForecaster to predict
reliability F from process chain trajectories.

Usage:
    python train_surrogate.py [options]

Or import components:
    from causaliT.surrogate_training.data_generator import TrajectoryDataGenerator
    from train_surrogate import SurrogateTrainer
"""

from configs.surrogate_config import SURROGATE_CONFIG

__all__ = ['SURROGATE_CONFIG']
