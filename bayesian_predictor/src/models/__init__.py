"""
Models module for Bayesian Neural Network

This module exports the main model classes and convenience functions.
"""

from .bayesian_nn import (
    BayesianLinear,
    BayesianPredictor,
    BayesianELBOLoss,
    create_small_bayesian_model,
    create_medium_bayesian_model,
    create_large_bayesian_model
)

__all__ = [
    'BayesianLinear',
    'BayesianPredictor',
    'BayesianELBOLoss',
    'create_small_bayesian_model',
    'create_medium_bayesian_model',
    'create_large_bayesian_model'
]
