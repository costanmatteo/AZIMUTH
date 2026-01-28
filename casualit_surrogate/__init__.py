"""
CasualiT Surrogate - Learned reliability prediction using CasualiT transformer.

This module provides a CasualiT-based surrogate that learns to predict
reliability F from process chain trajectories. Trained on data labeled
by the mathematical reliability_function.

Usage:
    from casualit_surrogate import CasualiTSurrogate

    surrogate = CasualiTSurrogate.load('checkpoints/best_model.ckpt')
    F = surrogate.predict_reliability(trajectory)
"""

from .src.models.casualit_surrogate import CasualiTSurrogate

__all__ = ['CasualiTSurrogate']
