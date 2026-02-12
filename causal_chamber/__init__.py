"""
Causal Chamber — Causal Validation Module for Azimuth.

Inspired by the Causal Chamber framework (Gamella, Peters, Buhlmann —
Nature Machine Intelligence, 2025), adapted for validating CausaliT
attention-based causal discovery against SCM ground truth.

This module is independent: it imports from scm_ds, causaliT,
reliability_function, and uncertainty_predictor, but does not modify them.
"""

__version__ = '0.1.0'

from causal_chamber.ground_truth import (
    get_ground_truth_edges,
    get_ground_truth_adjacency,
    get_ground_truth_adjacency_parent_convention,
    get_all_observable_vars,
    PROCESS_ORDER,
    PROCESS_DATASETS,
    PROCESS_OBSERVABLE_VARS,
)

from causal_chamber.metrics import (
    compute_all_metrics,
    edge_precision,
    edge_recall,
    edge_f1,
    structural_hamming_distance,
)

__all__ = [
    # Ground truth
    'get_ground_truth_edges',
    'get_ground_truth_adjacency',
    'get_ground_truth_adjacency_parent_convention',
    'get_all_observable_vars',
    'PROCESS_ORDER',
    'PROCESS_DATASETS',
    'PROCESS_OBSERVABLE_VARS',
    # Metrics
    'compute_all_metrics',
    'edge_precision',
    'edge_recall',
    'edge_f1',
    'structural_hamming_distance',
]
