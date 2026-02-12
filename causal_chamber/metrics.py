"""
Causal Discovery Metrics.

Compares an estimated graph (adjacency matrix) against the ground truth DAG.
Implements: precision, recall, F1 of edges, and Structural Hamming Distance (SHD).

Optionally runs classical causal discovery algorithms (GES, PC via causal-learn)
as baselines, if the library is installed.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Optional, Tuple

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _to_binary(A: np.ndarray) -> np.ndarray:
    """Convert any adjacency matrix to binary (0/1)."""
    return (np.asarray(A) != 0).astype(int)


def edge_precision(estimated: np.ndarray, truth: np.ndarray) -> float:
    """
    Precision = TP / (TP + FP).

    An edge is a True Positive if it exists in both estimated and truth.
    """
    est = _to_binary(estimated)
    gt = _to_binary(truth)
    tp = np.logical_and(est, gt).sum()
    total_est = est.sum()
    if total_est == 0:
        return 0.0
    return float(tp / total_est)


def edge_recall(estimated: np.ndarray, truth: np.ndarray) -> float:
    """
    Recall = TP / (TP + FN).
    """
    est = _to_binary(estimated)
    gt = _to_binary(truth)
    tp = np.logical_and(est, gt).sum()
    total_gt = gt.sum()
    if total_gt == 0:
        return 1.0  # no edges to recover
    return float(tp / total_gt)


def edge_f1(estimated: np.ndarray, truth: np.ndarray) -> float:
    """F1 = 2 * precision * recall / (precision + recall)."""
    p = edge_precision(estimated, truth)
    r = edge_recall(estimated, truth)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def structural_hamming_distance(estimated: np.ndarray, truth: np.ndarray) -> int:
    """
    Structural Hamming Distance (SHD): number of edge insertions,
    deletions, and flips needed to transform estimated into truth.

    For directed graphs, we count mismatches in the adjacency matrix.
    """
    est = _to_binary(estimated)
    gt = _to_binary(truth)
    return int(np.sum(est != gt))


def compute_all_metrics(
    estimated: np.ndarray,
    truth: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all causal discovery metrics.

    Parameters
    ----------
    estimated : np.ndarray
        Estimated adjacency matrix.
    truth : np.ndarray
        Ground truth adjacency matrix.

    Returns
    -------
    dict
        Keys: 'precision', 'recall', 'f1', 'shd', 'n_edges_estimated', 'n_edges_truth'.
    """
    est = _to_binary(estimated)
    gt = _to_binary(truth)
    return {
        'precision': edge_precision(est, gt),
        'recall': edge_recall(est, gt),
        'f1': edge_f1(est, gt),
        'shd': structural_hamming_distance(est, gt),
        'n_edges_estimated': int(est.sum()),
        'n_edges_truth': int(gt.sum()),
    }


def confusion_edges(
    estimated: np.ndarray,
    truth: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Classify edges into True Positives, False Positives, False Negatives.

    Returns three binary matrices of the same shape:
    - tp: edges present in both
    - fp: edges present in estimated but not truth
    - fn: edges present in truth but not estimated
    """
    est = _to_binary(estimated)
    gt = _to_binary(truth)
    tp = np.logical_and(est, gt).astype(int)
    fp = np.logical_and(est, ~gt.astype(bool)).astype(int)
    fn = np.logical_and(~est.astype(bool), gt).astype(int)
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Classical causal discovery baselines (optional)
# ---------------------------------------------------------------------------

def _check_causal_learn():
    """Check if causal-learn is available."""
    try:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.search.ScoreBased.GES import ges
        return True
    except ImportError:
        return False


CAUSAL_LEARN_AVAILABLE = _check_causal_learn()


def run_pc_algorithm(
    data: np.ndarray,
    alpha: float = 0.05,
    node_names: Optional[list] = None,
) -> np.ndarray:
    """
    Run the PC algorithm from causal-learn.

    Parameters
    ----------
    data : np.ndarray
        Data matrix (n_samples, n_variables).
    alpha : float
        Significance level for conditional independence tests.
    node_names : list, optional
        Variable names.

    Returns
    -------
    np.ndarray
        Estimated adjacency matrix (standard convention: A[i,j]=1 means i->j).
    """
    if not CAUSAL_LEARN_AVAILABLE:
        warnings.warn("causal-learn not installed. PC algorithm unavailable.")
        return None

    from causallearn.search.ConstraintBased.PC import pc

    cg = pc(data, alpha=alpha, node_names=node_names)
    # cg.G.graph is the adjacency matrix in causal-learn format
    # Convert: -1 means tail, 1 means arrowhead
    G = cg.G.graph
    n = G.shape[0]
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if G[i, j] == -1 and G[j, i] == 1:
                # i -> j
                adj[i, j] = 1
    return adj


def run_ges_algorithm(
    data: np.ndarray,
    node_names: Optional[list] = None,
) -> np.ndarray:
    """
    Run the GES algorithm from causal-learn.

    Parameters
    ----------
    data : np.ndarray
        Data matrix (n_samples, n_variables).
    node_names : list, optional
        Variable names.

    Returns
    -------
    np.ndarray
        Estimated adjacency matrix (standard convention: A[i,j]=1 means i->j).
    """
    if not CAUSAL_LEARN_AVAILABLE:
        warnings.warn("causal-learn not installed. GES algorithm unavailable.")
        return None

    from causallearn.search.ScoreBased.GES import ges

    record = ges(data)
    G = record['G'].graph
    n = G.shape[0]
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if G[i, j] == -1 and G[j, i] == 1:
                adj[i, j] = 1
    return adj


def run_classical_baselines(
    data: np.ndarray,
    node_names: Optional[list] = None,
    alpha: float = 0.05,
) -> Dict[str, Optional[np.ndarray]]:
    """
    Run available classical causal discovery algorithms.

    Returns
    -------
    dict
        Keys are algorithm names, values are adjacency matrices (or None if unavailable).
    """
    results = {}

    if CAUSAL_LEARN_AVAILABLE:
        try:
            results['PC'] = run_pc_algorithm(data, alpha=alpha, node_names=node_names)
        except Exception as e:
            warnings.warn(f"PC algorithm failed: {e}")
            results['PC'] = None

        try:
            results['GES'] = run_ges_algorithm(data, node_names=node_names)
        except Exception as e:
            warnings.warn(f"GES algorithm failed: {e}")
            results['GES'] = None
    else:
        warnings.warn(
            "causal-learn not installed. Classical baselines (PC, GES) skipped. "
            "Install with: pip install causal-learn"
        )

    return results
