"""
Causal discovery evaluation metrics.

Provides Structural Hamming Distance (SHD), precision, recall, and F1 for
comparing an estimated graph against a ground truth adjacency matrix.
These are the standard metrics used in causal-chamber-paper.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def structural_hamming_distance(
    adj_true: np.ndarray,
    adj_est: np.ndarray,
    ignore_diagonal: bool = True,
) -> int:
    """Structural Hamming Distance between two adjacency matrices.

    SHD counts the number of edge additions, deletions, and reversals needed
    to transform ``adj_est`` into ``adj_true``.  For *directed* graphs this
    simplifies to the number of entries where the two binary matrices differ.

    Parameters
    ----------
    adj_true, adj_est : ndarray
        Binary adjacency matrices of equal shape.
        Convention: ``A[i, j] = 1``  <=>  ``j -> i``.
    ignore_diagonal : bool
        If True, self-loops are ignored.

    Returns
    -------
    int
        SHD value (lower is better; 0 = perfect).
    """
    A = np.asarray(adj_true, dtype=int)
    B = np.asarray(adj_est, dtype=int)
    assert A.shape == B.shape, f"Shape mismatch: {A.shape} vs {B.shape}"

    diff = np.abs(A - B)
    if ignore_diagonal:
        np.fill_diagonal(diff, 0)
    return int(diff.sum())


def edge_precision_recall_f1(
    adj_true: np.ndarray,
    adj_est: np.ndarray,
    ignore_diagonal: bool = True,
) -> Dict[str, float]:
    """Precision, recall, and F1 for edge detection.

    An *edge* is any position ``(i, j)`` with value 1 in the matrix
    (excluding diagonal if ``ignore_diagonal`` is True).

    Parameters
    ----------
    adj_true, adj_est : ndarray
        Binary adjacency matrices.

    Returns
    -------
    dict
        ``{"precision": ..., "recall": ..., "f1": ...}``
        Each value in ``[0, 1]``.
    """
    A = np.asarray(adj_true, dtype=int)
    B = np.asarray(adj_est, dtype=int)
    assert A.shape == B.shape

    if ignore_diagonal:
        mask = ~np.eye(A.shape[0], dtype=bool)
    else:
        mask = np.ones(A.shape, dtype=bool)

    true_edges = A[mask]
    est_edges = B[mask]

    tp = int(np.sum((true_edges == 1) & (est_edges == 1)))
    fp = int(np.sum((true_edges == 0) & (est_edges == 1)))
    fn = int(np.sum((true_edges == 1) & (est_edges == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_all_metrics(
    adj_true: np.ndarray,
    adj_est: np.ndarray,
    ignore_diagonal: bool = True,
) -> Dict[str, float]:
    """Compute SHD, precision, recall, and F1 in one call.

    Parameters
    ----------
    adj_true, adj_est : ndarray
        Binary adjacency matrices.

    Returns
    -------
    dict
        ``{"shd": int, "precision": float, "recall": float, "f1": float,
           "n_true_edges": int, "n_est_edges": int}``
    """
    shd = structural_hamming_distance(adj_true, adj_est, ignore_diagonal)
    prf = edge_precision_recall_f1(adj_true, adj_est, ignore_diagonal)

    mask = ~np.eye(adj_true.shape[0], dtype=bool) if ignore_diagonal else np.ones(adj_true.shape, dtype=bool)

    return {
        "shd": shd,
        **prf,
        "n_true_edges": int(np.asarray(adj_true, dtype=int)[mask].sum()),
        "n_est_edges": int(np.asarray(adj_est, dtype=int)[mask].sum()),
    }


# ---------------------------------------------------------------------------
# DataFrame-aware wrappers
# ---------------------------------------------------------------------------

def _align_matrices(
    adj_true: np.ndarray | pd.DataFrame,
    adj_est: np.ndarray | pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Align two adjacency matrices (possibly DataFrames with different
    variable orderings) to a common variable set.

    Missing variables in either matrix are treated as having no edges.

    Returns
    -------
    A_true, A_est : ndarray
        Aligned binary matrices.
    labels : list[str]
        Variable labels in the aligned order.
    """
    if isinstance(adj_true, pd.DataFrame) and isinstance(adj_est, pd.DataFrame):
        all_vars = list(dict.fromkeys(list(adj_true.index) + list(adj_est.index)))
        n = len(all_vars)
        A_true = np.zeros((n, n), dtype=int)
        A_est = np.zeros((n, n), dtype=int)

        t_idx = {v: i for i, v in enumerate(adj_true.index)}
        e_idx = {v: i for i, v in enumerate(adj_est.index)}
        a_idx = {v: i for i, v in enumerate(all_vars)}

        for v in adj_true.index:
            for u in adj_true.columns:
                if u in a_idx and v in a_idx:
                    A_true[a_idx[v], a_idx[u]] = int(adj_true.loc[v, u])

        for v in adj_est.index:
            for u in adj_est.columns:
                if u in a_idx and v in a_idx:
                    A_est[a_idx[v], a_idx[u]] = int(adj_est.loc[v, u])

        return A_true, A_est, all_vars

    # Both plain arrays
    return np.asarray(adj_true, dtype=int), np.asarray(adj_est, dtype=int), []


def compare_graphs(
    adj_true: np.ndarray | pd.DataFrame,
    adj_est: np.ndarray | pd.DataFrame,
    label: str = "",
) -> Dict[str, float | str]:
    """Compare two graphs, optionally aligning variable sets.

    Parameters
    ----------
    adj_true, adj_est : ndarray or DataFrame
        Ground truth and estimated adjacency matrices.
    label : str
        Optional label for the comparison (e.g. method name).

    Returns
    -------
    dict
        Metrics dict augmented with ``"label"`` key.
    """
    A_true, A_est, _ = _align_matrices(adj_true, adj_est)
    result = compute_all_metrics(A_true, A_est)
    if label:
        result["label"] = label
    return result


# ---------------------------------------------------------------------------
# Optional: classical baselines via causal-learn
# ---------------------------------------------------------------------------

def run_classical_discovery(
    data: np.ndarray,
    method: str = "GES",
    variable_names: Optional[list[str]] = None,
) -> np.ndarray:
    """Run a classical causal discovery algorithm as baseline.

    Requires the ``causal-learn`` package.

    Parameters
    ----------
    data : ndarray
        ``(n_samples, n_vars)`` i.i.d. observations.
    method : str
        ``"GES"`` (Greedy Equivalence Search) or ``"PC"``.
    variable_names : list[str], optional
        Names for each column (for labelling only).

    Returns
    -------
    adj : ndarray
        Estimated adjacency matrix.
    """
    try:
        if method.upper() == "GES":
            from causallearn.search.ScoreBased.GES import ges
            record = ges(data)
            # GES returns a GeneralGraph; extract adjacency
            adj_raw = record["G"].graph
        elif method.upper() == "PC":
            from causallearn.search.ConstraintBased.PC import pc
            cg = pc(data)
            adj_raw = cg.G.graph
        else:
            raise ValueError(f"Unsupported method: {method}. Use 'GES' or 'PC'.")
    except ImportError:
        raise ImportError(
            f"causal-learn is required for classical discovery ({method}). "
            "Install with: pip install causal-learn"
        )

    # causal-learn uses: -1 = tail, 1 = arrowhead, 0 = no edge
    # Convert to binary: A[i,j]=1 if j->i
    n = adj_raw.shape[0]
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if adj_raw[i, j] == -1 and adj_raw[j, i] == 1:
                # j -> i
                adj[i, j] = 1
            elif adj_raw[i, j] == 1 and adj_raw[j, i] == -1:
                # i -> j
                adj[j, i] = 1
            elif adj_raw[i, j] == -1 and adj_raw[j, i] == -1:
                # undirected: count as bidirectional
                adj[i, j] = 1
                adj[j, i] = 1

    return adj
