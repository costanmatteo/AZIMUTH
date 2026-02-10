"""
Causal Discovery Metrics.

Implements metrics for comparing an estimated causal graph against a
ground-truth DAG, following the evaluation approach of
causal-chamber-paper:

* **Precision / Recall / F1** over edges
* **Structural Hamming Distance (SHD)**

All functions accept adjacency matrices in either ``np.ndarray`` or
``pd.DataFrame`` form (with matching variable order).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd


def _to_array(adj: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """Normalise input to a binary numpy array."""
    if isinstance(adj, pd.DataFrame):
        return adj.values.astype(int)
    return np.asarray(adj, dtype=int)


def _align_dataframes(
    estimated: pd.DataFrame,
    true: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align two DataFrames to a common variable ordering.

    Variables present in one but not the other are dropped.
    """
    common_vars = [v for v in true.columns if v in estimated.columns]
    est = estimated.loc[common_vars, common_vars].values.astype(int)
    gt = true.loc[common_vars, common_vars].values.astype(int)
    return est, gt


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def structural_hamming_distance(
    estimated: Union[np.ndarray, pd.DataFrame],
    true: Union[np.ndarray, pd.DataFrame],
) -> int:
    """Structural Hamming Distance (SHD).

    Counts the number of edge insertions, deletions, and flips needed
    to convert ``estimated`` into ``true``.  For DAGs (directed), each
    position ``(i, j)`` where the two matrices disagree contributes 1.

    Parameters
    ----------
    estimated, true : array-like
        Binary adjacency matrices (0/1).  May be DataFrames with matching
        variable names.

    Returns
    -------
    int
        SHD value (lower is better; 0 = perfect match).
    """
    if isinstance(estimated, pd.DataFrame) and isinstance(true, pd.DataFrame):
        est, gt = _align_dataframes(estimated, true)
    else:
        est, gt = _to_array(estimated), _to_array(true)
    return int(np.sum(est != gt))


def edge_precision_recall_f1(
    estimated: Union[np.ndarray, pd.DataFrame],
    true: Union[np.ndarray, pd.DataFrame],
) -> Dict[str, float]:
    """Compute edge-level precision, recall, and F1.

    An edge is ``(i, j)`` where ``adj[i, j] == 1``.

    Parameters
    ----------
    estimated, true : array-like
        Binary adjacency matrices.

    Returns
    -------
    dict
        ``{"precision": float, "recall": float, "f1": float}``
    """
    if isinstance(estimated, pd.DataFrame) and isinstance(true, pd.DataFrame):
        est, gt = _align_dataframes(estimated, true)
    else:
        est, gt = _to_array(estimated), _to_array(true)

    tp = int(np.sum((est == 1) & (gt == 1)))
    fp = int(np.sum((est == 1) & (gt == 0)))
    fn = int(np.sum((est == 0) & (gt == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}


def compare_graphs(
    estimated: Union[np.ndarray, pd.DataFrame],
    true: Union[np.ndarray, pd.DataFrame],
    label: str = "",
) -> Dict[str, float]:
    """Compute all metrics comparing an estimated graph to ground truth.

    Parameters
    ----------
    estimated, true : array-like
        Binary adjacency matrices.
    label : str
        Optional label to prefix keys.

    Returns
    -------
    dict
        ``{prefix+"precision": ..., prefix+"recall": ...,
          prefix+"f1": ..., prefix+"shd": ...}``
    """
    prf = edge_precision_recall_f1(estimated, true)
    shd = structural_hamming_distance(estimated, true)
    prefix = f"{label}_" if label else ""
    return {
        f"{prefix}precision": prf["precision"],
        f"{prefix}recall": prf["recall"],
        f"{prefix}f1": prf["f1"],
        f"{prefix}shd": shd,
    }


# ---------------------------------------------------------------------------
# Classical baseline runners (optional dependencies)
# ---------------------------------------------------------------------------

def run_ges_baseline(
    data: np.ndarray,
    variable_names: Optional[list] = None,
) -> pd.DataFrame:
    """Run GES (Greedy Equivalence Search) as a classical baseline.

    Requires ``causal-learn`` package (``pip install causal-learn``).

    Parameters
    ----------
    data : ndarray
        ``(n_samples, n_variables)`` observational data.
    variable_names : list of str, optional
        Variable names for the returned DataFrame.

    Returns
    -------
    pd.DataFrame
        Estimated binary adjacency matrix.
    """
    try:
        from causallearn.search.ScoreBased.GES import ges as ges_search
    except ImportError:
        raise ImportError(
            "causal-learn is required for GES baseline. "
            "Install with: pip install causal-learn"
        )

    result = ges_search(data, score_func="local_score_BIC")
    adj = result["G"].graph  # adjacency matrix
    # causal-learn uses: -1 = tail, 1 = arrowhead, 0 = no edge
    # Convert to binary: edge present if any non-zero entry
    binary_adj = (np.abs(adj) > 0).astype(int)
    if variable_names:
        return pd.DataFrame(binary_adj, index=variable_names, columns=variable_names)
    return pd.DataFrame(binary_adj)


def run_pc_baseline(
    data: np.ndarray,
    alpha: float = 0.05,
    variable_names: Optional[list] = None,
) -> pd.DataFrame:
    """Run the PC algorithm as a classical baseline.

    Requires ``causal-learn`` package.

    Parameters
    ----------
    data : ndarray
        ``(n_samples, n_variables)``.
    alpha : float
        Significance level for conditional independence tests.
    variable_names : list of str, optional
        Variable names for the DataFrame.

    Returns
    -------
    pd.DataFrame
        Estimated binary adjacency matrix.
    """
    try:
        from causallearn.search.ConstraintBased.PC import pc as pc_search
    except ImportError:
        raise ImportError(
            "causal-learn is required for PC baseline. "
            "Install with: pip install causal-learn"
        )

    result = pc_search(data, alpha=alpha, indep_test="fisherz")
    adj = result.G.graph
    binary_adj = (np.abs(adj) > 0).astype(int)
    if variable_names:
        return pd.DataFrame(binary_adj, index=variable_names, columns=variable_names)
    return pd.DataFrame(binary_adj)
