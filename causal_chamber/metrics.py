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


# ---------------------------------------------------------------------------
# UT-IGSP with grid search (reproduces Causal Chamber paper methodology)
# ---------------------------------------------------------------------------

def _check_causaldag():
    """Check if causaldag is available."""
    try:
        from causaldag import unknown_target_igsp, hsic_test
        return True
    except ImportError:
        return False


CAUSALDAG_AVAILABLE = _check_causaldag()


def run_ut_igsp(
    observational_data: np.ndarray,
    interventional_data_list: list,
    alpha_ci: float = 0.01,
    alpha_inv: float = 0.01,
    test: str = 'hsic',
) -> Optional[np.ndarray]:
    """
    Run UT-IGSP (Unknown Target Interventional GSP) from causaldag.

    Reproduces the Causal Chamber paper methodology (Gamella et al. 2025,
    Appendix IV / causal_discovery_iid.ipynb).

    Parameters
    ----------
    observational_data : np.ndarray
        Observational (reference) samples, shape (n_obs, p).
    interventional_data_list : list of np.ndarray
        List of interventional samples, each shape (n_int, p).
    alpha_ci : float
        Significance level for conditional independence tests.
    alpha_inv : float
        Significance level for invariance tests.
    test : str
        'hsic' for kernel-based (non-parametric) or 'gauss' for Gaussian.

    Returns
    -------
    np.ndarray or None
        Estimated adjacency matrix (A[i,j]=1 means i->j) or None if unavailable.
    """
    if not CAUSALDAG_AVAILABLE:
        warnings.warn("causaldag not installed. UT-IGSP unavailable.")
        return None

    from causaldag import (
        unknown_target_igsp,
        hsic_test, hsic_invariance_test,
        gauss_invariance_suffstat, gauss_invariance_test,
        partial_correlation_suffstat, partial_correlation_test,
        MemoizedCI_Tester, MemoizedInvarianceTester,
    )

    # Combine into data list: [observational, interventional_1, ...]
    data = [observational_data] + interventional_data_list
    p = observational_data.shape[1]
    nodes = set(range(p))

    if test == 'gauss':
        ci_suffstat = partial_correlation_suffstat(observational_data)
        invariance_suffstat = gauss_invariance_suffstat(
            observational_data, interventional_data_list,
        )
        ci_tester = MemoizedCI_Tester(
            partial_correlation_test, ci_suffstat, alpha=alpha_ci,
        )
        invariance_tester = MemoizedInvarianceTester(
            gauss_invariance_test, invariance_suffstat, alpha=alpha_inv,
        )
    elif test == 'hsic':
        ci_tester = MemoizedCI_Tester(
            hsic_test, observational_data, alpha=alpha_ci,
        )
        suffstat = {i: sample for i, sample in enumerate(interventional_data_list)}
        suffstat['obs_samples'] = observational_data
        invariance_tester = MemoizedInvarianceTester(
            hsic_invariance_test, suffstat, alpha=alpha_inv,
        )
    else:
        raise ValueError(f'Invalid test type: {test}')

    # Run UT-IGSP
    setting_list = [dict(known_interventions=[])] * len(interventional_data_list)
    estimated_dag, est_targets_list = unknown_target_igsp(
        setting_list, nodes, ci_tester, invariance_tester,
    )
    return estimated_dag.to_amat()[0]


def run_ut_igsp_grid_search(
    observational_data: np.ndarray,
    interventional_data_list: list,
    truth: np.ndarray,
    n_alphas: int = 10,
    alpha_range: tuple = (1e-4, 1e-2),
    test: str = 'hsic',
) -> Dict:
    """
    Grid search over (alpha_ci, alpha_inv) for UT-IGSP, selecting the
    hyperparameters that minimize L2 distance to (1,1) in precision-recall
    space.

    Reproduces the Causal Chamber paper methodology exactly
    (Gamella et al. 2025, causal_discovery_iid.ipynb).

    Parameters
    ----------
    observational_data : np.ndarray
        Observational (reference) samples, shape (n_obs, p).
    interventional_data_list : list of np.ndarray
        List of interventional samples, each shape (n_int, p).
    truth : np.ndarray
        Ground truth adjacency matrix.
    n_alphas : int
        Grid resolution (n_alphas x n_alphas).
    alpha_range : tuple
        (min_alpha, max_alpha) for both CI and invariance tests.
    test : str
        'hsic' or 'gauss'.

    Returns
    -------
    dict with keys:
        'best_dag': np.ndarray — best estimated adjacency matrix
        'best_alpha_ci': float
        'best_alpha_inv': float
        'best_metrics': dict with precision, recall, f1, shd
        'all_dags': np.ndarray of shape (n_alphas, n_alphas, p, p)
        'all_metrics': np.ndarray of shape (n_alphas, n_alphas, 2)
        'l2_distances': np.ndarray of shape (n_alphas, n_alphas)
        'alphas': np.ndarray
        'betas': np.ndarray
    """
    if not CAUSALDAG_AVAILABLE:
        warnings.warn("causaldag not installed. UT-IGSP grid search unavailable.")
        return None

    alphas = np.linspace(alpha_range[0], alpha_range[1], n_alphas)
    betas = np.linspace(alpha_range[0], alpha_range[1], n_alphas)
    p = observational_data.shape[1]

    all_dags = np.zeros((n_alphas, n_alphas, p, p))
    all_metrics = np.zeros((n_alphas, n_alphas, 2))  # precision, recall

    for i, alpha_ci in enumerate(alphas):
        for j, alpha_inv in enumerate(betas):
            try:
                dag = run_ut_igsp(
                    observational_data, interventional_data_list,
                    alpha_ci=alpha_ci, alpha_inv=alpha_inv, test=test,
                )
                if dag is not None:
                    all_dags[i, j] = dag
                    all_metrics[i, j, 0] = edge_precision(dag, truth)
                    all_metrics[i, j, 1] = edge_recall(dag, truth)
            except Exception as e:
                warnings.warn(f"UT-IGSP failed at alpha_ci={alpha_ci:.4f}, "
                              f"alpha_inv={alpha_inv:.4f}: {e}")

    # Select best by L2 distance to (1,1) in P/R space
    l2_dist = np.sqrt(((1 - all_metrics) ** 2).sum(axis=2))
    best_idx = np.unravel_index(np.argmin(l2_dist), l2_dist.shape)
    i_best, j_best = best_idx

    best_dag = all_dags[i_best, j_best]
    best_prec = all_metrics[i_best, j_best, 0]
    best_rec = all_metrics[i_best, j_best, 1]
    best_f1 = 2 * best_prec * best_rec / (best_prec + best_rec) if (best_prec + best_rec) > 0 else 0.0

    return {
        'best_dag': best_dag,
        'best_alpha_ci': float(alphas[i_best]),
        'best_alpha_inv': float(betas[j_best]),
        'best_metrics': {
            'precision': float(best_prec),
            'recall': float(best_rec),
            'f1': float(best_f1),
            'shd': structural_hamming_distance(best_dag, truth),
        },
        'all_dags': all_dags,
        'all_metrics': all_metrics,
        'l2_distances': l2_dist,
        'alphas': alphas,
        'betas': betas,
    }
