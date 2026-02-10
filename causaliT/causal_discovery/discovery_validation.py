"""
Causal discovery validation pipeline.

Generates i.i.d. data from the SCM trajectory, applies discovery methods
(attention-based + optional classical baselines), compares against ground
truth, and produces summary tables.

Inspired by ``causal_discovery_iid.ipynb`` and ``causal_discovery_time.ipynb``
from causal-chamber-paper.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scm_ds.scm import SCMDataset

from .ground_truth import (
    build_ground_truth_dag,
    get_observable_variables,
    PROCESS_ORDER,
    _DEFAULT_OBSERVABLE,
)
from .attention_discovery import (
    extract_attention_weights,
    aggregate_attention,
    attention_to_adjacency,
)
from .metrics import compute_all_metrics, compare_graphs, run_classical_discovery


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------

def generate_iid_data(
    datasets: Dict[str, SCMDataset],
    n_samples: int = 2000,
    seed: int = 42,
    process_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Generate i.i.d. observational samples from all processes.

    Concatenates the observable columns of each process into a single
    DataFrame.

    Parameters
    ----------
    datasets : dict
        ``{"laser": ds_laser, "plasma": ds_plasma, ...}``
    n_samples : int
        Number of samples per process.
    seed : int
        Random seed.
    process_order : list[str], optional
        Defaults to ``PROCESS_ORDER``.

    Returns
    -------
    DataFrame
        Columns are the observable variables of each process.
    """
    order = process_order or PROCESS_ORDER
    frames = {}
    for proc in order:
        ds = datasets[proc]
        df = ds.sample(n_samples, seed=seed)
        # Keep only observable columns
        obs_cols = ds.input_labels + ds.target_labels
        for col in obs_cols:
            col_name = col
            # Disambiguate shared names (e.g. Duration)
            if col_name in frames:
                col_name = f"{col}_{proc}"
            frames[col_name] = df[col].values

    return pd.DataFrame(frames)


# ---------------------------------------------------------------------------
# Validation pipeline
# ---------------------------------------------------------------------------

def validate_attention_discovery(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    input_vars_map: Dict[str, int],
    target_vars_map: Dict[str, int],
    adj_true: np.ndarray | pd.DataFrame,
    thresholds: Optional[List[float]] = None,
    aggregation: str = "mean",
    device: str = "cpu",
) -> pd.DataFrame:
    """Evaluate attention-based discovery at multiple thresholds.

    Parameters
    ----------
    model : nn.Module
        Trained CausaliT/ProT.
    input_tensor, target_tensor : Tensor
        Batch of trajectory data.
    input_vars_map, target_vars_map : dict
        Variable name -> ID mappings.
    adj_true : ndarray or DataFrame
        Ground truth adjacency.
    thresholds : list[float], optional
        Thresholds to sweep; defaults to ``[0.01, 0.05, 0.1, 0.2, 0.3, 0.5]``.
    aggregation : str
        ``"mean"`` or ``"max"`` for attention aggregation.
    device : str
        Inference device.

    Returns
    -------
    DataFrame
        One row per (attention_type, threshold) with SHD, precision, recall, F1.
    """
    if thresholds is None:
        thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]

    atts = extract_attention_weights(model, input_tensor, target_tensor, device)

    results = []
    for att_type in ["enc_self_att", "dec_self_att", "dec_cross_att"]:
        raw = atts[att_type]
        agg = aggregate_attention(raw, method=aggregation)

        for thr in thresholds:
            adj_est = attention_to_adjacency(agg, threshold=thr, method="absolute")

            # Pad/align if needed (est may have different shape than truth)
            A_true = adj_true.values if isinstance(adj_true, pd.DataFrame) else adj_true
            n_true = A_true.shape[0]
            n_est = adj_est.shape[0]

            if n_est < n_true:
                # Pad estimated with zeros
                padded = np.zeros((n_true, n_true), dtype=int)
                padded[:n_est, :adj_est.shape[1]] = adj_est
                adj_est = padded
            elif n_est > n_true:
                # Crop estimated
                adj_est = adj_est[:n_true, :n_true]

            m = compute_all_metrics(A_true, adj_est)
            m["attention_type"] = att_type
            m["threshold"] = thr
            m["aggregation"] = aggregation
            results.append(m)

    return pd.DataFrame(results)


def validate_classical_baselines(
    data: np.ndarray,
    adj_true: np.ndarray,
    methods: Optional[List[str]] = None,
    variable_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run classical causal discovery algorithms and evaluate.

    Parameters
    ----------
    data : ndarray
        ``(n_samples, n_vars)`` i.i.d. data.
    adj_true : ndarray
        Ground truth adjacency.
    methods : list[str]
        Algorithms to try (default: ``["GES", "PC"]``).
    variable_names : list[str], optional
        Names for each variable.

    Returns
    -------
    DataFrame
        One row per method with SHD, precision, recall, F1.
    """
    if methods is None:
        methods = ["GES", "PC"]

    results = []
    for method in methods:
        try:
            adj_est = run_classical_discovery(data, method=method, variable_names=variable_names)

            # Align sizes
            n_true = adj_true.shape[0]
            n_est = adj_est.shape[0]
            if n_est < n_true:
                padded = np.zeros((n_true, n_true), dtype=int)
                padded[:n_est, :n_est] = adj_est
                adj_est = padded
            elif n_est > n_true:
                adj_est = adj_est[:n_true, :n_true]

            m = compute_all_metrics(adj_true, adj_est)
            m["method"] = method
            results.append(m)
        except ImportError:
            results.append({
                "method": method,
                "shd": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "n_true_edges": np.nan,
                "n_est_edges": np.nan,
                "error": "causal-learn not installed",
            })
        except Exception as e:
            results.append({
                "method": method,
                "shd": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "n_true_edges": np.nan,
                "n_est_edges": np.nan,
                "error": str(e),
            })

    return pd.DataFrame(results)


def run_full_validation(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    input_vars_map: Dict[str, int],
    target_vars_map: Dict[str, int],
    datasets: Optional[Dict[str, SCMDataset]] = None,
    n_samples: int = 2000,
    seed: int = 42,
    thresholds: Optional[List[float]] = None,
    run_classical: bool = True,
    device: str = "cpu",
) -> Dict[str, pd.DataFrame]:
    """Run full discovery validation pipeline.

    Parameters
    ----------
    model : nn.Module
        Trained CausaliT/ProT.
    input_tensor, target_tensor : Tensor
        Trajectory batch for attention extraction.
    input_vars_map, target_vars_map : dict
        Variable mappings.
    datasets : dict, optional
        SCMDataset dict for classical baselines.
    n_samples : int
        Samples for classical methods.
    seed : int
        Random seed.
    thresholds : list[float], optional
        Threshold sweep.
    run_classical : bool
        Whether to run GES/PC baselines.
    device : str
        Inference device.

    Returns
    -------
    dict
        ``{"attention_results": DataFrame, "classical_results": DataFrame,
           "ground_truth": ndarray}``
    """
    adj_true = build_ground_truth_dag(as_dataframe=False)

    att_results = validate_attention_discovery(
        model=model,
        input_tensor=input_tensor,
        target_tensor=target_tensor,
        input_vars_map=input_vars_map,
        target_vars_map=target_vars_map,
        adj_true=adj_true,
        thresholds=thresholds,
        device=device,
    )

    classical_results = None
    if run_classical and datasets is not None:
        iid_data = generate_iid_data(datasets, n_samples=n_samples, seed=seed)
        classical_results = validate_classical_baselines(
            data=iid_data.values,
            adj_true=adj_true,
            variable_names=list(iid_data.columns),
        )

    return {
        "attention_results": att_results,
        "classical_results": classical_results,
        "ground_truth": adj_true,
    }
