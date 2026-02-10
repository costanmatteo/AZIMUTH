"""
Out-of-Distribution (OOD) analysis for the AZIMUTH pipeline.

Tests robustness under distributional shift by generating trajectories with
shifted input distributions and measuring degradation in:
- Uncertainty predictor accuracy (R^2, calibration ratio, coverage)
- CausaliT F-prediction accuracy
- Attention structure stability (does the discovered DAG change?)

Inspired by ``ood_sensors.ipynb``, ``ood_images.ipynb``, and
``ood_impulses.ipynb`` from causal-chamber-paper.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scm_ds.scm import SCMDataset, SCM, NoiseModel, NodeSpec

from .attention_discovery import (
    extract_attention_weights,
    aggregate_attention,
    attention_to_adjacency,
)
from .metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Shifted data generation
# ---------------------------------------------------------------------------

def create_shifted_dataset(
    dataset: SCMDataset,
    shift_specs: Dict[str, Dict[str, float]],
) -> SCMDataset:
    """Create a copy of an SCMDataset with shifted input distributions.

    Parameters
    ----------
    dataset : SCMDataset
        Original dataset.
    shift_specs : dict
        ``{variable_name: {"low": ..., "high": ...}}``
        specifying new uniform distribution ranges for the named variables.

    Returns
    -------
    SCMDataset
        New dataset with modified noise samplers for the specified variables.
    """
    # Deep copy noise model internals
    original_singles = {}
    if isinstance(dataset.scm.noise, NoiseModel):
        original_singles = dict(dataset.scm.noise.singles)
    elif isinstance(dataset.scm.noise, dict):
        original_singles = dict(dataset.scm.noise)

    new_singles = dict(original_singles)

    for var_name, spec in shift_specs.items():
        lo = spec["low"]
        hi = spec["high"]
        new_singles[var_name] = lambda rng, n, lo=lo, hi=hi: rng.uniform(
            low=lo, high=hi, size=n
        )

    # Reconstruct SCM with new noise
    new_noise_model = NoiseModel(singles=new_singles)
    new_scm = SCM(
        list(dataset.scm.specs.values()),
        noise_model=new_noise_model,
    )

    # Create a lightweight wrapper
    shifted = _ShiftedDataset(new_scm, dataset.input_labels, dataset.target_labels)
    return shifted


class _ShiftedDataset:
    """Minimal dataset wrapper for shifted SCMs."""

    def __init__(self, scm: SCM, input_labels: list, target_labels: list):
        self.scm = scm
        self.input_labels = input_labels
        self.target_labels = target_labels

    def sample(self, n: int, seed: int = 42) -> pd.DataFrame:
        return self.scm.sample(n, seed=seed)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_var: Optional[np.ndarray] = None,
    confidence_level: float = 0.95,
) -> Dict[str, float]:
    """Compute prediction quality and uncertainty metrics.

    Parameters
    ----------
    y_true : ndarray
        Ground truth values, shape ``(n,)`` or ``(n, d)``.
    y_pred : ndarray
        Predicted mean values, same shape.
    y_var : ndarray, optional
        Predicted variance, same shape.  If provided, calibration and
        coverage metrics are computed.
    confidence_level : float
        Confidence level for prediction interval coverage (default 95%).

    Returns
    -------
    dict
        ``{"r2": ..., "rmse": ..., "mae": ..., [calibration metrics]}``.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    result = {"r2": r2, "rmse": rmse, "mae": mae}

    if y_var is not None:
        y_var = np.asarray(y_var).flatten()
        y_std = np.sqrt(np.maximum(y_var, 0))

        # Calibration ratio: mean(variance) / MSE
        mse = np.mean((y_true - y_pred) ** 2)
        mean_var = np.mean(y_var)
        calibration_ratio = mean_var / mse if mse > 0 else np.nan

        # Coverage: fraction of true values within CI
        from scipy.stats import norm
        z = norm.ppf(0.5 + confidence_level / 2)
        lower = y_pred - z * y_std
        upper = y_pred + z * y_std
        coverage = np.mean((y_true >= lower) & (y_true <= upper))

        result.update({
            "calibration_ratio": calibration_ratio,
            "coverage": coverage,
            "mean_variance": mean_var,
        })

    return result


# ---------------------------------------------------------------------------
# Attention stability
# ---------------------------------------------------------------------------

def compare_attention_stability(
    model: nn.Module,
    in_dist_input: torch.Tensor,
    in_dist_target: torch.Tensor,
    ood_input: torch.Tensor,
    ood_target: torch.Tensor,
    threshold: float = 0.1,
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    """Compare attention-derived DAGs between in-distribution and OOD data.

    Parameters
    ----------
    model : nn.Module
        Trained CausaliT/ProT.
    in_dist_input, in_dist_target : Tensor
        In-distribution trajectory data.
    ood_input, ood_target : Tensor
        OOD trajectory data.
    threshold : float
        Threshold for attention -> adjacency conversion.
    device : str
        Inference device.

    Returns
    -------
    dict
        ``{att_type: {"shd": ..., "jaccard": ..., ...}}``
        for each attention block.
    """
    atts_id = extract_attention_weights(model, in_dist_input, in_dist_target, device)
    atts_ood = extract_attention_weights(model, ood_input, ood_target, device)

    results = {}
    for att_type in ["enc_self_att", "dec_self_att", "dec_cross_att"]:
        agg_id = aggregate_attention(atts_id[att_type])
        agg_ood = aggregate_attention(atts_ood[att_type])

        adj_id = attention_to_adjacency(agg_id, threshold=threshold)
        adj_ood = attention_to_adjacency(agg_ood, threshold=threshold)

        # Align shapes
        n = max(adj_id.shape[0], adj_ood.shape[0])
        A = np.zeros((n, n), dtype=int)
        B = np.zeros((n, n), dtype=int)
        A[:adj_id.shape[0], :adj_id.shape[1]] = adj_id
        B[:adj_ood.shape[0], :adj_ood.shape[1]] = adj_ood

        metrics = compute_all_metrics(A, B)

        # Jaccard similarity of edge sets
        edge_union = np.sum((A | B) > 0)
        edge_inter = np.sum((A & B) > 0)
        jaccard = edge_inter / edge_union if edge_union > 0 else 1.0

        # Frobenius distance of raw attention maps
        min_l = min(agg_id.shape[0], agg_ood.shape[0])
        min_s = min(agg_id.shape[1], agg_ood.shape[1])
        frob = np.linalg.norm(agg_id[:min_l, :min_s] - agg_ood[:min_l, :min_s])

        results[att_type] = {
            **metrics,
            "jaccard_similarity": jaccard,
            "frobenius_distance": frob,
        }

    return results


# ---------------------------------------------------------------------------
# Full OOD analysis
# ---------------------------------------------------------------------------

# Default shift configurations
DEFAULT_OOD_SHIFTS = {
    "laser_temp_shift": {
        "process": "laser",
        "description": "AmbientTemp shifted from [15,35] to [35,50]",
        "shifts": {"AmbientTemp": {"low": 35.0, "high": 50.0}},
    },
    "laser_power_shift": {
        "process": "laser",
        "description": "PowerTarget shifted from [0.1,1.0] to [0.8,1.5]",
        "shifts": {"PowerTarget": {"low": 0.8, "high": 1.5}},
    },
    "plasma_power_shift": {
        "process": "plasma",
        "description": "RF_Power shifted from [100,400] to [400,600]",
        "shifts": {"RF_Power": {"low": 400.0, "high": 600.0}},
    },
    "microetch_temp_shift": {
        "process": "microetch",
        "description": "Temperature shifted from [293,323] to [323,353]",
        "shifts": {"Temperature": {"low": 323.0, "high": 353.0}},
    },
}


def run_ood_analysis(
    datasets: Dict[str, SCMDataset],
    shift_configs: Optional[Dict[str, dict]] = None,
    n_samples: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run OOD analysis with distributional shifts.

    Generates in-distribution and OOD data from the SCM, compares
    statistics of output variables.

    Parameters
    ----------
    datasets : dict
        ``{"laser": ds_laser, ...}``
    shift_configs : dict, optional
        Shift specifications.  Defaults to ``DEFAULT_OOD_SHIFTS``.
    n_samples, seed : int
        Sampling parameters.

    Returns
    -------
    DataFrame
        Summary of distributional shifts and output changes.
    """
    configs = shift_configs or DEFAULT_OOD_SHIFTS
    results = []

    for shift_name, cfg in configs.items():
        proc = cfg["process"]
        ds = datasets[proc]

        # In-distribution
        df_id = ds.sample(n_samples, seed=seed)

        # OOD
        ds_ood = create_shifted_dataset(ds, cfg["shifts"])
        df_ood = ds_ood.sample(n_samples, seed=seed)

        # Compare output variable
        out_var = ds.target_labels[0]
        if out_var in df_id.columns and out_var in df_ood.columns:
            id_vals = df_id[out_var].values
            ood_vals = df_ood[out_var].values

            row = {
                "shift_name": shift_name,
                "process": proc,
                "description": cfg["description"],
                "output_var": out_var,
                "id_mean": np.mean(id_vals),
                "id_std": np.std(id_vals),
                "ood_mean": np.mean(ood_vals),
                "ood_std": np.std(ood_vals),
                "delta_mean": np.mean(ood_vals) - np.mean(id_vals),
                "delta_std": np.std(ood_vals) - np.std(id_vals),
            }

            try:
                from scipy.stats import ks_2samp
                ks_stat, ks_pval = ks_2samp(id_vals, ood_vals)
                row["ks_statistic"] = ks_stat
                row["ks_pvalue"] = ks_pval
            except ImportError:
                row["ks_statistic"] = np.nan
                row["ks_pvalue"] = np.nan

            results.append(row)

    return pd.DataFrame(results)
