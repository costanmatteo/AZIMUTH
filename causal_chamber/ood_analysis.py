"""
Out-of-Distribution (OOD) Analysis.

Generates trajectory data with shifted input distributions and measures
degradation in: uncertainty predictor accuracy, CausaliT F prediction,
and stability of attention-based causal structure.

Inspired by ood_sensors.ipynb from the Causal Chamber paper
(Gamella et al., Nature Machine Intelligence 2025).
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error

import sys

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from causal_chamber.ground_truth import (
    PROCESS_ORDER, PROCESS_DATASETS, PROCESS_OBSERVABLE_VARS,
)


# ---------------------------------------------------------------------------
# OOD shift definitions
# ---------------------------------------------------------------------------

# For each process, define ID ranges and OOD shifts.
# Based on the noise sampler ranges in scm_ds/datasets.py.
OOD_SHIFTS = {
    'laser': {
        'variable': 'AmbientTemp',
        'id_range': (15.0, 35.0),     # original training range
        'ood_range': (35.0, 45.0),    # shifted upward
        'description': 'AmbientTemp shifted from [15,35] to [35,45]',
    },
    'plasma': {
        'variable': 'RF_Power',
        'id_range': (100.0, 400.0),
        'ood_range': (400.0, 600.0),
        'description': 'RF_Power shifted from [100,400] to [400,600]',
    },
    'galvanic': {
        'variable': 'CurrentDensity',
        'id_range': (1.0, 5.0),
        'ood_range': (5.0, 8.0),
        'description': 'CurrentDensity shifted from [1,5] to [5,8]',
    },
    'microetch': {
        'variable': 'Temperature',
        'id_range': (293.0, 323.0),
        'ood_range': (323.0, 353.0),
        'description': 'Temperature shifted from [293,323] to [323,353]',
    },
}


def _make_shifted_noise_model(
    process_name: str,
    variable: str,
    new_range: Tuple[float, float],
):
    """
    Create a modified noise model with a shifted range for one variable.

    Returns a new NoiseModel with the specified variable's sampler replaced.
    """
    from scm_ds.scm import NoiseModel

    ds = PROCESS_DATASETS[process_name]
    original = ds.noise_model

    # Copy singles
    new_singles = dict(original.singles)
    low, high = new_range
    new_singles[variable] = lambda rng, n, lo=low, hi=high: rng.uniform(lo, hi, size=n)

    return NoiseModel(singles=new_singles, groups=original.groups)


def sample_ood_data(
    process_name: str,
    n: int = 1000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sample in-distribution (ID) and out-of-distribution (OOD) data.

    Parameters
    ----------
    process_name : str
        Process name.
    n : int
        Number of samples per condition.
    seed : int
        Random seed.

    Returns
    -------
    id_df, ood_df : pd.DataFrame
        In-distribution and OOD samples.
    """
    from scm_ds.scm import SCM

    ds = PROCESS_DATASETS[process_name]
    shift = OOD_SHIFTS[process_name]

    # ID data (standard sampling)
    id_df = ds.sample(n, seed=seed)

    # OOD data (shifted distribution)
    ood_noise_model = _make_shifted_noise_model(
        process_name, shift['variable'], shift['ood_range'],
    )
    ood_scm = SCM(
        list(ds.scm.specs.values()),
        noise_model=ood_noise_model,
    )
    rng = np.random.default_rng(seed + 1)
    eps_draws = ood_noise_model.sample_all(rng, n)
    ctx = {}
    ood_scm.forward(ctx, eps_draws)
    ood_df = pd.DataFrame({k: np.asarray(v).reshape(n) for k, v in ctx.items()})

    return id_df, ood_df


def compute_prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute R², MAE for predictions."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Handle edge cases
    if len(y_true) == 0 or np.std(y_true) < 1e-10:
        return {'r2': float('nan'), 'mae': float('nan')}

    return {
        'r2': float(r2_score(y_true, y_pred)),
        'mae': float(mean_absolute_error(y_true, y_pred)),
    }


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_var: np.ndarray,
    coverage_levels: List[float] = None,
) -> Dict[str, float]:
    """
    Compute calibration metrics for uncertainty estimates.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred_mean : np.ndarray
        Predicted means.
    y_pred_var : np.ndarray
        Predicted variances.
    coverage_levels : list of float
        Nominal coverage levels (default [0.5, 0.9, 0.95]).

    Returns
    -------
    dict with calibration ratio and coverage metrics.
    """
    if coverage_levels is None:
        coverage_levels = [0.5, 0.9, 0.95]

    y_true = np.asarray(y_true).ravel()
    y_pred_mean = np.asarray(y_pred_mean).ravel()
    y_pred_var = np.asarray(y_pred_var).ravel()
    y_pred_std = np.sqrt(np.maximum(y_pred_var, 1e-10))

    # Standardized residuals
    z = (y_true - y_pred_mean) / y_pred_std

    # Calibration ratio: ratio of actual std to predicted std
    actual_std = np.std(y_true - y_pred_mean)
    pred_std_mean = np.mean(y_pred_std)
    calibration_ratio = actual_std / pred_std_mean if pred_std_mean > 0 else float('nan')

    # Coverage at different levels
    from scipy.stats import norm
    coverages = {}
    for level in coverage_levels:
        z_crit = norm.ppf((1 + level) / 2)
        in_interval = np.abs(z) <= z_crit
        coverages[f'coverage_{level:.0%}'] = float(in_interval.mean())

    return {
        'calibration_ratio': float(calibration_ratio),
        **coverages,
    }


def analyze_attention_stability(
    id_attention: Optional[np.ndarray],
    ood_attention: Optional[np.ndarray],
) -> Dict[str, float]:
    """
    Compare attention matrices between ID and OOD conditions.

    Measures how stable the learned causal structure is under distribution shift.

    Returns
    -------
    dict with 'frobenius_diff', 'max_diff', 'correlation'.
    """
    if id_attention is None or ood_attention is None:
        return {'frobenius_diff': float('nan'), 'max_diff': float('nan'), 'correlation': float('nan')}

    diff = id_attention - ood_attention
    frob = float(np.linalg.norm(diff, 'fro'))
    max_diff = float(np.abs(diff).max())

    # Pearson correlation between flattened matrices
    id_flat = id_attention.ravel()
    ood_flat = ood_attention.ravel()
    if np.std(id_flat) > 1e-10 and np.std(ood_flat) > 1e-10:
        corr = float(np.corrcoef(id_flat, ood_flat)[0, 1])
    else:
        corr = float('nan')

    return {
        'frobenius_diff': frob,
        'max_diff': max_diff,
        'correlation': corr,
    }


def run_ood_analysis(
    n: int = 1000,
    seed: int = 42,
) -> Dict:
    """
    Run OOD analysis for all processes.

    For each process:
    1. Sample ID and OOD data
    2. Compare output distributions
    3. Report shift statistics

    CausaliT attention stability analysis requires a checkpoint and is
    handled separately via run_analysis.py.

    Returns
    -------
    dict with keys:
        'per_process': dict of process -> OOD analysis results
        'summary_table': pd.DataFrame
    """
    per_process = {}
    summary_rows = []

    for proc_name in PROCESS_ORDER:
        shift = OOD_SHIFTS[proc_name]
        info = PROCESS_OBSERVABLE_VARS[proc_name]
        out_var = info['outputs'][0]

        id_df, ood_df = sample_ood_data(proc_name, n=n, seed=seed)

        # Compare output distributions
        id_out = id_df[out_var].values
        ood_out = ood_df[out_var].values

        from scipy.stats import ks_2samp
        ks_stat, ks_pval = ks_2samp(id_out, ood_out)

        result = {
            'process': proc_name,
            'shift': shift,
            'id_data': id_df,
            'ood_data': ood_df,
            'id_output_stats': {
                'mean': float(id_out.mean()),
                'std': float(id_out.std()),
                'min': float(id_out.min()),
                'max': float(id_out.max()),
            },
            'ood_output_stats': {
                'mean': float(ood_out.mean()),
                'std': float(ood_out.std()),
                'min': float(ood_out.min()),
                'max': float(ood_out.max()),
            },
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pval),
        }
        per_process[proc_name] = result

        summary_rows.append({
            'process': proc_name,
            'shifted_variable': shift['variable'],
            'id_range': f"[{shift['id_range'][0]}, {shift['id_range'][1]}]",
            'ood_range': f"[{shift['ood_range'][0]}, {shift['ood_range'][1]}]",
            f'{out_var}_id_mean': result['id_output_stats']['mean'],
            f'{out_var}_ood_mean': result['ood_output_stats']['mean'],
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pval),
        })

    summary_table = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()

    return {
        'per_process': per_process,
        'summary_table': summary_table,
    }
