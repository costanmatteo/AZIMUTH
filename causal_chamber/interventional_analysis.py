"""
Interventional Validation Analysis.

Uses Pearl's do-operator (SCM.do()) to generate data under interventions,
validates that CausaliT and the reliability formula capture interventional
effects correctly. Inspired by the causal_validation.ipynb from the
Causal Chamber paper (Gamella et al., Nature Machine Intelligence 2025).
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from scipy import stats

import sys

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from causal_chamber.ground_truth import (
    PROCESS_ORDER, PROCESS_DATASETS, PROCESS_OBSERVABLE_VARS,
)


# ---------------------------------------------------------------------------
# Intervention definitions
# ---------------------------------------------------------------------------

DEFAULT_INTERVENTIONS = {
    'laser': [
        {'PowerTarget': 0.3},
        {'PowerTarget': 0.7},
        {'AmbientTemp': 20.0},
        {'AmbientTemp': 30.0},
    ],
    'plasma': [
        {'RF_Power': 150.0},
        {'RF_Power': 350.0},
        {'Duration': 20.0},
        {'Duration': 50.0},
    ],
    'galvanic': [
        {'CurrentDensity': 2.0},
        {'CurrentDensity': 4.0},
        {'Duration': 1200.0},
        {'Duration': 3000.0},
    ],
    'microetch': [
        {'Temperature': 300.0},
        {'Temperature': 315.0},
        {'Concentration': 1.0},
        {'Concentration': 2.5},
    ],
}


def generate_observational_data(
    process_name: str,
    n: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample observational data from a single process SCM."""
    ds = PROCESS_DATASETS[process_name]
    return ds.sample(n, seed=seed)


def generate_interventional_data(
    process_name: str,
    interventions: Dict[str, float],
    n: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample data from a process SCM under do-interventions.

    Parameters
    ----------
    process_name : str
        Process name (e.g., 'laser').
    interventions : dict
        Mapping variable_name -> constant value for do-operator.
    n : int
        Number of samples.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Interventional samples.
    """
    ds = PROCESS_DATASETS[process_name]
    scm_intervened = ds.scm.do(interventions)
    return scm_intervened.sample(n, seed=seed)


def compare_distributions(
    obs_data: np.ndarray,
    int_data: np.ndarray,
    test: str = 'ks',
) -> Dict[str, float]:
    """
    Compare observational and interventional distributions.

    Parameters
    ----------
    obs_data : np.ndarray
        Observational samples (1D).
    int_data : np.ndarray
        Interventional samples (1D).
    test : str
        'ks' for Kolmogorov-Smirnov, 'mannwhitney' for Mann-Whitney U.

    Returns
    -------
    dict with keys: 'statistic', 'p_value', 'obs_mean', 'int_mean',
                    'obs_std', 'int_std', 'effect_size'.
    """
    obs = np.asarray(obs_data).ravel()
    intv = np.asarray(int_data).ravel()

    if test == 'ks':
        stat, pval = stats.ks_2samp(obs, intv)
    elif test == 'mannwhitney':
        stat, pval = stats.mannwhitneyu(obs, intv, alternative='two-sided')
    else:
        raise ValueError(f"Unknown test: {test}")

    # Cohen's d effect size
    pooled_std = np.sqrt((obs.std()**2 + intv.std()**2) / 2)
    if pooled_std > 0:
        effect_size = (intv.mean() - obs.mean()) / pooled_std
    else:
        effect_size = 0.0

    return {
        'statistic': float(stat),
        'p_value': float(pval),
        'obs_mean': float(obs.mean()),
        'obs_std': float(obs.std()),
        'int_mean': float(intv.mean()),
        'int_std': float(intv.std()),
        'effect_size': float(effect_size),
    }


def run_single_intervention_analysis(
    process_name: str,
    interventions: Dict[str, float],
    n: int = 2000,
    seed: int = 42,
) -> Dict:
    """
    Run analysis for a single intervention on a single process.

    Returns
    -------
    dict with keys:
        'process': process name
        'intervention': intervention dict
        'obs_data': observational DataFrame
        'int_data': interventional DataFrame
        'comparisons': dict of variable -> distribution comparison results
    """
    obs_df = generate_observational_data(process_name, n=n, seed=seed)
    int_df = generate_interventional_data(process_name, interventions, n=n, seed=seed + 1)

    info = PROCESS_OBSERVABLE_VARS[process_name]
    downstream_vars = info['outputs']

    comparisons = {}
    for var in downstream_vars:
        if var in obs_df.columns and var in int_df.columns:
            comparisons[var] = compare_distributions(
                obs_df[var].values,
                int_df[var].values,
            )

    return {
        'process': process_name,
        'intervention': interventions,
        'obs_data': obs_df,
        'int_data': int_df,
        'comparisons': comparisons,
    }


def compute_reliability_under_intervention(
    process_name: str,
    interventions: Dict[str, float],
    n: int = 1000,
    seed: int = 42,
    device: str = 'cpu',
) -> Dict:
    """
    Compute reliability F under intervention using the mathematical formula.

    Generates full pipeline data with the specified process intervened upon,
    then computes F using ReliabilityFunction.

    Returns
    -------
    dict with 'F_obs' (observational F), 'F_int' (interventional F),
         'obs_outputs', 'int_outputs' (per-process outputs).
    """
    try:
        import torch
        from reliability_function import compute_reliability, ReliabilityFunction
    except ImportError as e:
        warnings.warn(f"Dependency not available for F computation: {e}")
        return None

    obs_outputs = {}
    int_outputs = {}

    for proc in PROCESS_ORDER:
        ds = PROCESS_DATASETS[proc]
        info = PROCESS_OBSERVABLE_VARS[proc]
        out_var = info['outputs'][0]

        # Observational
        obs_df = ds.sample(n, seed=seed)
        obs_outputs[proc] = obs_df[out_var].values

        # Interventional (only the target process is intervened)
        if proc == process_name:
            int_scm = ds.scm.do(interventions)
            int_df = int_scm.sample(n, seed=seed + 1)
        else:
            int_df = ds.sample(n, seed=seed + 1)
        int_outputs[proc] = int_df[out_var].values

    # Compute F for observational and interventional
    def _outputs_to_trajectory(outputs_dict):
        trajectory = {}
        for proc, vals in outputs_dict.items():
            t = torch.tensor(vals, dtype=torch.float32, device=device).unsqueeze(-1)
            trajectory[proc] = {
                'outputs_mean': t,
                'outputs_sampled': t,
                'outputs_var': torch.zeros_like(t),
            }
        return trajectory

    try:
        rf = ReliabilityFunction(device=device)
        F_obs = rf.compute_reliability(_outputs_to_trajectory(obs_outputs)).detach().cpu().numpy()
        F_int = rf.compute_reliability(_outputs_to_trajectory(int_outputs)).detach().cpu().numpy()
    except Exception as e:
        warnings.warn(f"Reliability computation failed: {e}")
        F_obs = None
        F_int = None

    return {
        'F_obs': F_obs,
        'F_int': F_int,
        'obs_outputs': obs_outputs,
        'int_outputs': int_outputs,
    }


def run_interventional_analysis(
    interventions: Optional[Dict[str, List[Dict[str, float]]]] = None,
    n: int = 2000,
    seed: int = 42,
    device: str = 'cpu',
) -> Dict:
    """
    Run full interventional validation analysis across all processes.

    Parameters
    ----------
    interventions : dict, optional
        Mapping process_name -> list of intervention dicts.
        Defaults to DEFAULT_INTERVENTIONS.
    n : int
        Samples per condition.
    seed : int
        Base random seed.
    device : str
        Compute device.

    Returns
    -------
    dict with keys:
        'per_process': dict of process -> list of single-intervention results
        'reliability': dict of process -> list of reliability comparison results
        'summary_table': pd.DataFrame with summary statistics
    """
    if interventions is None:
        interventions = DEFAULT_INTERVENTIONS

    per_process = {}
    reliability_results = {}
    summary_rows = []

    for proc_name, int_list in interventions.items():
        if proc_name not in PROCESS_DATASETS:
            warnings.warn(f"Unknown process: {proc_name}. Skipping.")
            continue

        per_process[proc_name] = []
        reliability_results[proc_name] = []

        for i, intv in enumerate(int_list):
            # Distribution comparison
            result = run_single_intervention_analysis(
                proc_name, intv, n=n, seed=seed + i * 100,
            )
            per_process[proc_name].append(result)

            # Build summary row
            intv_str = ', '.join(f"do({k}={v})" for k, v in intv.items())
            for var, comp in result['comparisons'].items():
                summary_rows.append({
                    'process': proc_name,
                    'intervention': intv_str,
                    'variable': var,
                    'obs_mean': comp['obs_mean'],
                    'int_mean': comp['int_mean'],
                    'effect_size': comp['effect_size'],
                    'p_value': comp['p_value'],
                    'significant': comp['p_value'] < 0.05,
                })

            # Reliability comparison
            rel_result = compute_reliability_under_intervention(
                proc_name, intv, n=min(n, 500), seed=seed + i * 100,
                device=device,
            )
            if rel_result is not None:
                reliability_results[proc_name].append({
                    'intervention': intv,
                    **rel_result,
                })

    summary_table = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()

    return {
        'per_process': per_process,
        'reliability': reliability_results,
        'summary_table': summary_table,
    }
