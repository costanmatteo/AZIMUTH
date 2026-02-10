"""
Interventional data generation and causal validation.

Uses ``SCM.do()`` to generate data under interventions, compares
pre/post-intervention distributions on downstream variables, and verifies
that CausaliT correctly captures intervention effects on F.

Inspired by ``causal_validation.ipynb`` from causal-chamber-paper
(Appendix V style tables).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scm_ds.scm import SCMDataset
from reliability_function.src.compute_reliability import ReliabilityFunction


# ---------------------------------------------------------------------------
# Data generation under interventions
# ---------------------------------------------------------------------------

def generate_interventional_data(
    dataset: SCMDataset,
    interventions: Dict[str, float],
    n_samples: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample from an SCM under do-interventions.

    Parameters
    ----------
    dataset : SCMDataset
        Process dataset (contains the SCM).
    interventions : dict
        ``{variable_name: constant_value}`` for the do-operator.
    n_samples : int
        Number of samples.
    seed : int
        Random seed.

    Returns
    -------
    DataFrame
        Interventional samples.
    """
    scm_do = dataset.scm.do(interventions)
    return scm_do.sample(n_samples, seed=seed)


def compare_distributions(
    obs_data: pd.DataFrame,
    int_data: pd.DataFrame,
    variables: List[str],
) -> pd.DataFrame:
    """Compare observational vs interventional distributions.

    Parameters
    ----------
    obs_data, int_data : DataFrame
        Observational and interventional samples.
    variables : list[str]
        Variables to compare.

    Returns
    -------
    DataFrame
        Summary statistics (mean, std, median, KS test) per variable.
    """
    rows = []
    for var in variables:
        if var not in obs_data.columns or var not in int_data.columns:
            continue

        obs = obs_data[var].values
        int_ = int_data[var].values

        row = {
            "variable": var,
            "obs_mean": np.mean(obs),
            "obs_std": np.std(obs),
            "int_mean": np.mean(int_),
            "int_std": np.std(int_),
            "delta_mean": np.mean(int_) - np.mean(obs),
            "delta_std": np.std(int_) - np.std(obs),
        }

        # KS test for distributional shift
        try:
            from scipy.stats import ks_2samp
            ks_stat, ks_pval = ks_2samp(obs, int_)
            row["ks_statistic"] = ks_stat
            row["ks_pvalue"] = ks_pval
        except ImportError:
            row["ks_statistic"] = np.nan
            row["ks_pvalue"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Intervention effect on F
# ---------------------------------------------------------------------------

def _build_trajectory_from_samples(
    process_samples: Dict[str, pd.DataFrame],
    process_order: List[str],
    output_map: Dict[str, str],
) -> Dict[str, dict]:
    """Convert per-process DataFrames into a trajectory dict for
    ``ReliabilityFunction``.

    Parameters
    ----------
    process_samples : dict
        ``{"laser": df_laser, ...}`` with observational samples.
    process_order : list[str]
        Order of processes.
    output_map : dict
        ``{"laser": "ActualPower", ...}`` mapping process to its output column.

    Returns
    -------
    dict
        Trajectory in ReliabilityFunction format.
    """
    trajectory = {}
    for proc in process_order:
        df = process_samples[proc]
        out_col = output_map[proc]
        trajectory[proc] = {
            "outputs_mean": torch.tensor(
                df[out_col].values, dtype=torch.float32
            ).unsqueeze(-1),
        }
    return trajectory


def compute_F_under_intervention(
    datasets: Dict[str, SCMDataset],
    intervention_process: str,
    interventions: Dict[str, float],
    n_samples: int = 2000,
    seed: int = 42,
    process_order: Optional[List[str]] = None,
    output_map: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """Compute F from the SCM under an intervention and compare to baseline.

    Parameters
    ----------
    datasets : dict
        ``{"laser": ds_laser, ...}``
    intervention_process : str
        Which process receives the intervention.
    interventions : dict
        ``{var: value}`` for ``scm.do()``.
    n_samples, seed : int
        Sampling parameters.
    process_order : list[str], optional
        Defaults to the standard 4-process chain.
    output_map : dict, optional
        ``{process_name: output_column_name}``.

    Returns
    -------
    dict
        ``{"F_obs": float, "F_int": float, "delta_F": float,
           "process": str, "interventions": dict}``
    """
    order = process_order or ["laser", "plasma", "galvanic", "microetch"]
    omap = output_map or {
        "laser": "ActualPower",
        "plasma": "RemovalRate",
        "galvanic": "Thickness",
        "microetch": "RemovalDepth",
    }

    rf = ReliabilityFunction()

    # Observational
    obs_samples = {}
    for proc in order:
        obs_samples[proc] = datasets[proc].sample(n_samples, seed=seed)

    traj_obs = _build_trajectory_from_samples(obs_samples, order, omap)
    F_obs = rf.compute_reliability(traj_obs)
    F_obs_val = F_obs.mean().item()

    # Interventional
    int_samples = {}
    for proc in order:
        if proc == intervention_process:
            int_samples[proc] = generate_interventional_data(
                datasets[proc], interventions, n_samples, seed
            )
        else:
            int_samples[proc] = datasets[proc].sample(n_samples, seed=seed)

    traj_int = _build_trajectory_from_samples(int_samples, order, omap)
    F_int = rf.compute_reliability(traj_int)
    F_int_val = F_int.mean().item()

    return {
        "F_obs": F_obs_val,
        "F_int": F_int_val,
        "delta_F": F_int_val - F_obs_val,
        "process": intervention_process,
        "interventions": interventions,
    }


# ---------------------------------------------------------------------------
# CausaliT prediction under interventions
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_F_with_causalit(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    device: str = "cpu",
) -> np.ndarray:
    """Run CausaliT forward and extract predictions.

    Parameters
    ----------
    model : nn.Module
        Trained CausaliT/ProT model.
    input_tensor, target_tensor : Tensor
        Input and target tensors.
    device : str
        Inference device.

    Returns
    -------
    ndarray
        Predictions of shape ``(B, out_dim)``.
    """
    model = model.to(device).eval()
    input_tensor = input_tensor.to(device)
    target_tensor = target_tensor.to(device)

    forecast, _, _, _ = model(input_tensor, target_tensor)
    return forecast.cpu().numpy()


# ---------------------------------------------------------------------------
# Full intervention validation
# ---------------------------------------------------------------------------

DEFAULT_INTERVENTIONS = [
    {"process": "laser",    "interventions": {"PowerTarget": 0.5}},
    {"process": "laser",    "interventions": {"AmbientTemp": 30.0}},
    {"process": "plasma",   "interventions": {"RF_Power": 250.0}},
    {"process": "galvanic", "interventions": {"CurrentDensity": 3.0}},
    {"process": "microetch", "interventions": {"Temperature": 310.0}},
]


def run_intervention_validation(
    datasets: Dict[str, SCMDataset],
    intervention_specs: Optional[List[Dict]] = None,
    n_samples: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run a suite of interventions and produce validation table.

    Parameters
    ----------
    datasets : dict
        SCMDataset dict.
    intervention_specs : list[dict], optional
        Each dict has ``"process"`` and ``"interventions"`` keys.
        Defaults to ``DEFAULT_INTERVENTIONS``.
    n_samples, seed : int
        Sampling parameters.

    Returns
    -------
    DataFrame
        One row per intervention with F_obs, F_int, delta_F.
    """
    specs = intervention_specs or DEFAULT_INTERVENTIONS
    results = []
    for spec in specs:
        result = compute_F_under_intervention(
            datasets=datasets,
            intervention_process=spec["process"],
            interventions=spec["interventions"],
            n_samples=n_samples,
            seed=seed,
        )
        # Flatten interventions for display
        int_str = ", ".join(f"{k}={v}" for k, v in spec["interventions"].items())
        result["intervention_str"] = int_str
        results.append(result)

    return pd.DataFrame(results)


def run_distributional_comparison(
    datasets: Dict[str, SCMDataset],
    intervention_specs: Optional[List[Dict]] = None,
    n_samples: int = 2000,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """For each intervention, compare obs vs int distributions of
    downstream variables.

    Returns
    -------
    dict
        ``{intervention_label: DataFrame}`` with distributional comparisons.
    """
    specs = intervention_specs or DEFAULT_INTERVENTIONS
    comparisons = {}

    for spec in specs:
        proc = spec["process"]
        ds = datasets[proc]

        obs_data = ds.sample(n_samples, seed=seed)
        int_data = generate_interventional_data(ds, spec["interventions"], n_samples, seed)

        # Compare all observable variables
        all_vars = ds.input_labels + ds.target_labels
        comp = compare_distributions(obs_data, int_data, all_vars)

        label = f"do({', '.join(f'{k}={v}' for k, v in spec['interventions'].items())})"
        comparisons[label] = comp

    return comparisons
