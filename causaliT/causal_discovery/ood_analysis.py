"""
Out-of-Distribution (OOD) Analysis.

Inspired by ``ood_sensors.ipynb``, ``ood_images.ipynb``, and
``ood_impulses.ipynb`` from causal-chamber-paper.

Tests the robustness of the Azimuth pipeline under distributional shift
by generating trajectories with shifted input distributions and measuring
the degradation of:

1. Uncertainty predictor accuracy (R^2, calibration ratio, coverage).
2. CausaliT reliability-F prediction accuracy.
3. Attention graph stability (does the discovered structure change?).

Produces an in-distribution vs OOD comparison report.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scm_ds.scm import SCMDataset, NoiseModel, SCM

from .ground_truth import DEFAULT_PROCESS_ORDER, _prefixed, get_observable_variables
from .metrics import compare_graphs


# ---------------------------------------------------------------------------
# Shift specifications
# ---------------------------------------------------------------------------

class DistributionShift:
    """Specification for a distributional shift on an SCM input variable.

    Parameters
    ----------
    variable : str
        Name of the variable whose noise distribution is shifted.
    id_range : tuple
        ``(low, high)`` for the in-distribution range.
    ood_range : tuple
        ``(low, high)`` for the out-of-distribution range.
    description : str
        Human-readable description.
    """

    def __init__(
        self,
        variable: str,
        id_range: Tuple[float, float],
        ood_range: Tuple[float, float],
        description: str = "",
    ):
        self.variable = variable
        self.id_range = id_range
        self.ood_range = ood_range
        self.description = description or f"{variable}: ID={id_range} -> OOD={ood_range}"

    def __repr__(self):
        return (
            f"DistributionShift({self.variable}, "
            f"id={self.id_range}, ood={self.ood_range})"
        )


# ---------------------------------------------------------------------------
# Default shifts
# ---------------------------------------------------------------------------

DEFAULT_SHIFTS = [
    DistributionShift(
        "AmbientTemp",
        id_range=(15.0, 25.0),
        ood_range=(25.0, 35.0),
        description="Laser: AmbientTemp shift to higher range",
    ),
    DistributionShift(
        "PowerTarget",
        id_range=(0.10, 0.6),
        ood_range=(0.6, 1.0),
        description="Laser: PowerTarget shift to higher range",
    ),
    DistributionShift(
        "RF_Power",
        id_range=(100.0, 250.0),
        ood_range=(250.0, 400.0),
        description="Plasma: RF_Power shift to higher range",
    ),
    DistributionShift(
        "Temperature",
        id_range=(293.0, 308.0),
        ood_range=(308.0, 323.0),
        description="Microetch: Temperature shift to higher range",
    ),
]


class OODAnalyzer:
    """Analyse OOD robustness of the Azimuth pipeline.

    Parameters
    ----------
    datasets : dict
        ``{process_name: SCMDataset}``.
    process_order : list of str, optional
        Manufacturing chain ordering.
    """

    def __init__(
        self,
        datasets: Dict[str, SCMDataset],
        process_order: Optional[List[str]] = None,
    ):
        self.datasets = datasets
        self.process_order = process_order or DEFAULT_PROCESS_ORDER
        self.obs_vars = get_observable_variables(
            datasets, self.process_order, include_F=False
        )

    # ------------------------------------------------------------------
    # Modified sampling
    # ------------------------------------------------------------------

    def _make_shifted_sampler(
        self, low: float, high: float
    ) -> Callable:
        """Return a noise sampler for Uniform(low, high)."""
        def sampler(rng, n):
            return rng.uniform(low=low, high=high, size=n)
        return sampler

    def _sample_with_shift(
        self,
        shift: DistributionShift,
        use_ood: bool,
        n_samples: int,
        seed: int,
    ) -> pd.DataFrame:
        """Sample observable data with optionally shifted input distribution.

        Parameters
        ----------
        shift : DistributionShift
            Which variable to shift and how.
        use_ood : bool
            If *True*, use the OOD range; otherwise use the ID range.
        n_samples : int
            Number of samples.
        seed : int
            Random seed.

        Returns
        -------
        pd.DataFrame
            Observable variables.
        """
        rng_range = shift.ood_range if use_ood else shift.id_range

        merged = pd.DataFrame(index=range(n_samples))
        for proc in self.process_order:
            if proc not in self.datasets:
                continue
            ds = self.datasets[proc]
            scm = ds.scm

            # Check if the shift variable belongs to this process
            if shift.variable in scm.specs:
                # Build modified noise model
                new_sampler = self._make_shifted_sampler(rng_range[0], rng_range[1])
                if isinstance(scm.noise, NoiseModel):
                    modified_singles = dict(scm.noise.singles)
                    modified_singles[shift.variable] = new_sampler
                    modified_nm = NoiseModel(singles=modified_singles, groups=scm.noise.groups)
                    modified_scm = SCM(list(scm.specs.values()), noise_model=modified_nm)
                elif isinstance(scm.noise, dict):
                    modified_noise = dict(scm.noise)
                    modified_noise[shift.variable] = new_sampler
                    modified_scm = SCM(list(scm.specs.values()))
                    modified_scm.noise = modified_noise
                else:
                    modified_scm = scm

                df_full = modified_scm.sample(n_samples, seed=seed)
            else:
                df_full = scm.sample(n_samples, seed=seed)

            obs = [v for v in ds.input_labels + ds.target_labels if v in df_full.columns]
            for col in obs:
                merged[_prefixed(proc, col)] = df_full[col].values

        return merged[self.obs_vars]

    # ------------------------------------------------------------------
    # OOD comparison
    # ------------------------------------------------------------------

    def compare_distributions_ood(
        self,
        shift: DistributionShift,
        n_samples: int = 5000,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Compare ID vs OOD distributions for all observable variables.

        Parameters
        ----------
        shift : DistributionShift
            Shift specification.
        n_samples : int
            Samples per condition.
        seed : int
            Random seed.

        Returns
        -------
        pd.DataFrame
            Comparison statistics per variable.
        """
        from scipy.stats import ks_2samp

        df_id = self._sample_with_shift(shift, use_ood=False, n_samples=n_samples, seed=seed)
        df_ood = self._sample_with_shift(shift, use_ood=True, n_samples=n_samples, seed=seed + 100)

        rows = []
        for col in df_id.columns:
            id_vals = df_id[col].values
            ood_vals = df_ood[col].values
            ks_stat, ks_p = ks_2samp(id_vals, ood_vals)
            rows.append({
                "variable": col,
                "id_mean": id_vals.mean(),
                "id_std": id_vals.std(),
                "ood_mean": ood_vals.mean(),
                "ood_std": ood_vals.std(),
                "mean_shift": ood_vals.mean() - id_vals.mean(),
                "relative_shift": (
                    abs(ood_vals.mean() - id_vals.mean()) / (abs(id_vals.mean()) + 1e-8)
                ),
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_p,
            })
        return pd.DataFrame(rows).set_index("variable")

    def evaluate_model_degradation(
        self,
        shift: DistributionShift,
        predict_fn: Callable[[pd.DataFrame], np.ndarray],
        target_var: str,
        n_samples: int = 5000,
        seed: int = 42,
    ) -> Dict[str, Dict[str, float]]:
        """Measure prediction accuracy degradation under OOD shift.

        Parameters
        ----------
        shift : DistributionShift
            Shift specification.
        predict_fn : callable
            ``fn(df) -> predictions`` (array of shape ``(n,)``).
        target_var : str
            Ground-truth variable name in the DataFrame.
        n_samples : int
            Samples per condition.
        seed : int
            Random seed.

        Returns
        -------
        dict
            ``{"id": {R2, MAE, RMSE}, "ood": {R2, MAE, RMSE}}``.
        """
        results = {}
        for label, use_ood in [("id", False), ("ood", True)]:
            df = self._sample_with_shift(shift, use_ood, n_samples, seed)
            y_true = df[target_var].values
            y_pred = predict_fn(df)

            mse = float(np.mean((y_true - y_pred) ** 2))
            mae = float(np.mean(np.abs(y_true - y_pred)))
            rmse = float(np.sqrt(mse))
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

            results[label] = {"R2": r2, "MAE": mae, "RMSE": rmse}

        return results

    def evaluate_attention_stability(
        self,
        shift: DistributionShift,
        extract_graph_fn: Callable[[pd.DataFrame], pd.DataFrame],
        ground_truth: pd.DataFrame,
        n_samples: int = 5000,
        seed: int = 42,
    ) -> Dict[str, Dict[str, float]]:
        """Check whether the attention-discovered graph changes under OOD.

        Parameters
        ----------
        shift : DistributionShift
            Shift specification.
        extract_graph_fn : callable
            ``fn(data_df) -> binary_adj_df`` that runs the full
            extraction pipeline.
        ground_truth : pd.DataFrame
            Ground-truth adjacency.
        n_samples : int
            Samples.
        seed : int
            Seed.

        Returns
        -------
        dict
            ``{"id": metrics, "ood": metrics, "graph_changed": bool}``.
        """
        results = {}
        adjs = {}
        for label, use_ood in [("id", False), ("ood", True)]:
            df = self._sample_with_shift(shift, use_ood, n_samples, seed)
            adj = extract_graph_fn(df)
            adjs[label] = adj
            results[label] = compare_graphs(adj, ground_truth, label=label)

        # Check structural change between ID and OOD graphs
        id_arr = adjs["id"].values if isinstance(adjs["id"], pd.DataFrame) else adjs["id"]
        ood_arr = adjs["ood"].values if isinstance(adjs["ood"], pd.DataFrame) else adjs["ood"]
        results["graph_changed"] = not np.array_equal(id_arr, ood_arr)
        results["id_vs_ood_shd"] = int(np.sum(id_arr != ood_arr))

        return results

    def run_full_ood_analysis(
        self,
        shifts: Optional[List[DistributionShift]] = None,
        n_samples: int = 5000,
        seed: int = 42,
    ) -> Dict[str, pd.DataFrame]:
        """Run distributional comparison for all specified shifts.

        Parameters
        ----------
        shifts : list of DistributionShift, optional
            Shifts to test.  Defaults to :data:`DEFAULT_SHIFTS`.
        n_samples, seed : int
            Sampling parameters.

        Returns
        -------
        dict
            ``{shift_description: comparison_df}``.
        """
        shifts = shifts or DEFAULT_SHIFTS
        results = {}
        for shift in shifts:
            results[shift.description] = self.compare_distributions_ood(
                shift, n_samples, seed
            )
        return results
