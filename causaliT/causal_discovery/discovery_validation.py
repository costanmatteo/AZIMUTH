"""
Causal Discovery Validation.

Replicates the analysis style of ``causal_discovery_iid.ipynb`` and
``causal_discovery_time.ipynb`` from causal-chamber-paper, adapted
to the Azimuth pipeline:

1. Generate i.i.d. data from the SCM trajectory.
2. Apply attention-based discovery (primary) and optionally classical
   algorithms (GES, PC) as baselines.
3. Compare estimated graphs to the ground-truth DAG.
4. Report precision, recall, F1, SHD in a summary table.
5. Optionally include interventional data (from point 3 of the pipeline).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scm_ds.scm import SCMDataset

from .ground_truth import (
    DEFAULT_PROCESS_ORDER,
    extract_ground_truth_dag,
    get_observable_variables,
)
from .metrics import (
    compare_graphs,
    run_ges_baseline,
    run_pc_baseline,
)


class DiscoveryValidator:
    """Orchestrates the full causal discovery validation pipeline.

    Parameters
    ----------
    datasets : dict
        ``{process_name: SCMDataset}``.
    process_order : list of str, optional
        Manufacturing chain order.
    """

    def __init__(
        self,
        datasets: Dict[str, SCMDataset],
        process_order: Optional[List[str]] = None,
    ):
        self.datasets = datasets
        self.process_order = process_order or DEFAULT_PROCESS_ORDER
        self.ground_truth = extract_ground_truth_dag(
            datasets, self.process_order, include_F=False
        )
        self.obs_vars = get_observable_variables(
            datasets, self.process_order, include_F=False
        )

    # ------------------------------------------------------------------
    # Data generation helpers
    # ------------------------------------------------------------------

    def generate_iid_data(
        self,
        n_samples: int = 5000,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate i.i.d. observational data from all processes.

        Samples each process SCM independently and concatenates columns
        for the observable variables.

        Parameters
        ----------
        n_samples : int
            Number of i.i.d. samples.
        seed : int
            Random seed.

        Returns
        -------
        pd.DataFrame
            ``(n_samples, n_obs_vars)`` data frame.
        """
        dfs: Dict[str, pd.DataFrame] = {}
        for proc in self.process_order:
            if proc not in self.datasets:
                continue
            ds = self.datasets[proc]
            df_full = ds.scm.sample(n_samples, seed=seed)
            obs = [v for v in ds.input_labels + ds.target_labels if v in df_full.columns]
            dfs[proc] = df_full[obs]

        # Merge on sample index, keeping unique columns
        merged = pd.DataFrame(index=range(n_samples))
        for proc in self.process_order:
            if proc in dfs:
                for col in dfs[proc].columns:
                    if col not in merged.columns:
                        merged[col] = dfs[proc][col].values

        return merged[self.obs_vars]

    def generate_interventional_data(
        self,
        intervention_var: str,
        intervention_value: float,
        n_samples: int = 5000,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate data under a do-intervention.

        Parameters
        ----------
        intervention_var : str
            Variable to intervene on.
        intervention_value : float
            Constant value to set.
        n_samples : int
            Number of samples.
        seed : int
            Random seed.

        Returns
        -------
        pd.DataFrame
            Interventional data.
        """
        dfs: Dict[str, pd.DataFrame] = {}
        for proc in self.process_order:
            if proc not in self.datasets:
                continue
            ds = self.datasets[proc]
            if intervention_var in ds.scm.specs:
                scm_do = ds.scm.do({intervention_var: intervention_value})
                df_full = scm_do.sample(n_samples, seed=seed)
            else:
                df_full = ds.scm.sample(n_samples, seed=seed)
            obs = [v for v in ds.input_labels + ds.target_labels if v in df_full.columns]
            dfs[proc] = df_full[obs]

        merged = pd.DataFrame(index=range(n_samples))
        for proc in self.process_order:
            if proc in dfs:
                for col in dfs[proc].columns:
                    if col not in merged.columns:
                        merged[col] = dfs[proc][col].values
        return merged[[v for v in self.obs_vars if v in merged.columns]]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_attention_graph(
        self,
        estimated_adj: Union[np.ndarray, pd.DataFrame],
        label: str = "attention",
    ) -> Dict[str, float]:
        """Compare an attention-derived adjacency to ground truth.

        Parameters
        ----------
        estimated_adj : array or DataFrame
            Binary adjacency from :class:`AttentionGraphExtractor`.
        label : str
            Prefix for the result keys.

        Returns
        -------
        dict
            Metrics (precision, recall, F1, SHD).
        """
        return compare_graphs(estimated_adj, self.ground_truth, label=label)

    def validate_classical_baselines(
        self,
        data: pd.DataFrame,
        run_ges: bool = True,
        run_pc: bool = True,
        pc_alpha: float = 0.05,
    ) -> Dict[str, Dict[str, float]]:
        """Run classical discovery algorithms and compare to ground truth.

        Parameters
        ----------
        data : DataFrame
            Observational data (n_samples, n_obs_vars).
        run_ges : bool
            Run GES.
        run_pc : bool
            Run PC.
        pc_alpha : float
            Significance for PC.

        Returns
        -------
        dict
            ``{algorithm_name: metrics_dict}``.
        """
        results: Dict[str, Dict[str, float]] = {}
        X = data.values

        if run_ges:
            try:
                est_ges = run_ges_baseline(X, variable_names=list(data.columns))
                results["GES"] = compare_graphs(est_ges, self.ground_truth, label="GES")
                results["GES"]["adjacency"] = est_ges
            except ImportError as e:
                results["GES"] = {"error": str(e)}

        if run_pc:
            try:
                est_pc = run_pc_baseline(X, alpha=pc_alpha, variable_names=list(data.columns))
                results["PC"] = compare_graphs(est_pc, self.ground_truth, label="PC")
                results["PC"]["adjacency"] = est_pc
            except ImportError as e:
                results["PC"] = {"error": str(e)}

        return results

    def run_full_validation(
        self,
        estimated_adj: Union[np.ndarray, pd.DataFrame],
        n_samples: int = 5000,
        seed: int = 42,
        run_classical: bool = True,
        pc_alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Run the complete validation pipeline.

        Parameters
        ----------
        estimated_adj : array or DataFrame
            Attention-based adjacency matrix.
        n_samples : int
            Samples for classical baselines.
        seed : int
            Random seed.
        run_classical : bool
            Whether to include GES/PC baselines.
        pc_alpha : float
            PC significance level.

        Returns
        -------
        pd.DataFrame
            Summary table with one row per method and columns for each metric.
        """
        rows = []

        # 1. Attention-based
        att_metrics = self.validate_attention_graph(estimated_adj, label="")
        rows.append({"method": "Attention", **att_metrics})

        # 2. Classical baselines
        if run_classical:
            data = self.generate_iid_data(n_samples, seed)
            classical = self.validate_classical_baselines(data, pc_alpha=pc_alpha)
            for algo, metrics in classical.items():
                if "error" not in metrics:
                    clean = {k: v for k, v in metrics.items() if k != "adjacency"}
                    # Strip algorithm prefix from keys
                    clean_keys = {}
                    for k, v in clean.items():
                        stripped = k.replace(f"{algo}_", "")
                        clean_keys[stripped] = v
                    rows.append({"method": algo, **clean_keys})
                else:
                    rows.append({"method": algo, "error": metrics["error"]})

        return pd.DataFrame(rows).set_index("method")

    def sweep_thresholds(
        self,
        var_attention: pd.DataFrame,
        thresholds: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """Evaluate attention discovery across multiple thresholds.

        Parameters
        ----------
        var_attention : DataFrame
            Variable-level attention weights (continuous).
        thresholds : list of float, optional
            Threshold values to sweep.

        Returns
        -------
        pd.DataFrame
            Metrics for each threshold value.
        """
        if thresholds is None:
            thresholds = np.arange(0.01, 0.5, 0.02).tolist()

        from .attention_discovery import AttentionGraphExtractor

        rows = []
        for thr in thresholds:
            adj = AttentionGraphExtractor.threshold_to_adjacency(var_attention, threshold=thr)
            metrics = compare_graphs(adj, self.ground_truth)
            rows.append({"threshold": thr, **metrics})

        return pd.DataFrame(rows)
