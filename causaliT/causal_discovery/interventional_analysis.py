"""
Interventional Data Generation & Causal Validation.

Inspired by ``causal_validation.ipynb`` (Appendix V) from
causal-chamber-paper.  Uses ``SCM.do()`` to generate data under
interventions and validates:

1. That CausaliT correctly captures the causal effect on downstream
   variables and on the reliability score *F*.
2. That the statistical footprint of interventions matches ground-truth
   expectations (parents are affected, non-descendants are not).

Produces validation tables analogous to Appendix V of causal-chamber-paper.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None  # torch-dependent methods will raise at call time

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scm_ds.scm import SCMDataset

from .ground_truth import DEFAULT_PROCESS_ORDER, get_observable_variables


class InterventionalAnalyzer:
    """Analyse the effects of do-interventions on the Azimuth pipeline.

    Parameters
    ----------
    datasets : dict
        ``{process_name: SCMDataset}``.
    process_order : list of str, optional
        Chain order.
    reliability_fn : callable, optional
        A callable ``reliability_fn(trajectory) -> F`` that mirrors
        :class:`ReliabilityFunction.compute_reliability`.
        If *None*, F-related validations are skipped.
    """

    def __init__(
        self,
        datasets: Dict[str, SCMDataset],
        process_order: Optional[List[str]] = None,
        reliability_fn=None,
    ):
        self.datasets = datasets
        self.process_order = process_order or DEFAULT_PROCESS_ORDER
        self.reliability_fn = reliability_fn
        self.obs_vars = get_observable_variables(
            datasets, self.process_order, include_F=False
        )

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample_process(
        self,
        process_name: str,
        n: int,
        seed: int,
        interventions: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """Sample from a single process, optionally with interventions."""
        ds = self.datasets[process_name]
        if interventions:
            applicable = {k: v for k, v in interventions.items() if k in ds.scm.specs}
            if applicable:
                scm_do = ds.scm.do(applicable)
                return scm_do.sample(n, seed=seed)
        return ds.scm.sample(n, seed=seed)

    def _sample_observable(
        self,
        n: int,
        seed: int,
        interventions: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """Sample all processes and return observable-variable DataFrame."""
        merged = pd.DataFrame(index=range(n))
        for proc in self.process_order:
            if proc not in self.datasets:
                continue
            df_full = self._sample_process(proc, n, seed, interventions)
            ds = self.datasets[proc]
            obs = [v for v in ds.input_labels + ds.target_labels if v in df_full.columns]
            for col in obs:
                if col not in merged.columns:
                    merged[col] = df_full[col].values
        return merged[[v for v in self.obs_vars if v in merged.columns]]

    # ------------------------------------------------------------------
    # Intervention effect analysis
    # ------------------------------------------------------------------

    def compare_distributions(
        self,
        var: str,
        value: float,
        n_samples: int = 5000,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Compare observational vs interventional distributions.

        For each observable variable, computes the mean shift and a
        two-sample Kolmogorov-Smirnov test p-value.

        Parameters
        ----------
        var : str
            Variable to intervene on.
        value : float
            Intervention value.
        n_samples : int
            Number of samples.
        seed : int
            Random seed.

        Returns
        -------
        pd.DataFrame
            One row per observable variable with columns:
            ``variable, obs_mean, obs_std, int_mean, int_std,
              mean_shift, ks_statistic, ks_pvalue``.
        """
        from scipy.stats import ks_2samp

        df_obs = self._sample_observable(n_samples, seed)
        df_int = self._sample_observable(n_samples, seed + 1, interventions={var: value})

        rows = []
        for col in df_obs.columns:
            obs = df_obs[col].values
            intv = df_int[col].values
            ks_stat, ks_p = ks_2samp(obs, intv)
            rows.append(
                {
                    "variable": col,
                    "obs_mean": obs.mean(),
                    "obs_std": obs.std(),
                    "int_mean": intv.mean(),
                    "int_std": intv.std(),
                    "mean_shift": intv.mean() - obs.mean(),
                    "ks_statistic": ks_stat,
                    "ks_pvalue": ks_p,
                }
            )
        return pd.DataFrame(rows).set_index("variable")

    def run_all_interventions(
        self,
        intervention_specs: Optional[List[Tuple[str, float]]] = None,
        n_samples: int = 5000,
        seed: int = 42,
    ) -> Dict[str, pd.DataFrame]:
        """Run interventions on multiple variables.

        Parameters
        ----------
        intervention_specs : list of (var, value), optional
            Variables and values to intervene on.  If *None*, uses
            sensible defaults for each process's inputs.
        n_samples : int
            Samples per condition.
        seed : int
            Base seed.

        Returns
        -------
        dict
            ``{intervention_label: distribution_comparison_df}``.
        """
        if intervention_specs is None:
            # Default interventions: mid-range values for each process input
            intervention_specs = []
            for proc in self.process_order:
                if proc not in self.datasets:
                    continue
                ds = self.datasets[proc]
                for inp in ds.input_labels:
                    # Use a fixed representative value
                    intervention_specs.append((inp, 0.5))

        results = {}
        for i, (var, value) in enumerate(intervention_specs):
            label = f"do({var}={value})"
            results[label] = self.compare_distributions(
                var, value, n_samples, seed=seed + i
            )
        return results

    # ------------------------------------------------------------------
    # CausaliT prediction under intervention
    # ------------------------------------------------------------------

    def validate_causalit_intervention(
        self,
        model: torch.nn.Module,
        var: str,
        value: float,
        n_samples: int = 1000,
        seed: int = 42,
        model_forward_fn=None,
    ) -> Dict[str, float]:
        """Validate that CausaliT captures interventional effects on F.

        Generates trajectories under ``do(var=value)`` and compares:
        - F predicted by CausaliT (surrogate)
        - F computed by the ReliabilityFunction

        Parameters
        ----------
        model : nn.Module
            Trained CausaliT/surrogate model.
        var : str
            Intervention variable.
        value : float
            Intervention value.
        n_samples : int
            Batch size.
        seed : int
            Random seed.
        model_forward_fn : callable, optional
            Custom ``fn(model, data) -> F_pred``.  If *None*, uses
            ``model(data)``.

        Returns
        -------
        dict
            ``{"F_pred_mean", "F_pred_std", "F_true_mean", "F_true_std",
              "mae", "correlation"}``.
        """
        if self.reliability_fn is None:
            raise ValueError(
                "reliability_fn must be provided for CausaliT intervention validation"
            )

        # Sample interventional trajectory data
        df_int = self._sample_observable(n_samples, seed, interventions={var: value})

        # Build trajectory for reliability function
        trajectory = {}
        for proc in self.process_order:
            if proc not in self.datasets:
                continue
            ds = self.datasets[proc]
            output_cols = [c for c in ds.target_labels if c in df_int.columns]
            if output_cols:
                trajectory[proc] = {
                    "outputs_mean": torch.tensor(
                        df_int[output_cols].values, dtype=torch.float32
                    ),
                }

        # Compute ground truth F
        F_true = self.reliability_fn(trajectory)
        if isinstance(F_true, tuple):
            F_true = F_true[0]
        F_true = F_true.detach().cpu().numpy()

        # Compute CausaliT prediction
        input_tensor = torch.tensor(df_int.values, dtype=torch.float32)
        if model_forward_fn:
            F_pred = model_forward_fn(model, input_tensor)
        else:
            with torch.no_grad():
                F_pred = model(input_tensor)
        F_pred = F_pred.detach().cpu().numpy()

        # Metrics
        mae = float(np.mean(np.abs(F_pred - F_true)))
        corr = float(np.corrcoef(F_pred.flatten(), F_true.flatten())[0, 1])

        return {
            "F_pred_mean": float(F_pred.mean()),
            "F_pred_std": float(F_pred.std()),
            "F_true_mean": float(F_true.mean()),
            "F_true_std": float(F_true.std()),
            "mae": mae,
            "correlation": corr,
        }

    # ------------------------------------------------------------------
    # P-value matrix (Appendix V style)
    # ------------------------------------------------------------------

    def compute_pvalue_matrix(
        self,
        intervention_vars: Optional[List[str]] = None,
        n_samples: int = 5000,
        seed: int = 42,
        significance: float = 0.05,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Build a p-value matrix for interventional validation.

        For each pair ``(intervention_var, response_var)``, performs a
        Kolmogorov-Smirnov test between observational and interventional
        distributions.  This mirrors the Appendix V analysis from
        causal-chamber-paper.

        Parameters
        ----------
        intervention_vars : list of str, optional
            Variables to intervene on.  Defaults to all input labels.
        n_samples : int
            Samples per condition.
        seed : int
            Base seed.
        significance : float
            Significance threshold for highlighting.

        Returns
        -------
        pvalue_matrix : pd.DataFrame
            ``(n_interventions, n_obs_vars)`` of p-values.
        significant_matrix : pd.DataFrame
            Boolean matrix where ``True`` means ``p < significance``.
        """
        from scipy.stats import ks_2samp

        if intervention_vars is None:
            intervention_vars = []
            for proc in self.process_order:
                if proc not in self.datasets:
                    continue
                intervention_vars.extend(self.datasets[proc].input_labels)

        # Observational reference
        df_obs = self._sample_observable(n_samples, seed)

        pvals = np.ones((len(intervention_vars), len(self.obs_vars)))
        for i, ivar in enumerate(intervention_vars):
            # Determine a meaningful intervention value
            if ivar in df_obs.columns:
                int_value = float(df_obs[ivar].quantile(0.9))
            else:
                int_value = 1.0
            df_int = self._sample_observable(
                n_samples, seed + i + 1, interventions={ivar: int_value}
            )
            for j, rvar in enumerate(self.obs_vars):
                if rvar in df_obs.columns and rvar in df_int.columns:
                    _, p = ks_2samp(df_obs[rvar].values, df_int[rvar].values)
                    pvals[i, j] = p

        pvalue_df = pd.DataFrame(
            pvals, index=intervention_vars, columns=self.obs_vars
        )
        sig_df = pvalue_df < significance
        return pvalue_df, sig_df
