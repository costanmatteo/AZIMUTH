"""
Symbolic Regression Analysis.

Inspired by ``symbolic_regression.ipynb`` from causal-chamber-paper.

For each SCM process, generates (input, output) pairs and tests whether
symbolic regression can rediscover the known structural equations.
Compares discovered expressions against the ground truth in
``NodeSpec.expr``.

Supports:
- PySR (optional, high-quality symbolic regression)
- Polynomial regression as a fallback baseline

Metrics: R^2 of the discovered expression, symbolic match quality.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scm_ds.scm import SCMDataset

from .ground_truth import DEFAULT_PROCESS_ORDER


class SymbolicAnalyzer:
    """Symbolic regression analysis for SCM equation recovery.

    Parameters
    ----------
    datasets : dict
        ``{process_name: SCMDataset}``.
    process_order : list of str, optional
        Chain ordering.
    """

    def __init__(
        self,
        datasets: Dict[str, SCMDataset],
        process_order: Optional[List[str]] = None,
    ):
        self.datasets = datasets
        self.process_order = process_order or DEFAULT_PROCESS_ORDER

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------

    def generate_io_pairs(
        self,
        process_name: str,
        n_samples: int = 10000,
        seed: int = 42,
        noise_free: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate (input, output) pairs for a specific process.

        Parameters
        ----------
        process_name : str
            Name of the process.
        n_samples : int
            Number of samples.
        seed : int
            Random seed.
        noise_free : bool
            If *True*, zero out process noise variables so the data
            reflects the deterministic structural equation only.

        Returns
        -------
        X : pd.DataFrame
            Input variables ``(n_samples, n_inputs)``.
        Y : pd.DataFrame
            Output variables ``(n_samples, n_outputs)``.
        """
        ds = self.datasets[process_name]
        scm = ds.scm

        if noise_free and hasattr(ds, "process_noise_vars"):
            # Zero out process noise by intervening
            interventions = {}
            for nvar in ds.process_noise_vars:
                if nvar in scm.specs:
                    # Set noise-sourced nodes to their deterministic value (0)
                    # For standard-normal noise vars, this zeroes the noise
                    interventions[nvar] = 0.0
            if interventions:
                scm_clean = scm.do(interventions)
                df = scm_clean.sample(n_samples, seed=seed)
            else:
                df = scm.sample(n_samples, seed=seed)
        else:
            df = scm.sample(n_samples, seed=seed)

        X = df[ds.input_labels]
        Y = df[ds.target_labels]
        return X, Y

    def get_ground_truth_expression(self, process_name: str) -> Dict[str, str]:
        """Retrieve the ground-truth structural equation for each output.

        Parameters
        ----------
        process_name : str
            Process name.

        Returns
        -------
        dict
            ``{output_var: expression_string}``.
        """
        ds = self.datasets[process_name]
        expressions = {}
        for target in ds.target_labels:
            if target in ds.scm.specs:
                expressions[target] = ds.scm.specs[target].expr
        return expressions

    # ------------------------------------------------------------------
    # Regression methods
    # ------------------------------------------------------------------

    def fit_polynomial(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        max_degree: int = 4,
    ) -> Dict[str, Any]:
        """Fit polynomial regression and return best-degree result.

        Parameters
        ----------
        X : DataFrame
            Input features.
        y : array
            Target values.
        max_degree : int
            Maximum polynomial degree to try.

        Returns
        -------
        dict
            ``{"degree", "r2_train", "r2_test", "coefficients",
              "expression"}``.
        """
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split

        X_arr = X.values
        X_train, X_test, y_train, y_test = train_test_split(
            X_arr, y, test_size=0.2, random_state=42
        )

        best_result = None
        best_r2_test = -np.inf

        for deg in range(1, max_degree + 1):
            poly = PolynomialFeatures(degree=deg, include_bias=False)
            X_train_poly = poly.fit_transform(X_train)
            X_test_poly = poly.transform(X_test)

            lr = LinearRegression()
            lr.fit(X_train_poly, y_train)

            r2_train = lr.score(X_train_poly, y_train)
            r2_test = lr.score(X_test_poly, y_test)

            if r2_test > best_r2_test:
                best_r2_test = r2_test
                feature_names = poly.get_feature_names_out(X.columns)
                # Build expression string
                terms = []
                if abs(lr.intercept_) > 1e-6:
                    terms.append(f"{lr.intercept_:.4g}")
                for coef, fname in zip(lr.coef_, feature_names):
                    if abs(coef) > 1e-6:
                        terms.append(f"{coef:.4g}*{fname}")
                expr = " + ".join(terms) if terms else "0"

                best_result = {
                    "degree": deg,
                    "r2_train": r2_train,
                    "r2_test": r2_test,
                    "coefficients": lr.coef_,
                    "intercept": lr.intercept_,
                    "expression": expr,
                }

        return best_result

    def fit_pysr(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        **pysr_kwargs,
    ) -> Dict[str, Any]:
        """Fit symbolic regression using PySR.

        Parameters
        ----------
        X : DataFrame
            Input features.
        y : array
            Target values.
        **pysr_kwargs
            Additional keyword arguments for ``PySRRegressor``.

        Returns
        -------
        dict
            ``{"expression", "r2", "complexity", "all_equations"}``.

        Raises
        ------
        ImportError
            If PySR is not installed.
        """
        try:
            from pysr import PySRRegressor
        except ImportError:
            raise ImportError(
                "PySR is required for symbolic regression. "
                "Install with: pip install pysr"
            )

        default_kwargs = {
            "niterations": 40,
            "binary_operators": ["+", "-", "*", "/"],
            "unary_operators": ["exp", "log", "sqrt", "abs"],
            "maxsize": 30,
            "populations": 15,
            "procs": 1,
            "verbosity": 0,
            "temp_equation_file": True,
        }
        default_kwargs.update(pysr_kwargs)

        model = PySRRegressor(**default_kwargs)
        model.fit(X.values, y, variable_names=list(X.columns))

        # Best equation
        best_eq = model.get_best()
        r2 = float(1.0 - best_eq["loss"])  # PySR loss is MSE-based

        # All equations (Pareto front)
        equations_df = model.equations_
        if equations_df is not None:
            all_eqs = equations_df[["equation", "loss", "complexity"]].to_dict("records")
        else:
            all_eqs = []

        return {
            "expression": str(best_eq["equation"]),
            "r2": r2,
            "complexity": int(best_eq["complexity"]),
            "all_equations": all_eqs,
        }

    # ------------------------------------------------------------------
    # Analysis pipeline
    # ------------------------------------------------------------------

    def analyze_process(
        self,
        process_name: str,
        n_samples: int = 10000,
        seed: int = 42,
        use_pysr: bool = False,
        max_poly_degree: int = 4,
        **pysr_kwargs,
    ) -> Dict[str, Any]:
        """Run symbolic regression analysis for a single process.

        Parameters
        ----------
        process_name : str
            Process to analyze.
        n_samples : int
            Number of samples.
        seed : int
            Random seed.
        use_pysr : bool
            Whether to try PySR (falls back to polynomial if unavailable).
        max_poly_degree : int
            Maximum polynomial degree for baseline.
        **pysr_kwargs
            PySR options.

        Returns
        -------
        dict
            ``{"process", "ground_truth_expr", "polynomial", "pysr"}``.
        """
        X, Y = self.generate_io_pairs(process_name, n_samples, seed, noise_free=True)
        gt_exprs = self.get_ground_truth_expression(process_name)

        result = {
            "process": process_name,
            "ground_truth_expr": gt_exprs,
            "n_inputs": X.shape[1],
            "n_outputs": Y.shape[1],
            "input_vars": list(X.columns),
            "output_vars": list(Y.columns),
        }

        # Run for each output
        for target in Y.columns:
            y = Y[target].values

            # Polynomial baseline
            poly_result = self.fit_polynomial(X, y, max_degree=max_poly_degree)
            result[f"{target}_polynomial"] = poly_result

            # PySR (optional)
            if use_pysr:
                try:
                    pysr_result = self.fit_pysr(X, y, **pysr_kwargs)
                    result[f"{target}_pysr"] = pysr_result
                except ImportError:
                    result[f"{target}_pysr"] = {
                        "error": "PySR not available",
                        "expression": "N/A",
                    }

        return result

    def run_full_analysis(
        self,
        n_samples: int = 10000,
        seed: int = 42,
        use_pysr: bool = False,
        max_poly_degree: int = 4,
    ) -> pd.DataFrame:
        """Run symbolic regression on all processes.

        Returns
        -------
        pd.DataFrame
            Summary table with one row per (process, output) pair.
        """
        rows = []
        for proc in self.process_order:
            if proc not in self.datasets:
                continue
            result = self.analyze_process(
                proc, n_samples, seed, use_pysr, max_poly_degree
            )
            ds = self.datasets[proc]
            for target in ds.target_labels:
                row = {
                    "process": proc,
                    "output_var": target,
                    "ground_truth_expr": result["ground_truth_expr"].get(target, "N/A"),
                    "input_vars": ", ".join(result["input_vars"]),
                }

                poly = result.get(f"{target}_polynomial")
                if poly:
                    row["poly_degree"] = poly["degree"]
                    row["poly_r2_train"] = poly["r2_train"]
                    row["poly_r2_test"] = poly["r2_test"]
                    row["poly_expression"] = poly["expression"]

                pysr = result.get(f"{target}_pysr")
                if pysr and "error" not in pysr:
                    row["pysr_expression"] = pysr["expression"]
                    row["pysr_r2"] = pysr["r2"]
                    row["pysr_complexity"] = pysr["complexity"]
                elif pysr:
                    row["pysr_expression"] = pysr.get("expression", "N/A")
                    row["pysr_r2"] = None

                rows.append(row)

        return pd.DataFrame(rows)
