"""
Symbolic regression analysis for SCM equation recovery.

For each process in the SCM, generates (input, output) pairs and tests
whether symbolic regression methods can rediscover the known structural
equations.  Compares the discovered equation against ``NodeSpec.expr``.

Inspired by ``symbolic_regression.ipynb`` from causal-chamber-paper.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scm_ds.scm import SCMDataset, NodeSpec


# ---------------------------------------------------------------------------
# Equation extraction from SCM specs
# ---------------------------------------------------------------------------

def get_structural_equations(
    dataset: SCMDataset,
) -> Dict[str, Dict]:
    """Extract the structural equations from an SCMDataset.

    Returns a dict mapping each node to its parents and expression string.

    Parameters
    ----------
    dataset : SCMDataset
        Process dataset.

    Returns
    -------
    dict
        ``{node_name: {"parents": [...], "expr": "...", "is_output": bool}}``
    """
    equations = {}
    for name, spec in dataset.scm.specs.items():
        equations[name] = {
            "parents": spec.parents,
            "expr": spec.expr,
            "is_output": name in dataset.target_labels,
        }
    return equations


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_regression_data(
    dataset: SCMDataset,
    target_node: str,
    n_samples: int = 5000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Generate (inputs, output) data for symbolic regression of a single node.

    Parameters
    ----------
    dataset : SCMDataset
        Process dataset.
    target_node : str
        Node whose structural equation we want to recover.
    n_samples : int
        Number of samples.
    seed : int
        Random seed.

    Returns
    -------
    X : ndarray
        Input features, shape ``(n_samples, n_parents)``.
    y : ndarray
        Target values, shape ``(n_samples,)``.
    feature_names : list[str]
        Names of the parent variables (columns of X).
    """
    df = dataset.sample(n_samples, seed=seed)
    spec = dataset.scm.specs[target_node]

    feature_names = spec.parents
    X = df[feature_names].values
    y = df[target_node].values

    return X, y, feature_names


# ---------------------------------------------------------------------------
# Symbolic regression backends
# ---------------------------------------------------------------------------

def _run_pysr(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    max_complexity: int = 20,
    niterations: int = 40,
    populations: int = 15,
) -> Dict:
    """Run PySR symbolic regression.

    Requires the ``pysr`` package.

    Returns
    -------
    dict
        ``{"equation": str, "r2": float, "complexity": int, "all_equations": DataFrame}``
    """
    try:
        from pysr import PySRRegressor
    except ImportError:
        raise ImportError(
            "PySR is required for symbolic regression. "
            "Install with: pip install pysr"
        )

    model = PySRRegressor(
        niterations=niterations,
        populations=populations,
        maxsize=max_complexity,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "sqrt", "abs", "sin"],
        extra_sympy_mappings={},
        variable_names=feature_names,
        verbosity=0,
        progress=False,
    )
    model.fit(X, y)

    best = model.get_best()
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "equation": str(best["equation"]),
        "r2": r2,
        "complexity": int(best["complexity"]),
        "all_equations": model.equations_,
    }


def _run_polynomial_fit(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    max_degree: int = 3,
) -> Dict:
    """Polynomial regression as a simple fallback for symbolic regression.

    Fits polynomial features up to ``max_degree`` and reports R^2.

    Returns
    -------
    dict
        ``{"equation": str, "r2": float, "complexity": int, "coefficients": dict}``
    """
    from itertools import combinations_with_replacement

    n_features = X.shape[1]

    # Build polynomial features manually
    poly_features = []
    poly_names = []

    # Degree 0 (intercept)
    poly_features.append(np.ones(X.shape[0]))
    poly_names.append("1")

    # Degree 1 to max_degree
    for deg in range(1, max_degree + 1):
        for combo in combinations_with_replacement(range(n_features), deg):
            feat = np.ones(X.shape[0])
            name_parts = []
            for idx in combo:
                feat *= X[:, idx]
                name_parts.append(feature_names[idx])
            poly_features.append(feat)
            poly_names.append("*".join(name_parts))

    X_poly = np.column_stack(poly_features)

    # Least squares fit
    coeffs, residuals, rank, sv = np.linalg.lstsq(X_poly, y, rcond=None)

    y_pred = X_poly @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Build equation string from significant terms
    terms = []
    for coeff, name in zip(coeffs, poly_names):
        if abs(coeff) > 1e-6:
            terms.append(f"{coeff:.4f}*{name}")
    equation = " + ".join(terms) if terms else "0"

    # Non-zero coefficients
    coeff_dict = {
        name: float(coeff)
        for coeff, name in zip(coeffs, poly_names)
        if abs(coeff) > 1e-6
    }

    return {
        "equation": equation,
        "r2": r2,
        "complexity": len(coeff_dict),
        "coefficients": coeff_dict,
    }


# ---------------------------------------------------------------------------
# Equation comparison
# ---------------------------------------------------------------------------

def compare_equations(
    true_expr: str,
    discovered_expr: str,
    X: np.ndarray,
    y_true: np.ndarray,
    feature_names: List[str],
) -> Dict[str, float]:
    """Compare a discovered equation against the true structural equation.

    Evaluates the discovered equation on the same data and computes
    fit quality metrics.

    Parameters
    ----------
    true_expr : str
        Original SymPy expression string from NodeSpec.
    discovered_expr : str
        Equation found by symbolic regression.
    X : ndarray
        Input features.
    y_true : ndarray
        True output values.
    feature_names : list[str]
        Variable names for X columns.

    Returns
    -------
    dict
        ``{"r2_discovered": float, "rmse_discovered": float,
           "true_expr": str, "discovered_expr": str}``
    """
    result = {
        "true_expr": true_expr,
        "discovered_expr": discovered_expr,
    }

    # Try to evaluate discovered equation
    try:
        import sympy as sp

        # Build symbol mapping
        symbols = {name: sp.symbols(name) for name in feature_names}
        expr = sp.sympify(discovered_expr, locals=symbols)
        fn = sp.lambdify(list(symbols.values()), expr, "numpy")

        y_disc = fn(*[X[:, i] for i in range(X.shape[1])])
        y_disc = np.asarray(y_disc, dtype=float).flatten()

        ss_res = np.sum((y_true - y_disc) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        result["r2_discovered"] = r2
        result["rmse_discovered"] = float(np.sqrt(np.mean((y_true - y_disc) ** 2)))
    except Exception as e:
        result["r2_discovered"] = np.nan
        result["rmse_discovered"] = np.nan
        result["eval_error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Full symbolic analysis
# ---------------------------------------------------------------------------

def run_symbolic_analysis(
    datasets: Dict[str, SCMDataset],
    process_order: Optional[List[str]] = None,
    n_samples: int = 5000,
    seed: int = 42,
    use_pysr: bool = False,
    poly_degree: int = 3,
) -> pd.DataFrame:
    """Run symbolic regression analysis across all processes.

    For each process, fits the output variable as a function of the
    observable input variables.

    Parameters
    ----------
    datasets : dict
        ``{"laser": ds_laser, ...}``
    process_order : list[str], optional
        Order of processes.
    n_samples, seed : int
        Sampling parameters.
    use_pysr : bool
        If True, use PySR; otherwise use polynomial fit fallback.
    poly_degree : int
        Max polynomial degree (only used if ``use_pysr=False``).

    Returns
    -------
    DataFrame
        One row per process with discovered equation, R^2, and comparison
        to true expression.
    """
    order = process_order or ["laser", "plasma", "galvanic", "microetch"]
    results = []

    for proc in order:
        ds = datasets[proc]
        out_var = ds.target_labels[0]

        # Get the true expression for the output node
        true_expr = ds.scm.specs[out_var].expr
        true_parents = ds.scm.specs[out_var].parents

        # Sample data and use only observable inputs
        df = ds.sample(n_samples, seed=seed)
        feature_names = ds.input_labels
        X = df[feature_names].values
        y = df[out_var].values

        # Run symbolic regression
        try:
            if use_pysr:
                sr_result = _run_pysr(X, y, feature_names)
            else:
                sr_result = _run_polynomial_fit(X, y, feature_names, max_degree=poly_degree)
        except Exception as e:
            sr_result = {
                "equation": f"ERROR: {e}",
                "r2": np.nan,
                "complexity": np.nan,
            }

        # Compare with true equation
        row = {
            "process": proc,
            "output_variable": out_var,
            "true_expression": true_expr,
            "true_parents": ", ".join(true_parents),
            "observable_inputs": ", ".join(feature_names),
            "discovered_equation": sr_result["equation"],
            "r2_fit": sr_result["r2"],
            "complexity": sr_result.get("complexity", np.nan),
            "method": "PySR" if use_pysr else f"Polynomial(deg={poly_degree})",
        }
        results.append(row)

    return pd.DataFrame(results)
