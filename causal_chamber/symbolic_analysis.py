"""
Symbolic Regression Analysis.

For each process SCM, generates (input, output) data and tests whether
symbolic regression can rediscover the known structural equations.
Uses PySR if available, otherwise falls back to polynomial fitting.

Inspired by symbolic_regression.ipynb from the Causal Chamber paper
(Gamella et al., Nature Machine Intelligence 2025).
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

import sys

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from causal_chamber.ground_truth import (
    PROCESS_ORDER, PROCESS_DATASETS, PROCESS_OBSERVABLE_VARS,
)


# ---------------------------------------------------------------------------
# Known structural equations (ground truth descriptions)
# ---------------------------------------------------------------------------

KNOWN_EQUATIONS = {
    'laser': {
        'output': 'ActualPower',
        'inputs': ['PowerTarget', 'AmbientTemp'],
        'equation_latex': r'$P = \eta_0 (1 - \alpha_T \Delta T)(I - I_0 e^{k_T \Delta T}) \cdot Z_{\ln} + \varepsilon$',
        'equation_description': 'L-I-T model: P = eta0 * (1 - alphaT*dT) * (I - I0*exp(kT*dT)) * Zln + noise',
        'key_features': ['exponential threshold', 'linear efficiency', 'multiplicative noise'],
    },
    'plasma': {
        'output': 'RemovalRate',
        'inputs': ['RF_Power', 'Duration'],
        'equation_latex': r'$R = k_0 e^{-\lambda_p P^\beta} \tau \cdot Z_{\ln} + \varepsilon$',
        'equation_description': 'Removal rate: R = k0 * exp(-lambdaP * P^beta) * tau * Zln + noise',
        'key_features': ['stretched exponential', 'linear time', 'multiplicative noise'],
    },
    'galvanic': {
        'output': 'Thickness',
        'inputs': ['CurrentDensity', 'Duration'],
        'equation_latex': r'$t_{Cu} = \frac{\eta_{dep} j \tau M_{Cu}}{n F \rho_{Cu}} (1 + g) + \varepsilon$',
        'equation_description': 'Faraday law: t = (eta*j*tau*M)/(n*F*rho) * (1+g) + noise',
        'key_features': ['Faraday law (linear in j*tau)', 'spatial variation', 'ripple'],
    },
    'microetch': {
        'output': 'RemovalDepth',
        'inputs': ['Temperature', 'Concentration', 'Duration'],
        'equation_latex': r'$R = k_{etch} e^{-E_a/(RT)} C^\alpha \tau \cdot Z_{\ln} + \varepsilon_t$',
        'equation_description': 'Arrhenius: R = k * exp(-Ea/(R*T)) * C^alpha * tau * Zln + noise',
        'key_features': ['Arrhenius exponential', 'power-law concentration', 'linear time'],
    },
}


# ---------------------------------------------------------------------------
# PySR availability check
# ---------------------------------------------------------------------------

def _check_pysr():
    try:
        from pysr import PySRRegressor
        return True
    except ImportError:
        return False


PYSR_AVAILABLE = _check_pysr()


# ---------------------------------------------------------------------------
# Symbolic regression methods
# ---------------------------------------------------------------------------

def polynomial_fit(
    X: np.ndarray,
    y: np.ndarray,
    degree: int = 3,
) -> Dict:
    """
    Polynomial regression as a baseline/fallback.

    Parameters
    ----------
    X : np.ndarray
        Input features (n_samples, n_features).
    y : np.ndarray
        Target values (n_samples,).
    degree : int
        Maximum polynomial degree.

    Returns
    -------
    dict with 'model', 'r2', 'equation_str', 'predictions'.
    """
    model = make_pipeline(
        PolynomialFeatures(degree, include_bias=True),
        LinearRegression(),
    )
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = float(r2_score(y, y_pred))

    # Build equation string
    poly = model.named_steps['polynomialfeatures']
    lr = model.named_steps['linearregression']
    feature_names = poly.get_feature_names_out()
    coefs = lr.coef_
    intercept = lr.intercept_

    terms = []
    if abs(intercept) > 1e-6:
        terms.append(f"{intercept:.4f}")
    for fname, coef in zip(feature_names, coefs):
        if abs(coef) > 1e-6:
            terms.append(f"{coef:.4f}*{fname}")

    equation_str = ' + '.join(terms) if terms else '0'

    return {
        'model': model,
        'r2': r2,
        'equation_str': equation_str,
        'predictions': y_pred,
        'method': f'polynomial(degree={degree})',
    }


def pysr_fit(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    niterations: int = 40,
    maxsize: int = 25,
) -> Optional[Dict]:
    """
    Symbolic regression using PySR.

    Parameters
    ----------
    X : np.ndarray
        Input features.
    y : np.ndarray
        Target values.
    feature_names : list of str
        Input variable names.
    niterations : int
        Number of PySR iterations.
    maxsize : int
        Maximum expression size.

    Returns
    -------
    dict or None if PySR not available.
    """
    if not PYSR_AVAILABLE:
        return None

    from pysr import PySRRegressor

    model = PySRRegressor(
        niterations=niterations,
        binary_operators=['+', '-', '*', '/', 'pow'],
        unary_operators=['exp', 'log', 'sqrt', 'abs'],
        maxsize=maxsize,
        populations=15,
        population_size=33,
        progress=False,
        verbosity=0,
        temp_equation_file=True,
    )

    try:
        model.fit(X, y, variable_names=feature_names)
        y_pred = model.predict(X)
        r2 = float(r2_score(y, y_pred))

        best_eq = str(model.sympy())
        latex_eq = model.latex()

        return {
            'model': model,
            'r2': r2,
            'equation_str': best_eq,
            'equation_latex': latex_eq,
            'predictions': y_pred,
            'method': 'PySR',
        }
    except Exception as e:
        warnings.warn(f"PySR fitting failed: {e}")
        return None


def run_symbolic_analysis_single(
    process_name: str,
    n: int = 5000,
    seed: int = 42,
    use_pysr: bool = True,
    poly_degree: int = 3,
) -> Dict:
    """
    Run symbolic regression for a single process.

    Generates data from the SCM and attempts to rediscover the structural
    equation using symbolic regression.

    Returns
    -------
    dict with analysis results.
    """
    ds = PROCESS_DATASETS[process_name]
    info = PROCESS_OBSERVABLE_VARS[process_name]
    known = KNOWN_EQUATIONS[process_name]

    # Sample data
    df = ds.sample(n, seed=seed)

    # Extract inputs and output
    input_cols = info['inputs']
    output_col = info['outputs'][0]

    X = df[input_cols].values
    y = df[output_col].values

    results = {
        'process': process_name,
        'input_vars': input_cols,
        'output_var': output_col,
        'known_equation': known,
        'n_samples': n,
        'X': X,
        'y_true': y,
        'fits': {},
    }

    # Polynomial fit (always available)
    poly_result = polynomial_fit(X, y, degree=poly_degree)
    results['fits']['polynomial'] = poly_result

    # PySR fit (optional)
    if use_pysr and PYSR_AVAILABLE:
        pysr_result = pysr_fit(X, y, feature_names=input_cols)
        if pysr_result is not None:
            results['fits']['pysr'] = pysr_result
    elif use_pysr and not PYSR_AVAILABLE:
        warnings.warn(
            "PySR not installed. Using polynomial fit only. "
            "Install with: pip install pysr"
        )

    # Best fit
    best_method = max(results['fits'].keys(), key=lambda k: results['fits'][k]['r2'])
    results['best_fit'] = results['fits'][best_method]
    results['best_method'] = best_method

    return results


def run_symbolic_analysis(
    n: int = 5000,
    seed: int = 42,
    use_pysr: bool = True,
    poly_degree: int = 3,
) -> Dict:
    """
    Run symbolic regression analysis for all processes.

    Returns
    -------
    dict with keys:
        'per_process': dict of process_name -> analysis results
        'summary_table': pd.DataFrame with summary
    """
    per_process = {}
    summary_rows = []

    for proc_name in PROCESS_ORDER:
        result = run_symbolic_analysis_single(
            proc_name, n=n, seed=seed, use_pysr=use_pysr,
            poly_degree=poly_degree,
        )
        per_process[proc_name] = result

        known = KNOWN_EQUATIONS[proc_name]
        row = {
            'process': proc_name,
            'output': known['output'],
            'true_equation': known['equation_description'],
            'best_method': result['best_method'],
            'best_r2': result['best_fit']['r2'],
            'discovered_equation': result['best_fit']['equation_str'][:80],
        }
        # Add polynomial R² always
        row['poly_r2'] = result['fits']['polynomial']['r2']
        if 'pysr' in result['fits']:
            row['pysr_r2'] = result['fits']['pysr']['r2']

        summary_rows.append(row)

    summary_table = pd.DataFrame(summary_rows)

    return {
        'per_process': per_process,
        'summary_table': summary_table,
    }
