"""
Out-of-Distribution (OOD) Analysis.

Reproduces the OOD generalization methodology from the Causal Chamber paper
(Gamella et al., Nature Machine Intelligence 2025, ood_sensors.ipynb):

1. Train regression models on observational (in-distribution) data
2. Evaluate prediction MAE on multiple OOD environments
3. Compare model robustness across distribution shifts
4. Radar/spider plots for multi-environment comparison

All analyses use joint pipeline trajectories (inputs -> outputs -> F).
Multiple random seeds are used for variance estimation.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from sklearn.metrics import mean_absolute_error

import sys

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from causal_chamber.ground_truth import (
    PROCESS_ORDER, PROCESS_DATASETS, PROCESS_OBSERVABLE_VARS,
)
from causal_chamber.generate_data import (
    sample_joint_pipeline, sample_joint_ood_pipeline,
)


# ---------------------------------------------------------------------------
# OOD shift definitions (environments)
# ---------------------------------------------------------------------------

OOD_SHIFTS = {
    'laser': {
        'variable': 'AmbientTemp',
        'id_range': (15.0, 35.0),
        'ood_range': (35.0, 45.0),
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


# ---------------------------------------------------------------------------
# Predictor-set definitions (analogous to the paper's {R}, {R,G}, ... sets)
# ---------------------------------------------------------------------------

def _get_predictor_sets(process_name: str) -> Dict[str, List[str]]:
    """
    Define multiple predictor sets for a process's output,
    analogous to the paper's {R}, {R,G}, {R,G,B}, ... sets.
    """
    info = PROCESS_OBSERVABLE_VARS[process_name]
    inputs = info['inputs']

    models = {'mean': []}  # baseline: predict mean
    for i in range(1, len(inputs) + 1):
        name = '+'.join(inputs[:i])
        models[name] = inputs[:i]

    return models


def _get_F_predictor_sets() -> Dict[str, List[str]]:
    """
    Define predictor sets for F, using increasing subsets of process outputs.
    """
    outputs = []
    for proc in PROCESS_ORDER:
        outputs.extend(PROCESS_OBSERVABLE_VARS[proc]['outputs'])

    models = {'mean': []}
    for i in range(1, len(outputs) + 1):
        name = '+'.join(outputs[:i])
        models[name] = outputs[:i]

    return models


# ---------------------------------------------------------------------------
# OLS regression (following the paper's statsmodels.OLS approach)
# ---------------------------------------------------------------------------

def _fit_ols(X_train, y_train):
    """Fit OLS regression (with intercept), return fitted model."""
    import statsmodels.api as sm
    X = sm.add_constant(X_train)
    model = sm.OLS(y_train, X).fit()
    return model


def _predict_ols(model, X_test):
    """Predict using OLS model."""
    import statsmodels.api as sm
    X = sm.add_constant(X_test)
    return model.predict(X)


def _predict_mean(y_train, n_test):
    """Baseline: predict the training mean for all test samples."""
    return np.full(n_test, y_train.mean())


# ---------------------------------------------------------------------------
# Core OOD analysis
# ---------------------------------------------------------------------------

def run_ood_analysis(
    n_train: int = 100,
    n_test: int = 1000,
    seed: int = 42,
    n_seeds: int = 8,
) -> Dict:
    """
    Run OOD analysis following the Causal Chamber paper methodology.

    For each process (and for F):
    1. Sample ID training data (small, like the paper's 100 samples)
    2. Sample ID validation + OOD test environments
    3. Train OLS models with different predictor sets
    4. Evaluate MAE across all environments
    5. Repeat for n_seeds and report mean +/- std

    Parameters
    ----------
    n_train : int
        Training samples (default 100, matching the paper).
    n_test : int
        Test samples per environment.
    seed : int
        Base random seed.
    n_seeds : int
        Number of random seeds for variance estimation.

    Returns
    -------
    dict with keys:
        'per_process': dict of process -> OOD results
        'reliability': F-level OOD results
        'summary_table': pd.DataFrame
    """
    env_names = ['ID'] + [f'OOD_{p}' for p in PROCESS_ORDER]
    per_process = {}
    summary_rows = []

    for proc_name in PROCESS_ORDER:
        info = PROCESS_OBSERVABLE_VARS[proc_name]
        output_var = info['outputs'][0]
        predictor_sets = _get_predictor_sets(proc_name)

        # Accumulate MAE per (model, env, seed)
        all_env_results = {m: {e: [] for e in env_names} for m in predictor_sets}

        for s in range(n_seeds):
            seed_s = seed + s * 1000

            train_df = sample_joint_pipeline(n=n_train, seed=seed_s)
            id_val_df = sample_joint_pipeline(n=n_test, seed=seed_s + 1)

            envs = {'ID': id_val_df}
            for ood_proc in PROCESS_ORDER:
                shift = OOD_SHIFTS[ood_proc]
                envs[f'OOD_{ood_proc}'] = sample_joint_ood_pipeline(
                    n=n_test, seed=seed_s + 2,
                    ood_process=ood_proc,
                    ood_variable=shift['variable'],
                    ood_range=shift['ood_range'],
                )

            y_train = train_df[output_var].values

            for model_name, predictors in predictor_sets.items():
                for env_name, env_df in envs.items():
                    y_test = env_df[output_var].values

                    if len(predictors) == 0:
                        y_pred = _predict_mean(y_train, len(y_test))
                    else:
                        model = _fit_ols(train_df[predictors].values, y_train)
                        y_pred = _predict_ols(model, env_df[predictors].values)

                    mae = float(mean_absolute_error(y_test, y_pred))
                    all_env_results[model_name][env_name].append(mae)

        # Aggregate across seeds: mean +/- std
        model_results = {}
        for model_name, env_dict in all_env_results.items():
            model_results[model_name] = {}
            for env_name, mae_list in env_dict.items():
                model_results[model_name][env_name] = {
                    'mean': float(np.mean(mae_list)),
                    'std': float(np.std(mae_list)),
                }

        per_process[proc_name] = {
            'output_var': output_var,
            'predictor_sets': predictor_sets,
            'model_results': model_results,
            'env_names': env_names,
            'n_seeds': n_seeds,
        }

        # Summary row for best non-baseline model
        best_model = None
        best_id_mae = float('inf')
        for model_name, env_dict in model_results.items():
            if model_name == 'mean':
                continue
            id_mae = env_dict.get('ID', {}).get('mean', float('inf'))
            if id_mae < best_id_mae:
                best_id_mae = id_mae
                best_model = model_name

        if best_model:
            row = {'process': proc_name, 'output': output_var, 'best_model': best_model}
            for env_name in env_names:
                row[f'{env_name}_MAE'] = model_results[best_model][env_name]['mean']
                row[f'{env_name}_std'] = model_results[best_model][env_name]['std']
            summary_rows.append(row)

    # ===================== F-level OOD analysis =====================
    F_predictor_sets = _get_F_predictor_sets()
    F_env_results = {m: {e: [] for e in env_names} for m in F_predictor_sets}

    for s in range(n_seeds):
        seed_s = seed + s * 1000
        train_df = sample_joint_pipeline(n=n_train, seed=seed_s)
        id_val_df = sample_joint_pipeline(n=n_test, seed=seed_s + 1)

        envs = {'ID': id_val_df}
        for ood_proc in PROCESS_ORDER:
            shift = OOD_SHIFTS[ood_proc]
            envs[f'OOD_{ood_proc}'] = sample_joint_ood_pipeline(
                n=n_test, seed=seed_s + 2,
                ood_process=ood_proc,
                ood_variable=shift['variable'],
                ood_range=shift['ood_range'],
            )

        y_train = train_df['F'].values
        for model_name, predictors in F_predictor_sets.items():
            for env_name, env_df in envs.items():
                y_test = env_df['F'].values
                if len(predictors) == 0:
                    y_pred = _predict_mean(y_train, len(y_test))
                else:
                    model = _fit_ols(train_df[predictors].values, y_train)
                    y_pred = _predict_ols(model, env_df[predictors].values)
                mae = float(mean_absolute_error(y_test, y_pred))
                F_env_results[model_name][env_name].append(mae)

    F_model_results = {}
    for model_name, env_dict in F_env_results.items():
        F_model_results[model_name] = {}
        for env_name, mae_list in env_dict.items():
            F_model_results[model_name][env_name] = {
                'mean': float(np.mean(mae_list)),
                'std': float(np.std(mae_list)),
            }

    reliability_results = {
        'output_var': 'F',
        'predictor_sets': F_predictor_sets,
        'model_results': F_model_results,
        'env_names': env_names,
        'n_seeds': n_seeds,
    }

    summary_table = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()

    return {
        'per_process': per_process,
        'reliability': reliability_results,
        'summary_table': summary_table,
    }


# ---------------------------------------------------------------------------
# Utility functions kept for backwards compatibility
# ---------------------------------------------------------------------------

def compute_prediction_metrics(y_true, y_pred):
    """Compute R², MAE for predictions."""
    from sklearn.metrics import r2_score
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if len(y_true) == 0 or np.std(y_true) < 1e-10:
        return {'r2': float('nan'), 'mae': float('nan')}
    return {
        'r2': float(r2_score(y_true, y_pred)),
        'mae': float(mean_absolute_error(y_true, y_pred)),
    }


def analyze_attention_stability(id_attention, ood_attention):
    """Compare attention matrices between ID and OOD conditions."""
    if id_attention is None or ood_attention is None:
        return {'frobenius_diff': float('nan'), 'max_diff': float('nan'), 'correlation': float('nan')}
    diff = id_attention - ood_attention
    frob = float(np.linalg.norm(diff, 'fro'))
    max_diff = float(np.abs(diff).max())
    id_flat = id_attention.ravel()
    ood_flat = ood_attention.ravel()
    if np.std(id_flat) > 1e-10 and np.std(ood_flat) > 1e-10:
        corr = float(np.corrcoef(id_flat, ood_flat)[0, 1])
    else:
        corr = float('nan')
    return {'frobenius_diff': frob, 'max_diff': max_diff, 'correlation': corr}
