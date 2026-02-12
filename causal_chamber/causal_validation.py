"""
Causal Validation via Interventional Two-Sample Tests.

Reproduces the edge validation methodology from the Causal Chamber paper
(Gamella et al., Nature Machine Intelligence 2025, Appendix V / Tables 5-8):

For each edge (parent -> child) in the ground truth DAG, we:
  1. Generate observational (control) data
  2. Generate interventional (treatment) data with do(parent = value)
  3. Apply a two-sample test (KS) on the child variable
  4. If p < alpha, the edge is validated

Two levels of validation:
  - **Intra-process**: do(input) on a single SCM, test outputs within that process.
  - **Pipeline-level**: do(input) on one process, sample the full trajectory
    through all 4 processes, compute F via ReliabilityFunction, and test whether
    F and downstream outputs are affected.  This validates the inter-process
    edges and the output -> F edges in the ground truth DAG.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict
from pathlib import Path
from scipy import stats

import sys

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from causal_chamber.ground_truth import (
    PROCESS_ORDER, PROCESS_DATASETS, PROCESS_OBSERVABLE_VARS,
    get_ground_truth_edges,
)
from causal_chamber.generate_data import (
    INTERVENTION_SCHEDULE,
    sample_trajectory,
    compute_F,
    trajectory_to_dataframe,
)


def compute_pvalue_matrix(
    n: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict:
    """
    Compute p-value matrices for causal edge validation.

    Two matrices are built:
      1. Per-process matrices (intra-process edges only, each SCM independently)
      2. Pipeline matrix (all outputs + F vs all inputs via joint trajectory)

    Returns
    -------
    dict with keys:
        'pvalue_matrices': dict of process_name -> pd.DataFrame
        'pipeline_pvalue_matrix': pd.DataFrame (all outputs+F x all inputs)
        'validated_edges': list of (parent, child, p_value, edge_type)
        'invalidated_edges': list of (parent, child, p_value, edge_type)
        'summary': dict with counts
    """
    all_validated = []
    all_invalidated = []
    per_process = {}

    gt_edges = get_ground_truth_edges()
    gt_edge_set = set(gt_edges)

    # ====================================================================
    # 1. Intra-process validation (each process independently)
    # ====================================================================
    seed_offset = 0
    for proc_name in PROCESS_ORDER:
        ds = PROCESS_DATASETS[proc_name]
        info = PROCESS_OBSERVABLE_VARS[proc_name]
        schedule = INTERVENTION_SCHEDULE[proc_name]

        obs_vars = info['inputs'] + info['outputs']

        # One intervention per input variable (strongest level)
        intervention_targets = []
        for var_name, interventions in schedule.items():
            intv = interventions[-1]
            intervention_targets.append({
                'variable': var_name,
                'value': intv['value'],
                'label': intv['name'],
            })

        col_labels = [
            f"do({t['variable']}={t['value']})" for t in intervention_targets
        ]

        pval_matrix = pd.DataFrame(
            np.ones((len(obs_vars), len(intervention_targets))),
            index=obs_vars,
            columns=col_labels,
        )

        for j, intv in enumerate(intervention_targets):
            seed_offset += 1
            var = intv['variable']
            val = intv['value']

            df_obs = ds.sample(n, seed=seed + seed_offset * 10)
            scm_int = ds.scm.do({var: val})
            df_int = scm_int.sample(n, seed=seed + seed_offset * 10 + 1)

            for i, target in enumerate(obs_vars):
                if target in df_obs.columns and target in df_int.columns:
                    _, p_val = stats.ks_2samp(
                        df_obs[target].values,
                        df_int[target].values,
                    )
                    pval_matrix.iloc[i, j] = p_val

        per_process[proc_name] = pval_matrix

        # Check ground truth intra-process edges
        for parent, child in gt_edge_set:
            if parent in info['inputs'] and child in info['outputs']:
                col_label = None
                for j, intv in enumerate(intervention_targets):
                    if intv['variable'] == parent:
                        col_label = col_labels[j]
                        break
                if col_label is not None and child in pval_matrix.index:
                    p = pval_matrix.loc[child, col_label]
                    if p < alpha:
                        all_validated.append((parent, child, float(p), 'intra'))
                    else:
                        all_invalidated.append((parent, child, float(p), 'intra'))

    # ====================================================================
    # 2. Pipeline-level validation (joint trajectory + F)
    # ====================================================================
    # Response variables: process outputs + F
    all_outputs = []
    for proc in PROCESS_ORDER:
        all_outputs.extend(PROCESS_OBSERVABLE_VARS[proc]['outputs'])
    all_outputs.append('F')

    # Intervention schedule (strongest per input, prefixed to avoid dupes)
    all_interventions = []  # (proc, var, value, label)
    for proc in PROCESS_ORDER:
        schedule = INTERVENTION_SCHEDULE[proc]
        for var_name, interventions in schedule.items():
            intv = interventions[-1]
            all_interventions.append((proc, var_name, intv['value'], intv['name']))

    col_labels_pipeline = [
        f"do({proc}.{var}={val})" for proc, var, val, _ in all_interventions
    ]

    pipeline_pvals = pd.DataFrame(
        np.ones((len(all_outputs), len(all_interventions))),
        index=all_outputs,
        columns=col_labels_pipeline,
    )

    # Reference (observational) joint trajectory
    traj_obs = sample_trajectory(n, seed=seed)
    df_obs_pipeline = trajectory_to_dataframe(traj_obs)
    F_obs = compute_F(traj_obs).numpy()
    df_obs_pipeline['F'] = F_obs

    for j, (proc, var, val, label) in enumerate(all_interventions):
        traj_int = sample_trajectory(
            n, seed=seed + (j + 1) * 100,
            interventions={proc: {var: val}},
        )
        df_int_pipeline = trajectory_to_dataframe(traj_int)
        F_int = compute_F(traj_int).numpy()
        df_int_pipeline['F'] = F_int

        for i, target in enumerate(all_outputs):
            if target in df_obs_pipeline.columns and target in df_int_pipeline.columns:
                _, p_val = stats.ks_2samp(
                    df_obs_pipeline[target].values,
                    df_int_pipeline[target].values,
                )
                pipeline_pvals.iloc[i, j] = p_val

    # Validate inter-process and -> F edges from pipeline matrix
    for parent, child in gt_edge_set:
        # Skip intra-process (already validated above)
        is_intra = any(
            parent in PROCESS_OBSERVABLE_VARS[proc]['inputs'] and
            child in PROCESS_OBSERVABLE_VARS[proc]['outputs']
            for proc in PROCESS_ORDER
        )
        if is_intra:
            continue

        # Find the best intervention for this parent
        best_p = 1.0

        # Case 1: parent is an input variable (direct do)
        for j, (proc, var, val, _) in enumerate(all_interventions):
            if var == parent:
                col = col_labels_pipeline[j]
                if child in pipeline_pvals.index:
                    best_p = min(best_p, pipeline_pvals.loc[child, col])

        # Case 2: parent is an output variable — intervene on the
        # upstream process's inputs to shift the parent, then check child
        if best_p >= 1.0:
            for proc in PROCESS_ORDER:
                if parent in PROCESS_OBSERVABLE_VARS[proc]['outputs']:
                    for jj, (iproc, ivar, ival, _) in enumerate(all_interventions):
                        if iproc == proc:
                            col = col_labels_pipeline[jj]
                            if child in pipeline_pvals.index:
                                best_p = min(best_p, pipeline_pvals.loc[child, col])

        if best_p < alpha:
            all_validated.append((parent, child, float(best_p), 'pipeline'))
        else:
            all_invalidated.append((parent, child, float(best_p), 'pipeline'))

    n_total_checked = len(all_validated) + len(all_invalidated)
    summary = {
        'n_ground_truth_edges': len(gt_edges),
        'n_validated': len(all_validated),
        'n_invalidated': len(all_invalidated),
        'n_total_checked': n_total_checked,
        'n_intra': sum(1 for *_, t in all_validated if t == 'intra') +
                   sum(1 for *_, t in all_invalidated if t == 'intra'),
        'n_pipeline': sum(1 for *_, t in all_validated if t == 'pipeline') +
                      sum(1 for *_, t in all_invalidated if t == 'pipeline'),
        'validation_rate': len(all_validated) / max(1, n_total_checked),
    }

    return {
        'pvalue_matrices': per_process,
        'pipeline_pvalue_matrix': pipeline_pvals,
        'validated_edges': all_validated,
        'invalidated_edges': all_invalidated,
        'summary': summary,
    }


def format_pvalue(p: float) -> str:
    """Format p-value in LaTeX notation like the paper."""
    if p < 1e-300:
        return '< 10^{-300}'
    elif p < 0.001:
        exp = int(math.floor(math.log10(p)))
        base = p / (10 ** exp)
        return f'{base:.1f} x 10^{{{exp}}}'
    else:
        return f'{p:.3f}'


def print_validation_table(results: Dict):
    """Print a validation table similar to the paper's Tables 5-8."""
    print('\nCausal Edge Validation Summary')
    print('=' * 70)

    summary = results['summary']
    print(f"Ground truth edges:   {summary['n_ground_truth_edges']}")
    print(f"Edges checked:        {summary['n_total_checked']}")
    print(f"  - intra-process:    {summary['n_intra']}")
    print(f"  - pipeline (F):     {summary['n_pipeline']}")
    print(f"Validated:            {summary['n_validated']}")
    print(f"Invalidated:          {summary['n_invalidated']}")
    print(f"Validation rate:      {summary['validation_rate']:.1%}")

    intra_edges = [(p, c, pv) for p, c, pv, t in results['validated_edges'] if t == 'intra']
    pipeline_edges = [(p, c, pv) for p, c, pv, t in results['validated_edges'] if t == 'pipeline']

    if intra_edges:
        print('\nValidated Intra-Process Edges:')
        print('-' * 70)
        for parent, child, p in sorted(intra_edges, key=lambda x: x[2]):
            print(f'  {parent:20s} -> {child:20s}  p = {format_pvalue(p)}')

    if pipeline_edges:
        print('\nValidated Pipeline Edges (inter-process + -> F):')
        print('-' * 70)
        for parent, child, p in sorted(pipeline_edges, key=lambda x: x[2]):
            print(f'  {parent:20s} -> {child:20s}  p = {format_pvalue(p)}')

    if results['invalidated_edges']:
        print('\nInvalidated Edges (p >= 0.05):')
        print('-' * 70)
        for parent, child, p, t in results['invalidated_edges']:
            print(f'  {parent:20s} -> {child:20s}  p = {format_pvalue(p)}  [{t}]')

    print('\nPer-Process P-Value Matrices:')
    print('-' * 70)
    for proc_name, pval_matrix in results['pvalue_matrices'].items():
        print(f'\n  [{proc_name}]')
        for target in pval_matrix.index:
            vals = '  '.join(
                f'{format_pvalue(pval_matrix.loc[target, col]):>16s}'
                for col in pval_matrix.columns
            )
            print(f'    {target:20s}: {vals}')

    # Print pipeline F row
    print('\n  [Pipeline -> F]')
    pm = results['pipeline_pvalue_matrix']
    if 'F' in pm.index:
        for col in pm.columns:
            p = pm.loc['F', col]
            print(f'    {col:40s}: p = {format_pvalue(p)}')
