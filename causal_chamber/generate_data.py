"""
Data Generation for Causal Chamber Analysis.

Generates observational and interventional datasets from all 4 SCM processes,
following the pattern of the Causal Chamber paper (Gamella et al., Nature
Machine Intelligence 2025).

Key addition over per-process sampling: **joint pipeline trajectories** that
assemble the same trajectory dict structure used by ProcessChain.forward()
elsewhere in AZIMUTH, then compute reliability F via ReliabilityFunction.

Trajectory dict structure (matches controller_optimization/process_chain.py):
    {
        'laser': {
            'inputs':          tensor (batch, input_dim),
            'outputs_mean':    tensor (batch, output_dim),
            'outputs_var':     tensor (batch, output_dim),
            'outputs_sampled': tensor (batch, output_dim),
        },
        'plasma': { ... },
        'galvanic': { ... },
        'microetch': { ... },
    }

Since the SCMs produce point samples (no analytical variance), we estimate
outputs_var empirically from the batch and set outputs_sampled = outputs_mean
(the SCM sample already includes stochastic noise).

Usage:
    python -m causal_chamber.generate_data --output_dir data/causal_chamber --n_obs 5000 --n_int 1000
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from causal_chamber.ground_truth import (
    PROCESS_ORDER, PROCESS_DATASETS, PROCESS_OBSERVABLE_VARS,
    get_ground_truth_edges, get_all_observable_vars,
    get_ground_truth_adjacency_parent_convention,
)
from reliability_function import ReliabilityFunction


# ---------------------------------------------------------------------------
# Intervention schedule per process
# ---------------------------------------------------------------------------

INTERVENTION_SCHEDULE = {
    'laser': {
        'PowerTarget': [
            {'name': 'low_power', 'value': 0.20},
            {'name': 'mid_power', 'value': 0.55},
            {'name': 'high_power', 'value': 0.90},
        ],
        'AmbientTemp': [
            {'name': 'cold', 'value': 18.0},
            {'name': 'warm', 'value': 30.0},
            {'name': 'hot', 'value': 34.0},
        ],
    },
    'plasma': {
        'RF_Power': [
            {'name': 'low_rf', 'value': 120.0},
            {'name': 'mid_rf', 'value': 250.0},
            {'name': 'high_rf', 'value': 380.0},
        ],
        'Duration': [
            {'name': 'short', 'value': 15.0},
            {'name': 'medium', 'value': 35.0},
            {'name': 'long', 'value': 55.0},
        ],
    },
    'galvanic': {
        'CurrentDensity': [
            {'name': 'low_current', 'value': 1.5},
            {'name': 'mid_current', 'value': 3.0},
            {'name': 'high_current', 'value': 4.5},
        ],
        'Duration': [
            {'name': 'short', 'value': 800.0},
            {'name': 'medium', 'value': 2100.0},
            {'name': 'long', 'value': 3400.0},
        ],
    },
    'microetch': {
        'Temperature': [
            {'name': 'cold', 'value': 296.0},
            {'name': 'warm', 'value': 308.0},
            {'name': 'hot', 'value': 320.0},
        ],
        'Concentration': [
            {'name': 'dilute', 'value': 0.8},
            {'name': 'medium', 'value': 1.75},
            {'name': 'concentrated', 'value': 2.8},
        ],
        'Duration': [
            {'name': 'short', 'value': 40.0},
            {'name': 'medium', 'value': 105.0},
            {'name': 'long', 'value': 170.0},
        ],
    },
}


# ---------------------------------------------------------------------------
# Joint pipeline trajectory sampling
# ---------------------------------------------------------------------------

def sample_trajectory(
    n: int,
    seed: int,
    interventions: dict = None,
) -> dict:
    """
    Sample a joint pipeline trajectory through all 4 processes.

    Produces the same dict structure as ProcessChain.forward() so that
    ReliabilityFunction.compute_reliability() can consume it directly.

    Since SCMs return point samples (no analytical variance), we:
      - Set outputs_mean = outputs_sampled = SCM sample (already noisy)
      - Estimate outputs_var empirically as batch variance

    Parameters
    ----------
    n : int
        Number of samples (batch size).
    seed : int
        Random seed.
    interventions : dict, optional
        {process_name: {variable: value}} for do-operator interventions.
        Only specified processes are intervened; others sample normally.

    Returns
    -------
    dict matching ProcessChain.forward() output structure.
    """
    interventions = interventions or {}
    trajectory = {}

    for proc_name in PROCESS_ORDER:
        ds = PROCESS_DATASETS[proc_name]
        info = PROCESS_OBSERVABLE_VARS[proc_name]

        # Sample from SCM (possibly intervened)
        if proc_name in interventions:
            scm_int = ds.scm.do(interventions[proc_name])
            df = scm_int.sample(n, seed=seed)
        else:
            df = ds.sample(n, seed=seed)

        # Extract inputs and outputs as tensors
        inputs_np = df[info['inputs']].values.astype(np.float32)
        outputs_np = df[info['outputs']].values.astype(np.float32)

        inputs_t = torch.tensor(inputs_np)
        outputs_t = torch.tensor(outputs_np)

        # Empirical variance across the batch (per output dimension)
        outputs_var = torch.var(outputs_t, dim=0, keepdim=True).expand_as(outputs_t)

        trajectory[proc_name] = {
            'inputs': inputs_t,
            'outputs_mean': outputs_t,
            'outputs_var': outputs_var,
            'outputs_sampled': outputs_t,
        }

    return trajectory


def trajectory_to_dataframe(trajectory: dict) -> pd.DataFrame:
    """
    Flatten a trajectory dict into a DataFrame with all observable columns.
    """
    data = {}
    for proc_name in PROCESS_ORDER:
        info = PROCESS_OBSERVABLE_VARS[proc_name]
        t = trajectory[proc_name]
        inputs = t['inputs'].numpy()
        outputs = t['outputs_sampled'].numpy()
        for i, col in enumerate(info['inputs']):
            data[col] = inputs[:, i]
        for i, col in enumerate(info['outputs']):
            data[col] = outputs[:, i]
    return pd.DataFrame(data)


def compute_F(trajectory: dict, return_quality_scores: bool = False):
    """
    Compute reliability F from a trajectory dict via ReliabilityFunction.

    This is the same computation used during controller training.
    """
    rf = ReliabilityFunction()
    with torch.no_grad():
        result = rf.compute_reliability(
            trajectory, return_quality_scores=return_quality_scores,
        )
    return result


def sample_joint_pipeline(
    n: int,
    seed: int,
    interventions: dict = None,
) -> pd.DataFrame:
    """
    Sample a joint pipeline: all 4 processes + reliability F.

    Convenience wrapper that returns a flat DataFrame.

    Parameters
    ----------
    n : int
        Number of samples.
    seed : int
        Random seed.
    interventions : dict, optional
        {process_name: {variable: value}} for do-operator interventions.

    Returns
    -------
    pd.DataFrame with all observable columns + 'F'.
    """
    trajectory = sample_trajectory(n, seed, interventions)
    df = trajectory_to_dataframe(trajectory)

    F = compute_F(trajectory)
    df['F'] = F.numpy()

    return df


def sample_joint_ood_pipeline(
    n: int,
    seed: int,
    ood_process: str,
    ood_variable: str,
    ood_range: tuple,
) -> pd.DataFrame:
    """
    Sample a joint pipeline with one process under OOD conditions.

    All processes except *ood_process* are sampled from their standard
    noise model.  The OOD process uses a shifted noise model for the
    specified variable.  Reliability F is computed on the full trajectory.

    Parameters
    ----------
    n : int
        Number of samples.
    seed : int
        Random seed.
    ood_process : str
        Process to shift (e.g. 'laser').
    ood_variable : str
        Variable whose noise range is shifted (e.g. 'AmbientTemp').
    ood_range : tuple of (float, float)
        New (low, high) range for the OOD variable.

    Returns
    -------
    pd.DataFrame with all observable columns + 'F'.
    """
    from scm_ds.scm import SCM, NoiseModel

    trajectory = {}
    for proc_name in PROCESS_ORDER:
        ds = PROCESS_DATASETS[proc_name]
        info = PROCESS_OBSERVABLE_VARS[proc_name]

        if proc_name == ood_process:
            # Build shifted noise model
            original = ds.noise_model
            new_singles = dict(original.singles)
            lo, hi = ood_range
            new_singles[ood_variable] = (
                lambda rng, n, lo=lo, hi=hi: rng.uniform(lo, hi, size=n)
            )
            ood_noise_model = NoiseModel(
                singles=new_singles, groups=original.groups,
            )
            ood_scm = SCM(
                list(ds.scm.specs.values()),
                noise_model=ood_noise_model,
            )
            rng = np.random.default_rng(seed + 1)
            eps_draws = ood_noise_model.sample_all(rng, n)
            ctx = {}
            ood_scm.forward(ctx, eps_draws)
            df = pd.DataFrame(
                {k: np.asarray(v).reshape(n) for k, v in ctx.items()},
            )
        else:
            df = ds.sample(n, seed=seed)

        inputs_np = df[info['inputs']].values.astype(np.float32)
        outputs_np = df[info['outputs']].values.astype(np.float32)

        inputs_t = torch.tensor(inputs_np)
        outputs_t = torch.tensor(outputs_np)
        outputs_var = torch.var(outputs_t, dim=0, keepdim=True).expand_as(outputs_t)

        trajectory[proc_name] = {
            'inputs': inputs_t,
            'outputs_mean': outputs_t,
            'outputs_var': outputs_var,
            'outputs_sampled': outputs_t,
        }

    df_out = trajectory_to_dataframe(trajectory)
    F = compute_F(trajectory)
    df_out['F'] = F.numpy()
    return df_out


# ---------------------------------------------------------------------------
# Per-process sampling (unchanged from paper pattern)
# ---------------------------------------------------------------------------

def generate_observational(process_name: str, n: int, seed: int) -> pd.DataFrame:
    """Sample observational (reference) data from a process SCM."""
    ds = PROCESS_DATASETS[process_name]
    return ds.sample(n, seed=seed)


def generate_interventional(
    process_name: str,
    variable: str,
    value: float,
    n: int,
    seed: int,
) -> pd.DataFrame:
    """Sample interventional data under do(variable=value)."""
    ds = PROCESS_DATASETS[process_name]
    scm_int = ds.scm.do({variable: value})
    return scm_int.sample(n, seed=seed)


def generate_validation_pair(
    process_name: str,
    variable: str,
    value: float,
    n: int,
    seed: int,
) -> pd.DataFrame:
    """
    Generate a validation dataset with flag column (0=control, 1=treatment).
    """
    df_obs = generate_observational(process_name, n, seed=seed)
    df_int = generate_interventional(process_name, variable, value, n, seed=seed + 1)

    df_obs['flag'] = 0
    df_int['flag'] = 1

    return pd.concat([df_obs, df_int], ignore_index=True)


# ---------------------------------------------------------------------------
# Full data generation
# ---------------------------------------------------------------------------

def generate_all_data(
    output_dir: Path,
    n_obs: int = 5000,
    n_int: int = 1000,
    seed: int = 42,
):
    """
    Generate all observational and interventional datasets.

    Directory structure:
        output_dir/
            meta.json
            ground_truth_edges.csv
            ground_truth_adjacency.csv
            <process>/
                observational.csv
                interventional_<var>_<name>.csv
                validate_<var>_<name>.csv
            pipeline/
                observational.csv                          (all outputs + F)
                interventional_<proc>_<var>_<name>.csv     (do on one process)
                validate_<proc>_<var>_<name>.csv           (flag=0/1 pairs)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng_seeds = np.random.SeedSequence(seed).spawn(len(PROCESS_ORDER))
    meta = {
        'created': datetime.now().isoformat(),
        'n_observational': n_obs,
        'n_interventional': n_int,
        'seed': seed,
        'processes': {},
    }

    # Save ground truth
    edges = get_ground_truth_edges()
    nodes = get_all_observable_vars()
    adj = get_ground_truth_adjacency_parent_convention(nodes, as_dataframe=True)

    edges_df = pd.DataFrame(edges, columns=['parent', 'child'])
    edges_df.to_csv(output_dir / 'ground_truth_edges.csv', index=False)
    adj.to_csv(output_dir / 'ground_truth_adjacency.csv')

    print(f'Ground truth: {len(edges)} edges, {len(nodes)} variables')

    # ----- Per-process datasets -----
    for proc_idx, proc_name in enumerate(PROCESS_ORDER):
        proc_dir = output_dir / proc_name
        proc_dir.mkdir(parents=True, exist_ok=True)

        proc_seed = int(rng_seeds[proc_idx].generate_state(1)[0])
        info = PROCESS_OBSERVABLE_VARS[proc_name]
        schedule = INTERVENTION_SCHEDULE[proc_name]

        proc_meta = {
            'inputs': info['inputs'],
            'outputs': info['outputs'],
            'experiments': [],
        }

        print(f'\n[{proc_name}] Generating observational data (n={n_obs})...')
        df_obs = generate_observational(proc_name, n_obs, seed=proc_seed)
        obs_cols = info['inputs'] + info['outputs']
        df_obs[obs_cols].to_csv(proc_dir / 'observational.csv', index=False)
        proc_meta['experiments'].append({
            'name': 'observational',
            'type': 'observational',
            'n_samples': n_obs,
            'file': 'observational.csv',
        })

        int_seed_offset = 0
        for var_name, interventions_list in schedule.items():
            for intv in interventions_list:
                intv_name = intv['name']
                intv_value = intv['value']
                int_seed_offset += 1

                print(f'  do({var_name}={intv_value}) [{intv_name}] (n={n_int})')
                df_int = generate_interventional(
                    proc_name, var_name, intv_value,
                    n=n_int, seed=proc_seed + int_seed_offset * 100,
                )
                int_fname = f'interventional_{var_name}_{intv_name}.csv'
                df_int[obs_cols].to_csv(proc_dir / int_fname, index=False)

                df_val = generate_validation_pair(
                    proc_name, var_name, intv_value,
                    n=n_int, seed=proc_seed + int_seed_offset * 100,
                )
                val_fname = f'validate_{var_name}_{intv_name}.csv'
                df_val[obs_cols + ['flag']].to_csv(proc_dir / val_fname, index=False)

                proc_meta['experiments'].append({
                    'name': f'do({var_name}={intv_value})',
                    'label': intv_name,
                    'type': 'interventional',
                    'target_variable': var_name,
                    'target_value': intv_value,
                    'n_samples': n_int,
                    'interventional_file': int_fname,
                    'validation_file': val_fname,
                })

        meta['processes'][proc_name] = proc_meta

    # ----- Joint pipeline datasets (all processes + F) -----
    print('\n[pipeline] Generating joint pipeline trajectories with F...')
    pipeline_dir = output_dir / 'pipeline'
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    # Observational reference trajectory
    df_pipeline_obs = sample_joint_pipeline(n_obs, seed=seed)
    df_pipeline_obs.to_csv(pipeline_dir / 'observational.csv', index=False)
    print(f'  Observational: {n_obs} samples, '
          f'F mean={df_pipeline_obs["F"].mean():.4f}')

    pipeline_meta = {
        'experiments': [{'name': 'observational', 'file': 'observational.csv'}],
    }

    # Interventional trajectories
    int_seed = seed + 1000
    for proc_name in PROCESS_ORDER:
        schedule = INTERVENTION_SCHEDULE[proc_name]
        for var_name, interventions_list in schedule.items():
            for intv in interventions_list:
                int_seed += 1
                intv_name = intv['name']
                intv_value = intv['value']

                df_int = sample_joint_pipeline(
                    n_int, seed=int_seed,
                    interventions={proc_name: {var_name: intv_value}},
                )
                fname = f'interventional_{proc_name}_{var_name}_{intv_name}.csv'
                df_int.to_csv(pipeline_dir / fname, index=False)

                # Validation pair (flag=0/1)
                df_obs_val = sample_joint_pipeline(n_int, seed=int_seed + 500)
                df_obs_val['flag'] = 0
                df_int_val = df_int.copy()
                df_int_val['flag'] = 1
                df_validate = pd.concat([df_obs_val, df_int_val], ignore_index=True)
                val_fname = f'validate_{proc_name}_{var_name}_{intv_name}.csv'
                df_validate.to_csv(pipeline_dir / val_fname, index=False)

                F_shift = df_int['F'].mean() - df_pipeline_obs['F'].mean()
                print(f'  do({proc_name}.{var_name}={intv_value}) [{intv_name}]: '
                      f'F={df_int["F"].mean():.4f} (delta={F_shift:+.4f})')

                pipeline_meta['experiments'].append({
                    'name': f'do({proc_name}.{var_name}={intv_value})',
                    'process': proc_name,
                    'variable': var_name,
                    'value': intv_value,
                    'label': intv_name,
                    'file': fname,
                    'validation_file': val_fname,
                })

    meta['pipeline'] = pipeline_meta

    # Save metadata
    with open(output_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    print(f'\nAll data saved to: {output_dir}')
    return meta


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate observational and interventional datasets from SCMs',
    )
    parser.add_argument('--output_dir', type=str, default='scm_ds/predictor_dataset/causal_chamber',
                        help='Output directory')
    parser.add_argument('--n_obs', type=int, default=5000,
                        help='Number of observational samples per process')
    parser.add_argument('--n_int', type=int, default=1000,
                        help='Number of interventional samples per experiment')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


def main():
    args = parse_args()
    print('=' * 60)
    print('Causal Chamber — Data Generation')
    print('=' * 60)
    meta = generate_all_data(
        output_dir=Path(args.output_dir),
        n_obs=args.n_obs,
        n_int=args.n_int,
        seed=args.seed,
    )
    n_total = sum(
        len(p['experiments']) for p in meta['processes'].values()
    )
    print(f'\nGenerated {n_total} per-process datasets + '
          f'{len(meta["pipeline"]["experiments"])} pipeline datasets.')


if __name__ == '__main__':
    main()
