#!/usr/bin/env python3
"""
PHASE 1 — Standalone SCM data generation.

Generates training data for all processes defined in PROCESSES and saves
them to disk as numpy arrays and a parquet file with full trajectories.

Output structure:
    data/scm_trajectories/
    ├── {process_name}/
    │   ├── inputs.npy          # shape (N, input_dim)
    │   └── outputs.npy         # shape (N, output_dim)
    └── trajectories_full.parquet   # all processes + reliability F

Usage:
    python scm_ds/generate_data.py [--n_samples 10000] [--seed 42] [--output_dir data/]
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add project root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from controller_optimization.configs.processes_config import (
    PROCESSES, DATASET_MODE, ST_DATASET_CONFIG, _build_st_processes,
)


def _instantiate_scm_dataset(process_config):
    """
    Instantiate the correct SCMDataset for a process config.

    Returns:
        (scm_dataset, input_columns, output_columns)
    """
    dataset_type = process_config['scm_dataset_type']

    if dataset_type == 'st':
        from scm_ds.datasets_st import STConfig, build_st_scm
        st_params = process_config.get('st_params')
        if st_params is None:
            raise ValueError(
                f"Process '{process_config['name']}' has dataset_type='st' "
                "but no 'st_params' in config"
            )
        scm_dataset = build_st_scm(STConfig(**st_params))
    else:
        from scm_ds.datasets import (
            ds_scm_1_to_1_ct,
            ds_scm_laser,
            ds_scm_plasma,
            ds_scm_galvanic,
            ds_scm_microetch,
        )
        _DATASET_MAP = {
            'one_to_one_ct': ds_scm_1_to_1_ct,
            'laser': ds_scm_laser,
            'plasma': ds_scm_plasma,
            'galvanic': ds_scm_galvanic,
            'microetch': ds_scm_microetch,
        }
        if dataset_type not in _DATASET_MAP:
            raise ValueError(f"Unknown SCM dataset type: {dataset_type}")
        scm_dataset = _DATASET_MAP[dataset_type]

    # Build column lists (same logic as generate_scm_data in preprocessing.py)
    input_columns = list(scm_dataset.input_labels)
    output_columns = list(scm_dataset.target_labels)

    # For ST datasets, include environmental variables as input
    if dataset_type == 'st' and hasattr(scm_dataset, 'structural_noise_vars'):
        structural_vars = list(scm_dataset.structural_noise_vars)
        input_columns = input_columns + structural_vars

    return scm_dataset, input_columns, output_columns


def _build_reliability_function(processes):
    """
    Build a ReliabilityFunction from process configs.

    For ST processes, uses surrogate_target/surrogate_scale/surrogate_weight
    and surrogate_adaptive_coefficients/surrogate_adaptive_baselines.
    For physical processes, uses default PROCESS_CONFIGS.
    """
    from reliability_function.src.compute_reliability import ReliabilityFunction

    if DATASET_MODE == 'physical':
        # Physical processes use the default configs from process_targets.py
        return ReliabilityFunction()

    # ST mode: build process_configs from process definitions
    process_configs = {}
    process_order = []

    for proc in processes:
        name = proc['name']
        process_order.append(name)

        entry = {
            'base_target': proc.get('surrogate_target', 0.0),
            'scale': proc.get('surrogate_scale', 1.0),
            'weight': proc.get('surrogate_weight', 1.0),
        }

        # Adaptive coefficients
        adaptive_coeffs = proc.get('surrogate_adaptive_coefficients', {})
        adaptive_baselines = proc.get('surrogate_adaptive_baselines', {})
        if adaptive_coeffs:
            entry['adaptive_coefficients'] = adaptive_coeffs
            entry['adaptive_baselines'] = adaptive_baselines

        process_configs[name] = entry

    return ReliabilityFunction(
        process_configs=process_configs,
        process_order=process_order,
    )


def generate_all_data(processes, n_samples, seed, output_dir, save_graphs=True):
    """
    Generate SCM data for all processes and save to disk.

    Args:
        processes: List of process config dicts (from PROCESSES).
        n_samples: Number of samples per process.
        seed: Random seed.
        output_dir: Base output directory.
        save_graphs: Whether to save DAG visualizations.

    Returns:
        dict: Statistics about the generated data.
    """
    output_path = Path(output_dir) / 'scm_trajectories'
    output_path.mkdir(parents=True, exist_ok=True)

    # For ST mode, all processes share the same SCM structure — we only
    # need to sample once and reuse the data for all identical processes.
    is_st_mode = DATASET_MODE == 'st'
    st_reference_data = None  # (X, y, scm_dataset)

    # Collect per-process data for the full-trajectory parquet
    all_process_data = {}
    process_names = []

    for proc in processes:
        process_name = proc['name']
        process_names.append(process_name)
        process_dir = output_path / process_name
        process_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  Processing: {process_name}")

        if is_st_mode and st_reference_data is not None:
            # Reuse data from first ST process (identical SCM)
            X, y, _ = st_reference_data
            print(f"    Reusing data from reference ST process")
        else:
            # Instantiate and sample
            scm_dataset, input_cols, output_cols = _instantiate_scm_dataset(proc)

            print(f"    Sampling {n_samples} trajectories from SCM ({proc['scm_dataset_type']})...")
            df = scm_dataset.sample(n=n_samples, seed=seed)

            X = df[input_cols].values
            y = df[output_cols].values

            print(f"    Input shape: {X.shape}, Output shape: {y.shape}")

            # Save DAG visualizations
            if save_graphs:
                try:
                    scm_dataset.save_dag_image(str(process_dir / 'dag'))
                except Exception as e:
                    print(f"    Warning: Could not save DAG image: {e}")
                try:
                    scm_dataset.save_dag_academic(str(process_dir / 'dag_academic'))
                except Exception:
                    pass
                try:
                    scm_dataset.save_dag_compact(str(process_dir / 'dag_compact'))
                except Exception:
                    pass

            if is_st_mode:
                st_reference_data = (X, y, scm_dataset)

        # Save per-process arrays
        np.save(process_dir / 'inputs.npy', X)
        np.save(process_dir / 'outputs.npy', y)
        print(f"    Saved: {process_dir}/inputs.npy, outputs.npy")

        all_process_data[process_name] = {'inputs': X, 'outputs': y}

    # --- Build full-trajectory parquet with reliability F ---
    print(f"\n  Computing reliability F for {n_samples} trajectories...")

    reliability_fn = _build_reliability_function(processes)

    # Build trajectory dict for ReliabilityFunction
    # Each sample is one trajectory across all processes
    trajectory = {}
    for proc_name in process_names:
        outputs = all_process_data[proc_name]['outputs']
        trajectory[proc_name] = {
            'outputs_mean': torch.tensor(outputs, dtype=torch.float32),
            'outputs_sampled': torch.tensor(outputs, dtype=torch.float32),
        }

    F_values = reliability_fn.compute_reliability(trajectory, use_sampled_outputs=True)
    F_numpy = F_values.detach().numpy()

    print(f"  F statistics: mean={F_numpy.mean():.4f}, std={F_numpy.std():.4f}, "
          f"min={F_numpy.min():.4f}, max={F_numpy.max():.4f}")

    # Build parquet DataFrame
    parquet_data = {}
    for proc_name in process_names:
        inputs = all_process_data[proc_name]['inputs']
        outputs = all_process_data[proc_name]['outputs']

        for j in range(inputs.shape[1]):
            parquet_data[f'{proc_name}_input_{j}'] = inputs[:, j]
        for j in range(outputs.shape[1]):
            parquet_data[f'{proc_name}_output_{j}'] = outputs[:, j]

    parquet_data['F'] = F_numpy

    df_full = pd.DataFrame(parquet_data)
    parquet_path = output_path / 'trajectories_full.parquet'
    df_full.to_parquet(parquet_path, index=False)
    print(f"  Saved full trajectories: {parquet_path} ({len(df_full)} rows)")

    # Save metadata
    metadata = {
        'n_samples': n_samples,
        'seed': seed,
        'dataset_mode': DATASET_MODE,
        'process_names': process_names,
        'process_input_dims': {p['name']: p['input_dim'] for p in processes},
        'process_output_dims': {p['name']: p['output_dim'] for p in processes},
        'F_mean': float(F_numpy.mean()),
        'F_std': float(F_numpy.std()),
    }
    import json
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Phase 1: Generate SCM training data for all processes'
    )
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='Number of samples per process (default: 10000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output_dir', type=str, default='data/',
                        help='Base output directory (default: data/)')
    parser.add_argument('--no_graphs', action='store_true',
                        help='Skip saving DAG graph visualizations')

    # ST dataset overrides (same as train_processes.py for consistency)
    parser.add_argument('--st_n', type=int, default=None,
                        help='ST input variables per process')
    parser.add_argument('--st_m', type=int, default=None,
                        help='ST cascaded stages per process')
    parser.add_argument('--st_rho', type=float, default=None,
                        help='ST noise intensity [0,1]')
    parser.add_argument('--st_n_processes', type=int, default=None,
                        help='Number of ST processes in sequence')

    args = parser.parse_args()

    # Handle ST overrides
    processes = PROCESSES
    _st_overrides = {
        k: v for k, v in [('n', args.st_n), ('m', args.st_m), ('rho', args.st_rho)]
        if v is not None
    }
    if (_st_overrides or args.st_n_processes is not None) and DATASET_MODE == 'st':
        import copy
        st_cfg = copy.deepcopy(ST_DATASET_CONFIG)
        st_cfg['st_params'].update(_st_overrides)
        if args.st_n_processes is not None:
            st_cfg['n_processes'] = args.st_n_processes
        processes = _build_st_processes(st_cfg)
        print(f"[ST Override] Rebuilt processes with: {_st_overrides}"
              f"{f', n_processes={args.st_n_processes}' if args.st_n_processes else ''}")

    print("=" * 70)
    print("PHASE 1: GENERATE SCM DATA")
    print("=" * 70)
    print(f"  Dataset mode: {DATASET_MODE}")
    print(f"  Processes: {[p['name'] for p in processes]}")
    print(f"  Samples per process: {args.n_samples}")
    print(f"  Seed: {args.seed}")
    print(f"  Output dir: {args.output_dir}")

    metadata = generate_all_data(
        processes=processes,
        n_samples=args.n_samples,
        seed=args.seed,
        output_dir=args.output_dir,
        save_graphs=not args.no_graphs,
    )

    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETED")
    print("=" * 70)
    print(f"  Samples: {metadata['n_samples']}")
    print(f"  Processes: {metadata['process_names']}")
    print(f"  F: {metadata['F_mean']:.4f} +/- {metadata['F_std']:.4f}")
    print(f"\nNext: Run train_processes.py (Phase 2) to train Uncertainty Predictors")


if __name__ == '__main__':
    main()
