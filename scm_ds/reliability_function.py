"""
Numpy-only reliability F computation and training trajectory generation.

Provides:
    - compute_F_numpy: compute chain reliability F from process outputs (no torch)
    - generate_training_trajectories: sample full-chain trajectories for UP/CausalIT

The F formula mirrors ReliabilityFunction in reliability_function/src/compute_reliability.py
but works with numpy arrays and accepts configs as explicit parameters.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple


def compute_F_numpy(
    outputs: Dict[str, np.ndarray],
    chain_configs: Dict[str, dict],
    process_order: List[str],
) -> np.ndarray:
    """
    Compute chain reliability F (numpy-only, no torch).

    Args:
        outputs: {process_name: np.ndarray of shape (n,)} — scalar output per sample.
        chain_configs: {process_name: {'base_target', 'scale', 'weight', ...}}
            Same format as ProTSurrogate._dynamic_configs in surrogate.py.
        process_order: ordered list of process names for dependency resolution.

    Returns:
        np.ndarray of shape (n,) — reliability score per sample.
    """
    quality_scores: Dict[str, np.ndarray] = {}

    for name in process_order:
        if name not in outputs:
            continue

        cfg = chain_configs.get(name, {})
        output = outputs[name]

        # Adaptive target: tau = base + sum_j coeff_j * (o_j - baseline_j)
        target = cfg.get('base_target', 0.0)
        for upstream, coeff in cfg.get('adaptive_coefficients', {}).items():
            if upstream in outputs:
                baseline = cfg.get('adaptive_baselines', {}).get(upstream, 0.0)
                target = target + coeff * (outputs[upstream] - baseline)

        scale = cfg.get('scale', 1.0)
        quality_scores[name] = np.exp(-((output - target) ** 2) / max(scale, 1e-8))

    # Weighted average
    total_wq = 0.0
    total_w = 0.0
    for name, q in quality_scores.items():
        w = chain_configs.get(name, {}).get('weight', 1.0)
        total_wq = total_wq + w * q
        total_w += w

    if total_w > 0:
        return total_wq / total_w
    return np.zeros(1)


def generate_training_trajectories(
    processes: List[dict],
    n_samples: int = 2000,
    seed: int = 42,
) -> Tuple[Dict[str, dict], np.ndarray]:
    """
    Generate full-chain trajectories for UP and CausalIT training.

    Each process is sampled independently from its SCM with active noise
    (realistic conditions). The input columns are built with the same logic
    as generate_scm_data() in preprocessing.py to guarantee dimensional
    compatibility with existing UP models.

    Args:
        processes: list of process config dicts (from PROCESSES in processes_config.py).
            Each dict must contain: name, scm_dataset_type, st_params (if ST),
            input_labels, output_labels, surrogate_target, surrogate_scale,
            surrogate_weight, and optionally surrogate_adaptive_coefficients/baselines.
        n_samples: number of samples to generate per process.
        seed: random seed for reproducibility.

    Returns:
        (trajectory, F_array) where:
            trajectory: {process_name: {'inputs': (n, d_in), 'outputs': (n, d_out)}}
            F_array: np.ndarray of shape (n_samples,)
    """
    from .datasets_st import STConfig, build_st_scm

    trajectory: Dict[str, dict] = {}

    for pc in processes:
        name = pc['name']
        scm_type = pc.get('scm_dataset_type', 'st')

        if scm_type == 'st':
            scm_dataset = build_st_scm(STConfig(**pc['st_params']))
        else:
            raise ValueError(
                f"generate_training_trajectories only supports ST datasets, "
                f"got '{scm_type}' for process '{name}'"
            )

        df = scm_dataset.sample(n=n_samples, seed=seed)

        # Build input columns with the same logic as preprocessing.generate_scm_data()
        input_cols = list(scm_dataset.input_labels)
        if hasattr(scm_dataset, 'structural_noise_vars') and scm_dataset.structural_noise_vars:
            input_cols = input_cols + list(scm_dataset.structural_noise_vars)
        output_cols = list(scm_dataset.target_labels)

        trajectory[name] = {
            'inputs': df[input_cols].values,
            'outputs': df[output_cols].values,
        }

    # Build chain_configs and process_order from the inter-process surrogate params
    chain_configs: Dict[str, dict] = {}
    process_order: List[str] = []

    for pc in processes:
        name = pc['name']
        process_order.append(name)
        entry = {
            'base_target': pc['surrogate_target'],
            'scale': pc['surrogate_scale'],
            'weight': pc.get('surrogate_weight', 1.0),
        }
        if 'surrogate_adaptive_coefficients' in pc:
            entry['adaptive_coefficients'] = pc['surrogate_adaptive_coefficients']
            entry['adaptive_baselines'] = pc['surrogate_adaptive_baselines']
        chain_configs[name] = entry

    # Extract scalar outputs for F computation (column 0, since p=1 for ST)
    outputs_for_F = {
        name: trajectory[name]['outputs'][:, 0]
        for name in process_order
    }

    F_array = compute_F_numpy(outputs_for_F, chain_configs, process_order)

    return trajectory, F_array
