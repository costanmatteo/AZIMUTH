"""
Generazione target trajectory e baseline trajectory.

- a* (target): SCM con noise=0 (ottimale, deterministico)
- a' (baseline): SCM con noise normale (per confronto con controller)
"""

import sys
from pathlib import Path
import numpy as np
import importlib.util

# Add uncertainty_predictor to path
REPO_ROOT = Path(__file__).parent.parent.parent.parent
UNCERTAINTY_PREDICTOR_PATH = REPO_ROOT / 'uncertainty_predictor'

# CRITICAL: Add uncertainty_predictor to sys.path FIRST
# This allows the SCM datasets module to import its dependencies
if str(UNCERTAINTY_PREDICTOR_PATH) not in sys.path:
    sys.path.insert(0, str(UNCERTAINTY_PREDICTOR_PATH))

# Load SCM datasets explicitly
spec_datasets = importlib.util.spec_from_file_location(
    "scm_datasets",
    UNCERTAINTY_PREDICTOR_PATH / "scm_ds" / "datasets.py"
)
scm_datasets = importlib.util.module_from_spec(spec_datasets)
sys.modules['scm_datasets'] = scm_datasets  # Add to sys.modules for nested imports
spec_datasets.loader.exec_module(scm_datasets)
ds_scm_laser = scm_datasets.ds_scm_laser
ds_scm_plasma = scm_datasets.ds_scm_plasma
ds_scm_galvanic = scm_datasets.ds_scm_galvanic
ds_scm_microetch = scm_datasets.ds_scm_microetch


def generate_target_trajectory(process_configs, n_samples=1, seed=42):
    """
    Genera target trajectory deterministico (noise=0).

    Questa è la trajectory OTTIMALE senza noise.

    Args:
        process_configs (list): Lista di config processi (da PROCESSES)
        n_samples (int): Numero di samples (default 1)
        seed (int): Random seed

    Returns:
        dict: {
            'laser': {
                'inputs': np.array,      # Shape: (n_samples, input_dim)
                'outputs': np.array,     # Shape: (n_samples, output_dim)
            },
            'plasma': {...},
            ...
        }
    """
    trajectory = {}

    for process_config in process_configs:
        process_name = process_config['name']
        scm_type = process_config['scm_dataset_type']
        input_labels = process_config['input_labels']
        output_labels = process_config['output_labels']

        # Get appropriate SCM dataset
        if scm_type == 'laser':
            ds_scm = ds_scm_laser
        elif scm_type == 'plasma':
            ds_scm = ds_scm_plasma
        elif scm_type == 'galvanic':
            ds_scm = ds_scm_galvanic
        elif scm_type == 'microetch':
            ds_scm = ds_scm_microetch
    
        else:
            raise ValueError(f"Unknown SCM dataset type: {scm_type}")

        # Backup original noise model
        original_singles = ds_scm.noise_model.singles.copy()
        original_groups = ds_scm.noise_model.groups.copy() if ds_scm.noise_model.groups else []

        try:
            # Override with very small noise (epsilon instead of 0 to avoid division by zero)
            epsilon = 1e-4
            tiny_noise_singles = {
                key: (lambda rng, n, eps=epsilon: rng.normal(0, eps, n))
                for key in original_singles.keys()
            }
            zero_groups = []  # No groups with noise

            ds_scm.noise_model.singles = tiny_noise_singles
            ds_scm.noise_model.groups = zero_groups

            # Generate samples with near-zero noise
            df = ds_scm.sample(n=n_samples, seed=seed)

            # Extract inputs and outputs
            inputs = df[input_labels].values  # Shape: (n_samples, input_dim)
            outputs = df[output_labels].values  # Shape: (n_samples, output_dim)

            trajectory[process_name] = {
                'inputs': inputs,
                'outputs': outputs,
            }

        finally:
            # Restore original noise model
            ds_scm.noise_model.singles = original_singles
            ds_scm.noise_model.groups = original_groups

    return trajectory


def generate_baseline_trajectory(process_configs, n_samples=1, seed=42):
    """
    Genera baseline trajectory CON noise normale.

    Questa rappresenta la trajectory SENZA controller (no adaptation).
    Usa gli stessi input della target ma con noise del processo.

    Args:
        process_configs (list): Lista di config processi
        n_samples (int): Numero di samples
        seed (int): Random seed

    Returns:
        dict: Stessa struttura di generate_target_trajectory()

    Note:
        Questa trajectory serve per il confronto finale:
        - a* = target trajectory (noise=0, ottimale)
        - a' = baseline trajectory (noise normale, NO controller)
        - a = actual trajectory (con policy generator, DA VALUTARE)

        Vogliamo dimostrare che: F(a) > F(a') ≈ F(a*)
    """
    trajectory = {}

    for process_config in process_configs:
        process_name = process_config['name']
        scm_type = process_config['scm_dataset_type']
        input_labels = process_config['input_labels']
        output_labels = process_config['output_labels']

        # Get appropriate SCM dataset
        if scm_type == 'laser':
            ds_scm = ds_scm_laser
        elif scm_type == 'plasma':
            ds_scm = ds_scm_plasma
        elif scm_type == 'galvanic':
            ds_scm = ds_scm_galvanic
        elif scm_type == 'microetch':
            ds_scm = ds_scm_microetch
        else:
            raise ValueError(f"Unknown SCM dataset type: {scm_type}")

        # Generate samples WITH normal noise (default behavior)
        df = ds_scm.sample(n=n_samples, seed=seed)

        # Extract inputs and outputs
        inputs = df[input_labels].values  # Shape: (n_samples, input_dim)
        outputs = df[output_labels].values  # Shape: (n_samples, output_dim)

        trajectory[process_name] = {
            'inputs': inputs,
            'outputs': outputs,
        }

    return trajectory


if __name__ == '__main__':
    # Test trajectory generation
    from controller_optimization.configs.processes_config import PROCESSES

    print("Testing trajectory generation...")

    # Generate target trajectory (noise=0)
    print("\nGenerating target trajectory (a*, noise=0)...")
    target_traj = generate_target_trajectory(PROCESSES, n_samples=5, seed=42)

    for process_name, data in target_traj.items():
        print(f"\n{process_name}:")
        print(f"  Inputs shape: {data['inputs'].shape}")
        print(f"  Inputs sample:\n{data['inputs'][:2]}")
        print(f"  Outputs shape: {data['outputs'].shape}")
        print(f"  Outputs sample:\n{data['outputs'][:2]}")

    # Generate baseline trajectory (noise normal)
    print("\n" + "="*70)
    print("\nGenerating baseline trajectory (a', normal noise)...")
    baseline_traj = generate_baseline_trajectory(PROCESSES, n_samples=5, seed=43)

    for process_name, data in baseline_traj.items():
        print(f"\n{process_name}:")
        print(f"  Inputs shape: {data['inputs'].shape}")
        print(f"  Inputs sample:\n{data['inputs'][:2]}")
        print(f"  Outputs shape: {data['outputs'].shape}")
        print(f"  Outputs sample:\n{data['outputs'][:2]}")

    # Compare: target should be more consistent
    print("\n" + "="*70)
    print("\nComparison (output variance):")
    for process_name in target_traj.keys():
        target_var = np.var(target_traj[process_name]['outputs'])
        baseline_var = np.var(baseline_traj[process_name]['outputs'])
        print(f"{process_name}:")
        print(f"  Target variance:   {target_var:.6f}")
        print(f"  Baseline variance: {baseline_var:.6f}")
