"""
Target and baseline trajectory generation with adaptive calibration targets.

Key Concepts:
- Target Trajectory (a*): A SINGLE trajectory sampled with seed_target.
  seed_target determines both environmental parameters and controllable inputs.
  Process noise is ZERO (ideal deterministic behavior). F* is NOT necessarily 1.

- Baseline Trajectories (a'): n_train baselines, each with:
  * SAME controllable inputs as target (copied)
  * DIFFERENT environmental parameters per baseline (sampled with seed_env)
  * ACTIVE process noise (sampled with seed_noise)

- Calibration: Sets base targets for individual processes (via SCM calibration).
  These become adaptive targets in the Q calculation:
  τ_i = base_target + Σ coeff_j × (Y_j - baseline_j)

Train:
  - Target: 1 sample (seed_target → env params + inputs)
  - Baselines: n_train, each with target inputs + different env params + noise

Test:
  - Baselines: n_test, same controllable inputs and SAME noise realization
    as train baselines, but DIFFERENT env params (seed_env offset by n_train)
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path for scm_ds import
REPO_ROOT = Path(__file__).parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import SCM datasets from the shared scm_ds package
from scm_ds.datasets import (
    ds_scm_laser,
    ds_scm_plasma,
    ds_scm_galvanic,
    ds_scm_microetch
)

# Cache per SCM ST già costruiti (evita di ricostruire ad ogni chiamata)
_st_scm_cache = {}


def get_scm_dataset(process_config):
    """Get SCM dataset for a process configuration.

    Per processi fisici ritorna l'istanza pre-built.
    Per processi ST, costruisce (e cachea) un SCMDataset da STConfig.
    """
    scm_type = process_config['scm_dataset_type']

    if scm_type == 'laser':
        return ds_scm_laser
    elif scm_type == 'plasma':
        return ds_scm_plasma
    elif scm_type == 'galvanic':
        return ds_scm_galvanic
    elif scm_type == 'microetch':
        return ds_scm_microetch
    elif scm_type == 'st':
        process_name = process_config['name']
        if process_name not in _st_scm_cache:
            from scm_ds.datasets_st import STConfig, build_st_scm
            st_params = process_config['st_params']
            _st_scm_cache[process_name] = build_st_scm(STConfig(**st_params))
        return _st_scm_cache[process_name]
    else:
        raise ValueError(f"Unknown SCM dataset type: {scm_type}")


def _get_scm_labels(process_config):
    """Get SCM-level input/output labels for a process.

    For ST processes, returns base labels (without suffix).
    For physical processes, returns the labels as-is.
    """
    is_st = process_config['scm_dataset_type'] == 'st'
    if is_st:
        return (process_config['_st_base_input_labels'],
                process_config['_st_base_output_labels'])
    else:
        return (process_config['input_labels'],
                process_config['output_labels'])


def _get_controllable_scm_labels(process_config):
    """Get which SCM input labels are controllable.

    Returns:
        set: SCM-level labels that are controllable
    """
    from configs.processes_config import get_controllable_inputs

    input_labels = process_config['input_labels']
    controllable = set(get_controllable_inputs(process_config))
    scm_input_labels, _ = _get_scm_labels(process_config)

    controllable_scm = set()
    for config_label, scm_label in zip(input_labels, scm_input_labels):
        if config_label in controllable:
            controllable_scm.add(scm_label)
    return controllable_scm


def _zero_process_noise(ds_scm, original_singles):
    """Create modified noise model with process noise zeroed out.

    Returns dict of modified singles (structural noise kept, process noise zeroed).
    """
    modified_singles = {}
    for var_name, noise_fn in original_singles.items():
        if var_name in ds_scm.structural_noise_vars:
            modified_singles[var_name] = noise_fn
        elif var_name in ds_scm.process_noise_vars:
            if var_name.startswith("Z_ln"):
                modified_singles[var_name] = lambda rng, n: np.ones(n)
            else:
                modified_singles[var_name] = lambda rng, n: np.zeros(n)
        else:
            modified_singles[var_name] = noise_fn
    return modified_singles


def generate_target_trajectory(process_configs, n_samples=1, seed=42):
    """
    Generate the target trajectory (a*).

    The target trajectory is a SINGLE sample (n_samples=1 by default).
    seed_target determines both environmental parameters and controllable inputs.
    Process noise is ZERO (ideal deterministic behavior).
    F* is computed from this trajectory and is NOT necessarily 1.

    Args:
        process_configs (list): List of process configurations (from PROCESSES)
        n_samples (int): Number of samples (default: 1, single target trajectory)
        seed (int): Random seed — determines both env params and inputs

    Returns:
        dict: {
            process_name: {
                'inputs': np.array of shape (n_samples, input_dim),
                'outputs': np.array of shape (n_samples, output_dim),
                'structural_conditions': dict of structural variable values per sample
            }
        }
    """
    trajectory = {}

    for process_config in process_configs:
        process_name = process_config['name']
        scm_input_labels, scm_output_labels = _get_scm_labels(process_config)

        ds_scm = get_scm_dataset(process_config)

        # Backup original noise model
        original_singles = ds_scm.noise_model.singles.copy()
        original_groups = ds_scm.noise_model.groups.copy() if ds_scm.noise_model.groups else []

        try:
            # Structural noise ACTIVE (determines env params + inputs)
            # Process noise ZERO (ideal deterministic behavior)
            modified_singles = _zero_process_noise(ds_scm, original_singles)

            # Restrict controllable inputs to action_domain if specified
            action_domain = process_config.get('action_domain')
            if action_domain is not None:
                a_lo, a_hi = action_domain
                controllable_scm = _get_controllable_scm_labels(process_config)
                for scm_label in controllable_scm:
                    if scm_label in modified_singles:
                        modified_singles[scm_label] = (
                            lambda rng, n, lo=a_lo, hi=a_hi: rng.uniform(lo, hi, size=n)
                        )

            ds_scm.noise_model.singles = modified_singles
            ds_scm.noise_model.groups = []

            # Sample: seed determines everything (env params + inputs)
            df = ds_scm.sample(n=n_samples, seed=seed)
            inputs = df[scm_input_labels].values
            outputs = df[scm_output_labels].values

            structural_conditions = {}
            for var in ds_scm.structural_noise_vars:
                if var in df.columns:
                    structural_conditions[var] = df[var].values

            print(f"Generated target trajectory for {process_name}:")

            trajectory[process_name] = {
                'inputs': inputs,
                'outputs': outputs,
                'structural_conditions': structural_conditions
            }

            print(f"  - Shape: {inputs.shape}")
            print(f"  - Structural vars: {list(structural_conditions.keys())}")
            if structural_conditions:
                for var, vals in structural_conditions.items():
                    print(f"    {var}: [{vals.min():.2f}, {vals.max():.2f}] (range)")

        finally:
            ds_scm.noise_model.singles = original_singles
            ds_scm.noise_model.groups = original_groups

    return trajectory


def generate_baseline_trajectories(process_configs, target_trajectory, n_baselines,
                                   seed_env, seed_noise):
    """
    Generate n_baselines baseline trajectories with different env params.

    Baseline 0 uses the SAME environmental parameters as the target trajectory,
    so the first scenario isolates the effect of process noise alone.
    Baselines 1..n_baselines-1 have DIFFERENT env params (sampled with seed_env).

    All baselines:
    - Copy CONTROLLABLE inputs from target (fixed)
    - Have ACTIVE process noise (sampled with seed_noise)
    - Copy non-controllable, non-structural inputs from target

    By separating seed_env and seed_noise, test baselines can reuse the
    same noise realization (seed_noise) while having different env params
    (different seed_env).

    Args:
        process_configs (list): Process configuration list
        target_trajectory (dict): Output from generate_target_trajectory() (1 sample)
        n_baselines (int): Number of baselines to generate
        seed_env (int): Seed for environmental parameters (baselines 1..n-1)
        seed_noise (int): Seed for process noise

    Returns:
        dict: {
            process_name: {
                'inputs': np.array of shape (n_baselines, input_dim),
                'outputs': np.array of shape (n_baselines, output_dim),
                'structural_conditions': dict of env param values per baseline
            }
        }
    """
    from configs.processes_config import get_controllable_inputs

    trajectory = {}

    for proc_idx, process_config in enumerate(process_configs):
        process_name = process_config['name']
        input_labels = process_config['input_labels']
        scm_input_labels, scm_output_labels = _get_scm_labels(process_config)

        ds_scm = get_scm_dataset(process_config)

        # Target inputs (1 sample)
        target_inputs = target_trajectory[process_name]['inputs']  # (1, input_dim)
        target_structural = target_trajectory[process_name].get('structural_conditions', {})

        # Determine which SCM labels are controllable
        controllable_scm = _get_controllable_scm_labels(process_config)

        # Step 1: Build env params array for all baselines
        # Baseline 0: SAME env params as target
        # Baselines 1..n-1: sampled with seed_env
        env_values = {}
        for var in ds_scm.structural_noise_vars:
            if var in ds_scm.noise_model.singles:
                # Target's env param value (baseline 0)
                target_env_val = target_structural.get(var, None)
                if target_env_val is not None:
                    target_val_0 = float(target_env_val[0])
                else:
                    # Fallback: extract from target inputs if var is an input label
                    var_idx = None
                    for i, scm_label in enumerate(scm_input_labels):
                        if scm_label == var:
                            var_idx = i
                            break
                    target_val_0 = float(target_inputs[0, var_idx]) if var_idx is not None else 0.0

                if n_baselines == 1:
                    # Only baseline 0: use target env params
                    env_values[var] = np.array([target_val_0])
                else:
                    # Sample n_baselines-1 new env params for baselines 1..n-1
                    rng_env = np.random.RandomState(seed_env + proc_idx)
                    noise_fn = ds_scm.noise_model.singles[var]
                    sampled_env = noise_fn(rng_env, n_baselines - 1)
                    # Prepend target env value as baseline 0
                    env_values[var] = np.concatenate([[target_val_0], sampled_env])

        # Step 2: Fix controllable inputs + env params, sample with process noise
        original_singles = ds_scm.noise_model.singles.copy()
        original_groups = ds_scm.noise_model.groups.copy() if ds_scm.noise_model.groups else []

        try:
            modified_singles = original_singles.copy()

            # Fix controllable inputs to target values (replicated for n_baselines)
            for i, scm_label in enumerate(scm_input_labels):
                if scm_label in controllable_scm:
                    target_val = float(target_inputs[0, i])

                    def make_const_sampler(v):
                        return lambda rng, n: np.full(n, v)

                    modified_singles[scm_label] = make_const_sampler(target_val)
                elif scm_label not in ds_scm.structural_noise_vars:
                    # Non-controllable, non-structural inputs: copy from target
                    target_val = float(target_inputs[0, i])

                    def make_const_sampler(v):
                        return lambda rng, n: np.full(n, v)

                    modified_singles[scm_label] = make_const_sampler(target_val)

            # Fix env params (baseline 0 = target, 1..n-1 = sampled)
            for var, vals in env_values.items():
                def make_env_sampler(values):
                    return lambda rng, n: values[:n]

                modified_singles[var] = make_env_sampler(vals)

            ds_scm.noise_model.singles = modified_singles
            ds_scm.noise_model.groups = []

            # Sample with seed_noise — only process noise is random here
            process_seed = seed_noise + proc_idx
            df = ds_scm.sample(n=n_baselines, seed=process_seed)

        finally:
            ds_scm.noise_model.singles = original_singles
            ds_scm.noise_model.groups = original_groups

        inputs = df[scm_input_labels].values
        outputs = df[scm_output_labels].values

        structural_conditions = {}
        for var in ds_scm.structural_noise_vars:
            if var in df.columns:
                structural_conditions[var] = df[var].values

        trajectory[process_name] = {
            'inputs': inputs,
            'outputs': outputs,
            'structural_conditions': structural_conditions
        }

        print(f"Generated {n_baselines} baseline trajectories for {process_name}:")
        print(f"  - Shape: {inputs.shape}")
        print(f"  - Controllable inputs: FIXED to target values")
        print(f"  - Baseline 0: SAME env params as target")
        if n_baselines > 1 and structural_conditions:
            print(f"  - Baselines 1..{n_baselines-1}: DIFFERENT env params")
            for var, vals in structural_conditions.items():
                print(f"    {var}: target={vals[0]:.4f}, range=[{vals[1:].min():.4f}, {vals[1:].max():.4f}]")

    return trajectory
