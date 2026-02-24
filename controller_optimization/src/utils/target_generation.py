"""
Single-target + multi-scenario trajectory generation.

Design:
- ONE ideal target: A single operating point with process noise = 0.
  Represents the quality specification (controllable inputs + expected outputs).
- N scenario conditions: Diverse structural conditions (temperature, etc.)
  generated independently. Define the environmental diversity.
- Expanded target: Combines the single target's controllable inputs
  (replicated N times) with each scenario's structural conditions.
  BC loss always compares to the SAME controllable target.
- Baseline trajectory: Same controllable inputs as target + same scenario
  structural conditions + ACTIVE process noise.
  Represents "actual performance WITHOUT controller".

Key functions:
- generate_target_trajectory(): Generate target or conditions via SCM sampling
- build_expanded_target(): Merge single target + N conditions into multi-scenario target
- generate_baseline_trajectory(): Baseline with process noise using expanded target inputs
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


def get_scm_dataset(process_config):
    """Get SCM dataset for a process configuration."""
    scm_type = process_config['scm_dataset_type']

    if scm_type == 'laser':
        return ds_scm_laser
    elif scm_type == 'plasma':
        return ds_scm_plasma
    elif scm_type == 'galvanic':
        return ds_scm_galvanic
    elif scm_type == 'microetch':
        return ds_scm_microetch
    else:
        raise ValueError(f"Unknown SCM dataset type: {scm_type}")


def generate_target_trajectory(process_configs, n_samples=50, seed=42):
    """
    Generate N target trajectories with diverse operating conditions.

    Each trajectory represents:
    - Different environmental conditions (structural noise ACTIVE)
    - Ideal deterministic behavior (process noise = 0)

    Example output for laser with n_samples=3:
    {
        'laser': {
            'inputs': [[0.5, 12.0],   # Scenario 1: Temp=12°C
                      [0.5, 15.0],   # Scenario 2: Temp=15°C
                      [0.5, 18.0]],  # Scenario 3: Temp=18°C
            'outputs': [[0.455],
                       [0.450],
                       [0.447]],
            'structural_conditions': {
                'AmbientTemp': [12.0, 15.0, 18.0]
            }
        }
    }

    Args:
        process_configs (list): List of process configurations (from PROCESSES)
        n_samples (int): Number of scenarios to generate (default: 50)
        seed (int): Random seed for reproducibility

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
        input_labels = process_config['input_labels']
        output_labels = process_config['output_labels']

        # Get SCM dataset
        ds_scm = get_scm_dataset(process_config)

        # Backup original noise model
        original_singles = ds_scm.noise_model.singles.copy()
        original_groups = ds_scm.noise_model.groups.copy() if ds_scm.noise_model.groups else []

        try:
            # Create modified noise model:
            # - Structural noise: KEEP ORIGINAL (for scenario diversity)
            # - Process noise: SET TO EPSILON (near zero, ideal behavior)
            # - Other variables: KEEP ORIGINAL (for safety)

            modified_singles = {}
            epsilon = 1e-6  # Very small value to avoid numerical issues

            for var_name, noise_fn in original_singles.items():
                if var_name in ds_scm.structural_noise_vars:
                    # Keep structural noise ACTIVE for scenario diversity
                    modified_singles[var_name] = noise_fn
                elif var_name in ds_scm.process_noise_vars:
                    # Zero out process noise for ideal deterministic behavior
                    modified_singles[var_name] = lambda rng, n, eps=epsilon: rng.normal(0, eps, n)
                else:
                    # Unknown variable - keep original for safety
                    # (This includes inputs, constants, intermediate nodes)
                    modified_singles[var_name] = noise_fn

            # Apply modified noise model
            ds_scm.noise_model.singles = modified_singles
            ds_scm.noise_model.groups = []  # No grouped noise

            # Generate samples with diverse structural conditions
            df = ds_scm.sample(n=n_samples, seed=seed)

            # Extract inputs and outputs
            inputs = df[input_labels].values  # Shape: (n_samples, input_dim)
            outputs = df[output_labels].values  # Shape: (n_samples, output_dim)

            # Extract structural conditions for alignment with baseline
            structural_conditions = {}
            for var in ds_scm.structural_noise_vars:
                if var in df.columns:
                    structural_conditions[var] = df[var].values

            trajectory[process_name] = {
                'inputs': inputs,
                'outputs': outputs,
                'structural_conditions': structural_conditions
            }

            print(f"Generated target trajectory for {process_name}:")
            print(f"  - Shape: {inputs.shape}")
            print(f"  - Structural vars: {list(structural_conditions.keys())}")
            if structural_conditions:
                for var, vals in structural_conditions.items():
                    print(f"    {var}: [{vals.min():.2f}, {vals.max():.2f}] (range)")

        finally:
            # Restore original noise model
            ds_scm.noise_model.singles = original_singles
            ds_scm.noise_model.groups = original_groups

    return trajectory


def generate_baseline_trajectory(process_configs, target_trajectory, n_samples=50, seed=43):
    """
    Generate baseline trajectories with SAME inputs as target but with process noise.

    For each scenario in target_trajectory:
    - Use EXACT SAME input values (all input variables fixed to target values)
    - Use SAME structural conditions (temperature, humidity, etc.)
    - Use ACTIVE process noise (realistic equipment variability)

    This represents: "What happens in reality WITHOUT the controller, with same inputs"

    Args:
        process_configs (list): Process configuration list
        target_trajectory (dict): Output from generate_target_trajectory() - inputs and structural conditions are copied
        n_samples (int): Must match target trajectory n_samples
        seed (int): Different from target seed to get different process noise

    Returns:
        dict: {
            process_name: {
                'inputs': np.array of shape (n_samples, input_dim),
                'outputs': np.array of shape (n_samples, output_dim)
            }
        }
    """
    trajectory = {}

    for process_config in process_configs:
        process_name = process_config['name']
        input_labels = process_config['input_labels']
        output_labels = process_config['output_labels']

        # Get SCM dataset
        ds_scm = get_scm_dataset(process_config)

        # Get inputs and structural conditions from target
        target_inputs = target_trajectory[process_name]['inputs']
        target_structural = target_trajectory[process_name]['structural_conditions']

        # Backup original noise model
        original_singles = ds_scm.noise_model.singles.copy()

        try:
            modified_singles = original_singles.copy()

            # CRITICAL: Fix ALL input variables to target values
            # This ensures exact same inputs between target and baseline
            for i, input_label in enumerate(input_labels):
                target_values = target_inputs[:, i]
                # Create a closure to capture target_values
                def make_input_sampler(values):
                    return lambda rng, n: values[:n]
                modified_singles[input_label] = make_input_sampler(target_values)

            # Also fix structural variables (if not already in inputs)
            for var_name, var_values in target_structural.items():
                if var_name not in input_labels:  # Only if not already fixed as input
                    # Create a closure to capture var_values
                    def make_structural_sampler(values):
                        return lambda rng, n: values[:n]
                    modified_singles[var_name] = make_structural_sampler(var_values)

            # Apply modified noise model
            ds_scm.noise_model.singles = modified_singles

            # Generate samples with fixed inputs and active process noise
            df = ds_scm.sample(n=n_samples, seed=seed)

        finally:
            # Restore original noise model
            ds_scm.noise_model.singles = original_singles

        # Extract inputs and outputs
        inputs = df[input_labels].values  # Shape: (n_samples, input_dim)
        outputs = df[output_labels].values  # Shape: (n_samples, output_dim)

        trajectory[process_name] = {
            'inputs': inputs,
            'outputs': outputs
        }

        print(f"Generated baseline trajectory for {process_name}:")
        print(f"  - Shape: {inputs.shape}")
        print(f"  - All inputs FIXED to target values")
        if target_structural:
            print(f"  - Aligned structural vars: {list(target_structural.keys())}")

    return trajectory


def build_expanded_target(single_target, scenario_conditions, process_configs):
    """
    Build an expanded multi-scenario target trajectory by combining:
    - Controllable inputs from single_target (replicated N times)
    - Non-controllable inputs from scenario_conditions (per scenario)
    - Outputs from single_target (replicated N times)

    This creates a target where:
    - All scenarios share the SAME controllable target (ideal operating point)
    - Each scenario has DIFFERENT structural conditions (environmental diversity)
    - All scenarios share the SAME target outputs (quality specification)

    The BC loss will always compare to the same controllable inputs,
    while the controller receives different structural conditions per scenario.

    Args:
        single_target: dict from generate_target_trajectory(n_samples=1)
            Contains the ideal operating point for each process.
        scenario_conditions: dict from generate_target_trajectory(n_samples=N)
            Only its non-controllable input values and structural_conditions are used.
        process_configs: list of process configurations (from PROCESSES)

    Returns:
        dict: expanded target with shape (N, ...) for each process:
            {
                process_name: {
                    'inputs': np.array (N, input_dim),
                    'outputs': np.array (N, output_dim),
                    'structural_conditions': dict of structural variable values
                }
            }
    """
    from controller_optimization.configs.processes_config import get_controllable_inputs

    n_scenarios = None
    expanded = {}

    for process_config in process_configs:
        process_name = process_config['name']
        input_labels = process_config['input_labels']
        controllable = get_controllable_inputs(process_config)

        # Get single target values
        target_inputs = single_target[process_name]['inputs']    # (1, input_dim)
        target_outputs = single_target[process_name]['outputs']  # (1, output_dim)

        # Get scenario conditions (full input arrays with diverse structural values)
        conditions_inputs = scenario_conditions[process_name]['inputs']  # (N, input_dim)

        if n_scenarios is None:
            n_scenarios = conditions_inputs.shape[0]

        # Build expanded inputs:
        # - Controllable: replicate single target value across all scenarios
        # - Non-controllable: keep from scenario_conditions (diverse per scenario)
        expanded_inputs = np.copy(conditions_inputs)

        for idx, label in enumerate(input_labels):
            if label in controllable:
                # Overwrite controllable values with single target's values
                expanded_inputs[:, idx] = target_inputs[0, idx]

        # Outputs: replicate single target outputs across all scenarios
        # (the quality specification is always the same)
        expanded_outputs = np.tile(target_outputs, (n_scenarios, 1))

        # Structural conditions: take from scenario_conditions
        structural_conditions = {}
        if 'structural_conditions' in scenario_conditions[process_name]:
            structural_conditions = scenario_conditions[process_name]['structural_conditions']

        expanded[process_name] = {
            'inputs': expanded_inputs,
            'outputs': expanded_outputs,
            'structural_conditions': structural_conditions
        }

        print(f"  Expanded target for {process_name}:")
        print(f"    Shape: {expanded_inputs.shape}")
        for idx, label in enumerate(input_labels):
            vals = expanded_inputs[:, idx]
            source = "target (fixed)" if label in controllable else "conditions (variable)"
            if label in controllable:
                print(f"    {label} [{source}]: {vals[0]:.4f} (same for all scenarios)")
            else:
                print(f"    {label} [{source}]: [{vals.min():.4f}, {vals.max():.4f}]")

    print(f"  Total: {n_scenarios} scenarios, single controllable target")

    return expanded


if __name__ == '__main__':
    # Test single-target + multi-scenario trajectory generation
    from controller_optimization.configs.processes_config import PROCESSES

    print("="*70)
    print("TESTING SINGLE-TARGET + MULTI-SCENARIO TRAJECTORY GENERATION")
    print("="*70)

    n_test_scenarios = 10

    # Step 1: Generate SINGLE ideal target (n=1)
    print("\n" + "="*70)
    print("STEP 1: SINGLE IDEAL TARGET (a*, n=1, process noise=0)")
    print("="*70)
    single_target = generate_target_trajectory(PROCESSES, n_samples=1, seed=42)

    for process_name, data in single_target.items():
        print(f"\n{process_name.upper()}:")
        print(f"  Inputs:  {data['inputs'][0]}")
        print(f"  Outputs: {data['outputs'][0]}")

    # Step 2: Generate N scenario conditions (diverse structural noise)
    print("\n" + "="*70)
    print(f"STEP 2: {n_test_scenarios} SCENARIO CONDITIONS (diverse structural noise)")
    print("="*70)
    scenario_conditions = generate_target_trajectory(PROCESSES, n_samples=n_test_scenarios, seed=542)

    for process_name, data in scenario_conditions.items():
        print(f"\n{process_name.upper()}:")
        print(f"  Shape: {data['inputs'].shape}")
        if data['structural_conditions']:
            for var, vals in data['structural_conditions'].items():
                print(f"  {var}: [{vals.min():.4f}, {vals.max():.4f}] (range)")

    # Step 3: Build expanded target
    print("\n" + "="*70)
    print("STEP 3: BUILD EXPANDED TARGET (single target × N conditions)")
    print("="*70)
    expanded_target = build_expanded_target(
        single_target=single_target,
        scenario_conditions=scenario_conditions,
        process_configs=PROCESSES
    )

    print("\n  VERIFICATION: Controllable inputs should be IDENTICAL across all scenarios")
    from controller_optimization.configs.processes_config import get_controllable_inputs
    for process_config in PROCESSES:
        process_name = process_config['name']
        controllable = get_controllable_inputs(process_config)
        inputs = expanded_target[process_name]['inputs']
        for idx, label in enumerate(process_config['input_labels']):
            vals = inputs[:, idx]
            if label in controllable:
                assert np.all(vals == vals[0]), f"{label} should be identical across scenarios!"
                print(f"  {process_name}.{label}: {vals[0]:.4f} (same for all {n_test_scenarios} scenarios) ✓")
            else:
                assert np.std(vals) > 0, f"{label} should vary across scenarios!"
                print(f"  {process_name}.{label}: [{vals.min():.4f}, {vals.max():.4f}] (varies) ✓")

    # Step 4: Generate baseline from expanded target
    print("\n" + "="*70)
    print("STEP 4: BASELINE TRAJECTORY (same inputs + active process noise)")
    print("="*70)
    baseline_traj = generate_baseline_trajectory(
        PROCESSES,
        target_trajectory=expanded_target,
        n_samples=n_test_scenarios,
        seed=43
    )

    # Compare
    print("\n" + "="*70)
    print("COMPARISON: EXPANDED TARGET vs BASELINE")
    print("="*70)
    for process_name in expanded_target.keys():
        target_outputs = expanded_target[process_name]['outputs']
        baseline_outputs = baseline_traj[process_name]['outputs']

        print(f"\n{process_name.upper()}:")
        print(f"  Target output std:   {np.std(target_outputs):.6f} (should be 0 — same single target)")
        print(f"  Baseline output std: {np.std(baseline_outputs):.6f} (varies due to process noise + structural)")

        # CRITICAL CHECK: Verify that ALL inputs are identical between target and baseline
        target_inputs = expanded_target[process_name]['inputs']
        baseline_inputs = baseline_traj[process_name]['inputs']

        input_labels = [p for p in PROCESSES if p['name'] == process_name][0]['input_labels']
        print(f"  INPUT ALIGNMENT CHECK (should be exactly 0.0):")
        for i, input_label in enumerate(input_labels):
            max_diff = np.max(np.abs(target_inputs[:, i] - baseline_inputs[:, i]))
            print(f"    {input_label}: max_diff={max_diff:.10f}")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
