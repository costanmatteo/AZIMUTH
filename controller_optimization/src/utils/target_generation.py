"""
Multi-scenario target and baseline trajectory generation.

Key Concepts:
- Structural Noise: Environmental conditions that create scenario diversity (ACTIVE in target)
- Process Noise: Measurement/actuator imperfections (ZERO in target, ACTIVE in baseline)
- Calibration: Separate step that fixes base_target values for Q computation.
  NOT injected into the target trajectory.

Target Trajectory (a*):
- seed_target selects BOTH environmental params AND controllable inputs
- Zero process noise (ideal deterministic behavior)
- Output NOT necessarily = 1 (depends on sampled conditions)
- Represents: "Best achievable performance under varying conditions"

Baseline Trajectory (a') - TRAIN:
- SAME structural conditions AND inputs as target (fair comparison)
- Active process noise via seed_baseline (realistic equipment variability)
- Represents: "Actual performance WITHOUT controller"

Baseline Trajectory (a') - TEST:
- SAME inputs as train baseline
- DIFFERENT environmental params (via seed offset)
- Active process noise (realistic equipment variability)
- Controller test: copies env params from test baseline, inputs from controller
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

        # Per processi ST, le label nel config sono suffissate (X_1_p1),
        # ma lo SCM usa le label base (X_1). Costruiamo il mapping.
        is_st = process_config['scm_dataset_type'] == 'st'
        if is_st:
            scm_input_labels = process_config['_st_base_input_labels']
            scm_output_labels = process_config['_st_base_output_labels']
        else:
            scm_input_labels = input_labels
            scm_output_labels = output_labels

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

            for var_name, noise_fn in original_singles.items():
                if var_name in ds_scm.structural_noise_vars:
                    # Keep structural noise ACTIVE for scenario diversity
                    modified_singles[var_name] = noise_fn
                elif var_name in ds_scm.process_noise_vars:
                    # Zero out process noise for ideal deterministic behavior
                    if var_name.startswith("Z_ln"):
                        # Lognormal multiplicative identity: Z_ln = 1.0
                        modified_singles[var_name] = lambda rng, n: np.ones(n)
                    else:
                        # Additive noise (Eps_add, Jump): set to 0.0
                        modified_singles[var_name] = lambda rng, n: np.zeros(n)
                else:
                    # Unknown variable - keep original for safety
                    # (This includes inputs, constants, intermediate nodes)
                    modified_singles[var_name] = noise_fn

            # Apply modified noise model
            ds_scm.noise_model.singles = modified_singles
            ds_scm.noise_model.groups = []  # No grouped noise

            # ── All processes: sample n_samples from SCM with seed ──
            # seed_target determines both environmental params and input values.
            # No calibration row injection — calibration is a separate concern
            # that fixes base_target values for Q computation.
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

    for proc_idx, process_config in enumerate(process_configs):
        process_name = process_config['name']
        input_labels = process_config['input_labels']
        output_labels = process_config['output_labels']

        # Per processi ST, mapping label suffissate → label base SCM
        is_st = process_config['scm_dataset_type'] == 'st'
        if is_st:
            scm_input_labels = process_config['_st_base_input_labels']
            scm_output_labels = process_config['_st_base_output_labels']
        else:
            scm_input_labels = input_labels
            scm_output_labels = output_labels

        # Get SCM dataset
        ds_scm = get_scm_dataset(process_config)

        # Get inputs and structural conditions from target
        target_inputs = target_trajectory[process_name]['inputs']
        target_structural = target_trajectory[process_name]['structural_conditions']

        # Use target trajectory size (may be 1 for ST calibration reference row)
        actual_n = target_inputs.shape[0]

        # Backup original noise model
        original_singles = ds_scm.noise_model.singles.copy()

        try:
            modified_singles = original_singles.copy()

            # CRITICAL: Fix ALL input variables to target values
            # This ensures exact same inputs between target and baseline
            # Usa le label SCM base per settare i sampler
            for i, scm_label in enumerate(scm_input_labels):
                target_values = target_inputs[:, i]
                # Create a closure to capture target_values
                def make_input_sampler(values):
                    return lambda rng, n: values[:n]
                modified_singles[scm_label] = make_input_sampler(target_values)

            # Also fix structural variables (if not already in inputs)
            for var_name, var_values in target_structural.items():
                if var_name not in scm_input_labels:  # Only if not already fixed as input
                    # Create a closure to capture var_values
                    def make_structural_sampler(values):
                        return lambda rng, n: values[:n]
                    modified_singles[var_name] = make_structural_sampler(var_values)

            # Apply modified noise model
            ds_scm.noise_model.singles = modified_singles

            # Use independent seed per process to get independent noise realizations
            # (otherwise identical SCMs with the same seed produce identical noise)
            process_seed = seed + proc_idx
            df = ds_scm.sample(n=actual_n, seed=process_seed)

        finally:
            # Restore original noise model
            ds_scm.noise_model.singles = original_singles

        # Extract inputs and outputs (usando le label SCM base)
        inputs = df[scm_input_labels].values  # Shape: (n_samples, input_dim)
        outputs = df[scm_output_labels].values  # Shape: (n_samples, output_dim)

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


def generate_test_baseline_trajectory(process_configs, train_baseline_trajectory,
                                      n_test, seed_env, seed_noise):
    """
    Generate test baseline trajectories.

    For each test scenario:
    - SAME controllable inputs as train baseline
    - DIFFERENT environmental params (sampled via seed_env)
    - Active process noise (via seed_noise)

    The controller test will then copy these environmental params
    but use its own inputs.

    Args:
        process_configs (list): Process configuration list
        train_baseline_trajectory (dict): Output from generate_baseline_trajectory() for train
        n_test (int): Number of test scenarios
        seed_env (int): Seed for sampling new environmental parameters
        seed_noise (int): Seed for process noise

    Returns:
        dict: {
            process_name: {
                'inputs': np.array (n_test, input_dim),
                'outputs': np.array (n_test, output_dim),
                'structural_conditions': dict of structural variable values per test scenario
            }
        }
    """
    trajectory = {}

    for proc_idx, process_config in enumerate(process_configs):
        process_name = process_config['name']

        is_st = process_config['scm_dataset_type'] == 'st'
        if is_st:
            scm_input_labels = process_config['_st_base_input_labels']
            scm_output_labels = process_config['_st_base_output_labels']
        else:
            scm_input_labels = process_config['input_labels']
            scm_output_labels = process_config['output_labels']

        ds_scm = get_scm_dataset(process_config)

        # Get train baseline inputs (to be reused in test)
        train_inputs = train_baseline_trajectory[process_name]['inputs']
        # Use the first row's inputs for all test scenarios (same controllable inputs)
        # If train had n_train>1, use the first scenario's inputs as reference
        reference_inputs = train_inputs[0]  # Shape: (input_dim,)

        # Step 1: Sample new environmental params using seed_env
        # with zero process noise (to get clean structural conditions)
        original_singles = ds_scm.noise_model.singles.copy()
        original_groups = ds_scm.noise_model.groups.copy() if ds_scm.noise_model.groups else []

        try:
            # Zero out process noise, keep structural noise active
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

            ds_scm.noise_model.singles = modified_singles
            ds_scm.noise_model.groups = []

            env_df = ds_scm.sample(n=n_test, seed=seed_env)

            # Extract structural conditions from this sample
            structural_conditions = {}
            for var in ds_scm.structural_noise_vars:
                if var in env_df.columns:
                    structural_conditions[var] = env_df[var].values

        finally:
            ds_scm.noise_model.singles = original_singles
            ds_scm.noise_model.groups = original_groups

        # Step 2: Now generate the actual test baseline with:
        #   - fixed inputs from train baseline
        #   - fixed structural conditions from step 1
        #   - active process noise via seed_noise
        original_singles = ds_scm.noise_model.singles.copy()

        try:
            modified_singles = original_singles.copy()

            # Fix all input variables to train baseline values
            for i, scm_label in enumerate(scm_input_labels):
                ref_val = reference_inputs[i]
                def make_const_sampler(val):
                    return lambda rng, n: np.full(n, val)
                modified_singles[scm_label] = make_const_sampler(ref_val)

            # Fix structural variables to test env params
            for var_name, var_values in structural_conditions.items():
                if var_name not in scm_input_labels:
                    def make_structural_sampler(values):
                        return lambda rng, n: values[:n]
                    modified_singles[var_name] = make_structural_sampler(var_values)

            ds_scm.noise_model.singles = modified_singles

            process_seed = seed_noise + proc_idx
            df = ds_scm.sample(n=n_test, seed=process_seed)

        finally:
            ds_scm.noise_model.singles = original_singles

        inputs = df[scm_input_labels].values
        outputs = df[scm_output_labels].values

        trajectory[process_name] = {
            'inputs': inputs,
            'outputs': outputs,
            'structural_conditions': structural_conditions
        }

        print(f"Generated test baseline trajectory for {process_name}:")
        print(f"  - Shape: {inputs.shape}")
        print(f"  - Inputs FIXED to train baseline values")
        print(f"  - Environmental params: NEW (seed_env={seed_env})")
        if structural_conditions:
            for var, vals in structural_conditions.items():
                print(f"    {var}: [{vals.min():.2f}, {vals.max():.2f}] (range)")

    return trajectory


def generate_test_target_trajectory(process_configs, test_baseline_trajectory, n_test, seed):
    """
    Generate test target trajectory that shares environmental params with test baseline.

    For test, the controller needs to know:
    - Environmental params (from test baseline — same conditions)
    - Initial controllable inputs (sampled from seed, not from train)

    The target trajectory for test serves as the "reference" for non-controllable
    inputs that the controller will observe.

    Args:
        process_configs (list): Process configuration list
        test_baseline_trajectory (dict): Output from generate_test_baseline_trajectory()
        n_test (int): Number of test scenarios
        seed (int): Seed for reproducibility

    Returns:
        dict: Same format as generate_target_trajectory()
    """
    trajectory = {}

    for process_config in process_configs:
        process_name = process_config['name']

        is_st = process_config['scm_dataset_type'] == 'st'
        if is_st:
            scm_input_labels = process_config['_st_base_input_labels']
            scm_output_labels = process_config['_st_base_output_labels']
        else:
            scm_input_labels = process_config['input_labels']
            scm_output_labels = process_config['output_labels']

        ds_scm = get_scm_dataset(process_config)

        # Get structural conditions from test baseline
        test_structural = test_baseline_trajectory[process_name]['structural_conditions']

        original_singles = ds_scm.noise_model.singles.copy()
        original_groups = ds_scm.noise_model.groups.copy() if ds_scm.noise_model.groups else []

        try:
            modified_singles = {}
            for var_name, noise_fn in original_singles.items():
                if var_name in ds_scm.structural_noise_vars:
                    # Keep structural noise but we'll override with test baseline values
                    modified_singles[var_name] = noise_fn
                elif var_name in ds_scm.process_noise_vars:
                    # Zero process noise for target (ideal behavior)
                    if var_name.startswith("Z_ln"):
                        modified_singles[var_name] = lambda rng, n: np.ones(n)
                    else:
                        modified_singles[var_name] = lambda rng, n: np.zeros(n)
                else:
                    modified_singles[var_name] = noise_fn

            # Fix structural variables to test baseline values
            for var_name, var_values in test_structural.items():
                def make_structural_sampler(values):
                    return lambda rng, n: values[:n]
                modified_singles[var_name] = make_structural_sampler(var_values)

            ds_scm.noise_model.singles = modified_singles
            ds_scm.noise_model.groups = []

            df = ds_scm.sample(n=n_test, seed=seed)

        finally:
            ds_scm.noise_model.singles = original_singles
            ds_scm.noise_model.groups = original_groups

        inputs = df[scm_input_labels].values
        outputs = df[scm_output_labels].values

        structural_conditions = {}
        for var in ds_scm.structural_noise_vars:
            if var in test_structural:
                structural_conditions[var] = test_structural[var]
            elif var in df.columns:
                structural_conditions[var] = df[var].values

        trajectory[process_name] = {
            'inputs': inputs,
            'outputs': outputs,
            'structural_conditions': structural_conditions
        }

        print(f"Generated test target trajectory for {process_name}:")
        print(f"  - Shape: {inputs.shape}")
        print(f"  - Environmental params: SAME as test baseline")

    return trajectory


if __name__ == '__main__':
    # Test multi-scenario trajectory generation
    from controller_optimization.configs.processes_config import PROCESSES

    print("="*70)
    print("TESTING MULTI-SCENARIO TRAJECTORY GENERATION")
    print("="*70)

    # Generate target trajectory with 10 scenarios (testing with smaller number)
    print("\n" + "="*70)
    print("GENERATING TARGET TRAJECTORY (a*, diverse structural + zero process noise)")
    print("="*70)
    n_test_scenarios = 10
    target_traj = generate_target_trajectory(PROCESSES, n_samples=n_test_scenarios, seed=42)

    print("\n" + "="*70)
    print("TARGET TRAJECTORY ANALYSIS")
    print("="*70)
    for process_name, data in target_traj.items():
        print(f"\n{process_name.upper()}:")
        print(f"  Inputs shape: {data['inputs'].shape}")
        print(f"  Outputs shape: {data['outputs'].shape}")

        # Show first 3 scenarios
        print(f"  First 3 scenarios:")
        for i in range(min(3, n_test_scenarios)):
            print(f"    Scenario {i}: inputs={data['inputs'][i]}, output={data['outputs'][i]}")

        # Show output variance (should still vary due to input/structural variation)
        output_var = np.var(data['outputs'])
        output_std = np.std(data['outputs'])
        print(f"  Output variance: {output_var:.6f} (std: {output_std:.6f})")

        # Show structural conditions
        if data['structural_conditions']:
            print(f"  Structural conditions:")
            for var, vals in data['structural_conditions'].items():
                print(f"    {var}: min={vals.min():.4f}, max={vals.max():.4f}, std={vals.std():.4f}")

    # Generate baseline trajectory (aligned structural conditions)
    print("\n" + "="*70)
    print("GENERATING BASELINE TRAJECTORY (a', same structural + active process noise)")
    print("="*70)
    baseline_traj = generate_baseline_trajectory(
        PROCESSES,
        target_trajectory=target_traj,
        n_samples=n_test_scenarios,
        seed=43
    )

    print("\n" + "="*70)
    print("BASELINE TRAJECTORY ANALYSIS")
    print("="*70)
    for process_name, data in baseline_traj.items():
        print(f"\n{process_name.upper()}:")
        print(f"  Inputs shape: {data['inputs'].shape}")
        print(f"  Outputs shape: {data['outputs'].shape}")

        # Show first 3 scenarios
        print(f"  First 3 scenarios:")
        for i in range(min(3, n_test_scenarios)):
            print(f"    Scenario {i}: inputs={data['inputs'][i]}, output={data['outputs'][i]}")

        # Show output variance (should be higher due to process noise)
        output_var = np.var(data['outputs'])
        output_std = np.std(data['outputs'])
        print(f"  Output variance: {output_var:.6f} (std: {output_std:.6f})")

    # Compare target vs baseline
    print("\n" + "="*70)
    print("COMPARISON: TARGET vs BASELINE")
    print("="*70)
    for process_name in target_traj.keys():
        target_outputs = target_traj[process_name]['outputs']
        baseline_outputs = baseline_traj[process_name]['outputs']

        print(f"\n{process_name.upper()}:")
        print(f"  Target output std:   {np.std(target_outputs):.6f}")
        print(f"  Baseline output std: {np.std(baseline_outputs):.6f}")
        print(f"  Std ratio (baseline/target): {np.std(baseline_outputs) / np.std(target_outputs):.3f}")

        # CRITICAL CHECK: Verify that ALL inputs are identical
        target_inputs = target_traj[process_name]['inputs']
        baseline_inputs = baseline_traj[process_name]['inputs']

        input_labels = PROCESSES[[p['name'] for p in PROCESSES].index(process_name)]['input_labels']
        print(f"\n  INPUT ALIGNMENT CHECK (should be exactly 0.0):")
        for i, input_label in enumerate(input_labels):
            target_vals = target_inputs[:, i]
            baseline_vals = baseline_inputs[:, i]
            max_diff = np.max(np.abs(target_vals - baseline_vals))
            mean_diff = np.mean(np.abs(target_vals - baseline_vals))
            print(f"    {input_label}: max_diff={max_diff:.10f}, mean_diff={mean_diff:.10f}")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
