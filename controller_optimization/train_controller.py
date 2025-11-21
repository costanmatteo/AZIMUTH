"""
Step 2: Training policy generators (controller).

Prerequisito: uncertainty predictors già addestrati (train_processes.py)

Usa: python train_controller.py

Output:
- Policy generators salvati
- Training history
- Report PDF con confronto completo:
  * a* (target trajectory)
  * a' (baseline trajectory, NO controller)
  * a (actual trajectory, CON controller)
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import torch
import numpy as np

# Add controller_optimization to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from controller_optimization.configs.processes_config import PROCESSES, get_filtered_processes
from controller_optimization.configs.controller_config import CONTROLLER_CONFIG
from controller_optimization.src.utils.target_generation import (
    generate_target_trajectory,
    generate_baseline_trajectory
)
from controller_optimization.src.utils.process_chain import ProcessChain
from controller_optimization.src.models.surrogate import ProTSurrogate
from controller_optimization.src.training.controller_trainer import ControllerTrainer
from controller_optimization.src.utils.metrics import (
    compute_final_metrics,
    compute_process_wise_metrics,
    convert_trajectory_to_numpy,
    compute_worst_case_gap,
    compute_success_rate,
    compute_train_test_gap,
    compute_scenario_diversity
)
from controller_optimization.src.utils.visualization import (
    plot_training_history,
    plot_trajectory_comparison,
    plot_reliability_comparison,
    plot_process_improvements,
    plot_target_vs_actual_scatter,
    plot_gap_distribution
)
from controller_optimization.src.utils.report_generator import generate_controller_report
from controller_optimization.src.utils.model_utils import convert_numpy_to_tensor
from controller_optimization.src.utils.scm_validation import validate_all_processes


def main():
    """
    Pipeline completo:

    1. Load config
    2. Generate target trajectory (a*, noise=0)
    3. Generate baseline trajectory (a', noise normale, NO controller)
    4. Create ProcessChain (carica uncertainty predictors frozen)
    5. Create ProTSurrogate
    6. Create ControllerTrainer
    7. Training loop
    8. FINAL EVALUATION:
       - Genera actual trajectory (a) con policy generators
       - Calcola F*, F', F
       - Compute metrics
       - Generate plots
       - Generate PDF report
    9. Save everything
    """

    print("="*70)
    print("CONTROLLER OPTIMIZATION - POLICY GENERATOR TRAINING")
    print("="*70)

    # Filter processes based on configuration
    process_names = CONTROLLER_CONFIG.get('process_names', None)
    selected_processes = get_filtered_processes(process_names)
    print(f"\nSelected processes: {[p['name'] for p in selected_processes]}")
    if process_names:
        print(f"  (filtered from PROCESSES using process_names: {process_names})")
    else:
        print(f"  (using all PROCESSES, no filter applied)")

    # Device setup
    device = CONTROLLER_CONFIG['training']['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # =========================================================================
    # SCM VALIDATION: Check causal consistency
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 0: SCM VALIDATION")
    print("="*70)
    validate_all_processes(selected_processes)

    checkpoint_dir = Path(CONTROLLER_CONFIG['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate TRAINING scenarios (diverse structural + zero process noise)
    print("\n[1/9] Generating training scenarios...")
    n_train = CONTROLLER_CONFIG['scenarios']['n_train']
    n_test = CONTROLLER_CONFIG['scenarios']['n_test']
    print(f"  Training scenarios: {n_train}")
    print(f"  Test scenarios: {n_test} (for future evaluation)")

    # Generate TRAIN target trajectory
    print("\n  Generating TRAIN target trajectory (a*)...")
    target_trajectory_train = generate_target_trajectory(
        process_configs=selected_processes,
        n_samples=n_train,
        seed=CONTROLLER_CONFIG['scenarios']['seed_target']
    )

    print("  Train target trajectory generated:")
    for process_name, data in target_trajectory_train.items():
        print(f"    {process_name}: inputs={data['inputs'].shape}, outputs={data['outputs'].shape}")
        if 'structural_conditions' in data and data['structural_conditions']:
            for var, vals in data['structural_conditions'].items():
                print(f"      {var}: [{vals.min():.2f}, {vals.max():.2f}] (range)")

    # Generate TRAIN baseline trajectory
    print("\n  Generating TRAIN baseline trajectory (a')...")
    baseline_trajectory_train = generate_baseline_trajectory(
        process_configs=selected_processes,
        target_trajectory=target_trajectory_train,
        n_samples=n_train,
        seed=CONTROLLER_CONFIG['scenarios']['seed_baseline']
    )

    print("  Train baseline trajectory generated:")
    for process_name, data in baseline_trajectory_train.items():
        print(f"    {process_name}: inputs={data['inputs'].shape}, outputs={data['outputs'].shape}")

    # 2. Generate TEST scenarios (for future evaluation, not used yet)
    print("\n[2/9] Generating test scenarios (not used in training)...")

    # Use seed offset from config to ensure test scenarios are truly unseen
    test_seed_offset = CONTROLLER_CONFIG['scenarios']['test_seed_offset']

    print(f"  Test seed offset: {test_seed_offset}")
    print(f"  Generating TEST target trajectory (a*)...")
    target_trajectory_test = generate_target_trajectory(
        process_configs=selected_processes,
        n_samples=n_test,
        seed=CONTROLLER_CONFIG['scenarios']['seed_target'] + test_seed_offset
    )

    print("  Test target trajectory generated:")
    for process_name, data in target_trajectory_test.items():
        print(f"    {process_name}: inputs={data['inputs'].shape}, outputs={data['outputs'].shape}")

    print(f"  Generating TEST baseline trajectory (a')...")
    baseline_trajectory_test = generate_baseline_trajectory(
        process_configs=selected_processes,
        target_trajectory=target_trajectory_test,
        n_samples=n_test,
        seed=CONTROLLER_CONFIG['scenarios']['seed_baseline'] + test_seed_offset
    )

    print("  Test baseline trajectory generated:")
    for process_name, data in baseline_trajectory_test.items():
        print(f"    {process_name}: inputs={data['inputs'].shape}, outputs={data['outputs'].shape}")

    # Use ONLY training trajectories for now
    target_trajectory = target_trajectory_train
    baseline_trajectory = baseline_trajectory_train
    n_scenarios = n_train

    # Save test scenarios for future evaluation
    test_scenarios_path = checkpoint_dir / 'test_scenarios.npz'
    test_data_to_save = {}
    for process_name in target_trajectory_test.keys():
        test_data_to_save[f'target_{process_name}_inputs'] = target_trajectory_test[process_name]['inputs']
        test_data_to_save[f'target_{process_name}_outputs'] = target_trajectory_test[process_name]['outputs']
        test_data_to_save[f'baseline_{process_name}_inputs'] = baseline_trajectory_test[process_name]['inputs']
        test_data_to_save[f'baseline_{process_name}_outputs'] = baseline_trajectory_test[process_name]['outputs']

    np.savez(test_scenarios_path, **test_data_to_save)

    print(f"\n  Using {n_scenarios} TRAIN scenarios for controller training")
    print(f"  Test scenarios saved: {test_scenarios_path}")

    # 3. Create ProcessChain (uses multi-scenario trajectory)
    print("\n[3/9] Building process chain...")
    try:
        process_chain = ProcessChain(
            processes_config=selected_processes,
            target_trajectory=target_trajectory,
            policy_config=CONTROLLER_CONFIG['policy_generator'],
            device=device
        )
        print(f"  ✓ Process chain created")
        print(f"    SCM functions: {len(process_chain.scm_functions)} (deterministic)")
        print(f"    Policy generators: {len(process_chain.policy_generators)} (trainable)")

        # Count parameters
        total_params = sum(p.numel() for p in process_chain.parameters())
        trainable_params = sum(p.numel() for p in process_chain.parameters() if p.requires_grad)
        print(f"    Total parameters: {total_params:,}")
        print(f"    Trainable parameters: {trainable_params:,}")

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease run train_processes.py first to train uncertainty predictors.")
        return

    # 4. Create Surrogate (computes F* for all scenarios)
    print("\n[4/9] Initializing surrogate model...")
    use_deterministic_sampling = CONTROLLER_CONFIG.get('surrogate', {}).get('use_deterministic_sampling', True)
    surrogate = ProTSurrogate(
        target_trajectory=target_trajectory,
        device=device,
        use_deterministic_sampling=use_deterministic_sampling
    )
    print(f"  Sampling mode: {'DETERMINISTIC (mean)' if use_deterministic_sampling else 'STOCHASTIC (reparameterization trick)'}")
    F_star_array = surrogate.F_star  # Now an array of (n_scenarios,)
    F_star_mean = np.mean(F_star_array)
    F_star_std = np.std(F_star_array)
    print(f"  ✓ Surrogate initialized")
    print(f"    F* per scenario computed for {n_scenarios} scenarios")
    print(f"    F* (mean): {F_star_mean:.6f} ± {F_star_std:.6f}")
    print(f"    F* (range): [{F_star_array.min():.6f}, {F_star_array.max():.6f}]")

    # 5. Create Trainer
    print("\n[5/9] Creating controller trainer...")
    trainer = ControllerTrainer(
        process_chain=process_chain,
        surrogate=surrogate,
        lambda_bc=CONTROLLER_CONFIG['training']['lambda_bc'],
        learning_rate=CONTROLLER_CONFIG['training']['learning_rate'],
        weight_decay=CONTROLLER_CONFIG['training']['weight_decay'],
        reliability_loss_scale=CONTROLLER_CONFIG['training']['reliability_loss_scale'],
        device=device
    )

    # 6. Training
    print("\n[6/9] Starting training...")
    print("-"*70)

    history = trainer.train(
        epochs=CONTROLLER_CONFIG['training']['epochs'],
        batch_size=CONTROLLER_CONFIG['training']['batch_size'],
        patience=CONTROLLER_CONFIG['training']['patience'],
        save_dir=checkpoint_dir,
        verbose=True
    )

    # 7. FINAL EVALUATION ACROSS ALL SCENARIOS
    print("\n[7/9] Final evaluation across all scenarios...")
    print("-"*70)

    # Load best model before evaluation
    print("  Loading best model...")
    trainer.load_checkpoint(checkpoint_dir)

    # Determine batch size for evaluation (use same as training)
    eval_batch_size = CONTROLLER_CONFIG['training']['batch_size']

    # Get plotting option from config
    plot_all_samples = CONTROLLER_CONFIG['report'].get('plot_all_batch_samples', True)

    # Evaluate controller on all scenarios
    mode_str = f"{n_scenarios} scenarios × {eval_batch_size} samples" if plot_all_samples else f"{n_scenarios} scenarios (aggregated)"
    print(f"  Evaluating controller on {mode_str}...")
    eval_results = trainer.evaluate_all_scenarios(
        batch_size=eval_batch_size,
        per_sample=plot_all_samples
    )

    F_actual_per_sample = eval_results['F_actual_per_sample']
    F_actual_mean = eval_results['F_actual_mean']
    F_actual_std = eval_results['F_actual_std']

    print(f"  Total samples evaluated: {len(F_actual_per_sample)}")

    # Compute baseline reliability for all scenarios (× samples if plotting all)
    mode_str = f"{n_scenarios} scenarios × {eval_batch_size} samples" if plot_all_samples else f"{n_scenarios} scenarios"
    print(f"  Computing baseline reliability for {mode_str}...")
    F_baseline_values = []
    F_star_values = []

    with torch.no_grad():
        for scenario_idx in range(n_scenarios):
            # Extract scenario from baseline and target trajectories
            baseline_scenario = {}
            target_scenario = {}

            for process_name, data in baseline_trajectory.items():
                baseline_scenario[process_name] = {
                    'inputs': data['inputs'][scenario_idx:scenario_idx+1],
                    'outputs': data['outputs'][scenario_idx:scenario_idx+1]
                }

            for process_name, data in target_trajectory.items():
                target_scenario[process_name] = {
                    'inputs': data['inputs'][scenario_idx:scenario_idx+1],
                    'outputs': data['outputs'][scenario_idx:scenario_idx+1]
                }

            # Convert to tensors
            baseline_scenario_tensor = convert_numpy_to_tensor(baseline_scenario, device=device)
            target_scenario_tensor = convert_numpy_to_tensor(target_scenario, device=device)

            # Compute reliability for baseline and target
            F_baseline_i = surrogate.compute_reliability(baseline_scenario_tensor).item()
            F_star_i = surrogate.compute_reliability(target_scenario_tensor).item()

            # If plotting all samples, replicate for each sample in the batch
            # Otherwise, just add once per scenario
            n_replicas = eval_batch_size if plot_all_samples else 1
            for _ in range(n_replicas):
                F_baseline_values.append(F_baseline_i)
                F_star_values.append(F_star_i)

    F_baseline_array = np.array(F_baseline_values)
    F_baseline_mean = np.mean(F_baseline_array)
    F_baseline_std = np.std(F_baseline_array)

    F_star_array = np.array(F_star_values)
    F_star_mean = np.mean(F_star_array)
    F_star_std = np.std(F_star_array)

    print(f"  Total baseline samples: {len(F_baseline_array)}")
    print(f"  Total target samples: {len(F_star_array)}")

    # Aggregate final metrics
    improvement = (F_actual_mean - F_baseline_mean) / abs(F_baseline_mean) * 100 if F_baseline_mean != 0 else 0
    target_gap = abs(F_star_mean - F_actual_mean) / F_star_mean * 100 if F_star_mean != 0 else 0

    # Print summary
    print("\n" + "="*70)
    print("FINAL RESULTS - AGGREGATED OVER ALL SAMPLES")
    print("="*70)
    print(f"Number of scenarios:           {n_scenarios}")
    print(f"Samples per scenario:          {eval_batch_size}")
    print(f"Total samples:                 {len(F_actual_per_sample)}")
    print(f"\nF* (target, optimal):")
    print(f"  Mean:  {F_star_mean:.6f} ± {F_star_std:.6f}")
    print(f"  Range: [{F_star_array.min():.6f}, {F_star_array.max():.6f}]")
    print(f"\nF' (baseline, no controller):")
    print(f"  Mean:  {F_baseline_mean:.6f} ± {F_baseline_std:.6f}")
    print(f"  Range: [{F_baseline_array.min():.6f}, {F_baseline_array.max():.6f}]")
    print(f"\nF  (actual, with controller):")
    print(f"  Mean:  {F_actual_mean:.6f} ± {F_actual_std:.6f}")
    print(f"  Range: [{F_actual_per_sample.min():.6f}, {F_actual_per_sample.max():.6f}]")
    print(f"\nImprovement over baseline:     {improvement:+.2f}%")
    print(f"Gap from optimal:              {target_gap:.2f}%")
    print(f"Robustness (std of F):         {F_actual_std:.6f}  (lower = more robust)")
    print("="*70)

    # 7b. Evaluate on TEST scenarios
    print("\n[7b/9] Evaluating on TEST scenarios...")
    print("-"*70)

    # Create temporary ProcessChain for test scenarios
    print(f"  Creating process chain for test scenarios...")
    process_chain_test = ProcessChain(
        processes_config=selected_processes,
        target_trajectory=target_trajectory_test,
        policy_config=CONTROLLER_CONFIG['policy_generator'],
        device=device
    )

    # Load trained policy generators into test chain
    for process_idx in range(len(process_chain.policy_generators)):
        process_chain_test.policy_generators[process_idx].load_state_dict(
            process_chain.policy_generators[process_idx].state_dict()
        )

    # Evaluate test scenarios
    F_star_test_values = []
    F_baseline_test_values = []
    F_actual_test_values = []

    print(f"  Evaluating on {n_test} test scenarios (never seen during training)...")

    with torch.no_grad():
        for scenario_idx in range(n_test):
            # Extract scenario from test trajectories
            target_test_scenario = {}
            baseline_test_scenario = {}

            for process_name, data in target_trajectory_test.items():
                target_test_scenario[process_name] = {
                    'inputs': data['inputs'][scenario_idx:scenario_idx+1],
                    'outputs': data['outputs'][scenario_idx:scenario_idx+1]
                }
                baseline_test_scenario[process_name] = {
                    'inputs': baseline_trajectory_test[process_name]['inputs'][scenario_idx:scenario_idx+1],
                    'outputs': baseline_trajectory_test[process_name]['outputs'][scenario_idx:scenario_idx+1]
                }

            # Convert to tensors
            target_test_tensor = convert_numpy_to_tensor(target_test_scenario, device=device)
            baseline_test_tensor = convert_numpy_to_tensor(baseline_test_scenario, device=device)

            # Compute F_star and F_baseline for this test scenario
            F_star_test_i = surrogate.compute_reliability(target_test_tensor).item()
            F_baseline_test_i = surrogate.compute_reliability(baseline_test_tensor).item()

            F_star_test_values.append(F_star_test_i)
            F_baseline_test_values.append(F_baseline_test_i)

            # Run controller on test scenario
            actual_test_trajectory = process_chain_test.forward(batch_size=1, scenario_idx=scenario_idx)
            F_actual_test_i = surrogate.compute_reliability(actual_test_trajectory).item()
            F_actual_test_values.append(F_actual_test_i)

    F_star_test_array = np.array(F_star_test_values)
    F_baseline_test_array = np.array(F_baseline_test_values)
    F_actual_test_array = np.array(F_actual_test_values)

    F_star_test_mean = np.mean(F_star_test_array)
    F_baseline_test_mean = np.mean(F_baseline_test_array)
    F_actual_test_mean = np.mean(F_actual_test_array)

    improvement_test = (F_actual_test_mean - F_baseline_test_mean) / abs(F_baseline_test_mean) * 100 if F_baseline_test_mean != 0 else 0

    print(f"\nTest Results:")
    print(f"  F* (test):        {F_star_test_mean:.6f}")
    print(f"  F' (test):        {F_baseline_test_mean:.6f}")
    print(f"  F  (test):        {F_actual_test_mean:.6f}")
    print(f"  Improvement:      {improvement_test:+.2f}%")

    # 7c. Compute advanced metrics
    print("\n[7c/9] Computing advanced metrics...")
    print("-"*70)

    # Compute per-scenario means for metrics
    if plot_all_samples:
        # If plotting all samples: F_actual_per_sample has shape (n_scenarios * batch_size,)
        # Reshape to (n_scenarios, batch_size) and take mean along batch dimension
        F_actual_per_scenario_mean = F_actual_per_sample.reshape(n_scenarios, eval_batch_size).mean(axis=1)
        F_star_per_scenario_mean = F_star_array.reshape(n_scenarios, eval_batch_size).mean(axis=1)
        F_baseline_per_scenario_mean = F_baseline_array.reshape(n_scenarios, eval_batch_size).mean(axis=1)
    else:
        # If using aggregated values: arrays already have shape (n_scenarios,)
        F_actual_per_scenario_mean = F_actual_per_sample
        F_star_per_scenario_mean = F_star_array
        F_baseline_per_scenario_mean = F_baseline_array

    # Get success rate threshold from config
    success_threshold = CONTROLLER_CONFIG['metrics']['success_rate_threshold']

    # Worst-case gap (train and test) - using scenario-level aggregates
    worst_case_train = compute_worst_case_gap(F_star_per_scenario_mean, F_actual_per_scenario_mean)
    worst_case_test = compute_worst_case_gap(F_star_test_array, F_actual_test_array)

    print(f"\nWorst-Case Gap:")
    print(f"  Train: {worst_case_train['worst_case_gap']:.6f} (scenario {worst_case_train['worst_case_scenario_idx']})")
    print(f"  Test:  {worst_case_test['worst_case_gap']:.6f} (scenario {worst_case_test['worst_case_scenario_idx']})")

    # Success rate (train and test) - using scenario-level aggregates
    success_rate_train = compute_success_rate(F_star_per_scenario_mean, F_actual_per_scenario_mean, threshold=success_threshold)
    success_rate_test = compute_success_rate(F_star_test_array, F_actual_test_array, threshold=success_threshold)

    print(f"\nSuccess Rate (threshold: {success_threshold*100:.0f}% of F_star):")
    print(f"  Train: {success_rate_train['success_rate_pct']:.1f}% ({success_rate_train['n_successful']}/{success_rate_train['n_total']} scenarios)")
    print(f"  Test:  {success_rate_test['success_rate_pct']:.1f}% ({success_rate_test['n_successful']}/{success_rate_test['n_total']} scenarios)")

    # Train-test gap - using scenario-level aggregates
    train_test_gap_metrics = compute_train_test_gap(F_star_per_scenario_mean, F_actual_per_scenario_mean,
                                                     F_star_test_array, F_actual_test_array)

    print(f"\nTrain-Test Gap:")
    print(f"  Mean gap (train): {train_test_gap_metrics['mean_gap_train']:.6f}")
    print(f"  Mean gap (test):  {train_test_gap_metrics['mean_gap_test']:.6f}")
    print(f"  Difference:       {train_test_gap_metrics['train_test_gap']:.6f}")
    if train_test_gap_metrics['train_test_gap'] > 0:
        print(f"    → Controller performs BETTER on test (generalizes well)")
    else:
        print(f"    → Controller performs WORSE on test (overfitting concern)")

    # Scenario diversity (train only, test separately)
    train_structural_conditions = {}
    test_structural_conditions = {}

    # Extract structural conditions from train scenarios
    for process_name, data in target_trajectory_train.items():
        if 'structural_conditions' in data and data['structural_conditions']:
            for var, vals in data['structural_conditions'].items():
                if var not in train_structural_conditions:
                    train_structural_conditions[var] = vals
                else:
                    train_structural_conditions[var] = np.concatenate([train_structural_conditions[var], vals])

    # Extract structural conditions from test scenarios
    for process_name, data in target_trajectory_test.items():
        if 'structural_conditions' in data and data['structural_conditions']:
            for var, vals in data['structural_conditions'].items():
                if var not in test_structural_conditions:
                    test_structural_conditions[var] = vals
                else:
                    test_structural_conditions[var] = np.concatenate([test_structural_conditions[var], vals])

    diversity_train = compute_scenario_diversity(train_structural_conditions)
    diversity_test = compute_scenario_diversity(test_structural_conditions)

    print(f"\nScenario Diversity Score:")
    print(f"  Train: {diversity_train['diversity_score']:.4f}")
    print(f"  Test:  {diversity_test['diversity_score']:.4f}")
    print(f"  Per-condition CV (train):")
    for var, cv in diversity_train['per_condition_cv'].items():
        stats = diversity_train['per_condition_stats'][var]
        print(f"    {var}: CV={cv:.4f}, mean={stats['mean']:.2f}, std={stats['std']:.2f}")

    # 8. Generate visualizations
    print("\n[8/9] Generating visualizations...")

    # Add F_star mean to history for plotting
    history['F_star'] = F_star_mean

    # Plot training history
    plot_training_history(
        history=history,
        save_path=str(checkpoint_dir / 'training_history.png')
    )

    # Plot reliability comparison (using mean values)
    plot_reliability_comparison(
        F_star=F_star_mean,
        F_baseline=F_baseline_mean,
        F_actual=F_actual_mean,
        save_path=str(checkpoint_dir / 'reliability_comparison.png')
    )

    # Plot trajectory comparison for representative scenario
    print("  Generating trajectory comparison plot...")

    # Select representative scenario (closest to mean F_actual)
    # F_actual_per_scenario_mean was already computed above for advanced metrics
    actual_trajectories = eval_results['trajectories']
    representative_idx = np.argmin(np.abs(F_actual_per_scenario_mean - F_actual_mean))

    print(f"    Using scenario {representative_idx} (F_mean={F_actual_per_scenario_mean[representative_idx]:.6f}, close to global mean {F_actual_mean:.6f})")

    # Extract representative scenario from target and baseline trajectories
    target_scenario = {}
    baseline_scenario = {}
    for process_name, data in target_trajectory.items():
        target_scenario[process_name] = {
            'inputs': data['inputs'][representative_idx:representative_idx+1],
            'outputs': data['outputs'][representative_idx:representative_idx+1]
        }

    for process_name, data in baseline_trajectory.items():
        baseline_scenario[process_name] = {
            'inputs': data['inputs'][representative_idx:representative_idx+1],
            'outputs': data['outputs'][representative_idx:representative_idx+1]
        }

    # Get actual trajectory for representative scenario (already a dict of tensors)
    # Note: actual_trajectories contains batches of eval_batch_size samples
    # Extract only the first sample for plotting (to match target and baseline)
    actual_scenario_batch = actual_trajectories[representative_idx]
    actual_scenario = {}
    for process_name, data in actual_scenario_batch.items():
        actual_scenario[process_name] = {
            'inputs': data['inputs'][0:1],
            'outputs': data['outputs'][0:1]
        }

    # Plot comparison
    plot_trajectory_comparison(
        target_trajectory=target_scenario,
        baseline_trajectory=baseline_scenario,
        actual_trajectory=actual_scenario,
        save_path=str(checkpoint_dir / 'trajectory_comparison.png')
    )

    print("  ✓ Basic visualizations generated")

    # 8a. Generate NEW advanced plots
    print("\n  Generating advanced plots...")

    # Scatter plot: Target vs Baseline & Actual (train) - ALL SAMPLES
    plot_target_vs_actual_scatter(
        F_star_per_scenario=F_star_array,
        F_baseline_per_scenario=F_baseline_array,
        F_actual_per_scenario=F_actual_per_sample,
        save_path=str(checkpoint_dir / 'target_vs_actual_scatter_train.png')
    )

    # Scatter plot: Target vs Baseline & Actual (test)
    plot_target_vs_actual_scatter(
        F_star_per_scenario=F_star_test_array,
        F_baseline_per_scenario=F_baseline_test_array,
        F_actual_per_scenario=F_actual_test_array,
        save_path=str(checkpoint_dir / 'target_vs_actual_scatter_test.png')
    )

    # Gap distribution (train) - ALL SAMPLES
    plot_gap_distribution(
        F_star_per_scenario=F_star_array,
        F_actual_per_scenario=F_actual_per_sample,
        save_path=str(checkpoint_dir / 'gap_distribution_train.png')
    )

    # Gap distribution (test)
    plot_gap_distribution(
        F_star_per_scenario=F_star_test_array,
        F_actual_per_scenario=F_actual_test_array,
        save_path=str(checkpoint_dir / 'gap_distribution_test.png')
    )

    print("  ✓ Advanced visualizations generated")

    # 8b. Generate PDF report (if enabled)
    if CONTROLLER_CONFIG['report']['generate_pdf']:
        print("\n  Generating PDF report...")

        # Prepare F values in dict format for multi-scenario
        F_star_dict = {
            'mean': float(F_star_mean),
            'std': float(F_star_std),
            'min': float(F_star_array.min()),
            'max': float(F_star_array.max())
        }

        F_baseline_dict = {
            'mean': float(F_baseline_mean),
            'std': float(F_baseline_std),
            'min': float(F_baseline_array.min()),
            'max': float(F_baseline_array.max())
        }

        F_actual_dict = {
            'mean': float(F_actual_mean),
            'std': float(F_actual_std),
            'min': float(F_actual_per_sample.min()),
            'max': float(F_actual_per_sample.max())
        }

        # Prepare final metrics for report
        report_final_metrics = {
            'improvement': improvement / 100,  # Convert back to fraction
            'target_gap': target_gap / 100
        }

        # Process-wise metrics not available for multi-scenario (optional)
        process_metrics = {}

        # Prepare advanced metrics for report
        advanced_metrics_for_report = {
            'worst_case_gap_train': worst_case_train,
            'worst_case_gap_test': worst_case_test,
            'success_rate_train': success_rate_train,
            'success_rate_test': success_rate_test,
            'train_test_gap': train_test_gap_metrics,
            'diversity_train': diversity_train,
            'diversity_test': diversity_test,
        }

        # Generate embedding visualization plots (if scenario encoder is enabled)
        print("\n[8.5/9] Generating embedding visualizations...")
        embedding_plots = {}

        # Check if scenario encoder is enabled in config
        use_scenario_encoder = CONTROLLER_CONFIG['policy_generator'].get('use_scenario_encoder', False)

        if not use_scenario_encoder:
            print("  ⚠ Scenario encoder is disabled in config. Skipping embedding plots.")

            # Remove old embedding plots if they exist (to prevent them from appearing in report)
            old_embedding_plots = [
                checkpoint_dir / 'embedding_tsne.png',
                checkpoint_dir / 'embedding_pca.png',
                checkpoint_dir / 'embedding_distances.png',
                checkpoint_dir / 'embedding_correlations.png',
                checkpoint_dir / 'embedding_evolution.png',
            ]
            for plot_path in old_embedding_plots:
                if plot_path.exists():
                    try:
                        plot_path.unlink()
                        print(f"    Removed old embedding plot: {plot_path.name}")
                    except Exception as e:
                        print(f"    Warning: Could not remove {plot_path.name}: {e}")
        else:
            try:
                from controller_optimization.src.utils.embedding_visualization import generate_all_embedding_plots

                # Load embedding data
                embedding_path = checkpoint_dir / 'embeddings.json'
                embedding_history_path = checkpoint_dir / 'embedding_history.npz'

                if embedding_path.exists():
                    with open(embedding_path, 'r') as f:
                        embedding_data = json.load(f)

                    embeddings = np.array(embedding_data['embeddings'])
                    structural_params = np.array(embedding_data['structural_params'])
                    scenario_indices = np.array(embedding_data['scenario_indices'])

                    # Check if we have enough scenarios for visualization
                    # t-SNE requires n_samples > perplexity (default perplexity=5)
                    if len(scenario_indices) < 2:
                        print(f"  ⚠ Not enough scenarios ({len(scenario_indices)}) for embedding visualization (need at least 2)")
                    else:
                        # Load embedding history if available
                        embedding_history = {}
                        if embedding_history_path.exists():
                            history_data = np.load(embedding_history_path)
                            for key in history_data.files:
                                epoch_num = int(key.split('_')[1])
                                embedding_history[epoch_num] = history_data[key]

                        # Determine parameter names from processes_config
                        param_names = []
                        for process_config in selected_processes:
                            from controller_optimization.configs.processes_config import get_controllable_inputs
                            input_labels = process_config['input_labels']
                            controllable = get_controllable_inputs(process_config)
                            for label in input_labels:
                                if label not in controllable:
                                    param_names.append(f"{process_config['name']}.{label}")

                        # Generate all embedding plots
                        embedding_plots = generate_all_embedding_plots(
                            embeddings=embeddings,
                            structural_params=structural_params,
                            scenario_indices=scenario_indices,
                            checkpoint_dir=checkpoint_dir,
                            embedding_history=embedding_history if len(embedding_history) > 0 else None,
                            param_names=param_names if len(param_names) > 0 else None
                        )
                        print(f"  ✓ Generated {len(embedding_plots)} embedding plots")
                else:
                    print("  ⚠ No embedding data found (scenario encoder may be disabled)")
            except Exception as e:
                print(f"  ✗ Warning: Failed to generate embedding plots: {e}")
                import traceback
                traceback.print_exc()

        # Generate report
        try:
            report_path = generate_controller_report(
                config=CONTROLLER_CONFIG,
                training_history=history,
                final_metrics=report_final_metrics,
                process_metrics=process_metrics,
                F_star=F_star_dict,
                F_baseline=F_baseline_dict,
                F_actual=F_actual_dict,
                checkpoint_dir=checkpoint_dir,
                timestamp=datetime.now(),
                n_scenarios=n_scenarios,
                advanced_metrics=advanced_metrics_for_report
            )
            print(f"  ✓ PDF report generated: {report_path}")
        except Exception as e:
            print(f"  ✗ Warning: Failed to generate PDF report: {e}")
            print(f"    Continuing without report...")

    # 9. Save all metrics to JSON
    print("\n[9/9] Saving final results...")

    # Convert history values to lists for JSON serialization
    history_serializable = {k: [float(v) for v in vals] for k, vals in history.items() if isinstance(vals, list)}
    history_serializable['F_star'] = float(F_star_mean)

    final_results = {
        'timestamp': datetime.now().isoformat(),
        'config': CONTROLLER_CONFIG,
        'n_train_scenarios': int(n_scenarios),
        'n_test_scenarios': int(n_test),

        # TRAIN metrics - Aggregated
        'train': {
            'F_star_mean': float(F_star_mean),
            'F_star_std': float(F_star_std),
            'F_baseline_mean': float(F_baseline_mean),
            'F_baseline_std': float(F_baseline_std),
            'F_actual_mean': float(F_actual_mean),
            'F_actual_std': float(F_actual_std),
            'improvement_pct': float(improvement),
            'target_gap_pct': float(target_gap),
            'robustness_std': float(F_actual_std),
        },

        # TRAIN metrics - Per sample (n_scenarios × batch_size)
        'train_per_sample': {
            'F_star': F_star_array.tolist(),
            'F_baseline': F_baseline_array.tolist(),
            'F_actual': F_actual_per_sample.tolist(),
            'batch_size': int(eval_batch_size),
        },

        # TRAIN metrics - Per scenario (aggregated means)
        'train_per_scenario_mean': {
            'F_star': F_star_per_scenario_mean.tolist(),
            'F_baseline': F_baseline_per_scenario_mean.tolist(),
            'F_actual': F_actual_per_scenario_mean.tolist(),
        },

        # TEST metrics - Aggregated
        'test': {
            'F_star_mean': float(F_star_test_mean),
            'F_baseline_mean': float(F_baseline_test_mean),
            'F_actual_mean': float(F_actual_test_mean),
            'improvement_pct': float(improvement_test),
        },

        # TEST metrics - Per scenario
        'test_per_scenario': {
            'F_star': F_star_test_array.tolist(),
            'F_baseline': F_baseline_test_array.tolist(),
            'F_actual': F_actual_test_array.tolist(),
        },

        # ADVANCED metrics
        'advanced_metrics': {
            # Worst-case gap
            'worst_case_gap_train': worst_case_train,
            'worst_case_gap_test': worst_case_test,

            # Success rate
            'success_rate_train': success_rate_train,
            'success_rate_test': success_rate_test,

            # Train-test gap
            'train_test_gap': train_test_gap_metrics,

            # Scenario diversity
            'diversity_train': diversity_train,
            'diversity_test': diversity_test,
        },

        # Training history
        'history': history_serializable,
    }

    results_path = checkpoint_dir / 'final_results.json'
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"  ✓ Final results saved: {results_path}")

    print("\n" + "="*70)
    print("MULTI-SCENARIO CONTROLLER TRAINING COMPLETED!")
    print("="*70)
    print(f"\nFiles saved in: {checkpoint_dir}/")
    print("  - policy_*.pth                     : Policy generators")
    print("  - training_history.json            : Training history")
    print("  - final_results.json               : All metrics (with per-scenario data)")
    print("  - *.png                            : Visualization plots")
    print(f"\nController trained on {n_scenarios} diverse scenarios")
    print(f"  → Generalizes across varying structural conditions")
    print(f"  → Robustness: {F_actual_std:.6f} (std across scenarios)")
    print(f"  → Mean improvement: {improvement:+.2f}%")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
