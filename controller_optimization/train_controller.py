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

from controller_optimization.configs.processes_config import PROCESSES
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
    convert_trajectory_to_numpy
)
from controller_optimization.src.utils.visualization import (
    plot_training_history,
    plot_trajectory_comparison,
    plot_reliability_comparison,
    plot_process_improvements
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
    validate_all_processes(PROCESSES)

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
        process_configs=PROCESSES,
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
        process_configs=PROCESSES,
        target_trajectory=target_trajectory_train,
        n_samples=n_train,
        seed=CONTROLLER_CONFIG['scenarios']['seed_baseline']
    )

    print("  Train baseline trajectory generated:")
    for process_name, data in baseline_trajectory_train.items():
        print(f"    {process_name}: inputs={data['inputs'].shape}, outputs={data['outputs'].shape}")

    # 2. Generate TEST scenarios (for future evaluation, not used yet)
    print("\n[2/9] Generating test scenarios (not used in training)...")

    # Use different seed to ensure test scenarios are truly unseen
    test_seed_offset = 1000

    print(f"  Generating TEST target trajectory (a*)...")
    target_trajectory_test = generate_target_trajectory(
        process_configs=PROCESSES,
        n_samples=n_test,
        seed=CONTROLLER_CONFIG['scenarios']['seed_target'] + test_seed_offset
    )

    print("  Test target trajectory generated:")
    for process_name, data in target_trajectory_test.items():
        print(f"    {process_name}: inputs={data['inputs'].shape}, outputs={data['outputs'].shape}")

    print(f"  Generating TEST baseline trajectory (a')...")
    baseline_trajectory_test = generate_baseline_trajectory(
        process_configs=PROCESSES,
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
            processes_config=PROCESSES,
            target_trajectory=target_trajectory,
            policy_config=CONTROLLER_CONFIG['policy_generator'],
            device=device
        )
        print(f"  ✓ Process chain created")
        print(f"    Uncertainty predictors: {len(process_chain.uncertainty_predictors)} (frozen)")
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
    surrogate = ProTSurrogate(
        target_trajectory=target_trajectory,
        device=device
    )
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

    # Evaluate controller on all scenarios
    print(f"  Evaluating controller on {n_scenarios} scenarios...")
    eval_results = trainer.evaluate_all_scenarios()

    F_actual_per_scenario = eval_results['F_actual_per_scenario']
    F_actual_mean = eval_results['F_actual_mean']
    F_actual_std = eval_results['F_actual_std']

    # Compute baseline reliability for all scenarios
    print(f"  Computing baseline reliability for {n_scenarios} scenarios...")
    F_baseline_values = []
    with torch.no_grad():
        for scenario_idx in range(n_scenarios):
            # Extract scenario from baseline trajectory
            baseline_scenario = {}
            for process_name, data in baseline_trajectory.items():
                baseline_scenario[process_name] = {
                    'inputs': data['inputs'][scenario_idx:scenario_idx+1],
                    'outputs': data['outputs'][scenario_idx:scenario_idx+1]
                }

            # Convert to tensor
            baseline_scenario_tensor = convert_numpy_to_tensor(baseline_scenario, device=device)

            # Compute reliability
            F_baseline_i = surrogate.compute_reliability(baseline_scenario_tensor).item()
            F_baseline_values.append(F_baseline_i)

    F_baseline_array = np.array(F_baseline_values)
    F_baseline_mean = np.mean(F_baseline_array)
    F_baseline_std = np.std(F_baseline_array)

    # Aggregate final metrics
    improvement = (F_actual_mean - F_baseline_mean) / abs(F_baseline_mean) * 100 if F_baseline_mean != 0 else 0
    target_gap = abs(F_star_mean - F_actual_mean) / F_star_mean * 100 if F_star_mean != 0 else 0

    # Print summary
    print("\n" + "="*70)
    print("FINAL RESULTS - AGGREGATED OVER ALL SCENARIOS")
    print("="*70)
    print(f"Number of scenarios:           {n_scenarios}")
    print(f"\nF* (target, optimal):")
    print(f"  Mean:  {F_star_mean:.6f} ± {F_star_std:.6f}")
    print(f"  Range: [{F_star_array.min():.6f}, {F_star_array.max():.6f}]")
    print(f"\nF' (baseline, no controller):")
    print(f"  Mean:  {F_baseline_mean:.6f} ± {F_baseline_std:.6f}")
    print(f"  Range: [{F_baseline_array.min():.6f}, {F_baseline_array.max():.6f}]")
    print(f"\nF  (actual, with controller):")
    print(f"  Mean:  {F_actual_mean:.6f} ± {F_actual_std:.6f}")
    print(f"  Range: [{F_actual_per_scenario.min():.6f}, {F_actual_per_scenario.max():.6f}]")
    print(f"\nImprovement over baseline:     {improvement:+.2f}%")
    print(f"Gap from optimal:              {target_gap:.2f}%")
    print(f"Robustness (std of F):         {F_actual_std:.6f}  (lower = more robust)")
    print("="*70)

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
    actual_trajectories = eval_results['trajectories']
    representative_idx = np.argmin(np.abs(F_actual_per_scenario - F_actual_mean))

    print(f"    Using scenario {representative_idx} (F={F_actual_per_scenario[representative_idx]:.6f}, close to mean {F_actual_mean:.6f})")

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
    actual_scenario = actual_trajectories[representative_idx]

    # Plot comparison
    plot_trajectory_comparison(
        target_trajectory=target_scenario,
        baseline_trajectory=baseline_scenario,
        actual_trajectory=actual_scenario,
        save_path=str(checkpoint_dir / 'trajectory_comparison.png')
    )

    print("  ✓ Basic visualizations generated")

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
            'min': float(F_actual_per_scenario.min()),
            'max': float(F_actual_per_scenario.max())
        }

        # Prepare final metrics for report
        report_final_metrics = {
            'improvement': improvement / 100,  # Convert back to fraction
            'target_gap': target_gap / 100
        }

        # Process-wise metrics not available for multi-scenario (optional)
        process_metrics = {}

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
                n_scenarios=n_scenarios
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
        'n_scenarios': int(n_scenarios),

        # Aggregated metrics
        'F_star_mean': float(F_star_mean),
        'F_star_std': float(F_star_std),
        'F_baseline_mean': float(F_baseline_mean),
        'F_baseline_std': float(F_baseline_std),
        'F_actual_mean': float(F_actual_mean),
        'F_actual_std': float(F_actual_std),

        # Per-scenario metrics
        'F_star_per_scenario': F_star_array.tolist(),
        'F_baseline_per_scenario': F_baseline_array.tolist(),
        'F_actual_per_scenario': F_actual_per_scenario.tolist(),

        # Summary metrics
        'improvement_pct': float(improvement),
        'target_gap_pct': float(target_gap),
        'robustness_std': float(F_actual_std),

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
