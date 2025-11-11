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

    checkpoint_dir = Path(CONTROLLER_CONFIG['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate target trajectory (a*)
    print("\n[1/9] Generating target trajectory (a*, noise=0)...")
    target_trajectory = generate_target_trajectory(
        process_configs=PROCESSES,
        n_samples=CONTROLLER_CONFIG['target']['n_samples'],
        seed=CONTROLLER_CONFIG['target']['seed']
    )

    print("  Target trajectory generated:")
    for process_name, data in target_trajectory.items():
        print(f"    {process_name}: inputs={data['inputs'].shape}, outputs={data['outputs'].shape}")

    # 2. Generate baseline trajectory (a')
    print("\n[2/9] Generating baseline trajectory (a', normal noise, NO controller)...")
    baseline_trajectory = generate_baseline_trajectory(
        process_configs=PROCESSES,
        n_samples=CONTROLLER_CONFIG['baseline']['n_samples'],
        seed=CONTROLLER_CONFIG['baseline']['seed']
    )

    print("  Baseline trajectory generated:")
    for process_name, data in baseline_trajectory.items():
        print(f"    {process_name}: inputs={data['inputs'].shape}, outputs={data['outputs'].shape}")

    # 3. Create ProcessChain
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

    # 4. Create Surrogate
    print("\n[4/9] Initializing surrogate model...")
    surrogate = ProTSurrogate(
        target_trajectory=target_trajectory,
        device=device
    )
    F_star = surrogate.F_star
    print(f"  ✓ Surrogate initialized")
    print(f"    F* (target reliability): {F_star:.6f}")

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
        n_batches_per_epoch=CONTROLLER_CONFIG['training']['n_batches_per_epoch'],
        batch_size=CONTROLLER_CONFIG['training']['batch_size'],
        patience=CONTROLLER_CONFIG['training']['patience'],
        save_dir=checkpoint_dir,
        verbose=True
    )

    # 7. FINAL EVALUATION
    print("\n[7/9] Final evaluation...")
    print("-"*70)

    # Generate actual trajectory con policy generators
    process_chain.eval()
    with torch.no_grad():
        actual_trajectory_tensor = process_chain.forward(batch_size=1)

    # Convert to numpy for metrics
    actual_trajectory = convert_trajectory_to_numpy(actual_trajectory_tensor)

    # Calculate reliability scores
    baseline_trajectory_tensor = convert_numpy_to_tensor(baseline_trajectory, device=device)

    with torch.no_grad():
        F_baseline = surrogate.compute_reliability(baseline_trajectory_tensor).item()
        F_actual = surrogate.compute_reliability(actual_trajectory_tensor).item()

    # Compute final metrics
    final_metrics = compute_final_metrics(
        target_trajectory=target_trajectory,
        baseline_trajectory=baseline_trajectory,
        actual_trajectory=actual_trajectory,
        F_star=F_star,
        F_baseline=F_baseline,
        F_actual=F_actual
    )

    # Print summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"F* (target, optimal):          {F_star:.6f}")
    print(f"F' (baseline, no controller):  {F_baseline:.6f}")
    print(f"F  (actual, with controller):  {F_actual:.6f}")
    print(f"\nImprovement over baseline:     {final_metrics['improvement_pct']:+.2f}%")
    print(f"Gap from optimal:              {final_metrics['target_gap_pct']:.2f}%")
    print("="*70)

    # Process-wise metrics
    process_metrics = final_metrics['process_metrics']

    # 8. Generate visualizations
    print("\n[8/9] Generating visualizations and report...")

    # Add F_star to history for plotting
    history['F_star'] = F_star

    plot_training_history(
        history=history,
        save_path=str(checkpoint_dir / 'training_history.png')
    )

    plot_trajectory_comparison(
        target_trajectory=target_trajectory,
        baseline_trajectory=baseline_trajectory,
        actual_trajectory=actual_trajectory,
        save_path=str(checkpoint_dir / 'trajectory_comparison.png')
    )

    plot_reliability_comparison(
        F_star=F_star,
        F_baseline=F_baseline,
        F_actual=F_actual,
        save_path=str(checkpoint_dir / 'reliability_comparison.png')
    )

    plot_process_improvements(
        process_metrics=process_metrics,
        save_path=str(checkpoint_dir / 'process_improvements.png')
    )

    # Generate PDF report
    if CONTROLLER_CONFIG['report']['generate_pdf']:
        report_path = generate_controller_report(
            config=CONTROLLER_CONFIG,
            training_history=history,
            final_metrics=final_metrics,
            process_metrics=process_metrics,
            F_star=F_star,
            F_baseline=F_baseline,
            F_actual=F_actual,
            checkpoint_dir=checkpoint_dir,
            timestamp=datetime.now()
        )
        print(f"\n  ✓ Controller report saved: {report_path}")

    # 9. Save all metrics to JSON
    print("\n[9/9] Saving final results...")

    # Convert history values to lists for JSON serialization
    history_serializable = {k: [float(v) for v in vals] for k, vals in history.items() if isinstance(vals, list)}
    history_serializable['F_star'] = float(F_star)

    final_results = {
        'timestamp': datetime.now().isoformat(),
        'config': CONTROLLER_CONFIG,
        'F_star': float(F_star),
        'F_baseline': float(F_baseline),
        'F_actual': float(F_actual),
        'final_metrics': final_metrics,
        'process_metrics': process_metrics,
        'history': history_serializable,
    }

    results_path = checkpoint_dir / 'final_results.json'
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"  ✓ Final results saved: {results_path}")

    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nFiles saved in: {checkpoint_dir}/")
    print("  - policy_*.pth                     : Policy generators")
    print("  - training_history.json            : Training history")
    print("  - final_results.json               : All metrics")
    print("  - controller_report.pdf            : Complete PDF report")
    print("  - *.png                            : Comparison plots")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
