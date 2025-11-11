"""
Step 1: Addestra uncertainty predictor per ogni processo.

Usa: python train_processes.py [--processes laser plasma] [--device cuda]

Output:
- Checkpoint per ogni processo in checkpoints/{process_name}/
- Report PDF per ogni processo
- Summary table con tutte le metriche
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime

# Add controller_optimization to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from controller_optimization.configs.processes_config import PROCESSES, get_process_by_name
from controller_optimization.src.training.process_trainer import train_single_process


def main():
    """
    Per ogni processo in PROCESSES:
    1. Controlla se checkpoint già esiste (skip se già addestrato)
    2. Chiama train_single_process()
    3. Raccoglie metriche
    4. Genera summary finale
    """
    parser = argparse.ArgumentParser(description='Train uncertainty predictors for all processes')
    parser.add_argument(
        '--processes',
        nargs='+',
        default=None,
        help='Specific processes to train (e.g., laser plasma). If not specified, trains all.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for training'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip processes that already have checkpoints'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Determine which processes to train
    if args.processes is None:
        processes_to_train = PROCESSES
        process_names = [p['name'] for p in PROCESSES]
    else:
        process_names = args.processes
        processes_to_train = [get_process_by_name(name) for name in process_names]

    print("="*70)
    print("CONTROLLER OPTIMIZATION - STEP 1: TRAIN UNCERTAINTY PREDICTORS")
    print("="*70)
    print(f"\nProcesses to train: {', '.join(process_names)}")
    print(f"Device: {args.device}")
    print(f"Skip existing: {args.skip_existing}")
    print(f"Random seed: {args.seed}")

    # Storage for results
    all_results = {}
    summary_data = []

    # Train each process
    for i, process_config in enumerate(processes_to_train, 1):
        process_name = process_config['name']

        print(f"\n{'='*70}")
        print(f"Process {i}/{len(processes_to_train)}: {process_name.upper()}")
        print(f"{'='*70}")

        checkpoint_dir = Path(process_config['checkpoint_dir'])
        model_path = checkpoint_dir / 'uncertainty_predictor.pth'

        # Check if already trained
        if args.skip_existing and model_path.exists():
            print(f"\n⏭  Checkpoint already exists for '{process_name}'. Skipping...")

            # Load existing metrics
            info_path = checkpoint_dir / 'training_info.json'
            if info_path.exists():
                with open(info_path, 'r') as f:
                    info = json.load(f)
                    metrics = info.get('metrics', {})
                    report_path = str(checkpoint_dir / 'training_report.pdf')

                    all_results[process_name] = {
                        'metrics': metrics,
                        'report_path': report_path,
                    }

                    summary_data.append({
                        'process': process_name,
                        'mse': metrics.get('mse', 0),
                        'rmse': metrics.get('rmse', 0),
                        'mae': metrics.get('mae', 0),
                        'r2': metrics.get('r2', 0),
                        'calibration_ratio': metrics.get('calibration_ratio', 0),
                        'report_path': report_path,
                    })
            continue

        # Train process
        try:
            result = train_single_process(
                process_config=process_config,
                device=args.device,
                verbose=True,
                seed=args.seed
            )

            all_results[process_name] = result

            # Add to summary
            summary_data.append({
                'process': process_name,
                'mse': result['metrics']['mse'],
                'rmse': result['metrics']['rmse'],
                'mae': result['metrics']['mae'],
                'r2': result['metrics']['r2'],
                'calibration_ratio': result['metrics']['calibration_ratio'],
                'report_path': result['report_path'],
            })

            print(f"\n✓ Training completed for '{process_name}'")

        except Exception as e:
            print(f"\n✗ Error training '{process_name}': {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)

    if not summary_data:
        print("\nNo processes were trained.")
        return

    # Print summary table
    print(f"\n{'Process':<12} {'MSE':<10} {'RMSE':<10} {'MAE':<10} {'R²':<10} {'Cal.Ratio':<12}")
    print("-" * 70)

    for data in summary_data:
        print(
            f"{data['process']:<12} "
            f"{data['mse']:<10.6f} "
            f"{data['rmse']:<10.6f} "
            f"{data['mae']:<10.6f} "
            f"{data['r2']:<10.6f} "
            f"{data['calibration_ratio']:<12.6f}"
        )

    # Print report paths
    print("\n" + "="*70)
    print("GENERATED REPORTS")
    print("="*70)

    for data in summary_data:
        print(f"  {data['process']:<12}: {data['report_path']}")

    # Save summary to JSON
    summary_path = Path('controller_optimization/checkpoints/processes_training_summary.json')
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary_output = {
        'timestamp': datetime.now().isoformat(),
        'device': args.device,
        'seed': args.seed,
        'processes': summary_data,
    }

    with open(summary_path, 'w') as f:
        json.dump(summary_output, f, indent=2)

    print(f"\n✓ Summary saved to: {summary_path}")

    print("\n" + "="*70)
    print("STEP 1 COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nNext step: Run train_controller.py to train policy generators")


if __name__ == '__main__':
    main()
