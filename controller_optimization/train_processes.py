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

from controller_optimization.configs.processes_config import (
    PROCESSES, DATASET_MODE, get_process_by_name,
    ST_DATASET_CONFIG, _build_st_processes,
)
from controller_optimization.src.training.process_trainer import train_single_process

# Import report combining function
import importlib.util
spec_report = importlib.util.spec_from_file_location(
    "report_generator",
    REPO_ROOT / "uncertainty_predictor" / "src" / "utils" / "report_generator.py"
)
report_gen = importlib.util.module_from_spec(spec_report)
spec_report.loader.exec_module(report_gen)
combine_process_reports = report_gen.combine_process_reports


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

    # ST dataset complexity overrides (for complexity sweep)
    parser.add_argument('--st_n', type=int, default=None,
                        help='ST input variables per process (overrides st_params.n)')
    parser.add_argument('--st_m', type=int, default=None,
                        help='ST cascaded stages per process (overrides st_params.m)')
    parser.add_argument('--st_rho', type=float, default=None,
                        help='ST noise intensity [0,1] (overrides st_params.rho)')
    parser.add_argument('--st_n_processes', type=int, default=None,
                        help='Number of ST processes in sequence (overrides n_processes)')
    parser.add_argument('--checkpoint_base_dir', type=str, default=None,
                        help='Override base checkpoint dir for all processes')

    args = parser.parse_args()

    # If ST dataset params are overridden via CLI, rebuild processes dynamically
    _st_overrides = {
        k: v for k, v in [('n', args.st_n), ('m', args.st_m), ('rho', args.st_rho)]
        if v is not None
    }
    _has_n_processes_override = args.st_n_processes is not None
    if (_st_overrides or _has_n_processes_override) and DATASET_MODE == 'st':
        import copy as _copy
        _st_cfg = _copy.deepcopy(ST_DATASET_CONFIG)
        _st_cfg['st_params'].update(_st_overrides)
        if _has_n_processes_override:
            _st_cfg['n_processes'] = args.st_n_processes
        _custom_processes = _build_st_processes(_st_cfg)
        # Monkey-patch so the rest of the script uses the new processes
        import controller_optimization.configs.processes_config as _proc_mod
        _proc_mod.PROCESSES = _custom_processes
        print(f"\n[ST Override] Rebuilt processes with: {_st_overrides}"
              f"{f', n_processes={args.st_n_processes}' if _has_n_processes_override else ''}")

    # Override checkpoint dirs if --checkpoint_base_dir is provided
    if args.checkpoint_base_dir is not None:
        import controller_optimization.configs.processes_config as _proc_mod
        for p in _proc_mod.PROCESSES:
            p['checkpoint_dir'] = str(Path(args.checkpoint_base_dir) / p['name'])
        print(f"[Checkpoint Override] Base dir: {args.checkpoint_base_dir}")

    # Re-read PROCESSES after potential monkey-patching
    from controller_optimization.configs.processes_config import PROCESSES as _current_processes

    # Determine which processes to train
    if args.processes is None:
        processes_to_train = _current_processes
        process_names = [p['name'] for p in _current_processes]
    else:
        process_names = args.processes
        # Use current (potentially monkey-patched) PROCESSES
        _proc_map = {p['name']: p for p in _current_processes}
        processes_to_train = [_proc_map[name] for name in process_names]

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

    # Per dataset ST, tutti i processi sono identici: alleniamo solo il primo
    # e copiamo i checkpoint per gli altri.
    is_st_mode = DATASET_MODE == 'st'
    st_reference_result = None  # Risultato del primo processo ST allenato

    # Train each process
    for i, process_config in enumerate(processes_to_train, 1):
        process_name = process_config['name']

        print(f"\n{'='*70}")
        print(f"Process {i}/{len(processes_to_train)}: {process_name.upper()}")
        print(f"{'='*70}")

        checkpoint_dir = Path(process_config['checkpoint_dir'])
        model_path = checkpoint_dir / 'uncertainty_predictor.pth'

        # Invalidate stale checkpoints: if the process config changed,
        # wipe the checkpoint dir so training uses the new parameters.
        # Skip staleness detection when --checkpoint_base_dir is set:
        # the config is already correct from CLI overrides, and parallel
        # jobs sharing the same UP dir would race on rmtree.
        config_path = checkpoint_dir / 'process_config.json'
        if checkpoint_dir.exists() and args.checkpoint_base_dir is None:
            # Build a comparable snapshot of the current config
            _cfg_snapshot = {
                k: v for k, v in process_config.items()
                if k not in ('uncertainty_predictor', 'checkpoint_dir')
            }
            _stale = False
            if config_path.exists():
                with open(config_path, 'r') as f:
                    _old = json.load(f)
                if _old != _cfg_snapshot:
                    _stale = True
                    print(f"\n♻  Config changed for '{process_name}' — wiping stale checkpoint.")
            elif model_path.exists():
                # Checkpoint exists but no config.json → unknown provenance
                _stale = True
                print(f"\n♻  No config.json for '{process_name}' — wiping old checkpoint.")

            if _stale:
                import shutil
                shutil.rmtree(checkpoint_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
                        'MSE': metrics.get('MSE', metrics.get('mse', 0)),
                        'RMSE': metrics.get('RMSE', metrics.get('rmse', 0)),
                        'MAE': metrics.get('MAE', metrics.get('mae', 0)),
                        'R2': metrics.get('R2', metrics.get('r2', 0)),
                        'Calibration_Ratio': metrics.get('Calibration_Ratio', metrics.get('calibration_ratio', 0)),
                        'report_path': report_path,
                    })
            continue

        # ST mode: se abbiamo già allenato il primo processo, copiamo i checkpoint
        if is_st_mode and st_reference_result is not None:
            import shutil
            ref_dir = Path(st_reference_result['checkpoint_dir'])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Copia tutti i file dal checkpoint di riferimento
            for src_file in ref_dir.iterdir():
                if src_file.is_file():
                    shutil.copy2(src_file, checkpoint_dir / src_file.name)

            print(f"\n✓ Copied checkpoint from '{st_reference_result['process_name']}' "
                  f"(identical ST process)")

            all_results[process_name] = {
                'metrics': st_reference_result['metrics'],
                'report_path': str(checkpoint_dir / 'training_report.pdf'),
            }
            summary_data.append({
                'process': process_name,
                'MSE': st_reference_result['metrics']['MSE'],
                'RMSE': st_reference_result['metrics']['RMSE'],
                'MAE': st_reference_result['metrics']['MAE'],
                'R2': st_reference_result['metrics']['R2'],
                'Calibration_Ratio': st_reference_result['metrics']['Calibration_Ratio'],
                'report_path': str(checkpoint_dir / 'training_report.pdf'),
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

            # Per ST mode, salva il risultato di riferimento
            if is_st_mode and st_reference_result is None:
                st_reference_result = {
                    'process_name': process_name,
                    'checkpoint_dir': process_config['checkpoint_dir'],
                    'metrics': result['metrics'],
                }

            # Add to summary
            summary_data.append({
                'process': process_name,
                'MSE': result['metrics']['MSE'],
                'RMSE': result['metrics']['RMSE'],
                'MAE': result['metrics']['MAE'],
                'R2': result['metrics']['R2'],
                'Calibration_Ratio': result['metrics']['Calibration_Ratio'],
                'report_path': result['report_path'],
            })

            # Save config snapshot for future staleness detection
            _cfg_snapshot = {
                k: v for k, v in process_config.items()
                if k not in ('uncertainty_predictor', 'checkpoint_dir')
            }
            with open(checkpoint_dir / 'process_config.json', 'w') as f:
                json.dump(_cfg_snapshot, f, indent=2)

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
            f"{data['MSE']:<10.6f} "
            f"{data['RMSE']:<10.6f} "
            f"{data['MAE']:<10.6f} "
            f"{data['R2']:<10.6f} "
            f"{data['Calibration_Ratio']:<12.6f}"
        )

    # Print report paths
    print("\n" + "="*70)
    print("GENERATED REPORTS")
    print("="*70)

    for data in summary_data:
        print(f"  {data['process']:<12}: {data['report_path']}")

    # Save summary to JSON
    if args.checkpoint_base_dir is not None:
        summary_path = Path(args.checkpoint_base_dir) / 'processes_training_summary.json'
    else:
        summary_path = Path('controller_optimization/checkpoints/processes_training_summary.json')
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Custom encoder to handle numpy float32/int types
    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as _np
            if isinstance(obj, (_np.floating,)):
                return float(obj)
            if isinstance(obj, (_np.integer,)):
                return int(obj)
            return super().default(obj)

    summary_output = {
        'timestamp': datetime.now().isoformat(),
        'device': args.device,
        'seed': args.seed,
        'processes': summary_data,
    }

    with open(summary_path, 'w') as f:
        json.dump(summary_output, f, indent=2, cls=_NumpyEncoder)

    print(f"\n✓ Summary saved to: {summary_path}")

    # Combine all process reports into a single PDF
    if summary_data and len(summary_data) > 1:
        print("\n" + "="*70)
        print("COMBINING PROCESS REPORTS")
        print("="*70)

        report_paths = [data['report_path'] for data in summary_data]
        process_names = [data['process'] for data in summary_data]
        combined_report_path = Path('controller_optimization/checkpoints/combined_training_report.pdf')

        try:
            combined_path = combine_process_reports(
                report_paths=report_paths,
                output_path=combined_report_path,
                process_names=process_names
            )
            if combined_path:
                print(f"\n✓ All process reports combined into: {combined_path}")
        except Exception as e:
            print(f"\n✗ Error combining reports: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("STEP 1 COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nNext step: Run train_controller.py to train policy generators")


if __name__ == '__main__':
    main()
