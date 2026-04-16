"""
Step 1: Addestra uncertainty predictor per ogni processo.

Carica i dataset pre-generati da data/per_process/ (prodotti da generate_dataset.py)
e addestra un UP per ogni processo.

Usa: python train_predictor.py [--processes st_1 st_2] [--device cuda]

Output:
- Checkpoint per ogni processo in checkpoints/{process_name}/
- Report PDF per ogni processo
- Summary table con tutte le metriche
"""

import sys
from pathlib import Path
import argparse
import copy
import json
from datetime import datetime
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import KFold, train_test_split

# Add project root to path
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from configs.processes_config import (
    PROCESSES, DATASET_MODE, get_process_by_name,
    ST_DATASET_CONFIG, _build_st_processes,
)
from configs.uncertainty_config import CV_CONFIG
from uncertainty_predictor.src.training.process_trainer import (
    train_single_process, DataPreprocessor
)

# Import report combining function
import importlib.util
spec_report = importlib.util.spec_from_file_location(
    "report_generator",
    REPO_ROOT / "uncertainty_predictor" / "src" / "utils" / "report_generator.py"
)
report_gen = importlib.util.module_from_spec(spec_report)
spec_report.loader.exec_module(report_gen)
combine_process_reports = report_gen.combine_process_reports


def build_loaders(inputs, outputs, batch_size, seed):
    """
    Preprocessing + split + scaling + DataLoader construction.

    Identical to what process_trainer.py used to do internally.
    """
    preprocessor = DataPreprocessor(scaling_method='standard')

    X = inputs.numpy() if isinstance(inputs, torch.Tensor) else inputs
    y = outputs.numpy() if isinstance(outputs, torch.Tensor) else outputs

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        X, y,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        random_state=seed
    )

    # Fit and transform
    X_train_scaled, y_train_scaled = preprocessor.fit_transform(X_train, y_train)
    X_val_scaled, y_val_scaled = preprocessor.transform(X_val, y_val)
    X_test_scaled, y_test_scaled = preprocessor.transform(X_test, y_test)

    # Create data loaders with deterministic shuffling
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train_scaled)),
        batch_size=batch_size,
        shuffle=True,
        generator=g,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val_scaled)),
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test_scaled)),
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader, preprocessor


# Metrics to aggregate over CV folds. These are available in
# result['metrics'] returned by train_single_process (see process_trainer.py).
_CV_METRIC_KEYS = ('MSE', 'RMSE', 'MAE', 'R2', 'Calibration_Ratio', 'NLL', 'Mean_Variance')


def _make_loaders_from_arrays(X_train, y_train, X_val, y_val, X_test, y_test,
                              batch_size, seed):
    """
    Fit a fresh DataPreprocessor on X_train/y_train only and build DataLoaders
    for train/val/test without leaking test/val statistics into the scaler.
    """
    preprocessor = DataPreprocessor(scaling_method='standard')

    X_train_scaled, y_train_scaled = preprocessor.fit_transform(X_train, y_train)
    X_val_scaled, y_val_scaled = preprocessor.transform(X_val, y_val)
    X_test_scaled, y_test_scaled = preprocessor.transform(X_test, y_test)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train_scaled)),
        batch_size=batch_size,
        shuffle=True,
        generator=g,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val_scaled)),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test_scaled)),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader, preprocessor


def run_cv_for_process(process_config, inputs, outputs, cv_folds, seed,
                       device, verbose=True, test_fraction=0.15):
    """
    Run K-fold cross-validation for a single process.

    Protocol:
    1. Hold out 15% of the data as a fixed test set (random_state=seed).
    2. KFold(K, shuffle=True, random_state=seed) on the remaining 85%.
    3. Per fold: fresh DataPreprocessor (no leakage), fresh checkpoint_dir,
       retrain from scratch via train_single_process.
    4. Final refit on the full 85% (internal train/val split) and evaluate
       on the hold-out test set — this populates the "official" checkpoint
       at the standard checkpoint_dir.

    Returns:
        dict: {
            'cv_results': payload written to cv_results.json,
            'refit_result': dict returned by train_single_process on refit,
        }
    """
    process_name = process_config['name']
    batch_size = process_config['uncertainty_predictor']['training']['batch_size']
    base_checkpoint_dir = Path(process_config['checkpoint_dir'])

    X = inputs.numpy() if isinstance(inputs, torch.Tensor) else np.asarray(inputs)
    y = outputs.numpy() if isinstance(outputs, torch.Tensor) else np.asarray(outputs)

    n_total = X.shape[0]

    # 1. Fixed hold-out test set
    X_cv, X_test, y_cv, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=seed
    )
    n_holdout = X_test.shape[0]
    n_cv = X_cv.shape[0]

    if verbose:
        print(f"\n  [CV] Total={n_total}, hold-out test={n_holdout}, CV pool={n_cv}")
        print(f"  [CV] Running {cv_folds}-fold cross-validation...")

    # IMPORTANT: we run the refit FIRST, then the K folds.
    # train_single_process() wipes its target checkpoint_dir on entry (to clear
    # stale plots/reports). If we ran the folds first into
    # checkpoints/{process_name}/cv_fold_k/, the subsequent refit (targeting
    # checkpoints/{process_name}/) would nuke the fold subdirs. Running the
    # refit first lets fold subdirs be created AFTER the main dir has been
    # finalized, keeping both sets of artifacts intact.

    # Refit: train on the full 85% pool, validate on an internal slice,
    # evaluate on the hold-out test. The internal train/val split keeps the
    # trainer's early-stopping signal meaningful. Using test_size of
    # 0.15/0.85 maps the pool back to the ~70/15 train/val ratio of the
    # original single-split flow.
    if verbose:
        print(f"\n  {'-'*66}")
        print(f"  [CV] Final refit on {n_cv} samples; hold-out test = {n_holdout}")
        print(f"  {'-'*66}")

    # Refit val fraction relative to the pool: test_fraction / (1 - test_fraction)
    # so that train/val sizes, as fractions of the full dataset, equal
    # (1 - 2*test_fraction) and test_fraction respectively — matching the
    # original single-split convention.
    pool_val_frac = test_fraction / (1.0 - test_fraction)
    X_refit_tr, X_refit_vl, y_refit_tr, y_refit_vl = train_test_split(
        X_cv, y_cv, test_size=pool_val_frac, random_state=seed
    )
    train_loader, val_loader, test_loader, preprocessor = _make_loaders_from_arrays(
        X_refit_tr, y_refit_tr,
        X_refit_vl, y_refit_vl,
        X_test, y_test,
        batch_size=batch_size, seed=seed,
    )

    refit_result = train_single_process(
        process_config=process_config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        preprocessor=preprocessor,
        device=device,
        verbose=verbose,
        seed=seed,
    )

    holdout_metrics = {
        key: float(refit_result['metrics'][key])
        for key in _CV_METRIC_KEYS if key in refit_result['metrics']
    }

    # Now run the K folds. Each writes to its own subdir under base_checkpoint_dir,
    # which the refit above has already populated and won't touch again.
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    per_fold = []
    for k, (train_idx, val_idx) in enumerate(kf.split(X_cv)):
        X_tr, X_vl = X_cv[train_idx], X_cv[val_idx]
        y_tr, y_vl = y_cv[train_idx], y_cv[val_idx]

        if verbose:
            print(f"\n  {'-'*66}")
            print(f"  [CV] Fold {k+1}/{cv_folds}: train={len(train_idx)}, val={len(val_idx)}")
            print(f"  {'-'*66}")

        # Fold-specific checkpoint dir on a copy of the config (never mutate
        # the caller's config so the main refit's checkpoint stays intact).
        fold_config = copy.deepcopy(process_config)
        fold_config['checkpoint_dir'] = str(base_checkpoint_dir / f'cv_fold_{k}')

        # Inside a fold the "val" set plays two roles: early-stopping signal
        # AND the held-out set whose metrics we report for this fold. This is
        # standard practice in K-fold CV (each fold has a single out-set).
        train_loader, val_loader, test_loader, preprocessor = _make_loaders_from_arrays(
            X_tr, y_tr, X_vl, y_vl, X_vl, y_vl,
            batch_size=batch_size, seed=seed,
        )

        fold_result = train_single_process(
            process_config=fold_config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            preprocessor=preprocessor,
            device=device,
            verbose=verbose,
            seed=seed,
        )

        fold_metrics = {
            key: float(fold_result['metrics'][key])
            for key in _CV_METRIC_KEYS if key in fold_result['metrics']
        }
        per_fold.append({
            'fold': k,
            'n_train': int(len(train_idx)),
            'n_val': int(len(val_idx)),
            'checkpoint_dir': fold_config['checkpoint_dir'],
            'metrics': fold_metrics,
        })

    # Aggregate CV metrics (mean ± std across folds), NaN/inf safe.
    # If a fold produces a non-finite metric (e.g. Calibration_Ratio when the
    # mean predicted variance is ~0 on a tiny val fold), we exclude it from
    # that metric's aggregate and record how many folds contributed.
    cv_aggregated = {}
    for key in _CV_METRIC_KEYS:
        raw_values = [fold['metrics'].get(key) for fold in per_fold]
        finite = [v for v in raw_values if v is not None and np.isfinite(v)]
        n_used = len(finite)
        n_excluded = sum(1 for v in raw_values if v is None or not np.isfinite(v))
        if n_used == 0:
            continue
        cv_aggregated[key] = {
            'mean': float(np.mean(finite)),
            'std': float(np.std(finite, ddof=1)) if n_used > 1 else 0.0,
            'n_folds_used': n_used,
            'n_folds_excluded': n_excluded,
        }
        if n_excluded and verbose:
            print(f"  [CV] WARNING: metric '{key}' excluded {n_excluded}/{cv_folds} "
                  f"non-finite fold values from aggregate.")

    cv_results = {
        'process_name': process_name,
        'n_folds': cv_folds,
        'seed': seed,
        'n_total': int(n_total),
        'n_holdout_test': int(n_holdout),
        'n_cv_pool': int(n_cv),
        'per_fold': per_fold,
        'cv_aggregated': cv_aggregated,
        'holdout_test': {
            'n': int(n_holdout),
            'metrics': holdout_metrics,
            # train_single_process needs a non-empty val_loader for early
            # stopping, so the refit splits the pool internally rather than
            # training on the entire pool. With test_size = f/(1-f), this
            # reproduces the original (1-2f)/f/f partition sizes — only the
            # test set is now the deterministic hold-out instead of a random
            # slice. The refit's training regime is therefore directly
            # comparable to a non-CV run.
            'trained_on': (
                f'{(1 - test_fraction) * 100:.0f}% CV pool, split internally '
                f'into {(1 - 2 * test_fraction) * 100:.0f}%/'
                f'{test_fraction * 100:.0f}% train/val of the full dataset '
                f'(matches the original single-split sizes); test = fixed hold-out'
            ),
            'test_fraction': float(test_fraction),
            'refit_train_frac': 1.0 - 2.0 * test_fraction,
            'refit_val_frac': float(test_fraction),
        },
        'timestamp': datetime.now().isoformat(),
    }

    # Write cv_results.json next to the "official" checkpoint.
    cv_results_path = base_checkpoint_dir / 'cv_results.json'
    cv_results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cv_results_path, 'w') as f:
        json.dump(cv_results, f, indent=2)

    if verbose:
        print(f"\n  [CV] Results saved to: {cv_results_path}")
        print(f"  [CV] Aggregated (mean ± std over {cv_folds} folds):")
        for key, stats in cv_aggregated.items():
            print(f"    {key:<20s} {stats['mean']:.6f} ± {stats['std']:.6f}")
        print(f"  [CV] Hold-out test metrics (refit model — official):")
        for key, val in holdout_metrics.items():
            print(f"    {key:<20s} {val:.6f}")

    return {
        'cv_results': cv_results,
        'refit_result': refit_result,
    }


def main():
    """
    Per ogni processo in PROCESSES:
    1. Carica dataset da scm_ds/predictor_dataset/per_process/{process_name}_dataset.pt
    2. Costruisce DataLoader (preprocessing, split, scaling)
    3. Chiama train_single_process() con i DataLoader
    4. Raccoglie metriche
    5. Genera summary finale
    """
    parser = argparse.ArgumentParser(description='Train uncertainty predictors for all processes')
    parser.add_argument(
        '--processes',
        nargs='+',
        default=None,
        help='Specific processes to train (e.g., st_1 st_2). If not specified, trains all.'
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
    parser.add_argument(
        '--data_dir',
        type=str,
        default='scm_ds/predictor_dataset/',
        help='Directory containing per-process datasets from generate_dataset.py'
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
    # CV defaults come from configs/uncertainty_config.py::CV_CONFIG.
    # CLI args, when passed, override the config.
    _cfg_cv_folds = CV_CONFIG.get('cv_folds')
    _cfg_test_fraction = CV_CONFIG.get('test_fraction', 0.15)
    parser.add_argument(
        '--cv_folds',
        type=int,
        default=_cfg_cv_folds,
        help=f'K for K-fold cross-validation per process. None/0 = single '
             f'(1-2f)/f/f split (original behavior). '
             f'Default from CV_CONFIG: {_cfg_cv_folds}.'
    )
    parser.add_argument(
        '--test_fraction',
        type=float,
        default=_cfg_test_fraction,
        help=f'Hold-out test fraction used by CV mode. '
             f'Default from CV_CONFIG: {_cfg_test_fraction}.'
    )

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
        import configs.processes_config as _proc_mod
        _proc_mod.PROCESSES = _custom_processes
        print(f"\n[ST Override] Rebuilt processes with: {_st_overrides}"
              f"{f', n_processes={args.st_n_processes}' if _has_n_processes_override else ''}")

    # Override checkpoint dirs if --checkpoint_base_dir is provided
    if args.checkpoint_base_dir is not None:
        import configs.processes_config as _proc_mod
        for p in _proc_mod.PROCESSES:
            p['checkpoint_dir'] = str(Path(args.checkpoint_base_dir) / p['name'])
        print(f"[Checkpoint Override] Base dir: {args.checkpoint_base_dir}")

    # Re-read PROCESSES after potential monkey-patching
    from configs.processes_config import PROCESSES as _current_processes

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
    print("AZIMUTH - STEP 1: TRAIN UNCERTAINTY PREDICTORS")
    print("="*70)
    print(f"\nProcesses to train: {', '.join(process_names)}")
    print(f"Device: {args.device}")
    print(f"Skip existing: {args.skip_existing}")
    print(f"Random seed: {args.seed}")
    print(f"Data dir: {args.data_dir}")
    if args.cv_folds is not None and args.cv_folds >= 2:
        print(f"Cross-validation: {args.cv_folds}-fold "
              f"(hold-out test = {args.test_fraction * 100:.0f}%)")

    # Storage for results
    all_results = {}
    summary_data = []

    # Per dataset ST, tutti i processi sono identici: alleniamo solo il primo
    # e copiamo i checkpoint per gli altri.
    is_st_mode = DATASET_MODE == 'st'
    st_reference_result = None  # Risultato del primo processo ST allenato

    data_dir = Path(args.data_dir)

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
        # Skip staleness detection when --checkpoint_base_dir is set.
        config_path = checkpoint_dir / 'process_config.json'
        if checkpoint_dir.exists() and args.checkpoint_base_dir is None:
            # Build a comparable snapshot of the current config
            _cfg_snapshot = {
                k: v for k, v in process_config.items()
                if k not in ('uncertainty_predictor', 'checkpoint_dir')
            }
            _stale = False
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        _old = json.load(f)
                    if _old != _cfg_snapshot:
                        _stale = True
                        print(f"\n  Config changed for '{process_name}' — wiping stale checkpoint.")
                except (json.JSONDecodeError, OSError):
                    _stale = True
                    print(f"\n  Corrupted config.json for '{process_name}' — wiping stale checkpoint.")
            elif model_path.exists():
                # Checkpoint exists but no config.json → unknown provenance
                _stale = True
                print(f"\n  No config.json for '{process_name}' — wiping old checkpoint.")

            if _stale:
                import shutil
                shutil.rmtree(checkpoint_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Check if already trained
        if args.skip_existing and model_path.exists():
            print(f"\n  Checkpoint already exists for '{process_name}'. Skipping...")

            # Load existing metrics
            _skip_ok = False
            info_path = checkpoint_dir / 'training_info.json'
            if info_path.exists():
                try:
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                except (json.JSONDecodeError, OSError):
                    print(f"\n  Corrupted training_info.json for '{process_name}' — retraining.")
                else:
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
                    _skip_ok = True
            else:
                _skip_ok = True  # No info file but model exists — skip anyway
            if _skip_ok:
                continue

        # ST mode: se abbiamo già allenato il primo processo, copiamo i checkpoint
        if is_st_mode and st_reference_result is not None:
            import shutil
            ref_dir = Path(st_reference_result['checkpoint_dir'])

            # Wipe the destination so files from previous runs that are no
            # longer produced by the reference process don't linger.
            if checkpoint_dir.exists():
                for _entry in checkpoint_dir.iterdir():
                    if _entry.is_file() or _entry.is_symlink():
                        try:
                            _entry.unlink()
                        except OSError:
                            pass
                    elif _entry.is_dir():
                        shutil.rmtree(_entry, ignore_errors=True)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Copia tutti i file dal checkpoint di riferimento
            for src_file in ref_dir.iterdir():
                if src_file.is_file():
                    shutil.copy2(src_file, checkpoint_dir / src_file.name)

            print(f"\n  Copied checkpoint from '{st_reference_result['process_name']}' "
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

        # Load dataset from disk
        dataset_path = data_dir / 'per_process' / f'{process_name}_dataset.pt'
        if not dataset_path.exists():
            print(f"\n  Dataset not found: {dataset_path}")
            print(f"  Run generate_dataset.py first!")
            continue

        print(f"\n  Loading dataset from {dataset_path}...")
        dataset = torch.load(dataset_path, weights_only=False)
        inputs = dataset['inputs']
        outputs = dataset['outputs']
        print(f"  Loaded: inputs {inputs.shape}, outputs {outputs.shape}")

        # Build DataLoaders (only needed for the non-CV path; the CV helper
        # builds its own loaders per fold with fold-specific scaling)
        training_config = process_config['uncertainty_predictor']['training']
        batch_size = training_config['batch_size']

        # Train process
        try:
            # Treat 0 as "disabled" to match the CV_CONFIG doc ("None or 0").
            cv_enabled = args.cv_folds is not None and args.cv_folds >= 2
            if cv_enabled:
                cv_output = run_cv_for_process(
                    process_config=process_config,
                    inputs=inputs,
                    outputs=outputs,
                    cv_folds=args.cv_folds,
                    seed=args.seed,
                    device=args.device,
                    verbose=True,
                    test_fraction=args.test_fraction,
                )
                result = cv_output['refit_result']
            else:
                train_loader, val_loader, test_loader, preprocessor = build_loaders(
                    inputs, outputs, batch_size, args.seed
                )
                result = train_single_process(
                    process_config=process_config,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    preprocessor=preprocessor,
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

            print(f"\n  Training completed for '{process_name}'")

        except Exception as e:
            print(f"\n  Error training '{process_name}': {e}")
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
        summary_path = Path('uncertainty_predictor/checkpoints/processes_training_summary.json')
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

    print(f"\n  Summary saved to: {summary_path}")

    # Combine all process reports into a single PDF
    if summary_data and len(summary_data) > 1:
        print("\n" + "="*70)
        print("COMBINING PROCESS REPORTS")
        print("="*70)

        report_paths = [data['report_path'] for data in summary_data]
        proc_names = [data['process'] for data in summary_data]
        combined_report_path = Path('uncertainty_predictor/checkpoints/combined_training_report.pdf')

        try:
            combined_path = combine_process_reports(
                report_paths=report_paths,
                output_path=combined_report_path,
                process_names=proc_names
            )
            if combined_path:
                print(f"\n  All process reports combined into: {combined_path}")
        except Exception as e:
            print(f"\n  Error combining reports: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("STEP 1 COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nNext step: Run train_controller.py to train policy generators")


if __name__ == '__main__':
    main()
