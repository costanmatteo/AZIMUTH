#!/usr/bin/env python3
"""
Train CasualiT as a Surrogate for Reliability F Prediction.

Loads a base causaliT YAML config and merges overrides from
configs/surrogate_config.py, then launches training via the
standard causaliT Lightning trainer.

Usage:
    python train_surrogate.py [options]

Options:
    --epochs INT              Max training epochs (default: from config)
    --batch_size INT          Batch size (default: from config)
    --learning_rate FLOAT     Learning rate (default: from config)
    --generate_data           Generate new training data before training
    --data_only               Generate data only (no training)
    --use_existing_dataset    Load data converted by convert_dataset.py instead of generating
    --output_dir PATH         Output directory
    --data_dir PATH           Data directory (parent of dataset folder)
    --device STR              Device (cpu/cuda/auto)
"""

import os
import warnings
warnings.filterwarnings("ignore", message=".*weights_only.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:torch.serialization"

import sys
import os
import json
import argparse
from pathlib import Path
import numpy as np

# Add paths
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'causaliT'))

# Check for required dependencies
try:
    import torch
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install: pip install torch")
    sys.exit(1)

from configs.surrogate_config import SURROGATE_CONFIG


def build_config(cli_overrides=None):
    """
    Build OmegaConf config by merging base YAML with surrogate overrides.

    Priority (highest wins): CLI args > surrogate_config overrides > base YAML defaults.
    """
    from omegaconf import OmegaConf

    base_yaml_path = SURROGATE_CONFIG['base_yaml']
    base = OmegaConf.load(base_yaml_path)
    overrides = OmegaConf.create(SURROGATE_CONFIG['overrides'])
    config = OmegaConf.merge(base, overrides)

    # Apply CLI overrides if provided
    if cli_overrides:
        config = OmegaConf.merge(config, OmegaConf.create(cli_overrides))

    # Derive d_model_enc/d_model_dec for ProT based on embedding composition mode
    from causaliT.training.experiment_control import update_config
    config = update_config(config)

    return config


def main():
    parser = argparse.ArgumentParser(description='Train CasualiT Surrogate')
    parser.add_argument('--epochs', type=int, default=None, help='Max epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--generate_data', action='store_true', help='Generate new training data')
    parser.add_argument('--data_only', action='store_true', help='Generate data only (no training)')
    parser.add_argument('--use_existing_dataset', action='store_true',
                       help='Use pre-existing full_trajectories.pt (converted by convert_dataset.py)')
    parser.add_argument('--output_dir', type=str, default='causaliT/checkpoints/surrogate',
                       help='Output directory')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Data directory (parent of dataset folder, default: causaliT/data)')
    parser.add_argument('--trajectories_path', type=str, default=None,
                       help='Path to full_trajectories.pt (overrides config default)')
    parser.add_argument('--device', type=str, default='auto', help='Device (cpu/cuda/auto)')

    # ST dataset complexity overrides (for complexity sweep)
    parser.add_argument('--st_n', type=int, default=None,
                        help='ST input variables per process (overrides st_params.n)')
    parser.add_argument('--st_m', type=int, default=None,
                        help='ST cascaded stages per process (overrides st_params.m)')
    parser.add_argument('--st_rho', type=float, default=None,
                        help='ST noise intensity [0,1] (overrides st_params.rho)')
    parser.add_argument('--st_n_processes', type=int, default=None,
                        help='Number of ST processes in sequence (overrides n_processes)')
    parser.add_argument('--up_checkpoint_dir', type=str, default=None,
                        help='Override UP checkpoint base dir (reads UPs from here)')

    args = parser.parse_args()

    # If ST dataset params are overridden via CLI, rebuild processes dynamically
    from configs.processes_config import DATASET_MODE, ST_DATASET_CONFIG, _build_st_processes
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
        import configs.processes_config as _proc_mod
        _proc_mod.PROCESSES = _custom_processes
        print(f"\n[ST Override] Rebuilt processes with: {_st_overrides}"
              f"{f', n_processes={args.st_n_processes}' if _has_n_processes_override else ''}")

    # Override UP checkpoint dirs if --up_checkpoint_dir is provided
    if args.up_checkpoint_dir is not None:
        import configs.processes_config as _proc_mod
        for p in _proc_mod.PROCESSES:
            p['checkpoint_dir'] = str(Path(args.up_checkpoint_dir) / p['name'])
        print(f"[UP Checkpoint Override] Base dir: {args.up_checkpoint_dir}")

    # Build CLI overrides dict
    cli_overrides = {}
    if args.epochs:
        cli_overrides.setdefault('training', {})['max_epochs'] = args.epochs
        cli_overrides.setdefault('experiment', {})['max_epochs'] = args.epochs
    if args.batch_size:
        cli_overrides.setdefault('training', {})['batch_size'] = args.batch_size
        cli_overrides.setdefault('experiment', {})['batch_size'] = args.batch_size
    if args.learning_rate:
        cli_overrides.setdefault('training', {})['lr'] = args.learning_rate
        cli_overrides.setdefault('experiment', {})['lr'] = args.learning_rate

    # Build merged config
    config = build_config(cli_overrides if cli_overrides else None)

    # Resolve data_dir
    data_dir = args.data_dir or str(REPO_ROOT / 'causaliT' / 'data')
    dataset_name = config['data']['dataset']

    print("=" * 70)
    print("CasualiT Surrogate Training")
    print("=" * 70)
    print(f"  Base YAML:  {SURROGATE_CONFIG['base_yaml']}")
    print(f"  Model:      {config['model']['model_object']}")
    print(f"  Dataset:    {dataset_name}")
    print(f"  Data dir:   {data_dir}")
    print(f"  Output dir: {args.output_dir}")

    # ── Data preparation ───────────────────────────────────────────────────
    dataset_path = Path(data_dir) / dataset_name

    if args.use_existing_dataset:
        from addition_to_causaliT.surrogate_training.convert_dataset import convert_trajectories_to_causalit_format

        traj_path = (args.trajectories_path
                     or 'scm_ds/predictor_dataset/trajectories/full_trajectories.pt')
        print(f"\n[1/2] Converting existing dataset from {traj_path}...")
        convert_trajectories_to_causalit_format(
            trajectories_path=traj_path,
            output_dir=str(dataset_path),
            model_type=config['model']['model_object'],
        )
        if args.data_only:
            print("\nData conversion complete.")
            return

    elif args.generate_data or args.data_only:
        from addition_to_causaliT.surrogate_training.data_generator import generate_all_datasets
        print(f"\n[1/2] Generating training data...")
        stats = generate_all_datasets(config, str(dataset_path), device=args.device)
        if args.data_only:
            print("\nData generation complete.")
            return

    else:
        # Check for required data files based on config
        train_file = config['data'].get('train_file')
        test_file = config['data'].get('test_file')
        if train_file and test_file:
            # Pre-split mode: need train and test files
            missing = [f for f in [train_file, test_file]
                       if not (dataset_path / f).exists()]
            if missing:
                print(f"\n[ERROR] Missing pre-split data files in {dataset_path}/:")
                for f in missing:
                    print(f"  - {f}")
                print("  Run with --use_existing_dataset to generate them from trajectories.")
                return
        elif not (dataset_path / 'ds.npz').exists():
            print(f"\n[ERROR] No data found at {dataset_path}/ds.npz")
            print("  Run generate_dataset.py first, or use --generate_data / --use_existing_dataset")
            return
        print(f"\n[1/2] Using existing data at {dataset_path}/")

    # ── Train ──────────────────────────────────────────────────────────────
    from causaliT.training.trainer import trainer as lightning_trainer

    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n[2/2] Training {config['model']['model_object']} with PyTorch Lightning...")

    results_df = lightning_trainer(
        config=config,
        data_dir=data_dir,
        save_dir=save_dir,
        cluster=False,
    )

    # Copy best checkpoint
    import shutil
    best_ckpt_src = Path(save_dir) / 'k_0' / 'checkpoints' / 'best_checkpoint.ckpt'
    best_ckpt_dst = Path(save_dir) / 'best_model.ckpt'
    if best_ckpt_src.exists():
        shutil.copy2(best_ckpt_src, best_ckpt_dst)
        print(f"  Best checkpoint copied to: {best_ckpt_dst}")

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"  Checkpoints: {save_dir}")
    if results_df is not None:
        print(f"  Results:\n{results_df.to_string()}")

    # ── Generate PDF report ───────────────────────────────────────────────
    try:
        _generate_report(config, results_df, save_dir, data_dir, dataset_name)
    except Exception as e:
        print(f"\nWarning: Report generation failed: {e}")
        import traceback
        traceback.print_exc()


def _generate_report(config, results_df, save_dir, data_dir, dataset_name):
    """Build history/eval dicts from training artifacts and generate the PDF report."""
    import pandas as pd
    from omegaconf import OmegaConf
    from addition_to_causaliT.surrogate_training.report_generator import generate_surrogate_training_report

    fold_dir = Path(save_dir) / 'k_0'

    # --- Training history from CSV logger ---
    csv_metrics_path = list(fold_dir.glob('logs/csv/*/metrics.csv'))
    history = {}
    if csv_metrics_path:
        df_log = pd.read_csv(csv_metrics_path[0])
        history['train_loss'] = df_log['train_loss'].dropna().tolist() if 'train_loss' in df_log else []
        history['val_loss'] = df_log['val_loss'].dropna().tolist() if 'val_loss' in df_log else []
        history['train_mae'] = df_log['train_mae'].dropna().tolist() if 'train_mae' in df_log else []
        history['val_mae'] = df_log['val_mae'].dropna().tolist() if 'val_mae' in df_log else []
        history['learning_rate'] = df_log['lr-Adam'].dropna().tolist() if 'lr-Adam' in df_log else []
        history['final_epoch'] = len(history['train_loss'])
        history['final_train_loss'] = history['train_loss'][-1] if history['train_loss'] else 0.0
        history['final_val_loss'] = history['val_loss'][-1] if history['val_loss'] else 0.0
        history['final_train_mae'] = history['train_mae'][-1] if history['train_mae'] else 0.0
        history['final_val_mae'] = history['val_mae'][-1] if history['val_mae'] else 0.0

    # --- Best epoch from best_metrics.json ---
    best_metrics_path = fold_dir / 'best_metrics.json'
    if best_metrics_path.exists():
        import json as _json
        with open(best_metrics_path) as f:
            best = _json.load(f)
        history['best_epoch'] = best.get('best_epoch', 0)
        history['best_val_loss'] = best.get('val_loss', history.get('final_val_loss', 0.0))
    else:
        history['best_epoch'] = history.get('final_epoch', 0)
        history['best_val_loss'] = history.get('final_val_loss', 0.0)

    # --- Eval results from results_df (fold 0) ---
    eval_results = {}
    if results_df is not None and len(results_df) > 0:
        row = results_df.iloc[0]
        eval_results['test_mse'] = float(row.get('test_loss', 0.0))
        eval_results['test_mae'] = float(row.get('test_mae', 0.0))
        eval_results['test_rmse'] = float(row.get('test_loss', 0.0)) ** 0.5
        eval_results['test_r2'] = float(row.get('test_r2_Y', row.get('test_r2', 0.0)))

    # Predictions/targets arrays (if evaluation saved them)
    pred_path = fold_dir / 'eval_predictions.npz'
    if pred_path.exists():
        pred_data = np.load(pred_path)
        eval_results['predictions'] = pred_data.get('predictions', np.array([]))
        eval_results['targets'] = pred_data.get('targets', np.array([]))
    else:
        eval_results['predictions'] = np.array([])
        eval_results['targets'] = np.array([])

    # --- Dataset split sizes ---
    ds_path = Path(data_dir) / dataset_name / 'ds.npz'
    if ds_path.exists():
        ds = np.load(ds_path)
        n_total = ds['x'].shape[0] if 'x' in ds else ds['s'].shape[0]
        input_dim = ds['x'].shape[1] if 'x' in ds else 0
    else:
        n_total = 0
        input_dim = 0

    # Read split sizes from data index tracker if available
    idx_path = fold_dir / 'data_indices.json'
    if idx_path.exists():
        import json as _json
        with open(idx_path) as f:
            idx_data = _json.load(f)
        n_train = len(idx_data.get('train_indices', []))
        n_val = len(idx_data.get('val_indices', []))
        n_test = len(idx_data.get('test_indices', []))
    else:
        # Fallback: estimate from config ratios
        test_ratio = config.get('data', {}).get('test_size', 0.15)
        n_test = int(n_total * test_ratio)
        n_train_val = n_total - n_test
        n_val = int(n_train_val * 0.2)
        n_train = n_train_val - n_val

    # --- Model params ---
    total_params = int(results_df.iloc[0].get('trainable_params', 0)) if results_df is not None and len(results_df) > 0 else 0

    # --- Build a flat config dict the report generator expects ---
    # OmegaConf configs nest arch fields under model.kwargs; the report
    # expects them directly under config['model'].
    if hasattr(config, '_metadata'):  # OmegaConf object
        cfg_dict = OmegaConf.to_container(config, resolve=True)
    else:
        cfg_dict = dict(config)

    # Flatten model.kwargs into model so the report finds d_model_enc, d_ff, etc.
    model_section = cfg_dict.get('model', {})
    kwargs = model_section.pop('kwargs', {})
    for k, v in kwargs.items():
        if k not in model_section:
            model_section[k] = v

    # Map surrogate_config override names → canonical report field names
    # (in case the experiment section wasn't interpolated into kwargs)
    exp = cfg_dict.get('experiment', {})
    _alias_map = {
        'd_model_enc': lambda: model_section.get('d_model_enc') or exp.get('d_model_set'),
        'd_model_dec': lambda: model_section.get('d_model_dec') or exp.get('d_model_set'),
        'd_ff':        lambda: model_section.get('d_ff') or exp.get('d_ff') or model_section.get('ffn_dim'),
        'd_qk':        lambda: model_section.get('d_qk') or exp.get('d_qk'),
        'e_layers':    lambda: model_section.get('e_layers') or exp.get('e_layers') or model_section.get('n_enc_layers'),
        'd_layers':    lambda: model_section.get('d_layers') or exp.get('d_layers') or model_section.get('n_dec_layers'),
        'n_heads':     lambda: model_section.get('n_heads') or exp.get('n_heads'),
    }
    for key, resolver in _alias_map.items():
        if not model_section.get(key):
            val = resolver()
            if val is not None:
                model_section[key] = val

    # Ensure dropout fields are populated from the single 'dropout' override
    _dropout = model_section.get('dropout') or exp.get('dropout')
    if _dropout is not None:
        for dk in ('dropout_emb', 'dropout_attn_out', 'dropout_ff'):
            if not model_section.get(dk):
                model_section[dk] = _dropout

    # Ensure training section has report-expected keys (lr alias)
    tr = cfg_dict.get('training', {})
    if not tr.get('learning_rate') and tr.get('lr'):
        tr['learning_rate'] = tr['lr']
    if not tr.get('max_epochs') and exp.get('max_epochs'):
        tr['max_epochs'] = exp['max_epochs']
    if not tr.get('patience') and tr.get('patience') is None:
        # patience comes from surrogate_config override; already present if set
        pass

    # --- Populate Processes / Trajectories / Scenarios from dataset metadata ---
    data_section = cfg_dict.get('data', {})
    meta_path = Path(data_dir) / dataset_name / 'dataset_metadata.json'
    if meta_path.exists() and not data_section.get('process_names'):
        with open(meta_path) as f:
            ds_meta = json.load(f)
        var_info = ds_meta.get('variable_info', {})
        if var_info.get('process_names'):
            data_section['process_names'] = var_info['process_names']
        if var_info.get('n_processes') and not data_section.get('n_scenarios'):
            data_section['n_scenarios'] = var_info['n_processes']

    # Fallback: read from processes_config if metadata doesn't have the info
    if not data_section.get('process_names'):
        try:
            from configs.processes_config import PROCESSES
            data_section['process_names'] = [p['name'] for p in PROCESSES]
        except Exception:
            pass

    # Trajectories = total number of samples in the dataset
    if not data_section.get('n_trajectories'):
        data_section['n_trajectories'] = n_total or (n_train + n_val + n_test)

    # Scenarios = number of processes if not already set
    if not data_section.get('n_scenarios') and data_section.get('process_names'):
        data_section['n_scenarios'] = len(data_section['process_names'])

    cfg_dict['model'] = model_section
    cfg_dict['data'] = data_section
    cfg_dict['training'] = tr

    # --- Compute Floor MSE (MSE restricted to top-decile F targets, F >= q80) ---
    floor_metrics = None
    targets = eval_results.get('targets', np.array([]))
    predictions = eval_results.get('predictions', np.array([]))
    if targets.size > 0 and predictions.size > 0:
        tgt_flat = targets.ravel()
        pred_flat = predictions.ravel()
        q80 = np.percentile(tgt_flat, 80)
        mask = tgt_flat >= q80
        if mask.sum() > 0:
            residuals = pred_flat[mask] - tgt_flat[mask]
            floor_mse = float(np.mean(residuals ** 2))
            floor_bias = float(np.mean(residuals))
            floor_metrics = {'floor_mse': floor_mse, 'floor_bias': floor_bias}

    # --- Early stopping info ---
    max_epochs = tr.get('max_epochs', exp.get('max_epochs'))
    patience = tr.get('patience')
    best_ep = history.get('best_epoch', 0)
    final_ep = history.get('final_epoch', 0)
    if max_epochs and patience and final_ep < int(max_epochs):
        history['early_stopped'] = True
    else:
        history['early_stopped'] = False

    generate_surrogate_training_report(
        config=cfg_dict,
        history=history,
        eval_results=eval_results,
        input_dim=input_dim,
        output_dim=1,
        total_params=total_params,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        checkpoint_dir=save_dir,
        floor_metrics=floor_metrics,
    )


if __name__ == '__main__':
    main()
