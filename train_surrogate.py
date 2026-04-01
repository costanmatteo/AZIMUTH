#!/usr/bin/env python3
"""
Train CasualiT as a Surrogate for Reliability F Prediction.

This script trains a model to predict reliability F from process chain
trajectories.  The model architecture is selected via surrogate_config
(casualit_model: proT | StageCausaliT | SingleCausalLayer).

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
    --device STR              Device (cpu/cuda/auto)
"""

import sys
import os
import json
import argparse
from pathlib import Path
import numpy as np

# Add paths
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

# Check for required dependencies
try:
    import torch
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install: pip install torch")
    sys.exit(1)

from configs.surrogate_config import SURROGATE_CONFIG
from causaliT.surrogate_training.data_generator import generate_all_datasets, TrajectoryDataGenerator


def _train_prot_lightning(config, args):
    """
    Train ProT (TransformerForecaster) using the PyTorch Lightning pipeline.

    Builds the OmegaConf config expected by causaliT.training.trainer.trainer()
    from the flat SURROGATE_CONFIG and the dataset_metadata.json produced by
    convert_dataset.py, then launches training.
    """
    from omegaconf import OmegaConf
    from causaliT.training.trainer import trainer as lightning_trainer

    # ── Load dataset metadata (created by convert_dataset.py) ─────────
    metadata_path = Path(args.data_dir) / 'dataset_metadata.json'
    if not metadata_path.exists():
        print(f"\n[ERROR] dataset_metadata.json not found at {metadata_path}")
        print("  Run with --use_existing_dataset or --generate_data first.")
        return
    with open(metadata_path) as f:
        metadata = json.load(f)

    n_features = metadata['variable_info']['n_features']
    n_processes = metadata['variable_info']['n_processes']

    # ── Read hyper-parameters from SURROGATE_CONFIG ───────────────────
    d_model_enc = config['model'].get('d_model_enc', 32)
    d_model_dec = config['model'].get('d_model_dec', 16)
    d_ff        = config['model'].get('d_ff', 64)
    d_qk        = config['model'].get('d_qk', 8)
    n_heads     = config['model'].get('n_heads', 4)
    e_layers    = config['model'].get('e_layers', 2)
    d_layers    = config['model'].get('d_layers', 1)
    dropout     = config['model'].get('dropout_emb', 0.3)
    activation  = config['model'].get('activation', 'gelu')
    norm        = config['model'].get('norm', 'batch')

    device_str = 'cuda' if (args.device == 'cuda' or
                            (args.device == 'auto' and torch.cuda.is_available())) else 'cpu'

    # ── Build encoder embeddings: one linear per feature column ───────
    enc_modules = []
    for i in range(n_features):
        enc_modules.append({
            'idx': i, 'embed': 'linear', 'label': f'feat_{i}',
            'kwargs': {'input_dim': 1, 'embedding_dim': d_model_enc},
        })

    # ── Build decoder embeddings: mask + linear for scalar F ──────────
    dec_modules = [
        {'idx': 0, 'embed': 'mask', 'label': 'F_mask', 'kwargs': {}},
        {'idx': 0, 'embed': 'linear', 'label': 'F_value',
         'kwargs': {'input_dim': 1, 'embedding_dim': d_model_dec}},
    ]

    # ── Path handling: Lightning trainer joins data_dir + dataset ─────
    data_parent  = str(Path(args.data_dir).parent)
    dataset_name = str(Path(args.data_dir).name)

    # ── Build the config dict expected by the Lightning pipeline ──────
    lt_config = OmegaConf.create({
        'model': {
            'model_object': 'proT',
            'kwargs': {
                'model': 'proT',

                # Encoder embeddings (one linear per feature column, summation)
                'ds_embed_enc': {
                    'setting': {'d_model': d_model_enc, 'sparse_grad': False},
                    'modules': enc_modules,
                },
                'comps_embed_enc': 'summation',

                # Decoder embeddings (mask + linear for F)
                'ds_embed_dec': {
                    'setting': {'d_model': d_model_dec, 'sparse_grad': False},
                    'modules': dec_modules,
                },
                'comps_embed_dec': 'summation',

                # Attention (plain scaled dot-product, no causal masks)
                'enc_attention_type':        'ScaledDotProduct',
                'enc_mask_type':             'Uniform',
                'dec_self_attention_type':    'ScaledDotProduct',
                'dec_self_mask_type':         'Uniform',
                'dec_cross_attention_type':   'ScaledDotProduct',
                'dec_cross_mask_type':        'Uniform',
                'n_heads':                    n_heads,
                'enc_causal_mask':            False,
                'dec_causal_mask':            False,

                # Architecture
                'e_layers':       e_layers,
                'd_layers':       d_layers,
                'activation':     activation,
                'norm':           norm,
                'use_final_norm': config['model'].get('use_final_norm', True),
                'device':         device_str,
                'out_dim':        1,
                'd_ff':           d_ff,
                'd_model_enc':    d_model_enc,
                'd_model_dec':    d_model_dec,
                'd_qk':           d_qk,

                # Sequence lengths for attention mask initialization
                'X_seq_len':      n_processes,
                'Y_seq_len':      1,

                # Dropout
                'dropout_emb':                  dropout,
                'dropout_data':                 dropout,
                'dropout_attn_out':             config['model'].get('dropout_attn_out', dropout),
                'dropout_ff':                   config['model'].get('dropout_ff', dropout),
                'enc_dropout_qkv':              dropout,
                'enc_attention_dropout':         dropout,
                'dec_self_dropout_qkv':         dropout,
                'dec_self_attention_dropout':    dropout,
                'dec_cross_dropout_qkv':        dropout,
                'dec_cross_attention_dropout':   dropout,
            },
        },

        'data': {
            'dataset':         dataset_name,
            'filename_input':  'ds.npz',
            'filename_target': 'ds.npz',   # same file, keys 'x' and 'y'
            'val_idx':         0,           # F value column in Y
            'test_ds_ixd':     None,
            'max_data_size':   None,
            'X_seq_len':       n_processes,
            'Y_seq_len':       1,
        },

        'training': {
            'optimizer':      'adamw',
            'lr':             config['training'].get('learning_rate', 5e-4),
            'weight_decay':   config['training'].get('weight_decay', 0.05),
            'use_scheduler':  config['training'].get('use_scheduler', True),
            'loss_fn':        'mse',
            'batch_size':     config['training'].get('batch_size', 32),
            'max_epochs':     config['training'].get('max_epochs', 200),
            'k_fold':         max(config['training'].get('k_fold', 2), 2),
            'seed':           config['training'].get('seed', 42),
            'save_ckpt_every_n_epochs': 50,
            'log_entropy':       False,
            'log_acyclicity':    False,
            'use_hard_masks':    False,
            'use_in_context_masks': False,
        },

        'special':    {'mode': []},
        'evaluation': {'functions': ['eval_train_metrics']},
    })

    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n[2/4] Training proT (TransformerForecaster) with PyTorch Lightning...")
    print(f"  d_model_enc={d_model_enc}, d_model_dec={d_model_dec}, n_heads={n_heads}, d_ff={d_ff}")
    print(f"  n_processes={n_processes}, n_features={n_features}")

    results_df = lightning_trainer(
        config=lt_config,
        data_dir=data_parent,
        save_dir=save_dir,
        cluster=False,
    )

    # Copy best checkpoint to the expected location
    best_ckpt_src = Path(save_dir) / 'k_0' / 'best_checkpoint.ckpt'
    best_ckpt_dst = Path(save_dir) / 'best_model.ckpt'
    if best_ckpt_src.exists():
        import shutil
        shutil.copy2(best_ckpt_src, best_ckpt_dst)
        print(f"  Best checkpoint copied to: {best_ckpt_dst}")

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"  Checkpoints: {save_dir}")
    if results_df is not None:
        print(f"  Results:\n{results_df.to_string()}")




def _train_stage_causal_lightning(casualit_model, config, args):
    """
    Train StageCausaliT or SingleCausalLayer using the PyTorch Lightning pipeline.

    Builds the OmegaConf config expected by causaliT.training.trainer.trainer()
    from the flat SURROGATE_CONFIG and the dataset_metadata.json produced by
    convert_dataset.py, then launches training.
    """
    from omegaconf import OmegaConf
    from causaliT.training.trainer import trainer as lightning_trainer

    if casualit_model == 'SingleCausalLayer':
        print(f"\n[INFO] SingleCausalLayer Lightning training not yet wired.")
        print(f"  Data has been prepared in {args.data_dir}.")
        return

    # ── Load dataset metadata (created by convert_dataset.py) ─────────
    metadata_path = Path(args.data_dir) / 'dataset_metadata.json'
    if not metadata_path.exists():
        print(f"\n[ERROR] dataset_metadata.json not found at {metadata_path}")
        print("  Run with --use_existing_dataset or --generate_data first.")
        return
    with open(metadata_path) as f:
        metadata = json.load(f)

    n_s = metadata['variable_info']['n_source']
    n_x = metadata['variable_info']['n_input']
    n_y = metadata['variable_info']['n_target']
    total_vars = n_s + n_x + n_y + 1  # +1 for padding_idx=0

    # ── Read hyper-parameters from SURROGATE_CONFIG ───────────────────
    d_model = config['model'].get('d_model_enc', 64)
    d_ff    = config['model'].get('d_ff', 128)
    d_qk    = config['model'].get('d_qk', 16)
    n_heads = config['model'].get('n_heads', 4)
    dropout = config['model'].get('dropout_emb', 0.1)

    # ── Path handling: Lightning trainer joins data_dir + dataset ─────
    # data lives directly in args.data_dir, so we split parent / name
    data_parent  = str(Path(args.data_dir).parent)   # e.g. "causaliT/data"
    dataset_name = str(Path(args.data_dir).name)      # e.g. "surrogate_training"

    # ── Build the config dict expected by the Lightning pipeline ──────
    lt_config = OmegaConf.create({
        'model': {
            'model_object': casualit_model,
            'kwargs': {
                'model': casualit_model,

                # Shared embedding (same table for S, X, Y – global variable IDs)
                'use_independent_embeddings': False,
                'ds_embed_shared': {
                    'setting': {'d_model': d_model, 'sparse_grad': False},
                    'modules': [
                        {'idx': 0, 'embed': 'linear', 'label': 'value',
                         'kwargs': {'input_dim': 1, 'embedding_dim': d_model}},
                        {'idx': 1, 'embed': 'nn_embedding', 'label': 'variable',
                         'kwargs': {'num_embeddings': total_vars,
                                    'embedding_dim': d_model,
                                    'padding_idx': 0, 'sparse': False,
                                    'max_norm': 1}},
                        {'idx': 0, 'embed': 'mask', 'label': 'value_missing',
                         'kwargs': {}},
                        {'idx': 1, 'embed': 'pass', 'label': 'order',
                         'kwargs': {}},
                    ],
                },
                'comps_embed_shared': 'summation',
                'val_idx_X': 0,  # value column to blank / predict

                # Attention (plain scaled dot-product, no causal masks)
                'dec1_cross_attention_type': 'ScaledDotProduct',
                'dec1_cross_mask_type':      'Uniform',
                'dec1_self_attention_type':   'ScaledDotProduct',
                'dec1_self_mask_type':        'Uniform',
                'dec2_cross_attention_type':  'ScaledDotProduct',
                'dec2_cross_mask_type':       'Uniform',
                'dec2_self_attention_type':   'ScaledDotProduct',
                'dec2_self_mask_type':        'Uniform',
                'n_heads':          n_heads,
                'dec1_causal_mask': False,
                'dec2_causal_mask': False,

                # Architecture
                'd1_layers':     config['model'].get('d_layers', 1),
                'd2_layers':     config['model'].get('d_layers', 1),
                'activation':    config['model'].get('activation', 'gelu'),
                'norm':          'layer',
                'use_final_norm': True,
                'device':         'cuda' if (args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available())) else 'cpu',

                # Dimensions
                'out_dim':  1,
                'd_ff':     d_ff,
                'd_model':  d_model,
                'd_qk':     d_qk,
                'S_seq_len': n_s,
                'X_seq_len': n_x,
                'Y_seq_len': n_y,

                # Dropout
                'dropout_emb':                  dropout,
                'dropout_attn_out':             dropout,
                'dropout_ff':                   dropout,
                'dec1_cross_dropout_qkv':       dropout,
                'dec1_cross_attention_dropout':  dropout,
                'dec1_self_dropout_qkv':        dropout,
                'dec1_self_attention_dropout':   dropout,
                'dec2_cross_dropout_qkv':       dropout,
                'dec2_cross_attention_dropout':  dropout,
                'dec2_self_dropout_qkv':        dropout,
                'dec2_self_attention_dropout':   dropout,
            },
        },

        'data': {
            'dataset':        dataset_name,
            'filename_input': 'ds.npz',
            'val_idx':   0,   # value column
            'val_idx_X': 0,
            'val_idx_Y': 0,
            'test_ds_ixd':    None,
            'max_data_size':  None,
            'S_seq_len': n_s,
            'X_seq_len': n_x,
            'Y_seq_len': n_y,
        },

        'training': {
            'optimizer':      'adamw',
            'lr':             config['training'].get('learning_rate', 1e-3),
            'weight_decay':   config['training'].get('weight_decay', 0.01),
            'use_scheduler':  config['training'].get('use_scheduler', True),
            'loss_fn':        'mse',
            'loss_weight_x':  config['training'].get('loss_weight_x', 1.0),
            'loss_weight_y':  config['training'].get('loss_weight_y', 1.0),
            'teacher_forcing': True,
            'batch_size':     config['training'].get('batch_size', 64),
            'max_epochs':     config['training'].get('max_epochs', 200),
            'k_fold':         max(config['training'].get('k_fold', 2), 2),
            'seed':           config['training'].get('seed', 42),
            'save_ckpt_every_n_epochs': 50,
            'log_entropy':       False,
            'log_acyclicity':    False,
            'use_hard_masks':    False,
            'use_in_context_masks': False,
        },

        'special':    {'mode': []},
        'evaluation': {'functions': ['eval_train_metrics']},
    })

    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n[2/4] Training {casualit_model} with PyTorch Lightning...")
    print(f"  d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}")
    print(f"  S_seq_len={n_s}, X_seq_len={n_x}, Y_seq_len={n_y}")
    print(f"  num_embeddings={total_vars} (shared)")

    results_df = lightning_trainer(
        config=lt_config,
        data_dir=data_parent,
        save_dir=save_dir,
        cluster=False,
    )

    # Copy best checkpoint to the expected location
    best_ckpt_src = Path(save_dir) / 'k_0' / 'best_checkpoint.ckpt'
    best_ckpt_dst = Path(save_dir) / 'best_model.ckpt'
    if best_ckpt_src.exists():
        import shutil
        shutil.copy2(best_ckpt_src, best_ckpt_dst)
        print(f"  Best checkpoint copied to: {best_ckpt_dst}")

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"  Checkpoints: {save_dir}")
    if results_df is not None:
        print(f"  Results:\n{results_df.to_string()}")


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
    parser.add_argument('--data_dir', type=str, default='causaliT/data/surrogate_training',
                       help='Data directory')
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

    # Load config
    config = SURROGATE_CONFIG.copy()

    # Override config with command line args
    if args.epochs:
        config['training']['max_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate

    casualit_model = config['model'].get('casualit_model', 'proT')
    # --generate_data takes priority over use_existing_dataset config
    if args.generate_data:
        use_existing = False
    else:
        use_existing = args.use_existing_dataset or config['data'].get('use_existing_dataset', False)

    print("="*70)
    print("CasualiT Surrogate Training")
    print("="*70)
    print(f"Model type: {casualit_model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Data directory: {args.data_dir}")
    print(f"Use existing dataset: {use_existing}")

    # ── Data preparation ───────────────────────────────────────────────────
    data_path = Path(args.data_dir)

    if use_existing:
        # Convert from full_trajectories.pt if needed
        from causaliT.surrogate_training.convert_dataset import convert_trajectories_to_causalit_format

        dataset_path = (args.trajectories_path
                        or config['data'].get('dataset_path',
                                              'scm_ds/data_predictor/trajectories/full_trajectories.pt'))
        print("\n[1/4] Converting existing dataset...")
        convert_trajectories_to_causalit_format(
            trajectories_path=dataset_path,
            output_dir=args.data_dir,
            model_type=casualit_model,
            train_frac=config['data'].get('train_frac', 0.70),
            val_frac=config['data'].get('val_frac', 0.15),
            test_frac=config['data'].get('test_frac', 0.15),
            seed=config['data'].get('random_seed', 42),
        )

        if args.data_only:
            print("\nData conversion complete.")
            return

    elif args.data_only or args.generate_data or not (data_path / 'ds.npz').exists():
        print("\n[1/4] Generating training data...")
        stats = generate_all_datasets(config, args.data_dir, device=args.device)

        if args.data_only:
            print("\n" + "="*70)
            print("Data Generation Complete!")
            print("="*70)
            print(f"Data saved to: {args.data_dir}")
            for split, s in stats.items():
                print(f"  {split}: {s['n_samples']} samples, F = {s['F_mean']:.4f} +/- {s['F_std']:.4f}")
            return
    else:
        print("\n[1/4] Using existing training data")

    # ── Load & train ───────────────────────────────────────────────────────
    if casualit_model in ('StageCausaliT', 'SingleCausalLayer'):
        _train_stage_causal_lightning(casualit_model, config, args)
        return

    # proT path: use Lightning pipeline (same as StageCausaliT)
    _train_prot_lightning(config, args)


if __name__ == '__main__':
    main()
