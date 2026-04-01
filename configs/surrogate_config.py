"""
Surrogate configuration overrides for causaliT training.

This file defines overrides that are merged ON TOP of a base causaliT YAML
config. Only parameters that differ from the base YAML need to be specified.

Usage:
    from configs.surrogate_config import SURROGATE_CONFIG

    base = OmegaConf.load(SURROGATE_CONFIG['base_yaml'])
    overrides = OmegaConf.create(SURROGATE_CONFIG['overrides'])
    config = OmegaConf.merge(base, overrides)

The 'base_yaml' field points to the YAML template inside causaliT/causaliT/config/.
The 'overrides' dict mirrors the YAML structure — only keys present here
will be overwritten; everything else keeps its YAML default.
"""

from pathlib import Path

_CAUSALIT_CONFIG_DIR = Path(__file__).parent.parent / 'causaliT' / 'causaliT' / 'config'

SURROGATE_CONFIG = {
    # Which YAML to use as base template
    'base_yaml': str(_CAUSALIT_CONFIG_DIR / 'config_stage_causaliT_dyconex.yaml'),

    # Overrides (same structure as the YAML)
    'overrides': {

        'experiment': {
            'd_model_set': 100,
            'd1_layers': 1,
            'd2_layers': 1,
            'n_heads': 1,
            'd_ff': 200,
            'd_qk': 100,
            'dropout': 0.1,
            'lr': 0.0001,
            'batch_size': 50,
            'max_epochs': 500,
            'loss_weight_x': 1.0,
            'loss_weight_y': 1.0,
        },

        'model': {
            'model_object': 'StageCausaliT',
        },

        'training': {
            'k_fold': 5,
            'seed': 42,
            'loss_fn': 'mse',
            'use_scheduler': False,
            'save_ckpt_every_n_epochs': 100,
        },

        'data': {
            'dataset': 'azimuth_surrogate',
            'filename_input': 'ds.npz',
            'train_file': None,
            'test_file': None,
        },

        'evaluation': {
            'functions': [
                'eval_train_metrics',
            ],
        },
    },
}
