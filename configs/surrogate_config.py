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
    'base_yaml': str(_CAUSALIT_CONFIG_DIR / 'config_noise_aware_example.yaml'),

    # Overrides (same structure as the YAML)
    'overrides': {

        'experiment': {
            'd_model_set': 24,
            'dec_layers': 1,
            'n_heads': 1,
            'd_ff': 48,
            'd_qk': 32,
            'dropout': 0.0,
            'lr': 0.001,
            'batch_size': 64,
            'max_epochs': 200,
        },

        'model': {
            'model_object': 'NoiseAwareSingleCausalLayer',
        },

        'training': {
            'k_fold': 3,
            'seed': 42,
            'loss_fn': 'gaussian_nll',
            'use_scheduler': True,
            'save_ckpt_every_n_epochs': 50,
        },

        'data': {
            'dataset': 'azimuth_surrogate',
        },

        'evaluation': {
            'functions': [
                'eval_train_metrics',
            ],
        },
    },
}
