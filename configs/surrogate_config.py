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
    'base_yaml': str(_CAUSALIT_CONFIG_DIR / 'config_proT.yaml'),

    # Overrides (same structure as the YAML)
    # Only keys present here will be overwritten; everything else keeps the YAML default.
    'overrides': {

        'experiment': {
            'd_model_set': 32,
            'e_layers': 2,
            'd_layers': 1,
            'n_heads': 4,
            'd_ff': 64,
            'd_qk': 8,
            'dropout': 0.3,
            'lr': 0.0005,
            'batch_size': 32,
            'max_epochs': 200,
        },

        'training': {
            'k_fold': 3,
            'seed': 42,
        },

        'data': {
            'dataset': 'azimuth_surrogate',
        },
    },
}
