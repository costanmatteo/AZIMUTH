"""
Surrogate configuration overrides for causaliT training.

This file defines overrides that are merged ON TOP of a base causaliT YAML
config. Only parameters that differ from the base YAML need to be specified.

Usage:
    from configs.surrogate_config import SURROGATE_CONFIG

    base = OmegaConf.load(SURROGATE_CONFIG['base_yaml'])
    overrides = OmegaConf.create(SURROGATE_CONFIG['overrides'])
    config = OmegaConf.merge(base, overrides)

Change 'model' to switch architecture — the correct base YAML is selected
automatically from _YAML_REGISTRY.
"""

from pathlib import Path

_CAUSALIT_CONFIG_DIR = Path(__file__).parent.parent / 'causaliT' / 'causaliT' / 'config'

# Maps model name → base YAML config
_YAML_REGISTRY = {
    'proT': _CAUSALIT_CONFIG_DIR / 'config_proT.yaml',
    'NoiseAwareSingleCausalLayer': _CAUSALIT_CONFIG_DIR / 'config_noise_aware_example.yaml',
    'StageCausaliT': _CAUSALIT_CONFIG_DIR / 'config_stage_causaliT_dyconex.yaml',
}

# ─── Choose model here ───────────────────────────────────────────────────
_MODEL = 'proT'
# ──────────────────────────────────────────────────────────────────────────

SURROGATE_CONFIG = {
    'model': _MODEL,
    'base_yaml': str(_YAML_REGISTRY[_MODEL]),

    'overrides': {

        'data': {
            'dataset': 'azimuth_surrogate',  # obbligatorio
            'num_workers': 0,
        },

        'model': {
            # ProT piccolo: 3 token di input (1 per processo), task semplice
            'model_dim': 32,       # dim embedding — 32 basta per 3 processi
            'n_heads': 4,          # teste attenzione (deve dividere model_dim)
            'n_enc_layers': 2,     # encoder layers — non serve profondo
            'n_dec_layers': 1,     # decoder layers
            'dropout': 0.1,
            'ffn_dim': 64,         # feed-forward dim interna (tipicamente 2× model_dim)
        },

        'training': {
            'batch_size': 128,
            'max_epochs': 200,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'patience': 20,        # early stopping
        },

    },
}
