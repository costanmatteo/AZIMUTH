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

    # Overrides (same structure as the YAML)
    # Only keys present here will be overwritten; everything else keeps
    # its YAML default.
    'overrides': {

        'data': {
            'dataset': 'azimuth_surrogate',
        },

    },
}
