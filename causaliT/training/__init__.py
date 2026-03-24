"""
ProT Training Infrastructure Package

This package contains all training-related components:
- forecasters: Lightning model wrappers (TransformerForecaster, StageCausalForecaster)
- callbacks: Training and model monitoring callbacks
- dataloader: Data loading utilities (ProcessDataModule, StageCausalDataModule)
- trainer: Main training orchestration
- experiment_control: Experiment management and sweeps
"""

def __getattr__(name):
    """Lazy imports — training components require pytorch_lightning."""
    _map = {
        'TransformerForecaster': ('.forecasters', 'TransformerForecaster'),
        'StageCausalForecaster': ('.forecasters', 'StageCausalForecaster'),
        'ProcessDataModule': ('.dataloader', 'ProcessDataModule'),
        'StageCausalDataModule': ('.stage_causal_dataloader', 'StageCausalDataModule'),
        'trainer': ('.trainer', 'trainer'),
        'get_model_class': ('.trainer', 'get_model_class'),
        'create_model_instance': ('.trainer', 'create_model_instance'),
        'get_dataloader': ('.trainer', 'get_dataloader'),
        'combination_sweep': ('.experiment_control', 'combination_sweep'),
        'update_config': ('.experiment_control', 'update_config'),
    }
    if name in _map:
        import importlib
        module = importlib.import_module(_map[name][0], __name__)
        return getattr(module, _map[name][1])
    raise AttributeError(f"module 'causaliT.training' has no attribute {name!r}")


__all__ = [
    'TransformerForecaster',
    'StageCausalForecaster',
    'ProcessDataModule',
    'StageCausalDataModule',
    'trainer',
    'get_model_class',
    'create_model_instance',
    'get_dataloader',
    'combination_sweep',
    'update_config',
]
