"""
ProT - Process Transformer Package

A transformer-based model for sequence prediction in process chains.
"""

# Import version info
__version__ = "0.1.0"

# Export commonly used paths for convenience
from causaliT.paths import (
    ROOT_DIR,
    DATA_DIR,
    EXPERIMENTS_DIR,
    LOGS_DIR,
    CONFIG_DIR,
    get_dirs,
)

# Export core components (no pytorch_lightning dependency)
from causaliT.core.model import ProT
from causaliT.core.architectures.stage_causal import StageCausaliT


def __getattr__(name):
    """Lazy imports for training components that require pytorch_lightning."""
    if name == 'TransformerForecaster':
        from causaliT.training.forecasters import TransformerForecaster
        return TransformerForecaster
    if name == 'StageCausalForecaster':
        from causaliT.training.forecasters import StageCausalForecaster
        return StageCausalForecaster
    if name == 'ProcessDataModule':
        from causaliT.training.dataloader import ProcessDataModule
        return ProcessDataModule
    raise AttributeError(f"module 'causaliT' has no attribute {name!r}")


__all__ = [
    # Paths
    'ROOT_DIR',
    'DATA_DIR',
    'EXPERIMENTS_DIR',
    'LOGS_DIR',
    'CONFIG_DIR',
    'get_dirs',
    # Core components
    'ProT',
    'StageCausaliT',
    'TransformerForecaster',
    'StageCausalForecaster',
    'ProcessDataModule',
]
