"""
Utility per load/save di modelli.
"""

import sys
from pathlib import Path
import torch
import pickle
import json
import numpy as np
import importlib.util

# Add uncertainty_predictor to path
REPO_ROOT = Path(__file__).parent.parent.parent.parent
UNCERTAINTY_PREDICTOR_PATH = REPO_ROOT / 'uncertainty_predictor'

# CRITICAL: Add uncertainty_predictor to sys.path FIRST
# This allows the loaded modules to import their dependencies
if str(UNCERTAINTY_PREDICTOR_PATH) not in sys.path:
    sys.path.insert(0, str(UNCERTAINTY_PREDICTOR_PATH))

# Load UncertaintyPredictor explicitly
spec_nn = importlib.util.spec_from_file_location(
    "uncertainty_nn",
    UNCERTAINTY_PREDICTOR_PATH / "src" / "models" / "uncertainty_nn.py"
)
uncertainty_nn = importlib.util.module_from_spec(spec_nn)
sys.modules['uncertainty_nn'] = uncertainty_nn  # Register for pickle
spec_nn.loader.exec_module(uncertainty_nn)
UncertaintyPredictor = uncertainty_nn.UncertaintyPredictor
EnsembleUncertaintyPredictor = uncertainty_nn.EnsembleUncertaintyPredictor
SWAGUncertaintyPredictor = uncertainty_nn.SWAGUncertaintyPredictor

# Load preprocessing module for pickle compatibility
spec_preprocessing = importlib.util.spec_from_file_location(
    "preprocessing",
    UNCERTAINTY_PREDICTOR_PATH / "src" / "data" / "preprocessing.py"
)
preprocessing = importlib.util.module_from_spec(spec_preprocessing)
sys.modules['preprocessing'] = preprocessing  # Register for pickle
spec_preprocessing.loader.exec_module(preprocessing)


def _infer_architecture_from_state_dict(state_dict):
    """
    Infer input_dim, output_dim, and hidden_sizes from a checkpoint's state_dict.

    Works for single, ensemble, and SWAG models by finding the first/last linear layers.

    Returns:
        (input_dim, output_dim, hidden_sizes) — any may be None if not inferrable
    """
    # Determine key prefix based on model type
    if any(k.startswith('models.') for k in state_dict):
        # Ensemble: use first sub-model (models.0.*)
        prefix = 'models.0.'
    elif any(k.startswith('base_model.') for k in state_dict):
        # SWAG: use base_model.*
        prefix = 'base_model.'
    else:
        prefix = ''

    # Collect shared_network linear layer weights in order
    layer_weights = {}
    for key, tensor in state_dict.items():
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix):]
        if suffix.startswith('shared_network.') and suffix.endswith('.weight'):
            if tensor.dim() == 2:  # Only Linear layers (skip BatchNorm which is 1D)
                # e.g. "shared_network.0.weight" → index 0
                parts = suffix.split('.')
                layer_idx = int(parts[1])
                layer_weights[layer_idx] = tensor.shape

    if not layer_weights:
        return None, None, None

    # input_dim from first layer's in_features
    first_idx = min(layer_weights.keys())
    input_dim = layer_weights[first_idx][1]

    # hidden_sizes from out_features of each linear layer
    hidden_sizes = [layer_weights[idx][0] for idx in sorted(layer_weights.keys())]

    # output_dim from mean_head
    mean_head_key = f'{prefix}mean_head.weight'
    output_dim = state_dict[mean_head_key].shape[0] if mean_head_key in state_dict else None

    return input_dim, output_dim, hidden_sizes


def load_uncertainty_predictor(checkpoint_path, input_dim, output_dim, model_config, device='cpu'):
    """
    Carica uncertainty predictor pre-addestrato.

    Automatically detects if the checkpoint is an ensemble model and loads
    the appropriate model type.

    Args:
        checkpoint_path (Path or str): Path to .pth file
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        model_config (dict): Model configuration (hidden_sizes, dropout, etc.)
        device (str): Device to load model on

    Returns:
        model (UncertaintyPredictor or EnsembleUncertaintyPredictor): Model with loaded weights, frozen
    """
    checkpoint_path = Path(checkpoint_path)

    # Load weights first to detect model type
    state_dict = torch.load(checkpoint_path, map_location=device)

    # Infer architecture from checkpoint to avoid config/checkpoint mismatches
    ckpt_input_dim, ckpt_output_dim, ckpt_hidden_sizes = _infer_architecture_from_state_dict(state_dict)
    if ckpt_input_dim is not None and ckpt_input_dim != input_dim:
        print(f"  ⚠ Checkpoint input_dim={ckpt_input_dim} differs from config input_dim={input_dim}, using checkpoint value")
        input_dim = ckpt_input_dim
    if ckpt_output_dim is not None and ckpt_output_dim != output_dim:
        print(f"  ⚠ Checkpoint output_dim={ckpt_output_dim} differs from config output_dim={output_dim}, using checkpoint value")
        output_dim = ckpt_output_dim
    if ckpt_hidden_sizes is not None and ckpt_hidden_sizes != model_config['hidden_sizes']:
        print(f"  ⚠ Checkpoint hidden_sizes={ckpt_hidden_sizes} differs from config {model_config['hidden_sizes']}, using checkpoint value")
        model_config = {**model_config, 'hidden_sizes': ckpt_hidden_sizes}

    # Detect model type from state_dict keys
    is_ensemble = any(key.startswith('models.') for key in state_dict.keys())
    is_swag = any(key.startswith('base_model.') for key in state_dict.keys())

    if is_ensemble:
        # Count number of models in ensemble
        model_indices = set()
        for key in state_dict.keys():
            if key.startswith('models.'):
                # Extract model index from "models.X.layer_name"
                idx = int(key.split('.')[1])
                model_indices.add(idx)
        n_models = len(model_indices)

        # Create ensemble model
        model = EnsembleUncertaintyPredictor(
            input_size=input_dim,
            output_size=output_dim,
            hidden_sizes=model_config['hidden_sizes'],
            dropout_rate=model_config.get('dropout_rate', 0.2),
            use_batchnorm=model_config.get('use_batchnorm', False),
            min_variance=model_config.get('min_variance', 1e-6),
            n_models=n_models
        )
    elif is_swag:
        # Create base model, then wrap in SWAG
        base_model = UncertaintyPredictor(
            input_size=input_dim,
            output_size=output_dim,
            hidden_sizes=model_config['hidden_sizes'],
            dropout_rate=model_config.get('dropout_rate', 0.2),
            use_batchnorm=model_config.get('use_batchnorm', False),
            min_variance=model_config.get('min_variance', 1e-6)
        )
        max_rank = state_dict['deviation_matrix'].shape[1] if 'deviation_matrix' in state_dict else 20
        model = SWAGUncertaintyPredictor(base_model, max_rank=max_rank)
    else:
        # Create single model
        model = UncertaintyPredictor(
            input_size=input_dim,
            output_size=output_dim,
            hidden_sizes=model_config['hidden_sizes'],
            dropout_rate=model_config.get('dropout_rate', 0.2),
            use_batchnorm=model_config.get('use_batchnorm', False),
            min_variance=model_config.get('min_variance', 1e-6)
        )

    # Load weights
    model.load_state_dict(state_dict)

    # Move to device
    model = model.to(device)

    # Freeze parameters
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model


def load_preprocessor(scaler_path):
    """
    Carica DataPreprocessor.

    Args:
        scaler_path (Path or str): Path to .pkl file

    Returns:
        preprocessor: DataPreprocessor instance
    """
    scaler_path = Path(scaler_path)

    with open(scaler_path, 'rb') as f:
        preprocessor = pickle.load(f)

    # Backward compatibility: ensure input_min and input_max attributes exist
    # Old saved preprocessors may not have these attributes
    if not hasattr(preprocessor, 'input_min'):
        preprocessor.input_min = None
    if not hasattr(preprocessor, 'input_max'):
        preprocessor.input_max = None

    return preprocessor
