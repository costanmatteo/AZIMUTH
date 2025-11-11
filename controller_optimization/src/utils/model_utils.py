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

# Load preprocessing module for pickle compatibility
spec_preprocessing = importlib.util.spec_from_file_location(
    "preprocessing",
    UNCERTAINTY_PREDICTOR_PATH / "src" / "data" / "preprocessing.py"
)
preprocessing = importlib.util.module_from_spec(spec_preprocessing)
sys.modules['preprocessing'] = preprocessing  # Register for pickle
spec_preprocessing.loader.exec_module(preprocessing)


def load_uncertainty_predictor(checkpoint_path, input_dim, output_dim, model_config, device='cpu'):
    """
    Carica uncertainty predictor pre-addestrato.

    Args:
        checkpoint_path (Path or str): Path to .pth file
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        model_config (dict): Model configuration (hidden_sizes, dropout, etc.)
        device (str): Device to load model on

    Returns:
        model (UncertaintyPredictor): Modello con pesi caricati, frozen
    """
    checkpoint_path = Path(checkpoint_path)

    # Create model
    model = UncertaintyPredictor(
        input_size=input_dim,
        output_size=output_dim,
        hidden_sizes=model_config['hidden_sizes'],
        dropout_rate=model_config.get('dropout_rate', 0.2),
        use_batchnorm=model_config.get('use_batchnorm', False),
        min_variance=model_config.get('min_variance', 1e-6)
    )

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location=device)
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

    return preprocessor


def save_policy_generator(policy, path, metadata=None):
    """
    Salva policy generator con metadata.

    Args:
        policy: PolicyGenerator instance
        path (Path or str): Save path
        metadata (dict): Optional metadata to save alongside
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save model weights
    torch.save(policy.state_dict(), path)

    # Save metadata if provided
    if metadata is not None:
        metadata_path = path.parent / f"{path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def load_policy_generator(path, input_dim, output_dim, model_config, device='cpu'):
    """
    Carica policy generator.

    Args:
        path (Path or str): Path to .pth file
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        model_config (dict): Model configuration
        device (str): Device to load model on

    Returns:
        policy: PolicyGenerator instance
    """
    from controller_optimization.src.models.policy_generator import PolicyGenerator

    path = Path(path)

    # Create model
    policy = PolicyGenerator(
        input_size=input_dim,
        output_size=output_dim,
        hidden_sizes=model_config.get('hidden_sizes', [64, 32]),
        dropout_rate=model_config.get('dropout', 0.1),
        use_batchnorm=model_config.get('use_batchnorm', False)
    )

    # Load weights
    state_dict = torch.load(path, map_location=device)
    policy.load_state_dict(state_dict)

    # Move to device
    policy = policy.to(device)

    return policy


def convert_numpy_to_tensor(trajectory, device='cpu'):
    """
    Converte trajectory da numpy arrays a torch tensors.

    Usato per passare baseline_trajectory (numpy) al surrogate (torch).

    Args:
        trajectory (dict): Trajectory with numpy arrays
        device (str): Device for tensors

    Returns:
        dict: Trajectory with torch tensors
    """
    tensor_trajectory = {}

    for process_name, data in trajectory.items():
        tensor_trajectory[process_name] = {
            'inputs': torch.tensor(data['inputs'], dtype=torch.float32, device=device),
            'outputs_mean': torch.tensor(data['outputs'], dtype=torch.float32, device=device),
            'outputs_var': torch.zeros_like(
                torch.tensor(data['outputs'], dtype=torch.float32, device=device)
            )  # Baseline has no variance prediction
        }

    return tensor_trajectory


def convert_tensor_to_numpy(trajectory):
    """
    Converte trajectory da torch tensors a numpy arrays.

    Args:
        trajectory (dict): Trajectory with torch tensors

    Returns:
        dict: Trajectory with numpy arrays
    """
    numpy_trajectory = {}

    for process_name, data in trajectory.items():
        numpy_trajectory[process_name] = {
            'inputs': data['inputs'].detach().cpu().numpy(),
            'outputs_mean': data['outputs_mean'].detach().cpu().numpy(),
            'outputs_var': data['outputs_var'].detach().cpu().numpy()
        }

    return numpy_trajectory


def count_parameters(model):
    """
    Conta parametri totali di un modello.

    Args:
        model: PyTorch model

    Returns:
        int: Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    """
    Conta parametri trainable di un modello.

    Args:
        model: PyTorch model

    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
