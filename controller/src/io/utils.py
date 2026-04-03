"""
Utility di conversione I/O tra numpy e torch, e conteggio parametri.
"""

import torch
import numpy as np


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
