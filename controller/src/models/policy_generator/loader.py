import json
import torch
from pathlib import Path


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
    from controller.src.models.policy_generator.policy_generator import PolicyGenerator

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
