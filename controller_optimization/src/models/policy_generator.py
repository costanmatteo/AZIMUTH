"""
Policy Generator: genera parametri di input ottimali per il processo successivo.

Architettura simile a UncertaintyPredictor ma con output bounded.
Output è limitato tra min e max derivati dall'UncertaintyPredictor.
"""

import torch
import torch.nn as nn


class PolicyGenerator(nn.Module):
    """
    Policy generator per controller optimization with bounded outputs.

    The output layer uses tanh activation to produce values in [-1, 1],
    which are then scaled to [output_min, output_max] bounds derived from
    the UncertaintyPredictor's training data.

    Args:
        input_size (int): Dimensione input concatenato
        hidden_sizes (list): Hidden layers
        output_size (int): Dimensione output (input per processo successivo)
        dropout_rate (float): Dropout
        use_batchnorm (bool): Batch normalization
        output_min (torch.Tensor): Minimum bounds for each output dimension
        output_max (torch.Tensor): Maximum bounds for each output dimension
    """

    def __init__(self, input_size, hidden_sizes, output_size,
                 dropout_rate=0.1, use_batchnorm=False,
                 output_min=None, output_max=None):
        super(PolicyGenerator, self).__init__()

        self.output_size = output_size

        # Register bounds as buffers (not parameters, but saved with model)
        if output_min is not None:
            self.register_buffer('output_min', output_min)
        else:
            self.register_buffer('output_min', None)

        if output_max is not None:
            self.register_buffer('output_max', output_max)
        else:
            self.register_buffer('output_max', None)

        # Build shared hidden layers (same architecture as UncertaintyPredictor)
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.SiLU())  # Same activation as UncertaintyPredictor
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        self.network = nn.Sequential(*layers)

        # Single output head (no variance head)
        self.output_head = nn.Linear(prev_size, output_size)

    def set_bounds(self, output_min, output_max):
        """
        Set or update output bounds.

        Args:
            output_min (torch.Tensor): Minimum bounds, shape (output_size,)
            output_max (torch.Tensor): Maximum bounds, shape (output_size,)
        """
        device = next(self.parameters()).device
        self.output_min = output_min.to(device)
        self.output_max = output_max.to(device)

    def forward(self, x):
        """
        Forward pass with bounded output.

        Uses tanh activation followed by affine scaling to enforce bounds:
            normalized = 0.5 * (tanh(raw) + 1)  -> [0, 1]
            bounded = min + normalized * (max - min)  -> [min, max]

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)

        Returns:
            torch.Tensor: Bounded actions for next process, shape (batch_size, output_size)
        """
        features = self.network(x)
        raw_actions = self.output_head(features)

        # Apply bounded activation if bounds are set
        if self.output_min is not None and self.output_max is not None:
            # tanh -> [-1, 1], then scale to [0, 1]
            normalized = 0.5 * (torch.tanh(raw_actions) + 1.0)
            # Scale to [min, max]
            actions = self.output_min + normalized * (self.output_max - self.output_min)
        else:
            # Fallback to unbounded (for backward compatibility)
            actions = raw_actions

        return actions


# Convenience functions for creating common architectures

def create_small_policy_generator(input_size, output_size, output_min=None, output_max=None):
    """Small policy generator for simple control tasks"""
    return PolicyGenerator(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=[32, 16],
        dropout_rate=0.05,
        use_batchnorm=False,
        output_min=output_min,
        output_max=output_max
    )


def create_medium_policy_generator(input_size, output_size, output_min=None, output_max=None):
    """Medium policy generator for moderate complexity tasks"""
    return PolicyGenerator(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=[64, 32],
        dropout_rate=0.1,
        use_batchnorm=False,
        output_min=output_min,
        output_max=output_max
    )


def create_large_policy_generator(input_size, output_size, output_min=None, output_max=None):
    """Large policy generator for complex control tasks"""
    return PolicyGenerator(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=[128, 64, 32],
        dropout_rate=0.15,
        use_batchnorm=False,
        output_min=output_min,
        output_max=output_max
    )


if __name__ == '__main__':
    # Test PolicyGenerator
    print("Testing PolicyGenerator...")

    # Example: previous process had 2 inputs, 1 output
    # Current input = concat([prev_inputs, prev_outputs_mean, prev_outputs_var])
    # = [2, 1, 1] = 4 features
    input_size = 4

    # Next process needs 2 inputs
    output_size = 2

    # Define bounds (e.g., from UncertaintyPredictor training data)
    output_min = torch.tensor([0.5, 30.0])   # e.g., Concentration [0.5, 3.0], Duration [30, 180]
    output_max = torch.tensor([3.0, 180.0])

    # Create model with bounds
    policy = create_medium_policy_generator(
        input_size, output_size,
        output_min=output_min,
        output_max=output_max
    )

    # Test forward pass
    batch_size = 8
    x = torch.randn(batch_size, input_size)

    actions = policy(x)

    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {actions.shape}")
    print(f"\nOutput bounds:")
    print(f"  Min: {output_min.numpy()}")
    print(f"  Max: {output_max.numpy()}")
    print(f"\nActual output range:")
    print(f"  Min: {actions.min(dim=0).values.detach().numpy()}")
    print(f"  Max: {actions.max(dim=0).values.detach().numpy()}")
    print(f"\nTotal parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Verify bounds are respected
    assert torch.all(actions >= output_min), "Output below minimum bound!"
    assert torch.all(actions <= output_max), "Output above maximum bound!"

    print("\n✓ PolicyGenerator bounded output test passed!")
