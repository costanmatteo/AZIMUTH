"""
Policy Generator: genera parametri di input ottimali per il processo successivo.

Architettura simile a UncertaintyPredictor ma output deterministico.
"""

import torch
import torch.nn as nn


class PolicyGenerator(nn.Module):
    """
    Policy generator per controller optimization.

    Args:
        input_size (int): Dimensione input concatenato
        hidden_sizes (list): Hidden layers
        output_size (int): Dimensione output (input per processo successivo)
        dropout_rate (float): Dropout
        use_batchnorm (bool): Batch normalization
    """

    def __init__(self, input_size, hidden_sizes, output_size,
                 dropout_rate=0.1, use_batchnorm=False):
        super(PolicyGenerator, self).__init__()

        self.output_size = output_size

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

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)

        Returns:
            torch.Tensor: Actions for next process, shape (batch_size, output_size)
        """
        features = self.network(x)
        actions = self.output_head(features)
        return actions


# Convenience functions for creating common architectures

def create_small_policy_generator(input_size, output_size):
    """Small policy generator for simple control tasks"""
    return PolicyGenerator(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=[32, 16],
        dropout_rate=0.05,
        use_batchnorm=False
    )


def create_medium_policy_generator(input_size, output_size):
    """Medium policy generator for moderate complexity tasks"""
    return PolicyGenerator(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=[64, 32],
        dropout_rate=0.1,
        use_batchnorm=False
    )


def create_large_policy_generator(input_size, output_size):
    """Large policy generator for complex control tasks"""
    return PolicyGenerator(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=[128, 64, 32],
        dropout_rate=0.15,
        use_batchnorm=False
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

    # Create model
    policy = create_medium_policy_generator(input_size, output_size)

    # Test forward pass
    batch_size = 8
    x = torch.randn(batch_size, input_size)

    actions = policy(x)

    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {actions.shape}")
    print(f"\nTotal parameters: {sum(p.numel() for p in policy.parameters()):,}")

    print("\n✓ PolicyGenerator test passed!")
