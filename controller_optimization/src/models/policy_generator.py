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


class ResidualPolicyGenerator(nn.Module):
    """
    Residual Policy Learning: action = BC_action(frozen) + residual(state)

    This architecture implements a two-phase training approach:
    1. BC Pre-training Phase: Train the base_policy with pure BC loss
    2. Residual Learning Phase: Freeze base_policy, train residual_network

    The final action is computed as:
        action = base_policy(state) + residual_scale * residual_network(state)

    Args:
        input_size (int): Dimension of input (state)
        output_size (int): Dimension of output (action)
        base_hidden_sizes (list): Hidden layer sizes for base BC policy
        residual_hidden_sizes (list): Hidden layer sizes for residual network
        dropout_rate (float): Dropout rate for both networks
        residual_scale (float): Scaling factor for residual output (default: 0.1)
                               Smaller values make learning more stable
        use_batchnorm (bool): Whether to use batch normalization
    """

    def __init__(self, input_size, output_size,
                 base_hidden_sizes=[64, 32],
                 residual_hidden_sizes=[32, 16],
                 dropout_rate=0.1,
                 residual_scale=0.1,
                 use_batchnorm=False):
        super(ResidualPolicyGenerator, self).__init__()

        self.output_size = output_size
        self.residual_scale = residual_scale
        self._residual_active = False  # Initially, only base policy is used

        # === Base Policy (BC Network) ===
        # This will be pre-trained with BC loss and then frozen
        base_layers = []
        prev_size = input_size
        for hidden_size in base_hidden_sizes:
            base_layers.append(nn.Linear(prev_size, hidden_size))
            if use_batchnorm:
                base_layers.append(nn.BatchNorm1d(hidden_size))
            base_layers.append(nn.SiLU())
            base_layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        self.base_network = nn.Sequential(*base_layers)
        self.base_output_head = nn.Linear(prev_size, output_size)

        # === Residual Network ===
        # This learns corrections to the base policy
        # Initialized to output near-zero values for stable start
        residual_layers = []
        prev_size = input_size
        for hidden_size in residual_hidden_sizes:
            residual_layers.append(nn.Linear(prev_size, hidden_size))
            if use_batchnorm:
                residual_layers.append(nn.BatchNorm1d(hidden_size))
            residual_layers.append(nn.SiLU())
            residual_layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        self.residual_network = nn.Sequential(*residual_layers)
        self.residual_output_head = nn.Linear(prev_size, output_size)

        # Initialize residual output head to near-zero for stable start
        # This ensures initial behavior is dominated by BC policy
        nn.init.zeros_(self.residual_output_head.weight)
        nn.init.zeros_(self.residual_output_head.bias)

    def freeze_base_policy(self):
        """
        Freeze the base (BC) policy parameters.
        Call this after BC pre-training phase to start residual learning.
        """
        for param in self.base_network.parameters():
            param.requires_grad = False
        for param in self.base_output_head.parameters():
            param.requires_grad = False
        self._residual_active = True

    def unfreeze_base_policy(self):
        """
        Unfreeze the base (BC) policy parameters.
        Used during BC pre-training phase.
        """
        for param in self.base_network.parameters():
            param.requires_grad = True
        for param in self.base_output_head.parameters():
            param.requires_grad = True
        self._residual_active = False

    def set_residual_active(self, active: bool):
        """
        Set whether residual network should be active.
        When False, only base policy output is used.
        When True, output = base_policy + residual_scale * residual.
        """
        self._residual_active = active

    def is_residual_active(self) -> bool:
        """Check if residual network is active."""
        return self._residual_active

    def get_base_action(self, x):
        """
        Get action from base (BC) policy only.

        Args:
            x (torch.Tensor): Input state, shape (batch_size, input_size)

        Returns:
            torch.Tensor: Base action, shape (batch_size, output_size)
        """
        features = self.base_network(x)
        return self.base_output_head(features)

    def get_residual(self, x):
        """
        Get residual correction only.

        Args:
            x (torch.Tensor): Input state, shape (batch_size, input_size)

        Returns:
            torch.Tensor: Residual correction, shape (batch_size, output_size)
        """
        features = self.residual_network(x)
        return self.residual_output_head(features)

    def forward(self, x):
        """
        Forward pass.

        During BC pre-training (residual_active=False):
            action = base_policy(x)

        During residual learning (residual_active=True):
            action = base_policy(x) + residual_scale * residual(x)

        Args:
            x (torch.Tensor): Input state, shape (batch_size, input_size)

        Returns:
            torch.Tensor: Actions for next process, shape (batch_size, output_size)
        """
        # Base policy action (may be frozen)
        base_action = self.get_base_action(x)

        if self._residual_active:
            # Add residual correction
            residual = self.get_residual(x)
            action = base_action + self.residual_scale * residual
        else:
            # BC pre-training phase: only base policy
            action = base_action

        return action

    def get_trainable_params_info(self):
        """
        Get info about trainable parameters in each component.

        Returns:
            dict: {
                'base_policy': {'params': int, 'trainable': bool},
                'residual_network': {'params': int, 'trainable': bool},
                'total_trainable': int
            }
        """
        base_params = sum(p.numel() for p in self.base_network.parameters()) + \
                      sum(p.numel() for p in self.base_output_head.parameters())
        base_trainable = sum(p.numel() for p in self.base_network.parameters() if p.requires_grad) + \
                         sum(p.numel() for p in self.base_output_head.parameters() if p.requires_grad)

        residual_params = sum(p.numel() for p in self.residual_network.parameters()) + \
                          sum(p.numel() for p in self.residual_output_head.parameters())
        residual_trainable = sum(p.numel() for p in self.residual_network.parameters() if p.requires_grad) + \
                             sum(p.numel() for p in self.residual_output_head.parameters() if p.requires_grad)

        return {
            'base_policy': {
                'params': base_params,
                'trainable': base_trainable > 0,
                'trainable_params': base_trainable
            },
            'residual_network': {
                'params': residual_params,
                'trainable': residual_trainable > 0,
                'trainable_params': residual_trainable
            },
            'total_trainable': base_trainable + residual_trainable
        }


def create_residual_policy_generator(input_size, output_size,
                                     base_hidden_sizes=[64, 32],
                                     residual_hidden_sizes=[32, 16],
                                     residual_scale=0.1):
    """
    Factory function for creating ResidualPolicyGenerator.

    Args:
        input_size (int): Input dimension
        output_size (int): Output dimension
        base_hidden_sizes (list): Hidden layers for base BC policy
        residual_hidden_sizes (list): Hidden layers for residual network
        residual_scale (float): Scaling factor for residual output

    Returns:
        ResidualPolicyGenerator: Configured residual policy generator
    """
    return ResidualPolicyGenerator(
        input_size=input_size,
        output_size=output_size,
        base_hidden_sizes=base_hidden_sizes,
        residual_hidden_sizes=residual_hidden_sizes,
        residual_scale=residual_scale
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

    # Test ResidualPolicyGenerator
    print("\n" + "="*60)
    print("Testing ResidualPolicyGenerator...")

    residual_policy = ResidualPolicyGenerator(
        input_size=input_size,
        output_size=output_size,
        base_hidden_sizes=[64, 32],
        residual_hidden_sizes=[32, 16],
        residual_scale=0.1
    )

    # Test BC pre-training phase (residual_active=False)
    print("\n--- BC Pre-training Phase ---")
    actions_bc = residual_policy(x)
    params_info = residual_policy.get_trainable_params_info()
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {actions_bc.shape}")
    print(f"Residual active: {residual_policy.is_residual_active()}")
    print(f"Base policy params: {params_info['base_policy']['params']:,} (trainable: {params_info['base_policy']['trainable']})")
    print(f"Residual params: {params_info['residual_network']['params']:,} (trainable: {params_info['residual_network']['trainable']})")
    print(f"Total trainable: {params_info['total_trainable']:,}")

    # Test residual learning phase
    print("\n--- Residual Learning Phase ---")
    residual_policy.freeze_base_policy()  # Freeze BC, activate residual
    actions_residual = residual_policy(x)
    params_info = residual_policy.get_trainable_params_info()
    print(f"Residual active: {residual_policy.is_residual_active()}")
    print(f"Base policy trainable: {params_info['base_policy']['trainable']}")
    print(f"Residual trainable: {params_info['residual_network']['trainable']}")
    print(f"Total trainable: {params_info['total_trainable']:,}")

    # Verify residual starts near zero
    base_action = residual_policy.get_base_action(x)
    residual = residual_policy.get_residual(x)
    print(f"\nBase action mean: {base_action.mean().item():.4f}")
    print(f"Residual mean: {residual.mean().item():.6f} (should be ~0)")
    print(f"Action = base + {residual_policy.residual_scale} * residual")

    print("\n✓ ResidualPolicyGenerator test passed!")
