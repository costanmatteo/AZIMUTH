"""
Neural Network for Machinery Output Prediction

This module contains the definition of the neural network that predicts
machinery output values (e.g. pressure, temperature) based on the input
parameters chosen to operate the machine.
"""

import torch
import torch.nn as nn


class MachineryPredictor(nn.Module):
    """
    Feedforward Neural Network for predicting machinery output.

    Architecture:
    - Input Layer: operational parameters (to be configured)
    - Hidden Layers: fully connected hidden layers
    - Output Layer: predicted values (pressure, temperature, etc.)

    Args:
        input_size (int): Number of input parameters
        hidden_sizes (list): List with dimensions of hidden layers
        output_size (int): Number of output values to predict
        dropout_rate (float): Dropout rate for regularization (default: 0.2)

    Example:
        >>> model = MachineryPredictor(
        ...     input_size=10,      # 10 operational parameters
        ...     hidden_sizes=[64, 32, 16],  # 3 hidden layers
        ...     output_size=5       # 5 output values
        ... )
        >>> x = torch.randn(32, 10)  # batch of 32 examples
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([32, 5])
    """

    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2, use_batchnorm=False):
        super(MachineryPredictor, self).__init__()

        # Build layers
        layers = []
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))  # normalize activations per batch
            layers.append(nn.ReLU())  # Funzione di attivazione
            layers.append(nn.Dropout(dropout_rate))  # Regolarizzazione
            prev_size = hidden_size

        # Output layer (no activation, linear regression)
        layers.append(nn.Linear(prev_size, output_size))

        # Combine all layers into a sequential module
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)

        Returns:
            torch.Tensor: Output predictions of shape (batch_size, output_size)
        """
        return self.network(x)


# Example common configurations
def create_small_model(input_size, output_size):
    """Small model for limited datasets"""
    return MachineryPredictor(
        input_size=input_size,
        hidden_sizes=[32, 16],
        output_size=output_size,
        dropout_rate=0.1
    )


def create_medium_model(input_size, output_size):
    """Medium model for medium-sized datasets"""
    return MachineryPredictor(
        input_size=input_size,
        hidden_sizes=[128, 64, 32],
        output_size=output_size,
        dropout_rate=0.2
    )


def create_large_model(input_size, output_size):
    """Large model for large datasets"""
    return MachineryPredictor(
        input_size=input_size,
        hidden_sizes=[256, 128, 64, 32],
        output_size=output_size,
        dropout_rate=0.3
    )
