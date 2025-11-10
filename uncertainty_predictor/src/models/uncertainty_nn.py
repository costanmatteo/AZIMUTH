"""
Neural Network with Uncertainty Quantification and Conditional Process Embeddings

This module contains the definition of a neural network that predicts
both the mean (μ) and variance (σ²) of the output, with optional conditioning
on process ID and environment variables for multi-process learning.

The network outputs two values:
- μ(x): estimated mean value
- σ²(x): estimated uncertainty (variance)

With conditioning enabled, the network learns process-specific and environment-aware
representations through:
- Process ID embeddings
- Environment variable embeddings (continuous + categorical + temporal)
- Conditional normalization layers
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List

# Import conditional layers
try:
    from .conditional_layers import (
        ContextEmbeddingModule,
        ConditionalLayerNorm,
        ConditionalBatchNorm1d
    )
    CONDITIONAL_LAYERS_AVAILABLE = True
except ImportError:
    CONDITIONAL_LAYERS_AVAILABLE = False
    print("Warning: conditional_layers not found. Conditioning will be disabled.")


class UncertaintyPredictor(nn.Module):
    """
    Neural Network with Uncertainty Quantification and optional Conditional Processing.

    Architecture:
    - (Optional) Context Embedding Module: process_id + env variables → context vector
    - Shared Hidden Layers with (Conditional) Normalization
    - Two output heads:
        * Mean head: predicts μ (linear output)
        * Variance head: predicts log(σ²) (then exponentiated for positivity)

    Args:
        input_size (int): Number of input features
        hidden_sizes (list): List with dimensions of shared hidden layers
        output_size (int): Number of output values to predict
        dropout_rate (float): Dropout rate for regularization (default: 0.2)
        use_batchnorm (bool): Whether to use batch normalization (default: False, deprecated if conditioning enabled)
        min_variance (float): Minimum allowed variance for numerical stability (default: 1e-6)

        # Conditioning parameters (optional)
        conditioning_config (dict): Configuration for conditional processing. If None, standard MLP is used.
            Expected keys:
            - 'enable': bool, master switch
            - 'num_processes': int
            - 'd_proc': int
            - 'env_continuous': list of str
            - 'd_env_float': int
            - 'env_categorical': dict {str: int}
            - 'd_env_cat_base': float
            - 'use_time': bool
            - 'time_periods': int
            - 'd_time': int
            - 'd_context': int
            - 'context_mlp_hidden': list
            - 'context_dropout': float
            - 'norm_type': str ('conditional_layer_norm', 'conditional_batch_norm', 'layer_norm', 'batch_norm', 'none')
            - 'use_missing_mask': bool
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout_rate: float = 0.2,
        use_batchnorm: bool = False,
        min_variance: float = 1e-6,
        conditioning_config: Optional[Dict] = None
    ):
        super(UncertaintyPredictor, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.min_variance = min_variance
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate

        # Parse conditioning configuration
        self.conditioning_enabled = False
        self.context_embedding = None
        self.norm_type = 'batch_norm' if use_batchnorm else 'none'

        if conditioning_config is not None and conditioning_config.get('enable', False):
            if not CONDITIONAL_LAYERS_AVAILABLE:
                raise RuntimeError("Conditioning enabled but conditional_layers module not available")

            self.conditioning_enabled = True
            self.norm_type = conditioning_config.get('norm_type', 'conditional_layer_norm')

            # Create context embedding module
            self.context_embedding = ContextEmbeddingModule(
                num_processes=conditioning_config.get('num_processes', 4),
                d_proc=conditioning_config.get('d_proc', 16),
                env_continuous_names=conditioning_config.get('env_continuous', []),
                d_env_float=conditioning_config.get('d_env_float', 16),
                env_categorical_specs=conditioning_config.get('env_categorical', {}),
                d_env_cat_base=conditioning_config.get('d_env_cat_base', 1.6),
                use_time=conditioning_config.get('use_time', True),
                time_periods=conditioning_config.get('time_periods', 4),
                d_time=conditioning_config.get('d_time', 8),
                d_context=conditioning_config.get('d_context', 64),
                context_mlp_hidden=conditioning_config.get('context_mlp_hidden', [128, 64]),
                context_dropout=conditioning_config.get('context_dropout', 0.1),
                use_missing_mask=conditioning_config.get('use_missing_mask', True)
            )

            self.d_context = conditioning_config.get('d_context', 64)
        else:
            self.d_context = None

        # Build shared hidden layers with appropriate normalization
        self.shared_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            self.shared_layers.append(nn.Linear(prev_size, hidden_size))

            # Normalization layer
            norm_layer = self._create_norm_layer(hidden_size)
            self.norm_layers.append(norm_layer)

            # Activation
            self.activation_layers.append(nn.SiLU())

            # Dropout
            self.dropout_layers.append(nn.Dropout(dropout_rate))

            prev_size = hidden_size

        # Output heads
        self.mean_head = nn.Linear(prev_size, output_size)
        self.log_variance_head = nn.Linear(prev_size, output_size)

    def _create_norm_layer(self, hidden_size: int) -> nn.Module:
        """Create appropriate normalization layer based on configuration"""
        if self.norm_type == 'conditional_layer_norm' and self.conditioning_enabled:
            return ConditionalLayerNorm(hidden_size, self.d_context)
        elif self.norm_type == 'conditional_batch_norm' and self.conditioning_enabled:
            return ConditionalBatchNorm1d(hidden_size, self.d_context)
        elif self.norm_type == 'layer_norm':
            return nn.LayerNorm(hidden_size)
        elif self.norm_type == 'batch_norm':
            return nn.BatchNorm1d(hidden_size)
        else:  # 'none'
            return nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        process_id: Optional[torch.Tensor] = None,
        env_continuous: Optional[torch.Tensor] = None,
        env_categorical: Optional[Dict[str, torch.Tensor]] = None,
        timestamp: Optional[torch.Tensor] = None,
        env_masks: Optional[torch.Tensor] = None
    ):
        """
        Forward pass with optional conditioning.

        Args:
            x: Input features of shape (batch_size, input_size)
            process_id: (batch_size,) long tensor with process IDs [0, num_processes-1]
            env_continuous: (batch_size, n_env_continuous) float tensor
            env_categorical: Dict {var_name: (batch_size,) long tensor}
            timestamp: (batch_size,) or (batch_size, 1) float tensor
            env_masks: (batch_size, n_env_continuous) boolean tensor (True=present, False=missing)

        Returns:
            tuple: (mean, variance)
                - mean: Predicted mean values of shape (batch_size, output_size)
                - variance: Predicted variance of shape (batch_size, output_size)
        """
        # Compute context vector if conditioning is enabled
        context = None
        if self.conditioning_enabled and self.context_embedding is not None:
            context = self.context_embedding(
                process_id=process_id,
                env_continuous=env_continuous,
                env_categorical=env_categorical,
                timestamp=timestamp,
                env_masks=env_masks
            )

        # Shared feature extraction with (conditional) normalization
        features = x
        for linear, norm, activation, dropout in zip(
            self.shared_layers,
            self.norm_layers,
            self.activation_layers,
            self.dropout_layers
        ):
            # Linear transformation
            features = linear(features)

            # Normalization (conditional or standard)
            if isinstance(norm, (ConditionalLayerNorm, ConditionalBatchNorm1d)):
                features = norm(features, context)
            else:
                features = norm(features)

            # Activation and dropout
            features = activation(features)
            features = dropout(features)

        # Predict mean
        mean = self.mean_head(features)

        # Predict variance (log-space for stability, then exp)
        log_variance = self.log_variance_head(features)
        variance = torch.exp(log_variance) + self.min_variance

        return mean, variance

    def predict_with_uncertainty(self, x, n_samples=100, **conditioning_kwargs):
        """
        Make predictions with uncertainty estimation using sampling.

        Args:
            x (torch.Tensor): Input tensor
            n_samples (int): Number of samples to draw (default: 100)
            **conditioning_kwargs: Additional conditioning arguments (process_id, env_continuous, etc.)

        Returns:
            dict: Dictionary containing mean, variance, std, samples, etc.
        """
        self.eval()
        with torch.no_grad():
            mean, variance = self.forward(x, **conditioning_kwargs)
            std = torch.sqrt(variance)

            # Sample from the predicted distribution
            samples = []
            for _ in range(n_samples):
                noise = torch.randn_like(mean)
                sample = mean + noise * std
                samples.append(sample)

            samples = torch.stack(samples)  # shape: (n_samples, batch_size, output_size)

            return {
                'mean': mean,
                'variance': variance,
                'std': std,
                'samples': samples,
                'sample_mean': samples.mean(dim=0),
                'sample_std': samples.std(dim=0)
            }


class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood Loss for Uncertainty Quantification.

    The loss is computed as:
        L = 0.5 * ((y - μ)² / σ² + α * log(σ²))

    Where:
    - μ: predicted mean
    - σ²: predicted variance
    - y: true target value
    - α: variance penalty weight (default: 1.0)

    Args:
        alpha (float): Weight for variance penalty term (default: 1.0)
        reduction (str): Reduction method ('mean', 'sum', 'none')
        epsilon (float): Small value for numerical stability
    """

    def __init__(self, alpha=1.0, reduction='mean', epsilon=1e-6):
        super(GaussianNLLLoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.epsilon = epsilon

        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")

        print(f"GaussianNLLLoss initialized with alpha={alpha:.3f}")
        if alpha < 1.0:
            print("  → Reduced penalty for large variances (encourages honest uncertainty)")
        elif alpha > 1.0:
            print("  → Increased penalty for large variances (encourages confidence)")
        else:
            print("  → Standard Gaussian NLL (balanced)")

    def forward(self, mean, variance, target):
        """
        Compute Gaussian NLL loss with weighted variance penalty.

        Args:
            mean: Predicted mean, shape (batch_size, output_size)
            variance: Predicted variance, shape (batch_size, output_size)
            target: True values, shape (batch_size, output_size)

        Returns:
            Loss value
        """
        variance = variance + self.epsilon

        squared_error_term = (target - mean) ** 2 / variance
        log_variance_term = self.alpha * torch.log(variance)
        loss = 0.5 * (squared_error_term + log_variance_term)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class EnergyScoreLoss(nn.Module):
    """
    Energy Score Loss for Uncertainty Quantification.

    The Energy Score is computed as:
        ES(F, y) = E[||X - y||] - β/2 * E[||X - X'||]

    Args:
        n_samples (int): Number of samples for Monte Carlo estimation (default: 50)
        beta (float): Weight for diversity term (default: 1.0)
        reduction (str): Reduction method ('mean', 'sum', 'none')
        epsilon (float): Small value for numerical stability
    """

    def __init__(self, n_samples=50, beta=1.0, reduction='mean', epsilon=1e-6):
        super(EnergyScoreLoss, self).__init__()
        self.n_samples = n_samples
        self.beta = beta
        self.reduction = reduction
        self.epsilon = epsilon

        if n_samples <= 1:
            raise ValueError(f"n_samples must be > 1, got {n_samples}")
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")

        print(f"EnergyScoreLoss initialized with n_samples={n_samples}, beta={beta:.3f}")
        if beta < 1.0:
            print("  → Reduced diversity penalty (allows wider uncertainty)")
        elif beta > 1.0:
            print("  → Increased diversity penalty (encourages tighter uncertainty)")
        else:
            print("  → Standard Energy Score (balanced)")
        print(f"  → Using {n_samples} Monte Carlo samples per prediction")

    def forward(self, mean, variance, target):
        """Compute Energy Score loss via Monte Carlo sampling"""
        variance = variance + self.epsilon
        std = torch.sqrt(variance)

        batch_size = mean.shape[0]
        output_size = mean.shape[1]

        # Generate samples from predicted Gaussian distribution
        noise = torch.randn(self.n_samples, batch_size, output_size,
                          device=mean.device, dtype=mean.dtype)
        samples = mean.unsqueeze(0) + noise * std.unsqueeze(0)

        # First term: E[|X - y|]
        abs_errors = torch.abs(samples - target.unsqueeze(0))
        first_term = abs_errors.mean(dim=0)

        # Second term: E[|X - X'|]
        pairwise_distances = torch.zeros_like(first_term)

        for i in range(self.n_samples):
            for j in range(i + 1, self.n_samples):
                pairwise_distances += torch.abs(samples[i] - samples[j])

        n_pairs = self.n_samples * (self.n_samples - 1) / 2
        second_term = pairwise_distances / n_pairs

        # Energy Score
        energy_score = first_term - (self.beta / 2.0) * second_term

        if self.reduction == 'mean':
            return energy_score.mean()
        elif self.reduction == 'sum':
            return energy_score.sum()
        else:
            return energy_score


# Convenience functions for creating models

def create_small_uncertainty_model(input_size, output_size, conditioning_config=None):
    """Small uncertainty model for limited datasets"""
    return UncertaintyPredictor(
        input_size=input_size,
        hidden_sizes=[32, 16],
        output_size=output_size,
        dropout_rate=0.1,
        conditioning_config=conditioning_config
    )


def create_medium_uncertainty_model(input_size, output_size, conditioning_config=None):
    """Medium uncertainty model for medium-sized datasets"""
    return UncertaintyPredictor(
        input_size=input_size,
        hidden_sizes=[128, 64, 32],
        output_size=output_size,
        dropout_rate=0.2,
        conditioning_config=conditioning_config
    )


def create_large_uncertainty_model(input_size, output_size, conditioning_config=None):
    """Large uncertainty model for large datasets"""
    return UncertaintyPredictor(
        input_size=input_size,
        hidden_sizes=[256, 128, 64, 32],
        output_size=output_size,
        dropout_rate=0.3,
        conditioning_config=conditioning_config
    )
