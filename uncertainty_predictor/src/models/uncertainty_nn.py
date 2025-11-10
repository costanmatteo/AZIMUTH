"""
Neural Network with Uncertainty Quantification

This module contains the definition of a neural network that predicts
both the mean (μ) and variance (σ²) of the output, enabling uncertainty
quantification for each prediction.

The network outputs two values:
- μ(x): estimated mean value
- σ²(x): estimated uncertainty (variance)

This allows the model to learn where the data is noisy or uncertain.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict

try:
    from .conditional_layers import ContextEmbeddingModule, ConditionalLayerNorm, ConditionalBatchNorm1d
except ImportError:
    # Fallback for when module is imported differently
    from conditional_layers import ContextEmbeddingModule, ConditionalLayerNorm, ConditionalBatchNorm1d


class UncertaintyPredictor(nn.Module):
    """
    Neural Network with Uncertainty Quantification.

    Instead of outputting only a single prediction ŷ, this network produces:
    - μ(x): mean prediction
    - σ²(x): variance (uncertainty) of the prediction

    The model learns to increase σ² in regions where data is noisy or uncertain,
    and decrease it where predictions are more reliable.

    Architecture:
    - Input Layer: operational parameters
    - Shared Hidden Layers: feature extraction
    - Two output heads:
        * Mean head: predicts μ (linear output)
        * Variance head: predicts log(σ²) (then exponentiated for positivity)

    **Conditional Embedding Support:**
    When `conditioning_config` is provided, the model incorporates conditional
    embeddings from process_id, environment variables, and timestamps. These
    embeddings modulate the normalization layers via ConditionalLayerNorm.

    Args:
        input_size (int): Number of input features
        hidden_sizes (list): List with dimensions of shared hidden layers
        output_size (int): Number of output values to predict
        dropout_rate (float): Dropout rate for regularization (default: 0.2)
        use_batchnorm (bool): Whether to use batch normalization (default: False)
        min_variance (float): Minimum allowed variance for numerical stability (default: 1e-6)
        conditioning_config (dict, optional): Configuration for conditional embeddings

    Example:
        >>> model = UncertaintyPredictor(
        ...     input_size=10,
        ...     hidden_sizes=[64, 32, 16],
        ...     output_size=5
        ... )
        >>> x = torch.randn(32, 10)  # batch of 32 examples
        >>> mean, variance = model(x)
        >>> print(mean.shape, variance.shape)  # both: torch.Size([32, 5])
    """

    def __init__(self, input_size, hidden_sizes, output_size,
                 dropout_rate=0.2, use_batchnorm=False, min_variance=1e-6,
                 conditioning_config=None):
        super(UncertaintyPredictor, self).__init__()

        self.output_size = output_size
        self.min_variance = min_variance
        self.conditioning_enabled = conditioning_config is not None and conditioning_config.get('enable', False)
        self.conditioning_config = conditioning_config or {}

        # Create conditional embedding module if enabled
        if self.conditioning_enabled:
            print("Initializing UncertaintyPredictor with Conditional Embeddings")
            self.context_embedder = ContextEmbeddingModule(
                num_processes=conditioning_config.get('num_processes', 4),
                d_proc=conditioning_config.get('d_proc', 16),
                env_continuous=conditioning_config.get('env_continuous', []),
                d_env_float=conditioning_config.get('d_env_float', 16),
                use_missing_mask=conditioning_config.get('use_missing_mask', True),
                env_categorical=conditioning_config.get('env_categorical', {}),
                d_env_cat_base=conditioning_config.get('d_env_cat_base', 1.6),
                use_time=conditioning_config.get('use_time', True),
                time_periods=conditioning_config.get('time_periods', 4),
                d_time=conditioning_config.get('d_time', 8),
                d_context=conditioning_config.get('d_context', 64),
                context_mlp_hidden=conditioning_config.get('context_mlp_hidden', [128, 64]),
                context_dropout=conditioning_config.get('context_dropout', 0.1)
            )
            d_context = conditioning_config.get('d_context', 64)
            norm_type = conditioning_config.get('norm_type', 'conditional_layer_norm')
            print(f"  • Context embedding dimension: {d_context}")
            print(f"  • Normalization type: {norm_type}")
        else:
            self.context_embedder = None
            d_context = None
            norm_type = None

        # Build shared hidden layers with conditional normalization
        self.shared_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            self.shared_layers.append(nn.Linear(prev_size, hidden_size))

            # Normalization layer (conditional or standard)
            if use_batchnorm:
                if self.conditioning_enabled:
                    if norm_type == 'conditional_layer_norm':
                        self.norm_layers.append(ConditionalLayerNorm(hidden_size, d_context))
                    elif norm_type == 'conditional_batch_norm':
                        self.norm_layers.append(ConditionalBatchNorm1d(hidden_size, d_context))
                    else:
                        # Fallback to standard LayerNorm
                        self.norm_layers.append(nn.LayerNorm(hidden_size))
                else:
                    self.norm_layers.append(nn.BatchNorm1d(hidden_size))
            else:
                self.norm_layers.append(None)

            # Activation
            self.activation_layers.append(nn.SiLU())

            # Dropout
            self.dropout_layers.append(nn.Dropout(dropout_rate))

            prev_size = hidden_size

        # Output heads
        # Mean head: linear output for the predicted mean μ
        self.mean_head = nn.Linear(prev_size, output_size)

        # Variance head: outputs log(σ²) which is then exponentiated
        # Using log-space helps with numerical stability
        self.log_variance_head = nn.Linear(prev_size, output_size)

    def forward(
        self,
        x,
        process_id=None,
        env_continuous=None,
        env_categorical=None,
        timestamp=None,
        env_masks=None
    ):
        """
        Forward pass of the uncertainty network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            process_id (torch.Tensor, optional): Process IDs (batch_size,)
            env_continuous (dict, optional): Continuous env vars {name: tensor (batch_size,)}
            env_categorical (dict, optional): Categorical env vars {name: tensor (batch_size,)}
            timestamp (torch.Tensor, optional): Timestamps (batch_size,)
            env_masks (dict, optional): Missing value masks {name: tensor (batch_size,)}

        Returns:
            tuple: (mean, variance)
                - mean: Predicted mean values of shape (batch_size, output_size)
                - variance: Predicted variance of shape (batch_size, output_size)
        """
        # Generate context vector if conditioning is enabled
        context = None
        if self.conditioning_enabled and self.context_embedder is not None:
            context = self.context_embedder(
                process_id=process_id,
                env_continuous=env_continuous,
                env_categorical=env_categorical,
                timestamp=timestamp,
                env_masks=env_masks
            )

        # Shared feature extraction with conditional normalization
        features = x
        for i, (linear, norm, activation, dropout) in enumerate(
            zip(self.shared_layers, self.norm_layers, self.activation_layers, self.dropout_layers)
        ):
            features = linear(features)

            # Apply normalization (conditional or standard)
            if norm is not None:
                if self.conditioning_enabled and isinstance(norm, (ConditionalLayerNorm, ConditionalBatchNorm1d)):
                    features = norm(features, context)
                else:
                    features = norm(features)

            features = activation(features)
            features = dropout(features)

        # Predict mean
        mean = self.mean_head(features)

        # Predict variance (log-space for stability, then exp)
        log_variance = self.log_variance_head(features)
        variance = torch.exp(log_variance) + self.min_variance  # ensure positivity

        return mean, variance

    def predict_with_uncertainty(self, x, n_samples=100):
        """
        Make predictions with uncertainty estimation using sampling.

        This method samples from the predicted Gaussian distribution to
        provide uncertainty estimates.

        Args:
            x (torch.Tensor): Input tensor
            n_samples (int): Number of samples to draw (default: 100)

        Returns:
            dict: Dictionary containing:
                - 'mean': Mean predictions
                - 'variance': Predicted variance
                - 'std': Standard deviation
                - 'samples': Sampled predictions (for further analysis)
        """
        self.eval()
        with torch.no_grad():
            mean, variance = self.forward(x)
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

    This loss function trains the network to predict both mean and variance.
    It penalizes:
    1. Large errors in the mean prediction
    2. Underestimation of uncertainty (too small variance)
    3. Overestimation of uncertainty (too large variance)

    The loss is computed as:
        L = 0.5 * ((y - μ)² / σ² + α * log(σ²))

    Where:
    - μ: predicted mean
    - σ²: predicted variance
    - y: true target value
    - α: variance penalty weight (default: 1.0)

    The α parameter controls the trade-off between prediction accuracy and
    uncertainty calibration:
    - α = 1.0: Standard Gaussian NLL (default)
    - α < 1.0: Reduces penalty for large variances, allowing the model to be
               more honest about its uncertainty (recommended for over-confident models)
    - α > 1.0: Increases penalty for large variances, pushing for more confident
               predictions

    This encourages the model to:
    - Predict accurate means where possible
    - Increase variance in uncertain regions
    - Balance confidence with calibration based on α

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
            mean (torch.Tensor): Predicted mean, shape (batch_size, output_size)
            variance (torch.Tensor): Predicted variance, shape (batch_size, output_size)
            target (torch.Tensor): True values, shape (batch_size, output_size)

        Returns:
            torch.Tensor: Loss value

        Formula:
            L = 0.5 * ((y - μ)² / σ² + α * log(σ²))

        When α < 1, the model can more easily increase variance without
        being heavily penalized, leading to better calibrated uncertainty.
        """
        # Ensure variance is positive
        variance = variance + self.epsilon

        # Weighted Gaussian NLL: 0.5 * ((target - mean)^2 / var + alpha * log(var))
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

    The Energy Score is a proper scoring rule for evaluating probabilistic predictions.
    Unlike Gaussian NLL which assumes and enforces a specific distribution shape,
    the Energy Score evaluates the quality of predictions without assuming a
    particular distribution form.

    The Energy Score is computed as:
        ES(F, y) = E[||X - y||] - β/2 * E[||X - X'||]

    Where:
    - F is the predicted distribution (Gaussian with mean μ and variance σ²)
    - y is the true target value
    - X and X' are independent samples from F
    - β is a weight for the diversity term (default: 1.0)
    - || || is a norm (we use L1: absolute value)

    Implementation via Monte Carlo sampling:
    1. Sample n_samples from N(μ, σ²)
    2. Compute mean absolute error between samples and target
    3. Compute mean pairwise distances between samples
    4. ES = MAE(samples, y) - β/2 * mean_pairwise_distance

    The β parameter controls the trade-off:
    - β = 1.0: Standard Energy Score (default, recommended)
    - β < 1.0: Less penalty for diverse predictions (allows wider distributions)
    - β > 1.0: More penalty for diverse predictions (encourages tighter distributions)

    Advantages over Gaussian NLL:
    - More robust to distribution misspecification
    - Direct evaluation of predictive distribution quality
    - Better calibrated uncertainty estimates in practice
    - Encourages proper uncertainty quantification naturally

    Args:
        n_samples (int): Number of samples for Monte Carlo estimation (default: 50)
        beta (float): Weight for diversity term (default: 1.0)
        reduction (str): Reduction method ('mean', 'sum', 'none')
        epsilon (float): Small value for numerical stability

    Reference:
        Gneiting, T., & Raftery, A. E. (2007). "Strictly proper scoring rules,
        prediction, and estimation." Journal of the American Statistical Association.
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
        """
        Compute Energy Score loss via Monte Carlo sampling.

        Args:
            mean (torch.Tensor): Predicted mean, shape (batch_size, output_size)
            variance (torch.Tensor): Predicted variance, shape (batch_size, output_size)
            target (torch.Tensor): True values, shape (batch_size, output_size)

        Returns:
            torch.Tensor: Energy Score loss value

        Formula:
            ES = E[|X - y|] - β/2 * E[|X - X'|]
        """
        # Ensure variance is positive
        variance = variance + self.epsilon
        std = torch.sqrt(variance)

        batch_size = mean.shape[0]
        output_size = mean.shape[1]

        # Generate samples from predicted Gaussian distribution
        # Shape: (n_samples, batch_size, output_size)
        noise = torch.randn(self.n_samples, batch_size, output_size,
                          device=mean.device, dtype=mean.dtype)
        samples = mean.unsqueeze(0) + noise * std.unsqueeze(0)

        # First term: E[|X - y|]
        # Mean absolute error between samples and target
        # Shape: (n_samples, batch_size, output_size)
        abs_errors = torch.abs(samples - target.unsqueeze(0))
        first_term = abs_errors.mean(dim=0)  # Average over samples

        # Second term: E[|X - X'|]
        # Mean pairwise distance between samples
        # Efficient computation: for each pair (i,j), compute |Xi - Xj|
        pairwise_distances = torch.zeros_like(first_term)

        # Compute mean pairwise distance
        # For efficiency, we compute this as:
        # E[|X - X'|] ≈ (1/n²) * Σᵢ Σⱼ |Xᵢ - Xⱼ|
        for i in range(self.n_samples):
            for j in range(i + 1, self.n_samples):
                pairwise_distances += torch.abs(samples[i] - samples[j])

        # Normalize: we computed sum over upper triangle, so divide by n_pairs
        n_pairs = self.n_samples * (self.n_samples - 1) / 2
        second_term = pairwise_distances / n_pairs

        # Energy Score: ES = first_term - (beta/2) * second_term
        energy_score = first_term - (self.beta / 2.0) * second_term

        # Apply reduction
        if self.reduction == 'mean':
            return energy_score.mean()
        elif self.reduction == 'sum':
            return energy_score.sum()
        else:
            return energy_score


# Convenience functions for creating common model architectures

def create_small_uncertainty_model(input_size, output_size):
    """Small uncertainty model for limited datasets"""
    return UncertaintyPredictor(
        input_size=input_size,
        hidden_sizes=[32, 16],
        output_size=output_size,
        dropout_rate=0.1
    )


def create_medium_uncertainty_model(input_size, output_size):
    """Medium uncertainty model for medium-sized datasets"""
    return UncertaintyPredictor(
        input_size=input_size,
        hidden_sizes=[128, 64, 32],
        output_size=output_size,
        dropout_rate=0.2
    )


def create_large_uncertainty_model(input_size, output_size):
    """Large uncertainty model for large datasets"""
    return UncertaintyPredictor(
        input_size=input_size,
        hidden_sizes=[256, 128, 64, 32],
        output_size=output_size,
        dropout_rate=0.3
    )
