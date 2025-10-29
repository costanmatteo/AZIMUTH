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

    Args:
        input_size (int): Number of input features
        hidden_sizes (list): List with dimensions of shared hidden layers
        output_size (int): Number of output values to predict
        dropout_rate (float): Dropout rate for regularization (default: 0.2)
        use_batchnorm (bool): Whether to use batch normalization (default: False)
        min_variance (float): Minimum allowed variance for numerical stability (default: 1e-6)

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
                 dropout_rate=0.2, use_batchnorm=False, min_variance=1e-6):
        super(UncertaintyPredictor, self).__init__()

        self.output_size = output_size
        self.min_variance = min_variance

        # Build shared hidden layers
        shared_layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            shared_layers.append(nn.Linear(prev_size, hidden_size))
            if use_batchnorm:
                shared_layers.append(nn.BatchNorm1d(hidden_size))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        self.shared_network = nn.Sequential(*shared_layers)

        # Output heads
        # Mean head: linear output for the predicted mean μ
        self.mean_head = nn.Linear(prev_size, output_size)

        # Variance head: outputs log(σ²) which is then exponentiated
        # Using log-space helps with numerical stability
        self.log_variance_head = nn.Linear(prev_size, output_size)

    def forward(self, x):
        """
        Forward pass of the uncertainty network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)

        Returns:
            tuple: (mean, variance)
                - mean: Predicted mean values of shape (batch_size, output_size)
                - variance: Predicted variance of shape (batch_size, output_size)
        """
        # Shared feature extraction
        features = self.shared_network(x)

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
    Gaussian Negative Log-Likelihood Loss for Uncertainty Quantification with Local Calibration.

    This loss function trains the network to predict both mean and variance.
    It penalizes:
    1. Large errors in the mean prediction
    2. Underestimation of uncertainty (too small variance)
    3. Overestimation of uncertainty (too large variance)
    4. Miscalibration between predicted variance and actual squared error

    The loss is computed as:
        L = 0.5 * ((y - μ)² / σ² + α * log(σ²)) + λ * |σ² - (y - μ)²|

    Where:
    - μ: predicted mean
    - σ²: predicted variance
    - y: true target value
    - α: variance penalty weight (default: 1.0)
    - λ: local calibration weight (default: 0.0)

    The α parameter controls the trade-off between prediction accuracy and
    uncertainty calibration:
    - α = 1.0: Standard Gaussian NLL (default)
    - α < 1.0: Reduces penalty for large variances, allowing the model to be
               more honest about its uncertainty (recommended for over-confident models)
    - α > 1.0: Increases penalty for large variances, pushing for more confident
               predictions

    The λ parameter controls local calibration:
    - λ = 0.0: No local calibration (standard loss)
    - λ > 0.0: Forces predicted variance to align with actual squared errors
               The term |σ² - (y - μ)²| penalizes when the predicted variance
               differs from the observed squared error

    This encourages the model to:
    - Predict accurate means where possible
    - Increase variance in uncertain regions
    - Balance confidence with calibration based on α
    - Align predicted uncertainty with actual errors based on λ

    Args:
        alpha (float): Weight for variance penalty term (default: 1.0)
        calibration_lambda (float): Weight for local calibration term (default: 0.0)
        reduction (str): Reduction method ('mean', 'sum', 'none')
        epsilon (float): Small value for numerical stability
    """

    def __init__(self, alpha=1.0, calibration_lambda=0.0, reduction='mean', epsilon=1e-6):
        super(GaussianNLLLoss, self).__init__()
        self.alpha = alpha
        self.calibration_lambda = calibration_lambda
        self.reduction = reduction
        self.epsilon = epsilon

        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if calibration_lambda < 0:
            raise ValueError(f"calibration_lambda must be non-negative, got {calibration_lambda}")

        print(f"GaussianNLLLoss initialized with alpha={alpha:.3f}, calibration_lambda={calibration_lambda:.4f}")
        if alpha < 1.0:
            print("  → Reduced penalty for large variances (encourages honest uncertainty)")
        elif alpha > 1.0:
            print("  → Increased penalty for large variances (encourages confidence)")
        else:
            print("  → Standard Gaussian NLL (balanced)")

        if calibration_lambda > 0:
            print(f"  → Local calibration enabled with λ={calibration_lambda:.4f}")
            print("    (forces predicted variance to align with actual squared errors)")
        else:
            print("  → No local calibration (λ=0)")

    def forward(self, mean, variance, target):
        """
        Compute Gaussian NLL loss with weighted variance penalty and local calibration.

        Args:
            mean (torch.Tensor): Predicted mean, shape (batch_size, output_size)
            variance (torch.Tensor): Predicted variance, shape (batch_size, output_size)
            target (torch.Tensor): True values, shape (batch_size, output_size)

        Returns:
            torch.Tensor: Loss value

        Formula:
            L = 0.5 * ((y - μ)² / σ² + α * log(σ²)) + λ * |σ² - (y - μ)²|

        When α < 1, the model can more easily increase variance without
        being heavily penalized, leading to better calibrated uncertainty.

        When λ > 0, the local calibration term forces the predicted variance
        to align with the actual squared errors, improving calibration.
        """
        # Ensure variance is positive
        variance = variance + self.epsilon

        # Compute squared error
        squared_error = (target - mean) ** 2

        # Weighted Gaussian NLL: 0.5 * ((target - mean)^2 / var + alpha * log(var))
        squared_error_term = squared_error / variance
        log_variance_term = self.alpha * torch.log(variance)
        nll_loss = 0.5 * (squared_error_term + log_variance_term)

        # Add local calibration term: λ * |σ² - (y - μ)²|
        if self.calibration_lambda > 0:
            calibration_term = self.calibration_lambda * torch.abs(variance - squared_error)
            loss = nll_loss + calibration_term
        else:
            loss = nll_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


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
