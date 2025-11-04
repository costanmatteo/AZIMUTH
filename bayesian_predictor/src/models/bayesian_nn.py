"""
Bayesian Neural Network for Machinery Output Prediction

This module implements a Bayesian Neural Network (BNN) where weights are
treated as probability distributions rather than point estimates. This allows
for principled uncertainty quantification through Bayesian inference.

Key Features:
- Variational Inference: learns distributions over weights
- Uncertainty Quantification: provides epistemic uncertainty estimates
- Monte Carlo Sampling: uses multiple forward passes for prediction
- ELBO Loss: Evidence Lower Bound (NLL + KL divergence)

The network learns a posterior distribution over weights given the data,
allowing it to capture both:
- Aleatoric uncertainty: inherent noise in the data
- Epistemic uncertainty: uncertainty due to limited data/knowledge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class BayesianLinear(nn.Module):
    """
    Bayesian Linear Layer with weight uncertainty.

    Instead of learning fixed weights W, this layer learns a distribution
    over weights: W ~ N(μ_w, σ_w²)

    During training:
    - Learns mean (μ) and log-variance (log σ²) for each weight
    - Samples weights from the learned distribution
    - Optimizes ELBO (Evidence Lower Bound)

    During inference:
    - Can sample different weights for uncertainty estimation
    - Multiple forward passes give different predictions

    Args:
        in_features (int): Size of input
        out_features (int): Size of output
        prior_std (float): Standard deviation of prior distribution (default: 1.0)
    """

    def __init__(self, in_features, out_features, prior_std=1.0):
        super(BayesianLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std

        # Weight parameters: mean and log-variance
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features))

        # Bias parameters: mean and log-variance
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_features))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Xavier initialization"""
        # Initialize means
        stdv = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.bias_mu.data.uniform_(-stdv, stdv)

        # Initialize log-variances (small initial uncertainty)
        self.weight_logvar.data.fill_(-5.0)  # exp(-5) ≈ 0.0067
        self.bias_logvar.data.fill_(-5.0)

    def forward(self, x, sample=True):
        """
        Forward pass with weight sampling.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
            sample (bool): Whether to sample weights (True) or use means (False)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        if sample:
            # Sample weights using reparameterization trick: w = μ + σ * ε, ε ~ N(0,1)
            weight_std = torch.exp(0.5 * self.weight_logvar)
            weight = self.weight_mu + weight_std * torch.randn_like(weight_std)

            bias_std = torch.exp(0.5 * self.bias_logvar)
            bias = self.bias_mu + bias_std * torch.randn_like(bias_std)
        else:
            # Use mean weights (MAP estimate)
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def kl_divergence(self):
        """
        Compute KL divergence between posterior q(w|θ) and prior p(w).

        Assumes:
        - Posterior: q(w|θ) = N(μ, σ²)
        - Prior: p(w) = N(0, prior_std²)

        KL[q||p] = 0.5 * (σ²/σ_prior² + μ²/σ_prior² - 1 - log(σ²/σ_prior²))

        Returns:
            torch.Tensor: KL divergence (scalar)
        """
        prior_var = self.prior_std ** 2

        # KL for weights
        weight_var = torch.exp(self.weight_logvar)
        weight_kl = 0.5 * (
            weight_var / prior_var +
            self.weight_mu ** 2 / prior_var -
            1 -
            torch.log(weight_var / prior_var)
        ).sum()

        # KL for biases
        bias_var = torch.exp(self.bias_logvar)
        bias_kl = 0.5 * (
            bias_var / prior_var +
            self.bias_mu ** 2 / prior_var -
            1 -
            torch.log(bias_var / prior_var)
        ).sum()

        return weight_kl + bias_kl


class BayesianPredictor(nn.Module):
    """
    Bayesian Neural Network for Machinery Prediction with Uncertainty.

    This network uses Bayesian inference to learn distributions over weights,
    providing principled uncertainty estimates. Unlike standard neural networks,
    BNNs can express how confident they are in their predictions.

    Architecture:
    - Input Layer: operational parameters
    - Hidden Layers: Bayesian linear layers with ReLU activations
    - Output Layer: Bayesian linear layer (regression output)

    The network is trained using Variational Inference:
    - Learns q(w|θ), an approximate posterior over weights
    - Minimizes ELBO: -log p(y|x,w) + β * KL[q(w|θ)||p(w)]

    Args:
        input_size (int): Number of input features
        hidden_sizes (list): List of hidden layer dimensions
        output_size (int): Number of output values
        prior_std (float): Standard deviation of weight prior (default: 1.0)
        dropout_rate (float): Dropout rate (default: 0.2)

    Example:
        >>> model = BayesianPredictor(
        ...     input_size=10,
        ...     hidden_sizes=[64, 32, 16],
        ...     output_size=5
        ... )
        >>> x = torch.randn(32, 10)
        >>>
        >>> # Training mode: sample weights
        >>> output = model(x, sample=True)
        >>>
        >>> # Inference: multiple samples for uncertainty
        >>> predictions = model.predict_with_uncertainty(x, n_samples=100)
        >>> mean = predictions['mean']
        >>> std = predictions['std']
    """

    def __init__(self, input_size, hidden_sizes, output_size,
                 prior_std=1.0, dropout_rate=0.2):
        super(BayesianPredictor, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.prior_std = prior_std

        # Build Bayesian layers
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(BayesianLinear(prev_size, hidden_size, prior_std))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Output layer (also Bayesian)
        self.output_layer = BayesianLinear(prev_size, output_size, prior_std)

    def forward(self, x, sample=True):
        """
        Forward pass through the Bayesian network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            sample (bool): Whether to sample weights (True) or use means (False)

        Returns:
            torch.Tensor: Output predictions of shape (batch_size, output_size)
        """
        for layer, dropout in zip(self.layers, self.dropouts):
            x = layer(x, sample=sample)
            x = F.relu(x)
            x = dropout(x)

        x = self.output_layer(x, sample=sample)
        return x

    def kl_divergence(self):
        """
        Compute total KL divergence for all layers.

        This is used as a regularization term in the ELBO loss.

        Returns:
            torch.Tensor: Total KL divergence (scalar)
        """
        kl = 0
        for layer in self.layers:
            kl += layer.kl_divergence()
        kl += self.output_layer.kl_divergence()
        return kl

    def predict_with_uncertainty(self, x, n_samples=100):
        """
        Make predictions with uncertainty estimation using Monte Carlo sampling.

        This method:
        1. Samples multiple weight configurations from the learned posterior
        2. Makes predictions with each configuration
        3. Aggregates results to estimate mean and uncertainty

        Args:
            x (torch.Tensor): Input tensor
            n_samples (int): Number of Monte Carlo samples (default: 100)

        Returns:
            dict: Dictionary containing:
                - 'mean': Mean prediction across samples
                - 'std': Standard deviation (total uncertainty)
                - 'samples': All sampled predictions
                - 'epistemic_uncertainty': Model uncertainty
                - 'aleatoric_uncertainty': Data uncertainty (if available)
        """
        self.eval()
        was_training = self.training

        with torch.no_grad():
            predictions = []

            for _ in range(n_samples):
                # Sample weights and make prediction
                pred = self.forward(x, sample=True)
                predictions.append(pred)

            # Stack predictions: shape (n_samples, batch_size, output_size)
            predictions = torch.stack(predictions)

            # Compute statistics
            mean = predictions.mean(dim=0)
            std = predictions.std(dim=0)

            # Epistemic uncertainty (uncertainty in the model)
            epistemic_uncertainty = std

            result = {
                'mean': mean,
                'std': std,
                'samples': predictions,
                'epistemic_uncertainty': epistemic_uncertainty,
                'confidence_intervals': {
                    '68%': (mean - std, mean + std),
                    '95%': (mean - 2*std, mean + 2*std),
                    '99%': (mean - 3*std, mean + 3*std)
                }
            }

        if was_training:
            self.train()

        return result

    def get_weight_statistics(self):
        """
        Get statistics about learned weight distributions.

        Useful for debugging and understanding what the network learned.

        Returns:
            dict: Statistics for each layer
        """
        stats = {}

        for i, layer in enumerate(self.layers):
            weight_std = torch.exp(0.5 * layer.weight_logvar)
            stats[f'hidden_{i+1}'] = {
                'weight_mu_mean': layer.weight_mu.mean().item(),
                'weight_mu_std': layer.weight_mu.std().item(),
                'weight_std_mean': weight_std.mean().item(),
                'weight_std_std': weight_std.std().item(),
            }

        weight_std = torch.exp(0.5 * self.output_layer.weight_logvar)
        stats['output'] = {
            'weight_mu_mean': self.output_layer.weight_mu.mean().item(),
            'weight_mu_std': self.output_layer.weight_mu.std().item(),
            'weight_std_mean': weight_std.mean().item(),
            'weight_std_std': weight_std.std().item(),
        }

        return stats


class BayesianELBOLoss(nn.Module):
    """
    ELBO Loss for Bayesian Neural Networks.

    The Evidence Lower Bound (ELBO) loss for variational inference:

        ELBO = -log p(y|x,w) + β * KL[q(w|θ)||p(w)]

    Where:
    - log p(y|x,w): Negative log-likelihood (reconstruction term)
    - KL[q||p]: KL divergence between posterior and prior (regularization)
    - β: Weight for KL term (default: 1/N for N training samples)

    The β parameter (KL weight) balances:
    - Fitting the data (NLL term)
    - Staying close to the prior (KL term)

    Common choices:
    - β = 1/N: Standard variational inference (recommended)
    - β < 1/N: Allow more deviation from prior (may overfit)
    - β > 1/N: Stronger regularization (may underfit)

    Args:
        kl_weight (float): Weight for KL divergence term (β)
        reduction (str): How to reduce batch losses ('mean' or 'sum')
    """

    def __init__(self, kl_weight=1.0, reduction='mean'):
        super(BayesianELBOLoss, self).__init__()
        self.kl_weight = kl_weight
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, predictions, targets, kl_divergence):
        """
        Compute ELBO loss.

        Args:
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth values
            kl_divergence (torch.Tensor): Total KL divergence from model

        Returns:
            tuple: (total_loss, nll, kl)
                - total_loss: ELBO loss
                - nll: Negative log-likelihood (data fit term)
                - kl: KL divergence (regularization term)
        """
        # Negative log-likelihood (using MSE for regression)
        nll = self.mse_loss(predictions, targets)

        # KL divergence scaled by kl_weight
        kl = self.kl_weight * kl_divergence

        # ELBO = NLL + KL
        loss = nll + kl

        return loss, nll, kl


# Convenience functions for creating common architectures

def create_small_bayesian_model(input_size, output_size, prior_std=1.0):
    """Small Bayesian model for limited datasets"""
    return BayesianPredictor(
        input_size=input_size,
        hidden_sizes=[32, 16],
        output_size=output_size,
        prior_std=prior_std,
        dropout_rate=0.1
    )


def create_medium_bayesian_model(input_size, output_size, prior_std=1.0):
    """Medium Bayesian model for medium-sized datasets"""
    return BayesianPredictor(
        input_size=input_size,
        hidden_sizes=[128, 64, 32],
        output_size=output_size,
        prior_std=prior_std,
        dropout_rate=0.2
    )


def create_large_bayesian_model(input_size, output_size, prior_std=1.0):
    """Large Bayesian model for large datasets"""
    return BayesianPredictor(
        input_size=input_size,
        hidden_sizes=[256, 128, 64, 32],
        output_size=output_size,
        prior_std=prior_std,
        dropout_rate=0.3
    )
