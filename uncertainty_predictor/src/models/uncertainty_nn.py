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
            shared_layers.append(nn.SiLU())
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


class EnsembleUncertaintyPredictor(nn.Module):
    """
    Deep Ensemble for Uncertainty Quantification.

    This class implements a Deep Ensemble approach where N independent neural networks
    are trained with different random initializations. The ensemble provides:
    - Better calibrated uncertainty estimates
    - Separation of aleatoric and epistemic uncertainty
    - Improved out-of-distribution detection

    The ensemble combines predictions using:
    - Mean: average of individual model means
    - Aleatoric uncertainty: average of individual model variances (data noise)
    - Epistemic uncertainty: variance of individual model means (model uncertainty)
    - Total uncertainty: aleatoric + epistemic

    Args:
        input_size (int): Number of input features
        hidden_sizes (list): List with dimensions of shared hidden layers
        output_size (int): Number of output values to predict
        n_models (int): Number of models in the ensemble (default: 5)
        dropout_rate (float): Dropout rate for regularization (default: 0.2)
        use_batchnorm (bool): Whether to use batch normalization (default: False)
        min_variance (float): Minimum allowed variance for numerical stability (default: 1e-6)

    Example:
        >>> ensemble = EnsembleUncertaintyPredictor(
        ...     input_size=10,
        ...     hidden_sizes=[64, 32],
        ...     output_size=5,
        ...     n_models=5
        ... )
        >>> x = torch.randn(32, 10)
        >>> mean, total_var, aleatoric, epistemic = ensemble.predict_with_decomposition(x)
    """

    def __init__(self, input_size, hidden_sizes, output_size, n_models=5,
                 dropout_rate=0.2, use_batchnorm=False, min_variance=1e-6):
        super(EnsembleUncertaintyPredictor, self).__init__()

        self.n_models = n_models
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm
        self.min_variance = min_variance

        # Create N independent models
        self.models = nn.ModuleList([
            UncertaintyPredictor(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                output_size=output_size,
                dropout_rate=dropout_rate,
                use_batchnorm=use_batchnorm,
                min_variance=min_variance
            )
            for _ in range(n_models)
        ])

    def forward(self, x, model_idx=None):
        """
        Forward pass through the ensemble.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            model_idx (int, optional): If specified, only use this model index.
                                       Used during training to train individual models.

        Returns:
            If model_idx is specified:
                tuple: (mean, variance) from the specified model
            Otherwise:
                tuple: (ensemble_mean, total_variance)
                    - ensemble_mean: Average of all model means
                    - total_variance: Aleatoric + Epistemic uncertainty
        """
        if model_idx is not None:
            # Training mode: forward through single model
            return self.models[model_idx](x)

        # Inference mode: aggregate all models
        return self.predict_ensemble(x)

    def predict_ensemble(self, x):
        """
        Make ensemble prediction by aggregating all models.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            tuple: (ensemble_mean, total_variance)
        """
        means = []
        variances = []

        for model in self.models:
            mean, variance = model(x)
            means.append(mean)
            variances.append(variance)

        # Stack predictions: shape (n_models, batch_size, output_size)
        means = torch.stack(means)
        variances = torch.stack(variances)

        # Ensemble mean: average of individual means
        ensemble_mean = means.mean(dim=0)

        # Aleatoric uncertainty: average of individual variances
        aleatoric = variances.mean(dim=0)

        # Epistemic uncertainty: variance of individual means
        epistemic = means.var(dim=0)

        # Total uncertainty
        total_variance = aleatoric + epistemic

        return ensemble_mean, total_variance

    def predict_with_decomposition(self, x):
        """
        Make predictions with full uncertainty decomposition.

        This method provides separate aleatoric and epistemic uncertainties,
        which is useful for understanding the source of uncertainty.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            tuple: (ensemble_mean, total_variance, aleatoric, epistemic)
                - ensemble_mean: Average prediction across ensemble
                - total_variance: Total uncertainty (aleatoric + epistemic)
                - aleatoric: Data/noise uncertainty (average of predicted variances)
                - epistemic: Model uncertainty (variance of predicted means)
        """
        self.eval()
        with torch.no_grad():
            means = []
            variances = []

            for model in self.models:
                mean, variance = model(x)
                means.append(mean)
                variances.append(variance)

            means = torch.stack(means)
            variances = torch.stack(variances)

            ensemble_mean = means.mean(dim=0)
            aleatoric = variances.mean(dim=0)
            epistemic = means.var(dim=0)
            total_variance = aleatoric + epistemic

            return ensemble_mean, total_variance, aleatoric, epistemic

    def predict_with_uncertainty(self, x, n_samples=100):
        """
        Make predictions with uncertainty estimation (compatible with single model API).

        This method maintains API compatibility with UncertaintyPredictor.

        Args:
            x (torch.Tensor): Input tensor
            n_samples (int): Not used, kept for API compatibility

        Returns:
            dict: Dictionary containing:
                - 'mean': Ensemble mean predictions
                - 'variance': Total variance (aleatoric + epistemic)
                - 'std': Standard deviation
                - 'aleatoric': Aleatoric uncertainty
                - 'epistemic': Epistemic uncertainty
                - 'individual_means': Means from each model
                - 'individual_variances': Variances from each model
        """
        self.eval()
        with torch.no_grad():
            means = []
            variances = []

            for model in self.models:
                mean, variance = model(x)
                means.append(mean)
                variances.append(variance)

            means_stacked = torch.stack(means)
            variances_stacked = torch.stack(variances)

            ensemble_mean = means_stacked.mean(dim=0)
            aleatoric = variances_stacked.mean(dim=0)
            epistemic = means_stacked.var(dim=0)
            total_variance = aleatoric + epistemic

            return {
                'mean': ensemble_mean,
                'variance': total_variance,
                'std': torch.sqrt(total_variance),
                'aleatoric': aleatoric,
                'epistemic': epistemic,
                'individual_means': means_stacked,
                'individual_variances': variances_stacked
            }

    def get_individual_predictions(self, x):
        """
        Get predictions from each individual model.

        Useful for visualization and debugging.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            list: List of (mean, variance) tuples from each model
        """
        self.eval()
        predictions = []
        with torch.no_grad():
            for model in self.models:
                mean, variance = model(x)
                predictions.append((mean, variance))
        return predictions


def create_ensemble_model(input_size, output_size, hidden_sizes, n_models=5,
                          dropout_rate=0.2, use_batchnorm=False, min_variance=1e-6):
    """
    Factory function to create an ensemble uncertainty model.

    Args:
        input_size (int): Number of input features
        output_size (int): Number of output features
        hidden_sizes (list): Hidden layer sizes
        n_models (int): Number of models in ensemble (default: 5)
        dropout_rate (float): Dropout rate (default: 0.2)
        use_batchnorm (bool): Use batch normalization (default: False)
        min_variance (float): Minimum variance (default: 1e-6)

    Returns:
        EnsembleUncertaintyPredictor: Configured ensemble model
    """
    return EnsembleUncertaintyPredictor(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        n_models=n_models,
        dropout_rate=dropout_rate,
        use_batchnorm=use_batchnorm,
        min_variance=min_variance
    )


class SWAGUncertaintyPredictor(nn.Module):
    """
    SWAG (Stochastic Weight Averaging - Gaussian) for Uncertainty Quantification.

    This class implements SWAG, which approximates the posterior distribution over
    neural network weights using a Gaussian with a low-rank plus diagonal covariance.

    Key idea: During the last phase of training, SGD iterates approximate samples
    from the posterior. SWAG captures the first two moments of these iterates.

    The covariance is approximated as:
        Σ ≈ (1/2) * (Σ_diag + Σ_low_rank)
    where:
        - Σ_diag: diagonal covariance (variance of each weight)
        - Σ_low_rank: low-rank approximation using K deviation vectors

    During inference:
        1. Sample weights from N(θ_SWA, Σ)
        2. Make prediction with sampled weights
        3. Repeat n_samples times
        4. Aggregate: mean → prediction, variance of means → epistemic,
                      mean of variances → aleatoric

    Reference:
        Maddox et al. (2019) "A Simple Baseline for Bayesian Uncertainty in Deep Learning"
        https://arxiv.org/abs/1902.02476

    Args:
        base_model (UncertaintyPredictor): The base model to wrap
        max_rank (int): Maximum rank for low-rank covariance approximation (default: 20)

    Example:
        >>> base_model = UncertaintyPredictor(input_size=10, hidden_sizes=[64, 32], output_size=5)
        >>> swag_model = SWAGUncertaintyPredictor(base_model, max_rank=20)
        >>> # After SWAG training...
        >>> mean, total_var, aleatoric, epistemic = swag_model.predict_with_decomposition(x)
    """

    def __init__(self, base_model, max_rank=20):
        super(SWAGUncertaintyPredictor, self).__init__()

        self.base_model = base_model
        self.max_rank = max_rank

        # Get total number of parameters
        self.n_params = sum(p.numel() for p in base_model.parameters())

        # SWA mean (running average of weights)
        self.register_buffer('swa_mean', torch.zeros(self.n_params))

        # Diagonal variance (running average of squared weights)
        self.register_buffer('swa_sq_mean', torch.zeros(self.n_params))

        # Low-rank deviation vectors (columns of D matrix)
        # Shape: (n_params, max_rank)
        self.register_buffer('deviation_matrix', torch.zeros(self.n_params, max_rank))

        # Number of models collected for SWA
        self.register_buffer('n_models_collected', torch.tensor(0))

        # Current column index for deviation matrix (circular buffer)
        self.register_buffer('deviation_idx', torch.tensor(0))

        # Flag to indicate if SWAG statistics have been collected
        self.swag_collected = False

    def _flatten_params(self):
        """Flatten all model parameters into a single vector."""
        return torch.cat([p.data.view(-1) for p in self.base_model.parameters()])

    def _unflatten_params(self, flat_params):
        """Restore flattened parameters back to model."""
        idx = 0
        for p in self.base_model.parameters():
            n = p.numel()
            p.data.copy_(flat_params[idx:idx + n].view(p.shape))
            idx += n

    def collect_model(self):
        """
        Collect current model weights for SWAG statistics.

        Call this at the end of each epoch during the SWA phase.
        Updates running mean, squared mean, and deviation matrix.
        """
        # Get current parameters as flat vector
        current_params = self._flatten_params()

        n = self.n_models_collected.item()

        if n == 0:
            # First collection: initialize
            self.swa_mean.copy_(current_params)
            self.swa_sq_mean.copy_(current_params ** 2)
        else:
            # Update running averages (online mean update)
            # new_mean = old_mean + (new_value - old_mean) / (n + 1)
            self.swa_mean.add_((current_params - self.swa_mean) / (n + 1))
            self.swa_sq_mean.add_((current_params ** 2 - self.swa_sq_mean) / (n + 1))

        # Store deviation from mean in low-rank matrix (circular buffer)
        deviation = current_params - self.swa_mean
        col_idx = self.deviation_idx.item() % self.max_rank
        self.deviation_matrix[:, col_idx] = deviation

        self.n_models_collected.add_(1)
        self.deviation_idx.add_(1)
        self.swag_collected = True

    def _compute_variance(self):
        """
        Compute the diagonal variance from collected statistics.

        Returns:
            torch.Tensor: Diagonal variance vector
        """
        # Variance = E[X^2] - E[X]^2
        variance = self.swa_sq_mean - self.swa_mean ** 2
        # Clamp to avoid negative variance due to numerical issues
        return torch.clamp(variance, min=1e-10)

    def sample_weights(self, scale=1.0):
        """
        Sample weights from the SWAG posterior distribution.

        The posterior is: N(θ_SWA, (1/2) * (Σ_diag + Σ_low_rank))

        Sampling procedure:
        1. Sample z1 ~ N(0, I_d) for diagonal part
        2. Sample z2 ~ N(0, I_K) for low-rank part
        3. θ = θ_SWA + (1/√2) * (√Σ_diag * z1 + D * z2 / √(K-1))

        Args:
            scale (float): Scale factor for perturbation (default: 1.0)

        Returns:
            torch.Tensor: Sampled parameter vector
        """
        if not self.swag_collected:
            raise RuntimeError("No SWAG statistics collected. Train with SWAG first.")

        device = self.swa_mean.device

        # Diagonal part: sample from N(0, Σ_diag)
        diag_var = self._compute_variance()
        z1 = torch.randn(self.n_params, device=device)
        diag_sample = torch.sqrt(diag_var) * z1

        # Low-rank part: sample from N(0, D @ D.T / (K-1))
        n_collected = min(self.n_models_collected.item(), self.max_rank)
        if n_collected > 1:
            z2 = torch.randn(n_collected, device=device)
            D = self.deviation_matrix[:, :n_collected]
            low_rank_sample = (D @ z2) / np.sqrt(n_collected - 1)
        else:
            low_rank_sample = torch.zeros(self.n_params, device=device)

        # Combine: θ = θ_SWA + scale * (1/√2) * (diag + low_rank)
        sampled_params = self.swa_mean + scale * (1.0 / np.sqrt(2.0)) * (diag_sample + low_rank_sample)

        return sampled_params

    def forward(self, x):
        """
        Forward pass using SWA mean weights.

        For uncertainty estimation, use predict_with_decomposition instead.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            tuple: (mean, variance) predictions using SWA mean weights
        """
        if self.swag_collected:
            # Use SWA mean weights
            self._unflatten_params(self.swa_mean)
        return self.base_model(x)

    def predict_with_decomposition(self, x, n_samples=30, scale=1.0):
        """
        Make predictions with full uncertainty decomposition using SWAG sampling.

        Samples n_samples weight configurations from the posterior, makes predictions
        with each, and aggregates to compute aleatoric and epistemic uncertainty.

        Args:
            x (torch.Tensor): Input tensor
            n_samples (int): Number of weight samples (default: 30)
            scale (float): Scale factor for weight perturbation (default: 1.0)

        Returns:
            tuple: (ensemble_mean, total_variance, aleatoric, epistemic)
                - ensemble_mean: Average prediction across weight samples
                - total_variance: Total uncertainty (aleatoric + epistemic)
                - aleatoric: Data/noise uncertainty (average of predicted variances)
                - epistemic: Model uncertainty (variance of predicted means)
        """
        if not self.swag_collected:
            raise RuntimeError("No SWAG statistics collected. Train with SWAG first.")

        self.base_model.eval()

        all_means = []
        all_variances = []

        # Save original weights to restore later
        original_params = self._flatten_params().clone()

        with torch.no_grad():
            for _ in range(n_samples):
                # Sample weights from posterior
                sampled_params = self.sample_weights(scale=scale)
                self._unflatten_params(sampled_params)

                # Forward pass with sampled weights
                mean, variance = self.base_model(x)
                all_means.append(mean)
                all_variances.append(variance)

        # Restore original weights
        self._unflatten_params(original_params)

        # Stack predictions: shape (n_samples, batch_size, output_size)
        means = torch.stack(all_means)
        variances = torch.stack(all_variances)

        # Aggregate predictions
        ensemble_mean = means.mean(dim=0)

        # Aleatoric: average of predicted variances (data noise)
        aleatoric = variances.mean(dim=0)

        # Epistemic: variance of predicted means (model uncertainty)
        epistemic = means.var(dim=0)

        # Total uncertainty
        total_variance = aleatoric + epistemic

        return ensemble_mean, total_variance, aleatoric, epistemic

    def predict_with_uncertainty(self, x, n_samples=30):
        """
        Make predictions with uncertainty estimation (compatible with other models API).

        Args:
            x (torch.Tensor): Input tensor
            n_samples (int): Number of weight samples (default: 30)

        Returns:
            dict: Dictionary containing:
                - 'mean': SWAG mean predictions
                - 'variance': Total variance (aleatoric + epistemic)
                - 'std': Standard deviation
                - 'aleatoric': Aleatoric uncertainty
                - 'epistemic': Epistemic uncertainty
        """
        ensemble_mean, total_var, aleatoric, epistemic = self.predict_with_decomposition(x, n_samples)

        return {
            'mean': ensemble_mean,
            'variance': total_var,
            'std': torch.sqrt(total_var),
            'aleatoric': aleatoric,
            'epistemic': epistemic
        }

    def get_swag_stats(self):
        """
        Get SWAG statistics for inspection/debugging.

        Returns:
            dict: Dictionary with SWAG statistics
        """
        return {
            'n_models_collected': self.n_models_collected.item(),
            'mean_weight_norm': torch.norm(self.swa_mean).item(),
            'mean_variance': self._compute_variance().mean().item(),
            'max_rank_used': min(self.n_models_collected.item(), self.max_rank)
        }


def create_swag_model(base_model, max_rank=20):
    """
    Factory function to create a SWAG uncertainty model.

    Args:
        base_model (UncertaintyPredictor): Base model to wrap
        max_rank (int): Maximum rank for covariance approximation (default: 20)

    Returns:
        SWAGUncertaintyPredictor: SWAG-wrapped model
    """
    return SWAGUncertaintyPredictor(base_model, max_rank=max_rank)
