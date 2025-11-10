"""
Conditional layers and embedding modules for process-aware neural networks.

This module provides:
- Time2Vec: Learnable temporal encoding
- ContextEmbeddingModule: Unified embedding for process_id, environment variables, and time
- ConditionalLayerNorm: Layer normalization modulated by context
- ConditionalBatchNorm1d: Batch normalization modulated by context
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple


class Time2Vec(nn.Module):
    """
    Time2Vec encoding with learnable frequencies and phases.

    Formula: [t, sin(ω₁·t+φ₁), ..., sin(ωₖ·t+φₖ)]

    Args:
        num_periods: Number of sinusoidal components (k)
    """

    def __init__(self, num_periods: int = 4):
        super().__init__()
        self.num_periods = num_periods

        # Learnable frequencies (ω) and phases (φ)
        self.frequencies = nn.Parameter(torch.randn(num_periods) * 0.01)
        self.phases = nn.Parameter(torch.randn(num_periods) * 0.01)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timestamps of shape (batch, 1) or (batch,)

        Returns:
            Encoded time of shape (batch, 1 + num_periods)
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)  # (batch,) -> (batch, 1)

        # Linear component
        t_linear = t  # (batch, 1)

        # Sinusoidal components
        # t: (batch, 1), frequencies: (num_periods,) -> (batch, num_periods)
        t_periodic = torch.sin(t * self.frequencies.unsqueeze(0) + self.phases.unsqueeze(0))

        # Concatenate: (batch, 1 + num_periods)
        return torch.cat([t_linear, t_periodic], dim=1)

    @property
    def output_dim(self) -> int:
        """Output dimension: 1 (linear) + num_periods (sinusoids)"""
        return 1 + self.num_periods


class ContextEmbeddingModule(nn.Module):
    """
    Unified context embedding module that handles:
    - Process ID (categorical)
    - Continuous environment variables (with missing value handling)
    - Categorical environment variables (batch_id, operator_id, shift, etc.)
    - Temporal information (via Time2Vec)

    The module is flexible: any subset of inputs can be provided.
    If all inputs are None, returns a zero context vector.

    Args:
        num_processes: Number of distinct processes (e.g., 4 for PCB manufacturing)
        d_proc: Embedding dimension for process_id
        env_continuous_names: List of continuous environment variable names
        d_env_float: Projection dimension for continuous env variables
        env_categorical_specs: Dict {var_name: cardinality} for categorical variables
        d_env_cat_base: Base multiplier for categorical embedding size
        use_time: Whether to use temporal encoding
        time_periods: Number of periods for Time2Vec
        d_time: Projection dimension for time encoding
        d_context: Final context vector dimension
        context_mlp_hidden: Hidden layer sizes for fusion MLP
        context_dropout: Dropout rate in fusion MLP
        use_missing_mask: Concatenate 0/1 mask for missing continuous variables
    """

    def __init__(
        self,
        num_processes: int = 4,
        d_proc: int = 16,
        env_continuous_names: Optional[List[str]] = None,
        d_env_float: int = 16,
        env_categorical_specs: Optional[Dict[str, int]] = None,
        d_env_cat_base: float = 1.6,
        use_time: bool = True,
        time_periods: int = 4,
        d_time: int = 8,
        d_context: int = 64,
        context_mlp_hidden: Optional[List[int]] = None,
        context_dropout: float = 0.1,
        use_missing_mask: bool = True
    ):
        super().__init__()

        # Store configuration
        self.num_processes = num_processes
        self.d_proc = d_proc
        self.env_continuous_names = env_continuous_names or []
        self.d_env_float = d_env_float
        self.env_categorical_specs = env_categorical_specs or {}
        self.use_time = use_time
        self.time_periods = time_periods
        self.d_time = d_time
        self.d_context = d_context
        self.use_missing_mask = use_missing_mask

        # === Process ID Embedding ===
        self.process_embedding = nn.Embedding(num_processes, d_proc)

        # === Continuous Environment Variables ===
        self.n_env_continuous = len(self.env_continuous_names)
        if self.n_env_continuous > 0:
            # Linear projection for continuous variables
            mask_dim = self.n_env_continuous if use_missing_mask else 0
            self.env_continuous_proj = nn.Linear(self.n_env_continuous + mask_dim, d_env_float)
        else:
            self.env_continuous_proj = None

        # === Categorical Environment Variables ===
        self.env_categorical_embeddings = nn.ModuleDict()
        self.env_categorical_dims = {}

        for var_name, cardinality in self.env_categorical_specs.items():
            # Formula: d = min(32, max(2, round(base * sqrt(cardinality))))
            d_cat = min(32, max(2, round(d_env_cat_base * np.sqrt(cardinality))))
            self.env_categorical_dims[var_name] = d_cat
            self.env_categorical_embeddings[var_name] = nn.Embedding(cardinality, d_cat)

        # === Temporal Encoding ===
        if self.use_time:
            self.time2vec = Time2Vec(num_periods=time_periods)
            self.time_proj = nn.Linear(self.time2vec.output_dim, d_time)
        else:
            self.time2vec = None
            self.time_proj = None

        # === Context Fusion MLP ===
        # Calculate total input dimension
        total_dim = d_proc  # Process ID always present as fallback (can be zero-embedding)

        if self.n_env_continuous > 0:
            total_dim += d_env_float

        total_dim += sum(self.env_categorical_dims.values())

        if self.use_time:
            total_dim += d_time

        # Build fusion MLP
        if context_mlp_hidden is None:
            context_mlp_hidden = [128, 64]

        layers = []
        in_dim = total_dim
        for hidden_dim in context_mlp_hidden:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(context_dropout)
            ])
            in_dim = hidden_dim

        # Final projection to context dimension
        layers.append(nn.Linear(in_dim, d_context))

        self.fusion_mlp = nn.Sequential(*layers)

    def forward(
        self,
        process_id: Optional[torch.Tensor] = None,
        env_continuous: Optional[torch.Tensor] = None,
        env_categorical: Optional[Dict[str, torch.Tensor]] = None,
        timestamp: Optional[torch.Tensor] = None,
        env_masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through context embedding module.

        Args:
            process_id: (batch,) long tensor with process IDs [0, num_processes-1]
            env_continuous: (batch, n_env_continuous) float tensor
            env_categorical: Dict {var_name: (batch,) long tensor}
            timestamp: (batch, 1) or (batch,) float tensor (Unix timestamp or normalized)
            env_masks: (batch, n_env_continuous) boolean tensor (True=present, False=missing)

        Returns:
            context: (batch, d_context) context vector
        """
        batch_size = self._infer_batch_size(process_id, env_continuous, env_categorical, timestamp)
        device = self._infer_device(process_id, env_continuous, env_categorical, timestamp)

        embeddings = []

        # === 1. Process ID Embedding ===
        if process_id is not None:
            proc_emb = self.process_embedding(process_id)  # (batch, d_proc)
        else:
            # Zero embedding if process_id not provided (backward compatibility)
            proc_emb = torch.zeros(batch_size, self.d_proc, device=device)

        embeddings.append(proc_emb)

        # === 2. Continuous Environment Variables ===
        if self.env_continuous_proj is not None and env_continuous is not None:
            # Handle missing values
            if self.use_missing_mask and env_masks is not None:
                # Zero-impute missing values
                env_continuous_imputed = env_continuous.clone()
                env_continuous_imputed[~env_masks] = 0.0

                # Concatenate data + mask (0/1 indicator)
                mask_float = env_masks.float()  # (batch, n_env_continuous)
                env_input = torch.cat([env_continuous_imputed, mask_float], dim=1)
            else:
                env_input = env_continuous

            env_cont_emb = self.env_continuous_proj(env_input)  # (batch, d_env_float)
            embeddings.append(env_cont_emb)

        # === 3. Categorical Environment Variables ===
        if env_categorical is not None:
            for var_name, var_tensor in env_categorical.items():
                if var_name in self.env_categorical_embeddings:
                    cat_emb = self.env_categorical_embeddings[var_name](var_tensor)
                    embeddings.append(cat_emb)

        # === 4. Temporal Encoding ===
        if self.time2vec is not None and timestamp is not None:
            time_enc = self.time2vec(timestamp)  # (batch, 1 + time_periods)
            time_emb = self.time_proj(time_enc)  # (batch, d_time)
            embeddings.append(time_emb)

        # === Fusion ===
        if len(embeddings) == 0:
            # Fallback: return zero context (should not happen with process_id fallback)
            return torch.zeros(batch_size, self.d_context, device=device)

        # Concatenate all embeddings
        context_concat = torch.cat(embeddings, dim=1)  # (batch, total_dim)

        # Pass through fusion MLP
        context = self.fusion_mlp(context_concat)  # (batch, d_context)

        return context

    def _infer_batch_size(
        self,
        process_id: Optional[torch.Tensor],
        env_continuous: Optional[torch.Tensor],
        env_categorical: Optional[Dict[str, torch.Tensor]],
        timestamp: Optional[torch.Tensor]
    ) -> int:
        """Infer batch size from any provided input"""
        if process_id is not None:
            return process_id.shape[0]
        if env_continuous is not None:
            return env_continuous.shape[0]
        if env_categorical is not None:
            for tensor in env_categorical.values():
                return tensor.shape[0]
        if timestamp is not None:
            return timestamp.shape[0]
        return 1  # Fallback

    def _infer_device(
        self,
        process_id: Optional[torch.Tensor],
        env_continuous: Optional[torch.Tensor],
        env_categorical: Optional[Dict[str, torch.Tensor]],
        timestamp: Optional[torch.Tensor]
    ) -> torch.device:
        """Infer device from any provided input"""
        if process_id is not None:
            return process_id.device
        if env_continuous is not None:
            return env_continuous.device
        if env_categorical is not None:
            for tensor in env_categorical.values():
                return tensor.device
        if timestamp is not None:
            return timestamp.device
        # Fallback to model parameters' device
        return next(self.parameters()).device


class ConditionalLayerNorm(nn.Module):
    """
    Layer Normalization with context-modulated scale (γ) and shift (β) parameters.

    Formula:
        LN(x) = (x - μ) / sqrt(σ² + ε)
        output = LN(x) * (γ₀ + Wγ·context) + (β₀ + Wβ·context)

    Args:
        hidden_dim: Dimension of input features
        context_dim: Dimension of context vector
        eps: Epsilon for numerical stability
    """

    def __init__(self, hidden_dim: int, context_dim: int, eps: float = 1e-5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.eps = eps

        # Base LayerNorm parameters (learnable)
        self.gamma_0 = nn.Parameter(torch.ones(hidden_dim))
        self.beta_0 = nn.Parameter(torch.zeros(hidden_dim))

        # Context-dependent modulation (learnable projections)
        self.W_gamma = nn.Linear(context_dim, hidden_dim)
        self.W_beta = nn.Linear(context_dim, hidden_dim)

        # Initialize modulation weights to small values (start close to standard LN)
        nn.init.zeros_(self.W_gamma.weight)
        nn.init.zeros_(self.W_gamma.bias)
        nn.init.zeros_(self.W_beta.weight)
        nn.init.zeros_(self.W_beta.bias)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, hidden_dim) or (batch, seq_len, hidden_dim)
            context: Context vector of shape (batch, context_dim). If None, behaves as standard LN.

        Returns:
            Normalized and modulated tensor with same shape as x
        """
        # Standard layer normalization
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        # Compute context-modulated parameters
        if context is not None:
            gamma_mod = self.W_gamma(context)  # (batch, hidden_dim)
            beta_mod = self.W_beta(context)    # (batch, hidden_dim)

            # Expand for broadcasting if x has sequence dimension
            if x.dim() == 3:  # (batch, seq_len, hidden_dim)
                gamma_mod = gamma_mod.unsqueeze(1)  # (batch, 1, hidden_dim)
                beta_mod = beta_mod.unsqueeze(1)    # (batch, 1, hidden_dim)

            gamma = self.gamma_0 + gamma_mod
            beta = self.beta_0 + beta_mod
        else:
            # Fallback to standard LayerNorm (no context modulation)
            gamma = self.gamma_0
            beta = self.beta_0

        # Apply affine transformation
        output = x_normalized * gamma + beta

        return output


class ConditionalBatchNorm1d(nn.Module):
    """
    Batch Normalization with context-modulated scale (γ) and shift (β) parameters.
    Maintains running statistics for batch normalization, but modulates affine parameters.

    Formula:
        BN(x) = (x - running_mean) / sqrt(running_var + ε)  [in eval mode]
        output = BN(x) * (γ₀ + Wγ·context) + (β₀ + Wβ·context)

    Args:
        hidden_dim: Dimension of input features
        context_dim: Dimension of context vector
        eps: Epsilon for numerical stability
        momentum: Momentum for running statistics update
    """

    def __init__(
        self,
        hidden_dim: int,
        context_dim: int,
        eps: float = 1e-5,
        momentum: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.eps = eps
        self.momentum = momentum

        # Running statistics (not learnable, updated during training)
        self.register_buffer('running_mean', torch.zeros(hidden_dim))
        self.register_buffer('running_var', torch.ones(hidden_dim))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        # Base affine parameters (learnable)
        self.gamma_0 = nn.Parameter(torch.ones(hidden_dim))
        self.beta_0 = nn.Parameter(torch.zeros(hidden_dim))

        # Context-dependent modulation (learnable projections)
        self.W_gamma = nn.Linear(context_dim, hidden_dim)
        self.W_beta = nn.Linear(context_dim, hidden_dim)

        # Initialize modulation weights to small values
        nn.init.zeros_(self.W_gamma.weight)
        nn.init.zeros_(self.W_gamma.bias)
        nn.init.zeros_(self.W_beta.weight)
        nn.init.zeros_(self.W_beta.bias)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, hidden_dim)
            context: Context vector of shape (batch, context_dim). If None, behaves as standard BN.

        Returns:
            Normalized and modulated tensor with same shape as x
        """
        # Update running statistics during training
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            # Update running statistics with momentum
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            self.num_batches_tracked += 1

            # Use batch statistics for normalization
            mean = batch_mean
            var = batch_var
        else:
            # Use running statistics in eval mode
            mean = self.running_mean
            var = self.running_var

        # Batch normalization
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        # Compute context-modulated parameters
        if context is not None:
            gamma_mod = self.W_gamma(context)  # (batch, hidden_dim)
            beta_mod = self.W_beta(context)    # (batch, hidden_dim)

            gamma = self.gamma_0 + gamma_mod
            beta = self.beta_0 + beta_mod
        else:
            # Fallback to standard BatchNorm (no context modulation)
            gamma = self.gamma_0
            beta = self.beta_0

        # Apply affine transformation
        output = x_normalized * gamma + beta

        return output
