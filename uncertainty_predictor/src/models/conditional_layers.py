"""
Conditional Normalization Layers and Context Embedding for Multi-Process Adaptation

This module implements conditioning mechanisms that allow a neural network to adapt
to different processes and environmental conditions via learned embeddings and
conditional normalization.

Components:
- Time2Vec: Learnable temporal encoding
- ConditionalLayerNorm: LayerNorm with context-dependent affine parameters
- ConditionalBatchNorm1d: BatchNorm1d with context-dependent affine parameters
- ContextEmbedding: Fusion of process_id, environmental, and temporal features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional


class Time2Vec(nn.Module):
    """
    Time2Vec: Learnable time representation with periodic components.

    Reference: Kazemi et al. "Time2Vec: Learning a Vector Representation of Time" (2019)

    out = [linear(t), sin(ω₁t + φ₁), ..., sin(ωₖt + φₖ)]

    where ω and φ are learnable parameters.
    """

    def __init__(self, num_periods: int = 3):
        """
        Args:
            num_periods: Number of periodic (sinusoidal) components
        """
        super().__init__()
        self.num_periods = num_periods

        # Linear component weight and bias
        self.linear_weight = nn.Parameter(torch.randn(1))
        self.linear_bias = nn.Parameter(torch.randn(1))

        # Periodic components: frequencies (ω) and phases (φ)
        self.periodic_weights = nn.Parameter(torch.randn(num_periods))
        self.periodic_biases = nn.Parameter(torch.randn(num_periods))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values, shape (batch_size,) or (batch_size, 1)

        Returns:
            Time encoding, shape (batch_size, 1 + num_periods)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (batch_size, 1)

        # Linear component
        linear_component = self.linear_weight * t + self.linear_bias  # (batch_size, 1)

        # Periodic components: sin(ω*t + φ)
        periodic_components = torch.sin(
            self.periodic_weights * t + self.periodic_biases
        )  # (batch_size, num_periods)

        # Concatenate
        encoding = torch.cat([linear_component, periodic_components], dim=-1)
        return encoding  # (batch_size, 1 + num_periods)

    @property
    def output_dim(self) -> int:
        return 1 + self.num_periods


class SinCosEncoding(nn.Module):
    """
    Simple sinusoidal time encoding: [sin(t/T), cos(t/T), sin(2t/T), cos(2t/T), ...]

    Alternative to Time2Vec with fixed (non-learnable) frequencies.
    """

    def __init__(self, num_periods: int = 3, base_period: float = 86400.0):
        """
        Args:
            num_periods: Number of sin/cos pairs
            base_period: Base period in seconds (default: 86400 = 1 day)
        """
        super().__init__()
        self.num_periods = num_periods
        self.base_period = base_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values in seconds, shape (batch_size,)

        Returns:
            Encoding, shape (batch_size, 2 * num_periods)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (batch_size, 1)

        encodings = []
        for k in range(1, self.num_periods + 1):
            freq = 2 * np.pi * k / self.base_period
            encodings.append(torch.sin(freq * t))
            encodings.append(torch.cos(freq * t))

        return torch.cat(encodings, dim=-1)  # (batch_size, 2 * num_periods)

    @property
    def output_dim(self) -> int:
        return 2 * self.num_periods


class ConditionalLayerNorm(nn.Module):
    """
    Conditional Layer Normalization with context-dependent affine parameters.

    Standard LayerNorm: out = γ * (x - μ) / σ + β
    Conditional variant: γ = γ₀ + Wγ(context), β = β₀ + Wβ(context)

    This allows the normalization to adapt to different processes and conditions.
    """

    def __init__(self, hidden_dim: int, context_dim: int, eps: float = 1e-5):
        """
        Args:
            hidden_dim: Dimension of input features
            context_dim: Dimension of context vector
            eps: Epsilon for numerical stability
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.eps = eps

        # Base affine parameters (learnable)
        self.gamma_0 = nn.Parameter(torch.ones(hidden_dim))
        self.beta_0 = nn.Parameter(torch.zeros(hidden_dim))

        # Context-dependent affine projections
        self.gamma_proj = nn.Linear(context_dim, hidden_dim)
        self.beta_proj = nn.Linear(context_dim, hidden_dim)

        # Initialize projections to small values (so initially γ ≈ γ₀, β ≈ β₀)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input features, shape (batch_size, hidden_dim)
            context: Context vector, shape (batch_size, context_dim) or None

        Returns:
            Normalized features, shape (batch_size, hidden_dim)
        """
        # Standard LayerNorm: normalize across features
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Compute context-dependent affine parameters
        if context is not None:
            gamma = self.gamma_0 + self.gamma_proj(context)
            beta = self.beta_0 + self.beta_proj(context)
        else:
            # If no context, use base parameters only (for compatibility)
            gamma = self.gamma_0
            beta = self.beta_0

        # Apply affine transformation
        return gamma * x_norm + beta


class ConditionalBatchNorm1d(nn.Module):
    """
    Conditional Batch Normalization with context-dependent affine parameters.

    Standard BatchNorm: out = γ * (x - μ_batch) / σ_batch + β
    Conditional variant: γ = γ₀ + Wγ(context), β = β₀ + Wβ(context)

    Running statistics (μ, σ) are shared, only affine params are conditioned.
    """

    def __init__(self, num_features: int, context_dim: int, eps: float = 1e-5, momentum: float = 0.1):
        """
        Args:
            num_features: Number of features (C from N×C input)
            context_dim: Dimension of context vector
            eps: Epsilon for numerical stability
            momentum: Momentum for running statistics
        """
        super().__init__()
        self.num_features = num_features
        self.context_dim = context_dim
        self.eps = eps
        self.momentum = momentum

        # Base affine parameters (learnable)
        self.gamma_0 = nn.Parameter(torch.ones(num_features))
        self.beta_0 = nn.Parameter(torch.zeros(num_features))

        # Running statistics (not learnable)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        # Context-dependent affine projections
        self.gamma_proj = nn.Linear(context_dim, num_features)
        self.beta_proj = nn.Linear(context_dim, num_features)

        # Initialize projections to small values
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input features, shape (batch_size, num_features)
            context: Context vector, shape (batch_size, context_dim) or None

        Returns:
            Normalized features, shape (batch_size, num_features)
        """
        # Use F.batch_norm for automatic handling of train/eval mode
        # In training: use batch statistics and update running stats
        # In eval: use running statistics
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            # Update running statistics
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
                self.num_batches_tracked += 1

            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Compute context-dependent affine parameters
        if context is not None:
            gamma = self.gamma_0 + self.gamma_proj(context)
            beta = self.beta_0 + self.beta_proj(context)
        else:
            gamma = self.gamma_0
            beta = self.beta_0

        # Apply affine transformation
        return gamma * x_norm + beta


class ContextEmbedding(nn.Module):
    """
    Context Embedding: Fusion of process ID, environmental features, and temporal information.

    Input components (all optional):
    - process_id: Categorical process identifier (embedded)
    - env_continuous: Continuous environmental features (projected)
    - env_categorical: Categorical environmental features (embedded separately)
    - timestamp: Temporal information (encoded via Time2Vec or SinCos)

    Output:
    - context: Fused context vector of dimension d_context
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary with keys:
                - num_processes: Number of process types
                - d_proc: Process embedding dimension
                - d_context: Final context dimension
                - env_continuous: dict with 'enabled', 'features', 'd_env_float', 'handle_missing'
                - env_categorical: dict with 'enabled', 'features', 'd_embedding_rule', 'd_embedding_fixed'
                - time_encoding: dict with 'enabled', 'method', 'd_time', 'num_periods'
        """
        super().__init__()
        self.config = config

        # Extract config parameters
        self.num_processes = config.get('num_processes', 4)
        self.d_proc = config.get('d_proc', 16)
        self.d_context = config.get('d_context', 64)

        # Process embedding
        self.process_embedding = nn.Embedding(self.num_processes, self.d_proc)

        # Environmental continuous features
        env_cont_config = config.get('env_continuous', {})
        self.env_cont_enabled = env_cont_config.get('enabled', False)
        if self.env_cont_enabled:
            self.env_cont_features = env_cont_config.get('features', [])
            self.n_env_cont = len(self.env_cont_features)
            self.d_env_float = env_cont_config.get('d_env_float', 32)
            self.handle_missing = env_cont_config.get('handle_missing', True)

            # Projection: concatenate [features, mask] if handle_missing, else just features
            input_dim = self.n_env_cont * 2 if self.handle_missing else self.n_env_cont
            self.env_cont_proj = nn.Linear(input_dim, self.d_env_float)
        else:
            self.n_env_cont = 0
            self.d_env_float = 0

        # Environmental categorical features
        env_cat_config = config.get('env_categorical', {})
        self.env_cat_enabled = env_cat_config.get('enabled', False)
        if self.env_cat_enabled:
            self.env_cat_features = env_cat_config.get('features', {})
            d_embedding_rule = env_cat_config.get('d_embedding_rule', 'sqrt')
            d_embedding_fixed = env_cat_config.get('d_embedding_fixed', 16)

            self.env_cat_embeddings = nn.ModuleDict()
            self.env_cat_dims = {}

            for feat_name, cardinality in self.env_cat_features.items():
                if d_embedding_rule == 'sqrt':
                    d_emb = min(32, max(4, int(np.ceil(1.6 * np.sqrt(cardinality)))))
                else:
                    d_emb = d_embedding_fixed

                self.env_cat_embeddings[feat_name] = nn.Embedding(cardinality, d_emb)
                self.env_cat_dims[feat_name] = d_emb

            self.d_env_cat = sum(self.env_cat_dims.values())
        else:
            self.d_env_cat = 0

        # Temporal encoding
        time_config = config.get('time_encoding', {})
        self.time_enabled = time_config.get('enabled', False)
        if self.time_enabled:
            time_method = time_config.get('method', 'time2vec')
            num_periods = time_config.get('num_periods', 3)

            if time_method == 'time2vec':
                self.time_encoder = Time2Vec(num_periods=num_periods)
            elif time_method == 'sincos':
                self.time_encoder = SinCosEncoding(num_periods=num_periods)
            else:
                raise ValueError(f"Unknown time encoding method: {time_method}")

            self.d_time = self.time_encoder.output_dim

            # Project time encoding to d_time config dimension
            d_time_config = time_config.get('d_time', 16)
            self.time_proj = nn.Linear(self.d_time, d_time_config)
            self.d_time = d_time_config
        else:
            self.d_time = 0

        # Fusion MLP: concatenate all embeddings → d_context
        total_dim = self.d_proc + self.d_env_float + self.d_env_cat + self.d_time
        self.fusion_mlp = nn.Sequential(
            nn.Linear(total_dim, self.d_context * 2),
            nn.ReLU(),
            nn.Linear(self.d_context * 2, self.d_context),
        )

    def forward(
        self,
        process_id: Optional[torch.Tensor] = None,
        env_cont: Optional[torch.Tensor] = None,
        env_cont_mask: Optional[torch.Tensor] = None,
        env_cat: Optional[Dict[str, torch.Tensor]] = None,
        timestamp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            process_id: Process IDs, shape (batch_size,), dtype long
            env_cont: Continuous env features, shape (batch_size, n_env_cont)
            env_cont_mask: Mask for missing values (True=valid), shape (batch_size, n_env_cont)
            env_cat: Dict of categorical env features, each shape (batch_size,)
            timestamp: Timestamps, shape (batch_size,)

        Returns:
            context: Fused context vector, shape (batch_size, d_context)
        """
        embeddings = []

        # Determine batch size
        if process_id is not None:
            batch_size = process_id.shape[0]
        elif env_cont is not None:
            batch_size = env_cont.shape[0]
        elif env_cat is not None and len(env_cat) > 0:
            batch_size = next(iter(env_cat.values())).shape[0]
        elif timestamp is not None:
            batch_size = timestamp.shape[0]
        else:
            raise ValueError("At least one input must be provided to determine batch size")

        device = next(self.parameters()).device

        # 1. Process embedding
        if process_id is not None:
            proc_emb = self.process_embedding(process_id)  # (batch_size, d_proc)
        else:
            proc_emb = torch.zeros(batch_size, self.d_proc, device=device)
        embeddings.append(proc_emb)

        # 2. Continuous environmental features
        if self.env_cont_enabled:
            if env_cont is not None:
                # Handle missing values
                if self.handle_missing:
                    # Replace NaN with 0 and create mask
                    env_cont_filled = torch.nan_to_num(env_cont, nan=0.0)

                    if env_cont_mask is None:
                        # Create mask: True where not NaN
                        env_cont_mask = ~torch.isnan(env_cont)

                    # Convert mask to float: 1=valid, 0=missing
                    mask_float = env_cont_mask.float()

                    # Concatenate [features, mask]
                    env_cont_input = torch.cat([env_cont_filled, mask_float], dim=-1)
                else:
                    # No missing handling: just use features (NaN → 0)
                    env_cont_input = torch.nan_to_num(env_cont, nan=0.0)

                env_cont_emb = self.env_cont_proj(env_cont_input)
            else:
                env_cont_emb = torch.zeros(batch_size, self.d_env_float, device=device)
            embeddings.append(env_cont_emb)

        # 3. Categorical environmental features
        if self.env_cat_enabled:
            env_cat_embs = []
            for feat_name, emb_layer in self.env_cat_embeddings.items():
                if env_cat is not None and feat_name in env_cat:
                    cat_emb = emb_layer(env_cat[feat_name])
                else:
                    cat_emb = torch.zeros(batch_size, self.env_cat_dims[feat_name], device=device)
                env_cat_embs.append(cat_emb)

            if env_cat_embs:
                embeddings.append(torch.cat(env_cat_embs, dim=-1))

        # 4. Temporal encoding
        if self.time_enabled:
            if timestamp is not None:
                time_enc = self.time_encoder(timestamp)
                time_emb = self.time_proj(time_enc)
            else:
                time_emb = torch.zeros(batch_size, self.d_time, device=device)
            embeddings.append(time_emb)

        # Concatenate all embeddings
        combined = torch.cat(embeddings, dim=-1)  # (batch_size, total_dim)

        # Fusion MLP
        context = self.fusion_mlp(combined)  # (batch_size, d_context)

        return context
