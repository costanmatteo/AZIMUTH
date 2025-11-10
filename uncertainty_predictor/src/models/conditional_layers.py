"""
Conditional embedding and normalization layers for multi-process training.

This module provides components for conditional learning across multiple processes:
- Time2Vec: Learnable temporal encoding
- ContextEmbeddingModule: Unified context fusion from process_id, env vars, and time
- ConditionalLayerNorm: Context-modulated layer normalization
- ConditionalBatchNorm1d: Context-modulated batch normalization
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple


class Time2Vec(nn.Module):
    """
    Time2Vec: Learnable temporal encoding with periodic and linear components.

    Reference: Kazemi et al. (2019) "Time2Vec: Learning a Vector Representation of Time"

    Args:
        d_time: Output dimension of temporal encoding
        time_periods: Number of periodic (sine/cosine) components (default: 4)
    """

    def __init__(self, d_time: int = 8, time_periods: int = 4):
        super().__init__()
        self.d_time = d_time
        self.time_periods = time_periods

        # Linear component: t2v[0] = w0*t + b0
        self.linear_w = nn.Parameter(torch.randn(1))
        self.linear_b = nn.Parameter(torch.randn(1))

        # Periodic components: t2v[i] = sin(wi*t + bi) for i=1..k
        n_periodic = d_time - 1
        self.periodic_w = nn.Parameter(torch.randn(n_periodic))
        self.periodic_b = nn.Parameter(torch.randn(n_periodic))

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: (batch_size,) normalized UNIX epoch timestamps

        Returns:
            encoding: (batch_size, d_time) temporal encoding
        """
        # Ensure timestamps is 2D
        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(-1)  # (batch_size, 1)

        # Linear component
        linear_term = self.linear_w * timestamps + self.linear_b  # (batch_size, 1)

        # Periodic components (sine activations)
        periodic_terms = torch.sin(self.periodic_w * timestamps + self.periodic_b)  # (batch_size, d_time-1)

        # Concatenate
        encoding = torch.cat([linear_term, periodic_terms], dim=-1)  # (batch_size, d_time)

        return encoding


class ContextEmbeddingModule(nn.Module):
    """
    Unified context embedding module that fuses:
    - Process ID embeddings
    - Continuous environmental variables (with missing value masks)
    - Categorical environmental variables
    - Temporal encoding (Time2Vec)

    All embeddings are fused through an MLP into a unified context vector.

    Args:
        num_processes: Number of distinct processes (e.g., 4 for Laser/Plasma/Galvanic/Microetch)
        d_proc: Embedding dimension for process IDs
        env_continuous: List of continuous env variable names (e.g., ['ambient_temp', 'humidity'])
        d_env_float: Embedding dimension for each continuous variable
        use_missing_mask: Whether to use missing value masks for continuous variables
        env_categorical: Dict mapping categorical variable names to number of categories
                        (e.g., {'batch_id': 10, 'operator_id': 5, 'shift': 3})
        d_env_cat_base: Base multiplier for categorical embedding dimension (dim = max(2, int(d_base * log(n))))
        use_time: Whether to use temporal encoding
        time_periods: Number of periodic components in Time2Vec
        d_time: Output dimension of temporal encoding
        d_context: Final unified context vector dimension
        context_mlp_hidden: Hidden layer sizes for context fusion MLP (e.g., [128, 64])
        context_dropout: Dropout rate in context MLP
    """

    def __init__(
        self,
        num_processes: int = 4,
        d_proc: int = 16,
        env_continuous: Optional[List[str]] = None,
        d_env_float: int = 16,
        use_missing_mask: bool = True,
        env_categorical: Optional[Dict[str, int]] = None,
        d_env_cat_base: float = 1.6,
        use_time: bool = True,
        time_periods: int = 4,
        d_time: int = 8,
        d_context: int = 64,
        context_mlp_hidden: Optional[List[int]] = None,
        context_dropout: float = 0.1
    ):
        super().__init__()

        self.num_processes = num_processes
        self.d_proc = d_proc
        self.env_continuous = env_continuous or []
        self.d_env_float = d_env_float
        self.use_missing_mask = use_missing_mask
        self.env_categorical = env_categorical or {}
        self.d_env_cat_base = d_env_cat_base
        self.use_time = use_time
        self.d_time = d_time
        self.d_context = d_context

        # Process ID embedding
        self.process_embedding = nn.Embedding(num_processes, d_proc)

        # Continuous environment variable embeddings
        self.env_float_projections = nn.ModuleDict()
        for var_name in self.env_continuous:
            input_dim = 2 if use_missing_mask else 1  # [value, mask] or just [value]
            self.env_float_projections[var_name] = nn.Sequential(
                nn.Linear(input_dim, d_env_float),
                nn.ReLU(),
                nn.Linear(d_env_float, d_env_float)
            )

        # Categorical environment variable embeddings
        self.env_cat_embeddings = nn.ModuleDict()
        self.cat_embed_dims = {}
        for var_name, n_categories in self.env_categorical.items():
            # Embedding dimension: max(2, int(d_base * log(n_categories)))
            d_cat = max(2, int(d_env_cat_base * np.log(n_categories)))
            self.cat_embed_dims[var_name] = d_cat
            self.env_cat_embeddings[var_name] = nn.Embedding(n_categories, d_cat)

        # Temporal encoding
        if self.use_time:
            self.time_encoder = Time2Vec(d_time=d_time, time_periods=time_periods)

        # Calculate total embedding dimension
        total_dim = d_proc
        total_dim += len(self.env_continuous) * d_env_float
        total_dim += sum(self.cat_embed_dims.values())
        if self.use_time:
            total_dim += d_time

        # Context fusion MLP
        if context_mlp_hidden is None:
            context_mlp_hidden = [128, 64]

        layers = []
        prev_dim = total_dim
        for hidden_dim in context_mlp_hidden:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(context_dropout)
            ])
            prev_dim = hidden_dim

        # Final projection to d_context
        layers.append(nn.Linear(prev_dim, d_context))

        self.context_mlp = nn.Sequential(*layers)

    def forward(
        self,
        process_id: torch.Tensor,
        env_continuous: Optional[Dict[str, torch.Tensor]] = None,
        env_categorical: Optional[Dict[str, torch.Tensor]] = None,
        timestamp: Optional[torch.Tensor] = None,
        env_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Args:
            process_id: (batch_size,) process IDs (0 to num_processes-1)
            env_continuous: Dict of continuous env variables, each (batch_size,)
            env_categorical: Dict of categorical env variables, each (batch_size,) with integer indices
            timestamp: (batch_size,) normalized UNIX timestamps
            env_masks: Dict of missing value masks for continuous variables, each (batch_size,)
                      1.0 = present, 0.0 = missing

        Returns:
            context: (batch_size, d_context) unified context vector
        """
        embeddings = []

        # 1. Process ID embedding
        proc_emb = self.process_embedding(process_id)  # (batch_size, d_proc)
        embeddings.append(proc_emb)

        # 2. Continuous environment variables
        if env_continuous is None:
            env_continuous = {}
        if env_masks is None:
            env_masks = {}

        for var_name in self.env_continuous:
            var_value = env_continuous.get(var_name, torch.zeros(process_id.size(0), device=process_id.device))

            if self.use_missing_mask:
                mask = env_masks.get(var_name, torch.ones(process_id.size(0), device=process_id.device))
                # Stack [value, mask]
                var_input = torch.stack([var_value, mask], dim=-1)  # (batch_size, 2)
            else:
                var_input = var_value.unsqueeze(-1)  # (batch_size, 1)

            var_emb = self.env_float_projections[var_name](var_input)  # (batch_size, d_env_float)
            embeddings.append(var_emb)

        # 3. Categorical environment variables
        if env_categorical is None:
            env_categorical = {}

        for var_name in self.env_categorical.keys():
            var_idx = env_categorical.get(var_name, torch.zeros(process_id.size(0), dtype=torch.long, device=process_id.device))
            var_emb = self.env_cat_embeddings[var_name](var_idx)  # (batch_size, d_cat)
            embeddings.append(var_emb)

        # 4. Temporal encoding
        if self.use_time and timestamp is not None:
            time_emb = self.time_encoder(timestamp)  # (batch_size, d_time)
            embeddings.append(time_emb)

        # Concatenate all embeddings
        concat_emb = torch.cat(embeddings, dim=-1)  # (batch_size, total_dim)

        # Fuse through MLP
        context = self.context_mlp(concat_emb)  # (batch_size, d_context)

        return context


class ConditionalLayerNorm(nn.Module):
    """
    Layer Normalization with context-modulated scale and shift parameters.

    Formula:
        normalized = (x - mean) / sqrt(var + eps)
        gamma = gamma_0 + W_gamma @ context
        beta = beta_0 + W_beta @ context
        output = gamma * normalized + beta

    Args:
        normalized_shape: Shape of input to normalize (e.g., feature dimension)
        d_context: Dimension of context vector
        eps: Epsilon for numerical stability
    """

    def __init__(self, normalized_shape: int, d_context: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.d_context = d_context
        self.eps = eps

        # Base parameters (γ₀, β₀)
        self.gamma_0 = nn.Parameter(torch.ones(normalized_shape))
        self.beta_0 = nn.Parameter(torch.zeros(normalized_shape))

        # Context modulation weights
        self.W_gamma = nn.Linear(d_context, normalized_shape, bias=False)
        self.W_beta = nn.Linear(d_context, normalized_shape, bias=False)

        # Initialize modulation weights to zero (start with standard LayerNorm behavior)
        nn.init.zeros_(self.W_gamma.weight)
        nn.init.zeros_(self.W_beta.weight)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, ..., normalized_shape) input tensor
            context: (batch_size, d_context) context vector

        Returns:
            output: (batch_size, ..., normalized_shape) normalized tensor
        """
        # Standard layer normalization
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        # Context-modulated parameters
        gamma = self.gamma_0 + self.W_gamma(context)  # (batch_size, normalized_shape)
        beta = self.beta_0 + self.W_beta(context)    # (batch_size, normalized_shape)

        # Apply affine transformation
        output = gamma * x_normalized + beta

        return output


class ConditionalBatchNorm1d(nn.Module):
    """
    Batch Normalization with context-modulated scale and shift parameters.

    Similar to ConditionalLayerNorm but uses batch statistics instead of layer statistics.
    Useful as an alternative normalization strategy.

    Args:
        num_features: Number of features (channels)
        d_context: Dimension of context vector
        eps: Epsilon for numerical stability
        momentum: Momentum for running statistics
    """

    def __init__(
        self,
        num_features: int,
        d_context: int,
        eps: float = 1e-5,
        momentum: float = 0.1
    ):
        super().__init__()
        self.num_features = num_features
        self.d_context = d_context
        self.eps = eps
        self.momentum = momentum

        # Running statistics (not modulated)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        # Base parameters (γ₀, β₀)
        self.gamma_0 = nn.Parameter(torch.ones(num_features))
        self.beta_0 = nn.Parameter(torch.zeros(num_features))

        # Context modulation weights
        self.W_gamma = nn.Linear(d_context, num_features, bias=False)
        self.W_beta = nn.Linear(d_context, num_features, bias=False)

        # Initialize modulation weights to zero
        nn.init.zeros_(self.W_gamma.weight)
        nn.init.zeros_(self.W_beta.weight)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_features) input tensor
            context: (batch_size, d_context) context vector

        Returns:
            output: (batch_size, num_features) normalized tensor
        """
        # Update running statistics during training
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
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        # Context-modulated parameters
        gamma = self.gamma_0 + self.W_gamma(context)  # (batch_size, num_features)
        beta = self.beta_0 + self.W_beta(context)    # (batch_size, num_features)

        # Apply affine transformation
        output = gamma * x_normalized + beta

        return output


def test_conditional_layers():
    """Test function to verify all components work correctly."""
    print("Testing Conditional Layers...")

    batch_size = 32
    d_context = 64

    # Test Time2Vec
    print("\n1. Testing Time2Vec...")
    time_encoder = Time2Vec(d_time=8, time_periods=4)
    timestamps = torch.randn(batch_size)  # Normalized timestamps
    time_encoding = time_encoder(timestamps)
    print(f"   Input timestamps shape: {timestamps.shape}")
    print(f"   Output encoding shape: {time_encoding.shape}")
    assert time_encoding.shape == (batch_size, 8), "Time2Vec output shape mismatch"
    print("   ✓ Time2Vec passed")

    # Test ContextEmbeddingModule
    print("\n2. Testing ContextEmbeddingModule...")
    context_module = ContextEmbeddingModule(
        num_processes=4,
        d_proc=16,
        env_continuous=['ambient_temp', 'humidity'],
        d_env_float=16,
        use_missing_mask=True,
        env_categorical={'batch_id': 10, 'operator_id': 5, 'shift': 3},
        d_env_cat_base=1.6,
        use_time=True,
        time_periods=4,
        d_time=8,
        d_context=d_context,
        context_mlp_hidden=[128, 64],
        context_dropout=0.1
    )

    process_id = torch.randint(0, 4, (batch_size,))
    env_continuous = {
        'ambient_temp': torch.randn(batch_size),
        'humidity': torch.randn(batch_size)
    }
    env_categorical = {
        'batch_id': torch.randint(0, 10, (batch_size,)),
        'operator_id': torch.randint(0, 5, (batch_size,)),
        'shift': torch.randint(0, 3, (batch_size,))
    }
    env_masks = {
        'ambient_temp': torch.ones(batch_size),
        'humidity': torch.bernoulli(torch.ones(batch_size) * 0.8)  # 20% missing
    }

    context = context_module(
        process_id=process_id,
        env_continuous=env_continuous,
        env_categorical=env_categorical,
        timestamp=timestamps,
        env_masks=env_masks
    )
    print(f"   Output context shape: {context.shape}")
    assert context.shape == (batch_size, d_context), "ContextEmbeddingModule output shape mismatch"
    print("   ✓ ContextEmbeddingModule passed")

    # Test ConditionalLayerNorm
    print("\n3. Testing ConditionalLayerNorm...")
    feature_dim = 128
    x = torch.randn(batch_size, feature_dim)
    cond_ln = ConditionalLayerNorm(normalized_shape=feature_dim, d_context=d_context)
    x_normed = cond_ln(x, context)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {x_normed.shape}")
    assert x_normed.shape == x.shape, "ConditionalLayerNorm output shape mismatch"
    print("   ✓ ConditionalLayerNorm passed")

    # Test ConditionalBatchNorm1d
    print("\n4. Testing ConditionalBatchNorm1d...")
    cond_bn = ConditionalBatchNorm1d(num_features=feature_dim, d_context=d_context)
    cond_bn.train()
    x_normed_bn = cond_bn(x, context)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {x_normed_bn.shape}")
    assert x_normed_bn.shape == x.shape, "ConditionalBatchNorm1d output shape mismatch"
    print("   ✓ ConditionalBatchNorm1d passed")

    print("\n✅ All tests passed successfully!")


if __name__ == "__main__":
    test_conditional_layers()
