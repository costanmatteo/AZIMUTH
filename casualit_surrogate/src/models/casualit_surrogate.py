"""
CasualiT Surrogate Model.

A transformer-based surrogate that learns to predict reliability F
from process chain trajectories using CasualiT's encoder-decoder architecture.

The model treats:
- Input: Trajectory sequence (n_processes, features_per_process)
- Target: Reliability F as a single-step "target sequence"
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Union

sys.path.insert(0, '/home/user/AZIMUTH')


class CasualiTSurrogate(nn.Module):
    """
    CasualiT-based surrogate for reliability prediction.

    Uses a transformer encoder-decoder architecture where:
    - Encoder processes the trajectory sequence (4 processes)
    - Decoder outputs reliability F (treating F as target sequence)

    Interface matches ProTSurrogate/ReliabilityFunction for drop-in replacement.
    """

    def __init__(self,
                 n_processes: int = 4,
                 features_per_process: int = 4,
                 embed_dim: int = 64,
                 n_heads: int = 4,
                 n_encoder_layers: int = 2,
                 n_decoder_layers: int = 2,
                 ff_dim: int = 128,
                 dropout: float = 0.1,
                 device: str = 'cpu'):
        """
        Args:
            n_processes: Number of processes in sequence (typically 4)
            features_per_process: Features per process step
            embed_dim: Transformer embedding dimension
            n_heads: Number of attention heads
            n_encoder_layers: Number of encoder layers
            n_decoder_layers: Number of decoder layers
            ff_dim: Feed-forward dimension
            dropout: Dropout rate
            device: Torch device
        """
        super().__init__()

        self.n_processes = n_processes
        self.features_per_process = features_per_process
        self.embed_dim = embed_dim
        self.device = device

        # Input embedding: project features to embed_dim
        self.input_embedding = nn.Linear(features_per_process, embed_dim)

        # Positional encoding for sequence positions
        self.pos_encoding = nn.Parameter(torch.randn(1, n_processes, embed_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        # Decoder for predicting F
        # We use a simple approach: learned query token + cross-attention
        self.decoder_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        # Output projection: embed_dim -> 1 (reliability F)
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, 1),
            nn.Sigmoid(),  # F is in [0, 1]
        )

        # Normalization stats for inference
        self.register_buffer('input_mean', None)
        self.register_buffer('input_std', None)

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: trajectory sequence -> reliability F.

        Args:
            x: Input trajectory, shape (batch, n_processes, features)

        Returns:
            F: Predicted reliability, shape (batch, 1)
        """
        batch_size = x.shape[0]

        # Embed input sequence
        x_embed = self.input_embedding(x)  # (batch, n_processes, embed_dim)

        # Add positional encoding
        x_embed = x_embed + self.pos_encoding

        # Encode trajectory
        memory = self.encoder(x_embed)  # (batch, n_processes, embed_dim)

        # Decode: use learned query token to attend to encoded trajectory
        query = self.decoder_query.expand(batch_size, -1, -1)  # (batch, 1, embed_dim)
        decoded = self.decoder(query, memory)  # (batch, 1, embed_dim)

        # Project to reliability F
        F = self.output_projection(decoded.squeeze(1))  # (batch, 1)

        return F

    def predict_reliability(self,
                           trajectory: Dict,
                           return_quality_scores: bool = False) -> torch.Tensor:
        """
        Predict reliability F from trajectory dict.

        Interface matches ReliabilityFunction/ProTSurrogate for drop-in replacement.

        Args:
            trajectory: Dict with process outputs (same format as ProTSurrogate)
                {
                    'laser': {'outputs_mean': tensor, 'outputs_var': tensor, ...},
                    'plasma': {...},
                    ...
                }
            return_quality_scores: If True, return empty dict (not supported)

        Returns:
            F: Predicted reliability tensor
        """
        # Convert trajectory dict to sequence tensor
        X = self._trajectory_to_tensor(trajectory)

        # Normalize if stats are available
        if self.input_mean is not None:
            X = (X - self.input_mean) / self.input_std

        # Forward pass
        with torch.no_grad():
            F = self.forward(X)

        if return_quality_scores:
            return F, {}  # Quality scores not supported for learned surrogate
        return F

    def compute_reliability(self,
                           trajectory: Dict,
                           return_quality_scores: bool = False) -> torch.Tensor:
        """Alias for predict_reliability (compatibility with ProTSurrogate)."""
        return self.predict_reliability(trajectory, return_quality_scores)

    def _trajectory_to_tensor(self, trajectory: Dict) -> torch.Tensor:
        """
        Convert trajectory dict to tensor format.

        Args:
            trajectory: Dict mapping process_name to data dict

        Returns:
            Tensor of shape (batch, n_processes, features)
        """
        process_order = ['laser', 'plasma', 'galvanic', 'microetch']
        sequences = []

        for process_name in process_order:
            if process_name not in trajectory:
                # Skip missing processes
                continue

            data = trajectory[process_name]

            # Get inputs if available
            inputs = data.get('inputs', torch.zeros(1, 2))
            if isinstance(inputs, np.ndarray):
                inputs = torch.tensor(inputs, dtype=torch.float32)

            # Get outputs
            outputs_mean = data.get('outputs_mean', data.get('outputs_sampled', torch.zeros(1, 1)))
            if isinstance(outputs_mean, np.ndarray):
                outputs_mean = torch.tensor(outputs_mean, dtype=torch.float32)

            outputs_var = data.get('outputs_var', torch.zeros_like(outputs_mean))
            if isinstance(outputs_var, np.ndarray):
                outputs_var = torch.tensor(outputs_var, dtype=torch.float32)

            # Concatenate features
            features = torch.cat([
                inputs.view(inputs.shape[0], -1),
                outputs_mean.view(outputs_mean.shape[0], -1),
                outputs_var.view(outputs_var.shape[0], -1),
            ], dim=-1)

            sequences.append(features)

        # Pad to same feature dimension
        max_features = max(s.shape[-1] for s in sequences)
        padded = []
        for s in sequences:
            if s.shape[-1] < max_features:
                padding = torch.zeros(s.shape[0], max_features - s.shape[-1], device=s.device)
                s = torch.cat([s, padding], dim=-1)
            padded.append(s)

        # Stack: (batch, n_processes, features)
        X = torch.stack(padded, dim=1)

        return X.to(self.device)

    def set_normalization_stats(self, mean: np.ndarray, std: np.ndarray):
        """Set normalization statistics for inference."""
        self.input_mean = torch.tensor(mean, dtype=torch.float32, device=self.device)
        self.input_std = torch.tensor(std, dtype=torch.float32, device=self.device)

    @classmethod
    def load(cls,
             checkpoint_path: str,
             device: str = 'cpu',
             config: Optional[Dict] = None) -> 'CasualiTSurrogate':
        """
        Load trained model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Torch device
            config: Model configuration (if not in checkpoint)

        Returns:
            Loaded CasualiTSurrogate instance
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Get config from checkpoint or use provided
        if 'config' in checkpoint:
            config = checkpoint['config']
        elif config is None:
            raise ValueError("Config not found in checkpoint and not provided")

        # Create model
        model = cls(
            n_processes=config.get('n_processes', 4),
            features_per_process=config.get('features_per_process', 4),
            embed_dim=config.get('embed_dim', 64),
            n_heads=config.get('n_heads', 4),
            n_encoder_layers=config.get('n_encoder_layers', 2),
            n_decoder_layers=config.get('n_decoder_layers', 2),
            ff_dim=config.get('ff_dim', 128),
            dropout=config.get('dropout', 0.1),
            device=device,
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load normalization stats if available
        if 'normalization_stats' in checkpoint:
            stats = checkpoint['normalization_stats']
            model.set_normalization_stats(stats['mean'], stats['std'])

        model.eval()
        return model

    def save(self,
             checkpoint_path: str,
             config: Dict,
             normalization_stats: Optional[Dict] = None,
             extra_data: Optional[Dict] = None):
        """
        Save model to checkpoint.

        Args:
            checkpoint_path: Path to save checkpoint
            config: Model configuration
            normalization_stats: Input normalization stats
            extra_data: Additional data to save
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': config,
        }

        if normalization_stats is not None:
            checkpoint['normalization_stats'] = normalization_stats

        if extra_data is not None:
            checkpoint.update(extra_data)

        torch.save(checkpoint, checkpoint_path)
