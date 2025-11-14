"""
Scenario Encoder: mappa parametri strutturali non-controllabili in scenario embeddings.

Questo modulo permette al policy generator di adattarsi a diverse condizioni
operative (temperatura, proprietà materiali, ecc.) che variano tra scenari.
"""

import torch
import torch.nn as nn


class ScenarioEncoder(nn.Module):
    """
    Encoder che mappa parametri strutturali in embedding denso.

    Parametri strutturali = parametri non-controllabili (es: temperatura ambiente,
    proprietà materiali) che variano tra scenari ma sono fissi all'interno di uno scenario.

    Args:
        n_structural_params (int): Numero di parametri strutturali in input
        embedding_dim (int): Dimensione dell'embedding output (default: 16)
        hidden_dim (int): Dimensione layer nascosto (default: 32)

    Input:
        structural_params: Tensor (batch_size, n_structural_params)

    Output:
        scenario_embedding: Tensor (batch_size, embedding_dim)
    """

    def __init__(self, n_structural_params, embedding_dim=16, hidden_dim=32):
        super(ScenarioEncoder, self).__init__()

        self.n_structural_params = n_structural_params
        self.embedding_dim = embedding_dim

        # Small MLP encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_structural_params, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, structural_params):
        """
        Forward pass.

        Args:
            structural_params: Tensor (batch_size, n_structural_params)

        Returns:
            Tensor (batch_size, embedding_dim)
        """
        return self.encoder(structural_params)


if __name__ == '__main__':
    # Test ScenarioEncoder
    print("Testing ScenarioEncoder...")

    # Example: 5 structural parameters (temps, material properties, etc.)
    n_params = 5
    embedding_dim = 16
    batch_size = 32

    # Create encoder
    encoder = ScenarioEncoder(
        n_structural_params=n_params,
        embedding_dim=embedding_dim
    )

    # Test forward pass
    structural_params = torch.randn(batch_size, n_params)
    embedding = encoder(structural_params)

    print(f"\nInput shape:  {structural_params.shape}")
    print(f"Output shape: {embedding.shape}")
    print(f"\nTotal parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    # Test with single sample
    single_params = torch.randn(1, n_params)
    single_embedding = encoder(single_params)
    print(f"\nSingle sample input:  {single_params.shape}")
    print(f"Single sample output: {single_embedding.shape}")

    print("\n✓ ScenarioEncoder test passed!")
