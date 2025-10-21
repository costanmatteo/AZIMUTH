"""
Rete Neurale per Predizione Output Macchinario

Questo modulo contiene la definizione della rete neurale che predice i valori
di output del macchinario (es. pressione, temperatura) basandosi sui parametri
di input scelti per operare la macchina.
"""

import torch
import torch.nn as nn


class MachineryPredictor(nn.Module):
    """
    Rete Neurale Feedforward per predire output del macchinario.

    Architettura:
    - Input Layer: parametri operativi (da configurare)
    - Hidden Layers: strati nascosti completamente connessi
    - Output Layer: valori predetti (pressione, temperatura, etc.)

    Args:
        input_size (int): Numero di parametri di input
        hidden_sizes (list): Lista con dimensioni dei layer nascosti
        output_size (int): Numero di valori di output da predire
        dropout_rate (float): Tasso di dropout per regolarizzazione (default: 0.2)

    Esempio:
        >>> model = MachineryPredictor(
        ...     input_size=10,      # 10 parametri operativi
        ...     hidden_sizes=[64, 32, 16],  # 3 layer nascosti
        ...     output_size=5       # 5 valori di output
        ... )
        >>> x = torch.randn(32, 10)  # batch di 32 esempi
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([32, 5])
    """

    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
        super(MachineryPredictor, self).__init__()

        # Costruzione dei layer
        layers = []
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())  # Funzione di attivazione
            layers.append(nn.Dropout(dropout_rate))  # Regolarizzazione
            prev_size = hidden_size

        # Output layer (nessuna attivazione, regressione lineare)
        layers.append(nn.Linear(prev_size, output_size))

        # Combina tutti i layer in un modulo sequenziale
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass della rete neurale.

        Args:
            x (torch.Tensor): Input tensor di shape (batch_size, input_size)

        Returns:
            torch.Tensor: Output predictions di shape (batch_size, output_size)
        """
        return self.network(x)


# Esempio di configurazioni comuni
def create_small_model(input_size, output_size):
    """Modello piccolo per dataset limitati"""
    return MachineryPredictor(
        input_size=input_size,
        hidden_sizes=[32, 16],
        output_size=output_size,
        dropout_rate=0.1
    )


def create_medium_model(input_size, output_size):
    """Modello medio per dataset di medie dimensioni"""
    return MachineryPredictor(
        input_size=input_size,
        hidden_sizes=[128, 64, 32],
        output_size=output_size,
        dropout_rate=0.2
    )


def create_large_model(input_size, output_size):
    """Modello grande per dataset ampi"""
    return MachineryPredictor(
        input_size=input_size,
        hidden_sizes=[256, 128, 64, 32],
        output_size=output_size,
        dropout_rate=0.3
    )
