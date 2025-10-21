"""
Dataset PyTorch per i dati del macchinario
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class MachineryDataset(Dataset):
    """
    Dataset PyTorch per dati del macchinario.

    Questo dataset gestisce i dati di input (parametri operativi) e output
    (valori misurati) del macchinario per l'uso con PyTorch DataLoader.

    Args:
        X (np.ndarray): Features di input, shape (n_samples, n_features)
        y (np.ndarray): Target di output, shape (n_samples, n_outputs)
        transform (callable, optional): Trasformazioni opzionali da applicare

    Esempio:
        >>> X = np.random.randn(1000, 10)
        >>> y = np.random.randn(1000, 5)
        >>> dataset = MachineryDataset(X, y)
        >>> print(len(dataset))  # 1000
        >>> x_sample, y_sample = dataset[0]
        >>> print(x_sample.shape, y_sample.shape)  # torch.Size([10]) torch.Size([5])
    """

    def __init__(self, X, y, transform=None):
        """
        Inizializza il dataset.

        Args:
            X: Input features (numpy array)
            y: Target outputs (numpy array)
            transform: Trasformazioni opzionali
        """
        # Converti in tensori PyTorch
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.transform = transform

        # Verifica che le dimensioni siano coerenti
        assert len(self.X) == len(self.y), \
            f"Mismatch: {len(self.X)} input samples vs {len(self.y)} output samples"

    def __len__(self):
        """Restituisce il numero di campioni nel dataset"""
        return len(self.X)

    def __getitem__(self, idx):
        """
        Restituisce un campione dal dataset.

        Args:
            idx (int): Indice del campione

        Returns:
            tuple: (input_tensor, output_tensor)
        """
        x_sample = self.X[idx]
        y_sample = self.y[idx]

        if self.transform:
            x_sample = self.transform(x_sample)

        return x_sample, y_sample

    def get_input_dim(self):
        """Restituisce la dimensione dell'input"""
        return self.X.shape[1]

    def get_output_dim(self):
        """Restituisce la dimensione dell'output"""
        return self.y.shape[1] if len(self.y.shape) > 1 else 1

    def get_statistics(self):
        """
        Restituisce statistiche sui dati.

        Returns:
            dict: Dizionario con statistiche di input e output
        """
        return {
            'n_samples': len(self),
            'input_dim': self.get_input_dim(),
            'output_dim': self.get_output_dim(),
            'input_mean': self.X.mean(dim=0),
            'input_std': self.X.std(dim=0),
            'output_mean': self.y.mean(dim=0),
            'output_std': self.y.std(dim=0),
        }
