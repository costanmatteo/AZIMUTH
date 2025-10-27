"""
PyTorch Dataset for machinery data
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class MachineryDataset(Dataset):
    """
    PyTorch Dataset for machinery data.

    This dataset handles input data (operational parameters) and output
    (measured values) of machinery for use with PyTorch DataLoader.

    Args:
        X (np.ndarray): Input features, shape (n_samples, n_features)
        y (np.ndarray): Target outputs, shape (n_samples, n_outputs)
        transform (callable, optional): Optional transformations to apply

    Example:
        >>> X = np.random.randn(1000, 10)
        >>> y = np.random.randn(1000, 5)
        >>> dataset = MachineryDataset(X, y)
        >>> print(len(dataset))  # 1000
        >>> x_sample, y_sample = dataset[0]
        >>> print(x_sample.shape, y_sample.shape)  # torch.Size([10]) torch.Size([5])
    """

    def __init__(self, X, y, transform=None):
        """
        Initialize the dataset.

        Args:
            X: Input features (numpy array)
            y: Target outputs (numpy array)
            transform: Optional transformations
        """
        # Convert to PyTorch tensors
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.transform = transform

        # Verify dimensions are consistent
        assert len(self.X) == len(self.y), \
            f"Mismatch: {len(self.X)} input samples vs {len(self.y)} output samples"

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.X)

    def __getitem__(self, idx):
        """
        Return a sample from the dataset.

        Args:
            idx (int): Sample index

        Returns:
            tuple: (input_tensor, output_tensor)
        """
        x_sample = self.X[idx]
        y_sample = self.y[idx]

        if self.transform:
            x_sample = self.transform(x_sample)

        return x_sample, y_sample

    def get_input_dim(self):
        """Return the input dimension"""
        return self.X.shape[1]

    def get_output_dim(self):
        """Return the output dimension"""
        return self.y.shape[1] if len(self.y.shape) > 1 else 1

    def get_statistics(self):
        """
        Return statistics about the data.

        Returns:
            dict: Dictionary with input and output statistics
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
