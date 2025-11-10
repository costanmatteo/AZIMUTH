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


class ConditionalMachineryDataset(Dataset):
    """
    PyTorch Dataset for machinery data with conditional variables.

    This dataset handles input data (operational parameters), output data,
    and additional conditioning variables (process_id, env variables, timestamp)
    for conditional uncertainty quantification.

    Args:
        data_dict (dict): Dictionary with keys:
            - 'X': numpy array (n_samples, n_features)
            - 'y': numpy array (n_samples, n_outputs)
            - 'process_id': numpy array (n_samples,) or None
            - 'env_continuous': numpy array (n_samples, n_env_cont) or None
            - 'env_continuous_masks': numpy array (n_samples, n_env_cont) boolean or None
            - 'env_categorical': dict {var_name: numpy array (n_samples,)} or None
            - 'timestamp': numpy array (n_samples,) or None
        transform (callable, optional): Optional transformations to apply

    Example:
        >>> data_dict = {
        ...     'X': np.random.randn(1000, 10),
        ...     'y': np.random.randn(1000, 5),
        ...     'process_id': np.random.randint(0, 4, 1000),
        ...     'env_continuous': np.random.randn(1000, 2),
        ...     'env_continuous_masks': np.ones((1000, 2), dtype=bool),
        ...     'env_categorical': {'batch_id': np.random.randint(0, 10, 1000)},
        ...     'timestamp': np.random.randn(1000)
        ... }
        >>> dataset = ConditionalMachineryDataset(data_dict)
        >>> sample = dataset[0]
        >>> print(sample.keys())  # dict_keys(['X', 'y', 'process_id', ...])
    """

    def __init__(self, data_dict, transform=None):
        """
        Initialize the conditional dataset.

        Args:
            data_dict: Dictionary with all data tensors
            transform: Optional transformations
        """
        # Required fields
        self.X = data_dict['X']
        self.y = data_dict['y']

        # Optional conditioning fields
        self.process_id = data_dict.get('process_id', None)
        self.env_continuous = data_dict.get('env_continuous', None)
        self.env_continuous_masks = data_dict.get('env_continuous_masks', None)
        self.env_categorical = data_dict.get('env_categorical', None)
        self.timestamp = data_dict.get('timestamp', None)

        self.transform = transform

        # Verify dimensions are consistent
        n_samples = len(self.X)
        assert len(self.y) == n_samples, \
            f"Mismatch: {len(self.X)} input samples vs {len(self.y)} output samples"

        if self.process_id is not None:
            assert len(self.process_id) == n_samples, \
                f"process_id length {len(self.process_id)} != {n_samples}"

        if self.env_continuous is not None:
            assert len(self.env_continuous) == n_samples, \
                f"env_continuous length {len(self.env_continuous)} != {n_samples}"

        if self.env_continuous_masks is not None:
            assert len(self.env_continuous_masks) == n_samples, \
                f"env_continuous_masks length {len(self.env_continuous_masks)} != {n_samples}"

        if self.env_categorical is not None:
            for var_name, var_values in self.env_categorical.items():
                assert len(var_values) == n_samples, \
                    f"env_categorical[{var_name}] length {len(var_values)} != {n_samples}"

        if self.timestamp is not None:
            assert len(self.timestamp) == n_samples, \
                f"timestamp length {len(self.timestamp)} != {n_samples}"

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.X)

    def __getitem__(self, idx):
        """
        Return a sample from the dataset as a dictionary.

        Args:
            idx (int): Sample index

        Returns:
            dict: Dictionary with keys 'X', 'y', and optional conditioning variables
        """
        sample = {
            'X': self.X[idx],
            'y': self.y[idx]
        }

        if self.process_id is not None:
            sample['process_id'] = self.process_id[idx]

        if self.env_continuous is not None:
            sample['env_continuous'] = self.env_continuous[idx]

        if self.env_continuous_masks is not None:
            sample['env_continuous_masks'] = self.env_continuous_masks[idx]

        if self.env_categorical is not None:
            sample['env_categorical'] = {
                var_name: var_values[idx]
                for var_name, var_values in self.env_categorical.items()
            }

        if self.timestamp is not None:
            sample['timestamp'] = self.timestamp[idx]

        if self.transform:
            sample['X'] = self.transform(sample['X'])

        return sample

    def get_input_dim(self):
        """Return the input dimension"""
        return self.X.shape[1]

    def get_output_dim(self):
        """Return the output dimension"""
        return self.y.shape[1] if len(self.y.shape) > 1 else 1

    def has_conditioning(self):
        """Check if dataset has any conditioning variables"""
        return (self.process_id is not None or
                self.env_continuous is not None or
                self.env_categorical is not None or
                self.timestamp is not None)

    def get_statistics(self):
        """
        Return statistics about the data including conditioning variables.

        Returns:
            dict: Dictionary with input, output, and conditioning statistics
        """
        stats = {
            'n_samples': len(self),
            'input_dim': self.get_input_dim(),
            'output_dim': self.get_output_dim(),
            'has_conditioning': self.has_conditioning(),
        }

        if self.process_id is not None:
            stats['num_processes'] = len(np.unique(self.process_id))
            stats['process_distribution'] = {
                int(pid): int(count)
                for pid, count in zip(*np.unique(self.process_id, return_counts=True))
            }

        if self.env_continuous is not None:
            stats['env_continuous_dim'] = self.env_continuous.shape[1]

        if self.env_categorical is not None:
            stats['env_categorical_vars'] = list(self.env_categorical.keys())
            stats['env_categorical_cardinalities'] = {
                var_name: len(np.unique(var_values))
                for var_name, var_values in self.env_categorical.items()
            }

        return stats
