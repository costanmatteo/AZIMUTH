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
    PyTorch Dataset for conditional multi-process machinery data.

    This dataset handles input/output data along with conditioning variables
    (process_id, environment variables, timestamps) for conditional embedding training.

    Args:
        data_dict (dict): Dictionary containing:
            - 'X': np.ndarray (n_samples, n_features) - input features
            - 'y': np.ndarray (n_samples, n_outputs) - target outputs
            - 'process_id': np.ndarray (n_samples,) - process IDs (optional)
            - 'timestamp': np.ndarray (n_samples,) - timestamps (optional)
            - 'env_continuous': dict of np.ndarray - continuous env vars (optional)
            - 'env_categorical': dict of np.ndarray - categorical env vars (optional)
            - 'env_masks': dict of np.ndarray - missing value masks (optional)

    Returns:
        dict: Sample dictionary with X, y, and all conditioning variables as tensors

    Example:
        >>> data_dict = {
        ...     'X': np.random.randn(1000, 10),
        ...     'y': np.random.randn(1000, 5),
        ...     'process_id': np.random.randint(0, 4, 1000),
        ...     'timestamp': np.random.rand(1000),
        ...     'env_continuous': {'temp': np.random.randn(1000)},
        ...     'env_categorical': {'batch': np.random.randint(0, 10, 1000)},
        ...     'env_masks': {'temp': np.ones(1000)}
        ... }
        >>> dataset = ConditionalMachineryDataset(data_dict)
        >>> sample = dataset[0]
        >>> print(sample.keys())  # dict_keys(['X', 'y', 'process_id', ...])
    """

    def __init__(self, data_dict, transform=None):
        """
        Initialize the conditional dataset.

        Args:
            data_dict: Dictionary with all data (X, y, conditioning vars)
            transform: Optional transformations
        """
        # Core features (required)
        self.X = torch.FloatTensor(data_dict['X'])
        self.y = torch.FloatTensor(data_dict['y'])

        # Conditioning variables (optional)
        self.process_id = None
        if data_dict.get('process_id') is not None:
            self.process_id = torch.LongTensor(data_dict['process_id'])

        self.timestamp = None
        if data_dict.get('timestamp') is not None:
            self.timestamp = torch.FloatTensor(data_dict['timestamp'])

        # Continuous environment variables
        self.env_continuous = None
        if data_dict.get('env_continuous') is not None:
            self.env_continuous = {
                var_name: torch.FloatTensor(values)
                for var_name, values in data_dict['env_continuous'].items()
            }

        # Categorical environment variables
        self.env_categorical = None
        if data_dict.get('env_categorical') is not None:
            self.env_categorical = {
                var_name: torch.LongTensor(values)
                for var_name, values in data_dict['env_categorical'].items()
            }

        # Environment masks (for missing values)
        self.env_masks = None
        if data_dict.get('env_masks') is not None:
            self.env_masks = {
                var_name: torch.FloatTensor(mask)
                for var_name, mask in data_dict['env_masks'].items()
            }

        self.transform = transform

        # Verify dimensions
        n_samples = len(self.X)
        assert len(self.y) == n_samples, \
            f"Mismatch: {len(self.X)} input samples vs {len(self.y)} output samples"

        if self.process_id is not None:
            assert len(self.process_id) == n_samples, \
                f"Mismatch: process_id length {len(self.process_id)} vs {n_samples} samples"

        if self.timestamp is not None:
            assert len(self.timestamp) == n_samples, \
                f"Mismatch: timestamp length {len(self.timestamp)} vs {n_samples} samples"

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.X)

    def __getitem__(self, idx):
        """
        Return a sample from the dataset.

        Args:
            idx (int): Sample index

        Returns:
            dict: Dictionary with all features and conditioning variables
        """
        sample = {
            'X': self.X[idx],
            'y': self.y[idx],
        }

        # Add conditioning variables if present
        if self.process_id is not None:
            sample['process_id'] = self.process_id[idx]

        if self.timestamp is not None:
            sample['timestamp'] = self.timestamp[idx]

        if self.env_continuous is not None:
            sample['env_continuous'] = {
                var_name: values[idx]
                for var_name, values in self.env_continuous.items()
            }

        if self.env_categorical is not None:
            sample['env_categorical'] = {
                var_name: values[idx]
                for var_name, values in self.env_categorical.items()
            }

        if self.env_masks is not None:
            sample['env_masks'] = {
                var_name: mask[idx]
                for var_name, mask in self.env_masks.items()
            }

        if self.transform:
            sample['X'] = self.transform(sample['X'])

        return sample

    def get_input_dim(self):
        """Return the input dimension"""
        return self.X.shape[1]

    def get_output_dim(self):
        """Return the output dimension"""
        return self.y.shape[1] if len(self.y.shape) > 1 else 1

    def get_num_processes(self):
        """Return the number of unique processes (if process_id exists)"""
        if self.process_id is not None:
            return int(self.process_id.max().item()) + 1
        return None

    def get_statistics(self):
        """
        Return statistics about the data.

        Returns:
            dict: Dictionary with input/output statistics and conditioning info
        """
        stats = {
            'n_samples': len(self),
            'input_dim': self.get_input_dim(),
            'output_dim': self.get_output_dim(),
            'input_mean': self.X.mean(dim=0),
            'input_std': self.X.std(dim=0),
            'output_mean': self.y.mean(dim=0),
            'output_std': self.y.std(dim=0),
            'has_process_id': self.process_id is not None,
            'has_timestamp': self.timestamp is not None,
            'has_env_continuous': self.env_continuous is not None,
            'has_env_categorical': self.env_categorical is not None,
        }

        # Add process-specific info if available
        if self.process_id is not None:
            stats['num_processes'] = self.get_num_processes()
            # Count samples per process
            process_counts = {}
            for pid in range(stats['num_processes']):
                count = (self.process_id == pid).sum().item()
                process_counts[f'process_{pid}_count'] = count
            stats.update(process_counts)

        return stats
