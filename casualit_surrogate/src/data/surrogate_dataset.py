"""
PyTorch Dataset and DataModule for CasualiT Surrogate Training.

Provides data loading for trajectory sequences with reliability labels.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict


class SurrogateDataset(Dataset):
    """
    PyTorch Dataset for trajectory-reliability pairs.

    Input X: (n_samples, n_processes, features_per_process)
    Target Y: (n_samples, 1) - reliability F
    """

    def __init__(self,
                 X: np.ndarray,
                 Y: np.ndarray,
                 normalize_inputs: bool = True,
                 input_stats: Optional[Dict] = None):
        """
        Args:
            X: Input trajectories, shape (n_samples, n_processes, features)
            Y: Target reliability, shape (n_samples, 1)
            normalize_inputs: Whether to normalize input features
            input_stats: Pre-computed normalization stats (mean, std)
                        If None and normalize_inputs=True, compute from data
        """
        self.normalize_inputs = normalize_inputs

        # Store raw data
        self.X_raw = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

        # Compute or use provided normalization stats
        if normalize_inputs:
            if input_stats is not None:
                self.input_mean = input_stats['mean']
                self.input_std = input_stats['std']
            else:
                # Compute per-feature statistics
                # Reshape to (n_samples * n_processes, features) for global stats
                X_flat = X.reshape(-1, X.shape[-1])
                self.input_mean = X_flat.mean(axis=0)
                self.input_std = X_flat.std(axis=0)
                self.input_std[self.input_std < 1e-8] = 1.0  # Avoid division by zero

            # Normalize
            self.X = (self.X_raw - self.input_mean) / self.input_std
        else:
            self.X = self.X_raw
            self.input_mean = None
            self.input_std = None

        # Convert to tensors
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.Y = torch.tensor(self.Y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]

    def get_normalization_stats(self) -> Optional[Dict]:
        """Return normalization statistics for use in inference."""
        if self.input_mean is not None:
            return {
                'mean': self.input_mean,
                'std': self.input_std,
            }
        return None

    def get_statistics(self) -> Dict:
        """Return dataset statistics."""
        return {
            'n_samples': len(self),
            'input_shape': tuple(self.X.shape[1:]),
            'n_processes': self.X.shape[1],
            'features_per_process': self.X.shape[2],
            'Y_mean': self.Y.mean().item(),
            'Y_std': self.Y.std().item(),
            'Y_min': self.Y.min().item(),
            'Y_max': self.Y.max().item(),
        }


class SurrogateDataModule:
    """
    Data module for CasualiT surrogate training.

    Handles train/val/test split and DataLoader creation.
    """

    def __init__(self,
                 X_train: np.ndarray,
                 Y_train: np.ndarray,
                 X_test: np.ndarray,
                 Y_test: np.ndarray,
                 batch_size: int = 64,
                 val_split: float = 0.2,
                 normalize_inputs: bool = True,
                 num_workers: int = 0,
                 seed: int = 42):
        """
        Args:
            X_train: Training trajectories
            Y_train: Training reliability labels
            X_test: Test trajectories
            Y_test: Test reliability labels
            batch_size: Batch size for DataLoaders
            val_split: Fraction of training data to use for validation
            normalize_inputs: Whether to normalize inputs
            num_workers: Number of DataLoader workers
            seed: Random seed for splitting
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize_inputs = normalize_inputs

        # Split training data into train/val
        n_train = len(X_train)
        n_val = int(n_train * val_split)

        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_train)

        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        X_train_split = X_train[train_indices]
        Y_train_split = Y_train[train_indices]
        X_val_split = X_train[val_indices]
        Y_val_split = Y_train[val_indices]

        # Create training dataset (computes normalization stats)
        self.train_dataset = SurrogateDataset(
            X_train_split, Y_train_split,
            normalize_inputs=normalize_inputs,
        )

        # Get normalization stats from training data
        self.normalization_stats = self.train_dataset.get_normalization_stats()

        # Create val and test datasets with same normalization
        self.val_dataset = SurrogateDataset(
            X_val_split, Y_val_split,
            normalize_inputs=normalize_inputs,
            input_stats=self.normalization_stats,
        )

        self.test_dataset = SurrogateDataset(
            X_test, Y_test,
            normalize_inputs=normalize_inputs,
            input_stats=self.normalization_stats,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_input_dim(self) -> Tuple[int, int]:
        """Return (n_processes, features_per_process)."""
        return self.train_dataset.X.shape[1], self.train_dataset.X.shape[2]

    def get_statistics(self) -> Dict:
        """Return statistics for all splits."""
        return {
            'train': self.train_dataset.get_statistics(),
            'val': self.val_dataset.get_statistics(),
            'test': self.test_dataset.get_statistics(),
        }
