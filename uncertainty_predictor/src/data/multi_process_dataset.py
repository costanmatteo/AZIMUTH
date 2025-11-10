"""
Multi-Process Dataset for Conditional Training

This module provides a PyTorch Dataset for training on multiple manufacturing
processes simultaneously, with support for conditioning on process ID and
environmental features.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class MultiProcessDataset(Dataset):
    """
    PyTorch Dataset for multi-process training with conditioning features.

    This dataset handles data from multiple manufacturing processes, each with
    different input/output features, and provides conditioning information
    (process_id, environmental features, timestamps) for each sample.

    Args:
        combined_df: DataFrame containing all processes with columns:
            - process_id: Process identifier (0, 1, 2, 3)
            - Input features (process-specific)
            - Output features (process-specific)
            - temperature, humidity, load_factor (may contain NaN)
            - batch_id, operator_id, shift (categorical)
            - timestamp (float)
        metadata: Dictionary with process information:
            - process_names: {0: 'laser', 1: 'plasma', ...}
            - input_columns: {0: ['PowerTarget', 'AmbientTemp'], ...}
            - output_columns: {0: ['ActualPower'], ...}
            - env_continuous: ['temperature', 'humidity', 'load_factor']
            - env_categorical: ['batch_id', 'operator_id', 'shift']
            - temporal: ['timestamp']
        input_scaler: Optional scaler for input features (fitted on training data)
        output_scaler: Optional scaler for output features (fitted on training data)
    """

    def __init__(
        self,
        combined_df: pd.DataFrame,
        metadata: Dict,
        input_scaler=None,
        output_scaler=None,
    ):
        super().__init__()

        self.df = combined_df.reset_index(drop=True)
        self.metadata = metadata
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler

        # Extract metadata
        self.process_names = metadata['process_names']
        self.input_columns_by_process = metadata['input_columns']
        self.output_columns_by_process = metadata['output_columns']
        self.env_continuous = metadata.get('env_continuous', [])
        self.env_categorical = metadata.get('env_categorical', [])
        self.temporal = metadata.get('temporal', [])

        # Determine maximum input/output sizes across all processes
        self.max_input_size = max(len(cols) for cols in self.input_columns_by_process.values())
        self.max_output_size = max(len(cols) for cols in self.output_columns_by_process.values())

        # Prepare data arrays for faster access
        self._prepare_data()

    def _prepare_data(self):
        """
        Pre-process and cache data for efficient access.
        """
        # Extract process IDs
        self.process_ids = self.df['process_id'].values

        # Pre-allocate arrays for inputs and outputs (padded to max size)
        n_samples = len(self.df)
        self.inputs = np.zeros((n_samples, self.max_input_size), dtype=np.float32)
        self.outputs = np.zeros((n_samples, self.max_output_size), dtype=np.float32)

        # Fill inputs and outputs per process
        for process_id in self.process_names.keys():
            # Get mask for this process
            mask = self.process_ids == process_id

            if not mask.any():
                continue

            # Get input and output columns for this process
            input_cols = self.input_columns_by_process[process_id]
            output_cols = self.output_columns_by_process[process_id]

            # Extract data
            input_data = self.df.loc[mask, input_cols].values
            output_data = self.df.loc[mask, output_cols].values

            # Store in pre-allocated arrays (remaining entries stay 0)
            self.inputs[mask, :len(input_cols)] = input_data
            self.outputs[mask, :len(output_cols)] = output_data

        # Apply scaling if provided
        if self.input_scaler is not None:
            self.inputs = self.input_scaler.transform(self.inputs)

        if self.output_scaler is not None:
            self.outputs = self.output_scaler.transform(self.outputs)

        # Extract environmental continuous features
        if self.env_continuous:
            self.env_cont_data = self.df[self.env_continuous].values.astype(np.float32)
        else:
            self.env_cont_data = None

        # Extract environmental categorical features
        if self.env_categorical:
            self.env_cat_data = {}
            for col in self.env_categorical:
                self.env_cat_data[col] = self.df[col].values.astype(np.int64)
        else:
            self.env_cat_data = None

        # Extract timestamps
        if self.temporal:
            self.timestamps = self.df[self.temporal[0]].values.astype(np.float32)
        else:
            self.timestamps = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample with all conditioning information.

        Returns:
            dict with keys:
                - 'x': Input features (torch.Tensor, shape: (max_input_size,))
                - 'y': Output features (torch.Tensor, shape: (max_output_size,))
                - 'process_id': Process ID (torch.Tensor, scalar)
                - 'env_cont': Continuous env features (torch.Tensor, shape: (n_env_cont,))
                - 'env_cont_mask': Mask for valid values (torch.Tensor, shape: (n_env_cont,))
                - 'env_cat': Dict of categorical features (each is torch.Tensor, scalar)
                - 'timestamp': Timestamp (torch.Tensor, scalar)
        """
        sample = {}

        # Input and output
        sample['x'] = torch.from_numpy(self.inputs[idx])
        sample['y'] = torch.from_numpy(self.outputs[idx])

        # Process ID
        sample['process_id'] = torch.tensor(self.process_ids[idx], dtype=torch.long)

        # Environmental continuous features
        if self.env_cont_data is not None:
            env_cont = self.env_cont_data[idx]
            sample['env_cont'] = torch.from_numpy(env_cont)

            # Create mask: True where not NaN
            env_cont_mask = ~np.isnan(env_cont)
            sample['env_cont_mask'] = torch.from_numpy(env_cont_mask)
        else:
            sample['env_cont'] = None
            sample['env_cont_mask'] = None

        # Environmental categorical features
        if self.env_cat_data is not None:
            env_cat = {}
            for col, data in self.env_cat_data.items():
                env_cat[col] = torch.tensor(data[idx], dtype=torch.long)
            sample['env_cat'] = env_cat
        else:
            sample['env_cat'] = None

        # Timestamp
        if self.timestamps is not None:
            sample['timestamp'] = torch.tensor(self.timestamps[idx], dtype=torch.float32)
        else:
            sample['timestamp'] = None

        return sample

    def get_process_mask(self, process_id: int) -> np.ndarray:
        """
        Get boolean mask for samples belonging to a specific process.

        Args:
            process_id: Process identifier

        Returns:
            Boolean array of shape (n_samples,)
        """
        return self.process_ids == process_id

    def get_input_dim(self) -> int:
        """Get maximum input dimension across all processes"""
        return self.max_input_size

    def get_output_dim(self) -> int:
        """Get maximum output dimension across all processes"""
        return self.max_output_size

    def get_process_distribution(self) -> Dict[int, int]:
        """
        Get count of samples per process.

        Returns:
            Dictionary {process_id: count}
        """
        unique, counts = np.unique(self.process_ids, return_counts=True)
        return {int(pid): int(count) for pid, count in zip(unique, counts)}


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for DataLoader.

    Handles batching of samples with optional conditioning features.

    Args:
        batch: List of samples from MultiProcessDataset.__getitem__()

    Returns:
        Batched dictionary with stacked tensors
    """
    batched = {}

    # Stack inputs and outputs
    batched['x'] = torch.stack([sample['x'] for sample in batch])
    batched['y'] = torch.stack([sample['y'] for sample in batch])

    # Stack process IDs
    batched['process_id'] = torch.stack([sample['process_id'] for sample in batch])

    # Stack environmental continuous features
    if batch[0]['env_cont'] is not None:
        batched['env_cont'] = torch.stack([sample['env_cont'] for sample in batch])
        batched['env_cont_mask'] = torch.stack([sample['env_cont_mask'] for sample in batch])
    else:
        batched['env_cont'] = None
        batched['env_cont_mask'] = None

    # Stack environmental categorical features
    if batch[0]['env_cat'] is not None:
        env_cat_batched = {}
        for key in batch[0]['env_cat'].keys():
            env_cat_batched[key] = torch.stack([sample['env_cat'][key] for sample in batch])
        batched['env_cat'] = env_cat_batched
    else:
        batched['env_cat'] = None

    # Stack timestamps
    if batch[0]['timestamp'] is not None:
        batched['timestamp'] = torch.stack([sample['timestamp'] for sample in batch])
    else:
        batched['timestamp'] = None

    return batched


def split_multi_process_dataset(
    combined_df: pd.DataFrame,
    metadata: Dict,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify_by_process: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split combined dataset into train/val/test while preserving process distribution.

    Args:
        combined_df: Combined DataFrame with all processes
        metadata: Metadata dictionary
        train_size: Fraction for training (default: 0.7)
        val_size: Fraction for validation (default: 0.15)
        test_size: Fraction for testing (default: 0.15)
        random_state: Random seed for reproducibility
        stratify_by_process: Whether to stratify split by process_id (default: True)

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    from sklearn.model_selection import train_test_split

    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    # Split into train and temp (val + test)
    if stratify_by_process:
        stratify = combined_df['process_id']
    else:
        stratify = None

    train_df, temp_df = train_test_split(
        combined_df,
        train_size=train_size,
        random_state=random_state,
        stratify=stratify,
    )

    # Split temp into val and test
    val_ratio = val_size / (val_size + test_size)

    if stratify_by_process:
        stratify_temp = temp_df['process_id']
    else:
        stratify_temp = None

    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio,
        random_state=random_state + 1,
        stratify=stratify_temp,
    )

    return train_df, val_df, test_df
