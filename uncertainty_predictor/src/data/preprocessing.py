"""
Utilities for machinery data preprocessing
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle


class DataPreprocessor:
    """
    Class for preprocessing machinery data.

    Features:
    - Normalization/Standardization
    - Missing values handling
    - Train/validation/test split
    - Scaler saving/loading for inference

    Args:
        scaling_method (str): 'standard' or 'minmax' (default: 'standard')
    """

    def __init__(self, scaling_method='standard'):
        self.scaling_method = scaling_method
        self.input_scaler = None
        self.output_scaler = None

        if scaling_method == 'standard':
            self.input_scaler = StandardScaler()
            self.output_scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.input_scaler = MinMaxScaler()
            self.output_scaler = MinMaxScaler()
        else:
            raise ValueError("scaling_method must be 'standard' or 'minmax'")

    def fit_transform(self, X, y):
        """
        Fit scalers on data and transform.

        Args:
            X (np.ndarray or pd.DataFrame): Input features
            y (np.ndarray or pd.DataFrame): Target outputs

        Returns:
            tuple: (X_scaled, y_scaled)
        """
        X = self._to_numpy(X)
        y = self._to_numpy(y)

        # Handle missing values
        X = self._handle_missing_values(X)
        y = self._handle_missing_values(y)

        # Fit and transform
        X_scaled = self.input_scaler.fit_transform(X)
        y_scaled = self.output_scaler.fit_transform(y)

        return X_scaled, y_scaled

    def transform(self, X, y=None):
        """
        Transform data using fitted scalers.

        Args:
            X (np.ndarray or pd.DataFrame): Input features
            y (np.ndarray or pd.DataFrame, optional): Target outputs

        Returns:
            np.ndarray or tuple: X_scaled or (X_scaled, y_scaled)
        """
        if self.input_scaler is None:
            raise ValueError("Scaler not yet fitted. Use fit_transform first.")

        X = self._to_numpy(X)
        X = self._handle_missing_values(X)
        X_scaled = self.input_scaler.transform(X)

        if y is not None:
            y = self._to_numpy(y)
            y = self._handle_missing_values(y)
            y_scaled = self.output_scaler.transform(y)
            return X_scaled, y_scaled

        return X_scaled

    def inverse_transform_output(self, y_scaled):
        """
        Convert predicted outputs back to original scale.

        Args:
            y_scaled (np.ndarray): Scaled outputs

        Returns:
            np.ndarray: Outputs in original scale
        """
        if self.output_scaler is None:
            raise ValueError("Output scaler not fitted.")

        return self.output_scaler.inverse_transform(y_scaled)

    def split_data(self, X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
        """
        Split data into train, validation and test sets.

        Args:
            X (np.ndarray): Features
            y (np.ndarray): Targets
            train_size (float): Training set proportion
            val_size (float): Validation set proportion
            test_size (float): Test set proportion
            random_state (int): Seed for reproducibility

        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
            "Proportions must sum to 1.0"

        # First split: train vs (val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(1 - train_size), random_state=random_state
        )

        # Second split: val vs test
        val_ratio = val_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio), random_state=random_state
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_scalers(self, filepath):
        """Save scalers for future use"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'input_scaler': self.input_scaler,
                'output_scaler': self.output_scaler,
                'scaling_method': self.scaling_method
            }, f)
        print(f"Scalers saved to: {filepath}")

    def load_scalers(self, filepath):
        """Load previously saved scalers"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.input_scaler = data['input_scaler']
            self.output_scaler = data['output_scaler']
            self.scaling_method = data['scaling_method']
        print(f"Scalers loaded from: {filepath}")

    @staticmethod
    def _to_numpy(data):
        """Convert DataFrame or lists to numpy array"""
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.values
        return np.array(data)

    @staticmethod
    def _handle_missing_values(data):
        """Handle missing values by replacing with mean"""
        if np.any(np.isnan(data)):
            print("Warning: Missing values found. Replaced with mean.")
            col_mean = np.nanmean(data, axis=0)
            inds = np.where(np.isnan(data))
            data[inds] = np.take(col_mean, inds[1])
        return data


def load_csv_data(filepath, input_columns, output_columns):
    """
    Load data from CSV file.

    Args:
        filepath (str): Path to CSV file
        input_columns (list): Names of input columns
        output_columns (list): Names of output columns

    Returns:
        tuple: (X, y) as numpy arrays
    """
    df = pd.read_csv(filepath)

    X = df[input_columns].values
    y = df[output_columns].values

    print(f"Data loaded: {X.shape[0]} samples")
    print(f"Input features: {X.shape[1]}")
    print(f"Output features: {y.shape[1]}")

    return X, y


def generate_scm_data(n_samples=5000, seed=42, dataset_type='one_to_one_ct'):
    """
    Generate synthetic data using Structural Causal Model (SCM).

    Args:
        n_samples (int): Number of samples to generate
        seed (int): Random seed for reproducibility
        dataset_type (str): Type of SCM dataset to use

    Returns:
        tuple: (X, y, input_columns, output_columns) as numpy arrays and column names
    """
    import sys
    from pathlib import Path

    # Add scm_ds to path
    scm_path = Path(__file__).parent.parent.parent / 'scm_ds'
    if str(scm_path) not in sys.path:
        sys.path.insert(0, str(scm_path))

    # Import all available datasets
    from datasets import ds_scm_1_to_1_ct

    # Select dataset based on type
    if dataset_type == 'one_to_one_ct':
        scm_dataset = ds_scm_1_to_1_ct

    # Add your custom datasets here:
    # elif dataset_type == 'simple_3to1':
    #     from datasets import ds_simple_3to1
    #     scm_dataset = ds_simple_3to1
    # elif dataset_type == 'nonlinear':
    #     from datasets import ds_nonlinear
    #     scm_dataset = ds_nonlinear

    else:
        raise ValueError(f"Unknown SCM dataset type: {dataset_type}")

    # Generate samples
    print(f"Generating {n_samples} synthetic samples using SCM...")
    df = scm_dataset.sample(n=n_samples, seed=seed)

    # Extract input and output columns
    input_columns = scm_dataset.input_labels
    output_columns = scm_dataset.target_labels

    X = df[input_columns].values
    y = df[output_columns].values

    print(f"SCM data generated: {X.shape[0]} samples")
    print(f"Input features: {X.shape[1]} - {input_columns}")
    print(f"Output features: {y.shape[1]} - {output_columns}")

    return X, y, input_columns, output_columns
