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


def load_process_data(process_name, data_dir='src/data/raw', output_columns=None):
    """
    Load data for a specific manufacturing process with automatic column mapping.

    This function automatically:
    1. Loads the correct CSV file for the process
    2. Excludes metadata columns (timestamps, IDs, etc.)
    3. Separates input and output columns based on process configuration

    Args:
        process_name (str): Name of the process ('laser', 'plasma', 'galvanic',
                           'multibond', 'microetch')
        data_dir (str): Directory containing the CSV files (default: 'src/data/raw')
        output_columns (list, optional): Override default output columns for the process.
                                        If None, uses columns from process_config.py

    Returns:
        tuple: (X, y, column_info) where:
            - X (np.ndarray): Input features
            - y (np.ndarray): Target outputs
            - column_info (dict): Dictionary with 'input_columns', 'output_columns',
                                 'metadata_columns' lists

    Example:
        >>> # Load laser process data with default output columns
        >>> X, y, info = load_process_data('laser')
        >>> print(f"Inputs: {info['input_columns']}")
        >>> print(f"Outputs: {info['output_columns']}")
        >>>
        >>> # Load plasma process data with custom output columns
        >>> X, y, info = load_process_data('plasma', output_columns=['Temperature'])

    Raises:
        ValueError: If process_name is not recognized
        FileNotFoundError: If CSV file doesn't exist
    """
    import os
    import sys

    # Import process configuration
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from configs.process_config import get_process_config, get_column_mapping, set_output_columns

    # Get process configuration
    config = get_process_config(process_name)

    # Override output columns if specified
    if output_columns is not None:
        set_output_columns(process_name, output_columns)
        config = get_process_config(process_name)

    # Build file path
    filepath = os.path.join(data_dir, config['filename'])

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"CSV file not found: {filepath}\n"
            f"Expected file: {config['filename']}\n"
            f"Make sure the data file exists in {data_dir}"
        )

    # Load CSV with process-specific settings
    df = pd.read_csv(filepath, sep=config['sep'], header=config['header'])

    print(f"\n{'='*60}")
    print(f"Loading process: {config['process_label']}")
    print(f"File: {config['filename']}")
    print(f"{'='*60}")

    # Get automatic column mapping
    all_columns = df.columns.tolist()
    input_cols, output_cols, metadata_cols = get_column_mapping(process_name, all_columns)

    # Check if output columns are defined
    if len(output_cols) == 0:
        print("\n⚠️  WARNING: No output columns defined for this process!")
        print("Please set output columns using one of these methods:")
        print(f"  1. In process_config.py: PROCESS_CONFIGS['{process_name}']['output_columns']")
        print(f"  2. When calling: load_process_data('{process_name}', output_columns=['col1', 'col2'])")
        print("\nAvailable columns (excluding metadata):")
        non_metadata = [col for col in all_columns if col not in metadata_cols]
        for col in non_metadata:
            print(f"  - {col}")
        raise ValueError(f"No output columns defined for process '{process_name}'")

    # Extract data
    X = df[input_cols].values
    y = df[output_cols].values

    # Print summary
    print(f"\nData loaded: {X.shape[0]} samples")
    print(f"\nMetadata columns (excluded): {len(metadata_cols)}")
    for col in metadata_cols:
        print(f"  ✗ {col}")

    print(f"\nInput features: {len(input_cols)}")
    for col in input_cols:
        print(f"  → {col}")

    print(f"\nOutput features: {len(output_cols)}")
    for col in output_cols:
        print(f"  ← {col}")

    print(f"{'='*60}\n")

    # Return column information for reference
    column_info = {
        'input_columns': input_cols,
        'output_columns': output_cols,
        'metadata_columns': metadata_cols
    }

    return X, y, column_info
