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


def generate_scm_data(n_samples=5000, seed=42, dataset_type='one_to_one_ct', save_graph_to=None):
    """
    Generate synthetic data using Structural Causal Model (SCM).

    Args:
        n_samples (int): Number of samples to generate
        seed (int): Random seed for reproducibility
        dataset_type (str): Type of SCM dataset to use. Available types:
            - 'one_to_one_ct': Simple one-to-one with cross-talk
            - 'laser': Laser drilling optical power (L-I-T model with physical noise)
            - 'plasma': Plasma cleaning residue removal (with micro-arcing jumps)
            - 'galvanic': Galvanic copper deposition (with spatial variation and ripple)
            - 'microetch': Micro-etching Cu removal (Arrhenius kinetics with Student-t noise)
        save_graph_to (str, optional): Directory path to save SCM graph visualization

    Returns:
        tuple: (X, y, input_columns, output_columns) as numpy arrays and column names
    """
    import sys
    from pathlib import Path

    # Add scm_ds to path
    scm_path = Path(__file__).parent.parent.parent / 'scm_ds'
    if str(scm_path) not in sys.path:
        sys.path.insert(0, str(scm_path))

    from scm_ds.datasets import (
        ds_scm_1_to_1_ct,
        ds_scm_laser,
        ds_scm_plasma,
        ds_scm_galvanic,
        ds_scm_microetch
    )

    # Select dataset based on type
    if dataset_type == 'one_to_one_ct':
        scm_dataset = ds_scm_1_to_1_ct
    elif dataset_type == 'laser':
        scm_dataset = ds_scm_laser
    elif dataset_type == 'plasma':
        scm_dataset = ds_scm_plasma
    elif dataset_type == 'galvanic':
        scm_dataset = ds_scm_galvanic
    elif dataset_type == 'microetch':
        scm_dataset = ds_scm_microetch
    else:
        raise ValueError(f"Unknown SCM dataset type: {dataset_type}. "
                         f"Available types: 'one_to_one_ct', 'laser', 'plasma', 'galvanic', 'microetch'")

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

    # Save SCM graph visualization if requested
    if save_graph_to is not None:
        try:
            from pathlib import Path
            from os.path import join
            save_dir = Path(save_graph_to)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Use matplotlib-based visualization (no external dependencies needed)
            scm_dataset.scm.save_graph_matplotlib(join(save_dir, 'scm_graph'))
            print(f"SCM graph saved to: {save_dir}/scm_graph.png")
        except Exception as e:
            print(f"Warning: Could not save SCM graph visualization.")
            print(f"  Error: {e}")
            print(f"  Continuing without graph...")

    return X, y, input_columns, output_columns


# =============================================================================
# CONDITIONAL EMBEDDING DATA UTILITIES
# =============================================================================


def generate_conditional_scm_data(
    process_selection='all',
    n_samples=2000,
    add_env_vars=True,
    seed=42
):
    """
    Generate SCM data for conditional embedding training.

    Args:
        process_selection (str): Process to generate data for:
            - 'all': Unified dataset with all 4 processes (Laser, Plasma, Galvanic, Microetch)
            - 'laser', 'plasma', 'galvanic', 'microetch': Single process
        n_samples (int): Number of samples to generate
            - For 'all': samples per process (total will be 4*n_samples)
            - For single process: total samples
        add_env_vars (bool): If True, add environment variables and process_id
        seed (int): Random seed for reproducibility

    Returns:
        pd.DataFrame: Generated data with all features including conditioning variables
    """
    import sys
    from pathlib import Path

    # Add scm_ds to path
    scm_path = Path(__file__).parent.parent.parent / 'scm_ds'
    if str(scm_path) not in sys.path:
        sys.path.insert(0, str(scm_path))

    if process_selection == 'all':
        from scm_ds.datasets import generate_unified_dataset
        print(f"Generating unified multi-process dataset...")
        df = generate_unified_dataset(
            n_samples_per_process=n_samples,
            add_env_vars=add_env_vars,
            seed=seed,
            mode='balanced'
        )
    elif process_selection in ['laser', 'plasma', 'galvanic', 'microetch']:
        from scm_ds.datasets import generate_single_process_dataset
        print(f"Generating single-process dataset: {process_selection}")
        df = generate_single_process_dataset(
            process_name=process_selection,
            n_samples=n_samples,
            add_env_vars=add_env_vars,
            seed=seed
        )
        print(f"  Generated {len(df)} samples for {process_selection}")
    else:
        raise ValueError(
            f"Invalid process_selection: {process_selection}. "
            f"Must be 'all' or one of ['laser', 'plasma', 'galvanic', 'microetch']"
        )

    return df


def prepare_conditional_tensors(
    df,
    input_columns,
    output_columns,
    conditioning_columns=None
):
    """
    Extract features and conditioning variables from DataFrame for conditional training.

    Args:
        df (pd.DataFrame): Input DataFrame with all features
        input_columns (list): Names of process-specific input feature columns
        output_columns (list): Names of process-specific output feature columns
        conditioning_columns (dict, optional): Dict specifying conditioning columns:
            {
                'process_id': 'process_id',
                'timestamp': 'timestamp',
                'env_continuous': ['ambient_temp', 'humidity'],
                'env_categorical': ['batch_id', 'operator_id', 'shift']
            }

    Returns:
        dict: {
            'X': np.ndarray (n_samples, n_input_features),
            'y': np.ndarray (n_samples, n_output_features),
            'process_id': np.ndarray (n_samples,) or None,
            'timestamp': np.ndarray (n_samples,) or None,
            'env_continuous': dict {var_name: np.ndarray (n_samples,)} or None,
            'env_categorical': dict {var_name: np.ndarray (n_samples,)} or None,
            'env_masks': dict {var_name: np.ndarray (n_samples,)} or None,
        }
    """
    result = {}

    # Extract X and y (process-specific features)
    # Handle case where different processes have different input columns
    # Use only available columns
    available_input_cols = [col for col in input_columns if col in df.columns]
    available_output_cols = [col for col in output_columns if col in df.columns]

    # For multi-process datasets, we need to handle different input/output columns per process
    # Strategy: use all non-conditioning columns as features
    if conditioning_columns is not None:
        # Get all conditioning column names
        cond_cols = set()
        if 'process_id' in conditioning_columns:
            cond_cols.add(conditioning_columns['process_id'])
        if 'timestamp' in conditioning_columns:
            cond_cols.add(conditioning_columns['timestamp'])
        if 'env_continuous' in conditioning_columns:
            cond_cols.update(conditioning_columns['env_continuous'])
        if 'env_categorical' in conditioning_columns:
            cond_cols.update(conditioning_columns['env_categorical'])

        # All numeric columns that are not conditioning variables are features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in cond_cols]

        # Separate input and output based on available columns
        if available_output_cols:
            output_cols_final = available_output_cols
            input_cols_final = [col for col in feature_cols if col not in output_cols_final]
        else:
            # If no output columns available, use last column as output
            input_cols_final = feature_cols[:-1]
            output_cols_final = [feature_cols[-1]]
    else:
        input_cols_final = available_input_cols
        output_cols_final = available_output_cols

    result['X'] = df[input_cols_final].values
    result['y'] = df[output_cols_final].values

    # Extract conditioning variables if provided
    if conditioning_columns is not None:
        # Process ID
        if 'process_id' in conditioning_columns:
            pid_col = conditioning_columns['process_id']
            if pid_col in df.columns:
                result['process_id'] = df[pid_col].values.astype(np.int64)
            else:
                result['process_id'] = None
        else:
            result['process_id'] = None

        # Timestamp
        if 'timestamp' in conditioning_columns:
            ts_col = conditioning_columns['timestamp']
            if ts_col in df.columns:
                # Normalize timestamps to [0, 1] range for better training
                timestamps = df[ts_col].values.astype(np.float32)
                ts_min = timestamps.min()
                ts_max = timestamps.max()
                if ts_max > ts_min:
                    timestamps = (timestamps - ts_min) / (ts_max - ts_min)
                result['timestamp'] = timestamps
            else:
                result['timestamp'] = None
        else:
            result['timestamp'] = None

        # Continuous environment variables
        if 'env_continuous' in conditioning_columns:
            env_cont = {}
            env_masks = {}
            for var_name in conditioning_columns['env_continuous']:
                if var_name in df.columns:
                    values = df[var_name].values.astype(np.float32)
                    # Create mask: 1.0 = present, 0.0 = missing
                    mask = ~np.isnan(values)
                    # Replace NaN with 0.0 (will be masked out in model)
                    values = np.nan_to_num(values, nan=0.0)
                    env_cont[var_name] = values
                    env_masks[var_name] = mask.astype(np.float32)
            result['env_continuous'] = env_cont if env_cont else None
            result['env_masks'] = env_masks if env_masks else None
        else:
            result['env_continuous'] = None
            result['env_masks'] = None

        # Categorical environment variables
        if 'env_categorical' in conditioning_columns:
            env_cat = {}
            for var_name in conditioning_columns['env_categorical']:
                if var_name in df.columns:
                    env_cat[var_name] = df[var_name].values.astype(np.int64)
            result['env_categorical'] = env_cat if env_cat else None
        else:
            result['env_categorical'] = None
    else:
        result['process_id'] = None
        result['timestamp'] = None
        result['env_continuous'] = None
        result['env_categorical'] = None
        result['env_masks'] = None

    return result


def create_conditional_collate_fn(conditioning_enabled=True):
    """
    Create a collate function for DataLoader that handles conditional data.

    Args:
        conditioning_enabled (bool): If True, return dict format with conditioning vars.
                                     If False, return standard (X, y) tuple.

    Returns:
        callable: Collate function for DataLoader
    """
    import torch

    if not conditioning_enabled:
        # Standard collate for non-conditional training
        def standard_collate(batch):
            # batch is a list of (X, y) tuples
            X_batch = torch.stack([item['X'] for item in batch])
            y_batch = torch.stack([item['y'] for item in batch])
            return X_batch, y_batch
        return standard_collate

    else:
        # Conditional collate function
        def conditional_collate(batch):
            # batch is a list of dicts with keys: X, y, process_id, etc.
            result = {}

            # Stack X and y
            result['X'] = torch.stack([item['X'] for item in batch])
            result['y'] = torch.stack([item['y'] for item in batch])

            # Process ID
            if 'process_id' in batch[0] and batch[0]['process_id'] is not None:
                result['process_id'] = torch.stack([item['process_id'] for item in batch])
            else:
                result['process_id'] = None

            # Timestamp
            if 'timestamp' in batch[0] and batch[0]['timestamp'] is not None:
                result['timestamp'] = torch.stack([item['timestamp'] for item in batch])
            else:
                result['timestamp'] = None

            # Continuous environment variables
            if 'env_continuous' in batch[0] and batch[0]['env_continuous'] is not None:
                env_cont = {}
                for var_name in batch[0]['env_continuous'].keys():
                    env_cont[var_name] = torch.stack([item['env_continuous'][var_name] for item in batch])
                result['env_continuous'] = env_cont
            else:
                result['env_continuous'] = None

            # Categorical environment variables
            if 'env_categorical' in batch[0] and batch[0]['env_categorical'] is not None:
                env_cat = {}
                for var_name in batch[0]['env_categorical'].keys():
                    env_cat[var_name] = torch.stack([item['env_categorical'][var_name] for item in batch])
                result['env_categorical'] = env_cat
            else:
                result['env_categorical'] = None

            # Environment masks
            if 'env_masks' in batch[0] and batch[0]['env_masks'] is not None:
                env_masks = {}
                for var_name in batch[0]['env_masks'].keys():
                    env_masks[var_name] = torch.stack([item['env_masks'][var_name] for item in batch])
                result['env_masks'] = env_masks
            else:
                result['env_masks'] = None

            return result

        return conditional_collate
