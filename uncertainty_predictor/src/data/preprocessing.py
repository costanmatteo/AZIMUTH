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


def generate_conditional_scm_data(
    process_selection='all',
    n_samples=5000,
    add_env_vars=True,
    seed=42
):
    """
    Generate SCM data with process_id and optional environment variables for conditional training.

    This function supports both:
    - Single process generation (backward compatible): process_selection='laser', 'plasma', etc.
    - Unified multi-process generation: process_selection='all'

    Args:
        process_selection (str): Process to generate. Options:
            - 'all': Unified dataset with all 4 processes (n_samples per process)
            - 'laser': Only laser drilling
            - 'plasma': Only plasma cleaning
            - 'galvanic': Only galvanic copper deposition
            - 'microetch': Only micro-etching
        n_samples (int): Number of samples. If process_selection='all', this is per process.
        add_env_vars (bool): Whether to add environment variables (temp, humidity, batch, operator, shift, timestamp)
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (df, input_columns, output_columns, conditioning_columns)
            - df: DataFrame with all data
            - input_columns: List of physical input features (PowerTarget, RF_Power, etc.)
            - output_columns: List of target outputs (ActualPower, RemovalRate, etc.)
            - conditioning_columns: Dict with conditioning column groups:
                - 'process_id': 'process_id'
                - 'env_continuous': list of continuous env vars
                - 'env_categorical': list of categorical env vars
                - 'timestamp': 'timestamp' or None
    """
    import sys
    from pathlib import Path

    # Add scm_ds to path
    scm_path = Path(__file__).parent.parent.parent / 'scm_ds'
    if str(scm_path) not in sys.path:
        sys.path.insert(0, str(scm_path))

    from scm_ds.datasets import (
        generate_single_process_dataset,
        generate_unified_dataset,
        get_process_info
    )

    # Generate dataset based on process selection
    if process_selection == 'all':
        print(f"Generating unified multi-process dataset: {n_samples} samples per process")
        df = generate_unified_dataset(
            n_samples_per_process=n_samples,
            add_env_vars=add_env_vars,
            seed=seed,
            mode='flat'
        )
    else:
        if process_selection not in ['laser', 'plasma', 'galvanic', 'microetch']:
            raise ValueError(
                f"Invalid process_selection '{process_selection}'. "
                f"Must be 'all', 'laser', 'plasma', 'galvanic', or 'microetch'"
            )
        print(f"Generating single process dataset: {process_selection} ({n_samples} samples)")
        df = generate_single_process_dataset(
            process_name=process_selection,
            n_samples=n_samples,
            add_env_vars=add_env_vars,
            seed=seed,
            mode='flat'
        )

    # Identify input and output columns
    process_info = get_process_info()

    if process_selection == 'all':
        # For unified dataset, we need to collect all possible input/output columns
        all_input_cols = set()
        all_output_cols = set()
        for proc_name, info in process_info.items():
            all_input_cols.update(info['inputs'])
            all_output_cols.update(info['outputs'])

        # Filter to only columns that exist in df
        input_columns = [col for col in all_input_cols if col in df.columns]
        output_columns = [col for col in all_output_cols if col in df.columns]
    else:
        # For single process, use the specific input/output columns
        info = process_info[process_selection]
        input_columns = info['inputs']
        output_columns = info['outputs']

    # Identify conditioning columns
    conditioning_columns = {
        'process_id': 'process_id' if 'process_id' in df.columns else None,
        'env_continuous': [],
        'env_categorical': [],
        'timestamp': None
    }

    if add_env_vars:
        # Continuous environment variables
        for col in ['ambient_temp', 'humidity']:
            if col in df.columns:
                conditioning_columns['env_continuous'].append(col)

        # Categorical environment variables
        for col in ['batch_id', 'operator_id', 'shift']:
            if col in df.columns:
                conditioning_columns['env_categorical'].append(col)

        # Timestamp
        if 'timestamp' in df.columns:
            conditioning_columns['timestamp'] = 'timestamp'

    print(f"Dataset generated: {df.shape[0]} samples")
    print(f"Input features ({len(input_columns)}): {input_columns}")
    print(f"Output features ({len(output_columns)}): {output_columns}")
    print(f"Conditioning features:")
    print(f"  - process_id: {conditioning_columns['process_id']}")
    print(f"  - env_continuous ({len(conditioning_columns['env_continuous'])}): {conditioning_columns['env_continuous']}")
    print(f"  - env_categorical ({len(conditioning_columns['env_categorical'])}): {conditioning_columns['env_categorical']}")
    print(f"  - timestamp: {conditioning_columns['timestamp']}")

    return df, input_columns, output_columns, conditioning_columns


def prepare_conditional_tensors(df, input_columns, output_columns, conditioning_columns):
    """
    Prepare data tensors for conditional training from DataFrame.

    Extracts and organizes:
    - Standard features (X)
    - Target outputs (y)
    - Process ID (if available)
    - Continuous environment variables (if available)
    - Categorical environment variables (if available)
    - Timestamp (if available)

    Args:
        df (pd.DataFrame): Input DataFrame
        input_columns (list): Physical input feature names
        output_columns (list): Target output names
        conditioning_columns (dict): Dict with conditioning column information

    Returns:
        dict: Dictionary with keys:
            - 'X': numpy array (n_samples, n_features)
            - 'y': numpy array (n_samples, n_outputs)
            - 'process_id': numpy array (n_samples,) or None
            - 'env_continuous': numpy array (n_samples, n_env_cont) or None
            - 'env_continuous_masks': numpy array (n_samples, n_env_cont) boolean or None
            - 'env_categorical': dict {var_name: numpy array (n_samples,)} or None
            - 'timestamp': numpy array (n_samples,) or None
    """
    import torch

    # Standard features and targets
    X = df[input_columns].values.astype(np.float32)
    y = df[output_columns].values.astype(np.float32)

    # Initialize output dict
    data = {
        'X': X,
        'y': y,
        'process_id': None,
        'env_continuous': None,
        'env_continuous_masks': None,
        'env_categorical': None,
        'timestamp': None
    }

    # Process ID
    if conditioning_columns['process_id'] is not None:
        data['process_id'] = df[conditioning_columns['process_id']].values.astype(np.int64)

    # Continuous environment variables
    if len(conditioning_columns['env_continuous']) > 0:
        env_cont = df[conditioning_columns['env_continuous']].values.astype(np.float32)
        data['env_continuous'] = env_cont

        # Create missing masks (True = value present, False = missing)
        masks = ~np.isnan(env_cont)
        data['env_continuous_masks'] = masks

        # Fill NaN with zeros (will be handled by mask in model)
        data['env_continuous'] = np.nan_to_num(env_cont, nan=0.0)

    # Categorical environment variables
    if len(conditioning_columns['env_categorical']) > 0:
        env_cat = {}
        for col in conditioning_columns['env_categorical']:
            env_cat[col] = df[col].values.astype(np.int64)
        data['env_categorical'] = env_cat

    # Timestamp
    if conditioning_columns['timestamp'] is not None:
        data['timestamp'] = df[conditioning_columns['timestamp']].values.astype(np.float32)

    return data


def create_conditional_collate_fn(conditioning_enabled=True):
    """
    Create a custom collate function for DataLoader that handles conditioning variables.

    Args:
        conditioning_enabled (bool): Whether conditioning is enabled

    Returns:
        callable: Collate function for torch DataLoader
    """
    import torch

    def collate_fn(batch):
        """
        Custom collate function that handles both standard and conditioning data.

        Args:
            batch: List of tuples from Dataset

        Returns:
            dict: Batched data with appropriate tensors
        """
        # Each item in batch is a dict with keys: X, y, process_id, env_continuous, etc.
        if not conditioning_enabled:
            # Standard mode: only X and y
            X_batch = torch.stack([torch.from_numpy(item['X']) for item in batch])
            y_batch = torch.stack([torch.from_numpy(item['y']) for item in batch])
            return {'X': X_batch, 'y': y_batch}

        # Conditional mode: gather all fields
        X_batch = torch.stack([torch.from_numpy(item['X']) for item in batch])
        y_batch = torch.stack([torch.from_numpy(item['y']) for item in batch])

        result = {'X': X_batch, 'y': y_batch}

        # Process ID
        if 'process_id' in batch[0] and batch[0]['process_id'] is not None:
            result['process_id'] = torch.tensor([item['process_id'] for item in batch], dtype=torch.long)

        # Continuous environment variables
        if 'env_continuous' in batch[0] and batch[0]['env_continuous'] is not None:
            result['env_continuous'] = torch.stack([
                torch.from_numpy(item['env_continuous']) for item in batch
            ])

        # Continuous environment masks
        if 'env_continuous_masks' in batch[0] and batch[0]['env_continuous_masks'] is not None:
            result['env_continuous_masks'] = torch.stack([
                torch.from_numpy(item['env_continuous_masks']) for item in batch
            ])

        # Categorical environment variables
        if 'env_categorical' in batch[0] and batch[0]['env_categorical'] is not None:
            env_cat_dict = {}
            for var_name in batch[0]['env_categorical'].keys():
                env_cat_dict[var_name] = torch.tensor([
                    item['env_categorical'][var_name] for item in batch
                ], dtype=torch.long)
            result['env_categorical'] = env_cat_dict

        # Timestamp
        if 'timestamp' in batch[0] and batch[0]['timestamp'] is not None:
            result['timestamp'] = torch.tensor([item['timestamp'] for item in batch], dtype=torch.float32)

        return result

    return collate_fn
