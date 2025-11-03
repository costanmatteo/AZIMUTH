"""
Synthetic dataset generators based on Structural Causal Models (SCM).

These generators create controlled synthetic data with known causal relationships
and varying levels of noise/uncertainty for testing uncertainty quantification models.
"""

import numpy as np
import pandas as pd


def ds_scm_1_to_1_ct(n_samples=1000, seed=42, noise_level=0.1):
    """
    Generate synthetic data based on a 1-to-1 continuous treatment SCM.

    This creates a simple causal structure:
    X -> Y with controllable heteroscedastic noise

    Args:
        n_samples (int): Number of samples to generate
        seed (int): Random seed for reproducibility
        noise_level (float): Base noise level (heteroscedastic noise varies by region)

    Returns:
        tuple: (X, y, input_columns, output_columns)
            - X: Input features as numpy array
            - y: Output targets as numpy array
            - input_columns: List of input column names
            - output_columns: List of output column names
    """
    np.random.seed(seed)

    # Generate input features
    x1 = np.random.uniform(-3, 3, n_samples)
    x2 = np.random.uniform(-3, 3, n_samples)
    x3 = np.random.uniform(-3, 3, n_samples)

    # Create non-linear relationship with heteroscedastic noise
    # The true function is: y = x1^2 + sin(x2) + x3/2
    # with noise that increases in certain regions
    y_true = x1**2 + np.sin(x2) + x3 / 2

    # Heteroscedastic noise: higher noise where |x1| is large
    noise_std = noise_level * (1 + np.abs(x1))
    noise = np.random.normal(0, noise_std)

    y = y_true + noise

    # Prepare data in the expected format
    X = np.column_stack([x1, x2, x3])
    y = y.reshape(-1, 1)

    input_columns = ['x1', 'x2', 'x3']
    output_columns = ['y']

    return X, y, input_columns, output_columns


def ds_scm_multivariate(n_samples=1000, seed=42, noise_level=0.1):
    """
    Generate synthetic multivariate data with multiple outputs.

    Creates a structure with multiple correlated outputs:
    X -> [Y1, Y2, Y3]

    Args:
        n_samples (int): Number of samples to generate
        seed (int): Random seed for reproducibility
        noise_level (float): Base noise level

    Returns:
        tuple: (X, y, input_columns, output_columns)
    """
    np.random.seed(seed)

    # Generate input features
    x1 = np.random.uniform(-2, 2, n_samples)
    x2 = np.random.uniform(-2, 2, n_samples)

    # Multiple outputs with different relationships
    y1 = 2 * x1 + np.random.normal(0, noise_level, n_samples)
    y2 = x1**2 + x2 + np.random.normal(0, noise_level * 1.5, n_samples)
    y3 = np.sin(x1 + x2) + np.random.normal(0, noise_level * 0.5, n_samples)

    X = np.column_stack([x1, x2])
    y = np.column_stack([y1, y2, y3])

    input_columns = ['x1', 'x2']
    output_columns = ['y1', 'y2', 'y3']

    return X, y, input_columns, output_columns


def ds_scm_high_noise_regions(n_samples=1000, seed=42):
    """
    Generate data with distinct high and low noise regions.

    Useful for testing if the model can learn to increase uncertainty
    in noisy regions and decrease it in clean regions.

    Args:
        n_samples (int): Number of samples to generate
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (X, y, input_columns, output_columns)
    """
    np.random.seed(seed)

    # Generate input in range [-3, 3]
    x = np.random.uniform(-3, 3, n_samples)

    # True function: simple quadratic
    y_true = x**2

    # Create distinct noise regions:
    # - Low noise for |x| < 1.5
    # - High noise for |x| >= 1.5
    noise = np.zeros(n_samples)
    low_noise_mask = np.abs(x) < 1.5

    noise[low_noise_mask] = np.random.normal(0, 0.05, low_noise_mask.sum())
    noise[~low_noise_mask] = np.random.normal(0, 0.5, (~low_noise_mask).sum())

    y = y_true + noise

    X = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    input_columns = ['x']
    output_columns = ['y']

    return X, y, input_columns, output_columns


# Dataset registry for easy access
AVAILABLE_DATASETS = {
    '1_to_1_ct': ds_scm_1_to_1_ct,
    'multivariate': ds_scm_multivariate,
    'high_noise_regions': ds_scm_high_noise_regions,
}


def get_dataset(dataset_type='1_to_1_ct', **kwargs):
    """
    Get a synthetic dataset by name.

    Args:
        dataset_type (str): Name of the dataset
        **kwargs: Arguments to pass to the dataset generator

    Returns:
        tuple: (X, y, input_columns, output_columns)
    """
    if dataset_type not in AVAILABLE_DATASETS:
        raise ValueError(
            f"Unknown dataset type: {dataset_type}. "
            f"Available: {list(AVAILABLE_DATASETS.keys())}"
        )

    return AVAILABLE_DATASETS[dataset_type](**kwargs)
