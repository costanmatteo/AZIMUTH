"""
SCM Data Generator

Generates synthetic supply chain management data with realistic
relationships between input parameters and output metrics.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class SCMDataGenerator:
    """
    Generator for synthetic Supply Chain Management data.

    This generator creates data that simulates a manufacturing or logistics
    process where:
    - x, y, z are input parameters (e.g., time, quantity, resource level)
    - res_1 is the output metric (e.g., efficiency, cost, quality)

    The data includes:
    - Non-linear relationships between inputs and output
    - Noise to simulate real-world measurement uncertainty
    - Heteroscedastic noise (uncertainty varies with input region)

    Args:
        random_seed (int): Seed for reproducibility
        noise_level (float): Base noise level (default: 0.1)
        heteroscedastic (bool): If True, noise varies with inputs (default: True)
    """

    def __init__(
        self,
        random_seed: int = 42,
        noise_level: float = 0.1,
        heteroscedastic: bool = True
    ):
        self.random_seed = random_seed
        self.noise_level = noise_level
        self.heteroscedastic = heteroscedastic
        np.random.seed(random_seed)

    def _compute_output(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Compute the true output function (without noise).

        This simulates a complex supply chain relationship where:
        - x represents time/scheduling factor
        - y represents quantity/demand factor
        - z represents resource/capacity factor
        - res_1 is the resulting efficiency/performance metric

        Args:
            x, y, z: Input parameter arrays

        Returns:
            Array of true output values
        """
        # Complex non-linear relationship
        # Simulate a supply chain optimization problem
        res_1 = (
            2.0 * x +                          # Linear time component
            3.5 * y +                          # Linear quantity component
            1.5 * z +                          # Linear resource component
            0.5 * x * y +                      # Interaction between time and quantity
            -0.3 * y * z +                     # Interaction between quantity and resource
            0.8 * np.sin(2 * np.pi * x / 10) + # Cyclical time patterns
            -0.4 * x**2 +                      # Quadratic time effect
            0.2 * y**2 +                       # Quadratic quantity effect
            5.0                                 # Baseline offset
        )

        return res_1

    def _compute_noise_scale(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Compute the noise scale for each sample (heteroscedastic noise).

        The noise is higher in certain regions to simulate regions of
        higher uncertainty (e.g., extreme operating conditions).

        Args:
            x, y, z: Input parameter arrays

        Returns:
            Array of noise scales
        """
        if not self.heteroscedastic:
            return np.ones_like(x) * self.noise_level

        # Noise increases with extreme values and certain interactions
        base_scale = self.noise_level

        # Higher noise at extreme values
        x_factor = 1.0 + 0.5 * np.abs(x - 5) / 5  # More noise away from center
        y_factor = 1.0 + 0.3 * (y / 10)           # More noise with high quantity
        z_factor = 1.0 + 0.2 * (z / 10)           # More noise with high resources

        scale = base_scale * x_factor * y_factor * z_factor

        return scale

    def generate_dataset(
        self,
        n_samples: int = 1000,
        x_range: Tuple[float, float] = (0, 10),
        y_range: Tuple[float, float] = (0, 10),
        z_range: Tuple[float, float] = (0, 10)
    ) -> pd.DataFrame:
        """
        Generate a complete synthetic dataset.

        Args:
            n_samples (int): Number of samples to generate
            x_range (tuple): Range for x values (min, max)
            y_range (tuple): Range for y values (min, max)
            z_range (tuple): Range for z values (min, max)

        Returns:
            pd.DataFrame: Dataset with columns ['x', 'y', 'z', 'res_1']
        """
        # Generate random input values
        x = np.random.uniform(x_range[0], x_range[1], n_samples)
        y = np.random.uniform(y_range[0], y_range[1], n_samples)
        z = np.random.uniform(z_range[0], z_range[1], n_samples)

        # Compute true output
        res_1_true = self._compute_output(x, y, z)

        # Add heteroscedastic noise
        noise_scale = self._compute_noise_scale(x, y, z)
        noise = np.random.normal(0, 1, n_samples) * noise_scale
        res_1 = res_1_true + noise

        # Create DataFrame
        df = pd.DataFrame({
            'x': x,
            'y': y,
            'z': z,
            'res_1': res_1
        })

        return df

    def generate_and_save(
        self,
        output_path: str,
        n_samples: int = 1000,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate dataset and save to CSV file.

        Args:
            output_path (str): Path where to save the CSV file
            n_samples (int): Number of samples to generate
            **kwargs: Additional arguments passed to generate_dataset

        Returns:
            pd.DataFrame: Generated dataset
        """
        df = self.generate_dataset(n_samples=n_samples, **kwargs)
        df.to_csv(output_path, index=False)
        print(f"Dataset generated and saved to: {output_path}")
        print(f"  Samples: {n_samples}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Shape: {df.shape}")
        print(f"\nSample statistics:")
        print(df.describe())

        return df


if __name__ == "__main__":
    # Example usage
    generator = SCMDataGenerator(
        random_seed=42,
        noise_level=0.15,
        heteroscedastic=True
    )

    df = generator.generate_and_save(
        output_path="scm_dataset.csv",
        n_samples=1000
    )
