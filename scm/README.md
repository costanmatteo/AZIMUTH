# SCM Dataset Generator

This module generates synthetic Supply Chain Management (SCM) datasets for training the uncertainty predictor model.

## Overview

The SCM dataset generator creates realistic synthetic data that simulates a manufacturing or logistics process with:
- Non-linear relationships between input parameters and output metrics
- Heteroscedastic noise (uncertainty varies across input space)
- Realistic supply chain dynamics

## Dataset Format

The generated dataset contains the following columns:

| Column | Description | Type |
|--------|-------------|------|
| `x` | Time/scheduling factor | float |
| `y` | Quantity/demand factor | float |
| `z` | Resource/capacity factor | float |
| `res_1` | Efficiency/performance metric (target) | float |

## Usage

### Quick Start

Generate a dataset with default parameters:

```bash
cd scm
python generate_dataset.py
```

This will create a dataset with 2000 samples at `scm/data/scm_dataset.csv`.

### Advanced Options

```bash
python generate_dataset.py --samples 5000 \
                          --output data/my_dataset.csv \
                          --noise-level 0.2 \
                          --seed 123
```

#### Parameters

- `--samples N`: Number of samples to generate (default: 2000)
- `--output PATH`: Output path for CSV file (default: data/scm_dataset.csv)
- `--noise-level FLOAT`: Base noise level (default: 0.15)
- `--seed INT`: Random seed for reproducibility (default: 42)
- `--no-heteroscedastic`: Disable varying noise (use constant noise)

### Programmatic Usage

```python
from scm import SCMDataGenerator

# Create generator
generator = SCMDataGenerator(
    random_seed=42,
    noise_level=0.15,
    heteroscedastic=True
)

# Generate dataset
df = generator.generate_dataset(n_samples=1000)

# Or generate and save directly
df = generator.generate_and_save(
    output_path="my_dataset.csv",
    n_samples=1000
)
```

## Integration with Uncertainty Predictor

After generating a dataset, update the uncertainty predictor configuration:

1. Open `uncertainty_predictor/configs/example_config.py`
2. Update the `csv_path`:
   ```python
   'csv_path': '../scm/data/scm_dataset.csv'
   ```
3. Train the model:
   ```bash
   cd uncertainty_predictor
   python train.py
   ```

## Data Generation Process

The generator creates data following this process:

1. **Input Generation**: Randomly samples `x`, `y`, `z` from specified ranges
2. **True Function**: Computes output using a complex non-linear function that includes:
   - Linear terms for each input
   - Interaction terms between inputs
   - Non-linear terms (quadratic, sinusoidal)
3. **Noise Addition**: Adds heteroscedastic Gaussian noise that varies with input location
4. **Output**: Returns DataFrame with all columns

### Mathematical Model

The true output function (before noise) is:

```
res_1 = 2.0*x + 3.5*y + 1.5*z
        + 0.5*x*y - 0.3*y*z
        + 0.8*sin(2πx/10)
        - 0.4*x² + 0.2*y²
        + 5.0
```

The noise scale is heteroscedastic:
```
σ(x,y,z) = σ_base * (1 + 0.5|x-5|/5) * (1 + 0.3*y/10) * (1 + 0.2*z/10)
```

## Directory Structure

```
scm/
├── __init__.py           # Module initialization
├── generator.py          # Core generator class
├── generate_dataset.py   # CLI script for generation
├── README.md             # This file
└── data/                 # Generated datasets (created automatically)
    └── scm_dataset.csv   # Default output location
```

## Examples

### Generate training dataset
```bash
python generate_dataset.py --samples 2000 --output data/train_dataset.csv
```

### Generate validation dataset with different seed
```bash
python generate_dataset.py --samples 500 --output data/val_dataset.csv --seed 999
```

### Generate low-noise dataset
```bash
python generate_dataset.py --samples 1000 --noise-level 0.05
```

### Generate dataset with constant noise
```bash
python generate_dataset.py --no-heteroscedastic
```

## Notes

- The generated data is synthetic and designed for testing/training the uncertainty predictor
- The heteroscedastic noise allows the uncertainty model to learn region-dependent uncertainties
- The random seed ensures reproducibility across runs
- All inputs are generated uniformly in the range [0, 10] by default
