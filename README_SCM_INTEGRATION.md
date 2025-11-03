# SCM Dataset Generation Integration

This document describes the integration between the SCM dataset generator and the uncertainty predictor.

## Overview

The `scm/` module generates synthetic Supply Chain Management (SCM) datasets that are automatically consumed by the `uncertainty_predictor/` module for training uncertainty quantification models.

## Architecture

```
AZIMUTH/
├── scm/                              # SCM Dataset Generator
│   ├── __init__.py                   # Module initialization
│   ├── generator.py                  # Core generator class
│   ├── generate_dataset.py           # CLI script for generation
│   ├── README.md                     # SCM module documentation
│   └── data/                         # Generated datasets
│       └── scm_dataset.csv           # Default output
│
├── uncertainty_predictor/            # Uncertainty Predictor
│   ├── configs/
│   │   └── example_config.py         # Config (points to SCM dataset)
│   ├── train.py                      # Training script
│   └── ...
│
└── generate_and_train.py             # Integrated pipeline script
```

## Workflow

### Method 1: Integrated Pipeline (Recommended)

Run the entire pipeline with a single command:

```bash
python generate_and_train.py
```

This will:
1. Generate a fresh SCM dataset (2000 samples)
2. Train the uncertainty predictor on the generated data

### Method 2: Manual Step-by-Step

#### Step 1: Generate Dataset

```bash
cd scm
python generate_dataset.py --samples 2000
```

Options:
- `--samples N`: Number of samples (default: 2000)
- `--noise-level FLOAT`: Base noise level (default: 0.15)
- `--seed INT`: Random seed (default: 42)
- `--output PATH`: Output path (default: data/scm_dataset.csv)

#### Step 2: Train Uncertainty Predictor

```bash
cd uncertainty_predictor
python train.py
```

The configuration automatically points to the SCM-generated dataset.

## Configuration

The integration is configured in `uncertainty_predictor/configs/example_config.py`:

```python
CONFIG = {
    'data': {
        'csv_path': '../scm/data/scm_dataset.csv',  # Points to SCM dataset
        'input_columns': ['x', 'y', 'z'],           # Input features
        'output_columns': ['res_1'],                # Target output
        # ... other settings
    },
    # ... other config
}
```

## Dataset Format

The SCM generator produces datasets with the following structure:

| Column | Type | Description |
|--------|------|-------------|
| `x` | float | Time/scheduling factor (0-10) |
| `y` | float | Quantity/demand factor (0-10) |
| `z` | float | Resource/capacity factor (0-10) |
| `res_1` | float | Efficiency/performance metric (target) |

### Data Characteristics

- **Non-linear relationships**: Complex interactions between inputs
- **Heteroscedastic noise**: Uncertainty varies across the input space
- **Realistic dynamics**: Simulates real supply chain behavior

The mathematical model is:

```
res_1 = 2.0*x + 3.5*y + 1.5*z
        + 0.5*x*y - 0.3*y*z
        + 0.8*sin(2πx/10)
        - 0.4*x² + 0.2*y²
        + 5.0
        + noise(x, y, z)
```

where `noise(x, y, z)` is heteroscedastic Gaussian noise.

## Customization

### Change Dataset Size

```bash
cd scm
python generate_dataset.py --samples 5000
```

### Change Noise Level

```bash
cd scm
python generate_dataset.py --noise-level 0.2
```

### Use Homoscedastic Noise

```bash
cd scm
python generate_dataset.py --no-heteroscedastic
```

### Modify Training Parameters

Edit `uncertainty_predictor/configs/example_config.py`:

```python
CONFIG = {
    'training': {
        'batch_size': 64,        # Increase batch size
        'epochs': 500,           # More epochs
        'learning_rate': 0.001,  # Higher learning rate
        # ...
    }
}
```

## Programmatic Usage

You can also use the SCM generator programmatically:

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

# Or generate and save
df = generator.generate_and_save(
    output_path="my_dataset.csv",
    n_samples=1000
)
```

## Benefits of This Integration

1. **Reproducibility**: SCM generator ensures consistent, reproducible datasets
2. **Control**: Fine-grained control over data characteristics
3. **No External Dependencies**: No need for external data files
4. **Flexibility**: Easy to generate datasets with different properties
5. **Testing**: Perfect for testing and experimenting with the uncertainty predictor

## Example Output

After running `python generate_and_train.py`, you'll see:

```
======================================================================
SCM DATASET GENERATION
======================================================================
Number of samples: 2000
Output path: data/scm_dataset.csv
...
======================================================================

======================================================================
UNCERTAINTY QUANTIFICATION TRAINING
======================================================================
...
Training completed successfully!
```

Results are saved in:
- `scm/data/scm_dataset.csv` - Generated dataset
- `uncertainty_predictor/checkpoints_uncertainty/` - Model checkpoints and visualizations
- `uncertainty_predictor/checkpoints_uncertainty/training_report.pdf` - Comprehensive report

## Troubleshooting

### Dataset not found

Ensure the dataset is generated before training:
```bash
cd scm && python generate_dataset.py
```

### Import errors

Install required dependencies:
```bash
pip install -r uncertainty_predictor/requirements.txt
```

### Training fails

Check the configuration file and ensure all paths are correct:
```bash
cd uncertainty_predictor
python -c "from configs.example_config import CONFIG; print(CONFIG['data']['csv_path'])"
```

## Further Reading

- [SCM Module Documentation](scm/README.md) - Detailed SCM generator docs
- [Uncertainty Predictor README](uncertainty_predictor/README.md) - Uncertainty predictor docs
