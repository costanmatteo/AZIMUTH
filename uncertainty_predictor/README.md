# Uncertainty Predictor

Neural network with **Uncertainty Quantification** for machinery prediction tasks in manufacturing process chains.

## Overview

This module provides neural networks that predict **both output values AND their uncertainty**. Instead of a single point estimate, the model outputs:

- **μ(x)**: Estimated mean value
- **σ²(x)**: Estimated uncertainty (variance)

This enables:
- Confidence bounds for each prediction
- Identifying regions where data is noisy or uncertain
- Uncertainty-aware downstream decision making
- Integration with the AZIMUTH controller optimization system

---

## How It Works

### Architecture

The `UncertaintyPredictor` network has:

```
Input Layer
    ↓
Shared Hidden Layers (feature extraction)
    ↓
    ├── Mean Head → μ (linear output)
    │
    └── Variance Head → log(σ²) → exp() → σ² (always positive)
```

### Loss Function: Gaussian Negative Log-Likelihood

```
L = 0.5 × (log(σ²) + (y - μ)² / σ²)
```

This loss function:
- Penalizes prediction errors: `(y - μ)²`
- Accounts for uncertainty: errors divided by `σ²`
- Prevents overconfidence: `log(σ²)` discourages unrealistically small variances

### Learning Behavior

The model automatically learns:
- **High σ²** in noisy/uncertain regions
- **Low σ²** where predictions are reliable

---

## Project Structure

```
uncertainty_predictor/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── uncertainty_nn.py        # UncertaintyPredictor + GaussianNLLLoss
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py               # PyTorch dataset
│   │   └── preprocessing.py         # Data preprocessing
│   ├── training/
│   │   ├── __init__.py
│   │   └── uncertainty_trainer.py   # Training loop
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py               # Metrics including calibration
│       ├── visualization.py         # Plots with uncertainty bounds
│       └── report_generator.py      # PDF report generation
│
├── scm_ds/                          # Structural Causal Model datasets
│   ├── __init__.py
│   ├── scm.py                       # SCM framework
│   └── datasets.py                  # Manufacturing process SCMs
│
├── configs/
│   └── example_config.py            # Configuration file
│
├── train.py                         # Training script
├── predict.py                       # Prediction interface
├── requirements.txt
└── README.md
```

---

## SCM Datasets

The module includes **Structural Causal Model (SCM) datasets** that simulate realistic manufacturing processes:

### Available Datasets

| Dataset | Process | Physical Model | Inputs | Outputs |
|---------|---------|----------------|--------|---------|
| `ds_scm_laser` | Laser Drilling | Light-Current-Temperature (L-I-T) | PowerTarget, AmbientTemp | ActualPower |
| `ds_scm_plasma` | Plasma Cleaning | Exponential removal rate | RF_Power, Duration | RemovalRate |
| `ds_scm_galvanic` | Galvanic Deposition | Faraday's law | CurrentDensity, Duration | Thickness |
| `ds_scm_microetch` | Micro-Etching | Arrhenius kinetics | Temperature, Concentration, Duration | RemovalDepth |

### Noise Types

Each SCM includes two types of noise for multi-scenario training:

**Structural Noise** (Environmental):
- `AmbientTemp` in laser (15-35°C)
- `Temperature` in microetch (293-323K)
- Creates diversity between operating scenarios

**Process Noise** (Measurement):
- Quantum noise, measurement uncertainty, thermal drift
- Zeroed in target trajectories for ideal behavior

### Usage Example

```python
from scm_ds.datasets import ds_scm_laser, ds_scm_plasma

# Generate laser data
X_laser, y_laser = ds_scm_laser.generate(n=1000, seed=42)

# Generate plasma data
X_plasma, y_plasma = ds_scm_plasma.generate(n=1000, seed=42)
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:**
```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
reportlab>=4.0.0
```

---

## Usage

### Training

```bash
python train.py
```

This will:
1. Load and preprocess data
2. Create uncertainty model
3. Train with Gaussian NLL loss
4. Evaluate on test set
5. Generate visualizations and PDF report

### Configuration

Edit `configs/example_config.py`:

```python
CONFIG = {
    'data': {
        'csv_path': 'data/your_data.csv',
        'input_columns': ['feature1', 'feature2', ...],
        'output_columns': ['target1', 'target2', ...],
        'scaling_method': 'standard',  # 'standard', 'minmax', 'robust'
    },
    'model': {
        'model_type': 'medium',  # 'small', 'medium', 'large', 'custom'
        'dropout_rate': 0.2,
        'min_variance': 1e-6,
    },
    'training': {
        'batch_size': 32,
        'epochs': 200,
        'learning_rate': 0.001,
        'patience': 20,
    },
    'uncertainty': {
        'confidence_level': 0.95,
    }
}
```

### Making Predictions

```bash
# From CSV file
python predict.py --checkpoint_dir checkpoints --input data/samples.csv

# From command line
python predict.py --checkpoint_dir checkpoints --input "1.2,3.4,5.6"

# With custom confidence level
python predict.py --checkpoint_dir checkpoints --input "1.2,3.4" --confidence 0.90
```

---

## Model Architecture Options

### Pre-configured Models

| Size | Hidden Layers | Best For |
|------|---------------|----------|
| **Small** | `[32, 16]` | < 1000 samples |
| **Medium** | `[128, 64, 32]` | 1000-10000 samples |
| **Large** | `[256, 128, 64, 32]` | > 10000 samples |

### Custom Architecture

```python
from src.models import UncertaintyPredictor

model = UncertaintyPredictor(
    input_size=10,
    hidden_sizes=[128, 64, 32, 16],
    output_size=5,
    dropout_rate=0.2,
    use_batchnorm=False,
    min_variance=1e-6
)
```

---

## Interpreting Results

### Calibration Metrics

The model computes calibration quality:

- **Calibration Ratio = MSE / Mean Variance**
  - ~1.0: Well calibrated
  - <0.8: Under-confident (too much uncertainty)
  - >1.2: Over-confident (too little uncertainty)

### Prediction Intervals

For 95% confidence intervals:
- ~95% of true values should fall within predicted bounds
- Check "Coverage" in evaluation output

### Output Example

```
Sample 1:
--------------------------------------------------
  Pressure:
    Mean (μ):           145.234567
    Std Dev (σ):          2.456789
    95% Interval:     [140.418765, 150.050369]
  Temperature:
    Mean (μ):            78.912345
    Std Dev (σ):          1.234567
    95% Interval:      [76.492593, 81.332097]
```

---

## Programmatic API

```python
from src.models import UncertaintyPredictor, GaussianNLLLoss
from src.training import UncertaintyTrainer

# Create model
model = UncertaintyPredictor(
    input_size=10,
    hidden_sizes=[128, 64, 32],
    output_size=5
)

# Create loss and trainer
criterion = GaussianNLLLoss()
trainer = UncertaintyTrainer(model, criterion, learning_rate=0.001)

# Train
history = trainer.train(train_loader, val_loader, epochs=100)

# Predict with uncertainty
mean, variance = trainer.predict(X_test, return_uncertainty=True)

# Compute calibration
calibration = trainer.compute_calibration_metrics(val_loader)
print(f"Calibration ratio: {calibration['calibration_ratio']:.4f}")
```

---

## Integration with Controller Optimization

This module is used by the `controller_optimization` module to:

1. **Train process predictors**: Each manufacturing process gets its own UncertaintyPredictor
2. **Provide uncertainty bounds**: Policy generators receive both μ and σ² as inputs
3. **Enable stochastic sampling**: The reparameterization trick (μ + σε, ε~N(0,1)) for differentiable sampling
4. **Support theoretical analysis**: Computing minimum achievable loss from predicted variances

### Controller Integration Example

```python
# In controller_optimization/train_processes.py
from uncertainty_predictor.src.models import UncertaintyPredictor
from uncertainty_predictor.scm_ds.datasets import ds_scm_laser

# Generate SCM data
X, y = ds_scm_laser.generate(n=5000, seed=42)

# Create and train predictor
model = UncertaintyPredictor(input_size=2, hidden_sizes=[64, 32], output_size=1)
# ... training code ...

# Save for controller use
torch.save(model.state_dict(), 'checkpoints/laser/uncertainty_predictor.pth')
```

---

## Comparison with Standard Predictor

| Feature | Standard Predictor | Uncertainty Predictor |
|---------|-------------------|----------------------|
| Output | Single value (ŷ) | Mean (μ) + Variance (σ²) |
| Loss | MSE/MAE | Gaussian NLL |
| Confidence | None | Prediction intervals |
| Noise handling | Fixed | Adaptive |
| Forward passes | 1 | 1 (same speed!) |
| Calibration | N/A | Built-in metrics |

---

## Key Advantages

1. **Quantified Uncertainty**: Know how confident each prediction is
2. **Automatic Calibration**: Model learns where to be uncertain
3. **Actionable Insights**: Identify regions needing more data
4. **Risk Management**: Make informed decisions based on confidence
5. **Single Forward Pass**: Efficient compared to ensemble methods
6. **Downstream Integration**: Feed uncertainty to decision-making systems

---

## Visualizations Generated

1. **Training History**: NLL loss and MSE over epochs
2. **Predictions with Uncertainty**: Time series with confidence bounds
3. **Scatter with Uncertainty**: True vs predicted, colored by uncertainty
4. **Uncertainty Distribution**: Histogram of predicted variances
5. **Calibration Plot**: Expected vs observed coverage

---

## Troubleshooting

### Training loss not decreasing

- Increase learning rate (0.01 instead of 0.001)
- Check data normalization
- Verify data loader is shuffling

### Over-confident predictions (calibration ratio > 1.2)

- Increase dropout rate
- Add more training data
- Use larger model

### Under-confident predictions (calibration ratio < 0.8)

- Decrease dropout rate
- Train longer
- Check for data quality issues

### NaN in variance output

- Increase `min_variance` parameter
- Check for extreme values in input data
- Reduce learning rate

---

## References

- Nix, D. A., & Weigend, A. S. (1994). "Estimating the mean and variance of the target probability distribution"
- Kendall, A., & Gal, Y. (2017). "What uncertainties do we need in Bayesian deep learning for computer vision?"
- Kingma, D. P., & Welling, M. (2014). "Auto-Encoding Variational Bayes" (reparameterization trick)

---

## Related Documentation

- [Main AZIMUTH README](../README.md)
- [Controller Optimization](../controller_optimization/README.md)
- [Noise Classification Analysis](../controller_optimization/NOISE_CLASSIFICATION_ANALYSIS.md)
- [Theoretical Loss Analysis](../controller_optimization/src/analysis/THEORETICAL_LOSS_ANALYSIS.md)

---

## License

Part of the AZIMUTH project - Universit&agrave; degli Studi di Milano, 2025.
