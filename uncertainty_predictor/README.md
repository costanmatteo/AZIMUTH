# Uncertainty Predictor

Neural network with **Uncertainty Quantification** for machinery prediction tasks.

## What is Uncertainty Quantification?

Instead of predicting only a single value ŷ, this model predicts **two outputs**:

- **μ(x)**: Estimated mean value
- **σ²(x)**: Estimated uncertainty (variance)

This allows the model to:
- Provide confidence bounds for each prediction
- Identify regions where data is noisy or uncertain
- Know when it doesn't know (epistemic uncertainty)

## How it works

### Architecture

The network has:
- **Shared hidden layers**: Extract features from input
- **Two output heads**:
  - Mean head: predicts μ (linear output)
  - Variance head: predicts log(σ²), then exponentiated for positivity

### Loss Function: Gaussian Negative Log-Likelihood

```
L = 0.5 * (log(σ²) + (y - μ)² / σ²)
```

This loss:
- Penalizes large prediction errors: `(y - μ)²`
- Accounts for uncertainty: errors divided by `σ²`
- Prevents overconfidence: `log(σ²)` term discourages predicting unrealistically small variances

### Learning Behavior

The model automatically learns:
- **High variance (σ²)** in noisy/uncertain regions
- **Low variance (σ²)** where predictions are reliable

## Project Structure

```
uncertainty_predictor/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── uncertainty_nn.py       # UncertaintyPredictor model + GaussianNLLLoss
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py              # PyTorch dataset
│   │   └── preprocessing.py        # Data preprocessing
│   ├── training/
│   │   ├── __init__.py
│   │   └── uncertainty_trainer.py  # Training loop for uncertainty model
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py              # Metrics including calibration
│       └── visualization.py        # Plots with uncertainty bounds
├── configs/
│   └── example_config.py           # Configuration file
├── train.py                        # Main training script
├── predict.py                      # Prediction script
├── requirements.txt
└── README.md
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your data:
   - CSV file with input features and target outputs
   - Update `configs/example_config.py` with your column names and paths

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
5. Generate visualizations:
   - Training history (NLL and MSE)
   - Predictions with uncertainty bounds
   - Scatter plots colored by uncertainty
   - Uncertainty distribution

### Making Predictions

```bash
# From CSV file
python predict.py --checkpoint_dir checkpoints_uncertainty --input data/new_samples.csv

# From command line (comma-separated values)
python predict.py --checkpoint_dir checkpoints_uncertainty --input "1.2,3.4,5.6,7.8,9.0"

# With custom confidence level
python predict.py --checkpoint_dir checkpoints_uncertainty --input "1.2,3.4,5.6" --confidence 0.90
```

Output includes:
- Mean prediction (μ)
- Standard deviation (σ)
- Prediction interval bounds

## Configuration

Edit `configs/example_config.py`:

```python
CONFIG = {
    'data': {
        'csv_path': 'data/your_data.csv',
        'input_columns': ['feature1', 'feature2', ...],
        'output_columns': ['target1', 'target2', ...],
        'scaling_method': 'standard',  # 'standard', 'minmax', or 'robust'
        ...
    },
    'model': {
        'model_type': 'medium',  # 'small', 'medium', 'large', or 'custom'
        'dropout_rate': 0.2,
        'min_variance': 1e-6,  # Minimum variance for numerical stability
        ...
    },
    'training': {
        'batch_size': 32,
        'epochs': 200,
        'learning_rate': 0.001,
        'patience': 20,
        ...
    },
    'uncertainty': {
        'confidence_level': 0.95,  # For prediction intervals
    }
}
```

## Model Architecture Options

### Pre-configured models:

- **Small**: `[32, 16]` hidden layers - for limited datasets
- **Medium**: `[128, 64, 32]` hidden layers - for medium datasets
- **Large**: `[256, 128, 64, 32]` hidden layers - for large datasets

### Custom architecture:

```python
model = UncertaintyPredictor(
    input_size=10,
    hidden_sizes=[128, 64, 32, 16],  # Your custom architecture
    output_size=5,
    dropout_rate=0.2,
    use_batchnorm=False
)
```

## Interpreting Results

### Calibration Metrics

The model computes calibration metrics:
- **Calibration Ratio = MSE / Mean Variance**
  - ~1.0: Well calibrated ✓
  - <0.8: Under-confident (too much uncertainty)
  - >1.2: Over-confident (too little uncertainty)

### Prediction Intervals

For 95% confidence intervals:
- Approximately 95% of true values should fall within the predicted bounds
- Check "Coverage" in evaluation output

### Visualizations

1. **Training History**: Shows both NLL loss and MSE over epochs
2. **Predictions with Uncertainty**: Time series with confidence bounds
3. **Scatter with Uncertainty**: True vs predicted, colored by uncertainty
4. **Uncertainty Distribution**: Histogram of predicted variances

## Key Advantages

1. **Quantified Uncertainty**: Know how confident each prediction is
2. **Automatic Calibration**: Model learns where to be uncertain
3. **Actionable Insights**: Identify regions needing more data
4. **Risk Management**: Make informed decisions based on prediction confidence
5. **Single Forward Pass**: Efficient compared to ensemble methods

## Comparison with Standard Predictor

| Feature | Standard Predictor | Uncertainty Predictor |
|---------|-------------------|----------------------|
| Output | Single value (ŷ) | Mean (μ) + Variance (σ²) |
| Loss | MSE/MAE | Gaussian NLL |
| Confidence | None | Prediction intervals |
| Noise handling | Fixed | Adaptive |
| Forward passes | 1 | 1 (same speed!) |

## Example Output

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

## Advanced Usage

### Programmatic API

```python
from models import UncertaintyPredictor, GaussianNLLLoss
from training import UncertaintyTrainer

# Create model
model = UncertaintyPredictor(
    input_size=10,
    hidden_sizes=[128, 64, 32],
    output_size=5
)

# Create loss
criterion = GaussianNLLLoss()

# Create trainer
trainer = UncertaintyTrainer(model, criterion, learning_rate=0.001)

# Train
history = trainer.train(train_loader, val_loader, epochs=100)

# Predict with uncertainty
mean, variance = trainer.predict(X_test, return_uncertainty=True)
```

### Calibration Analysis

```python
# Check calibration
calibration = trainer.compute_calibration_metrics(val_loader)
print(f"Calibration ratio: {calibration['calibration_ratio']:.4f}")
print(f"Status: {calibration['interpretation']}")
```

## References

- Nix, D. A., & Weigend, A. S. (1994). "Estimating the mean and variance of the target probability distribution"
- Kendall, A., & Gal, Y. (2017). "What uncertainties do we need in Bayesian deep learning for computer vision?"

## License

Same as AZIMUTH project.

## Contact

For questions or issues, please refer to the main AZIMUTH project documentation.
