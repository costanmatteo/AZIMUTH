# Predictor - Neural Network for Machinery Output Prediction

Standard feedforward neural network for predicting machinery output values based on operational parameters.

> **Note**: This is a **legacy module**. For production use in the AZIMUTH system, prefer the [`uncertainty_predictor`](../uncertainty_predictor/) module which provides uncertainty quantification in addition to point predictions.

---

## Overview

This module provides a basic PyTorch neural network for regression tasks in manufacturing contexts. It predicts output values (e.g., pressure, temperature, velocity) from input operational parameters.

### When to Use This Module

- **Learning/prototyping**: Understanding neural network basics
- **Simple predictions**: When uncertainty quantification is not needed
- **Baseline comparisons**: Comparing with uncertainty-aware models
- **Legacy compatibility**: Maintaining existing workflows

### When to Use `uncertainty_predictor` Instead

- **Production systems**: Controllers need uncertainty information
- **Risk-sensitive applications**: Need confidence bounds
- **Multi-scenario training**: AZIMUTH controller optimization
- **Calibrated predictions**: Need to know prediction reliability

---

## Project Structure

```
predictor/
├── src/
│   ├── models/           # Neural network definitions
│   │   ├── __init__.py
│   │   └── neural_network.py
│   ├── data/             # Dataset and preprocessing
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── training/         # Training scripts
│   │   ├── __init__.py
│   │   └── trainer.py
│   └── utils/            # Utilities
│       ├── __init__.py
│       ├── visualization.py
│       └── metrics.py
│
├── data/                 # Datasets
│   ├── raw/             # Raw data
│   └── processed/       # Preprocessed data
│
├── notebooks/            # Jupyter notebooks for experiments
├── configs/              # Configuration files
│   ├── config.yaml
│   └── example_config.py
│
├── train.py             # Main training script
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

---

## Framework Choice: PyTorch

This project uses **PyTorch** for the following reasons:

| Advantage | Description |
|-----------|-------------|
| Flexibility | More control over architecture design |
| Debug-friendly | Eager execution makes debugging easier |
| Research-ready | Widely used in academic research |
| Community | Large community and documentation |

---

## Installation

```bash
cd predictor
pip install -r requirements.txt
```

---

## Quick Start

### 1. Prepare Your Data

Create a CSV file with your data:

```csv
param1,param2,param3,param4,pressure,temperature,velocity
1.2,3.4,5.6,7.8,120.5,85.2,1500
2.1,4.3,6.5,8.7,125.3,87.1,1520
...
```

Save to `data/raw/machinery_data.csv`

### 2. Configure the Model

Edit `configs/example_config.py`:

```python
CONFIG = {
    'data': {
        'csv_path': 'data/raw/machinery_data.csv',
        'input_columns': ['param1', 'param2', 'param3', 'param4'],
        'output_columns': ['pressure', 'temperature', 'velocity'],
        'test_size': 0.2,
        'val_size': 0.1,
    },
    'model': {
        'hidden_sizes': [128, 64, 32],
        'dropout_rate': 0.2,
        'use_batchnorm': True,
    },
    'training': {
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.001,
        'patience': 10,
    }
}
```

### 3. Train the Model

```bash
python train.py
```

### 4. Use for Predictions

```python
import torch
from src.models import MachineryPredictor
from src.data import DataPreprocessor

# Load model
model = MachineryPredictor(input_size=4, hidden_sizes=[64, 32], output_size=3)
model.load_state_dict(torch.load('checkpoints/best_model.pth')['model_state_dict'])
model.eval()

# Load preprocessor
preprocessor = DataPreprocessor()
preprocessor.load_scalers('checkpoints/scalers.pkl')

# Make prediction
new_input = [[1.5, 3.8, 6.2, 8.1]]
new_input_scaled = preprocessor.transform(new_input)

with torch.no_grad():
    prediction_scaled = model(torch.FloatTensor(new_input_scaled))
    prediction = preprocessor.inverse_transform_output(prediction_scaled.numpy())

print(f"Prediction: {prediction}")
# Output: Prediction: [[122.3, 86.5, 1510.2]]
```

---

## Neural Network Architecture

The network is a **Feedforward Neural Network** (Multi-Layer Perceptron):

```
Input (n features)
       ↓
  [Linear + ReLU + Dropout] × Hidden Layers
       ↓
Output (m targets)
```

### Model Sizes

| Size | Hidden Layers | Dropout | Best For |
|------|---------------|---------|----------|
| **Small** | `[32, 16]` | 0.1 | < 1000 samples |
| **Medium** | `[128, 64, 32]` | 0.2 | 1000-10000 samples |
| **Large** | `[256, 128, 64, 32]` | 0.3 | > 10000 samples |

### Creating Models

```python
from src.models import create_small_model, create_medium_model, create_large_model

# Pre-configured
model = create_medium_model(input_size=10, output_size=5)

# Custom
from src.models import MachineryPredictor
model = MachineryPredictor(
    input_size=10,
    hidden_sizes=[128, 64, 32, 16],
    output_size=5,
    dropout_rate=0.2,
    use_batchnorm=True
)
```

---

## Training Components

### Loss Functions

| Function | Description | Use When |
|----------|-------------|----------|
| **MSE** | Mean Squared Error | General purpose, penalizes large errors |
| **MAE** | Mean Absolute Error | More robust to outliers |
| **Huber** | Smooth L1 | Compromise between MSE and MAE |

### Optimizer

- **Adam**: Adaptive learning rate optimizer (default)

### Regularization

- **Dropout**: Randomly disables neurons during training
- **Early Stopping**: Stops training when validation loss plateaus

---

## Complete Workflow Example

```python
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import load_csv_data, DataPreprocessor, MachineryDataset
from src.models import create_medium_model
from src.training import ModelTrainer
from src.utils import plot_training_history, plot_predictions, calculate_metrics

# 1. Load data
X, y = load_csv_data(
    'data/raw/machinery_data.csv',
    input_columns=['param1', 'param2', 'param3'],
    output_columns=['pressure', 'temperature']
)

# 2. Preprocessing
preprocessor = DataPreprocessor(scaling_method='standard')
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)

X_train_scaled, y_train_scaled = preprocessor.fit_transform(X_train, y_train)
X_val_scaled, y_val_scaled = preprocessor.transform(X_val, y_val)
X_test_scaled, y_test_scaled = preprocessor.transform(X_test, y_test)

# 3. Create datasets and dataloaders
train_dataset = MachineryDataset(X_train_scaled, y_train_scaled)
val_dataset = MachineryDataset(X_val_scaled, y_val_scaled)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 4. Create and train model
model = create_medium_model(input_size=3, output_size=2)
trainer = ModelTrainer(model, learning_rate=0.001, loss_fn='mse')

history = trainer.train(
    train_loader,
    val_loader,
    epochs=100,
    patience=10
)

# 5. Evaluate
y_pred = trainer.predict(X_test_scaled)
y_pred_original = preprocessor.inverse_transform_output(y_pred)

metrics = calculate_metrics(
    y_test,
    y_pred_original,
    output_names=['pressure', 'temperature']
)

# 6. Visualize
plot_training_history(history['train_losses'], history['val_losses'])
plot_predictions(y_test, y_pred_original, output_names=['pressure', 'temperature'])
```

---

## Evaluation Metrics

| Metric | Description | Ideal |
|--------|-------------|-------|
| **MSE** | Mean Squared Error | → 0 |
| **RMSE** | Root MSE (same units as data) | → 0 |
| **MAE** | Mean Absolute Error | → 0 |
| **R²** | Coefficient of determination | → 1 |
| **MAPE** | Mean Absolute Percentage Error | → 0% |

---

## Tips & Best Practices

1. **Always normalize data**: Use StandardScaler or MinMaxScaler
2. **Start small**: Begin with a small model, add complexity if needed
3. **Use early stopping**: Prevents overfitting automatically
4. **Monitor train vs val loss**: Large gap indicates overfitting
5. **Keep test set sacred**: Only use for final evaluation
6. **Save scalers**: Required for making predictions on new data
7. **Set random seed**: For reproducible results

---

## Troubleshooting

### Training loss not decreasing

- Increase learning rate (try 0.01)
- Increase model size
- Check data normalization

### Validation loss much worse than training

- Normal - this is overfitting
- Increase dropout rate
- Reduce model size
- Add more data

### All predictions are similar

- Model too small
- Learning rate too low
- Not enough training data

---

## Comparison with Uncertainty Predictor

| Feature | This Module | Uncertainty Predictor |
|---------|-------------|----------------------|
| Output | Single value ŷ | Mean μ + Variance σ² |
| Loss | MSE/MAE/Huber | Gaussian NLL |
| Confidence | None | Prediction intervals |
| Complexity | Simple | Moderate |
| Use case | Basic prediction | Production systems |

**Recommendation**: Use `uncertainty_predictor` for AZIMUTH controller optimization.

---

## Learning Resources

**PyTorch:**
- Official tutorials: https://pytorch.org/tutorials/
- 60-minute blitz: https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

**Deep Learning Concepts:**
- Neural Networks and Deep Learning: http://neuralnetworksanddeeplearning.com/
- Deep Learning Book: https://www.deeplearningbook.org/

---

## Related Documentation

- [Main AZIMUTH README](../README.md)
- [Uncertainty Predictor](../uncertainty_predictor/README.md) (recommended for production)
- [Controller Optimization](../controller_optimization/README.md)

---

## License

Part of the AZIMUTH project - Universit&agrave; degli Studi di Milano, 2025.
