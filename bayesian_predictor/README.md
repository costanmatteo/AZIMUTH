# Bayesian Neural Network for Machinery Prediction

A **Bayesian Neural Network (BNN)** implementation that provides not only predictions but also **uncertainty quantification** for machinery output prediction tasks.

## 🎯 What is a Bayesian Neural Network?

Unlike standard neural networks that learn fixed weights, Bayesian Neural Networks treat weights as **probability distributions**. This provides several key advantages:

### Key Features

✅ **Uncertainty Quantification**: Know how confident the model is in each prediction
✅ **Epistemic Uncertainty**: Captures model uncertainty (lack of knowledge)
✅ **Confidence Intervals**: 68%, 95%, and 99% confidence bands
✅ **Principled Inference**: Uses Bayesian inference for predictions
✅ **Better Calibration**: More reliable uncertainty estimates
✅ **Overfitting Prevention**: Built-in regularization through priors

### When to Use Bayesian Neural Networks?

Use BNNs when:
- You need to **quantify prediction uncertainty**
- **Safety is critical** (medical, industrial applications)
- You have **limited training data**
- You want to know **where the model is uncertain**
- You need **confidence intervals** for predictions

## 📊 Comparison with Other Predictors

| Feature | Standard NN | Uncertainty NN | **Bayesian NN** |
|---------|-------------|----------------|-----------------|
| Point Predictions | ✅ | ✅ | ✅ |
| Uncertainty Estimation | ❌ | ✅ | ✅ |
| Epistemic Uncertainty | ❌ | ❌ | ✅ |
| Confidence Intervals | ❌ | ⚠️ | ✅ |
| Theoretical Foundation | ✅ | ⚠️ | ✅ |
| Training Speed | Fast | Fast | Moderate |
| Inference Speed | Fast | Fast | Slow (MC sampling) |

**Legend**: ✅ Full support, ⚠️ Partial support, ❌ Not supported

## 🚀 Quick Start

### 1. Installation

```bash
cd bayesian_predictor
pip install -r requirements.txt
```

### 2. Configure Your Data

Edit `configs/example_config.py`:

```python
CONFIG = {
    'data': {
        'csv_path': 'path/to/your/data.csv',
        'input_columns': ['x', 'y', 'z'],
        'output_columns': ['output'],
        # ...
    },
    'model': {
        'hidden_sizes': [32, 16],
        'prior_std': 1.0,  # Bayesian prior
        # ...
    },
    'training': {
        'kl_weight': None,  # Auto: 1/N
        'kl_schedule': 'linear',
        'n_train_samples': 1,
        'n_val_samples': 10,
        # ...
    }
}
```

### 3. Train the Model

```bash
python train.py
```

This will:
- Train the Bayesian neural network
- Generate uncertainty visualizations
- Save the trained model and scalers
- Create comprehensive plots with confidence intervals

### 4. Make Predictions

```bash
python predict.py --input "1.2,3.4,5.6"
```

Output example:
```
BAYESIAN PREDICTION RESULTS
======================================================================
Input Parameters:
  x: 1.2000
  y: 3.4000
  z: 5.6000

Predicted Output (with Uncertainty):
----------------------------------------------------------------------
output:
  Mean (Best Estimate):  42.3567
  Std Dev (Uncertainty): 2.1234
  68% Confidence:        [40.2333, 44.4801]
  95% Confidence:        [38.1099, 46.6035]
  Relative Uncertainty:  5.01%
======================================================================
```

## 📁 Project Structure

```
bayesian_predictor/
├── src/
│   ├── data/                    # Data loading and preprocessing
│   ├── models/
│   │   ├── bayesian_nn.py      # Bayesian neural network implementation
│   │   └── __init__.py
│   ├── training/
│   │   ├── bayesian_trainer.py # Bayesian training logic
│   │   └── __init__.py
│   └── utils/
│       ├── visualization.py     # Standard plots
│       ├── bayesian_visualization.py  # Uncertainty plots
│       ├── metrics.py
│       └── report_generator.py
├── configs/
│   └── example_config.py       # Configuration with Bayesian params
├── train.py                    # Training script
├── predict.py                  # Prediction script
└── requirements.txt
```

## 🧠 How Bayesian Neural Networks Work

### 1. Weight Distributions

Standard NN:
```
Weight = fixed value (e.g., 0.523)
```

Bayesian NN:
```
Weight ~ N(μ, σ²)
  - μ (mean): most likely weight value
  - σ² (variance): uncertainty in the weight
```

### 2. Variational Inference

The network learns:
- **Posterior distribution** over weights: q(w|θ) ≈ p(w|D)
- Uses the **reparameterization trick** for gradient computation
- Optimizes the **ELBO (Evidence Lower Bound)**:

```
ELBO = -log p(y|x,w) + β * KL[q(w|θ)||p(w)]
       └─ Data fit ─┘      └─ Regularization ─┘
```

### 3. Monte Carlo Sampling

For prediction:
1. Sample N different weight configurations from learned distributions
2. Make N predictions (one per weight sample)
3. Aggregate results to get mean and uncertainty

## 📊 Output Visualizations

The training process generates several plots:

### 1. **Bayesian Training History**
- ELBO loss (total loss)
- NLL component (data fit)
- KL component (regularization)
- Validation uncertainty evolution

### 2. **Predictions with Uncertainty**
- Scatter plot with uncertainty color-coding
- Confidence bands (68%, 95%)
- Perfect prediction line

### 3. **Uncertainty Calibration**
- Shows how well predicted uncertainty matches actual errors
- Ideal: points lie on diagonal line
- Includes statistics (% within 1σ, 2σ)

### 4. **Epistemic Uncertainty Heatmap**
- Shows which samples have high model uncertainty
- Helps identify where more data is needed

## ⚙️ Configuration Guide

### Bayesian-Specific Parameters

#### `prior_std` (Prior Standard Deviation)
- Controls initial uncertainty in weights
- **Higher values** → More uncertainty, wider confidence intervals
- **Lower values** → Less uncertainty, tighter confidence intervals
- **Typical range**: 0.5 - 2.0
- **Default**: 1.0

```python
'prior_std': 1.0  # Moderate uncertainty
'prior_std': 2.0  # High uncertainty (conservative)
'prior_std': 0.5  # Low uncertainty (confident)
```

#### `kl_weight` (KL Divergence Weight)
- Weight for KL regularization term in ELBO
- **Higher values** → Stronger regularization, stays closer to prior
- **Lower values** → More flexibility, follows data more
- **Recommended**: `1/N` where N = training samples (auto if None)

```python
'kl_weight': None          # Auto: 1/N (recommended)
'kl_weight': 0.001         # For 1000 samples
'kl_weight': 1e-4          # More flexibility
```

#### `kl_schedule` (KL Weight Schedule)
- How KL weight changes during training
- **Options**:
  - `'constant'`: Fixed throughout training
  - `'linear'`: Gradual warmup from 0 to target (recommended)
  - `'cyclical'`: Periodic annealing (for multi-modal distributions)

```python
'kl_schedule': 'linear'    # Smooth warmup (recommended)
'kl_schedule': 'constant'  # No warmup (faster)
'kl_schedule': 'cyclical'  # Explore multiple modes
```

#### `kl_warmup_epochs` (KL Warmup Duration)
- Number of epochs for KL weight warmup (for 'linear' schedule)
- **Longer warmup** → Smoother training, less overfitting
- **Shorter warmup** → Faster convergence
- **Typical range**: 5 - 30

```python
'kl_warmup_epochs': 10   # Standard warmup
'kl_warmup_epochs': 20   # Longer, more stable
'kl_warmup_epochs': 5    # Faster convergence
```

#### `n_train_samples` (Training MC Samples)
- Monte Carlo samples per training batch
- **More samples** → More robust, but slower
- **Fewer samples** → Faster training
- **Recommended**: 1-3

```python
'n_train_samples': 1     # Fast, usually sufficient
'n_train_samples': 2     # More robust
'n_train_samples': 3     # Very noisy data
```

#### `n_val_samples` (Validation MC Samples)
- Monte Carlo samples per validation batch
- **More samples** → Better uncertainty estimates
- **Recommended**: 10-20

```python
'n_val_samples': 10      # Good balance
'n_val_samples': 20      # More accurate
'n_val_samples': 5       # Faster validation
```

#### `n_samples` (Test MC Samples)
- Monte Carlo samples for final evaluation and prediction
- **More samples** → More accurate uncertainty estimates
- **Recommended**: 100+

```python
'n_samples': 100         # Standard accuracy
'n_samples': 200         # High accuracy
'n_samples': 500         # Research-grade
```

## 🔬 Advanced Usage

### Custom Model Architecture

```python
from src.models import BayesianPredictor

model = BayesianPredictor(
    input_size=10,
    hidden_sizes=[128, 64, 32],
    output_size=5,
    prior_std=1.0,
    dropout_rate=0.2
)
```

### Prediction with Custom Samples

```python
from src.training import BayesianTrainer

trainer = BayesianTrainer(model, loss_fn)
results = trainer.predict_with_uncertainty(X, n_samples=200)

print(f"Mean: {results['mean']}")
print(f"Std: {results['std']}")
print(f"95% CI: {results['confidence_intervals']['95%']}")
```

## 📈 Tuning Tips

### If Predictions Are Too Uncertain:
- ✅ Decrease `prior_std` (e.g., 0.5)
- ✅ Increase `kl_weight`
- ✅ Use more training data
- ✅ Reduce `dropout_rate`

### If Predictions Are Too Confident:
- ✅ Increase `prior_std` (e.g., 2.0)
- ✅ Decrease `kl_weight`
- ✅ Increase `dropout_rate`
- ✅ Use longer `kl_warmup_epochs`

### If Training Is Unstable:
- ✅ Use `kl_schedule: 'linear'` with longer warmup
- ✅ Reduce `learning_rate`
- ✅ Increase `weight_decay`
- ✅ Use smaller `batch_size`

### If Training Is Too Slow:
- ✅ Reduce `n_train_samples` to 1
- ✅ Reduce `n_val_samples` to 5
- ✅ Use `kl_schedule: 'constant'`
- ✅ Increase `batch_size`

## 🎓 Understanding the Output

### Mean Prediction
The average prediction across all weight samples. This is your **best estimate**.

### Standard Deviation
The uncertainty in the prediction. Higher std = more uncertain.

### Confidence Intervals
- **68% CI**: ~1σ band, "likely" range
- **95% CI**: ~2σ band, "very likely" range
- **99% CI**: ~3σ band, "almost certain" range

### Relative Uncertainty
Uncertainty as percentage of predicted value:
- **< 5%**: Very confident
- **5-15%**: Moderate confidence
- **> 15%**: High uncertainty, use with caution

## 🔍 Comparison: Standard vs Bayesian

### Standard Neural Network
```python
prediction = model(x)
# Output: 42.35
# Question: How confident is this?
# Answer: Unknown!
```

### Bayesian Neural Network
```python
results = model.predict_with_uncertainty(x, n_samples=100)
# Mean: 42.35
# Std: 2.12
# 95% CI: [38.11, 46.60]
# Interpretation:
#   "The true value is very likely between 38 and 46"
#   "Model is 5% uncertain relative to prediction"
```

## 📚 References

- **Variational Inference**: Blundell et al. (2015) - "Weight Uncertainty in Neural Networks"
- **Bayesian Deep Learning**: Gal & Ghahramani (2016) - "Dropout as a Bayesian Approximation"
- **Uncertainty Quantification**: Kendall & Gal (2017) - "What Uncertainties Do We Need?"

## 🆚 Related Projects

- **predictor/**: Standard neural network (fastest, no uncertainty)
- **uncertainty_predictor/**: Uncertainty quantification (faster, aleatoric uncertainty)
- **bayesian_predictor/**: Full Bayesian inference (best uncertainty, slower)

## 📝 License

Same as parent project.

## 🤝 Contributing

Contributions welcome! This is a research-grade implementation suitable for:
- Industrial applications requiring uncertainty
- Academic research in Bayesian deep learning
- Safety-critical prediction tasks

## ⚠️ Important Notes

1. **Training Time**: Bayesian NNs take longer to train than standard NNs
2. **Inference Time**: Predictions are slower due to Monte Carlo sampling
3. **Memory**: Model stores 2x parameters (mean and variance for each weight)
4. **Hyperparameters**: More hyperparameters to tune than standard NNs

## 🎯 Best Practices

1. Start with default config
2. Use `n_train_samples=1` for training (faster)
3. Use `n_samples=100+` for final evaluation (accuracy)
4. Monitor uncertainty calibration plots
5. Tune `prior_std` based on your data quality
6. Use linear KL warmup for stable training

---

**Created with ❤️ for reliable, uncertainty-aware predictions**
