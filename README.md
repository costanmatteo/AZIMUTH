# AZIMUTH

**Controller Optimization System for Manufacturing Process Chains**

*Developed at Universit&agrave; degli Studi di Milano*

---

## Overview

AZIMUTH is a sophisticated system for optimizing sequences of manufacturing processes through neural network-based policy generators. The system learns optimal control parameters for multi-step manufacturing workflows including laser drilling, plasma cleaning, galvanic deposition, and micro-etching.

### Core Innovation

AZIMUTH employs a **two-level architecture** that separates uncertainty prediction from control policy generation:

1. **Uncertainty Predictors**: Neural networks that predict both output mean AND variance for each manufacturing process (frozen during controller training)
2. **Policy Generators**: Trainable neural networks that generate optimal process parameters based on previous process outputs and their uncertainties
3. **Surrogate Model**: Evaluates the reliability (quality) of complete manufacturing trajectories

This architecture enables the controller to make uncertainty-aware decisions, adapting its control strategy based on the confidence of upstream predictions.

---

## Project Structure

```
AZIMUTH/
├── controller_optimization/       # Main controller optimization module
│   ├── src/
│   │   ├── models/               # Policy generator, surrogate model
│   │   ├── training/             # Training loops for predictors and controllers
│   │   ├── analysis/             # Theoretical loss analysis module
│   │   └── utils/                # Process chain, target generation, metrics
│   ├── configs/                  # Process and controller configurations
│   ├── checkpoints/              # Trained models (generated)
│   ├── train_processes.py        # Step 1: Train uncertainty predictors
│   └── train_controller.py       # Step 2: Train policy generators
│
├── uncertainty_predictor/         # Uncertainty quantification module
│   ├── src/
│   │   ├── models/               # UncertaintyPredictor, GaussianNLLLoss
│   │   ├── data/                 # Dataset and preprocessing
│   │   ├── training/             # Uncertainty-aware training
│   │   └── utils/                # Metrics, visualization, reports
│   ├── scm_ds/                   # Structural Causal Models for datasets
│   │   ├── scm.py                # SCM framework
│   │   └── datasets.py           # Laser, Plasma, Galvanic, Microetch SCMs
│   ├── train.py                  # Training script
│   └── predict.py                # Prediction interface
│
├── predictor/                     # Basic neural network predictor (legacy)
│   ├── src/                      # Standard feedforward architecture
│   ├── train.py                  # Training script
│   └── README.md
│
└── docs/                          # Additional documentation
    └── azimuth_workflow.tex       # LaTeX workflow diagram
```

---

## Key Features

### Multi-Scenario Training
- Trains on **50 diverse operating conditions** instead of a single point
- Structural noise (environmental factors) creates scenario diversity
- Ensures robust generalization across varying conditions

### Uncertainty Quantification
- Predicts both mean (μ) and variance (σ²) for each process output
- Automatic calibration through Gaussian NLL loss
- Prediction intervals with confidence bounds

### Multi-Controller Architecture
- Separate policy generator for each process transition
- Shared uncertainty predictors across all phases
- Each controller trained to minimize reliability loss

### Theoretical Loss Analysis
- Computes minimum achievable loss (L_min) analytically
- Decomposes loss into reducible (Bias²) and irreducible (Var[F]) components
- Tracks training efficiency toward theoretical optimum

### Comprehensive Reporting
- PDF reports with configuration summaries
- Multi-scenario metrics with robustness statistics
- Training visualizations and trajectory comparisons

---

## Manufacturing Process Chain

AZIMUTH models a 4-process manufacturing chain:

```
┌─────────────────────────────────────────────────────────────────┐
│ PROCESS 1: Laser Drilling                                       │
│  Input: [PowerTarget, AmbientTemp]                              │
│  → UncertaintyPredictor → (ActualPower, σ²)                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│ POLICY GENERATOR 1 (trainable)                                  │
│  Input: [AmbientTemp, ActualPower, σ²]                          │
│  → Outputs: RF_Power, Duration                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│ PROCESS 2: Plasma Cleaning                                      │
│  Input: [RF_Power, Duration]                                    │
│  → UncertaintyPredictor → (RemovalRate, σ²)                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│ POLICY GENERATOR 2 (trainable)                                  │
│  → Outputs: CurrentDensity, Duration                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│ PROCESS 3: Galvanic Deposition                                  │
│  → UncertaintyPredictor → (Thickness, σ²)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│ POLICY GENERATOR 3 (trainable)                                  │
│  → Outputs: Temperature, Concentration, Duration                │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│ PROCESS 4: Micro-Etching                                        │
│  → UncertaintyPredictor → (RemovalDepth, σ²)                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│ SURROGATE MODEL: Compute Reliability F(trajectory)              │
│  • Per-process quality functions Q_i(output)                    │
│  • Combined weighted reliability score                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

```bash
# Python 3.9+ required
pip install -r controller_optimization/requirements.txt
```

### Step 1: Train Uncertainty Predictors

```bash
cd controller_optimization
python train_processes.py
```

This trains a separate uncertainty predictor for each process (laser, plasma, galvanic, microetch) using Structural Causal Model datasets.

**Output:**
- `checkpoints/{process}/uncertainty_predictor.pth` - Model weights
- `checkpoints/{process}/scalers.pkl` - Data preprocessors
- `checkpoints/{process}/training_report.pdf` - Training report

### Step 2: Train Controller (Policy Generators)

```bash
python train_controller.py
```

This trains policy generators that learn to optimize process parameters based on upstream outputs and uncertainties.

**Output:**
- `checkpoints/controller/policy_*.pth` - Policy generator weights
- `checkpoints/controller/controller_report.pdf` - Comprehensive report
- `checkpoints/controller/final_results.json` - Aggregated metrics

### Expected Results

```
FINAL RESULTS - AGGREGATED OVER ALL SCENARIOS
============================================
Number of scenarios:           50

F* (target, optimal):
  Mean:  0.849 ± 0.003
  Range: [0.843, 0.855]

F' (baseline, no controller):
  Mean:  0.821 ± 0.005
  Range: [0.810, 0.832]

F  (actual, with controller):
  Mean:  0.846 ± 0.004
  Range: [0.838, 0.854]

Improvement over baseline:     +3.05%
Gap from optimal:              0.35%
Robustness (std of F):         0.004
============================================
```

---

## Modules

### Controller Optimization (`controller_optimization/`)

The main module for training and evaluating the complete control system.

**Key Components:**
- `PolicyGenerator`: Neural network that generates optimal process parameters
- `ProTSurrogate`: Evaluates trajectory reliability using quality functions
- `ProcessChain`: Orchestrates multi-process pipeline with uncertainty propagation
- `ControllerTrainer`: Training loop with multi-scenario support

See [controller_optimization/README.md](controller_optimization/README.md) for detailed documentation.

### Uncertainty Predictor (`uncertainty_predictor/`)

Neural network with uncertainty quantification for machinery prediction tasks.

**Key Features:**
- Outputs both mean (μ) and variance (σ²)
- Gaussian Negative Log-Likelihood loss function
- Automatic calibration metrics

See [uncertainty_predictor/README.md](uncertainty_predictor/README.md) for detailed documentation.

### Basic Predictor (`predictor/`)

Standard feedforward neural network for baseline predictions (legacy module).

See [predictor/README.md](predictor/README.md) for documentation.

---

## SCM Datasets

AZIMUTH uses Structural Causal Model (SCM) datasets that model the physical processes:

| Dataset | Process | Physical Model | Key Variables |
|---------|---------|----------------|---------------|
| `ds_scm_laser` | Laser Drilling | Light-Current-Temperature | PowerTarget, AmbientTemp → ActualPower |
| `ds_scm_plasma` | Plasma Cleaning | Exponential removal rate | RF_Power, Duration → RemovalRate |
| `ds_scm_galvanic` | Galvanic Deposition | Faraday's law | CurrentDensity, Duration → Thickness |
| `ds_scm_microetch` | Micro-Etching | Arrhenius kinetics | Temperature, Concentration → RemovalDepth |

### Noise Classification

Each SCM includes two types of noise:

- **Structural Noise**: Environmental conditions (e.g., AmbientTemp) - creates scenario diversity
- **Process Noise**: Measurement/actuator imperfections - zeroed in target trajectories

See [NOISE_CLASSIFICATION_ANALYSIS.md](controller_optimization/NOISE_CLASSIFICATION_ANALYSIS.md) for detailed analysis.

---

## Configuration

### Process Configuration

Edit `controller_optimization/configs/processes_config.py`:

```python
PROCESSES = [
    {
        'name': 'laser',
        'scm_dataset_type': 'laser',
        'input_dim': 2,
        'output_dim': 1,
        'input_labels': ['PowerTarget', 'AmbientTemp'],
        'output_labels': ['ActualPower'],
        # ...
    },
    # ... more processes
]
```

### Controller Configuration

Edit `controller_optimization/configs/controller_config.py`:

```python
CONTROLLER_CONFIG = {
    'target': {
        'n_samples': 50,  # Number of training scenarios
        'seed': 42
    },
    'policy_generator': {
        'architecture': 'medium',  # 'small', 'medium', 'large', 'custom'
        'hidden_sizes': [64, 32, 16],
        'dropout': 0.1,
    },
    'training': {
        'epochs': 200,
        'learning_rate': 0.001,
        'lambda_bc': 0.1,  # Behavior cloning weight
        'reliability_loss_scale': 100.0,
    },
}
```

---

## Advanced Usage

### Train Specific Processes Only

```bash
python train_processes.py --processes laser plasma
```

### Skip Already Trained Processes

```bash
python train_processes.py --skip-existing
```

### Use GPU

```bash
python train_processes.py --device cuda
python train_controller.py  # Auto-detects GPU
```

### Quick Test (5 scenarios)

For faster testing, reduce scenarios in `controller_config.py`:

```python
'target': {'n_samples': 5, 'seed': 42},
```

---

## Theoretical Loss Analysis

AZIMUTH includes a theoretical framework for computing the minimum achievable loss when training with stochastic sampling.

### Key Concepts

- **L_min**: Minimum achievable loss (irreducible due to stochasticity)
- **Var[F]**: Variance component (from stochastic sampling)
- **Bias²**: Systematic bias component (reducible through training)
- **Efficiency**: L_min / observed loss (100% = optimal)

### Output Files

- `theoretical_analysis_data.json` - Complete analysis data
- `theoretical_analysis_report.txt` - Human-readable report
- `theoretical_*.png` - Visualization plots

See [THEORETICAL_LOSS_ANALYSIS.md](controller_optimization/src/analysis/THEORETICAL_LOSS_ANALYSIS.md) for mathematical details.

---

## Troubleshooting

### Error: "Uncertainty predictor not found"

Run `train_processes.py` first to train the uncertainty predictors.

### Training diverges (F decreases)

- Reduce learning rate: `learning_rate: 0.0005`
- Increase BC weight: `lambda_bc: 0.2`
- Use smaller architecture: `architecture: 'small'`

### Training is slow with 50 scenarios

Start with 5-10 scenarios for testing. Use full 50 for production.

### GPU out of memory

- Reduce batch size
- Use smaller model architecture
- Process fewer scenarios

---

## Dependencies

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
pypdf==6.2.0
```

---

## Citation

If you use AZIMUTH in your research, please cite:

```bibtex
@software{azimuth2025,
  title = {AZIMUTH: Controller Optimization System for Manufacturing Process Chains},
  author = {Universit\`{a} degli Studi di Milano},
  year = {2025},
  url = {https://github.com/costanmatteo/AZIMUTH}
}
```

---

## License

This project is developed at Universit&agrave; degli Studi di Milano.

---

## Contact

For questions or issues, please open an issue on the GitHub repository.
