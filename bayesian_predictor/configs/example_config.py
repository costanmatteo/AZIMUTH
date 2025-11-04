"""
Configuration for Bayesian Neural Network

BAYESIAN-SPECIFIC PARAMETERS:
- prior_std: Standard deviation of weight prior (typically 0.5-2.0)
- kl_weight: Weight for KL divergence (typically 1/N where N = training samples)
- kl_schedule: How KL weight changes ('constant', 'linear', 'cyclical')
- kl_warmup_epochs: Epochs to warm up KL weight (for 'linear' schedule)
- n_train_samples: Monte Carlo samples during training (1-3 recommended)
- n_val_samples: Monte Carlo samples during validation (10-20 recommended)
- n_test_samples: Monte Carlo samples for final evaluation (100+ recommended)

OPTIMIZED FOR: Small-medium datasets (500-5000 samples)
"""

CONFIG = {
    # Data configuration
    'data': {
        'csv_path': 'src/data/raw/Dati_prova - Sheet13.csv',
        'input_columns': [
            'x',
            'y',
            'z',
        ],
        'output_columns': [
            'res',
        ],
        'train_size': 0.7,      # 70% for training
        'val_size': 0.15,       # 15% for validation
        'test_size': 0.15,      # 15% for test
        'random_state': 42,
        'scaling_method': 'minmax',  # 'standard' or 'minmax'
    },

    # Bayesian Model configuration
    'model': {
        'hidden_sizes': [32, 16],       # Small network to prevent overfitting
        'dropout_rate': 0.1,            # Dropout for additional regularization
        'model_type': 'custom',         # 'small', 'medium', 'large', or 'custom'
        'prior_std': 1.0,               # Prior std dev for weights (Bayesian param)
    },

    # Bayesian Training configuration
    'training': {
        'epochs': 200,
        'batch_size': 16,
        'learning_rate': 0.001,         # Lower LR often works better for Bayesian
        'weight_decay': 1e-5,           # Small L2 regularization
        'patience': 15,                 # Early stopping patience
        'device': 'auto',
        'checkpoint_dir': 'checkpoints',

        # Bayesian-specific parameters
        'kl_weight': None,              # If None, will use 1/N (recommended)
        'kl_schedule': 'linear',        # 'constant', 'linear', or 'cyclical'
        'kl_warmup_epochs': 10,         # Warmup epochs for KL weight
        'n_train_samples': 1,           # MC samples per training batch (1-3)
        'n_val_samples': 10,            # MC samples per validation batch (10-20)
    },

    # Evaluation configuration
    'evaluation': {
        'n_samples': 100,               # MC samples for final test evaluation
    },

    # Other
    'misc': {
        'random_seed': 42,
        'num_workers': 0,
    }
}


# =============================================================================
# ALTERNATIVE BAYESIAN CONFIGURATIONS
# =============================================================================

# Option 1: HIGH UNCERTAINTY MODEL (more conservative, wider confidence intervals)
HIGH_UNCERTAINTY_CONFIG = {
    **CONFIG,
    'model': {
        **CONFIG['model'],
        'prior_std': 2.0,               # Higher prior std = more uncertainty
        'dropout_rate': 0.2,
    },
    'training': {
        **CONFIG['training'],
        'kl_weight': None,              # Auto: 1/N
        'kl_schedule': 'linear',
        'kl_warmup_epochs': 20,         # Longer warmup
    }
}

# Option 2: LOW UNCERTAINTY MODEL (more confident predictions)
LOW_UNCERTAINTY_CONFIG = {
    **CONFIG,
    'model': {
        **CONFIG['model'],
        'prior_std': 0.5,               # Lower prior std = less uncertainty
        'dropout_rate': 0.05,
    },
    'training': {
        **CONFIG['training'],
        'kl_weight': None,              # Auto: 1/N
        'kl_schedule': 'constant',      # No warmup
    }
}

# Option 3: DEEP BAYESIAN MODEL (for larger datasets)
DEEP_BAYESIAN_CONFIG = {
    **CONFIG,
    'model': {
        'hidden_sizes': [128, 64, 32, 16],
        'dropout_rate': 0.2,
        'model_type': 'custom',
        'prior_std': 1.0,
    },
    'training': {
        **CONFIG['training'],
        'epochs': 300,
        'batch_size': 32,
        'learning_rate': 0.0005,
        'weight_decay': 1e-4,
        'patience': 20,
        'kl_weight': None,
        'kl_schedule': 'linear',
        'kl_warmup_epochs': 30,
        'n_train_samples': 2,           # More samples for deeper model
        'n_val_samples': 20,
    },
    'evaluation': {
        'n_samples': 200,               # More samples for final evaluation
    }
}

# Option 4: FAST TRAINING (fewer MC samples, faster convergence)
FAST_BAYESIAN_CONFIG = {
    **CONFIG,
    'model': {
        'hidden_sizes': [32, 16],
        'dropout_rate': 0.1,
        'model_type': 'custom',
        'prior_std': 1.0,
    },
    'training': {
        **CONFIG['training'],
        'epochs': 150,
        'batch_size': 32,
        'learning_rate': 0.002,
        'kl_weight': None,
        'kl_schedule': 'constant',      # No warmup = faster
        'n_train_samples': 1,
        'n_val_samples': 5,             # Fewer samples = faster
    },
    'evaluation': {
        'n_samples': 50,
    }
}

# Option 5: CYCLICAL KL ANNEALING (helps explore multiple modes)
CYCLICAL_BAYESIAN_CONFIG = {
    **CONFIG,
    'training': {
        **CONFIG['training'],
        'epochs': 300,
        'kl_schedule': 'cyclical',      # Cyclical annealing
        'n_train_samples': 2,
        'n_val_samples': 15,
    },
    'evaluation': {
        'n_samples': 150,
    }
}


# =============================================================================
# USAGE GUIDE
# =============================================================================
"""
WHEN TO USE EACH CONFIGURATION:

1. CONFIG (default):
   - General purpose Bayesian model
   - Good balance between accuracy and uncertainty
   - Suitable for most small-medium datasets

2. HIGH_UNCERTAINTY_CONFIG:
   - When you want conservative predictions
   - When dataset is very noisy or sparse
   - When safety is critical (e.g., medical, industrial)

3. LOW_UNCERTAINTY_CONFIG:
   - When you have high-quality, low-noise data
   - When you want tighter confidence intervals
   - When you trust your data more than your prior

4. DEEP_BAYESIAN_CONFIG:
   - For larger datasets (>2000 samples)
   - When you need more model capacity
   - When you have complex relationships to model

5. FAST_BAYESIAN_CONFIG:
   - For quick prototyping
   - When computational resources are limited
   - When you need faster training cycles

6. CYCLICAL_BAYESIAN_CONFIG:
   - When your loss landscape has multiple modes
   - When you want to explore different solutions
   - For research and experimentation

TUNING TIPS:

- prior_std:
  * Increase if predictions are too confident
  * Decrease if uncertainty is too high
  * Typical range: 0.5-2.0

- kl_weight:
  * Leave as None (auto 1/N) for most cases
  * Increase if model overfits (predictions too flexible)
  * Decrease if model underfits (predictions too conservative)

- kl_warmup_epochs:
  * Longer warmup = smoother training
  * Shorter warmup = faster convergence
  * Typical range: 5-30

- n_train_samples:
  * More samples = more robust but slower
  * 1 sample is usually sufficient
  * Use 2-3 for very noisy data

- n_val_samples:
  * More samples = better uncertainty estimates
  * 10-20 is a good balance
  * Can reduce for faster validation

- n_test_samples (in evaluation):
  * Use 100+ for final evaluation
  * More samples = more accurate uncertainty
  * Computational cost scales linearly
"""
