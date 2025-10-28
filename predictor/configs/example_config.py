"""
Example configuration in Python (alternative to YAML)

OPTIMIZED FOR: 736 samples, 3 inputs (x,y,z) -> 1 output (res)
"""

CONFIG = {
    # Data configuration
    'data': {
        'csv_path': 'src/data/raw/Dati_prova - Foglio2.csv',
        'input_columns': [
            'x',
            'y',
            'z',
        ],
        'output_columns': [
            'res',
        ],
        'train_size': 0.7,      # 515 samples for training
        'val_size': 0.15,       # 110 samples for validation
        'test_size': 0.15,      # 111 samples for test
        'random_state': 42,
        'scaling_method': 'minmax',  # Better for inputs with different ranges
    },

    # Model configuration
    # OPTIMIZED: Small network for 3 inputs prevents overfitting
    'model': {
        'hidden_sizes': [64, 32, 16],  # Much smaller! Was [512, 256, 64, 32]
        'dropout_rate': 0.15,           # Reduced dropout for small dataset
        'model_type': 'custom',
        'use_batchnorm': False,         # Not needed for small network
    },

    # Training configuration
    'training': {
        'epochs': 300,                  # More epochs with smaller LR
        'batch_size': 16,               # Smaller batches for small dataset
        'learning_rate': 0.0005,        # Lower LR for more stable convergence
        'loss_function': 'mse',         # MSE works well for regression
        'patience': 30,                 # More patience with lower LR
        'device': 'auto',
        'checkpoint_dir': 'checkpoints',
    },

    # Other
    'misc': {
        'random_seed': 42,
        'num_workers': 4,
    }
}


# =============================================================================
# ALTERNATIVE CONFIGURATIONS TO TRY
# =============================================================================

# Option 1: MINIMAL MODEL (fastest, might be good enough)
MINIMAL_CONFIG = {
    **CONFIG,
    'model': {
        'hidden_sizes': [32, 16],       # Even smaller
        'dropout_rate': 0.1,
        'model_type': 'custom',
        'use_batchnorm': False,
    },
    'training': {
        **CONFIG['training'],
        'epochs': 200,
        'batch_size': 16,
        'learning_rate': 0.001,
        'patience': 25,
    }
}

# Option 2: DEEPER MODEL (if you want more capacity)
DEEPER_CONFIG = {
    **CONFIG,
    'model': {
        'hidden_sizes': [128, 64, 32, 16],  # More layers but smaller than original
        'dropout_rate': 0.2,
        'model_type': 'custom',
        'use_batchnorm': True,
    },
    'training': {
        **CONFIG['training'],
        'epochs': 400,
        'batch_size': 16,
        'learning_rate': 0.0003,        # Even lower LR
        'patience': 40,
    }
}

# Option 3: WITH MAE LOSS (if outliers are a problem)
MAE_CONFIG = {
    **CONFIG,
    'training': {
        **CONFIG['training'],
        'loss_function': 'mae',         # More robust to outliers
        'learning_rate': 0.001,
    }
}

# Option 4: AGGRESSIVE REGULARIZATION (if overfitting)
REGULARIZED_CONFIG = {
    **CONFIG,
    'model': {
        'hidden_sizes': [64, 32, 16],
        'dropout_rate': 0.3,            # Higher dropout
        'model_type': 'custom',
        'use_batchnorm': False,
    },
    'training': {
        **CONFIG['training'],
        'batch_size': 8,                # Even smaller batches
        'learning_rate': 0.0003,
    }
}
