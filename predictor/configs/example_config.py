"""
Example configuration in Python (alternative to YAML)

OPTIMIZED FOR: 736 samples, 3 inputs (x,y,z) -> 1 output (res)
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
        'train_size': 0.7,      #  samples for training
        'val_size': 0.15,       #  samples for validation
        'test_size': 0.15,      #  samples for test
        'random_state': 42,
        'scaling_method': 'minmax',  # 'standard' o 'minmax'
    },

    # Model configuration
    # OPTIMIZED: Small network for 3 inputs prevents overfitting
    'model': {
        'hidden_sizes': [32, 16],  
        'dropout_rate': 0.1,           # Reduced dropout for small dataset
        'model_type': 'custom',
        'use_batchnorm': False,         # Not needed for small network
    },

    # Training configuration
    'training': {
        'epochs': 200,                  # More epochs with smaller LR
        'batch_size': 16,               # Smaller batches for small dataset
        'learning_rate': 0.01,        # Lower LR for more stable convergence
        'weight_decay': 0.0001,         # L2 regularization (0.0 = no regularization)
        'loss_function': 'mae',         # mse, mae, huber
        'patience': 10,                 # More patience with lower LR
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

# Option 1: MINIMAL MODEL (fa
# stest, might be good enough)
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
        'weight_decay': 0.0,            # No L2 regularization for minimal model
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
        'weight_decay': 0.0005,         # More regularization for deeper model
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
        'weight_decay': 0.0001,
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
        'weight_decay': 0.001,          # Strong L2 regularization
    }
}
