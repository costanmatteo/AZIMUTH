"""
Example configuration in Python (alternative to YAML)
"""

CONFIG = {
    # Data configuration
    'data': {
        'csv_path': 'C:\\AZIMUTH\\AZIMUTH\\predictor\\src\\data\\raw\\Dati_prova - Sheet8.csv',
        'input_columns': [
            'x',
            'y',
            'z',
            # Add other parameters
        ],
        'output_columns': [
            'res',
            # Add other outputs
        ],
        'train_size': 0.7,
        'val_size': 0.15,
        'test_size': 0.15,
        'random_state': 42,
        'scaling_method': 'minmax',  #'standard' or 'minmax'
    },

    # Model configuration
    'model': {
        'hidden_sizes': [64, 32, 16],
        'dropout_rate': 0.1,
        'model_type': 'custom',  # 'small', 'medium', 'large', o 'custom'
        'use_batchnorm': True,
    },

    # Training configuration
    'training': {
        'epochs': 200,
        'batch_size': 32,
        'learning_rate': 0.001,
        'loss_function': 'mae',  # 'mse', 'mae', or 'huber'
        'patience': 15,
        'device': 'auto',  # 'cuda', 'cpu', or 'auto'
        'checkpoint_dir': 'checkpoints',
    },

    # Other
    'misc': {
        'random_seed': 42,
        'num_workers': 4,
    }
}


# Default configurations for different scenarios
SMALL_DATASET_CONFIG = {
    **CONFIG,
    'model': {
        'model_type': 'small',
        'dropout_rate': 0.1,
    },
    'training': {
        **CONFIG['training'],
        'batch_size': 16,
        'epochs': 100,
    }
}

LARGE_DATASET_CONFIG = {
    **CONFIG,
    'model': {
        'model_type': 'large',
        'dropout_rate': 0.3,
    },
    'training': {
        **CONFIG['training'],
        'batch_size': 64,
        'epochs': 300,
    }
}
