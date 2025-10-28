"""
Configuration file for Uncertainty Quantification training

Customize the parameters below for your specific use case.
"""

CONFIG = {
    # Data configuration
    'data': {
        'csv_path': 'src/data/raw/Dati_prova - Sheet19.csv',  # Path to your CSV file
        'input_columns': ['x', 'y', 'z'],  # Input features
        'output_columns': ['res_1', 'res_2'],  # Target outputs
        'scaling_method': 'standard',  # 'standard', 'minmax', or 'robust'
        'train_size': 0.7,  # 70% for training
        'val_size': 0.15,   # 15% for validation
        'test_size': 0.15,  # 15% for testing
        'random_state': 42
    },

    # Model configuration
    'model': {
        'model_type': 'custom',  # 'small', 'medium', 'large', or 'custom'
        # If 'custom', specify architecture below:
        'hidden_sizes': [32, 16],  # Used only if model_type='custom'
        'dropout_rate': 0.2,
        'use_batchnorm': False,
        'min_variance': 1e-6  # Minimum variance for numerical stability
    },

    # Training configuration
    'training': {
        'batch_size': 32,
        'epochs': 200,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,  # L2 regularization
        'patience': 20,  # Early stopping patience
        'device': 'auto',  # 'auto', 'cuda', or 'cpu'
        'checkpoint_dir': 'checkpoints_uncertainty'
    },

    # Uncertainty configuration
    'uncertainty': {
        'confidence_level': 0.95,  # For prediction intervals (95%)
    },

    # Miscellaneous
    'misc': {
        'random_seed': 42,
        'verbose': True
    }
}
