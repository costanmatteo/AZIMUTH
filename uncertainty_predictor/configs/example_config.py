"""
Configuration file for Uncertainty Quantification training

Customize the parameters below for your specific use case.
"""

CONFIG = {
    # Data configuration
    'data': {
        'csv_path': 'data/machinery_data.csv',  # Path to your CSV file
        'input_columns': ['param1', 'param2', 'param3', 'param4', 'param5'],  # Input features
        'output_columns': ['pressure', 'temperature', 'flow_rate'],  # Target outputs
        'scaling_method': 'standard',  # 'standard', 'minmax', or 'robust'
        'train_size': 0.7,  # 70% for training
        'val_size': 0.15,   # 15% for validation
        'test_size': 0.15,  # 15% for testing
        'random_state': 42
    },

    # Model configuration
    'model': {
        'model_type': 'medium',  # 'small', 'medium', 'large', or 'custom'
        # If 'custom', specify architecture below:
        'hidden_sizes': [128, 64, 32, 16],  # Used only if model_type='custom'
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
