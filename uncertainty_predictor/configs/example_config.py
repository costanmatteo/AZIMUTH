"""
Configuration file for Uncertainty Quantification training

Customize the parameters below for your specific use case.
"""

CONFIG = {
    # Data configuration
    'data': {
        'csv_path': 'src/data/raw/Dati_prova - Sheet44.csv',  # Path to your CSV file
        'input_columns': ['x', 'y', 'z'],  # Input features
        'output_columns': ['res_1'],  # Target outputs
        'scaling_method': 'standard',  # 'standard', 'minmax'
        'train_size': 0.7,  # 70% for training
        'val_size': 0.15,   # 15% for validation
        'test_size': 0.15,  # 15% for testing
        'random_state': 42
    },

    # Model configuration
    'model': {
        'model_type': 'custom',  # 'small', 'medium', 'large', or 'custom'
        # If 'custom', specify architecture below:
        'hidden_sizes': [256, 128, 64, 32, 16],  # Used only if model_type='custom'
        'dropout_rate': 0.0,
        'use_batchnorm': True,
        'min_variance': 1e-6  # Minimum variance for numerical stability
    },

    # Training configuration
    'training': {
        'batch_size': 8,
        'epochs': 500,
        'learning_rate': 0.001,
        'weight_decay': 0.0,  # L2 regularization
        'patience': 100,  # Early stopping patience
        'device': 'auto',  # 'auto', 'cuda', or 'cpu'
        'checkpoint_dir': 'checkpoints_uncertainty',
        # Variance penalty weight in loss function: L = (y-μ)²/σ² + α*log(σ²)
        # α = 1.0: Standard Gaussian NLL
        # α < 1.0: Reduces penalty for large variances (recommended for over-confident models)
        # α > 1.0: Increases penalty for large variances
        'variance_penalty_alpha': 100  # Try values like 0.1-0.5 for over-confident models
    },

    # Uncertainty configuration
    'uncertainty': {
        'confidence_level': 0.99,  # For prediction intervals (0.95, 0.99, etc.)
    },

    # Miscellaneous
    'misc': {
        'random_seed': 42,
        'verbose': True
    }
}
