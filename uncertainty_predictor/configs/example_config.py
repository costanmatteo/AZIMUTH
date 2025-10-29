"""
Configuration file for Uncertainty Quantification training

Customize the parameters below for your specific use case.
"""

CONFIG = {
    # Data configuration
    'data': {
        'csv_path': 'src/data/raw/Dati_prova - Sheet16.csv',  # Path to your CSV file
        'input_columns': ['x', 'y', 'z'],  # Input feature column names
        'output_columns': ['res_1'],  # Target output column names
        'scaling_method': 'standard',  # Feature scaling: 'standard' (zero mean, unit variance) or 'minmax' (0-1 range)
        'train_size': 0.7,  # Fraction of data for training (0.0-1.0)
        'val_size': 0.15,  # Fraction of data for validation (0.0-1.0)
        'test_size': 0.15,  # Fraction of data for testing (0.0-1.0)
        'random_state': 42  # Random seed for reproducible data splits
    },

    # Model configuration
    'model': {
        'model_type': 'custom',  # Model size preset: 'small', 'medium', 'large', or 'custom'
        'hidden_sizes': [32, 16],  # Hidden layer sizes (used only if model_type='custom')
        'dropout_rate': 0.2,  # Dropout probability for regularization (0.0-1.0)
        'use_batchnorm': True,  # Whether to use batch normalization layers
        'min_variance': 1e-6  # Minimum variance threshold for numerical stability
    },

    # Training configuration
    'training': {
        'batch_size': 16,  # Number of samples per training batch
        'epochs': 300,  # Maximum number of training epochs
        'learning_rate': 0.001,  # Initial learning rate for optimizer
        'weight_decay': 0.01,  # L2 regularization penalty (0.0 = no regularization)
        'patience': 20,  # Early stopping: epochs to wait without improvement
        'device': 'auto',  # Computing device: 'auto' (GPU if available), 'cuda', or 'cpu'
        'checkpoint_dir': 'checkpoints_uncertainty',  # Directory to save model checkpoints
        'variance_penalty_alpha': 1  # Weight for variance term in loss function (1.0 = standard Gaussian NLL)
    },

    # Uncertainty configuration
    'uncertainty': {
        'confidence_level': 0.99,  # Confidence level for prediction intervals (0.0-1.0, e.g., 0.95, 0.99)
    },

    # Miscellaneous
    'misc': {
        'random_seed': 42,  # Global random seed for reproducibility
        'verbose': True  # Whether to print detailed training information
    }
}