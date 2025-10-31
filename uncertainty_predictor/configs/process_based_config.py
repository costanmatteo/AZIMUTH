"""
Process-based configuration for Uncertainty Quantification training

This configuration demonstrates how to use the automatic process-based
data loading system. Simply specify the process type, and the system will
automatically:
- Load the correct CSV file
- Exclude metadata columns (timestamps, IDs)
- Separate input and output features

To use this config:
1. Set the 'process_type' to your desired process
2. Set 'output_columns' to define what the model should predict
3. Run training with: python train.py --config configs/process_based_config.py
"""

# =============================================================================
# SELECT YOUR PROCESS
# =============================================================================
# Available processes: 'laser', 'plasma', 'galvanic', 'multibond', 'microetch'

PROCESS_TYPE = 'laser'  # Change this to your desired process

# =============================================================================
# DEFINE OUTPUT COLUMNS (what the model should predict)
# =============================================================================
# Specify which columns should be predicted by the neural network
# All other columns (except metadata) will automatically become inputs

OUTPUT_COLUMNS = [
    # TODO: Replace with your actual target columns
    # Examples:
    # 'Temperature',
    # 'Quality_Score',
    # 'Thickness',
]

# =============================================================================
# CONFIGURATION (same structure as example_config.py)
# =============================================================================

CONFIG = {
    # Data configuration
    'data': {
        # Process-based loading
        'use_process_config': True,  # Enable automatic process-based loading
        'process_type': PROCESS_TYPE,
        'output_columns': OUTPUT_COLUMNS,
        'data_dir': 'src/data/raw',  # Directory containing CSV files

        # Preprocessing
        'scaling_method': 'standard',  # 'standard', 'minmax'
        'train_size': 0.7,  # 70% for training
        'val_size': 0.15,   # 15% for validation
        'test_size': 0.15,  # 15% for testing
        'random_state': 42
    },

    # Model configuration
    'model': {
        'model_type': 'medium',  # 'small', 'medium', 'large', or 'custom'
        # If 'custom', specify architecture below:
        'hidden_sizes': [128, 64, 32],  # Used only if model_type='custom'
        'dropout_rate': 0.2,
        'use_batchnorm': True,
        'min_variance': 1e-6  # Minimum variance for numerical stability
    },

    # Training configuration
    'training': {
        'batch_size': 32,
        'epochs': 400,
        'learning_rate': 0.001,
        'weight_decay': 0.01,  # L2 regularization
        'patience': 30,  # Early stopping patience
        'device': 'auto',  # 'auto', 'cuda', or 'cpu'
        'checkpoint_dir': f'checkpoints_{PROCESS_TYPE}',
        # Variance penalty weight in loss function
        'variance_penalty_alpha': 0.2
    },

    # Uncertainty configuration
    'uncertainty': {
        'confidence_level': 0.99,  # For prediction intervals
    },

    # Miscellaneous
    'misc': {
        'random_seed': 42,
        'verbose': True
    }
}
