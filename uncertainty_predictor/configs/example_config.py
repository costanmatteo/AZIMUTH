"""
Configuration file for Uncertainty Quantification training

Customize the parameters below for your specific use case.
"""

CONFIG = {
    # Data configuration
    'data': {
        'csv_path': None,  # Path to your CSV file (set to None to use SCM synthetic data)
        'input_columns': ['x', 'y', 'z'],  # Input features
        'output_columns': ['res_1'],  # Target outputs
        'scaling_method': 'standard',  # 'standard', 'minmax'
        'train_size': 0.8,  # 70% for training
        'val_size': 0.01,   # 15% for validation
        'test_size': 0.19,  # 15% for testing
        'random_state': 42,

        # SCM synthetic data generation (used if csv_path is None)
        'use_scm': True,  # Enable SCM data generation
        'scm': {
            'n_samples': 2000,  # Number of samples to generate
            'seed': 42,  # Random seed for reproducibility
            # Type of SCM dataset: 'one_to_one_ct', 'laser', 'plasma', 'galvanic', 'microetch'
            # Or 'all' for multi-process training (requires conditioning.enable=True)
            'dataset_type': 'microetch',
            # Multi-process configuration (used when conditioning is enabled)
            'process_selection': 'microetch',  # 'all', 'laser', 'plasma', 'galvanic', 'microetch'
            'add_env_vars': True  # Add environmental variables and timestamps
        }
    },

    # Model configuration
    'model': {
        'model_type': 'custom',  # 'small', 'medium', 'large', or 'custom'
        # If 'custom', specify architecture below:
        'hidden_sizes': [64, 32, 16],  # Used only if model_type='custom'
        'dropout_rate': 0.1,
        'use_batchnorm': True,
        'min_variance': 1e-6  # Minimum variance for numerical stability
    },

    # Training configuration
    'training': {
        'batch_size': 64,
        'epochs': 400,
        'learning_rate': 0.001,
        'weight_decay': 0.001,  # L2 regularization
        'patience': 30,  # Early stopping patience
        'device': 'auto',  # 'auto', 'cuda', or 'cpu'
        'checkpoint_dir': 'checkpoints_uncertainty',

        # Loss function type: 'gaussian_nll' or 'energy_score'
        'loss_type': 'gaussian_nll',  # Choose between 'gaussian_nll' or 'energy_score'

        # Gaussian NLL parameters (used only if loss_type='gaussian_nll')
        # Variance penalty weight in loss function: L = (y-μ)²/σ² + α*log(σ²)
        # α = 1.0: Standard Gaussian NLL
        # α < 1.0: Reduces penalty for large variances (recommended for over-confident models)
        # α > 1.0: Increases penalty for large variances
        'variance_penalty_alpha': 0.1,  

        # Energy Score parameters (used only if loss_type='energy_score')
        # Number of Monte Carlo samples for Energy Score computation
        # Higher values = more accurate but slower. Recommended: 50-200
        'energy_score_samples': 30,
        # β parameter: controls diversity penalty in ES = E[|X-y|] - β/2*E[|X-X'|]
        # β = 1.0: Standard Energy Score (recommended)
        # β < 1.0: Less penalty for diverse predictions (allows wider uncertainty)
        # β > 1.0: More penalty for diverse predictions (encourages tighter uncertainty)
        'energy_score_beta': 0.1
    },

    # Uncertainty configuration
    'uncertainty': {
        'confidence_level': 0.99,  # For prediction intervals (0.95, 0.99, etc.)
    },

    # Conditional embedding configuration (for multi-process training)
    # Set enable=True to train a single model on all 4 PCB processes simultaneously
    'conditioning': {
        'enable': False,  # Set to True to enable conditional embeddings

        # Process embedding configuration
        'num_processes': 4,  # Number of processes (Laser, Plasma, Galvanic, Microetch)
        'd_proc': 16,  # Process ID embedding dimension

        # Continuous environment variables
        'env_continuous': ['ambient_temp', 'humidity'],
        'd_env_float': 16,  # Embedding dimension for each continuous variable
        'use_missing_mask': True,  # Use missing value masks

        # Categorical environment variables
        'env_categorical': {
            'batch_id': 10,      # 10 possible batch IDs
            'operator_id': 5,    # 5 possible operators
            'shift': 3           # 3 shifts (morning, afternoon, night)
        },
        'd_env_cat_base': 1.6,  # Base multiplier for categorical embedding dimension

        # Temporal encoding
        'use_time': True,
        'time_column': 'timestamp',
        'time_periods': 4,  # Number of periodic components in Time2Vec
        'd_time': 8,  # Temporal encoding dimension

        # Context fusion
        'd_context': 64,  # Unified context vector dimension
        'context_mlp_hidden': [128, 64],  # Hidden layers for context fusion MLP
        'context_dropout': 0.1,  # Dropout in context MLP

        # Normalization type
        'norm_type': 'conditional_layer_norm'  # 'conditional_layer_norm', 'conditional_batch_norm', or 'layer_norm'
    },

    # Miscellaneous
    'misc': {
        'random_seed': 42,
        'verbose': True
    }
}

# =============================================================================
# EXAMPLE CONFIGURATIONS FOR DIFFERENT USE CASES
# =============================================================================

# To enable conditional multi-process training, set:
# CONFIG['conditioning']['enable'] = True
# CONFIG['data']['scm']['process_selection'] = 'all'
# CONFIG['data']['scm']['n_samples'] = 500  # samples per process (total: 2000)

# Minimal conditioning config (no env vars, no time):
# 'conditioning': {
#     'enable': True,
#     'num_processes': 4,
#     'd_proc': 16,
#     'env_continuous': [],
#     'env_categorical': {},
#     'use_time': False,
#     'd_context': 32,
#     'context_mlp_hidden': [64, 32],
#     'norm_type': 'conditional_layer_norm'
# }









