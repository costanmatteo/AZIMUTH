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
            'dataset_type': 'microetch'
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

    # Conditioning configuration (for multi-process and environmental adaptation)
    'conditioning': {
        # Process ID embedding (e.g., 4 PCB manufacturing processes)
        'num_processes': 4,  # Number of different processes
        'd_proc': 16,  # Dimension of process embedding

        # Context vector dimension (final fused representation)
        'd_context': 64,  # Dimension of context vector passed to conditional norms

        # Environmental features - CONTINUOUS
        'env_continuous': {
            'enabled': True,  # Enable continuous environmental features
            'features': ['temperature', 'humidity', 'load_factor'],  # List of feature names
            'd_env_float': 32,  # Projection dimension for continuous features
            'handle_missing': True,  # Enable missing value handling with mask
        },

        # Environmental features - CATEGORICAL
        'env_categorical': {
            'enabled': True,  # Enable categorical environmental features
            'features': {
                # Format: 'feature_name': cardinality
                'batch_id': 50,      # e.g., 50 different batches
                'operator_id': 10,   # e.g., 10 different operators
                'shift': 3,          # e.g., 3 shifts (morning/afternoon/night)
            },
            'd_embedding_rule': 'sqrt',  # Embedding size rule: 'sqrt' (1.6*sqrt(card)) or 'fixed'
            'd_embedding_fixed': 16,     # Used if d_embedding_rule='fixed'
        },

        # Temporal encoding (timestamp → learned representation)
        'time_encoding': {
            'enabled': True,  # Enable temporal encoding
            'method': 'time2vec',  # 'time2vec' or 'sincos'
            'd_time': 16,  # Dimension of time encoding
            'num_periods': 3,  # Number of periodic components (for Time2Vec)
        },

        # Conditional Normalization (replaces standard BatchNorm with conditional variant)
        'use_conditional_norm': True,  # Use Conditional LayerNorm/BatchNorm instead of standard
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

    # Miscellaneous
    'misc': {
        'random_seed': 42,
        'verbose': True
    }
}









