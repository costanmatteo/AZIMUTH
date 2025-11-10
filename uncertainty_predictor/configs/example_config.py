"""
Configuration file for Uncertainty Quantification training

Customize the parameters below for your specific use case.
"""

CONFIG = {
    # Data configuration
    'data': {
        'csv_path': None,  # Path to your CSV file (set to None to use SCM synthetic data)
        'input_columns': ['x', 'y', 'z'],  # Input features (used only if csv_path is provided)
        'output_columns': ['res_1'],  # Target outputs (used only if csv_path is provided)
        'scaling_method': 'standard',  # 'standard', 'minmax'
        'train_size': 0.8,  # 80% for training
        'val_size': 0.01,   # 1% for validation
        'test_size': 0.19,  # 19% for testing
        'random_state': 42,

        # SCM synthetic data generation (used if csv_path is None)
        'use_scm': True,  # Enable SCM data generation
        'scm': {
            'n_samples': 5000,  # Number of samples per process (if unified) or total (if single)
            'seed': 42,  # Random seed for reproducibility

            # Process selection mode:
            # - 'all': Generate unified dataset with all 4 processes (requires conditioning.enable=True)
            # - 'laser': Only laser drilling (process_id=0)
            # - 'plasma': Only plasma cleaning (process_id=1)
            # - 'galvanic': Only galvanic copper deposition (process_id=2)
            # - 'microetch': Only micro-etching (process_id=3)
            'process_selection': 'all',  # 'all', 'laser', 'plasma', 'galvanic', 'microetch'

            # Add environment variables (temperature, humidity, batch_id, operator_id, shift, timestamp)
            # If conditioning.enable=True, this should be True
            # If conditioning.enable=False, this can be False (backward compatible mode)
            'add_env_vars': True,
        }
    },

    # Model configuration
    'model': {
        'model_type': 'custom',  # 'small', 'medium', 'large', or 'custom'
        # If 'custom', specify architecture below:
        'hidden_sizes': [256, 128, 64, 32],  # Used only if model_type='custom'
        'dropout_rate': 0.2,
        'use_batchnorm': False,  # Deprecated when conditioning.enable=True (use conditioning.norm_type instead)
        'min_variance': 1e-6  # Minimum variance for numerical stability
    },

    # Conditional embedding and normalization configuration
    # This enables multi-process training with shared representations
    'conditioning': {
        # ============ GLOBAL ENABLE/DISABLE ============
        'enable': True,  # Master switch for conditional training
                         # If True: uses process_id + env embeddings + conditional normalization
                         # If False: standard MLP (backward compatible, requires process_selection != 'all')

        # ============ PROCESS ID EMBEDDING ============
        'num_processes': 4,  # Number of distinct PCB processes (laser, plasma, galvanic, microetch)
        'd_proc': 16,  # Embedding dimension for process_id

        # ============ CONTINUOUS ENVIRONMENT VARIABLES ============
        # List of continuous variable names to use (subset of: 'ambient_temp', 'humidity')
        # Examples:
        #   ['ambient_temp', 'humidity'] - use both temperature and humidity
        #   ['ambient_temp'] - use only temperature
        #   [] - don't use any continuous variables
        'env_continuous': ['ambient_temp', 'humidity'],
        'd_env_float': 16,  # Projection dimension for continuous env variables
        'use_missing_mask': True,  # Concatenate 0/1 mask for missing value handling

        # ============ CATEGORICAL ENVIRONMENT VARIABLES ============
        # Dictionary {variable_name: cardinality} for categorical variables
        # Available: 'batch_id' (10), 'operator_id' (5), 'shift' (3)
        # Examples:
        #   {'batch_id': 10, 'operator_id': 5, 'shift': 3} - use all categorical vars
        #   {'batch_id': 10} - use only batch_id
        #   {} - don't use any categorical variables
        'env_categorical': {
            'batch_id': 10,
            'operator_id': 5,
            'shift': 3
        },
        'd_env_cat_base': 1.6,  # Multiplier for categorical embedding size: d = min(32, round(base * sqrt(cardinality)))

        # ============ TEMPORAL ENCODING ============
        'use_time': True,  # Enable Time2Vec temporal encoding
        'time_column': 'timestamp',  # Column name for timestamp
        'time_periods': 4,  # Number of learnable sinusoidal periods in Time2Vec
        'd_time': 8,  # Projection dimension for time encoding

        # ============ CONTEXT FUSION MLP ============
        'd_context': 64,  # Final dimension of context vector (input to conditional norm layers)
        'context_mlp_hidden': [128, 64],  # Hidden layer sizes for embedding fusion MLP
        'context_dropout': 0.1,  # Dropout rate in context fusion MLP

        # ============ CONDITIONAL NORMALIZATION TYPE ============
        # Choose normalization strategy:
        #   'conditional_layer_norm': LayerNorm with context-modulated γ, β (RECOMMENDED for multi-process)
        #   'conditional_batch_norm': BatchNorm with context-modulated γ, β
        #   'layer_norm': Standard LayerNorm (no conditioning)
        #   'batch_norm': Standard BatchNorm (no conditioning)
        #   'none': No normalization
        'norm_type': 'conditional_layer_norm',
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









