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
        'train_size': 0.7,  # 70% for training
        'val_size': 0.15,   # 15% for validation
        'test_size': 0.15,  # 15% for testing
        'random_state': 42,

        # SCM synthetic data generation (used if csv_path is None)
        'use_scm': True,  # Enable SCM data generation
        'scm': {
            'n_samples': 1000,  # Number of samples to generate
            'seed': 42,  # Random seed for reproducibility
            'dataset_type': 'laser'  # Type of SCM dataset, either 'one_to_one_ct' or 'laser'
        }
    },

    # Model configuration
    'model': {
        'model_type': 'custom',  # 'small', 'medium', 'large', or 'custom'
        # If 'custom', specify architecture below:
        'hidden_sizes': [64, 32],  # Used only if model_type='custom'
        'dropout_rate': 0.1,
        'use_batchnorm': True,
        'min_variance': 1e-6,  # Minimum variance for numerical stability
        # Variance bounds to prevent numerical overflow
        'max_log_variance': 10.0,  # Maximum log(variance), max_variance ≈ 22026
        'min_log_variance': -10.0  # Minimum log(variance), min_variance ≈ 0.000045
    },

    # Training configuration
    'training': {
        'batch_size': 32,
        'epochs': 400,
        'learning_rate': 0.001,
        'weight_decay': 0.001,  # L2 regularization
        'patience': 30,  # Early stopping patience
        'device': 'auto',  # 'auto', 'cuda', or 'cpu'
        'checkpoint_dir': 'checkpoints_uncertainty',

        # Loss function type: 'gaussian_nll' or 'energy_score'
        'loss_type': 'energy_score',  # Choose between 'gaussian_nll' or 'energy_score'

        # Gaussian NLL parameters (used only if loss_type='gaussian_nll')
        # Variance penalty weight in loss function: L = (y-μ)²/σ² + α*log(σ²)
        # α = 1.0: Standard Gaussian NLL
        # α < 1.0: Reduces penalty for large variances (recommended for over-confident models)
        # α > 1.0: Increases penalty for large variances
        'variance_penalty_alpha': 5,  

        # Energy Score parameters (used only if loss_type='energy_score')
        # Number of Monte Carlo samples for Energy Score computation
        # Higher values = more accurate but slower. Recommended: 50-200
        'energy_score_samples': 30,
        # β parameter: controls diversity penalty in ES = E[|X-y|] - β/2*E[|X-X'|]
        # β = 1.0: Standard Energy Score (recommended)
        # β < 1.0: Less penalty for diverse predictions (allows wider uncertainty)
        # β > 1.0: More penalty for diverse predictions (encourages tighter uncertainty)
        'energy_score_beta': 1
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









