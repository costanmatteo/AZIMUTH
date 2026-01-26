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
        'min_variance': 1e-6,  # Minimum variance for numerical stability

        # Deep Ensemble configuration
        # Enable ensemble mode for better uncertainty quantification
        # Ensemble trains N independent models and combines their predictions
        'use_ensemble': False,  # Set to True to use Deep Ensemble

        # Number of models in the ensemble (recommended: 5)
        # More models = better uncertainty estimates but longer training
        # Diminishing returns beyond 5-10 models
        'n_ensemble_models': 5,

        # Base seed for ensemble (each model uses base_seed + model_idx * 1000)
        'ensemble_base_seed': 42
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









