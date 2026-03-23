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
        'hidden_sizes': [256, 128, 64, 32],  # Used only if model_type='custom'
        'dropout_rate': 0.05,
        'use_batchnorm': True,
        'min_variance': 1e-6,  # Minimum variance for numerical stability

        # ═══════════════════════════════════════════════════════════════════
        # UNCERTAINTY QUANTIFICATION METHOD
        # ═══════════════════════════════════════════════════════════════════
        # Choose ONE method for uncertainty quantification:
        #   - 'single': Single model, only aleatoric uncertainty
        #   - 'ensemble': Deep Ensemble, aleatoric + epistemic (trains N models)
        #   - 'swag': SWAG (SWA-Gaussian), aleatoric + epistemic (single training)
        #
        # Comparison:
        #   | Method   | Training Cost | Epistemic | Accuracy |
        #   |----------|---------------|-----------|----------|
        #   | single   | 1x            | No        | Good     |
        #   | ensemble | Nx            | Yes       | Best     |
        #   | swag     | ~1.25x        | Yes       | Good     |
        # ═══════════════════════════════════════════════════════════════════
        'uncertainty_method': 'swag',  # 'single', 'ensemble', or 'swag'

        # ─────────────────────────────────────────────────────────────────────
        # Deep Ensemble configuration (used if uncertainty_method='ensemble')
        # ─────────────────────────────────────────────────────────────────────
        # Enable ensemble mode for better uncertainty quantification
        # Ensemble trains N independent models and combines their predictions
        'use_ensemble': False,  # DEPRECATED: use uncertainty_method='ensemble' instead

        # Number of models in the ensemble (recommended: 5)
        # More models = better uncertainty estimates but longer training
        # Diminishing returns beyond 5-10 models
        'n_ensemble_models': 5,

        # Base seed for ensemble (each model uses base_seed + model_idx * 1000)
        'ensemble_base_seed': 42,

        # ─────────────────────────────────────────────────────────────────────
        # SWAG configuration (used if uncertainty_method='swag')
        # ─────────────────────────────────────────────────────────────────────
        # SWAG approximates the posterior over weights using a Gaussian with
        # low-rank + diagonal covariance. Much cheaper than ensemble.
        #
        # Reference: Maddox et al. (2019) "A Simple Baseline for Bayesian
        # Uncertainty in Deep Learning" https://arxiv.org/abs/1902.02476

        # Fraction of training to complete before starting SWA collection
        # 0.5 = start collecting weights at 50% of training
        'swag_start_epoch': 0.6,

        # Learning rate during SWA phase (typically higher than final LR)
        # Higher LR helps explore the posterior better
        'swag_learning_rate': 0.005,

        # Maximum rank for low-rank covariance approximation
        # Higher = more accurate but more memory. 20-30 is usually sufficient
        'swag_max_rank': 20,

        # Collect weights every N epochs during SWA phase
        'swag_collection_freq': 1,

        # Number of weight samples for prediction (more = slower but smoother)
        'swag_n_samples': 30,

        # Minimum number of weight samples to collect before allowing training to stop
        # If early stopping triggers before collecting this many samples, training
        # continues until this minimum is reached. Set to 0 to disable this check.
        'swag_min_samples': 20
    },

    # Training configuration
    'training': {
        'batch_size': 32,
        'epochs': 500,
        'learning_rate': 0.001,
        'weight_decay': 0.0005,  # L2 regularization
        'patience': 50,  # Early stopping patience
        'device': 'auto',  # 'auto', 'cuda', or 'cpu'
        'checkpoint_dir': 'checkpoints_uncertainty',

        # Loss function type: 'gaussian_nll' or 'energy_score'
        'loss_type': 'gaussian_nll',  # Choose between 'gaussian_nll' or 'energy_score'

        # Gaussian NLL parameters (used only if loss_type='gaussian_nll')
        # Variance penalty weight in loss function: L = (y-μ)²/σ² + α*log(σ²)
        # α = 1.0: Standard Gaussian NLL
        # α < 1.0: Reduces penalty for large variances (recommended for over-confident models)
        # α > 1.0: Increases penalty for large variances
        'variance_penalty_alpha': 1.5,

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
        'confidence_level': 0.95,  # For prediction intervals (0.95, 0.99, etc.)
    },

    # Miscellaneous
    'misc': {
        'random_seed': 42,
        'verbose': True
    }
}









