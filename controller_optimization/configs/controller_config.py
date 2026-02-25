"""
Configuration for controller training with multi-scenario support.

Key Parameters Guide:
===================

PROCESS SELECTION:
- process_names: Controls which processes to include in training
  * None (or omitted): Uses ALL processes defined in PROCESSES
  * ['laser', 'plasma']: Uses only laser and plasma (in that order)
  * ['laser', 'plasma', 'galvanic', 'microetch']: Uses all 4 processes
  NOTE: The order matters! Processes are chained in the order specified.
        Example: ['plasma', 'laser'] would put plasma BEFORE laser (unusual!)
  VALIDATION: Will raise ValueError if a specified process doesn't exist in PROCESSES

TRAINING:
- epochs: Number of training epochs (each epoch cycles through all training scenarios once)
- batch_size: Total number of samples per epoch, split equally across scenarios
              (samples_per_scenario = batch_size // n_scenarios)
              Higher values = smoother gradients but more memory
- learning_rate: Initial learning rate for optimizer
- lambda_bc: Behavior cloning weight (balances reliability vs target-following)
- reliability_loss_scale: Scale factor for reliability loss (F - F*)^2
                         Prevents vanishing gradients when delta F is small (~0.1)
                         Typical values: 100.0 (default), 1000.0 for very small deltas
- patience: Early stopping patience (epochs without improvement)
- gradient_clip_norm: Max gradient norm (None=no clipping, 1.0=typical value)
- lr_scheduler: Learning rate decay schedule (None or dict). Supported types:
                * 'step': {'type': 'step', 'step_size': 50, 'gamma': 0.5}
                          LR = LR * gamma every step_size epochs
                * 'exponential': {'type': 'exponential', 'gamma': 0.99}
                          LR = LR * gamma every epoch
                * 'cosine': {'type': 'cosine', 'T_max': 1500, 'eta_min': 0}
                          Cosine annealing from initial LR to eta_min over T_max epochs
                * 'reduce_on_plateau': {'type': 'reduce_on_plateau', 'factor': 0.5, 'patience': 10}
                          Reduce LR by factor when loss stops improving for patience epochs
- early_stopping_metric: 'F' (reliability, maximize) or 'loss' (minimize)
- eval_all_scenarios_every: Evaluate on all scenarios every N epochs (None=only at end)

OPTIMIZER:
- optimizer: 'adam' (default), 'adamw', or 'sgd'
- momentum: For SGD (typically 0.9)
- beta1, beta2: For Adam/AdamW (default 0.9, 0.999)

POLICY GENERATOR:
- architecture: 'small', 'medium', 'large', or 'custom'
- hidden_sizes: Layer sizes for 'custom' architecture
- dropout: Dropout rate for regularization
- use_batchnorm: Enable batch normalization
- use_scenario_encoder: Enable scenario context encoding (structural parameters → embedding)
                       Allows policy to adapt to different operating conditions
                       Recommended: True (default)
- scenario_embedding_dim: Dimension of scenario embedding vector (default: 16)
                         Higher = more expressive but more parameters

SCENARIOS:
- n_train: Number of scenarios for training (diverse operating conditions)
- n_test: Number of scenarios for final evaluation (never seen during training)
- seed_target: Seed for target trajectory generation (train scenarios)
- seed_baseline: Seed for baseline process noise (same inputs as target, different noise realization)
- test_seed_offset: Offset added to seeds for test scenarios (default 1000, ensures test != train)

MULTI-SCENARIO:
- shuffle_order: Shuffle scenario order each epoch (recommended: True)
- warmup_epochs: Train on single scenario first (0 = disabled)
- scenario_weights: Custom weights per scenario (None = equal coverage)

METRICS:
- success_rate_threshold: Threshold for success rate metric (0.95 = 95% of F_star)
                          Success means F_actual >= threshold * F_star

SURROGATE:
- type: Which surrogate to use for reliability computation
        * 'reliability_function': Mathematical formula (default)
        * 'casualit': Learned transformer model (uses causaliT/proT)
- use_deterministic_sampling: If True, uses mean values directly (deterministic)
                              If False, uses reparameterization trick (stochastic)
                              Recommended: True for stable training, False for uncertainty estimation
- casualit.checkpoint_path: Path to trained causaliT TransformerForecaster checkpoint

MISC:
- random_seed: Global random seed
- verbose: Print training progress
- save_intermediate_results: Save per-scenario metrics during training
- log_per_scenario_metrics: Log individual scenario F values (very verbose)

Typical Modifications:
=====================
- Increase batch_size for smoother gradients (needs more GPU memory)
- Add gradient clipping (1.0) if training is unstable
- Enable lr_scheduler for longer training (e.g., {'type': 'step', 'step_size': 50, 'gamma': 0.5})
- Increase n_train (80, 100) for even more diverse training scenarios
- Adjust n_test (20, 30) for more robust final evaluation
- Enable eval_all_scenarios_every (10) to monitor per-scenario progress during training
"""

CONTROLLER_CONFIG = {
    # Processi da includere (filtrati da PROCESSES in processes_config.py)
    # - None: usa tutti i processi definiti in PROCESSES
    # - ['laser', 'plasma']: usa solo laser e plasma (nell'ordine specificato)
    # - ['laser', 'plasma', 'galvanic', 'microetch']: usa tutti e 4 i processi
    'process_names': ['laser', 'plasma', 'galvanic', 'microetch'],  # All 4 processes

    # Policy generator architecture
    'policy_generator': {
        'architecture': 'custom',  # 'small', 'medium', 'large', 'custom'
        'hidden_sizes': [32, 16],  # Usato solo se 'custom'
        'dropout': 0.015735540546988648,
        'use_batchnorm': False,
        'use_scenario_encoder': False,  # Enable scenario context encoding
        'scenario_embedding_dim': 16,  # Dimension of scenario embedding vector
    },

    # Training parameters
    'training': {
        'epochs': 1500,  # Each epoch cycles through all training scenarios once
        'batch_size': 3000,  # Total samples per epoch (split equally across scenarios)
        'learning_rate': 0.0019017383571692538,
        'weight_decay': 0.001,
        'lambda_bc': 0.001,  # Behavior cloning weight
        'reliability_loss_scale': 100.0,  # Scale factor for reliability loss (F - F*)^2
        'patience': 400,
        'device': 'auto',
        'checkpoint_dir': 'controller_optimization/checkpoints/controller',

        # Optimizer settings
        'optimizer': 'adam',  # 'adam', 'adamw', 'sgd'
        'momentum': 0.9,  # For SGD
        'beta1': 0.9,  # For Adam/AdamW
        'beta2': 0.999,  # For Adam/AdamW

        # Gradient clipping (helps with training stability)
        'gradient_clip_norm': 3.0,  # None = no clipping, or float (e.g., 1.0)

        # Learning rate scheduler (see docstring for full options)
        # Examples:
        #   {'type': 'step', 'step_size': 50, 'gamma': 0.5}           - Reduce LR by 0.5 every 50 epochs
        #   {'type': 'cosine', 'T_max': 1500}                         - Cosine annealing over 1500 epochs
        #   {'type': 'reduce_on_plateau', 'factor': 0.5, 'patience': 50}  - Reduce on plateau
        'lr_scheduler': {'type': 'cosine', 'T_max': 2000} ,  # None = constant LR, or dict with scheduler config

        # Early stopping
        'early_stopping_metric': 'F',  # 'F' (maximize) or 'loss' (minimize)
        'early_stopping_delta': 0.0,  # Minimum improvement to count as better

        # Evaluation frequency
        'eval_all_scenarios_every': None,  # None = only at end, or int (e.g., 10 = every 10 epochs)

        # Curriculum Learning (gradual introduction of reliability loss)
        'curriculum_learning': {
            'enabled': True,  # Enable curriculum learning strategy
            'warmup_fraction': 0.1,  # First 10% of epochs = warm-up (BC only)
            'lambda_bc_start': 1.0,  # High BC weight during warm-up
            'lambda_bc_end': 0.01,  # Low BC weight at end of training
            'lambda_bc_end': 0.05,  # Low BC weight at end of training
            'decay_speed': 3.0,  # Speed of λ_BC decay: 1.0=normal, 2.0=2x faster, 3.0=3x faster
            'reliability_weight_curve': 'exponential',  # 'exponential', 'linear', 'sigmoid'
            'reliability_speed': 2.0,  # Speed of reliability weight increase: 1.0=normal, 2.0=2x faster
        },
    },

    # Scenario generation (train/test split)
    'scenarios': {
        'n_train': 30,        # Training scenarios (diverse operating conditions)
        'n_test': 1,          # Test scenarios (final evaluation, never seen during training)
        'seed_target': 64,     # Seed for target trajectory generation (train)
        'seed_baseline': 134,   # Seed for baseline process noise (same inputs, different noise)
        'test_seed_offset': 1000,  # Offset added to seeds for test scenarios (ensures different from train)
    },

    # Multi-scenario training
    'multi_scenario': {
        'shuffle_order': True,  # Shuffle scenario order each epoch (recommended)
        'warmup_epochs': 0,  # Train on single scenario for first N epochs (0 = disabled)
        'scenario_weights': None,  # None = equal weights, or dict {scenario_idx: weight} for weighted sampling
    },

    # Metrics
    'metrics': {
        'success_rate_threshold': 0.95,  # Threshold for success rate (F_actual >= threshold * F_star)
    },

    # Validation settings (for overfitting detection)
    'validation': {
        # Cross-scenario validation (uses separate test scenarios with different conditions)
        'cross_scenario_enabled': False,  # Enable validation on test scenarios (different conditions)

        # Within-scenario validation (splits training samples into train/val)
        'within_scenario_enabled': True,  # Enable train/val split within same scenarios
        'within_scenario_split': 0.2,  # Fraction of training samples to use as validation (0.2 = 20%)
                                        # If multiple scenarios: same number of val samples per scenario
    },

    # Surrogate model
    'surrogate': {
        # Surrogate type: 'reliability_function' (mathematical formula) or 'casualit' (learned transformer)
        'type': 'reliability_function',

        # Common settings
        'use_deterministic_sampling': False,  # True = use mean (stable), False = use sampling (stochastic)

        # CasualiT surrogate settings (used if type='casualit')
        # Requires a trained causaliT model checkpoint
        'casualit': {
            'checkpoint_path': 'causaliT/checkpoints/surrogate/best_model.ckpt',
        },
    },

    # Theoretical analysis
    'theoretical_analysis': {
        'enabled': False,  # Enable/disable theoretical L_min analysis (computation + plots)
        'use_correlation_for_L_min': True,  # Use estimated process correlations for L_min calculation
                                            # True = accounts for correlated process errors (more accurate)
                                            # False = assumes independence (ρᵢⱼ = 0, simpler)
    },

    # Report generation
    'report': {
        'generate_pdf': True,
        'include_plots': True,
        'plot_all_batch_samples': False,  # If True, plot all individual batch samples; if False, plot only aggregated values per scenario
    },

    # Miscellaneous
    'misc': {
        'random_seed': 42,
        'verbose': True,
        'save_intermediate_results': False,  # Save per-scenario results during training
        'log_per_scenario_metrics': False,  # Log individual scenario F values during training (verbose)
    }
}
