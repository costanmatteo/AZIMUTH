"""
Configuration for controller training with multi-scenario support.

Key Parameters Guide:
===================

TRAINING:
- epochs: Number of training epochs (each epoch cycles through all scenarios once)
- batch_size: Number of samples per scenario (different process noise realizations)
- learning_rate: Initial learning rate for optimizer
- lambda_bc: Behavior cloning weight (balances reliability vs target-following)
- patience: Early stopping patience (epochs without improvement)
- gradient_clip_norm: Max gradient norm (None=no clipping, 1.0=typical value)
- lr_scheduler: Learning rate decay schedule (None or dict with 'type', 'step_size', 'gamma')
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

TARGET/BASELINE:
- n_samples: Number of scenarios (50 = diverse operating conditions)
- seed: Random seed for reproducibility

MULTI-SCENARIO:
- shuffle_order: Shuffle scenario order each epoch (recommended: True)
- warmup_epochs: Train on single scenario first (0 = disabled)
- scenario_weights: Custom weights per scenario (None = equal coverage)

MISC:
- random_seed: Global random seed
- verbose: Print training progress
- save_intermediate_results: Save per-scenario metrics during training
- log_per_scenario_metrics: Log individual scenario F values (very verbose)

Typical Modifications:
=====================
- Increase batch_size (64, 128) for smoother gradients (needs more GPU memory)
- Add gradient clipping (1.0) if training is unstable
- Enable lr_scheduler for longer training (e.g., {'type': 'step', 'step_size': 50, 'gamma': 0.5})
- Increase n_samples (100, 200) for even more diverse scenarios
- Enable eval_all_scenarios_every (10) to monitor per-scenario progress during training
"""

CONTROLLER_CONFIG = {
    # Processi da includere (presi da PROCESSES)
    'process_names': ['laser', 'plasma', 'galvanic', 'microetch'],

    # Policy generator architecture
    'policy_generator': {
        'architecture': 'custom',  # 'small', 'medium', 'large', 'custom'
        'hidden_sizes': [32, 16],  # Usato solo se 'custom'
        'dropout': 0.1,
        'use_batchnorm': False,
    },

    # Training parameters
    'training': {
        'epochs': 200,  # Increased from 100 to maintain total batches (200 epochs × 50 scenarios = 10,000 batches)
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 0.001,
        'lambda_bc': 0.01,  # Behavior cloning weight
        'patience': 30,
        'device': 'auto',
        'checkpoint_dir': 'controller_optimization/checkpoints/controller',

        # Optimizer settings
        'optimizer': 'adam',  # 'adam', 'adamw', 'sgd'
        'momentum': 0.9,  # For SGD
        'beta1': 0.9,  # For Adam/AdamW
        'beta2': 0.999,  # For Adam/AdamW

        # Gradient clipping (helps with training stability)
        'gradient_clip_norm': None,  # None = no clipping, or float (e.g., 1.0)

        # Learning rate scheduler
        'lr_scheduler': None,  # None, or {'type': 'step', 'step_size': 30, 'gamma': 0.1}
                               # Options: 'step', 'exponential', 'cosine', 'reduce_on_plateau'

        # Early stopping
        'early_stopping_metric': 'F',  # 'F' (maximize) or 'loss' (minimize)
        'early_stopping_delta': 0.0,  # Minimum improvement to count as better

        # Evaluation frequency
        'eval_all_scenarios_every': None,  # None = only at end, or int (e.g., 10 = every 10 epochs)
    },

    # Target trajectory
    'target': {
        'n_samples': 1,  # Multi-scenario training for generalization
        'seed': 42,
    },

    # Baseline trajectory (per comparison)
    'baseline': {
        'n_samples': 1,  # Must match target for structural alignment
        'seed': 43,  # Diverso seed per noise diverso
    },

    # Multi-scenario training
    'multi_scenario': {
        'shuffle_order': True,  # Shuffle scenario order each epoch (recommended)
        'warmup_epochs': 0,  # Train on single scenario for first N epochs (0 = disabled)
        'scenario_weights': None,  # None = equal weights, or dict {scenario_idx: weight} for weighted sampling
    },

    # Report generation
    'report': {
        'generate_pdf': True,
        'include_plots': True,
    },

    # Miscellaneous
    'misc': {
        'random_seed': 42,
        'verbose': True,
        'save_intermediate_results': False,  # Save per-scenario results during training
        'log_per_scenario_metrics': False,  # Log individual scenario F values during training (verbose)
    }
}
