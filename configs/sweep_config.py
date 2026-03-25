"""
Configuration for hyperparameter sweep.

This file defines the sweep strategy and the search space for each parameter.
It supports both grid search and Optuna-based Bayesian optimization.

SWEEP MODES:
- 'grid': Cartesian product of all parameter lists (exhaustive search)
- 'optuna': Bayesian optimization via Optuna (efficient search)

PARAMETER TYPES (used by Optuna):
- 'categorical': Discrete choices from a list          → values: [v1, v2, ...]
- 'float': Continuous range                             → min, max, (optional: log=True)
- 'int': Integer range                                  → min, max, (optional: step)

For grid mode, only 'values' is used (list of values to sweep over).

Usage:
    from configs.sweep_config import SWEEP_CONFIG

    # Access sweep mode
    mode = SWEEP_CONFIG['sweep']['mode']

    # Access search space
    space = SWEEP_CONFIG['search_space']

    # Access Optuna settings
    optuna_cfg = SWEEP_CONFIG['optuna']
"""

SWEEP_CONFIG = {
    # =========================================================================
    # SWEEP STRATEGY
    # =========================================================================
    'sweep': {
        # 'grid' = exhaustive cartesian product of all values
        # 'optuna' = Bayesian optimization (Optuna)
        'mode': 'optuna',

        # Target to optimize: which config module to sweep
        # 'controller' → sweeps over CONTROLLER_CONFIG parameters
        # 'surrogate'  → sweeps over SURROGATE_CONFIG parameters
        'target': 'controller',

        # Number of random seeds per configuration (for statistical robustness)
        'n_seeds': 1,
        'seed_start': 42,
    },

    # =========================================================================
    # SEARCH SPACE
    # =========================================================================
    # Each entry maps to a parameter path in the target config.
    # Nested keys use dot notation: 'training.learning_rate' → config['training']['learning_rate']
    #
    # Format per parameter:
    #   'type':   'categorical' | 'float' | 'int'
    #   'values': [v1, v2, ...]           (used by grid mode, and categorical in optuna)
    #   'min':    lower bound             (used by float/int in optuna)
    #   'max':    upper bound             (used by float/int in optuna)
    #   'log':    True/False              (log-uniform sampling, float only)
    #   'step':   int                     (discrete step, int only)
    #
    # To disable a parameter from the sweep, comment it out or set 'enabled': False.
    # =========================================================================
    'search_space': {

        # --- Training parameters ---
        'training.learning_rate': {
            'type': 'float',
            'values': [1e-4, 5e-4, 1e-3, 5e-3],  # grid mode
            'min': 1e-5,                            # optuna mode
            'max': 1e-2,
            'log': True,
        },
        'training.weight_decay': {
            'type': 'float',
            'values': [0.001, 0.01, 0.05],
            'min': 1e-4,
            'max': 0.1,
            'log': True,
        },
        'training.batch_size': {
            'type': 'categorical',
            'values': [1000, 2000, 3000, 5000],
        },
        'training.epochs': {
            'type': 'categorical',
            'values': [100, 200, 500],
        },
        'training.lambda_bc': {
            'type': 'float',
            'values': [0.0, 0.001, 0.01, 0.1],
            'min': 0.0,
            'max': 0.5,
        },
        'training.reliability_loss_scale': {
            'type': 'categorical',
            'values': [100.0, 500.0, 1000.0],
        },
        'training.gradient_clip_norm': {
            'type': 'categorical',
            'values': [None, 1.0, 3.0, 5.0],
        },
        'training.optimizer': {
            'type': 'categorical',
            'values': ['adam', 'adamw'],
        },

        # --- Policy generator architecture ---
        'policy_generator.hidden_sizes': {
            'type': 'categorical',
            'values': [
                [32, 16],
                [64, 32],
                [128, 64],
                [128, 64, 32],
                [256, 128, 64],
            ],
        },
        'policy_generator.dropout': {
            'type': 'float',
            'values': [0.0, 0.05, 0.1, 0.2, 0.4],
            'min': 0.0,
            'max': 0.4,
        },
        'policy_generator.use_batchnorm': {
            'type': 'categorical',
            'values': [True, False],
        },
        'policy_generator.use_scenario_encoder': {
            'type': 'categorical',
            'values': [True, False],
        },
        'policy_generator.scenario_embedding_dim': {
            'type': 'int',
            'values': [8, 16, 32, 64],
            'min': 8,
            'max': 64,
            'step': 8,
        },

        # --- Curriculum learning ---
        'training.curriculum_learning.enabled': {
            'type': 'categorical',
            'values': [True, False],
        },
        'training.curriculum_learning.decay_speed': {
            'type': 'float',
            'values': [1.0, 2.0, 3.0, 5.0],
            'min': 1.0,
            'max': 5.0,
        },
        'training.curriculum_learning.reliability_weight_curve': {
            'type': 'categorical',
            'values': ['exponential', 'linear', 'sigmoid'],
        },

        # --- Scenario settings ---
        'scenarios.n_train': {
            'type': 'categorical',
            'values': [1, 5, 10, 20, 50],
        },
        'scenarios.seed_target': {
            'type': 'int',
            'values': [1, 10, 25, 42, 50, 75, 100],
            'min': 1,
            'max': 100,
        },
        'scenarios.seed_baseline': {
            'type': 'int',
            'values': [1, 10, 25, 42, 50, 75, 100],
            'min': 1,
            'max': 100,
        },

        # --- Surrogate model parameters (used when target='surrogate') ---
        'model.d_model_enc': {
            'type': 'categorical',
            'values': [16, 32, 64, 128],
        },
        'model.d_model_dec': {
            'type': 'categorical',
            'values': [8, 16, 32, 64],
        },
        'model.d_ff': {
            'type': 'categorical',
            'values': [32, 64, 128, 256],
        },
        'model.n_heads': {
            'type': 'categorical',
            'values': [1, 2, 4],
        },
        'model.e_layers': {
            'type': 'categorical',
            'values': [1, 2, 3],
        },
        'model.dropout_emb': {
            'type': 'float',
            'values': [0.0, 0.1, 0.2, 0.3],
            'min': 0.0,
            'max': 0.5,
        },
        'training.loss_weight_x': {
            'type': 'float',
            'values': [0.1, 0.3, 0.5, 1.0],
            'min': 0.0,
            'max': 1.0,
        },
        'training.loss_weight_y': {
            'type': 'float',
            'values': [0.5, 1.0, 2.0],
            'min': 0.5,
            'max': 2.0,
        },

        # --- LR Scheduler ---
        'training.lr_scheduler.type': {
            'type': 'categorical',
            'values': ['step', 'cosine', 'reduce_on_plateau'],
        },
    },

    # =========================================================================
    # OPTUNA SETTINGS (used only when mode='optuna')
    # =========================================================================
    'optuna': {
        'n_trials': 100,                # Total number of trials
        'study_name': 'azimuth_sweep',  # Study name (for distributed execution)
        'direction': 'maximize',        # 'maximize' (F reliability) or 'minimize' (loss)
        'sampler': 'tpe',               # 'tpe' (Tree-structured Parzen Estimator), 'random', 'cmaes'
        'pruner': 'median',             # 'median', 'hyperband', None (no pruning)
        'storage': None,                # None = in-memory, or 'sqlite:///optuna_sweep.db' for persistence
        'reduced_epochs': None,         # If set, use fewer epochs per trial (faster but less accurate)
        'timeout': None,                # Max seconds for entire study (None = no limit)
    },

    # =========================================================================
    # GRID SETTINGS (used only when mode='grid')
    # =========================================================================
    'grid': {
        # Maximum number of combinations to run (safety limit for large grids)
        'max_combinations': 500,
        # Run combinations in random order (useful if stopping early)
        'shuffle': True,
    },

    # =========================================================================
    # EXECUTION
    # =========================================================================
    'execution': {
        'device': 'auto',              # 'auto', 'cuda', 'cpu'
        'n_parallel_jobs': 1,           # Number of parallel trials (1 = sequential)
        'verbose': False,               # Print training progress for each trial
        'save_all_results': True,       # Save results for every trial (not just best)
        'results_dir': 'sweep_results', # Directory for sweep outputs
    },

    # =========================================================================
    # LOGGING & CHECKPOINTS
    # =========================================================================
    'logging': {
        'log_to_file': True,
        'log_file': 'sweep_results/sweep_log.txt',
        'save_best_config': True,       # Save the best config as a standalone file
        'save_top_k': 5,                # Save top K configurations
    },
}
