"""
Configuration for CasualiT surrogate training.

Multi-scenario support similar to controller_optimization.

Key Parameters:
===============

PROCESS SELECTION:
- process_names: Controls which processes to include (same as controller_config)
  * None: Uses ALL processes defined in PROCESSES
  * ['laser', 'plasma', ...]: Uses specified processes in order

SCENARIOS:
- n_train: Number of training scenarios (diverse operating conditions)
- n_test: Number of test scenarios for evaluation
- n_trajectories_per_scenario: How many trajectories to generate per scenario
- seed: Random seed for reproducibility

SAMPLING:
- strategy: How to sample controllable parameters
  * 'uniform': Uniform random within boundaries
  * 'latin_hypercube': LHS sampling for better coverage
- use_scm_boundaries: Use boundaries from scm_ds/datasets.py

MODEL:
- sequence_length: Number of processes (typically 4)
- features_per_process: Computed from input_dim + output_dim per process

TRAINING:
- batch_size, learning_rate, max_epochs, patience: Standard training params
- loss_fn: Loss function ('mse' for regression)
"""

SURROGATE_CONFIG = {
    # Process selection (same format as controller_config)
    'process_names': ['laser', 'plasma', 'galvanic', 'microetch'],

    # Scenario generation (configurable like controller optimization)
    'scenarios': {
        'n_train': 50,                    # Training scenarios
        'n_test': 10,                     # Test scenarios
        'n_trajectories_per_scenario': 200,  # Trajectories per scenario
        'seed': 42,
        'seed_offset_test': 1000,         # Offset for test seeds
    },

    # Controllable parameter sampling
    'sampling': {
        'strategy': 'uniform',            # 'uniform' or 'latin_hypercube'
        'use_scm_boundaries': True,       # Get bounds from scm_ds
    },

    # CasualiT model configuration
    'model': {
        # Architecture (encoder-decoder transformer)
        'embed_dim': 64,                  # Embedding dimension
        'n_heads': 4,                     # Attention heads
        'n_encoder_layers': 2,            # Encoder layers
        'n_decoder_layers': 2,            # Decoder layers
        'ff_dim': 128,                    # Feed-forward dimension
        'dropout': 0.1,

        # Input/output configuration
        'sequence_length': 4,             # Number of processes
        # features_per_process computed dynamically from process configs
    },

    # Training configuration
    'training': {
        'batch_size': 64,
        'max_epochs': 200,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'patience': 30,                   # Early stopping patience
        'loss_fn': 'mse',                 # Mean squared error for regression
        'checkpoint_dir': 'casualit_surrogate/checkpoints',

        # Optimizer
        'optimizer': 'adamw',

        # Learning rate scheduler
        'lr_scheduler': {
            'type': 'cosine',
            'T_max': 200,
            'eta_min': 1e-6,
        },

        # Validation
        'val_split': 0.2,                 # Validation split ratio
    },

    # Data processing
    'data': {
        'normalize_inputs': True,         # Normalize trajectory features
        'normalize_targets': False,       # F is already in [0, 1]
    },

    # Miscellaneous
    'misc': {
        'device': 'auto',                 # 'auto', 'cuda', 'cpu'
        'verbose': True,
        'save_every_n_epochs': 10,        # Save checkpoint every N epochs
    },
}


# Input boundaries for controllable parameters (from scm_ds/datasets.py)
# These define the valid ranges for random sampling
INPUT_BOUNDARIES = {
    'laser': {
        'PowerTarget': (0.10, 1.0),       # Controllable
        'AmbientTemp': (15.0, 35.0),      # Environmental/structural
    },
    'plasma': {
        'RF_Power': (100.0, 400.0),       # Controllable
        'Duration': (10.0, 60.0),         # Controllable
    },
    'galvanic': {
        'CurrentDensity': (1.0, 5.0),     # Controllable
        'Duration': (600.0, 3600.0),      # Controllable
    },
    'microetch': {
        'Temperature': (293.0, 323.0),    # Environmental/structural
        'Concentration': (0.5, 3.0),      # Controllable
        'Duration': (30.0, 180.0),        # Controllable
    },
}

# Which inputs are controllable vs environmental (from processes_config.py)
CONTROLLABLE_INPUTS = {
    'laser': ['PowerTarget'],
    'plasma': ['RF_Power'],
    'galvanic': ['CurrentDensity', 'Duration'],
    'microetch': ['Concentration', 'Duration'],
}

# Environmental/structural inputs (sampled differently for scenarios)
STRUCTURAL_INPUTS = {
    'laser': ['AmbientTemp'],
    'plasma': ['Duration'],              # Typically set per scenario
    'galvanic': [],
    'microetch': ['Temperature'],
}
