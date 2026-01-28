"""
Configuration for CasualiT surrogate training.

This trains a TransformerForecaster to predict reliability F from process trajectories.
The model learns to map:
    X (trajectory: inputs + outputs for each process) -> Y (reliability F)

Data Generation:
- Trajectories are generated using random controllable parameters within uncertainty predictor bounds
- F is computed using the mathematical formula (ProTSurrogate)
- Multi-scenario support for diverse training conditions

Model Architecture:
- Uses simplified ProT architecture for sequence-to-scalar prediction
- Encoder processes the trajectory sequence (each process = one sequence step)
- Decoder predicts F as a single scalar output
"""

SURROGATE_CONFIG = {
    # Data generation
    'data': {
        'n_trajectories': 10000,       # Number of training trajectories
        'n_val_trajectories': 2000,    # Validation trajectories
        'n_test_trajectories': 2000,   # Test trajectories
        'batch_size_generation': 100,  # Batch size for trajectory generation
        'random_seed': 42,

        # Scenario diversity
        'n_scenarios': 50,             # Number of different scenario conditions
        'scenario_seed_offset': 1000,  # Seed offset between scenarios

        # Process selection
        'process_names': ['laser', 'plasma', 'galvanic', 'microetch'],
    },

    # Model architecture
    'model': {
        # Embedding dimensions
        'd_model_enc': 64,             # Encoder model dimension
        'd_model_dec': 32,             # Decoder model dimension
        'd_ff': 128,                   # Feed-forward dimension
        'd_qk': 32,                    # Query/key dimension

        # Transformer layers
        'e_layers': 2,                 # Encoder layers
        'd_layers': 1,                 # Decoder layers
        'n_heads': 4,                  # Attention heads

        # Regularization
        'dropout_emb': 0.1,
        'dropout_attn_out': 0.1,
        'dropout_ff': 0.1,

        # Architecture options
        'activation': 'gelu',
        'norm': 'batch',
        'use_final_norm': True,
    },

    # Training
    'training': {
        'max_epochs': 500,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'weight_decay': 0.01,
        'loss_fn': 'mse',
        'k_fold': 3,                   # K-fold cross validation
        'seed': 42,
        'patience': 50,                # Early stopping patience

        # Scheduler
        'use_scheduler': True,
        'scheduler_factor': 0.5,
        'scheduler_patience': 10,

        # Entropy regularization
        'entropy_regularizer': False,
        'gamma': 0.05,
    },

    # Checkpoints
    'checkpoints': {
        'save_dir': 'causaliT/checkpoints/surrogate',
        'save_every_n_epochs': 50,
        'keep_top_k': 3,               # Keep top k checkpoints
    },

    # Report
    'report': {
        'generate_pdf': True,
        'output_dir': 'causaliT/reports/surrogate',
        'include_plots': True,
        'plot_dpi': 150,
    },

    # Logging
    'logging': {
        'log_every_n_steps': 10,
        'verbose': True,
    },
}
