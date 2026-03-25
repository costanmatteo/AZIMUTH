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
    # Data generation (minimal for pipeline testing)
    'data': {
        'n_trajectories': 200,         # Minimal for pipeline test
        'n_val_trajectories': 50,      # Minimal validation
        'n_test_trajectories': 50,     # Minimal test
        'batch_size_generation': 200,  # Generate all in one batch
        'random_seed': 42,

        # Scenario diversity
        'n_scenarios': 2,              # Just 2 scenarios for speed
        'scenario_seed_offset': 1000,

        # Process selection
        'process_names': ['laser', 'plasma', 'galvanic', 'microetch'],
    },

    # Model architecture (tiny)
    'model': {
        'd_model_enc': 16,
        'd_model_dec': 8,
        'd_ff': 32,
        'd_qk': 8,

        'e_layers': 1,
        'd_layers': 1,
        'n_heads': 2,

        'dropout_emb': 0.0,
        'dropout_attn_out': 0.0,
        'dropout_ff': 0.0,

        'activation': 'gelu',
        'norm': 'batch',
        'use_final_norm': True,
    },

    # Training (fast)
    'training': {
        'max_epochs': 20,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'weight_decay': 0.01,
        'loss_fn': 'mse',
        'k_fold': 1,
        'seed': 42,
        'patience': 10,

        'use_scheduler': False,
        'scheduler_factor': 0.5,
        'scheduler_patience': 10,

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
