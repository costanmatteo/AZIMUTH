"""
Configurazione per training policy generators (controller).
"""

CONTROLLER_CONFIG = {
    # Processi da includere (presi da PROCESSES)
    'process_names': ['laser', 'plasma', 'galvanic', 'microetch'],

    # Policy generator architecture
    'policy_generator': {
        'architecture': 'medium',  # 'small', 'medium', 'large', 'custom'
        'hidden_sizes': [64, 32],  # Usato solo se 'custom'
        'dropout': 0.1,
        'use_batchnorm': False,
    },

    # Training parameters
    'training': {
        'epochs': 200,  # Increased from 100 to maintain total batches (200 epochs × 50 scenarios = 10,000 batches)
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'lambda_bc': 0.1,  # Behavior cloning weight
        'patience': 20,
        'device': 'auto',
        'checkpoint_dir': 'controller_optimization/checkpoints/controller',
    },

    # Target trajectory
    'target': {
        'n_samples': 50,  # Multi-scenario training for generalization
        'seed': 42,
    },

    # Baseline trajectory (per comparison)
    'baseline': {
        'n_samples': 50,  # Must match target for structural alignment
        'seed': 43,  # Diverso seed per noise diverso
    },

    # Report generation
    'report': {
        'generate_pdf': True,
        'include_plots': True,
    }
}
