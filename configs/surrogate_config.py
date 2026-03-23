"""
Configurazione per il training del surrogato CasualiT.

Il surrogato apprende a predire la reliability F data una traiettoria completa:
    (inputs, env, outputs) per ogni processo -> F

Il dataset viene caricato da data/trajectories/full_trajectories.pt,
lo stesso generato da generate_dataset.py e usato per allenare gli
uncertainty predictor.

Architettura:
- Transformer encoder: processa la sequenza di processi
- Output head: predice F come scalare in [0, 1]
"""

SURROGATE_CONFIG = {
    # Model architecture
    'model': {
        'd_model_enc': 64,
        'd_ff': 128,
        'n_heads': 4,
        'e_layers': 2,
        'dropout_emb': 0.1,
        'activation': 'gelu',
    },

    # Training
    'training': {
        'max_epochs': 200,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'weight_decay': 0.01,
        'loss_fn': 'mse',
        'seed': 42,
        'patience': 20,

        'use_scheduler': True,
        'scheduler_factor': 0.5,
        'scheduler_patience': 10,
    },

    # Data split (applied to full_trajectories.pt)
    'data': {
        'train_size': 0.7,
        'val_size': 0.15,
        'test_size': 0.15,
    },

    # Checkpoints
    'checkpoints': {
        'save_dir': 'checkpoints/surrogate',
    },

    # Report
    'report': {
        'generate_pdf': True,
        'output_dir': 'checkpoints/surrogate',
    },
}
