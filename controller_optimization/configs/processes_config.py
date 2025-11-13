"""
Definizione centralizzata di tutti i processi.

Ogni processo specifica:
- Nome identificativo
- Dataset SCM da usare
- Dimensioni input/output
- Configurazione modello
- Configurazione training
"""

PROCESSES = [
    {
        'name': 'laser',
        'scm_dataset_type': 'laser',
        'input_dim': 2,
        'output_dim': 1,
        'input_labels': ['PowerTarget', 'AmbientTemp'],
        'output_labels': ['ActualPower'],

        'uncertainty_predictor': {
            'model': {
                'hidden_sizes': [64, 32],
                'dropout_rate': 0.1,
                'use_batchnorm': False,
                'min_variance': 1e-6,
            },
            'training': {
                'n_samples': 2000,
                'batch_size': 64,
                'epochs': 200,
                'learning_rate': 0.001,
                'weight_decay': 0.001,
                'patience': 30,
                'loss_type': 'gaussian_nll',
                'variance_penalty_alpha': 1,
            }
        },

        'checkpoint_dir': 'checkpoints/laser',
    },

    {
        'name': 'plasma',
        'scm_dataset_type': 'plasma',
        'input_dim': 2,
        'output_dim': 1,
        'input_labels': ['RF_Power', 'Duration'],
        'output_labels': ['RemovalRate'],

        'uncertainty_predictor': {
            'model': {
                'hidden_sizes': [64, 32],
                'dropout_rate': 0.1,
                'use_batchnorm': False,
                'min_variance': 1e-6,
            },
            'training': {
                'n_samples': 2000,
                'batch_size': 64,
                'epochs': 200,
                'learning_rate': 0.001,
                'weight_decay': 0.001,
                'patience': 30,
                'loss_type': 'gaussian_nll',
                'variance_penalty_alpha': 0.5,
            }
        },

        'checkpoint_dir': 'checkpoints/plasma',
    },

    {
        'name': 'galvanic',
        'scm_dataset_type': 'galvanic',
        'input_dim': 2,
        'output_dim': 1,
        'input_labels': ['CurrentDensity', 'Duration'],
        'output_labels': ['Thickness'],

        'uncertainty_predictor': {
            'model': {
                'hidden_sizes': [32, 16],
                'dropout_rate': 0.1,
                'use_batchnorm': False,
                'min_variance': 1e-6,
            },
            'training': {
                'n_samples': 2000,
                'batch_size': 645,
                'epochs': 200,
                'learning_rate': 0.001,
                'weight_decay': 0.001,
                'patience': 30,
                'loss_type': 'gaussian_nll',
                'variance_penalty_alpha': 2,
            }
        },

        'checkpoint_dir': 'checkpoints/galvanic',
    },

    {
        'name': 'microetch',
        'scm_dataset_type': 'microetch',
        'input_dim': 3,
        'output_dim': 1,
        'input_labels': ['Temperature', 'Concentration', 'Duration'],
        'output_labels': ['RemovalDepth'],

        'uncertainty_predictor': {
            'model': {
                'hidden_sizes': [64, 32, 16],
                'dropout_rate': 0.1,
                'use_batchnorm': False,
                'min_variance': 1e-6,
            },
            'training': {
                'n_samples': 2000,
                'batch_size': 64,
                'epochs': 200,
                'learning_rate': 0.001,
                'weight_decay': 0.001,
                'patience': 30,
                'loss_type': 'gaussian_nll',
                'variance_penalty_alpha': 2,
            }
        },

        'checkpoint_dir': 'checkpoints/microetch',

    },
]


def get_process_by_name(name):
    """Recupera config di un processo per nome"""
    for p in PROCESSES:
        if p['name'] == name:
            return p
    raise ValueError(f"Process '{name}' not found")


def get_process_sequence():
    """Ritorna lista ordinata dei nomi dei processi"""
    return [p['name'] for p in PROCESSES]
