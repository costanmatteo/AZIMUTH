"""
Esempio di configurazione in Python (alternativa a YAML)
"""

CONFIG = {
    # Configurazione dati
    'data': {
        'csv_path': 'C:/COMPASS/COMPASS/predictor/src/data/raw/dataset_sintetico_xyz.csv',
        'input_columns': [
            'x',
            'y',
            'z',
            # Aggiungi altri parametri
        ],
        'output_columns': [
            'res',
            # Aggiungi altri output
        ],
        'train_size': 0.7,
        'val_size': 0.15,
        'test_size': 0.15,
        'random_state': 42,
        'scaling_method': 'standard',  # o 'minmax'
    },

    # Configurazione modello
    'model': {
        'hidden_sizes': [512, 256, 64],
        'dropout_rate': 0.2,
        'model_type': 'custom',  # 'small', 'medium', 'large', o 'custom'
        'use_batchnorm': True,
    },

    # Configurazione training
    'training': {
        'epochs': 200,
        'batch_size': 32,
        'learning_rate': 0.001,
        'loss_function': 'mse',  # 'mse', 'mae', o 'huber'
        'patience': 20,
        'device': 'auto',  # 'cuda', 'cpu', o 'auto'
        'checkpoint_dir': 'checkpoints',
    },

    # Altro
    'misc': {
        'random_seed': 42,
        'num_workers': 4,
    }
}


# Configurazioni predefinite per diversi scenari
SMALL_DATASET_CONFIG = {
    **CONFIG,
    'model': {
        'model_type': 'small',
        'dropout_rate': 0.1,
    },
    'training': {
        **CONFIG['training'],
        'batch_size': 16,
        'epochs': 100,
    }
}

LARGE_DATASET_CONFIG = {
    **CONFIG,
    'model': {
        'model_type': 'large',
        'dropout_rate': 0.3,
    },
    'training': {
        **CONFIG['training'],
        'batch_size': 64,
        'epochs': 300,
    }
}
