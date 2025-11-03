"""
Configurazione esempio per dati di produzione
"""

CONFIG = {
    # Data configuration
    'data': {
        # OPZIONE 1: Usa i dati di esempio
        'csv_path': 'example_production_data.csv',
        'input_columns': ['temperatura', 'pressione', 'velocita'],
        'output_columns': ['qualita'],

        # OPZIONE 2: Oppure usa i tuoi dati (decommentare)
        # 'csv_path': 'path/to/your/data.csv',
        # 'input_columns': ['col1', 'col2', 'col3'],
        # 'output_columns': ['target'],

        'scaling_method': 'standard',
        'train_size': 0.7,
        'val_size': 0.15,
        'test_size': 0.15,
        'random_state': 42
    },

    # Model configuration
    'model': {
        'model_type': 'small',  # 'small', 'medium', 'large'
        'dropout_rate': 0.1,
        'use_batchnorm': False,
        'min_variance': 1e-6
    },

    # Training configuration
    'training': {
        'batch_size': 4,  # Piccolo perché abbiamo pochi dati
        'epochs': 200,
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'patience': 30,
        'device': 'auto',
        'checkpoint_dir': 'checkpoints_production',
        'variance_penalty_alpha': 0.2
    },

    # Uncertainty configuration
    'uncertainty': {
        'confidence_level': 0.95,
    },

    # Miscellaneous
    'misc': {
        'random_seed': 42,
        'verbose': True
    }
}
