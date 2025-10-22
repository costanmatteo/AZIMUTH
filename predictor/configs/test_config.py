"""
Configurazione per testare il modello con dati semplici

Funzione: output = 2*x + 3*y - z + rumore
"""

CONFIG = {
    # Configurazione dati
    'data': {
        'csv_path': 'data/raw/test_data.csv',

        # Input: 3 variabili semplici
        'input_columns': [
            'x',
            'y',
            'z',
        ],

        # Output: 1 valore
        'output_columns': [
            'output',
        ],

        'train_size': 0.7,
        'val_size': 0.15,
        'test_size': 0.15,
        'random_state': 42,
        'scaling_method': 'standard',
    },

    # Configurazione modello - SEMPLICE per questo test
    'model': {
        'hidden_sizes': [16, 8],  # Rete piccola (è un problema semplice!)
        'dropout_rate': 0.1,
        'model_type': 'custom',
    },

    # Configurazione training
    'training': {
        'epochs': 100,             # Poche epoche (dovrebbe convergere velocemente)
        'batch_size': 32,
        'learning_rate': 0.01,     # Learning rate più alto (problema semplice)
        'loss_function': 'mse',
        'patience': 15,
        'device': 'auto',
        'checkpoint_dir': 'checkpoints_test',  # Cartella separata per test
    },

    # Altro
    'misc': {
        'random_seed': 42,
        'num_workers': 0,
    }
}
