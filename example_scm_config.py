"""
ESEMPIO: Configurazioni SCM per diversi scenari
Copia una di queste configurazioni in configs/example_config.py
"""

# ============================================================================
# SCENARIO 1: Dataset piccolo per testing veloce
# ============================================================================
CONFIG_SMALL = {
    'data': {
        'csv_path': None,  # Usa SCM
        'use_scm': True,
        'scm': {
            'n_samples': 1000,      # Solo 1000 campioni → training veloce
            'seed': 42,
            'dataset_type': 'one_to_one_ct'
        },
        'scaling_method': 'standard',
        'train_size': 0.7,
        'val_size': 0.15,
        'test_size': 0.15,
    },
    'training': {
        'epochs': 50,               # Poche epoche per test
        'batch_size': 32,
    }
}

# ============================================================================
# SCENARIO 2: Dataset grande per training serio
# ============================================================================
CONFIG_LARGE = {
    'data': {
        'csv_path': None,  # Usa SCM
        'use_scm': True,
        'scm': {
            'n_samples': 50000,     # 50k campioni → training robusto
            'seed': 42,
            'dataset_type': 'one_to_one_ct'
        },
        'scaling_method': 'standard',
        'train_size': 0.7,
        'val_size': 0.15,
        'test_size': 0.15,
    },
    'training': {
        'epochs': 400,              # Molte epoche
        'batch_size': 128,          # Batch grande
    }
}

# ============================================================================
# SCENARIO 3: Passare da SCM a CSV (dati reali)
# ============================================================================
CONFIG_CSV = {
    'data': {
        'csv_path': 'data/my_real_data.csv',  # ← Usa CSV invece di SCM
        'input_columns': ['x', 'y', 'z'],
        'output_columns': ['res_1'],
        'scaling_method': 'standard',
        'train_size': 0.7,
        'val_size': 0.15,
        'test_size': 0.15,
    }
}

# ============================================================================
# SCENARIO 4: Generare dataset SCM riproducibile
# ============================================================================
CONFIG_REPRODUCIBLE = {
    'data': {
        'csv_path': None,
        'use_scm': True,
        'scm': {
            'n_samples': 10000,
            'seed': 123,            # ← Cambia seed per dataset diversi
            'dataset_type': 'one_to_one_ct'
        },
        'scaling_method': 'standard',
        'train_size': 0.7,
        'val_size': 0.15,
        'test_size': 0.15,
        'random_state': 123,        # ← Stesso seed per split
    },
    'misc': {
        'random_seed': 123,         # ← Stesso seed per training
    }
}
