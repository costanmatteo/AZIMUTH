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
        'controllable_inputs': ['PowerTarget'],  # AmbientTemp is environmental (non-controllable)

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
        'controllable_inputs': ['RF_Power'],  # All inputs controllable

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
        'controllable_inputs': ['CurrentDensity', 'Duration'],  # All inputs controllable

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
        'controllable_inputs': ['Concentration', 'Duration'],  # Temperature is environmental (non-controllable)

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


def get_controllable_inputs(process_config):
    """
    Ritorna lista degli input controllabili dal controller.

    Default: se 'controllable_inputs' non è specificato,
    assume tutti gli input come controllabili.

    Args:
        process_config: Dizionario di configurazione del processo

    Returns:
        list: Lista di nomi delle variabili controllabili
    """
    return process_config.get('controllable_inputs', process_config['input_labels'])


def get_filtered_processes(process_names=None):
    """
    Ritorna lista di processi filtrata per nomi.

    Se process_names è None, ritorna tutti i processi.
    Se process_names è fornito, ritorna solo i processi specificati nell'ordine dato.

    Args:
        process_names (list[str], optional): Lista di nomi processi da includere.
                                            Se None, ritorna tutti i PROCESSES.

    Returns:
        list: Lista di configurazioni processi filtrata e ordinata

    Raises:
        ValueError: Se un nome processo specificato non esiste

    Example:
        >>> filtered = get_filtered_processes(['laser', 'plasma'])
        >>> [p['name'] for p in filtered]
        ['laser', 'plasma']
    """
    if process_names is None:
        return PROCESSES

    # Validate: tutti i nomi devono esistere
    available_names = {p['name'] for p in PROCESSES}
    for name in process_names:
        if name not in available_names:
            raise ValueError(
                f"Process '{name}' not found in PROCESSES. "
                f"Available: {sorted(available_names)}"
            )

    # Filtra e ordina secondo process_names
    process_map = {p['name']: p for p in PROCESSES}
    return [process_map[name] for name in process_names]
