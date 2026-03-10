"""
Definizione centralizzata di tutti i processi.

Ogni processo specifica:
- Nome identificativo
- Dataset SCM da usare
- Dimensioni input/output
- Configurazione modello
- Configurazione training

DATASET_MODE controlla quale famiglia di dataset SCM usare:
  - 'physical': 4 processi fisici hardcoded (laser, plasma, galvanic, microetch)
  - 'st': N processi identici basati su Styblinski-Tang, configurabili via ST_DATASET_CONFIG
"""

import sys
from pathlib import Path

# Add project root to path for scm_ds import
_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET MODE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════
# 'physical' → usa i 4 processi fisici (laser, plasma, galvanic, microetch)
# 'st'       → usa N processi identici basati su Styblinski-Tang
DATASET_MODE = 'st'

# ═══════════════════════════════════════════════════════════════════════════════
# ST DATASET CONFIGURATION (usata solo se DATASET_MODE == 'st')
# ═══════════════════════════════════════════════════════════════════════════════
ST_DATASET_CONFIG = {
    # Numero di processi in sequenza
    'n_processes': 3,

    # Parametri STConfig — ogni processo usa la stessa configurazione
    'st_params': {
        'n': 4,                     # variabili di input per processo
        'm': 1,                     # stadi ST per processo (1 = singolo stadio)
        'p': 1,                     # output per processo
        'me': 1,                    # variabili ambientali
        'env_mode': 'A',            # modalità ambiente (A|B|C|D)
        'env_overlap': 0.0,
        'alpha': 0.3,               # ampiezza shift additivo
        'gamma': 0.3,               # ampiezza moltiplicativa
        'rho': 0.2,                 # intensità rumore [0,1]
        'width_profile': 'uniform',
        'carry_beta': 1.0,
        'x_domain': (-5.0, 5.0),
        'e_domain': (-1.0, 1.0),
        'cal_n': 2000,
        'cal_percentile': 10.0,
        'cal_width_factor': 1.0,
    },

    # Configurazione uncertainty predictor (uguale per tutti i processi ST)
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
}

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL UNCERTAINTY CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
# These settings apply to ALL processes unless overridden in process-specific config
GLOBAL_UNCERTAINTY_CONFIG = {
    # Uncertainty quantification method: 'single', 'ensemble', or 'swag'
    'uncertainty_method': 'single',

    # Deep Ensemble configuration (used if uncertainty_method='ensemble')
    'use_ensemble': False,  # DEPRECATED: use uncertainty_method='ensemble'
    'n_ensemble_models': 5,
    'ensemble_base_seed': 42,

    # SWAG configuration (used if uncertainty_method='swag')
    'swag_start_epoch': 0.5,      # Start SWA at 50% of training
    'swag_learning_rate': 0.01,   # LR during SWA phase
    'swag_max_rank': 20,          # Low-rank covariance dimension
    'swag_collection_freq': 1,    # Collect weights every N epochs
    'swag_n_samples': 30,         # Weight samples for prediction
    'swag_min_samples': 20,       # Minimum samples before training can stop
}
# ═══════════════════════════════════════════════════════════════════════════════

_PHYSICAL_PROCESSES = [
    {
        'name': 'laser',
        'scm_dataset_type': 'laser',
        'input_dim': 2,
        'output_dim': 1,
        'input_labels': ['PowerTarget', 'AmbientTemp'],
        'output_labels': ['ActualPower'],
        'controllable_inputs': ['PowerTarget'],  # AmbientTemp is environmental (non-controllable)

        # Process-specific model settings (override global if needed)
        'uncertainty_predictor': {
            'model': {
                'hidden_sizes': [64, 32],
                'dropout_rate': 0.1,
                'use_batchnorm': False,
                'min_variance': 1e-6,
                # Override global uncertainty_method here if needed for this process only
                # 'uncertainty_method': 'swag',
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

        'checkpoint_dir': 'controller_optimization/checkpoints/laser',
    },

    {
        'name': 'plasma',
        'scm_dataset_type': 'plasma',
        'input_dim': 2,
        'output_dim': 1,
        'input_labels': ['RF_Power', 'Duration'],
        'output_labels': ['RemovalRate'],
        'controllable_inputs': ['RF_Power'],

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

        'checkpoint_dir': 'controller_optimization/checkpoints/plasma',
    },

    {
        'name': 'galvanic',
        'scm_dataset_type': 'galvanic',
        'input_dim': 2,
        'output_dim': 1,
        'input_labels': ['CurrentDensity', 'Duration'],
        'output_labels': ['Thickness'],
        'controllable_inputs': ['CurrentDensity', 'Duration'],

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

        'checkpoint_dir': 'controller_optimization/checkpoints/galvanic',
    },

    {
        'name': 'microetch',
        'scm_dataset_type': 'microetch',
        'input_dim': 3,
        'output_dim': 1,
        'input_labels': ['Temperature', 'Concentration', 'Duration'],
        'output_labels': ['RemovalDepth'],
        'controllable_inputs': ['Concentration', 'Duration'],

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

        'checkpoint_dir': 'controller_optimization/checkpoints/microetch',
    },
]


def _build_st_processes(st_dataset_config):
    """
    Genera dinamicamente una lista di processi ST identici in sequenza.

    Ogni processo è un SCM Styblinski-Tang indipendente con la stessa configurazione.
    I nomi seguono il pattern: st_1, st_2, ..., st_N.
    Le label degli input/output sono suffissate con l'indice del processo per
    evitare collisioni nella catena (es. X_1_p1, X_2_p1, Y_p1).

    Args:
        st_dataset_config: dizionario con chiavi 'n_processes', 'st_params',
                           'uncertainty_predictor'

    Returns:
        list: lista di dizionari di configurazione processo (stessa struttura di _PHYSICAL_PROCESSES)
    """
    from scm_ds.datasets_st import STConfig, build_st_scm
    import copy

    n_procs = st_dataset_config['n_processes']
    st_params = st_dataset_config['st_params']
    up_config = st_dataset_config['uncertainty_predictor']

    # Costruisci un SCM di riferimento per ricavare labels e dimensioni
    cfg = STConfig(**st_params)
    ref_scm = build_st_scm(cfg)

    # Label di riferimento (senza suffisso processo)
    # input_labels dello SCM contiene solo X_i, ma per coerenza con i processi
    # fisici (es. laser include AmbientTemp in input_labels) aggiungiamo anche
    # le variabili ambientali E_j come input osservabili non controllabili.
    base_scm_input_labels = list(ref_scm.input_labels)   # es. ["X_1", "X_2", ...]
    base_structural = list(ref_scm.structural_noise_vars)  # es. ["E_1"]
    base_input_labels = base_scm_input_labels + base_structural  # es. ["X_1", ..., "E_1"]
    base_output_labels = list(ref_scm.target_labels)  # es. ["Y"]

    # Input controllabili = solo gli X_i (non gli E_j)
    base_controllable = list(base_scm_input_labels)

    input_dim = len(base_input_labels)
    output_dim = len(base_output_labels)

    # Estrai target e scale calibrati dal nodo output dell'SCM.
    # Per p=1, il nodo output è "Y"; per p>1 sarebbe "Y_1", "Y_2", ecc.
    output_node = base_output_labels[0]  # es. "Y"
    scm_proc_cfg = ref_scm.process_configs.get(output_node, {})
    calibrated_target = scm_proc_cfg.get('base_target', 0.0)
    calibrated_scale = scm_proc_cfg.get('scale', 1.0)
    calibrated_weight = scm_proc_cfg.get('weight', 1.0)

    processes = []
    for i in range(1, n_procs + 1):
        suffix = f"_p{i}"

        # Suffissa le label per questo processo
        input_labels = [f"{l}{suffix}" for l in base_input_labels]
        output_labels = [f"{l}{suffix}" for l in base_output_labels]
        controllable = [f"{l}{suffix}" for l in base_controllable]

        process = {
            'name': f'st_{i}',
            'scm_dataset_type': 'st',
            'input_dim': input_dim,
            'output_dim': output_dim,
            'input_labels': input_labels,
            'output_labels': output_labels,
            'controllable_inputs': controllable,

            # Parametri ST salvati per ricostruire lo SCM a runtime
            'st_params': copy.deepcopy(st_params),

            # Label originali (senza suffisso) per mapping con lo SCM
            '_st_base_input_labels': base_input_labels,
            '_st_base_output_labels': base_output_labels,
            '_st_base_structural_vars': base_structural,

            # Target e scale calibrati dall'SCM (usati da ProTSurrogate)
            'surrogate_target': calibrated_target,
            'surrogate_scale': calibrated_scale,
            'surrogate_weight': calibrated_weight,

            'uncertainty_predictor': copy.deepcopy(up_config),

            'checkpoint_dir': f'controller_optimization/checkpoints/st_{i}',
        }
        processes.append(process)

    return processes


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESSES — selezionato automaticamente in base a DATASET_MODE
# ═══════════════════════════════════════════════════════════════════════════════
if DATASET_MODE == 'physical':
    PROCESSES = _PHYSICAL_PROCESSES
elif DATASET_MODE == 'st':
    PROCESSES = _build_st_processes(ST_DATASET_CONFIG)
else:
    raise ValueError(f"DATASET_MODE sconosciuto: '{DATASET_MODE}'. Usa 'physical' o 'st'.")


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
