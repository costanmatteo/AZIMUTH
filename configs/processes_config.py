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

La configurazione dell'uncertainty predictor è definita in configs/uncertainty_config.py
e viene iniettata automaticamente nei dizionari di processo.
"""

import sys
from pathlib import Path

# Add project root to path for scm_ds import
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from configs.uncertainty_config import (  # noqa: E402
    GLOBAL_UNCERTAINTY_CONFIG,
    DEFAULT_ST_UNCERTAINTY_PREDICTOR,
    PHYSICAL_UNCERTAINTY_PREDICTORS,
)

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

    # Numero di campioni da generare per processo
    'n_samples': 2000,

    # Coefficiente adattivo inter-processo per il surrogate.
    # τ_i = base_target + (adaptive_coeff / (i-1)) × Σ_{j<i} (Y_j - base_target)  per i > 1
    # Se 0.0, tutti i processi usano un target fisso (base_target).
    'adaptive_coeff': 0.3,

    # Parametri STConfig — ogni processo usa la stessa configurazione
    'st_params': {
        'n': 4,                     # variabili di input per processo
        'm': 1,                     # stadi ST per processo (1 = singolo stadio)
        'p': 1,                     # output per processo
        'me': 1,                    # variabili ambientali
        'env_mode': 'A',            # modalità ambiente (A|B|C|D)
        'env_overlap': 0.0,
        'output_overlap': 0.0,          # sovrapposizione confine output [0,1]
        'alpha': 0.3,               # ampiezza shift additivo
        'gamma': 0.3,               # ampiezza moltiplicativa
        'rho': 0.2,                 # intensità rumore [0,1]
        'width_profile': 'uniform',
        'carry_beta': 0.3,
        'x_domain': (-2.0, 2.0),
        'e_domain': (-1.0, 1.0),
        'cal_n': 2000,
        'cal_percentile': 10.0,
        'cal_width_factor': 1.0,
    },

    # ── Modalità adattiva non-lineare ──────────────────────────────────────────
    # Tipo di funzione usata per calcolare l'aggiustamento del target adattivo.
    # Opzioni: 'linear' (default), 'polynomial', 'power', 'softplus', 'deadband', 'tanh'
    # Se omesso o 'linear', il comportamento è identico a prima.
    'adaptive_mode': 'linear',

    # Parametri aggiuntivi per i mode non-lineari (ignorati se mode='linear').
    # Ogni dict mappa nome_processo_upstream → valore del parametro.
    # I nomi upstream vengono generati automaticamente (st_1, st_2, ...).
    #
    # 'adaptive_coefficients2': {},   # polynomial: coeff2 per upstream
    # 'adaptive_power': {},           # power: alpha per upstream (default 0.5)
    # 'adaptive_sharpness': {},       # softplus: k per upstream (default 2.0)
    # 'adaptive_band': {},            # deadband: band width per upstream
    # 'adaptive_max_shift': {},       # tanh: saturazione massima per upstream

    # Configurazione uncertainty predictor: importata da uncertainty_config.py
    'uncertainty_predictor': DEFAULT_ST_UNCERTAINTY_PREDICTOR,
}

# ═══════════════════════════════════════════════════════════════════════════════

_PHYSICAL_PROCESSES = [
    {
        'name': 'laser',
        'scm_dataset_type': 'laser',
        'n_samples': 2000,
        'input_dim': 2,
        'output_dim': 1,
        'input_labels': ['PowerTarget', 'AmbientTemp'],
        'output_labels': ['ActualPower'],
        'controllable_inputs': ['PowerTarget'],  # AmbientTemp is environmental (non-controllable)
        'uncertainty_predictor': PHYSICAL_UNCERTAINTY_PREDICTORS['laser'],
        'checkpoint_dir': 'checkpoints/predictor/laser',
    },

    {
        'name': 'plasma',
        'scm_dataset_type': 'plasma',
        'n_samples': 2000,
        'input_dim': 2,
        'output_dim': 1,
        'input_labels': ['RF_Power', 'Duration'],
        'output_labels': ['RemovalRate'],
        'controllable_inputs': ['RF_Power'],
        'uncertainty_predictor': PHYSICAL_UNCERTAINTY_PREDICTORS['plasma'],
        'checkpoint_dir': 'checkpoints/predictor/plasma',
    },

    {
        'name': 'galvanic',
        'scm_dataset_type': 'galvanic',
        'n_samples': 2000,
        'input_dim': 2,
        'output_dim': 1,
        'input_labels': ['CurrentDensity', 'Duration'],
        'output_labels': ['Thickness'],
        'controllable_inputs': ['CurrentDensity', 'Duration'],
        'uncertainty_predictor': PHYSICAL_UNCERTAINTY_PREDICTORS['galvanic'],
        'checkpoint_dir': 'checkpoints/predictor/galvanic',
    },

    {
        'name': 'microetch',
        'scm_dataset_type': 'microetch',
        'n_samples': 2000,
        'input_dim': 3,
        'output_dim': 1,
        'input_labels': ['Temperature', 'Concentration', 'Duration'],
        'output_labels': ['RemovalDepth'],
        'controllable_inputs': ['Concentration', 'Duration'],
        'uncertainty_predictor': PHYSICAL_UNCERTAINTY_PREDICTORS['microetch'],
        'checkpoint_dir': 'checkpoints/predictor/microetch',
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
    adaptive_coeff = st_dataset_config.get('adaptive_coeff', 0.0)

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

    # Estrai target e scale calibrati da ciascun nodo output dell'SCM.
    # Per p=1, il nodo output è "Y"; per p>1 sono "Y_1", "Y_2", ecc.
    calibrated_targets = []
    calibrated_scales  = []
    calibrated_weights = []
    for out_node in base_output_labels:
        cfg_node = ref_scm.process_configs.get(out_node, {})
        calibrated_targets.append(cfg_node.get('base_target', 0.0))
        calibrated_scales.append(cfg_node.get('scale', 1.0))
        calibrated_weights.append(cfg_node.get('weight', 1.0))

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
            'n_samples': st_dataset_config.get('n_samples', 2000),
            'input_dim': input_dim,
            'output_dim': output_dim,
            'input_labels': input_labels,
            'output_labels': output_labels,
            'controllable_inputs': controllable,

            # Controller action domain: restrict to [-1, 1] to avoid
            # zero-nullification at boundaries of sin(π/2 · x).
            'action_domain': (-1.0, 1.0),

            # Parametri ST salvati per ricostruire lo SCM a runtime
            'st_params': copy.deepcopy(st_params),

            # Label originali (senza suffisso) per mapping con lo SCM
            '_st_base_input_labels': base_input_labels,
            '_st_base_output_labels': base_output_labels,
            '_st_base_structural_vars': base_structural,

            # Target e scale calibrati dall'SCM (usati da ProTSurrogate)
            # Liste di lunghezza output_dim (p); per p=1 liste con 1 elemento.
            'surrogate_target': calibrated_targets,
            'surrogate_scale':  calibrated_scales,
            'surrogate_weight': calibrated_weights,

            'uncertainty_predictor': copy.deepcopy(up_config),

            'checkpoint_dir': f'checkpoints/predictor/st_{i}',

            # Pre-built SCM instance (avoids rebuilding in target_generation.py)
            '_scm_instance': ref_scm,
        }

        # Target adattivo inter-processo (tutti i precedenti, peso normalizzato):
        # τ_i = base_target + (coeff / (i-1)) × Σ_{j<i} (Y_j - base_target)
        # Il baseline adattivo usa la media dei target calibrati (scalare)
        # per coerenza con il calcolo scalare dell'adaptive target.
        if i > 1 and adaptive_coeff != 0.0:
            n_prev = i - 1
            coeff_per_proc = adaptive_coeff / n_prev
            calibrated_target_mean = sum(calibrated_targets) / len(calibrated_targets)
            process['surrogate_adaptive_coefficients'] = {
                f'st_{j}': coeff_per_proc for j in range(1, i)
            }
            process['surrogate_adaptive_baselines'] = {
                f'st_{j}': calibrated_target_mean for j in range(1, i)
            }

            # Propagate non-linear adaptive mode params if configured
            for src_key, dst_key in [
                ('adaptive_mode',           'surrogate_adaptive_mode'),
                ('adaptive_coefficients2',  'surrogate_adaptive_coefficients2'),
                ('adaptive_power',          'surrogate_adaptive_power'),
                ('adaptive_band',           'surrogate_adaptive_band'),
                ('adaptive_sharpness',      'surrogate_adaptive_sharpness'),
                ('adaptive_max_shift',      'surrogate_adaptive_max_shift'),
            ]:
                if src_key in st_dataset_config:
                    process[dst_key] = st_dataset_config[src_key]

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
