"""
Configurazione centralizzata per l'uncertainty predictor.

Contiene:
- GLOBAL_UNCERTAINTY_CONFIG: impostazioni globali (metodo, ensemble, SWAG)
- DEFAULT_ST_UNCERTAINTY_PREDICTOR: configurazione modello/training per processi ST
- PHYSICAL_UNCERTAINTY_PREDICTORS: configurazione modello/training per processo fisico
"""

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL UNCERTAINTY CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
# These settings apply to ALL processes unless overridden in process-specific config
GLOBAL_UNCERTAINTY_CONFIG = {
    # Uncertainty quantification method: 'single', 'ensemble', or 'swag'
    'uncertainty_method': 'swag',

    # Deep Ensemble configuration (used if uncertainty_method='ensemble')
    'use_ensemble': False,  # DEPRECATED: use uncertainty_method='ensemble'
    'n_ensemble_models': 5,
    'ensemble_base_seed': 42,

    # SWAG configuration (used if uncertainty_method='swag')
    'swag_start_epoch': 0.15,     # SWA starts early via early-stop, fraction is fallback
    'swag_learning_rate': 0.015,  # LR during SWA phase
    'swag_max_rank': 30,          # Low-rank covariance dimension
    'swag_collection_freq': 1,    # Collect weights every N epochs
    'swag_n_samples': 60,         # Weight samples for prediction
    'swag_min_samples': 40,       # Minimum samples before training can stop
}

# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT ST UNCERTAINTY PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
# Configurazione modello/training condivisa da tutti i processi ST
DEFAULT_ST_UNCERTAINTY_PREDICTOR = {
    'model': {
        'hidden_sizes': [32, 16],
        'dropout_rate': 0.2,
        'use_batchnorm': False,
        'min_variance': 1e-3,
    },
    'training': {
        'batch_size': 64,
        'epochs': 200,
        'learning_rate': 0.001,
        'weight_decay': 5e-3,
        'patience': 15,
        'loss_type': 'gaussian_nll',
        'variance_penalty_alpha': 0.7,
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICAL PROCESS UNCERTAINTY PREDICTORS
# ═══════════════════════════════════════════════════════════════════════════════
# Configurazione per-processo dei modelli di incertezza (processi fisici)
PHYSICAL_UNCERTAINTY_PREDICTORS = {
    'laser': {
        'model': {
            'hidden_sizes': [64, 32],
            'dropout_rate': 0.1,
            'use_batchnorm': False,
            'min_variance': 1e-6,
        },
        'training': {
            'batch_size': 64,
            'epochs': 200,
            'learning_rate': 0.001,
            'weight_decay': 0.001,
            'patience': 30,
            'loss_type': 'gaussian_nll',
            'variance_penalty_alpha': 1,
        }
    },

    'plasma': {
        'model': {
            'hidden_sizes': [64, 32],
            'dropout_rate': 0.1,
            'use_batchnorm': False,
            'min_variance': 1e-6,
        },
        'training': {
            'batch_size': 64,
            'epochs': 200,
            'learning_rate': 0.001,
            'weight_decay': 0.001,
            'patience': 30,
            'loss_type': 'gaussian_nll',
            'variance_penalty_alpha': 0.5,
        }
    },

    'galvanic': {
        'model': {
            'hidden_sizes': [32, 16],
            'dropout_rate': 0.1,
            'use_batchnorm': False,
            'min_variance': 1e-6,
        },
        'training': {
            'batch_size': 645,
            'epochs': 200,
            'learning_rate': 0.001,
            'weight_decay': 0.001,
            'patience': 30,
            'loss_type': 'gaussian_nll',
            'variance_penalty_alpha': 2,
        }
    },

    'microetch': {
        'model': {
            'hidden_sizes': [64, 32, 16],
            'dropout_rate': 0.1,
            'use_batchnorm': False,
            'min_variance': 1e-6,
        },
        'training': {
            'batch_size': 64,
            'epochs': 200,
            'learning_rate': 0.001,
            'weight_decay': 0.001,
            'patience': 30,
            'loss_type': 'gaussian_nll',
            'variance_penalty_alpha': 2,
        }
    },
}
