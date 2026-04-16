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

    # SWAG: più campioni, posterior più larga, più samples a inference
    'swag_start_epoch': 0.15,
    'swag_learning_rate': 0.015,
    'swag_max_rank': 30,
    'swag_collection_freq': 1,
    'swag_n_samples': 60,
    'swag_min_samples': 40,
}

# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
# Controls K-fold CV in train_predictor.py. Used as the default when
# --cv_folds is not passed on the CLI; CLI always overrides.
CV_CONFIG = {
    # None (or 0) → single 70/15/15 split (original behavior).
    # K >= 2      → K-fold CV on the 85% pool with a fixed 15% hold-out test.
    'cv_folds': None,

    # Hold-out test fraction set aside before CV. Also applied to the refit's
    # internal val split (0.15/0.85 of the pool) so the refit reproduces the
    # 70/15/15 partition sizes of the non-CV flow.
    'test_fraction': 0.15,
}
# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT ST UNCERTAINTY PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
# Configurazione modello/training condivisa da tutti i processi ST
DEFAULT_ST_UNCERTAINTY_PREDICTOR = {
    'model': {
        'hidden_sizes': [32, 16, 8],
        'dropout_rate': 0.1,
        'use_batchnorm': True,
        'min_variance': 1e-3,
    },
    'training': {
        'batch_size': 32,
        'epochs': 200,
        'learning_rate': 0.001,
        'weight_decay': 5e-3,
        'patience': 120,
        'loss_type': 'gaussian_nll',
        'variance_penalty_alpha': 0.7,
        'grad_clip_max_norm': 1.0,
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
            'grad_clip_max_norm': 1.0,
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
            'grad_clip_max_norm': 1.0,
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
            'grad_clip_max_norm': 1.0,
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
            'grad_clip_max_norm': 1.0,
        }
    },
}
