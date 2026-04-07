"""
Process-specific configuration for reliability computation.

Defines targets, scales, and weights for each process in the chain.
These values are based on typical ranges from the SCM models.
"""

PROCESS_CONFIGS = {
    'laser': {
        'base_target': 0.8,      # ActualPower target
        'scale': 0.1,            # Quality scale (smaller = more sensitive)
        'weight': 1.0,           # Relative importance
        # Adaptive target: fixed (first process)
    },
    'plasma': {
        'base_target': 3.0,      # RemovalRate target
        'scale': 2.0,
        'weight': 1.0,
        # Adaptive target depends on: laser
        'adaptive_coefficients': {
            'laser': 0.2,        # plasma_target += 0.2 * (laser_output - 0.8)
        },
        'adaptive_baselines': {
            'laser': 0.8,
        },
        # Demo: non-linear adaptive mode (tanh saturates the shift at ±0.5)
        'adaptive_mode': 'tanh',
        'adaptive_max_shift': {
            'laser': 0.5,
        },
    },
    'galvanic': {
        'base_target': 10.0,     # Thickness target (μm)
        'scale': 4.0,
        'weight': 1.5,           # More important (final product quality)
        # Adaptive target depends on: laser, plasma
        'adaptive_coefficients': {
            'plasma': 0.5,       # galvanic_target += 0.5 * (plasma_output - 5.0)
            'laser': 0.4,        # galvanic_target += 0.4 * (laser_output - 0.5)
        },
        'adaptive_baselines': {
            'plasma': 5.0,
            'laser': 0.5,
        },
    },
    'microetch': {
        'base_target': 20.0,     # Depth target
        'scale': 4.0,
        'weight': 1.0,
        # Adaptive target depends on: laser, plasma, galvanic
        'adaptive_coefficients': {
            'laser': 1.5,        # microetch_target += 1.5 * (laser_output - 0.5)
            'plasma': 0.3,       # microetch_target += 0.3 * (plasma_output - 5.0)
            'galvanic': -0.15,   # microetch_target -= 0.15 * (galvanic_output - 10.0)
        },
        'adaptive_baselines': {
            'laser': 0.5,
            'plasma': 5.0,
            'galvanic': 10.0,
        },
    },
}

# Process order (for sequential dependency resolution)
PROCESS_ORDER = ['laser', 'plasma', 'galvanic', 'microetch']
