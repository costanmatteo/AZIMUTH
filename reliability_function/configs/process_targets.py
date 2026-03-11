"""
Process-specific configuration for reliability computation.

Defines base_target and scale for each process in the chain.
Adaptive target: τ_i = base_target_i + β × (Y_{i-1} - τ_{i-1})
β is configured globally, not per-process.
"""

PROCESS_CONFIGS = {
    'laser': {
        'base_target': 0.8,      # ActualPower target
        'scale': 0.1,            # Quality scale (smaller = more sensitive)
    },
    'plasma': {
        'base_target': 3.0,      # RemovalRate target
        'scale': 2.0,
    },
    'galvanic': {
        'base_target': 10.0,     # Thickness target (μm)
        'scale': 4.0,
    },
    'microetch': {
        'base_target': 20.0,     # Depth target
        'scale': 4.0,
    },
}

# Process order (for sequential target computation)
PROCESS_ORDER = ['laser', 'plasma', 'galvanic', 'microetch']
