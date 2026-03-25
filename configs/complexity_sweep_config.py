"""
Configuration for the SLURM-based complexity sensitivity sweep.

This sweep tests how controller win rate varies with dataset complexity
parameters (n, m, rho, n_processes) using Latin Hypercube Sampling (LHS)
combined with a reduced seed grid for each configuration.

Total runs = n_lhs_configs * n_seeds^2

Workflow:
    1. Edit this config to define complexity ranges and seed grid
    2. Run: python euler/complexity_sweep/generate_complexity_sweep_params.py --from-config
    3. Submit: sbatch euler/complexity_sweep/complexity_sweep.sh
    4. Report: python euler/complexity_sweep/generate_complexity_sweep_report.py

Every key in 'st_params' defines the range for a dataset complexity parameter.
The seed grid in 'seeds' controls robustness evaluation per configuration.
"""

COMPLEXITY_SWEEP_CONFIG = {
    # =========================================================================
    # SLURM JOB SETTINGS
    # =========================================================================
    'slurm': {
        'job_name': 'complexity_sweep',
        'account': 'es_mohr',
        'time': '00:30:00',            # Max wall time per job (HH:MM:SS)
        'ntasks': 1,
        'cpus_per_task': 1,
        'mem_per_cpu': '4G',
        'output_log': 'logs/complexity_%A_%a.out',
        'error_log': 'logs/complexity_%A_%a.err',
    },

    # =========================================================================
    # MODULE / ENVIRONMENT
    # =========================================================================
    'environment': {
        'modules': ['stack/2024-05', 'gcc/13.2.0', 'python/3.11.6_cuda'],
        'venv': None,  # None = auto-detect (venv or .venv), or explicit path
    },

    # =========================================================================
    # LATIN HYPERCUBE SAMPLING (LHS) SETTINGS
    # =========================================================================
    'lhs': {
        'n_configs': 30,    # Number of LHS configurations to sample
        'seed': 42,         # Random seed for LHS generation (reproducibility)
    },

    # =========================================================================
    # ST COMPLEXITY PARAMETERS (ranges for LHS sampling)
    # =========================================================================
    # Each parameter has: min, max, type ('int' or 'float')
    # Constraint enforced: m <= n (required by ST SCM)
    'st_params': {
        'n': {
            'min': 2,
            'max': 8,
            'type': 'int',
            'description': 'Number of input variables per process',
        },
        'm': {
            'min': 1,
            'max': 4,
            'type': 'int',
            'description': 'Number of cascaded stages per process',
        },
        'rho': {
            'min': 0.0,
            'max': 0.5,
            'type': 'float',
            'description': 'Noise intensity (correlation strength)',
        },
        'n_processes': {
            'min': 2,
            'max': 5,
            'type': 'int',
            'description': 'Number of processes in the chain',
        },
    },

    # =========================================================================
    # SEED GRID (for robustness evaluation per ST configuration)
    # =========================================================================
    # n_seeds^2 seed pairs are generated per LHS configuration.
    # Seeds are evenly spaced from 1 to ~100.
    'seeds': {
        'n_seeds': 5,   # Number of seed values per axis (total pairs = n_seeds^2 = 25)
    },

    # =========================================================================
    # OUTPUT
    # =========================================================================
    'output': {
        'output_dir': 'controller_optimization/checkpoints/complexity_sweep',
        'params_file': 'euler/complexity_sweep/complexity_sweep_params.txt',
        'sweep_script': 'euler/complexity_sweep/complexity_sweep.sh',
        # Run name template: use {param_name} placeholders
        'run_name_template': 'cfg{cfg_idx:02d}_n{n}_m{m}_p{n_processes}_r{rho:.2f}_t{seed_target:02d}_b{seed_baseline:02d}',
    },

    # =========================================================================
    # FIXED PARAMS (applied to ALL runs, not swept)
    # =========================================================================
    # These are passed as CLI args to every controller training run.
    'fixed_params': {
        'no_pdf': True,
    },
}
