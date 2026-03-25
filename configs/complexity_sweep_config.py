"""
Configuration for the complexity sensitivity sweep.

This config controls the Latin Hypercube Sampling (LHS) parameters and SLURM
settings for the dataset complexity sensitivity analysis. It determines how
controller win rate varies with ST complexity parameters (n, m, rho, n_processes).

Workflow:
    1. Edit this config to define parameter ranges and sampling settings
    2. Run: python euler/complexity_sweep/generate_complexity_sweep_params.py
    3. Submit: sbatch euler/complexity_sweep/complexity_sweep.sh
    4. After completion: python euler/complexity_sweep/generate_complexity_sweep_report.py
"""

COMPLEXITY_SWEEP_CONFIG = {
    # =========================================================================
    # SLURM JOB SETTINGS
    # =========================================================================
    'slurm': {
        'job_name': 'complexity_sweep',
        'account': 'es_mohr',
        'time': '00:30:00',
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
    # LHS SAMPLING SETTINGS
    # =========================================================================
    'sampling': {
        'n_lhs': 30,           # Number of Latin Hypercube configurations
        'n_seeds': 5,          # Number of seed values per axis (total pairs = n_seeds^2)
        'seed': 42,            # Random seed for LHS generation
    },

    # =========================================================================
    # ST COMPLEXITY PARAMETER RANGES
    # =========================================================================
    # Each parameter is defined as (min, max).
    # LHS samples uniformly within these ranges.
    # Constraint: m <= n is enforced after sampling.
    'param_ranges': {
        'n': (2, 8),               # ST input variables
        'm': (1, 4),               # ST cascaded stages (must be <= n)
        'rho': (0.0, 0.5),         # ST noise intensity
        'n_processes': (2, 5),     # Number of processes in sequence
    },

    # =========================================================================
    # OUTPUT
    # =========================================================================
    'output': {
        'output_dir': 'controller_optimization/checkpoints/complexity_sweep',
        'params_file': 'euler/complexity_sweep/complexity_sweep_params.txt',
        'sweep_script': 'euler/complexity_sweep/complexity_sweep.sh',
        'report_dir': None,  # None = same as output_dir
    },

    # =========================================================================
    # FIXED PARAMS (applied to ALL runs, not swept)
    # =========================================================================
    'fixed_params': {
        'no_pdf': True,
    },
}
