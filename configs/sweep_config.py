"""
Configuration for the SLURM-based controller sweep.

This config controls which parameters are swept and how sweep_params.txt
and sweep.sh are generated. Each parameter listed in 'params' produces a
grid (cartesian product) of all combinations, which are then written to
sweep_params.txt and submitted as a SLURM array job.

Workflow:
    1. Edit this config to define the parameters you want to sweep
    2. Run: python euler/sweep/generate_sweep_params.py --from-config
    3. Submit: sbatch euler/sweep/sweep.sh

Every key in 'params' must match a CLI argument of train_controller.py
(without the -- prefix). The values are lists of values to sweep over.

Example: if params = {'seed_target': [1, 50], 'learning_rate': [1e-3, 1e-4]}
         this generates 2 x 2 = 4 combinations in sweep_params.txt.
"""

SWEEP_CONFIG = {
    # =========================================================================
    # SLURM JOB SETTINGS
    # =========================================================================
    'slurm': {
        'job_name': 'ctrl_sweep',
        'account': 'es_mohr',
        'time': '00:10:00',            # Max wall time per job (HH:MM:SS)
        'ntasks': 1,
        'cpus_per_task': 1,
        'mem_per_cpu': '2G',
        'output_log': 'logs/sweep_%A_%a.out',
        'error_log': 'logs/sweep_%A_%a.err',
        # GPU (uncomment if needed):
        # 'gpus_per_node': 1,
        # 'partition': 'gpu',
    },

    # =========================================================================
    # MODULE / ENVIRONMENT
    # =========================================================================
    'environment': {
        'modules': ['stack/2024-05', 'gcc/13.2.0', 'python/3.11.6_cuda'],
        'venv': None,  # None = auto-detect (venv or .venv), or explicit path
    },

    # =========================================================================
    # SWEEP PARAMETERS
    # =========================================================================
    # Each key is a CLI argument of train_controller.py.
    # Each value is a list of values to sweep over.
    # The cartesian product of all lists determines the total number of runs.
    #
    # Supported CLI args:
    #   Training:    learning_rate, epochs, batch_size, lambda_bc, weight_decay,
    #                reliability_loss_scale, patience, gradient_clip_norm
    #   Architecture: dropout, hidden_sizes, use_scenario_encoder,
    #                 scenario_embedding_dim
    #   Scenarios:   n_train, n_test, seed, seed_target, seed_baseline
    #   Curriculum:  curriculum_enabled, no_curriculum
    #   ST dataset:  st_n, st_m, st_rho, st_n_processes
    #   Checkpoints: up_checkpoint_dir, surrogate_checkpoint_dir
    #   Output:      output_dir, run_name, no_pdf, quiet
    # =========================================================================
    'params': {
        'seed_target': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37,
                        41, 45, 49, 53, 57, 61, 65, 69, 73, 77,
                        81, 85, 89, 93, 97, 101, 105 , 109, 113, 117, 121, 125, 129, 133, 137, 141, 145, 149, 153, 157, 161,],
        'seed_baseline': [1, 11, 21, 31, 41],
    },

    # =========================================================================
    # OUTPUT
    # =========================================================================
    'output': {
        'output_dir': 'controller_optimization/checkpoints/sweep',
        'params_file': 'euler/sweep/sweep_params.txt',
        'sweep_script': 'euler/sweep/sweep.sh',
        # Run name template: use {param_name} placeholders
        # Available: any key from 'params' above
        'run_name_template': 'seed_t{seed_target:02d}_b{seed_baseline:02d}',
    },

    # =========================================================================
    # FIXED PARAMS (applied to ALL runs, not swept)
    # =========================================================================
    # These are passed as CLI args to every run but are not part of the grid.
    # Useful for setting a common config across the entire sweep.
    'fixed_params': {
        'no_pdf': True,
        'quiet': True,
    },
}
