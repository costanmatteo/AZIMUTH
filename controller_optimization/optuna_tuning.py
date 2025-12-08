#!/usr/bin/env python3
"""
Optuna Hyperparameter Tuning for Controller Optimization.

This script uses Optuna to find optimal hyperparameters for the PolicyGenerator
neural network. It supports both local execution and distributed execution on
Euler HPC cluster.

Hyperparameters optimized:
- hidden_sizes: Architecture of the policy generator
- dropout: Dropout rate for regularization
- use_batchnorm: Whether to use batch normalization
- scenario_embedding_dim: Dimension of scenario embedding
- learning_rate: Learning rate for optimizer

Usage:
    # Local execution (sequential)
    python optuna_tuning.py --n-trials 50

    # Create study for distributed execution (Euler)
    python optuna_tuning.py --create-study --study-name "controller_hpo"

    # Run single trial (called by SLURM jobs)
    python optuna_tuning.py --study-name "controller_hpo" --single-trial

    # Check study status
    python optuna_tuning.py --study-name "controller_hpo" --status

    # Generate report after optimization
    python optuna_tuning.py --study-name "controller_hpo" --report
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import json
import copy

import numpy as np
import torch

# Add project root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    import optuna
    from optuna.trial import TrialState
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate,
        plot_slice,
        plot_contour,
    )
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not installed. Run: pip install optuna")

from controller_optimization.configs.processes_config import PROCESSES, get_filtered_processes
from controller_optimization.configs.controller_config import CONTROLLER_CONFIG
from controller_optimization.src.utils.target_generation import (
    generate_target_trajectory,
    generate_baseline_trajectory
)
from controller_optimization.src.utils.process_chain import ProcessChain
from controller_optimization.src.models.surrogate import ProTSurrogate
from controller_optimization.src.training.controller_trainer import ControllerTrainer


# =============================================================================
# HYPERPARAMETER SEARCH SPACE
# =============================================================================

SEARCH_SPACE = {
    'hidden_sizes': [
        [32, 16],
        [64, 32],
        [128, 64],
        [128, 64, 32],
        [256, 128, 64],
    ],
    'dropout': (0.0, 0.4),  # Uniform range
    'use_batchnorm': [True, False],
    'scenario_embedding_dim': (8, 64, 8),  # (min, max, step)
    'learning_rate': (1e-5, 1e-2),  # Log-uniform range
}


# =============================================================================
# OBJECTIVE FUNCTION
# =============================================================================

def create_objective(base_config: dict, device: str = 'auto',
                     reduced_epochs: int = None, verbose: bool = False):
    """
    Create Optuna objective function.

    Args:
        base_config: Base configuration dict (CONTROLLER_CONFIG)
        device: Device to use ('auto', 'cuda', 'cpu')
        reduced_epochs: If set, use fewer epochs for faster trials
        verbose: Print training progress

    Returns:
        Callable objective function for Optuna
    """

    def objective(trial: optuna.Trial) -> float:
        """
        Objective function for a single Optuna trial.

        Args:
            trial: Optuna trial object

        Returns:
            Final loss value to minimize
        """
        # Create a deep copy of config
        cfg = copy.deepcopy(base_config)

        # =================================================================
        # SUGGEST HYPERPARAMETERS
        # =================================================================

        # Hidden sizes (categorical)
        hidden_sizes = trial.suggest_categorical(
            'hidden_sizes',
            [str(h) for h in SEARCH_SPACE['hidden_sizes']]
        )
        hidden_sizes = eval(hidden_sizes)  # Convert string back to list

        # Dropout (float)
        dropout = trial.suggest_float(
            'dropout',
            SEARCH_SPACE['dropout'][0],
            SEARCH_SPACE['dropout'][1]
        )

        # Batch normalization (categorical/boolean)
        use_batchnorm = trial.suggest_categorical(
            'use_batchnorm',
            SEARCH_SPACE['use_batchnorm']
        )

        # Scenario embedding dimension (int with step)
        scenario_embedding_dim = trial.suggest_int(
            'scenario_embedding_dim',
            SEARCH_SPACE['scenario_embedding_dim'][0],
            SEARCH_SPACE['scenario_embedding_dim'][1],
            step=SEARCH_SPACE['scenario_embedding_dim'][2]
        )

        # Learning rate (log-uniform)
        learning_rate = trial.suggest_float(
            'learning_rate',
            SEARCH_SPACE['learning_rate'][0],
            SEARCH_SPACE['learning_rate'][1],
            log=True
        )

        # =================================================================
        # UPDATE CONFIG WITH SUGGESTED HYPERPARAMETERS
        # =================================================================

        cfg['policy_generator']['architecture'] = 'custom'
        cfg['policy_generator']['hidden_sizes'] = hidden_sizes
        cfg['policy_generator']['dropout'] = dropout
        cfg['policy_generator']['use_batchnorm'] = use_batchnorm
        cfg['policy_generator']['use_scenario_encoder'] = True  # Enable for embedding_dim
        cfg['policy_generator']['scenario_embedding_dim'] = scenario_embedding_dim
        cfg['training']['learning_rate'] = learning_rate

        # Reduce epochs for faster trials if specified
        if reduced_epochs is not None:
            cfg['training']['epochs'] = reduced_epochs
            # Adjust patience proportionally
            original_patience = base_config['training']['patience']
            original_epochs = base_config['training']['epochs']
            cfg['training']['patience'] = max(50, int(original_patience * reduced_epochs / original_epochs))

        # Disable PDF report generation during trials
        cfg['report']['generate_pdf'] = False

        # Set device
        actual_device = device
        if device == 'auto':
            actual_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cfg['training']['device'] = actual_device

        # Create unique checkpoint directory for this trial
        trial_dir = Path(cfg['training']['checkpoint_dir']).parent / 'optuna_trials' / f'trial_{trial.number}'
        cfg['training']['checkpoint_dir'] = str(trial_dir)
        trial_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"\n{'='*60}")
            print(f"TRIAL {trial.number}")
            print(f"{'='*60}")
            print(f"  hidden_sizes: {hidden_sizes}")
            print(f"  dropout: {dropout:.4f}")
            print(f"  use_batchnorm: {use_batchnorm}")
            print(f"  scenario_embedding_dim: {scenario_embedding_dim}")
            print(f"  learning_rate: {learning_rate:.6f}")
            print(f"  epochs: {cfg['training']['epochs']}")
            print(f"  device: {actual_device}")

        try:
            # =============================================================
            # RUN TRAINING
            # =============================================================

            # Get selected processes
            process_names = cfg.get('process_names', None)
            selected_processes = get_filtered_processes(process_names)

            # Generate trajectories
            n_train = cfg['scenarios']['n_train']
            target_trajectory = generate_target_trajectory(
                process_configs=selected_processes,
                n_samples=n_train,
                seed=cfg['scenarios']['seed_target']
            )

            baseline_trajectory = generate_baseline_trajectory(
                process_configs=selected_processes,
                target_trajectory=target_trajectory,
                n_samples=n_train,
                seed=cfg['scenarios']['seed_baseline']
            )

            # Create ProcessChain
            process_chain = ProcessChain(
                processes_config=selected_processes,
                target_trajectory=target_trajectory,
                policy_config=cfg['policy_generator'],
                device=actual_device
            )

            # Create Surrogate
            surrogate = ProTSurrogate(
                target_trajectory=target_trajectory,
                device=actual_device,
                use_deterministic_sampling=cfg.get('surrogate', {}).get('use_deterministic_sampling', True)
            )

            # Get curriculum config
            curriculum_config = cfg['training'].get('curriculum_learning', {
                'enabled': False,
                'warmup_fraction': 0.1,
                'lambda_bc_start': 10.0,
                'lambda_bc_end': 0.001,
                'reliability_weight_curve': 'exponential'
            })

            lr_scheduler_config = cfg['training'].get('lr_scheduler', None)

            # Create Trainer
            trainer = ControllerTrainer(
                process_chain=process_chain,
                surrogate=surrogate,
                lambda_bc=cfg['training']['lambda_bc'],
                learning_rate=learning_rate,
                weight_decay=cfg['training']['weight_decay'],
                reliability_loss_scale=cfg['training']['reliability_loss_scale'],
                device=actual_device,
                curriculum_config=curriculum_config,
                lr_scheduler_config=lr_scheduler_config
            )

            # =============================================================
            # TRAINING LOOP WITH PRUNING
            # =============================================================

            epochs = cfg['training']['epochs']
            batch_size = cfg['training']['batch_size']
            patience = cfg['training']['patience']

            # Calculate warmup epochs for curriculum learning
            warmup_epochs = 0
            if curriculum_config.get('enabled', False):
                warmup_epochs = int(epochs * curriculum_config['warmup_fraction'])

            best_loss = float('inf')
            epochs_without_improvement = 0

            for epoch in range(1, epochs + 1):
                # Get dynamic loss weights
                lambda_bc, reliability_weight, phase = trainer.get_loss_weights(epoch, epochs)

                # Train one epoch
                avg_total_loss, avg_rel_loss, avg_bc_loss, avg_F = trainer.train_epoch(
                    batch_size=batch_size,
                    reliability_weight=reliability_weight,
                    lambda_bc=lambda_bc
                )

                # Track history
                trainer.history['total_loss'].append(avg_total_loss)
                trainer.history['reliability_loss'].append(avg_rel_loss)
                trainer.history['bc_loss'].append(avg_bc_loss)
                trainer.history['F_values'].append(avg_F)

                # Step scheduler
                if trainer.scheduler is not None:
                    from torch.optim.lr_scheduler import ReduceLROnPlateau
                    if isinstance(trainer.scheduler, ReduceLROnPlateau):
                        trainer.scheduler.step(avg_total_loss)
                    else:
                        trainer.scheduler.step()

                # Report intermediate value for pruning
                # Only report after warm-up phase when reliability loss is active
                if epoch > warmup_epochs:
                    trial.report(avg_total_loss, epoch)

                    # Check if trial should be pruned
                    if trial.should_prune():
                        if verbose:
                            print(f"  Trial {trial.number} pruned at epoch {epoch}")
                        raise optuna.TrialPruned()

                # Early stopping logic
                patience_active = epoch > warmup_epochs and reliability_weight >= 0.9

                if patience_active:
                    if avg_total_loss < best_loss:
                        best_loss = avg_total_loss
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1

                    if epochs_without_improvement >= patience:
                        if verbose:
                            print(f"  Early stopping at epoch {epoch}")
                        break

                # Print progress occasionally
                if verbose and epoch % 100 == 0:
                    print(f"  Epoch {epoch}/{epochs}: loss={avg_total_loss:.6f}, F={avg_F:.6f}")

            # =============================================================
            # FINAL EVALUATION
            # =============================================================

            # Get final loss (average of last 10 epochs for stability)
            final_losses = trainer.history['total_loss'][-10:]
            final_loss = np.mean(final_losses)

            if verbose:
                print(f"  Final loss: {final_loss:.6f}")
                print(f"  Final F: {trainer.history['F_values'][-1]:.6f}")

            # Save trial results
            trial_results = {
                'trial_number': trial.number,
                'params': trial.params,
                'final_loss': float(final_loss),
                'final_F': float(trainer.history['F_values'][-1]),
                'epochs_run': len(trainer.history['total_loss']),
                'history': {
                    'total_loss': [float(x) for x in trainer.history['total_loss']],
                    'F_values': [float(x) for x in trainer.history['F_values']],
                }
            }

            with open(trial_dir / 'trial_results.json', 'w') as f:
                json.dump(trial_results, f, indent=2)

            return final_loss

        except Exception as e:
            if verbose:
                print(f"  Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()

    return objective


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def generate_optuna_report(study: optuna.Study, output_dir: Path, verbose: bool = True):
    """
    Generate comprehensive report with Optuna visualizations.

    Args:
        study: Completed Optuna study
        output_dir: Directory to save report
        verbose: Print progress
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nGenerating Optuna report in {output_dir}...")

    # Get completed trials
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

    if len(completed_trials) == 0:
        print("  No completed trials found!")
        return

    # =================================================================
    # 1. OPTIMIZATION HISTORY
    # =================================================================
    try:
        fig = plot_optimization_history(study)
        fig.write_image(str(output_dir / 'optimization_history.png'), scale=2)
        fig.write_html(str(output_dir / 'optimization_history.html'))
        if verbose:
            print("  - Optimization history saved")
    except Exception as e:
        print(f"  Warning: Could not generate optimization history: {e}")

    # =================================================================
    # 2. PARAMETER IMPORTANCES
    # =================================================================
    try:
        if len(completed_trials) >= 2:
            fig = plot_param_importances(study)
            fig.write_image(str(output_dir / 'param_importances.png'), scale=2)
            fig.write_html(str(output_dir / 'param_importances.html'))
            if verbose:
                print("  - Parameter importances saved")
    except Exception as e:
        print(f"  Warning: Could not generate parameter importances: {e}")

    # =================================================================
    # 3. PARALLEL COORDINATE PLOT
    # =================================================================
    try:
        fig = plot_parallel_coordinate(study)
        fig.write_image(str(output_dir / 'parallel_coordinate.png'), scale=2)
        fig.write_html(str(output_dir / 'parallel_coordinate.html'))
        if verbose:
            print("  - Parallel coordinate plot saved")
    except Exception as e:
        print(f"  Warning: Could not generate parallel coordinate plot: {e}")

    # =================================================================
    # 4. SLICE PLOTS
    # =================================================================
    try:
        fig = plot_slice(study)
        fig.write_image(str(output_dir / 'slice_plot.png'), scale=2)
        fig.write_html(str(output_dir / 'slice_plot.html'))
        if verbose:
            print("  - Slice plot saved")
    except Exception as e:
        print(f"  Warning: Could not generate slice plot: {e}")

    # =================================================================
    # 5. CONTOUR PLOTS (for pairs of important parameters)
    # =================================================================
    try:
        # Only generate for top 2 parameters if we have enough trials
        if len(completed_trials) >= 5:
            fig = plot_contour(study, params=['learning_rate', 'dropout'])
            fig.write_image(str(output_dir / 'contour_lr_dropout.png'), scale=2)
            fig.write_html(str(output_dir / 'contour_lr_dropout.html'))
            if verbose:
                print("  - Contour plot (lr vs dropout) saved")
    except Exception as e:
        print(f"  Warning: Could not generate contour plot: {e}")

    # =================================================================
    # 6. SUMMARY REPORT (TEXT)
    # =================================================================
    summary_path = output_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("OPTUNA HYPERPARAMETER OPTIMIZATION SUMMARY\n")
        f.write("="*70 + "\n\n")

        f.write(f"Study name: {study.study_name}\n")
        f.write(f"Direction: {study.direction.name}\n")
        f.write(f"Total trials: {len(study.trials)}\n")
        f.write(f"Completed trials: {len(completed_trials)}\n")
        f.write(f"Pruned trials: {len([t for t in study.trials if t.state == TrialState.PRUNED])}\n")
        f.write(f"Failed trials: {len([t for t in study.trials if t.state == TrialState.FAIL])}\n\n")

        f.write("-"*70 + "\n")
        f.write("BEST TRIAL\n")
        f.write("-"*70 + "\n\n")

        best_trial = study.best_trial
        f.write(f"Trial number: {best_trial.number}\n")
        f.write(f"Best value (loss): {best_trial.value:.6f}\n\n")

        f.write("Best hyperparameters:\n")
        for key, value in best_trial.params.items():
            f.write(f"  {key}: {value}\n")

        f.write("\n" + "-"*70 + "\n")
        f.write("TOP 5 TRIALS\n")
        f.write("-"*70 + "\n\n")

        sorted_trials = sorted(completed_trials, key=lambda t: t.value)[:5]
        for i, trial in enumerate(sorted_trials, 1):
            f.write(f"{i}. Trial {trial.number}: loss={trial.value:.6f}\n")
            for key, value in trial.params.items():
                f.write(f"     {key}: {value}\n")
            f.write("\n")

    if verbose:
        print(f"  - Summary saved to {summary_path}")

    # =================================================================
    # 7. BEST PARAMS JSON
    # =================================================================
    best_params_path = output_dir / 'best_params.json'
    best_params = {
        'best_trial_number': study.best_trial.number,
        'best_value': study.best_trial.value,
        'best_params': study.best_trial.params,
        'study_stats': {
            'n_trials': len(study.trials),
            'n_completed': len(completed_trials),
            'n_pruned': len([t for t in study.trials if t.state == TrialState.PRUNED]),
        }
    }

    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=2)

    if verbose:
        print(f"  - Best params saved to {best_params_path}")
        print(f"\n{'='*70}")
        print("BEST HYPERPARAMETERS FOUND")
        print(f"{'='*70}")
        print(f"Best loss: {study.best_trial.value:.6f}")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")


def print_study_status(study: optuna.Study):
    """Print current status of an Optuna study."""
    trials = study.trials
    completed = [t for t in trials if t.state == TrialState.COMPLETE]
    running = [t for t in trials if t.state == TrialState.RUNNING]
    pruned = [t for t in trials if t.state == TrialState.PRUNED]
    failed = [t for t in trials if t.state == TrialState.FAIL]

    print(f"\n{'='*60}")
    print(f"STUDY STATUS: {study.study_name}")
    print(f"{'='*60}")
    print(f"Total trials:     {len(trials)}")
    print(f"Completed:        {len(completed)}")
    print(f"Running:          {len(running)}")
    print(f"Pruned:           {len(pruned)}")
    print(f"Failed:           {len(failed)}")

    if len(completed) > 0:
        best_trial = study.best_trial
        print(f"\nBest trial so far:")
        print(f"  Trial {best_trial.number}: loss={best_trial.value:.6f}")
        print(f"  Params: {best_trial.params}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Optuna Hyperparameter Tuning for Controller Optimization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Execution mode
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of trials for local execution')
    parser.add_argument('--create-study', action='store_true',
                        help='Create a new study (for distributed execution)')
    parser.add_argument('--single-trial', action='store_true',
                        help='Run a single trial (for SLURM job array)')
    parser.add_argument('--status', action='store_true',
                        help='Print study status')
    parser.add_argument('--report', action='store_true',
                        help='Generate report from completed study')

    # Study configuration
    parser.add_argument('--study-name', type=str, default='controller_hpo',
                        help='Name of the Optuna study')
    parser.add_argument('--storage', type=str, default=None,
                        help='SQLite storage path (default: optuna_results/<study_name>.db)')

    # Training configuration
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for training')
    parser.add_argument('--reduced-epochs', type=int, default=None,
                        help='Use fewer epochs for faster trials (default: use config value)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed training progress')

    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')

    args = parser.parse_args()

    if not OPTUNA_AVAILABLE:
        print("Error: Optuna is required. Install with: pip install optuna")
        sys.exit(1)

    # Set up paths
    base_dir = Path(__file__).parent
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = base_dir / 'optuna_results' / args.study_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up storage
    if args.storage:
        storage = args.storage
    else:
        storage = f"sqlite:///{output_dir / 'study.db'}"

    # =================================================================
    # CREATE STUDY MODE
    # =================================================================
    if args.create_study:
        print(f"Creating new study: {args.study_name}")
        print(f"Storage: {storage}")

        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            direction='minimize',
            load_if_exists=False,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=100,
                interval_steps=10
            )
        )

        print(f"Study created successfully!")
        print(f"\nTo run optimization on Euler, use:")
        print(f"  sbatch optuna_sweep.sh")
        return

    # =================================================================
    # STATUS MODE
    # =================================================================
    if args.status:
        try:
            study = optuna.load_study(
                study_name=args.study_name,
                storage=storage
            )
            print_study_status(study)
        except Exception as e:
            print(f"Error loading study: {e}")
        return

    # =================================================================
    # REPORT MODE
    # =================================================================
    if args.report:
        try:
            study = optuna.load_study(
                study_name=args.study_name,
                storage=storage
            )
            generate_optuna_report(study, output_dir)
        except Exception as e:
            print(f"Error generating report: {e}")
            import traceback
            traceback.print_exc()
        return

    # =================================================================
    # SINGLE TRIAL MODE (for SLURM jobs)
    # =================================================================
    if args.single_trial:
        print(f"Running single trial for study: {args.study_name}")

        study = optuna.load_study(
            study_name=args.study_name,
            storage=storage
        )

        objective = create_objective(
            base_config=CONTROLLER_CONFIG,
            device=args.device,
            reduced_epochs=args.reduced_epochs,
            verbose=args.verbose
        )

        study.optimize(objective, n_trials=1)

        print(f"Trial completed. Current best: {study.best_trial.value:.6f}")
        return

    # =================================================================
    # LOCAL EXECUTION MODE (default)
    # =================================================================
    print(f"{'='*70}")
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print(f"{'='*70}")
    print(f"Study name: {args.study_name}")
    print(f"Storage: {storage}")
    print(f"N trials: {args.n_trials}")
    print(f"Device: {args.device}")
    print(f"Output dir: {output_dir}")

    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction='minimize',
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=100,
            interval_steps=10
        )
    )

    # Calculate remaining trials
    completed_trials = len([t for t in study.trials if t.state == TrialState.COMPLETE])
    remaining_trials = max(0, args.n_trials - completed_trials)

    if remaining_trials == 0:
        print(f"\nStudy already has {completed_trials} completed trials.")
        print("Generating report...")
        generate_optuna_report(study, output_dir)
        return

    print(f"\nCompleted trials: {completed_trials}")
    print(f"Remaining trials: {remaining_trials}")

    # Create objective function
    objective = create_objective(
        base_config=CONTROLLER_CONFIG,
        device=args.device,
        reduced_epochs=args.reduced_epochs,
        verbose=args.verbose
    )

    # Run optimization
    print(f"\nStarting optimization...")
    study.optimize(
        objective,
        n_trials=remaining_trials,
        show_progress_bar=True
    )

    # Generate report
    print(f"\nOptimization completed!")
    generate_optuna_report(study, output_dir)


if __name__ == '__main__':
    main()
