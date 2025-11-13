"""
Visualizzazioni per controller optimization.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_training_history(history, save_path=None):
    """
    Plot training history (total loss, reliability loss, BC loss, F values).

    Supports both old format (single metrics) and new format (train/val split).

    Args:
        history (dict): Training history
        save_path (str): Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Determine if we have train/val split
    has_validation = 'val_total_loss' in history and len(history['val_total_loss']) > 0

    # Total loss
    if has_validation:
        axes[0, 0].plot(history['train_total_loss'], label='Train Total Loss', color='blue')
        axes[0, 0].plot(history['val_total_loss'], label='Val Total Loss', color='cyan', linestyle='--')
    else:
        # Backward compatibility: check both old and new format
        loss_key = 'train_total_loss' if 'train_total_loss' in history else 'total_loss'
        axes[0, 0].plot(history[loss_key], label='Total Loss', color='blue')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Reliability loss
    if has_validation:
        axes[0, 1].plot(history['train_reliability_loss'], label='Train Reliability Loss', color='red')
        axes[0, 1].plot(history['val_reliability_loss'], label='Val Reliability Loss', color='orange', linestyle='--')
    else:
        rel_loss_key = 'train_reliability_loss' if 'train_reliability_loss' in history else 'reliability_loss'
        axes[0, 1].plot(history[rel_loss_key], label='Reliability Loss', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Reliability Loss (F - F*)²')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # BC loss
    if has_validation:
        axes[1, 0].plot(history['train_bc_loss'], label='Train BC Loss', color='green')
        axes[1, 0].plot(history['val_bc_loss'], label='Val BC Loss', color='lime', linestyle='--')
    else:
        bc_loss_key = 'train_bc_loss' if 'train_bc_loss' in history else 'bc_loss'
        axes[1, 0].plot(history[bc_loss_key], label='Behavior Cloning Loss', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Behavior Cloning Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # F values
    F_values_key = 'train_F_values' if 'train_F_values' in history else 'F_values'
    if F_values_key in history:
        axes[1, 1].plot(history[F_values_key], label='F (Train)', color='purple')
        if has_validation:
            axes[1, 1].plot(history['val_F_values'], label='F (Val)', color='magenta', linestyle='--')
        if 'F_star' in history:
            axes[1, 1].axhline(y=history['F_star'], color='gold', linestyle='--', label='F* (Target)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Reliability F')
        axes[1, 1].set_title('Reliability Evolution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_trajectory_comparison(target_trajectory, baseline_trajectories,
                               actual_trajectories, save_path=None):
    """
    Plot confronto tra le trajectories:
    - a* (target, verde) - single trajectory
    - a' (baseline, rosso) - multiple trajectories (list)
    - a (actual, blu) - multiple trajectories (list)

    Un subplot separato per ogni singolo input e output di ogni processo.

    Args:
        target_trajectory: Single target trajectory (dict)
        baseline_trajectories: List of baseline trajectories (list of dicts)
        actual_trajectories: List of actual trajectories (list of dicts)
        save_path: Path to save the plot
    """
    from controller_optimization.src.utils.metrics import convert_trajectory_to_numpy
    from controller_optimization.configs.processes_config import get_process_by_name

    # Convert target to numpy
    target = convert_trajectory_to_numpy(target_trajectory)

    # Convert baseline trajectories to numpy (list of dicts)
    baselines = [convert_trajectory_to_numpy(baseline_traj) for baseline_traj in baseline_trajectories]

    # Convert actual trajectories to numpy (list of dicts)
    actuals = [convert_trajectory_to_numpy(actual_traj) for actual_traj in actual_trajectories]

    process_names = list(target.keys())

    # Calculate total number of subplots needed
    total_subplots = 0
    process_subplot_info = []

    for process_name in process_names:
        process_config = get_process_by_name(process_name)
        n_inputs = process_config['input_dim']
        n_outputs = process_config['output_dim']
        n_plots_for_process = n_inputs + n_outputs

        process_subplot_info.append({
            'name': process_name,
            'config': process_config,
            'n_inputs': n_inputs,
            'n_outputs': n_outputs,
            'n_plots': n_plots_for_process,
            'start_idx': total_subplots
        })
        total_subplots += n_plots_for_process

    # Create figure with dynamic number of subplots
    n_cols = 3  # 3 columns for better layout
    n_rows = int(np.ceil(total_subplots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = axes.flatten() if total_subplots > 1 else [axes]

    subplot_idx = 0

    for proc_info in process_subplot_info:
        process_name = proc_info['name']
        process_config = proc_info['config']
        n_inputs = proc_info['n_inputs']
        n_outputs = proc_info['n_outputs']

        target_inputs = target[process_name]['inputs']  # shape: (n_samples, input_dim)
        target_outputs = target[process_name]['outputs_mean']  # shape: (n_samples, output_dim)

        n_samples = target_inputs.shape[0]
        x = np.arange(n_samples)

        # Plot each input dimension separately
        for i in range(n_inputs):
            ax = axes[subplot_idx]

            input_label = process_config['input_labels'][i] if i < len(process_config['input_labels']) else f'Input {i}'

            # Plot target (single trajectory, solid line)
            ax.plot(x, target_inputs[:, i], 'o-', color='green', label='Target (a*)',
                   linewidth=2.5, markersize=5, zorder=10)

            # Plot multiple baseline trajectories (semi-transparent)
            for j, baseline in enumerate(baselines):
                baseline_inputs = baseline[process_name]['inputs']
                label = 'Baseline (a\')' if j == 0 else None  # Only label first one
                ax.plot(x, baseline_inputs[:, i], 's--', color='red',
                       linewidth=1.5, alpha=0.4, markersize=3, label=label, zorder=5)

            # Plot multiple actual trajectories (semi-transparent)
            for j, actual in enumerate(actuals):
                actual_inputs = actual[process_name]['inputs']
                label = 'Controller (a)' if j == 0 else None  # Only label first one
                ax.plot(x, actual_inputs[:, i], '^-', color='blue',
                       linewidth=1.5, alpha=0.4, markersize=3, label=label, zorder=5)

            ax.set_title(f'{process_name.capitalize()} - {input_label}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel(input_label)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

            subplot_idx += 1

        # Plot each output dimension separately
        for i in range(n_outputs):
            ax = axes[subplot_idx]

            output_label = process_config['output_labels'][i] if i < len(process_config['output_labels']) else f'Output {i}'

            # Plot target (single trajectory, solid line)
            ax.plot(x, target_outputs[:, i], 'o-', color='green', label='Target (a*)',
                   linewidth=2.5, markersize=5, zorder=10)

            # Plot multiple baseline trajectories (semi-transparent)
            for j, baseline in enumerate(baselines):
                baseline_outputs = baseline[process_name]['outputs_mean']
                label = 'Baseline (a\')' if j == 0 else None  # Only label first one
                ax.plot(x, baseline_outputs[:, i], 's--', color='red',
                       linewidth=1.5, alpha=0.4, markersize=3, label=label, zorder=5)

            # Plot multiple actual trajectories (semi-transparent)
            for j, actual in enumerate(actuals):
                actual_outputs = actual[process_name]['outputs_mean']
                label = 'Controller (a)' if j == 0 else None  # Only label first one
                ax.plot(x, actual_outputs[:, i], '^-', color='blue',
                       linewidth=1.5, alpha=0.4, markersize=3, label=label, zorder=5)

            ax.set_title(f'{process_name.capitalize()} - {output_label}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel(output_label)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

            subplot_idx += 1

    # Hide any unused subplots
    for idx in range(subplot_idx, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_reliability_comparison(F_star, F_baseline, F_actual, save_path=None):
    """
    Bar chart confronto reliability:
    - F* (target)
    - F' (baseline)
    - F (actual)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = ['Target\n(F*)', 'Baseline\n(F\')', 'Controller\n(F)']
    values = [F_star, F_baseline, F_actual]
    colors = ['green', 'red', 'blue']

    bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{value:.6f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add improvement annotation
    if F_baseline != 0:
        improvement = ((F_actual - F_baseline) / F_baseline) * 100
        ax.text(2, F_actual * 0.5, f'Improvement:\n{improvement:+.2f}%',
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    ax.set_ylabel('Reliability Score (F)', fontsize=14)
    ax.set_title('Reliability Comparison', fontsize=16, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_process_improvements(process_metrics, save_path=None):
    """
    Plot miglioramenti per processo:
    - MSE improvement
    - Output accuracy improvement
    """
    process_names = list(process_metrics['baseline'].keys())
    n_processes = len(process_names)

    baseline_mse = [process_metrics['baseline'][p]['combined_mse'] for p in process_names]
    actual_mse = [process_metrics['actual'][p]['combined_mse'] for p in process_names]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(n_processes)
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_mse, width, label='Baseline', color='red', alpha=0.7)
    bars2 = ax.bar(x + width/2, actual_mse, width, label='Controller', color='blue', alpha=0.7)

    ax.set_xlabel('Process', fontsize=14)
    ax.set_ylabel('Combined MSE', fontsize=14)
    ax.set_title('Process-wise MSE Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in process_names])
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved: {save_path}")
    else:
        plt.show()

    plt.close()
