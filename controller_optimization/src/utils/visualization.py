"""
Visualizzazioni per controller optimization.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_training_history(history, save_path=None):
    """
    Plot training history (total loss, reliability loss, BC loss, F values).

    Args:
        history (dict): Training history
        save_path (str): Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Total loss
    axes[0, 0].plot(history['total_loss'], label='Total Loss', color='blue')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Reliability loss
    axes[0, 1].plot(history['reliability_loss'], label='Reliability Loss', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Reliability Loss (F - F*)²')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # BC loss
    axes[1, 0].plot(history['bc_loss'], label='Behavior Cloning Loss', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Behavior Cloning Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # F values
    if 'F_values' in history:
        axes[1, 1].plot(history['F_values'], label='F (Actual)', color='purple')
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


def plot_trajectory_comparison(target_trajectory, baseline_trajectory,
                               actual_trajectory, save_path=None):
    """
    Plot confronto tra le 3 trajectories:
    - a* (target, verde)
    - a' (baseline, rosso)
    - a (actual, blu)

    Un subplot per ogni processo, mostrando inputs e outputs.
    """
    from controller_optimization.src.utils.metrics import convert_trajectory_to_numpy

    # Convert to numpy
    target = convert_trajectory_to_numpy(target_trajectory)
    baseline = convert_trajectory_to_numpy(baseline_trajectory)
    actual = convert_trajectory_to_numpy(actual_trajectory)

    process_names = list(target.keys())
    n_processes = len(process_names)

    fig, axes = plt.subplots(n_processes, 2, figsize=(14, 5 * n_processes))

    if n_processes == 1:
        axes = axes.reshape(1, -1)

    for i, process_name in enumerate(process_names):
        # Inputs plot
        ax_in = axes[i, 0]
        target_inputs = target[process_name]['inputs'].flatten()
        baseline_inputs = baseline[process_name]['inputs'].flatten()
        actual_inputs = actual[process_name]['inputs'].flatten()

        x = np.arange(len(target_inputs))
        ax_in.plot(x, target_inputs, 'o-', color='green', label='Target (a*)', linewidth=2)
        ax_in.plot(x, baseline_inputs, 's--', color='red', label='Baseline (a\')', linewidth=2, alpha=0.7)
        ax_in.plot(x, actual_inputs, '^-', color='blue', label='Controller (a)', linewidth=2, alpha=0.7)

        ax_in.set_title(f'{process_name.capitalize()} - Inputs')
        ax_in.set_xlabel('Sample Index')
        ax_in.set_ylabel('Input Value')
        ax_in.legend()
        ax_in.grid(True, alpha=0.3)

        # Outputs plot
        ax_out = axes[i, 1]
        target_outputs = target[process_name]['outputs_mean'].flatten()
        baseline_outputs = baseline[process_name]['outputs_mean'].flatten()
        actual_outputs = actual[process_name]['outputs_mean'].flatten()

        ax_out.plot(x, target_outputs, 'o-', color='green', label='Target (a*)', linewidth=2)
        ax_out.plot(x, baseline_outputs, 's--', color='red', label='Baseline (a\')', linewidth=2, alpha=0.7)
        ax_out.plot(x, actual_outputs, '^-', color='blue', label='Controller (a)', linewidth=2, alpha=0.7)

        ax_out.set_title(f'{process_name.capitalize()} - Outputs')
        ax_out.set_xlabel('Sample Index')
        ax_out.set_ylabel('Output Value')
        ax_out.legend()
        ax_out.grid(True, alpha=0.3)

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
