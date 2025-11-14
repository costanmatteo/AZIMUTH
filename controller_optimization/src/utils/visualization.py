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

    Un subplot separato per ogni singolo input e output di ogni processo.
    """
    from controller_optimization.src.utils.metrics import convert_trajectory_to_numpy
    from controller_optimization.configs.processes_config import get_process_by_name

    # Convert to numpy
    target = convert_trajectory_to_numpy(target_trajectory)
    baseline = convert_trajectory_to_numpy(baseline_trajectory)
    actual = convert_trajectory_to_numpy(actual_trajectory)

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
        baseline_inputs = baseline[process_name]['inputs']
        actual_inputs = actual[process_name]['inputs']

        target_outputs = target[process_name]['outputs_mean']  # shape: (n_samples, output_dim)
        baseline_outputs = baseline[process_name]['outputs_mean']
        actual_outputs = actual[process_name]['outputs_mean']

        n_samples = target_inputs.shape[0]
        x = np.arange(n_samples)

        # Plot each input dimension separately
        for i in range(n_inputs):
            ax = axes[subplot_idx]

            input_label = process_config['input_labels'][i] if i < len(process_config['input_labels']) else f'Input {i}'

            ax.plot(x, target_inputs[:, i], 'o-', color='green', label='Target (a*)', linewidth=2, markersize=4)
            ax.plot(x, baseline_inputs[:, i], 's--', color='red', label='Baseline (a\')', linewidth=2, alpha=0.7, markersize=4)
            ax.plot(x, actual_inputs[:, i], '^-', color='blue', label='Controller (a)', linewidth=2, alpha=0.7, markersize=4)

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

            ax.plot(x, target_outputs[:, i], 'o-', color='green', label='Target (a*)', linewidth=2, markersize=4)
            ax.plot(x, baseline_outputs[:, i], 's--', color='red', label='Baseline (a\')', linewidth=2, alpha=0.7, markersize=4)
            ax.plot(x, actual_outputs[:, i], '^-', color='blue', label='Controller (a)', linewidth=2, alpha=0.7, markersize=4)

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


def plot_target_vs_actual_scatter(F_star_per_scenario, F_actual_per_scenario, save_path=None):
    """
    Scatter plot: F_star (target) vs F_actual (controller).

    Shows how well the controller matches the target for each scenario.
    Points on the diagonal line (y=x) indicate perfect matching.
    Color indicates the gap (F_star - F_actual).

    Works with both single and multiple scenarios.

    Args:
        F_star_per_scenario (array-like): Target reliability for each scenario
        F_actual_per_scenario (array-like): Actual reliability for each scenario
        save_path (str): Path to save figure
    """
    F_star_arr = np.atleast_1d(F_star_per_scenario)
    F_actual_arr = np.atleast_1d(F_actual_per_scenario)

    # Compute gaps for colormap
    gaps = F_star_arr - F_actual_arr

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot with colormap (green -> yellow -> red)
    scatter = ax.scatter(F_star_arr, F_actual_arr,
                        c=gaps,
                        cmap='RdYlGn_r',  # Reverse: green (small gap) -> red (large gap)
                        s=100,
                        alpha=0.7,
                        edgecolors='black',
                        linewidths=1.5)

    # Diagonal line (perfect match: F_actual = F_star)
    min_val = min(F_star_arr.min(), F_actual_arr.min())
    max_val = max(F_star_arr.max(), F_actual_arr.max())
    margin = (max_val - min_val) * 0.1
    ax.plot([min_val - margin, max_val + margin],
            [min_val - margin, max_val + margin],
            'k--', linewidth=2, label='Perfect Match (F_actual = F_star)', alpha=0.5)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Gap (F_star - F_actual)', fontsize=12, fontweight='bold')

    # Axis labels and title
    ax.set_xlabel('F_star (Target Reliability)', fontsize=14, fontweight='bold')
    ax.set_ylabel('F_actual (Controller Reliability)', fontsize=14, fontweight='bold')
    ax.set_title('Target vs Actual Reliability', fontsize=16, fontweight='bold')

    # Equal aspect ratio and grid
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=11)

    # Set limits with margin
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)

    # Add statistics annotation
    mean_gap = np.mean(gaps)
    worst_gap = np.max(gaps)
    n_scenarios = len(F_star_arr)

    stats_text = f'Scenarios: {n_scenarios}\nMean Gap: {mean_gap:.6f}\nWorst Gap: {worst_gap:.6f}'
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_gap_distribution(F_star_per_scenario, F_actual_per_scenario, save_path=None):
    """
    Histogram of gap distribution: F_star - F_actual.

    Shows the distribution of gaps between target and actual reliability.
    Includes mean, median, and worst-case annotations.

    Works with both single and multiple scenarios.

    Args:
        F_star_per_scenario (array-like): Target reliability for each scenario
        F_actual_per_scenario (array-like): Actual reliability for each scenario
        save_path (str): Path to save figure
    """
    F_star_arr = np.atleast_1d(F_star_per_scenario)
    F_actual_arr = np.atleast_1d(F_actual_per_scenario)

    # Compute gaps
    gaps = F_star_arr - F_actual_arr

    fig, ax = plt.subplots(figsize=(12, 7))

    # Histogram
    n_scenarios = len(gaps)

    # Determine number of bins (adaptive)
    if n_scenarios == 1:
        # Single scenario: show as single bar
        n_bins = 1
    elif n_scenarios < 10:
        n_bins = n_scenarios
    else:
        n_bins = min(20, n_scenarios // 2)

    counts, bins, patches = ax.hist(gaps, bins=n_bins,
                                     color='steelblue',
                                     alpha=0.7,
                                     edgecolor='black',
                                     linewidth=1.5)

    # Color bars based on gap value (green = small, red = large)
    gap_max = gaps.max()
    gap_min = gaps.min()
    gap_range = gap_max - gap_min if gap_max > gap_min else 1.0

    for patch, bin_left in zip(patches, bins[:-1]):
        # Normalize bin position to [0, 1]
        normalized_pos = (bin_left - gap_min) / gap_range if gap_range > 0 else 0
        # Green (0) -> Yellow (0.5) -> Red (1)
        if normalized_pos < 0.5:
            # Green to yellow
            r = 2 * normalized_pos
            g = 1.0
        else:
            # Yellow to red
            r = 1.0
            g = 2 * (1 - normalized_pos)
        color = (r, g, 0.0)
        patch.set_facecolor(color)

    # Statistics
    mean_gap = np.mean(gaps)
    median_gap = np.median(gaps)
    worst_gap = np.max(gaps)
    best_gap = np.min(gaps)
    std_gap = np.std(gaps)

    # Add vertical lines for statistics
    y_max = ax.get_ylim()[1]

    ax.axvline(mean_gap, color='darkblue', linestyle='--', linewidth=2,
               label=f'Mean: {mean_gap:.6f}')
    ax.axvline(median_gap, color='purple', linestyle='--', linewidth=2,
               label=f'Median: {median_gap:.6f}')
    ax.axvline(worst_gap, color='darkred', linestyle='--', linewidth=2,
               label=f'Worst: {worst_gap:.6f}')

    # Labels and title
    ax.set_xlabel('Gap (F_star - F_actual)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Scenarios', fontsize=14, fontweight='bold')
    ax.set_title('Distribution of Target-Actual Gap', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    # Add statistics box
    stats_text = (f'Scenarios: {n_scenarios}\n'
                 f'Mean: {mean_gap:.6f}\n'
                 f'Std: {std_gap:.6f}\n'
                 f'Median: {median_gap:.6f}\n'
                 f'Best: {best_gap:.6f}\n'
                 f'Worst: {worst_gap:.6f}')

    ax.text(0.98, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved: {save_path}")
    else:
        plt.show()

    plt.close()
