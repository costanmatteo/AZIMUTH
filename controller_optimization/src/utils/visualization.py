"""
Visualizzazioni per controller optimization.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_training_history(history, save_path=None):
    """
    Plot training history (total loss, reliability loss, BC loss, F values, curriculum weights).

    Args:
        history (dict): Training history
        save_path (str): Path to save figure
    """
    # Check if curriculum learning weights are available
    has_curriculum_weights = 'lambda_bc' in history and 'reliability_weight' in history

    # Create figure with appropriate number of subplots
    if has_curriculum_weights:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes = axes.flatten()

    # Total loss
    axes[0].plot(history['total_loss'], label='Total Loss', color='blue')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Reliability loss
    axes[1].plot(history['reliability_loss'], label='Reliability Loss', color='red')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Reliability Loss (F - F*)²')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # BC loss
    axes[2].plot(history['bc_loss'], label='Behavior Cloning Loss', color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Behavior Cloning Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # F values
    if 'F_values' in history:
        axes[3].plot(history['F_values'], label='F (Actual)', color='purple')
        if 'F_star' in history:
            axes[3].axhline(y=history['F_star'], color='gold', linestyle='--', label='F* (Target)')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('Reliability F')
        axes[3].set_title('Reliability Evolution')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

    # Curriculum learning weights (if available)
    if has_curriculum_weights:
        # Lambda BC (log scale)
        axes[4].plot(history['lambda_bc'], label='λ_BC', color='orange', linewidth=2)
        axes[4].set_xlabel('Epoch')
        axes[4].set_ylabel('λ_BC')
        axes[4].set_title('Behavior Cloning Weight (λ_BC)')
        axes[4].set_yscale('log')
        axes[4].legend()
        axes[4].grid(True, alpha=0.3, which='both')

        # Reliability weight
        axes[5].plot(history['reliability_weight'], label='Reliability Weight', color='brown', linewidth=2)
        axes[5].set_xlabel('Epoch')
        axes[5].set_ylabel('Reliability Weight')
        axes[5].set_title('Reliability Loss Weight Schedule')
        axes[5].set_ylim(-0.05, 1.05)
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)

        # Add vertical line at end of warm-up (where reliability_weight starts to increase)
        if len(history['reliability_weight']) > 0:
            # Find first epoch where reliability_weight > 0
            warmup_end = 0
            for i, w in enumerate(history['reliability_weight']):
                if w > 0:
                    warmup_end = i
                    break

            if warmup_end > 0:
                axes[4].axvline(x=warmup_end, color='red', linestyle='--', alpha=0.5, label='Warm-up End')
                axes[5].axvline(x=warmup_end, color='red', linestyle='--', alpha=0.5, label='Warm-up End')

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


def plot_target_vs_actual_scatter(F_star_per_scenario, F_baseline_per_scenario, F_actual_per_scenario, save_path=None):
    """
    Scatter plot: F_star (target) vs F_baseline and F_actual.

    Y-axis: F_star (target reliability)
    X-axis: F_baseline (red) and F_actual (blue)
    Shows how well both baseline and controller match the target.

    Works with both single and multiple scenarios.

    Args:
        F_star_per_scenario (array-like): Target reliability for each scenario
        F_baseline_per_scenario (array-like): Baseline reliability for each scenario
        F_actual_per_scenario (array-like): Actual reliability for each scenario
        save_path (str): Path to save figure
    """
    F_star_arr = np.atleast_1d(F_star_per_scenario)
    F_baseline_arr = np.atleast_1d(F_baseline_per_scenario)
    F_actual_arr = np.atleast_1d(F_actual_per_scenario)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Scatter plots
    # Baseline points (red)
    ax.scatter(F_baseline_arr, F_star_arr,
               c='red',
               s=120,
               alpha=0.6,
               edgecolors='darkred',
               linewidths=2,
               label='Baseline (no controller)',
               marker='s')  # square

    # Actual points (blue)
    ax.scatter(F_actual_arr, F_star_arr,
               c='blue',
               s=120,
               alpha=0.6,
               edgecolors='darkblue',
               linewidths=2,
               label='Controller',
               marker='o')  # circle

    # Diagonal line (perfect match)
    all_values = np.concatenate([F_star_arr, F_baseline_arr, F_actual_arr])
    min_val = all_values.min()
    max_val = all_values.max()
    margin = (max_val - min_val) * 0.1

    ax.plot([min_val - margin, max_val + margin],
            [min_val - margin, max_val + margin],
            'k--', linewidth=2, label='Perfect Match (y = x)', alpha=0.5)

    # Axis labels and title
    ax.set_xlabel('F (Reliability)', fontsize=14, fontweight='bold')
    ax.set_ylabel('F_star (Target Reliability)', fontsize=14, fontweight='bold')
    ax.set_title('Target vs Baseline & Controller Reliability', fontsize=16, fontweight='bold')

    # Grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)

    # Set limits with margin
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)

    # Add statistics annotation
    n_scenarios = len(F_star_arr)
    gap_baseline = np.mean(F_star_arr - F_baseline_arr)
    gap_actual = np.mean(F_star_arr - F_actual_arr)
    improvement = ((gap_baseline - gap_actual) / abs(gap_baseline)) * 100 if gap_baseline != 0 else 0

    stats_text = (f'Scenarios: {n_scenarios}\n'
                 f'Mean Gap (Baseline): {gap_baseline:.6f}\n'
                 f'Mean Gap (Controller): {gap_actual:.6f}\n'
                 f'Improvement: {improvement:.1f}%')

    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

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
        normalized_pos = np.clip(normalized_pos, 0, 1)  # Ensure in [0, 1] range

        # Green (0) -> Yellow (0.5) -> Red (1)
        if normalized_pos < 0.5:
            # Green to yellow
            r = 2 * normalized_pos
            g = 1.0
        else:
            # Yellow to red
            r = 1.0
            g = 2 * (1 - normalized_pos)

        # Ensure RGB values are in [0, 1] range
        r = np.clip(r, 0, 1)
        g = np.clip(g, 0, 1)
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


def plot_training_progression(progression_path, save_path=None):
    """
    Plot training progression showing how inputs/outputs evolve through epochs.

    Shows snapshots at key epochs (start, warmup end, curriculum start, middle, end)
    to visualize how the controller learns.

    Args:
        progression_path (str): Path to training_progression.npz file
        save_path (str): Path to save figure
    """
    from controller_optimization.configs.processes_config import get_process_by_name

    # Load progression data
    data = np.load(progression_path, allow_pickle=True)

    # Extract snapshot information
    # Keys format: snapshot_0_epoch_1_epoch, snapshot_0_epoch_1_phase, etc.
    snapshots = []

    # Find all unique prefixes (snapshot_X_epoch_Y)
    prefixes = set()
    for key in data.files:
        # Extract prefix (everything before the last underscore + field name)
        # E.g., "snapshot_0_epoch_1_epoch" -> "snapshot_0_epoch_1"
        if key.count('_') >= 3:  # At least snapshot_X_epoch_Y_field
            parts = key.split('_')
            if len(parts) >= 4 and parts[0] == 'snapshot' and parts[2] == 'epoch':
                # Reconstruct prefix: snapshot_X_epoch_Y
                prefix = '_'.join(parts[:4])  # snapshot_X_epoch_Y
                prefixes.add(prefix)

    # Sort prefixes by epoch number
    def get_epoch_from_prefix(prefix):
        # Extract epoch from "snapshot_X_epoch_Y"
        parts = prefix.split('_')
        return int(parts[3])

    sorted_prefixes = sorted(prefixes, key=get_epoch_from_prefix)

    for prefix in sorted_prefixes:
        snapshot = {
            'epoch': int(data[f'{prefix}_epoch']),
            'phase': str(data[f'{prefix}_phase']),
            'lambda_bc': float(data[f'{prefix}_lambda_bc']),
            'reliability_weight': float(data[f'{prefix}_reliability_weight']),
            'F_actual': float(data[f'{prefix}_F_actual']),
            'F_star': float(data[f'{prefix}_F_star']),
            'processes': {}
        }

        # Extract process data
        for key in data.files:
            if key.startswith(prefix) and '_inputs' in key and '_target_inputs' not in key:
                # Extract process name from key like "snapshot_0_epoch_1_laser_inputs"
                # Remove prefix and "_inputs" suffix
                remaining = key.replace(f'{prefix}_', '')
                if remaining.endswith('_inputs'):
                    process_name = remaining.replace('_inputs', '')
                    snapshot['processes'][process_name] = {
                        'inputs': data[f'{prefix}_{process_name}_inputs'],
                        'outputs': data[f'{prefix}_{process_name}_outputs'],
                        'target_inputs': data[f'{prefix}_{process_name}_target_inputs'],
                        'target_outputs': data[f'{prefix}_{process_name}_target_outputs']
                    }

        snapshots.append(snapshot)

    if len(snapshots) == 0:
        print("No progression snapshots found")
        return

    # Count total plots needed
    n_snapshots = len(snapshots)
    process_names = list(snapshots[0]['processes'].keys())
    n_processes = len(process_names)

    # Get input/output dimensions for each process
    total_plots = 0
    process_dims = {}
    for process_name in process_names:
        config = get_process_by_name(process_name)
        n_inputs = config['input_dim']
        n_outputs = config['output_dim']
        process_dims[process_name] = {'n_inputs': n_inputs, 'n_outputs': n_outputs}
        total_plots += n_inputs + n_outputs

    # Create figure with subplots (one row per input/output, one column per snapshot)
    fig = plt.figure(figsize=(4 * n_snapshots, 3 * total_plots))

    plot_idx = 1

    for process_name in process_names:
        config = get_process_by_name(process_name)
        n_inputs = process_dims[process_name]['n_inputs']
        n_outputs = process_dims[process_name]['n_outputs']

        # Plot each input dimension
        for input_idx in range(n_inputs):
            input_label = config['input_labels'][input_idx] if input_idx < len(config['input_labels']) else f'Input {input_idx}'

            for snap_idx, snapshot in enumerate(snapshots):
                ax = plt.subplot(total_plots, n_snapshots, plot_idx)

                process_data = snapshot['processes'][process_name]
                actual_inputs = process_data['inputs'][0, :, input_idx]  # First sample
                target_inputs = process_data['target_inputs'][0, :, input_idx]

                x = np.arange(len(actual_inputs))
                ax.plot(x, target_inputs, 'o-', color='green', label='Target', linewidth=2, markersize=4, alpha=0.7)
                ax.plot(x, actual_inputs, '^-', color='blue', label='Actual', linewidth=2, markersize=4)

                # Title with epoch and phase info
                phase_label = f"[{snapshot['phase'].upper()}]" if snapshot['phase'] != 'standard' else ""
                title = f"Epoch {snapshot['epoch']} {phase_label}\n"
                title += f"λ_BC={snapshot['lambda_bc']:.3f}, Rel={snapshot['reliability_weight']:.2f}\n"
                title += f"F={snapshot['F_actual']:.4f} (F*={snapshot['F_star']:.4f})"
                ax.set_title(title, fontsize=8)

                if snap_idx == 0:
                    ax.set_ylabel(f"{process_name.capitalize()}\n{input_label}", fontsize=9)

                if snap_idx == 0 and plot_idx <= n_snapshots:
                    ax.legend(fontsize=7, loc='best')

                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=7)

                plot_idx += 1

        # Plot each output dimension
        for output_idx in range(n_outputs):
            output_label = config['output_labels'][output_idx] if output_idx < len(config['output_labels']) else f'Output {output_idx}'

            for snap_idx, snapshot in enumerate(snapshots):
                ax = plt.subplot(total_plots, n_snapshots, plot_idx)

                process_data = snapshot['processes'][process_name]
                actual_outputs = process_data['outputs'][0, :, output_idx]  # First sample
                target_outputs = process_data['target_outputs'][0, :, output_idx]

                x = np.arange(len(actual_outputs))
                ax.plot(x, target_outputs, 'o-', color='green', label='Target', linewidth=2, markersize=4, alpha=0.7)
                ax.plot(x, actual_outputs, '^-', color='blue', label='Actual', linewidth=2, markersize=4)

                # Title with epoch and phase info
                phase_label = f"[{snapshot['phase'].upper()}]" if snapshot['phase'] != 'standard' else ""
                title = f"Epoch {snapshot['epoch']} {phase_label}\n"
                title += f"λ_BC={snapshot['lambda_bc']:.3f}, Rel={snapshot['reliability_weight']:.2f}\n"
                title += f"F={snapshot['F_actual']:.4f} (F*={snapshot['F_star']:.4f})"
                ax.set_title(title, fontsize=8)

                if snap_idx == 0:
                    ax.set_ylabel(f"{process_name.capitalize()}\n{output_label}", fontsize=9)

                if snap_idx == 0 and plot_idx <= n_snapshots:
                    ax.legend(fontsize=7, loc='best')

                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=7)

                plot_idx += 1

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Training progression plot saved: {save_path}")
    else:
        plt.show()

    plt.close()
