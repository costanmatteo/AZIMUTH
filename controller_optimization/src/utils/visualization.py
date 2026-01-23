"""
Visualizzazioni per controller optimization.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_training_history(history, save_path=None):
    """
    Plot training history with combined loss/weight graph and separate reliability graph.

    Layout:
    - Top: Combined graph with losses (left Y-axis) and weights (right Y-axis)
    - Bottom: Reliability Evolution (F values)

    Args:
        history (dict): Training history
        save_path (str): Path to save figure
    """
    # Check if curriculum learning weights are available
    has_curriculum_weights = 'lambda_bc' in history and 'reliability_weight' in history

    # Create figure with 2 rows: combined losses/weights on top, reliability on bottom
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # ============ TOP PLOT: Combined Losses and Weights ============
    ax_loss = axes[0]

    epochs = range(len(history['total_loss']))

    # Plot losses on primary Y-axis (left)
    line_total, = ax_loss.plot(epochs, history['total_loss'], label='Total Loss', color='blue', linewidth=2)
    line_rel, = ax_loss.plot(epochs, history['reliability_loss'], label='Reliability Loss', color='red', linewidth=2)
    line_bc, = ax_loss.plot(epochs, history['bc_loss'], label='BC Loss', color='green', linewidth=2)

    ax_loss.set_xlabel('Epoch', fontsize=12)
    ax_loss.set_ylabel('Loss', fontsize=12, color='black')
    ax_loss.tick_params(axis='y', labelcolor='black')
    ax_loss.grid(True, alpha=0.3)

    lines = [line_total, line_rel, line_bc]
    labels = ['Total Loss', 'Reliability Loss', 'BC Loss']

    # Plot weights on secondary Y-axis (right) if available
    if has_curriculum_weights:
        ax_weight = ax_loss.twinx()

        line_lambda, = ax_weight.plot(epochs, history['lambda_bc'], label='λ_BC', color='orange',
                                       linewidth=2, linestyle='--')
        line_relw, = ax_weight.plot(epochs, history['reliability_weight'], label='Reliability Weight',
                                     color='brown', linewidth=2, linestyle='--')

        ax_weight.set_ylabel('Weight', fontsize=12, color='gray')
        ax_weight.tick_params(axis='y', labelcolor='gray')
        ax_weight.set_ylim(-0.05, max(max(history['lambda_bc']), 1.0) * 1.1)

        lines += [line_lambda, line_relw]
        labels += ['λ_BC', 'Reliability Weight']

        # Add vertical line at end of warm-up
        if len(history['reliability_weight']) > 0:
            warmup_end = 0
            for i, w in enumerate(history['reliability_weight']):
                if w > 0:
                    warmup_end = i
                    break

            if warmup_end > 0:
                ax_loss.axvline(x=warmup_end, color='gray', linestyle=':', alpha=0.7, linewidth=2)
                # Add annotation for warm-up end
                y_pos = ax_loss.get_ylim()[1] * 0.95
                ax_loss.annotate('Warm-up End', xy=(warmup_end, y_pos), fontsize=9,
                                ha='left', va='top', color='gray')

    # Combined legend
    ax_loss.legend(lines, labels, loc='upper right', fontsize=10, framealpha=0.9)
    ax_loss.set_title('Training Losses and Weights', fontsize=14, fontweight='bold')

    # ============ BOTTOM PLOT: Reliability Evolution ============
    ax_rel = axes[1]

    if 'F_values' in history:
        ax_rel.plot(epochs, history['F_values'], label='F (Actual)', color='purple', linewidth=2)
        if 'F_star' in history:
            ax_rel.axhline(y=history['F_star'], color='gold', linestyle='--', linewidth=2, label='F* (Target)')

        # Add warm-up end marker if curriculum weights available
        if has_curriculum_weights and len(history['reliability_weight']) > 0:
            warmup_end = 0
            for i, w in enumerate(history['reliability_weight']):
                if w > 0:
                    warmup_end = i
                    break
            if warmup_end > 0:
                ax_rel.axvline(x=warmup_end, color='gray', linestyle=':', alpha=0.7, linewidth=2)

    ax_rel.set_xlabel('Epoch', fontsize=12)
    ax_rel.set_ylabel('Reliability F', fontsize=12)
    ax_rel.set_title('Reliability Evolution', fontsize=14, fontweight='bold')
    ax_rel.legend(loc='best', fontsize=10)
    ax_rel.grid(True, alpha=0.3)

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
    Plot training progression showing how controllable inputs evolve across epochs.

    X-axis: Epoch number
    Y-axis: Input value for each of 3 fixed samples
    One subplot per controllable input dimension.
    Shows convergence toward target values with warmup phase marker.

    Supports both old format (single sample) and new format (3 fixed samples).

    Args:
        progression_path (str): Path to training_progression.npz file
        save_path (str): Path to save figure
    """
    from controller_optimization.configs.processes_config import get_process_by_name

    # Load progression data
    data = np.load(progression_path, allow_pickle=True)

    # Extract snapshot information
    snapshots = []

    # Find all unique prefixes (snapshot_X_epoch_Y)
    prefixes = set()
    for key in data.files:
        if key.count('_') >= 3:
            parts = key.split('_')
            if len(parts) >= 4 and parts[0] == 'snapshot' and parts[2] == 'epoch':
                prefix = '_'.join(parts[:4])
                prefixes.add(prefix)

    def get_epoch_from_prefix(prefix):
        parts = prefix.split('_')
        return int(parts[3])

    sorted_prefixes = sorted(prefixes, key=get_epoch_from_prefix)

    for prefix in sorted_prefixes:
        # Check if new format (with n_samples) or old format
        n_samples_key = f'{prefix}_n_samples'
        if n_samples_key in data.files:
            n_samples = int(data[n_samples_key])
        else:
            n_samples = 1  # Old format

        # Get F values for all samples
        if f'{prefix}_F_actuals' in data.files:
            F_actuals = data[f'{prefix}_F_actuals']
        else:
            F_actuals = [float(data[f'{prefix}_F_actual'])]

        snapshot = {
            'epoch': int(data[f'{prefix}_epoch']),
            'phase': str(data[f'{prefix}_phase']),
            'lambda_bc': float(data[f'{prefix}_lambda_bc']),
            'reliability_weight': float(data[f'{prefix}_reliability_weight']),
            'F_actuals': list(F_actuals),
            'F_actual': float(F_actuals[0]),
            'F_star': float(data[f'{prefix}_F_star']),
            'n_samples': n_samples,
            'samples': []  # List of process data dicts, one per sample
        }

        # Extract process data for each sample
        for sample_idx in range(n_samples):
            sample_processes = {}

            # Find process names for this sample
            sample_prefix = f'{prefix}_sample{sample_idx}_'

            for key in data.files:
                if key.startswith(sample_prefix) and key.endswith('_inputs'):
                    # Extract process name: "prefix_sample0_plasma_inputs" -> "plasma"
                    remaining = key.replace(sample_prefix, '').replace('_inputs', '')
                    process_name = remaining

                    sample_processes[process_name] = {
                        'inputs': data[f'{sample_prefix}{process_name}_inputs'],
                        'outputs': data[f'{sample_prefix}{process_name}_outputs'],
                    }

            # Get target inputs (same for all samples)
            for process_name in sample_processes.keys():
                target_key = f'{prefix}_{process_name}_target_inputs'
                if target_key in data.files:
                    sample_processes[process_name]['target_inputs'] = data[target_key]
                    sample_processes[process_name]['target_outputs'] = data[f'{prefix}_{process_name}_target_outputs']

            snapshot['samples'].append(sample_processes)

        # Backward compatibility: if old format, extract process data directly
        if n_samples == 1 and len(snapshot['samples'][0]) == 0:
            sample_processes = {}
            for key in data.files:
                if key.startswith(prefix) and '_inputs' in key and '_target_inputs' not in key and 'sample' not in key:
                    remaining = key.replace(f'{prefix}_', '')
                    if remaining.endswith('_inputs'):
                        process_name = remaining.replace('_inputs', '')
                        sample_processes[process_name] = {
                            'inputs': data[f'{prefix}_{process_name}_inputs'],
                            'outputs': data[f'{prefix}_{process_name}_outputs'],
                            'target_inputs': data[f'{prefix}_{process_name}_target_inputs'],
                            'target_outputs': data[f'{prefix}_{process_name}_target_outputs']
                        }
            snapshot['samples'] = [sample_processes]

        snapshots.append(snapshot)

    if len(snapshots) == 0:
        print("No progression snapshots found")
        return

    # Setup
    process_names = list(snapshots[0]['samples'][0].keys())
    epochs = [s['epoch'] for s in snapshots]
    n_samples = snapshots[0]['n_samples']

    # Sample colors and markers
    sample_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    sample_markers = ['o', 's', '^']  # Circle, Square, Triangle
    sample_labels = ['Sample 1 (seed=42)', 'Sample 2 (seed=123)', 'Sample 3 (seed=456)']

    # Get input/output dimensions for each process
    total_input_plots = 0
    process_dims = {}
    for process_name in process_names:
        config = get_process_by_name(process_name)
        n_inputs = config['input_dim']
        process_dims[process_name] = {'n_inputs': n_inputs}
        total_input_plots += n_inputs

    # Create figure with subplots
    fig, axes = plt.subplots(total_input_plots, 1, figsize=(12, 4 * total_input_plots))
    if total_input_plots == 1:
        axes = [axes]

    plot_idx = 0

    for process_name in process_names:
        config = get_process_by_name(process_name)
        n_inputs = process_dims[process_name]['n_inputs']

        for input_idx in range(n_inputs):
            ax = axes[plot_idx]
            input_label = config['input_labels'][input_idx] if input_idx < len(config['input_labels']) else f'Input {input_idx}'

            # Plot each sample separately
            for sample_idx in range(n_samples):
                sample_values = []

                for snapshot in snapshots:
                    if sample_idx < len(snapshot['samples']):
                        process_data = snapshot['samples'][sample_idx].get(process_name, {})
                        if 'inputs' in process_data:
                            inputs = process_data['inputs']
                            if len(inputs.shape) == 2:
                                val = inputs[:, input_idx]
                            else:
                                val = inputs[0, :, input_idx]
                            sample_values.append(np.mean(val))
                        else:
                            sample_values.append(np.nan)
                    else:
                        sample_values.append(np.nan)

                color = sample_colors[sample_idx % len(sample_colors)]
                marker = sample_markers[sample_idx % len(sample_markers)]
                label = sample_labels[sample_idx] if sample_idx < len(sample_labels) else f'Sample {sample_idx+1}'

                ax.plot(epochs, sample_values, marker=marker, linestyle='-', color=color,
                       label=label, linewidth=2, markersize=6, alpha=0.8)

            # Plot target value as horizontal line
            target_values = []
            for snapshot in snapshots:
                process_data = snapshot['samples'][0].get(process_name, {})
                if 'target_inputs' in process_data:
                    target_inputs = process_data['target_inputs']
                    if len(target_inputs.shape) == 2:
                        val = target_inputs[:, input_idx]
                    else:
                        val = target_inputs[0, :, input_idx]
                    target_values.append(np.mean(val))

            if target_values:
                target_mean = np.mean(target_values)
                ax.axhline(y=target_mean, color='gray', linestyle='--',
                          linewidth=2.5, alpha=0.7, label=f'Target ({target_mean:.3f})')

            # Mark warmup end
            if snapshots[0]['phase'] == 'warmup':
                for i, snapshot in enumerate(snapshots):
                    if snapshot['phase'] == 'curriculum':
                        ax.axvline(x=snapshot['epoch'], color='red', linestyle=':',
                                  linewidth=2.5, alpha=0.6, label='Warmup end')
                        break

            ax.set_ylabel(f"{process_name.capitalize()}\n{input_label}", fontsize=11, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc='best')

            # Title with F progression (mean across samples)
            F_start = np.mean(snapshots[0]['F_actuals'])
            F_end = np.mean(snapshots[-1]['F_actuals'])
            F_star = snapshots[0]['F_star']
            ax.set_title(f"F: {F_start:.4f} → {F_end:.4f} (F*={F_star:.4f})", fontsize=10)

            plot_idx += 1

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Training progression plot saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_loss_chart(history, save_path=None):
    """
    Plot training vs validation loss chart to help identify overfitting.

    Similar to uncertainty predictor's plot_training_history, this shows:
    - Train loss vs Validation loss (if available)
    - Helps identify when the model starts overfitting

    Layout:
    - Top: Total Loss (train vs validation)
    - Bottom: Reliability Loss (train vs validation)

    Args:
        history (dict): Training history containing:
            - total_loss: Training total loss per epoch
            - val_total_loss: Validation total loss per epoch (optional)
            - reliability_loss: Training reliability loss per epoch
            - val_reliability_loss: Validation reliability loss per epoch (optional)
        save_path (str): Path to save figure
    """
    # Check if validation data is available
    has_validation = 'val_total_loss' in history and len(history.get('val_total_loss', [])) > 0

    # Determine number of plots
    n_plots = 2  # Total loss and Reliability loss

    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 5 * n_plots))

    epochs = range(1, len(history['total_loss']) + 1)

    # ============ Plot 1: Total Loss ============
    ax1 = axes[0]

    # Training loss
    ax1.plot(epochs, history['total_loss'], label='Train Loss', color='blue', linewidth=2)

    # Validation loss if available
    if has_validation:
        val_epochs = range(1, len(history['val_total_loss']) + 1)
        ax1.plot(val_epochs, history['val_total_loss'], label='Validation Loss',
                 color='red', linewidth=2, linestyle='--')

        # Highlight overfitting region (where val loss > train loss)
        train_arr = np.array(history['total_loss'])
        val_arr = np.array(history['val_total_loss'])
        min_len = min(len(train_arr), len(val_arr))

        if min_len > 0:
            overfit_mask = val_arr[:min_len] > train_arr[:min_len]
            if np.any(overfit_mask):
                # Find first overfitting epoch
                overfit_start = np.argmax(overfit_mask) + 1
                ax1.axvline(x=overfit_start, color='orange', linestyle=':',
                           linewidth=2, alpha=0.7, label=f'Overfitting starts (epoch {overfit_start})')

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Total Loss', fontsize=12)
    ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Add annotation for final gap
    if has_validation and len(history['val_total_loss']) > 0:
        final_train = history['total_loss'][-1]
        final_val = history['val_total_loss'][-1]
        gap = final_val - final_train
        gap_pct = (gap / final_train) * 100 if final_train != 0 else 0

        status = "⚠ Overfitting" if gap > 0 else "✓ Good generalization"
        ax1.text(0.02, 0.98, f'Final Train: {final_train:.6f}\nFinal Val: {final_val:.6f}\nGap: {gap:+.6f} ({gap_pct:+.1f}%)\n{status}',
                transform=ax1.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # ============ Plot 2: Reliability Loss ============
    ax2 = axes[1]

    # Training reliability loss
    ax2.plot(epochs, history['reliability_loss'], label='Train Reliability Loss',
             color='purple', linewidth=2)

    # Validation reliability loss if available
    if has_validation and 'val_reliability_loss' in history:
        val_rel_loss = history['val_reliability_loss']
        if len(val_rel_loss) > 0:
            val_epochs = range(1, len(val_rel_loss) + 1)
            ax2.plot(val_epochs, val_rel_loss, label='Validation Reliability Loss',
                     color='magenta', linewidth=2, linestyle='--')

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Reliability Loss', fontsize=12)
    ax2.set_title('Reliability Loss: Train vs Validation', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Add warm-up marker if curriculum learning data is available
    if 'reliability_weight' in history and len(history['reliability_weight']) > 0:
        # Find end of warm-up (first epoch with reliability_weight > 0)
        warmup_end = 0
        for i, w in enumerate(history['reliability_weight']):
            if w > 0:
                warmup_end = i + 1  # 1-indexed epoch
                break

        if warmup_end > 0:
            ax2.axvline(x=warmup_end, color='gray', linestyle=':', alpha=0.7, linewidth=2)
            ax2.annotate('Warm-up End', xy=(warmup_end, ax2.get_ylim()[1] * 0.9),
                        fontsize=9, ha='left', va='top', color='gray')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Loss chart saved: {save_path}")
    else:
        plt.show()

    plt.close()
