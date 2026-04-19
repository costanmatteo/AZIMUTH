"""
Visualizzazioni per controller optimization.
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

# Rebuild font cache once at import time so new fonts are picked up
fm._load_fontmanager(try_read_cache=False)


def apply_plot_style():
    plt.rcParams.update({
        'font.family':           'sans-serif',
        'font.sans-serif':       ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size':             6,
        'axes.titlesize':        6.5,
        'axes.titleweight':      'normal',
        'axes.titlelocation':    'left',
        'axes.labelsize':        6,
        'axes.labelweight':      'normal',
        'axes.linewidth':        0.4,
        'axes.spines.top':       False,
        'axes.spines.right':     False,
        'axes.grid':             True,
        'grid.color':            '#DDDDDD',
        'grid.linewidth':        0.3,
        'grid.alpha':            1.0,
        'xtick.labelsize':       5.5,
        'ytick.labelsize':       5.5,
        'xtick.major.width':     0.3,
        'ytick.major.width':     0.3,
        'xtick.major.size':      2.5,
        'ytick.major.size':      2.5,
        'legend.fontsize':       5.5,
        'legend.framealpha':     0.9,
        'legend.edgecolor':      '#DDDDDD',
        'legend.fancybox':       False,
        'legend.borderpad':      0.3,
        'figure.facecolor':      'white',
        'axes.facecolor':        'white',
        'savefig.facecolor':     'white',
        'savefig.dpi':           150,
        'savefig.bbox':          'tight',
        'lines.linewidth':       0.8,
    })


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
    apply_plot_style()
    # Check if curriculum learning weights are available
    has_curriculum_weights = 'lambda_bc' in history and 'reliability_weight' in history

    # Create figure with 2 rows: combined losses/weights on top, reliability on bottom
    fig, axes = plt.subplots(2, 1, figsize=(20, 5))

    for _ax in axes.flatten():
        _ax.spines['top'].set_visible(False)
        _ax.spines['right'].set_visible(False)

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
    ax_loss.set_title('Training Losses and Weights')

    # ============ BOTTOM PLOT: Reliability Evolution ============
    ax_rel = axes[1]

    if 'F_values' in history:
        ax_rel.plot(epochs, history['F_values'], label='F (Surrogate)', color='purple', linewidth=2)
        if 'F_formula_values' in history and len(history['F_formula_values']) > 0:
            f_formula_epochs = range(len(history['F_formula_values']))
            ax_rel.plot(f_formula_epochs, history['F_formula_values'],
                       label='F (Formula)', color='darkorange', linewidth=2, linestyle='-.')
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
    ax_rel.set_title('Reliability Evolution')
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
    from controller.src.evaluation.metrics import convert_trajectory_to_numpy
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


def plot_target_vs_actual_scatter(F_star_per_scenario, F_baseline_per_scenario, F_actual_per_scenario,
                                  F_formula_per_scenario=None, save_path=None, figsize=(8, 10)):
    """
    Scatter plot: Baseline vs Controller reliability.

    Y-axis: F_star (target reliability)
    X-axis: F values (baseline red, controller blue)

    When F_formula_per_scenario is provided, adds hollow blue circles
    showing the ProT formula-based F estimate alongside the solid
    surrogate-based controller dots.

    Args:
        F_star_per_scenario (array-like): Target reliability for each scenario
        F_baseline_per_scenario (array-like): Baseline reliability for each scenario
        F_actual_per_scenario (array-like): Controller reliability (surrogate) for each scenario
        F_formula_per_scenario (array-like, optional): Controller reliability (ProT formula)
        save_path (str): Path to save figure
    """
    apply_plot_style()
    F_star_arr = np.atleast_1d(F_star_per_scenario)
    F_baseline_arr = np.atleast_1d(F_baseline_per_scenario)
    F_actual_arr = np.atleast_1d(F_actual_per_scenario)

    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Baseline points (red squares)
    ax.scatter(F_baseline_arr, F_star_arr,
               c='red',
               s=8,
               alpha=0.6,
               edgecolors='darkred',
               linewidths=0.3,
               label='Baseline (no controller)',
               marker='s')

    # Controller points — surrogate F (solid blue circles)
    ax.scatter(F_actual_arr, F_star_arr,
               c='blue',
               s=8,
               alpha=0.6,
               edgecolors='darkblue',
               linewidths=0.3,
               label='Controller (surrogate)',
               marker='o')

    # Controller points — ProT formula F (hollow blue circles)
    if F_formula_per_scenario is not None:
        F_formula_arr = np.atleast_1d(F_formula_per_scenario)
        # Match F_star length to F_formula length (may differ if per_sample)
        if len(F_formula_arr) == len(F_star_arr):
            F_star_for_formula = F_star_arr
        else:
            F_star_for_formula = np.full_like(F_formula_arr, F_star_arr[0])
        ax.scatter(F_formula_arr, F_star_for_formula,
                   facecolors='none',
                   s=8,
                   alpha=0.7,
                   edgecolors='blue',
                   linewidths=0.5,
                   label='Controller (ProT formula)',
                   marker='o')

    # Diagonal line (perfect match)
    all_values_list = [F_star_arr, F_baseline_arr, F_actual_arr]
    if F_formula_per_scenario is not None:
        all_values_list.append(np.atleast_1d(F_formula_per_scenario))
    all_values = np.concatenate(all_values_list)
    min_val = all_values.min()
    max_val = all_values.max()
    margin = (max_val - min_val) * 0.1

    ax.plot([min_val - margin, max_val + margin],
            [min_val - margin, max_val + margin],
            'k--', label='Perfect Match (y = x)', alpha=0.5)

    ax.set_xlabel('F (Reliability)')
    ax.set_ylabel('F_star (Target Reliability)')
    ax.set_title('Baseline vs Controller Reliability')

    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', framealpha=0.9)

    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_gap_distribution(F_star_per_scenario, F_actual_per_scenario, save_path=None, figsize=(8, 10)):
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
    apply_plot_style()
    F_star_arr = np.atleast_1d(F_star_per_scenario)
    F_actual_arr = np.atleast_1d(F_actual_per_scenario)

    # Compute gaps
    gaps = F_star_arr - F_actual_arr

    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

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
                                     linewidth=0.5)

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

    ax.axvline(mean_gap, color='darkblue', linestyle='--',
               label=f'Mean: {mean_gap:.6f}')
    ax.axvline(median_gap, color='purple', linestyle='--',
               label=f'Median: {median_gap:.6f}')
    ax.axvline(worst_gap, color='darkred', linestyle='--',
               label=f'Worst: {worst_gap:.6f}')

    # Labels and title
    ax.set_xlabel('Gap (F_star - F_actual)')
    ax.set_ylabel('Number of Scenarios')
    ax.set_title('Distribution of Target-Actual Gap')
    ax.legend(loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)

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

    # Create figure with subplots (cap height to avoid matplotlib pixel limit)
    # matplotlib limit: 2^16 = 65536 pixels per dimension; at 150 dpi, max ~436 inches
    # Use conservative cap of 200 inches (= 30000 pixels at 150 dpi)
    fig_height = min(4 * total_input_plots, 200)
    fig, axes = plt.subplots(total_input_plots, 1, figsize=(12, fig_height))
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

    Shows two types of validation:
    1. Cross-scenario validation: Test scenarios with different conditions
    2. Within-scenario validation: Held-out samples from training scenarios

    Layout:
    - Top: Total Loss (train vs both validations)
    - Bottom: Reliability Loss (train vs both validations)

    Args:
        history (dict): Training history containing:
            - total_loss: Training total loss per epoch
            - val_total_loss: Cross-scenario validation loss (optional)
            - val_within_total_loss: Within-scenario validation loss (optional)
            - reliability_loss: Training reliability loss per epoch
            - val_reliability_loss: Cross-scenario validation reliability loss (optional)
            - val_within_reliability_loss: Within-scenario validation reliability loss (optional)
        save_path (str): Path to save figure
    """
    apply_plot_style()
    # Check which validation data is available
    has_cross_val = 'val_total_loss' in history and len(history.get('val_total_loss', [])) > 0
    has_within_val = 'val_within_total_loss' in history and len(history.get('val_within_total_loss', [])) > 0

    # Determine number of plots
    n_plots = 2  # Total loss and Reliability loss

    fig, axes = plt.subplots(n_plots, 1, figsize=(20, 5))

    for _ax in (axes.flatten() if hasattr(axes, 'flatten') else [axes]):
        _ax.spines['top'].set_visible(False)
        _ax.spines['right'].set_visible(False)

    epochs = range(1, len(history['total_loss']) + 1)

    # ============ Plot 1: Total Loss ============
    ax1 = axes[0]

    # Training loss
    ax1.plot(epochs, history['total_loss'], label='Train Loss', color='blue', linewidth=2)

    # Cross-scenario validation loss
    if has_cross_val:
        val_epochs = range(1, len(history['val_total_loss']) + 1)
        ax1.plot(val_epochs, history['val_total_loss'], label='Val Loss (cross-scenario)',
                 color='red', linewidth=2, linestyle='--')

    # Within-scenario validation loss
    if has_within_val:
        val_within_epochs = range(1, len(history['val_within_total_loss']) + 1)
        ax1.plot(val_within_epochs, history['val_within_total_loss'], label='Val Loss (within-scenario)',
                 color='orange', linewidth=2, linestyle=':')

    # Highlight overfitting region (prefer within-scenario for detection, fallback to cross-scenario)
    val_for_overfit = None
    val_label = ""
    if has_within_val:
        val_for_overfit = history['val_within_total_loss']
        val_label = "within"
    elif has_cross_val:
        val_for_overfit = history['val_total_loss']
        val_label = "cross"

    if val_for_overfit is not None:
        train_arr = np.array(history['total_loss'])
        val_arr = np.array(val_for_overfit)
        min_len = min(len(train_arr), len(val_arr))

        if min_len > 0:
            overfit_mask = val_arr[:min_len] > train_arr[:min_len]
            if np.any(overfit_mask):
                overfit_start = np.argmax(overfit_mask) + 1
                ax1.axvline(x=overfit_start, color='darkred', linestyle=':',
                           linewidth=2, alpha=0.7, label=f'Overfitting ({val_label}, epoch {overfit_start})')

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Total Loss', fontsize=12)
    ax1.set_title('Training vs Validation Loss (Overfitting Detection)')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Add annotation for final gaps
    annotation_lines = []
    final_train = history['total_loss'][-1]
    annotation_lines.append(f'Final Train: {final_train:.6f}')

    if has_cross_val and len(history['val_total_loss']) > 0:
        final_cross = history['val_total_loss'][-1]
        gap_cross = final_cross - final_train
        gap_cross_pct = (gap_cross / final_train) * 100 if final_train != 0 else 0
        status_cross = "overfitting" if gap_cross > 0 else "OK"
        annotation_lines.append(f'Val (cross): {final_cross:.6f} ({gap_cross_pct:+.1f}% {status_cross})')

    if has_within_val and len(history['val_within_total_loss']) > 0:
        final_within = history['val_within_total_loss'][-1]
        gap_within = final_within - final_train
        gap_within_pct = (gap_within / final_train) * 100 if final_train != 0 else 0
        status_within = "overfitting" if gap_within > 0 else "OK"
        annotation_lines.append(f'Val (within): {final_within:.6f} ({gap_within_pct:+.1f}% {status_within})')

    if len(annotation_lines) > 1:
        ax1.text(0.02, 0.98, '\n'.join(annotation_lines),
                transform=ax1.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='square,pad=0.3', facecolor='white',
                          edgecolor='#CCCCCC', linewidth=0.5, alpha=0.9))

    # ============ Plot 2: Reliability Loss ============
    ax2 = axes[1]

    # Training reliability loss
    ax2.plot(epochs, history['reliability_loss'], label='Train Reliability Loss',
             color='purple', linewidth=2)

    # Cross-scenario validation reliability loss
    if has_cross_val and 'val_reliability_loss' in history:
        val_rel_loss = history['val_reliability_loss']
        if len(val_rel_loss) > 0:
            val_epochs = range(1, len(val_rel_loss) + 1)
            ax2.plot(val_epochs, val_rel_loss, label='Val Reliability (cross-scenario)',
                     color='magenta', linewidth=2, linestyle='--')

    # Within-scenario validation reliability loss
    if has_within_val and 'val_within_reliability_loss' in history:
        val_within_rel_loss = history['val_within_reliability_loss']
        if len(val_within_rel_loss) > 0:
            val_within_epochs = range(1, len(val_within_rel_loss) + 1)
            ax2.plot(val_within_epochs, val_within_rel_loss, label='Val Reliability (within-scenario)',
                     color='darkorange', linewidth=2, linestyle=':')

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Reliability Loss', fontsize=12)
    ax2.set_title('Reliability Loss: Train vs Validation')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Add warm-up marker if curriculum learning data is available
    if 'reliability_weight' in history and len(history['reliability_weight']) > 0:
        warmup_end = 0
        for i, w in enumerate(history['reliability_weight']):
            if w > 0:
                warmup_end = i + 1
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


class _SquareHandler:
    """Legend handler that draws a small filled square instead of a line."""
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        from matplotlib.patches import Rectangle
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        w, h = handlebox.width * 0.5, handlebox.height * 0.7
        patch = Rectangle((x0, y0 + h * 0.15), w, h,
                           facecolor=orig_handle.get_color(),
                           edgecolor='none', transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        return patch


def generate_process_evolution_plots(training_progression, controllable_info,
                                     checkpoint_dir, n_scenarios=1,
                                     row_height_pt=14, plot_width_in=3.5,
                                     uniform_height=True,
                                     y_range=None,
                                     warmup_end=None,
                                     n_epochs=None):
    """
    Generate per-process evolution plots showing controllable inputs and outputs
    across training epochs, for each scenario.

    Args:
        training_progression (list): List of epoch snapshots from trainer.
        controllable_info (dict): per-process info (input_labels, controllable_indices, etc.)
        checkpoint_dir (Path): Directory to save plots
        n_scenarios (int): Number of scenarios
        row_height_pt (float): (kept for backward compatibility, unused)
        plot_width_in (float): (kept for backward compatibility, unused)
        uniform_height (bool): (kept for backward compatibility, unused)
        y_range (tuple, optional): (kept for backward compatibility, unused — Y
            axis is fixed to (-2, 2) for thesis-figure consistency).
        warmup_end (int, optional): Epoch at which the warm-up phase ends.
            None disables the warm-up shading/marker.
        n_epochs (int, optional): Total number of training epochs, used for
            X-axis limits and ticks. If None, inferred from max epoch in
            training_progression.

    Returns:
        tuple: (plot_paths, color_maps) where
            plot_paths: {(scenario_idx, process_name): plot_path}
            color_maps: {process_name: {variable_label: hex_color}}
    """
    from matplotlib.lines import Line2D

    apply_plot_style()
    checkpoint_dir = Path(checkpoint_dir)
    plot_paths = {}
    color_maps = {}  # {process_name: {var_label: hex_color}}

    if not training_progression:
        return plot_paths, color_maps

    # Extract process names from first snapshot
    first_snap = training_progression[0]
    per_sc = first_snap.get('per_scenario', {})
    if not per_sc:
        return plot_paths, color_maps
    proc_names = list(per_sc[0].keys()) if 0 in per_sc else []
    if not proc_names:
        return plot_paths, color_maps

    epochs = [s['epoch'] for s in training_progression]
    if n_epochs is None:
        n_epochs = int(max(epochs)) if epochs else 200
    n_epochs = max(int(n_epochs), 1)

    # One color per process (solid = input, dashed = output); fallback to tab10.
    PROC_PALETTE = ['#1f77b4', '#d62728', '#2ca02c']
    tab10 = list(plt.get_cmap('tab10').colors)
    process_colors = {}
    for i, pname in enumerate(proc_names):
        process_colors[pname] = (
            PROC_PALETTE[i] if i < len(PROC_PALETTE) else tab10[i % len(tab10)]
        )

    # X ticks: 5 evenly spaced between 0 and n_epochs
    xticks = np.unique(np.linspace(0, n_epochs, 5).round().astype(int))

    for scenario_idx in range(n_scenarios):
        for proc_idx, proc_name in enumerate(proc_names):
            p_info = controllable_info.get(proc_name, {})
            ctrl_indices = p_info.get('controllable_indices', [])
            input_labels = p_info.get('input_labels', [])
            output_labels = p_info.get('output_labels', [])

            # Collect per-epoch values
            ctrl_series = {ci: [] for ci in ctrl_indices}
            out_series = {oi: [] for oi in range(len(output_labels))}

            for snap in training_progression:
                sc_data = snap.get('per_scenario', {}).get(scenario_idx)
                if sc_data is None:
                    for ci in ctrl_indices:
                        ctrl_series[ci].append(np.nan)
                    for oi in range(len(output_labels)):
                        out_series[oi].append(np.nan)
                    continue

                proc_data = sc_data.get(proc_name, {})
                inputs = proc_data.get('inputs', np.array([[]]))
                outputs = proc_data.get('outputs_mean', proc_data.get('outputs', np.array([[]])))

                # inputs shape: (1, input_dim) or (input_dim,)
                inp = np.asarray(inputs).flatten()
                out = np.asarray(outputs).flatten()

                for ci in ctrl_indices:
                    ctrl_series[ci].append(float(inp[ci]) if ci < len(inp) else np.nan)
                for oi in range(len(output_labels)):
                    out_series[oi].append(float(out[oi]) if oi < len(out) else np.nan)

            if len(ctrl_indices) + len(output_labels) == 0:
                continue

            # Target inputs (a_t*) from the first available snapshot.
            # Same for all epochs/scenarios; use axhlines as references.
            target_values = {}  # ci -> float
            for snap in training_progression:
                tgt_traj = snap.get('target_trajectory', {}) or {}
                tgt_proc = tgt_traj.get(proc_name, {}) or {}
                tgt_inp = tgt_proc.get('inputs', None)
                if tgt_inp is not None and np.asarray(tgt_inp).size > 0:
                    t_flat = np.asarray(tgt_inp).flatten()
                    for ci in ctrl_indices:
                        if ci < len(t_flat):
                            target_values[ci] = float(t_flat[ci])
                    break

            fig, ax = plt.subplots(figsize=(6.5, 2.6))

            pcolor = process_colors[proc_name]
            proc_var_colors = {}

            # Warm-up shading (behind everything)
            if warmup_end is not None and warmup_end > 0:
                ax.axvspan(0, warmup_end, color='#F5F5F5', zorder=0)
                ax.axvline(warmup_end, color='#888888', linestyle=':',
                           linewidth=0.5, zorder=1)
                ax.annotate('Warm-up end',
                            xy=(warmup_end + 2, 1.85),
                            fontsize=5, color='#666666')

            # Horizontal grid at integer ticks (drawn via yaxis grid below),
            # plus explicit y=0 reference line.
            ax.axhline(y=0, color='#CCCCCC', linewidth=0.3, zorder=0.5)

            # Target reference lines (a_t*), dotted, same color as process
            for ci in ctrl_indices:
                if ci in target_values:
                    ax.axhline(y=target_values[ci], color=pcolor,
                               linestyle=(0, (1, 2)), linewidth=0.3,
                               alpha=0.35, zorder=1.5)

            # Controllable inputs (solid)
            for ci in ctrl_indices:
                lbl = input_labels[ci] if ci < len(input_labels) else f"X_{ci}"
                ax.plot(epochs, ctrl_series[ci], color=pcolor, linewidth=0.9,
                        alpha=0.9, zorder=3)
                proc_var_colors[lbl] = pcolor

            # Outputs (dashed)
            for oi in range(len(output_labels)):
                lbl = output_labels[oi]
                ax.plot(epochs, out_series[oi], color=pcolor, linewidth=0.9,
                        linestyle='--', alpha=0.75, zorder=3)
                proc_var_colors[lbl] = pcolor

            color_maps[proc_name] = proc_var_colors

            # Axes
            ax.set_xlim(0, n_epochs)
            ax.set_ylim(-2, 2)
            ax.set_xticks(xticks)
            ax.set_yticks([-2, -1, 0, 1, 2])
            ax.set_xlabel('Epoch')
            ax.set_ylabel(r'Value')
            ax.xaxis.set_label_coords(0.5, -0.14)
            ax.yaxis.set_label_coords(-0.06, 0.5)

            # Grid: only horizontal
            ax.yaxis.grid(True, color='#EEEEEE', linewidth=0.3, zorder=0)
            ax.xaxis.grid(False)

            # Spines: keep top/right off, left/bottom default (at axes edge)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['left'].set_linewidth(0.4)
            ax.spines['bottom'].set_visible(True)
            ax.spines['bottom'].set_position(('outward', 0))
            ax.spines['bottom'].set_linewidth(0.4)

            # Legend: only the current process (solid line in its color) plus
            # three style entries (input / output / target reference). A single
            # horizontal row below the X axis, no frame.
            legend_handles = [
                Line2D([0], [0], color=pcolor, linewidth=1.0,
                       label=proc_name.capitalize()),
                Line2D([0], [0], color='#555555', linewidth=1.0,
                       label=r'Input $\hat{a}$'),
                Line2D([0], [0], color='#555555', linewidth=1.0,
                       linestyle='--', label=r'Output $\hat{o}$'),
                Line2D([0], [0], color='#888888', linewidth=0.5,
                       linestyle=(0, (1, 2)), label='Target (ref.)'),
            ]
            ax.legend(handles=legend_handles, loc='upper center',
                      ncol=len(legend_handles),
                      bbox_to_anchor=(0.5, -0.28), frameon=False,
                      columnspacing=1.6, handlelength=1.8,
                      handletextpad=0.5)

            plt.subplots_adjust(bottom=0.25)

            fname = f'evolution_sc{scenario_idx}_{proc_name}.png'
            fpath = checkpoint_dir / fname
            plt.savefig(str(fpath), dpi=200, bbox_inches='tight')
            plt.close()

            plot_paths[(scenario_idx, proc_name)] = str(fpath)

    return plot_paths, color_maps
