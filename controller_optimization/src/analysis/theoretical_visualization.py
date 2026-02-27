"""
Visualization functions for Theoretical Loss Analysis.

Creates plots comparing observed loss vs theoretical minimum (L_min).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


def plot_loss_vs_L_min(
    epochs: List[int],
    observed_loss: List[float],
    theoretical_L_min: List[float],
    save_path: Optional[str] = None,
    title: str = "Loss vs Theoretical Minimum",
    figsize: Tuple[int, int] = (10, 6),
    bellman_lmin: Optional[Dict[str, Any]] = None,
) -> plt.Figure:
    """
    Plot observed loss and theoretical L_min over epochs.

    Shows:
    - Observed loss curve (solid blue)
    - Theoretical L_min curve (dashed red)
    - Shaded area showing reducible gap
    - Bellman L_min horizontal line (if available)

    Args:
        epochs: List of epoch numbers
        observed_loss: List of observed loss values
        theoretical_L_min: List of L_min values (Var[F] + Bias²)
        save_path: Path to save figure (optional)
        title: Plot title
        figsize: Figure size
        bellman_lmin: Dict with Bellman results (keys: L_min_bellman, L_min_naive, L_min_forward)

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = np.array(epochs)
    observed = np.array(observed_loss)
    theoretical = np.array(theoretical_L_min)

    # Plot observed loss
    ax.plot(epochs, observed, 'b-', linewidth=2, label='Observed Loss', marker='o', markersize=3)

    # Plot theoretical L_min (empirical)
    ax.plot(epochs, theoretical, 'r--', linewidth=2, label='L_min (empirical)')

    # Plot Bellman L_min lines if available
    if bellman_lmin is not None:
        bellman_val = bellman_lmin.get('L_min_bellman', None)
        naive_val = bellman_lmin.get('L_min_naive', None)
        if bellman_val is not None:
            ax.axhline(y=bellman_val, color='green', linestyle='-.',
                       linewidth=2.5, label=f'L_min Bellman = {bellman_val:.4f}')
        if naive_val is not None:
            ax.axhline(y=naive_val, color='purple', linestyle=':',
                       linewidth=2, label=f'L_min naive = {naive_val:.4f}')

    # Fill area between L_min and observed (reducible gap)
    ax.fill_between(
        epochs,
        theoretical,
        observed,
        alpha=0.3,
        color='orange',
        label='Reducible Gap'
    )

    all_vals = np.concatenate([observed, theoretical])
    if bellman_lmin is not None:
        extra = [v for v in [bellman_lmin.get('L_min_bellman'), bellman_lmin.get('L_min_naive')] if v is not None]
        if extra:
            all_vals = np.concatenate([all_vals, extra])

    # Labels and legend
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set reasonable y-axis limits
    y_max = max(all_vals) * 1.1
    y_min = min(min(all_vals), 0) * 0.9 if min(all_vals) < 0 else 0
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


def plot_efficiency_over_time(
    epochs: List[int],
    efficiency: List[float],
    save_path: Optional[str] = None,
    title: str = "Training Efficiency (L_min / Loss)",
    figsize: Tuple[int, int] = (10, 5)
) -> plt.Figure:
    """
    Plot efficiency (L_min / observed_loss) over epochs.

    Efficiency of 1.0 means loss equals theoretical minimum.

    Args:
        epochs: List of epoch numbers
        efficiency: List of efficiency values (0 to 1+)
        save_path: Path to save figure (optional)
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = np.array(epochs)
    eff = np.array(efficiency)

    # Clip efficiency for visualization (handle division issues)
    eff_clipped = np.clip(eff, 0, 1.5)

    # Plot efficiency curve
    ax.plot(epochs, eff_clipped, 'g-', linewidth=2, marker='o', markersize=3)

    # Add horizontal line at y=1 (theoretical limit)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Theoretical Limit (100%)')

    # Add horizontal lines at 90% and 95%
    ax.axhline(y=0.9, color='orange', linestyle=':', linewidth=1, alpha=0.7, label='90% Efficiency')
    ax.axhline(y=0.95, color='purple', linestyle=':', linewidth=1, alpha=0.7, label='95% Efficiency')

    # Fill area above current efficiency (room for improvement)
    ax.fill_between(
        epochs,
        eff_clipped,
        1.0,
        where=(eff_clipped < 1.0),
        alpha=0.2,
        color='red',
        label='Room for Improvement'
    )

    # Labels and legend
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Efficiency (L_min / Loss)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Set y-axis from 0 to 1.1
    ax.set_ylim(0, 1.1)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


def plot_loss_decomposition(
    Var_F: float,
    Bias2: float,
    gap: float,
    loss_scale: float = 100.0,
    save_path: Optional[str] = None,
    title: str = "Loss Decomposition",
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Bar chart showing decomposition of loss into components.

    Components:
    - Var(F): Irreducible variance due to stochastic sampling
    - Bias^2: Irreducible bias from sampling
    - Gap: Reducible gap (actual - L_min)

    Args:
        Var_F: Variance component (already scaled)
        Bias2: Bias squared component (already scaled)
        gap: Reducible gap (already scaled)
        loss_scale: Scale factor (for display purposes)
        save_path: Path to save figure
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Data
    components = ['Var(F)\n(Irreducible)', 'Bias²\n(Irreducible)', 'Gap\n(Reducible)']
    values = [Var_F, Bias2, max(gap, 0)]  # Ensure gap is non-negative
    colors = ['#ff6b6b', '#feca57', '#48dbfb']

    # Create bars
    bars = ax.bar(components, values, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f'{val:.4f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=11, fontweight='bold'
        )

    # Compute percentages
    total = sum(values)
    L_min = Var_F + Bias2

    # Add percentage annotations
    if total > 0:
        pct_var = 100 * Var_F / total
        pct_bias = 100 * Bias2 / total
        pct_gap = 100 * max(gap, 0) / total

        # Add text below bars
        ax.text(0, -0.1 * max(values), f'{pct_var:.1f}%', ha='center', fontsize=10, transform=ax.get_xaxis_transform())
        ax.text(1, -0.1 * max(values), f'{pct_bias:.1f}%', ha='center', fontsize=10, transform=ax.get_xaxis_transform())
        ax.text(2, -0.1 * max(values), f'{pct_gap:.1f}%', ha='center', fontsize=10, transform=ax.get_xaxis_transform())

    # Add horizontal line for L_min
    ax.axhline(y=L_min, color='red', linestyle='--', linewidth=2, label=f'L_min = {L_min:.4f}')

    # Labels
    ax.set_ylabel('Loss Value', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', fontsize=10)

    # Add annotations explaining components
    textstr = (
        f'L_min (irreducible) = {L_min:.4f}\n'
        f'Total Loss = {total:.4f}\n'
        f'Efficiency = {100*L_min/total:.1f}%' if total > 0 else ''
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


def plot_loss_scatter(
    observed_loss: List[float],
    theoretical_L_min: List[float],
    epochs: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    title: str = "Observed Loss vs Theoretical L_min",
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    Scatter plot of observed loss vs theoretical L_min.

    Points should lie above the diagonal y=x line.
    Points on the diagonal = optimal (loss = L_min).

    Args:
        observed_loss: List of observed loss values
        theoretical_L_min: List of theoretical minimum values
        epochs: List of epoch numbers (for coloring)
        save_path: Path to save figure
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    observed = np.array(observed_loss)
    theoretical = np.array(theoretical_L_min)

    # Determine axis limits
    all_vals = np.concatenate([observed, theoretical])
    min_val = max(0, min(all_vals) * 0.9)
    max_val = max(all_vals) * 1.1

    # Plot diagonal line y=x (theoretical limit)
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y = x (Limit)')

    # Scatter plot with color by epoch
    if epochs is not None:
        epochs_arr = np.array(epochs)
        scatter = ax.scatter(
            theoretical, observed,
            c=epochs_arr, cmap='viridis',
            s=50, alpha=0.7, edgecolors='black', linewidth=0.5
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Epoch', fontsize=10)
    else:
        ax.scatter(
            theoretical, observed,
            c='blue', s=50, alpha=0.7, edgecolors='black', linewidth=0.5
        )

    # Labels
    ax.set_xlabel('Theoretical L_min', fontsize=12)
    ax.set_ylabel('Observed Loss', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Equal aspect ratio
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal')

    # Add annotation about violations
    n_violations = np.sum(observed < theoretical * 0.99)
    n_total = len(observed)
    ax.text(
        0.98, 0.02,
        f'Violations: {n_violations}/{n_total}\n(points below diagonal)',
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


def plot_empirical_vs_theoretical(
    empirical_E_F: List[float],
    theoretical_E_F: List[float],
    empirical_Var_F: List[float],
    theoretical_Var_F: List[float],
    epochs: List[int],
    save_path: Optional[str] = None,
    title: str = "Empirical vs Theoretical Statistics",
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot empirical and theoretical E[F] and Var[F] over epochs.

    Two subplots:
    - Left: E[F] empirical vs theoretical
    - Right: Var[F] empirical vs theoretical

    Args:
        empirical_E_F: Empirical E[F] values
        theoretical_E_F: Theoretical E[F] values
        empirical_Var_F: Empirical Var[F] values
        theoretical_Var_F: Theoretical Var[F] values
        epochs: Epoch numbers
        save_path: Path to save figure
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    epochs = np.array(epochs)

    # Left plot: E[F]
    ax1.plot(epochs, empirical_E_F, 'b-', linewidth=2, label='Empirical E[F]', marker='o', markersize=3)
    ax1.plot(epochs, theoretical_E_F, 'r--', linewidth=2, label='Theoretical E[F]')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('E[F]', fontsize=11)
    ax1.set_title('Expected Reliability E[F]', fontsize=12)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right plot: Var[F]
    ax2.plot(epochs, empirical_Var_F, 'b-', linewidth=2, label='Empirical Var[F]', marker='o', markersize=3)
    ax2.plot(epochs, theoretical_Var_F, 'r--', linewidth=2, label='Theoretical Var[F]')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Var[F]', fontsize=11)
    ax2.set_title('Variance of Reliability Var[F]', fontsize=12)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


def create_summary_figure(
    tracker_data: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Create a comprehensive 2x2 summary figure.

    Quadrants:
    1. Loss vs L_min over time
    2. Efficiency over time
    3. Loss decomposition (bar chart)
    4. Scatter plot

    Args:
        tracker_data: Dictionary from TheoreticalLossTracker.to_dict()
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)

    epochs = tracker_data['epochs']
    observed_loss = tracker_data['observed_loss']
    theoretical_L_min = tracker_data['theoretical_L_min']
    efficiency = tracker_data['efficiency']

    # Get final values for decomposition
    if len(epochs) > 0:
        final_Var_F = tracker_data['theoretical_Var_F'][-1]
        final_Bias2 = tracker_data['theoretical_Bias2'][-1]
        final_gap = tracker_data['gap'][-1]
    else:
        final_Var_F = 0
        final_Bias2 = 0
        final_gap = 0

    # Extract Bellman data if available
    bellman_data = tracker_data.get('bellman_lmin', None)

    # 1. Loss vs L_min (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(epochs, observed_loss, 'b-', linewidth=2, label='Observed Loss', marker='o', markersize=2)
    ax1.plot(epochs, theoretical_L_min, 'r--', linewidth=2, label='L_min (empirical)')
    if bellman_data is not None:
        bellman_val = bellman_data.get('L_min_bellman', None)
        naive_val = bellman_data.get('L_min_naive', None)
        if bellman_val is not None:
            ax1.axhline(y=bellman_val, color='green', linestyle='-.', linewidth=2,
                        label=f'L_min Bellman = {bellman_val:.4f}')
        if naive_val is not None:
            ax1.axhline(y=naive_val, color='purple', linestyle=':', linewidth=1.5,
                        label=f'L_min naive = {naive_val:.4f}')
    ax1.fill_between(epochs, theoretical_L_min, observed_loss, alpha=0.3, color='orange', label='Gap')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs Theoretical Minimum')
    ax1.legend(loc='upper right', fontsize=7)
    ax1.grid(True, alpha=0.3)

    # 2. Efficiency (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(epochs, efficiency, 'g-', linewidth=2, marker='o', markersize=2, label='Efficiency (empirical)')
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='100%')
    ax2.axhline(y=0.9, color='orange', linestyle=':', linewidth=1, alpha=0.7)
    if bellman_data is not None and bellman_data.get('L_min_bellman') is not None:
        # Show Bellman-based efficiency for the final loss
        bellman_val = bellman_data['L_min_bellman']
        if len(observed_loss) > 0 and observed_loss[-1] > 0:
            bellman_eff = bellman_val / observed_loss[-1]
            ax2.axhline(y=bellman_eff, color='green', linestyle='-.', linewidth=2,
                        label=f'Bellman eff = {bellman_eff*100:.1f}%')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Efficiency (L_min / Loss)')
    ax2.set_title('Training Efficiency')
    ax2.legend(loc='lower right', fontsize=7)
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)

    # 3. Loss decomposition (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    components = ['Var(F)', 'Bias²', 'Gap']
    values = [final_Var_F, final_Bias2, max(final_gap, 0)]
    colors_bar = ['#ff6b6b', '#feca57', '#48dbfb']
    bars = ax3.bar(components, values, color=colors_bar, edgecolor='black')
    for bar, val in zip(bars, values):
        ax3.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    ax3.axhline(y=final_Var_F + final_Bias2, color='red', linestyle='--', linewidth=2,
                label=f'L_min emp = {final_Var_F + final_Bias2:.4f}')
    if bellman_data is not None and bellman_data.get('L_min_bellman') is not None:
        ax3.axhline(y=bellman_data['L_min_bellman'], color='green', linestyle='-.',
                    linewidth=2, label=f'L_min Bellman = {bellman_data["L_min_bellman"]:.4f}')
    ax3.set_ylabel('Loss Value')
    ax3.set_title('Loss Decomposition (Final)')
    ax3.legend(loc='upper right', fontsize=7)

    # 4. Scatter plot (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    scatter = ax4.scatter(theoretical_L_min, observed_loss, c=epochs, cmap='viridis',
                         s=30, alpha=0.7, edgecolors='black', linewidth=0.3)
    all_vals = np.concatenate([observed_loss, theoretical_L_min])
    min_val, max_val = min(all_vals) * 0.9, max(all_vals) * 1.1
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax4.set_xlabel('Theoretical L_min')
    ax4.set_ylabel('Observed Loss')
    ax4.set_title('Loss vs L_min Scatter')
    ax4.set_xlim(min_val, max_val)
    ax4.set_ylim(min_val, max_val)
    ax4.set_aspect('equal')
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Epoch', fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


def generate_all_theoretical_plots(
    tracker_data: Dict[str, Any],
    checkpoint_dir: Path,
    verbose: bool = True
) -> Dict[str, Path]:
    """
    Generate all theoretical analysis plots.

    Args:
        tracker_data: Dictionary from TheoreticalLossTracker.to_dict()
        checkpoint_dir: Directory to save plots
        verbose: Print progress messages

    Returns:
        Dict mapping plot name to file path
    """
    checkpoint_dir = Path(checkpoint_dir)
    plots = {}

    if verbose:
        print("  Generating theoretical analysis plots...")

    epochs = tracker_data['epochs']
    if len(epochs) == 0:
        print("  Warning: No epochs in tracker data, skipping plots")
        return plots

    # Extract Bellman data if available
    bellman_data = tracker_data.get('bellman_lmin', None)

    # 1. Loss vs L_min
    path = checkpoint_dir / 'loss_vs_L_min.png'
    plot_loss_vs_L_min(
        epochs=tracker_data['epochs'],
        observed_loss=tracker_data['observed_loss'],
        theoretical_L_min=tracker_data['theoretical_L_min'],
        save_path=str(path),
        bellman_lmin=bellman_data,
    )
    plots['loss_vs_L_min'] = path
    plt.close()

    # 2. Efficiency
    path = checkpoint_dir / 'training_efficiency.png'
    plot_efficiency_over_time(
        epochs=tracker_data['epochs'],
        efficiency=tracker_data['efficiency'],
        save_path=str(path)
    )
    plots['training_efficiency'] = path
    plt.close()

    # 3. Loss decomposition (final values)
    path = checkpoint_dir / 'loss_decomposition.png'
    plot_loss_decomposition(
        Var_F=tracker_data['theoretical_Var_F'][-1],
        Bias2=tracker_data['theoretical_Bias2'][-1],
        gap=tracker_data['gap'][-1],
        save_path=str(path)
    )
    plots['loss_decomposition'] = path
    plt.close()

    # 4. Scatter plot
    path = checkpoint_dir / 'loss_scatter.png'
    plot_loss_scatter(
        observed_loss=tracker_data['observed_loss'],
        theoretical_L_min=tracker_data['theoretical_L_min'],
        epochs=tracker_data['epochs'],
        save_path=str(path)
    )
    plots['loss_scatter'] = path
    plt.close()

    # 5. Empirical vs theoretical
    if 'empirical_E_F' in tracker_data and len(tracker_data['empirical_E_F']) > 0:
        path = checkpoint_dir / 'empirical_vs_theoretical.png'
        plot_empirical_vs_theoretical(
            empirical_E_F=tracker_data['empirical_E_F'],
            theoretical_E_F=tracker_data['theoretical_E_F'],
            empirical_Var_F=tracker_data['empirical_Var_F'],
            theoretical_Var_F=tracker_data['theoretical_Var_F'],
            epochs=tracker_data['epochs'],
            save_path=str(path)
        )
        plots['empirical_vs_theoretical'] = path
        plt.close()

    # 6. Summary figure (2x2)
    path = checkpoint_dir / 'theoretical_analysis_summary.png'
    create_summary_figure(tracker_data, save_path=str(path))
    plots['theoretical_summary'] = path
    plt.close()

    if verbose:
        print(f"  Generated {len(plots)} theoretical analysis plots")

    return plots


if __name__ == '__main__':
    # Test visualization functions
    print("Testing Theoretical Visualization")
    print("="*60)

    # Create dummy data
    np.random.seed(42)
    n_epochs = 50
    epochs = list(range(1, n_epochs + 1))

    # Simulate decreasing loss
    theoretical_L_min = [0.1] * n_epochs  # Constant theoretical minimum
    observed_loss = [0.5 * np.exp(-0.05 * e) + 0.1 + 0.02 * np.random.randn() for e in epochs]
    efficiency = [l_min / obs if obs > 0 else 0 for l_min, obs in zip(theoretical_L_min, observed_loss)]

    # Create test directory
    test_dir = Path('/tmp/theoretical_viz_test')
    test_dir.mkdir(exist_ok=True)

    # Test individual plots
    print("\n1. Testing loss vs L_min plot...")
    plot_loss_vs_L_min(epochs, observed_loss, theoretical_L_min, str(test_dir / 'test_loss.png'))
    plt.close()

    print("2. Testing efficiency plot...")
    plot_efficiency_over_time(epochs, efficiency, str(test_dir / 'test_efficiency.png'))
    plt.close()

    print("3. Testing decomposition plot...")
    plot_loss_decomposition(0.05, 0.02, 0.08, save_path=str(test_dir / 'test_decomp.png'))
    plt.close()

    print("4. Testing scatter plot...")
    plot_loss_scatter(observed_loss, theoretical_L_min, epochs, str(test_dir / 'test_scatter.png'))
    plt.close()

    print(f"\nTest plots saved to: {test_dir}")
    print("Test passed!")
