"""
Visualization functions for Empirical Loss Analysis.

Creates plots of observed loss and empirical statistics (E[F], Var[F], Bias²).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


def plot_observed_loss(
    epochs: List[int],
    observed_loss: List[float],
    save_path: Optional[str] = None,
    title: str = "Observed Loss Over Training",
    figsize: Tuple[int, int] = (10, 6),
    bellman_lmin: Optional[Dict[str, Any]] = None,
) -> plt.Figure:
    """
    Plot observed loss over epochs.

    Shows:
    - Observed loss curve (solid blue)
    - Bellman L_min horizontal line (if available)

    Args:
        epochs: List of epoch numbers
        observed_loss: List of observed loss values
        save_path: Path to save figure (optional)
        title: Plot title
        figsize: Figure size
        bellman_lmin: Dict with Bellman results (keys: L_min_bellman, L_min_forward)

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = np.array(epochs)
    observed = np.array(observed_loss)

    # Plot observed loss
    ax.plot(epochs, observed, 'b-', linewidth=2, label='Observed Loss', marker='o', markersize=3)

    # Plot Bellman L_min lines if available
    if bellman_lmin is not None:
        bellman_val = bellman_lmin.get('L_min_bellman', None)
        if bellman_val is not None:
            ax.axhline(y=bellman_val, color='green', linestyle='-.',
                       linewidth=2.5, label=f'L_min Bellman = {bellman_val:.4f}')

    all_vals = observed.copy()
    if bellman_lmin is not None:
        extra = [v for v in [bellman_lmin.get('L_min_bellman')] if v is not None]
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


def plot_loss_decomposition(
    Var_F: float,
    Bias2: float,
    loss_scale: float = 100.0,
    save_path: Optional[str] = None,
    title: str = "Empirical Statistics: Var[F] vs Bias²",
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Bar chart showing decomposition into Var[F] and Bias².

    Components:
    - Var(F): Variance due to stochastic sampling
    - Bias²: (E[F] - F*)²

    Args:
        Var_F: Variance component (already scaled)
        Bias2: Bias squared component (already scaled)
        loss_scale: Scale factor (for display purposes)
        save_path: Path to save figure
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Data
    components = ['Var(F)', 'Bias²']
    values = [Var_F, Bias2]
    colors = ['#ff6b6b', '#feca57']

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

    # Add percentage annotations
    if total > 0:
        pct_var = 100 * Var_F / total
        pct_bias = 100 * Bias2 / total

        # Add text below bars
        ax.text(0, -0.1 * max(values), f'{pct_var:.1f}%', ha='center', fontsize=10, transform=ax.get_xaxis_transform())
        ax.text(1, -0.1 * max(values), f'{pct_bias:.1f}%', ha='center', fontsize=10, transform=ax.get_xaxis_transform())

    # Labels
    ax.set_ylabel('Value (scaled)', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add annotations
    textstr = (
        f'Var[F] + Bias² = {total:.4f}\n'
        f'Var[F] = {100*Var_F/total:.1f}%\n'
        f'Bias² = {100*Bias2/total:.1f}%'
    ) if total > 0 else ''
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

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
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Create a summary figure with observed loss and decomposition.

    Two panels:
    1. Observed loss over time (+ Bellman L_min if available)
    2. Var[F] vs Bias² decomposition (bar chart)

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

    # Get final empirical values
    if len(epochs) > 0:
        final_Var_F = tracker_data.get('empirical_Var_F', [0])[-1]
        final_Bias2 = tracker_data.get('empirical_Bias2', [0])[-1]
    else:
        final_Var_F = 0
        final_Bias2 = 0

    # Extract Bellman data if available
    bellman_data = tracker_data.get('bellman_lmin', None)

    # 1. Observed loss (left)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(epochs, observed_loss, 'b-', linewidth=2, label='Observed Loss', marker='o', markersize=2)
    if bellman_data is not None:
        bellman_val = bellman_data.get('L_min_bellman', None)
        if bellman_val is not None:
            ax1.axhline(y=bellman_val, color='green', linestyle='-.',
                        linewidth=2, label=f'L_min Bellman = {bellman_val:.4f}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Observed Loss Over Training')
    ax1.legend(loc='upper right', fontsize=7)
    ax1.grid(True, alpha=0.3)

    # 2. Decomposition (right)
    ax2 = fig.add_subplot(1, 2, 2)
    components = ['Var(F)', 'Bias²']
    values = [final_Var_F, final_Bias2]
    colors_bar = ['#ff6b6b', '#feca57']
    bars = ax2.bar(components, values, color=colors_bar, edgecolor='black')
    for bar, val in zip(bars, values):
        ax2.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    ax2.set_ylabel('Value (scaled)')
    ax2.set_title('Empirical Statistics (Final)')

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
    Generate all analysis plots.

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
        print("  Generating analysis plots...")

    epochs = tracker_data['epochs']
    if len(epochs) == 0:
        print("  Warning: No epochs in tracker data, skipping plots")
        return plots

    # Extract Bellman data if available
    bellman_data = tracker_data.get('bellman_lmin', None)

    # 1. Observed loss
    path = checkpoint_dir / 'observed_loss.png'
    plot_observed_loss(
        epochs=tracker_data['epochs'],
        observed_loss=tracker_data['observed_loss'],
        save_path=str(path),
        bellman_lmin=bellman_data,
    )
    plots['observed_loss'] = path
    plt.close()

    # 2. Var[F] vs Bias² decomposition (final values)
    if 'empirical_Var_F' in tracker_data and len(tracker_data['empirical_Var_F']) > 0:
        path = checkpoint_dir / 'loss_decomposition.png'
        plot_loss_decomposition(
            Var_F=tracker_data['empirical_Var_F'][-1],
            Bias2=tracker_data['empirical_Bias2'][-1],
            save_path=str(path)
        )
        plots['loss_decomposition'] = path
        plt.close()

    # 3. Empirical vs theoretical (if both available)
    if ('empirical_E_F' in tracker_data and len(tracker_data['empirical_E_F']) > 0
            and 'theoretical_E_F' in tracker_data and len(tracker_data.get('theoretical_E_F', [])) > 0):
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

    # 4. Summary figure
    path = checkpoint_dir / 'theoretical_analysis_summary.png'
    create_summary_figure(tracker_data, save_path=str(path))
    plots['theoretical_summary'] = path
    plt.close()

    if verbose:
        print(f"  Generated {len(plots)} analysis plots")

    return plots


if __name__ == '__main__':
    # Test visualization functions
    print("Testing Visualization")
    print("="*60)

    # Create dummy data
    np.random.seed(42)
    n_epochs = 50
    epochs = list(range(1, n_epochs + 1))

    # Simulate decreasing loss
    observed_loss = [0.5 * np.exp(-0.05 * e) + 0.1 + 0.02 * np.random.randn() for e in epochs]

    # Create test directory
    test_dir = Path('/tmp/theoretical_viz_test')
    test_dir.mkdir(exist_ok=True)

    # Test individual plots
    print("\n1. Testing observed loss plot...")
    plot_observed_loss(epochs, observed_loss, str(test_dir / 'test_loss.png'))
    plt.close()

    print("2. Testing decomposition plot...")
    plot_loss_decomposition(0.05, 0.02, save_path=str(test_dir / 'test_decomp.png'))
    plt.close()

    print(f"\nTest plots saved to: {test_dir}")
    print("Test passed!")
