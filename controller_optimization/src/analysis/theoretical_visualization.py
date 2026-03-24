"""
Visualization functions for Theoretical Loss Analysis.

Creates plots comparing observed loss vs theoretical minimum (L_min).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


def apply_plot_style():
    plt.rcParams.update({
        'font.family':           'monospace',
        'font.size':             8,
        'axes.titlesize':        9,
        'axes.titleweight':      'normal',
        'axes.titlelocation':    'left',
        'axes.labelsize':        8,
        'axes.labelweight':      'normal',
        'axes.linewidth':        0.5,
        'axes.spines.top':       False,
        'axes.spines.right':     False,
        'axes.grid':             True,
        'grid.color':            '#DDDDDD',
        'grid.linewidth':        0.4,
        'grid.alpha':            1.0,
        'xtick.labelsize':       7.5,
        'ytick.labelsize':       7.5,
        'xtick.major.width':     0.4,
        'ytick.major.width':     0.4,
        'xtick.major.size':      3,
        'ytick.major.size':      3,
        'legend.fontsize':       7.5,
        'legend.framealpha':     0.9,
        'legend.edgecolor':      '#DDDDDD',
        'legend.fancybox':       False,
        'legend.borderpad':      0.4,
        'figure.facecolor':      'white',
        'axes.facecolor':        'white',
        'savefig.facecolor':     'white',
        'savefig.dpi':           150,
        'savefig.bbox':          'tight',
        'lines.linewidth':       1.4,
    })


def _get_surrogate_subtitle(surrogate_type: Optional[str],
                            bellman_lmin: Optional[Dict[str, Any]],
                            surrogate_lmin: Optional[float]) -> str:
    """Build subtitle annotation indicating active surrogate mode."""
    if surrogate_type == 'casualit' and surrogate_lmin is not None:
        return "Surrogate: CasualiT  |  L\u0302_min via surrogate sampling (N=500)"
    elif surrogate_type == 'reliability_function' and bellman_lmin is not None and bellman_lmin.get('L_min_bellman') is not None:
        return "Surrogate: reliability_function  |  L_min via Bellman backward induction"
    elif surrogate_type == 'casualit':
        return "Surrogate: CasualiT  |  (L\u0302_min not available)"
    elif surrogate_type == 'reliability_function':
        return "Surrogate: reliability_function  |  (Bellman L_min not available)"
    return ""


def plot_loss_vs_L_min(
    epochs: List[int],
    observed_loss: List[float],
    theoretical_L_min: List[float],
    save_path: Optional[str] = None,
    title: str = "Loss vs Theoretical Minimum",
    figsize: Tuple[int, int] = (10, 6),
    bellman_lmin: Optional[Dict[str, Any]] = None,
    surrogate_lmin: Optional[float] = None,
    var_F: Optional[float] = None,
    surrogate_type: Optional[str] = None,
) -> plt.Figure:
    """
    Plot observed loss and the complete loss hierarchy over epochs.

    Shows (bottom to top, only if computed):
    1. Var(F) horizontal line — irreducible noise floor (always)
    2. L_min Bellman horizontal line — only for reliability_function surrogate
    3. L̂_min empirical horizontal line — only for casualit surrogate
    4. Observed Loss curve (always)
    Shaded fill between lowest reference and observed loss.

    Args:
        epochs: List of epoch numbers
        observed_loss: List of observed loss values
        theoretical_L_min: List of L_min values (Var[F] + Bias²)
        save_path: Path to save figure (optional)
        title: Plot title
        figsize: Figure size
        bellman_lmin: Dict with Bellman results (keys: L_min_bellman, L_min_forward)
        surrogate_lmin: Empirical L̂_min from surrogate sampling (casualit only)
        var_F: Var(F) value for irreducible noise floor
        surrogate_type: 'reliability_function' or 'casualit'

    Returns:
        Matplotlib Figure object
    """
    apply_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    epochs = np.array(epochs)
    observed = np.array(observed_loss)

    # Track lowest reference line for shaded fill
    lowest_ref = None

    # 1. Var(F) horizontal line — ALWAYS shown if available and > 0
    if var_F is not None and var_F > 0:
        ax.axhline(y=var_F, color='grey', linestyle='-', linewidth=1.5,
                   label=f'Var(F) = {var_F:.4f} \u2014 irreducible noise floor')
        lowest_ref = var_F

    # 2. L_min Bellman — only for reliability_function surrogate
    bellman_val = None
    if surrogate_type == 'reliability_function' and bellman_lmin is not None:
        bellman_val = bellman_lmin.get('L_min_bellman', None)
        if bellman_val is not None:
            ax.axhline(y=bellman_val, color='green', linestyle='--', linewidth=2,
                       label=f'L_min Bellman = {bellman_val:.4f} \u2014 optimal Gaussian policy')
            lowest_ref = min(lowest_ref, bellman_val) if lowest_ref is not None else bellman_val

    # 3. L̂_min empirical (surrogate) — only for casualit surrogate
    if surrogate_type == 'casualit' and surrogate_lmin is not None:
        ax.axhline(y=surrogate_lmin, color='purple', linestyle='-.', linewidth=2,
                   label=f'L\u0302_min empirical = {surrogate_lmin:.4f} \u2014 surrogate-based floor')
        lowest_ref = min(lowest_ref, surrogate_lmin) if lowest_ref is not None else surrogate_lmin

    # 4. Observed Loss curve — ALWAYS shown
    ax.plot(epochs, observed, 'b-', linewidth=2, label='Observed Loss', marker='o', markersize=3)

    # Shaded fill between lowest reference line and observed loss
    if lowest_ref is not None:
        ax.fill_between(
            epochs,
            lowest_ref,
            observed,
            alpha=0.2,
            color='orange',
            label='Reducible gap (suboptimality)'
        )

    # Collect all values for y-axis limits
    all_vals = list(observed)
    if var_F is not None and var_F > 0:
        all_vals.append(var_F)
    if bellman_val is not None:
        all_vals.append(bellman_val)
    if surrogate_type == 'casualit' and surrogate_lmin is not None:
        all_vals.append(surrogate_lmin)
    all_vals = np.array(all_vals)

    # Labels and legend
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Subtitle annotation indicating active mode
    subtitle = _get_surrogate_subtitle(surrogate_type, bellman_lmin, surrogate_lmin)
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, fontsize=8,
                ha='center', va='bottom', style='italic', color='#555555')

    ax.legend(loc='upper right', fontsize=9)
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
    figsize: Tuple[int, int] = (10, 5),
    bellman_lmin: Optional[Dict[str, Any]] = None,
    observed_loss: Optional[List[float]] = None,
    surrogate_lmin: Optional[float] = None,
    var_F: Optional[float] = None,
    surrogate_type: Optional[str] = None,
) -> plt.Figure:
    """
    Plot up to three training efficiency curves over epochs.

    Hierarchy: eta_Var <= eta_min <= eta_emp <= 1
      eta_Var  = Var(F) / L(Phi, epoch)          — always shown
      eta_min  = L_min_Bellman / L(Phi, epoch)    — reliability_function only
      eta_emp  = L_hat_min / L(Phi, epoch)        — casualit only

    Args:
        epochs: List of epoch numbers
        efficiency: List of empirical efficiency values (L_min_emp / Loss)
        save_path: Path to save figure (optional)
        title: Plot title
        figsize: Figure size
        bellman_lmin: Dict with Bellman results (keys: L_min_bellman, L_min_forward)
        observed_loss: List of observed loss values (needed for computing efficiencies)
        surrogate_lmin: Empirical L̂_min from surrogate sampling (casualit only)
        var_F: Var(F) value for irreducible noise floor
        surrogate_type: 'reliability_function' or 'casualit'

    Returns:
        Matplotlib Figure object
    """
    apply_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    epochs = np.array(epochs)
    obs_arr = np.array(observed_loss) if observed_loss is not None else None

    # η_Var — ALWAYS shown if var_F available and observed_loss provided
    if var_F is not None and var_F > 0 and obs_arr is not None and len(obs_arr) == len(epochs):
        eta_var = np.where(obs_arr > 0, var_F / obs_arr, 0.0)
        eta_var_clipped = np.clip(eta_var, 0, 1.05)
        ax.plot(epochs, eta_var_clipped, color='grey', linestyle='-', linewidth=1.8,
                label='\u03B7_Var = Var(F)/L(\u03A6) \u2014 irreducible floor')

    # η_min — only for reliability_function with Bellman
    bellman_val = None
    if surrogate_type == 'reliability_function' and bellman_lmin is not None:
        bellman_val = bellman_lmin.get('L_min_bellman', None)
    if bellman_val is not None and obs_arr is not None and len(obs_arr) == len(epochs):
        eta_min = np.where(obs_arr > 0, bellman_val / obs_arr, 0.0)
        eta_min_clipped = np.clip(eta_min, 0, 1.05)
        ax.plot(epochs, eta_min_clipped, color='green', linestyle='--', linewidth=2,
                label='\u03B7_min = L_min Bellman / L(\u03A6)')

    # η_emp — only for casualit with surrogate_lmin
    if surrogate_type == 'casualit' and surrogate_lmin is not None and obs_arr is not None and len(obs_arr) == len(epochs):
        eta_emp = np.where(obs_arr > 0, surrogate_lmin / obs_arr, 0.0)
        eta_emp_clipped = np.clip(eta_emp, 0, 1.05)
        ax.plot(epochs, eta_emp_clipped, color='purple', linestyle='-', linewidth=2,
                label='\u03B7_emp = L\u0302_min / L(\u03A6) \u2014 empirical efficiency')

    # Reference lines
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
               label='Theoretical limit 100%')
    ax.axhline(y=0.9, color='orange', linestyle=':', linewidth=1, alpha=0.7,
               label='90%')

    # Labels and legend
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Efficiency', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Subtitle annotation
    subtitle = _get_surrogate_subtitle(surrogate_type, bellman_lmin, surrogate_lmin)
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, fontsize=8,
                ha='center', va='bottom', style='italic', color='#555555')

    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # y-axis clipped to [0, 1.05]
    ax.set_ylim(0, 1.05)

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
    apply_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

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
            fontsize=11
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
    props = dict(boxstyle='square,pad=0.3', facecolor='white',
                 edgecolor='#CCCCCC', linewidth=0.5, alpha=0.9)
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
    apply_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

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
        bbox=dict(boxstyle='square,pad=0.3', facecolor='white',
                  edgecolor='#CCCCCC', linewidth=0.5, alpha=0.9)
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
    figsize: Tuple[int, int] = (14, 10),
    surrogate_lmin: Optional[float] = None,
) -> plt.Figure:
    """
    Create a comprehensive 2x2 summary figure.

    Quadrants:
    1. Loss vs L_min over time (complete hierarchy)
    2. Efficiency over time (up to 3 curves)
    3. Loss decomposition (bar chart)
    4. Scatter plot

    Args:
        tracker_data: Dictionary from TheoreticalLossTracker.to_dict()
        save_path: Path to save figure
        figsize: Figure size
        surrogate_lmin: Empirical L̂_min from surrogate sampling (casualit only)

    Returns:
        Matplotlib Figure object
    """
    apply_plot_style()
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

    # Extract Bellman data and surrogate info
    bellman_data = tracker_data.get('bellman_lmin', None)
    if surrogate_lmin is None:
        surrogate_lmin = tracker_data.get('surrogate_lmin', None)

    # Determine surrogate type from available data
    if bellman_data is not None and bellman_data.get('L_min_bellman') is not None:
        surrogate_type = 'reliability_function'
    elif surrogate_lmin is not None:
        surrogate_type = 'casualit'
    else:
        surrogate_type = None

    var_F = final_Var_F if final_Var_F > 0 else None
    obs_arr = np.array(observed_loss)
    subtitle = _get_surrogate_subtitle(surrogate_type, bellman_data, surrogate_lmin)

    # 1. Loss vs L_min (top left) — complete hierarchy
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    lowest_ref = None

    # Var(F) line — always
    if var_F is not None:
        ax1.axhline(y=var_F, color='grey', linestyle='-', linewidth=1.2,
                     label=f'Var(F) = {var_F:.4f}')
        lowest_ref = var_F

    # L_min Bellman — reliability_function only
    bellman_val = None
    if surrogate_type == 'reliability_function' and bellman_data is not None:
        bellman_val = bellman_data.get('L_min_bellman', None)
        if bellman_val is not None:
            ax1.axhline(y=bellman_val, color='green', linestyle='--', linewidth=1.8,
                         label=f'L_min Bellman = {bellman_val:.4f}')
            lowest_ref = min(lowest_ref, bellman_val) if lowest_ref is not None else bellman_val

    # L̂_min surrogate — casualit only
    if surrogate_type == 'casualit' and surrogate_lmin is not None:
        ax1.axhline(y=surrogate_lmin, color='purple', linestyle='-.', linewidth=1.8,
                     label=f'L\u0302_min emp = {surrogate_lmin:.4f}')
        lowest_ref = min(lowest_ref, surrogate_lmin) if lowest_ref is not None else surrogate_lmin

    # Observed loss
    ax1.plot(epochs, observed_loss, 'b-', linewidth=2, label='Observed Loss', marker='o', markersize=2)

    # Shaded fill
    if lowest_ref is not None:
        ax1.fill_between(epochs, lowest_ref, observed_loss, alpha=0.2, color='orange',
                         label='Reducible gap')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs Theoretical Minimum')
    if subtitle:
        ax1.text(0.5, 1.02, subtitle, transform=ax1.transAxes, fontsize=6,
                 ha='center', va='bottom', style='italic', color='#555555')
    ax1.legend(loc='upper right', fontsize=7)
    ax1.grid(True, alpha=0.3)

    # 2. Efficiency (top right) — up to 3 curves
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # η_Var — always
    if var_F is not None and len(obs_arr) > 0:
        eta_var = np.where(obs_arr > 0, var_F / obs_arr, 0.0)
        ax2.plot(epochs, np.clip(eta_var, 0, 1.05), color='grey', linestyle='-', linewidth=1.5,
                 label='\u03B7_Var')

    # η_min — reliability_function only
    if surrogate_type == 'reliability_function' and bellman_val is not None and len(obs_arr) > 0:
        eta_min = np.where(obs_arr > 0, bellman_val / obs_arr, 0.0)
        ax2.plot(epochs, np.clip(eta_min, 0, 1.05), color='green', linestyle='--', linewidth=1.8,
                 label='\u03B7_min (Bellman)')

    # η_emp — casualit only
    if surrogate_type == 'casualit' and surrogate_lmin is not None and len(obs_arr) > 0:
        eta_emp = np.where(obs_arr > 0, surrogate_lmin / obs_arr, 0.0)
        ax2.plot(epochs, np.clip(eta_emp, 0, 1.05), color='purple', linestyle='-', linewidth=1.8,
                 label='\u03B7_emp (surrogate)')

    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='100%')
    ax2.axhline(y=0.9, color='orange', linestyle=':', linewidth=1, alpha=0.7, label='90%')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Efficiency')
    ax2.set_title('Training Efficiency')
    if subtitle:
        ax2.text(0.5, 1.02, subtitle, transform=ax2.transAxes, fontsize=6,
                 ha='center', va='bottom', style='italic', color='#555555')
    ax2.legend(loc='lower right', fontsize=7)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    # 3. Loss decomposition (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
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
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
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

    # Extract surrogate_lmin (casualit) if available
    surrogate_lmin_val = tracker_data.get('surrogate_lmin', None)

    # Determine surrogate type from available data
    if bellman_data is not None and bellman_data.get('L_min_bellman') is not None:
        surrogate_type = 'reliability_function'
    elif surrogate_lmin_val is not None:
        surrogate_type = 'casualit'
    else:
        surrogate_type = None

    # Extract Var(F) — constant across epochs
    var_F = None
    var_F_list = tracker_data.get('theoretical_Var_F', [])
    if len(var_F_list) > 0:
        var_F = var_F_list[-1]
        if var_F <= 0:
            var_F = None

    # 1. Loss vs L_min
    path = checkpoint_dir / 'loss_vs_L_min.png'
    plot_loss_vs_L_min(
        epochs=tracker_data['epochs'],
        observed_loss=tracker_data['observed_loss'],
        theoretical_L_min=tracker_data['theoretical_L_min'],
        save_path=str(path),
        bellman_lmin=bellman_data,
        surrogate_lmin=surrogate_lmin_val,
        var_F=var_F,
        surrogate_type=surrogate_type,
    )
    plots['loss_vs_L_min'] = path
    plt.close()

    # 2. Efficiency
    path = checkpoint_dir / 'training_efficiency.png'
    plot_efficiency_over_time(
        epochs=tracker_data['epochs'],
        efficiency=tracker_data['efficiency'],
        save_path=str(path),
        bellman_lmin=bellman_data,
        observed_loss=tracker_data['observed_loss'],
        surrogate_lmin=surrogate_lmin_val,
        var_F=var_F,
        surrogate_type=surrogate_type,
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
    create_summary_figure(tracker_data, save_path=str(path), surrogate_lmin=surrogate_lmin_val)
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
