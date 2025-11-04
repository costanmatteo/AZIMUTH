"""
Visualization utilities specific to Bayesian Neural Networks

This module provides specialized plotting functions for Bayesian predictions
including uncertainty bands, confidence intervals, and epistemic uncertainty.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Tuple


def plot_bayesian_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_nlls: List[float] = None,
    train_kls: List[float] = None,
    val_nlls: List[float] = None,
    val_kls: List[float] = None,
    val_uncertainties: List[float] = None,
    save_path: Optional[str] = None
):
    """
    Plot comprehensive Bayesian training history.

    Shows:
    - Total loss (ELBO)
    - NLL (data fit) component
    - KL divergence (regularization) component
    - Validation uncertainty

    Args:
        train_losses: Training ELBO losses
        val_losses: Validation ELBO losses
        train_nlls: Training NLL component (optional)
        train_kls: Training KL component (optional)
        val_nlls: Validation NLL component (optional)
        val_kls: Validation KL component (optional)
        val_uncertainties: Validation uncertainties (optional)
        save_path: Path to save the plot
    """
    epochs = range(1, len(train_losses) + 1)

    # Determine number of subplots
    n_plots = 1
    if train_nlls is not None and train_kls is not None:
        n_plots = 2
    if val_uncertainties is not None:
        n_plots = max(2, n_plots)

    fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # Plot 1: Total Loss (ELBO)
    ax = axes[0]
    ax.plot(epochs, train_losses, 'b-', label='Train ELBO', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Val ELBO', linewidth=2)
    ax.set_title('ELBO Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: NLL and KL components
    if n_plots >= 2 and train_nlls is not None:
        ax = axes[1]
        ax.plot(epochs, train_nlls, 'b-', label='Train NLL', linewidth=2, alpha=0.7)
        ax.plot(epochs, train_kls, 'b--', label='Train KL', linewidth=2, alpha=0.7)
        if val_nlls is not None:
            ax.plot(epochs, val_nlls, 'r-', label='Val NLL', linewidth=2, alpha=0.7)
        if val_kls is not None:
            ax.plot(epochs, val_kls, 'r--', label='Val KL', linewidth=2, alpha=0.7)
        ax.set_title('Loss Components', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Plot 3: Uncertainty evolution (if we have 3 plots)
    if n_plots >= 3 and val_uncertainties is not None:
        ax = axes[2]
        ax.plot(epochs, val_uncertainties, 'g-', label='Val Uncertainty', linewidth=2)
        ax.set_title('Validation Uncertainty', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Average Std Dev', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    elif n_plots == 2 and val_uncertainties is not None:
        # If we only have 2 plots, add uncertainty to the second plot
        ax = axes[1]
        ax2 = ax.twinx()
        ax2.plot(epochs, val_uncertainties, 'g-', label='Val Uncertainty', linewidth=2, alpha=0.7)
        ax2.set_ylabel('Uncertainty (Std Dev)', fontsize=12, color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bayesian training history saved to: {save_path}")

    plt.close()


def plot_predictions_with_uncertainty(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    output_names: Optional[List[str]] = None,
    confidence_level: float = 0.95,
    save_path: Optional[str] = None
):
    """
    Plot predictions with uncertainty bands.

    Args:
        y_true: Actual values (n_samples, n_outputs)
        y_pred_mean: Predicted mean values (n_samples, n_outputs)
        y_pred_std: Predicted standard deviations (n_samples, n_outputs)
        output_names: Names for each output
        confidence_level: Confidence level for bands (0.68, 0.95, or 0.99)
        save_path: Path to save the plot
    """
    n_outputs = y_true.shape[1] if len(y_true.shape) > 1 else 1

    if n_outputs == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred_mean = y_pred_mean.reshape(-1, 1)
        y_pred_std = y_pred_std.reshape(-1, 1)

    # Z-score for confidence level
    z_scores = {0.68: 1.0, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence_level, 1.96)

    # Determine subplot layout
    n_cols = min(3, n_outputs)
    n_rows = (n_outputs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 6*n_rows))
    if n_outputs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(n_outputs):
        ax = axes[i]

        # Sort by true values for better visualization
        sort_idx = np.argsort(y_true[:, i])
        y_true_sorted = y_true[sort_idx, i]
        y_pred_sorted = y_pred_mean[sort_idx, i]
        y_std_sorted = y_pred_std[sort_idx, i]

        # Scatter plot with color based on uncertainty
        scatter = ax.scatter(
            y_true[:, i],
            y_pred_mean[:, i],
            c=y_pred_std[:, i],
            cmap='viridis',
            alpha=0.6,
            s=30,
            edgecolors='black',
            linewidth=0.5
        )

        # Perfect prediction line
        min_val = min(y_true[:, i].min(), y_pred_mean[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred_mean[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect', alpha=0.7)

        # Confidence band
        lower_bound = y_pred_sorted - z * y_std_sorted
        upper_bound = y_pred_sorted + z * y_std_sorted
        ax.fill_between(
            y_true_sorted,
            lower_bound,
            upper_bound,
            alpha=0.2,
            color='blue',
            label=f'{int(confidence_level*100)}% Confidence'
        )

        # Colorbar for uncertainty
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Prediction Std Dev', fontsize=9)

        # Labels
        title = output_names[i] if output_names else f'Output {i+1}'
        ax.set_title(f'{title} - with Uncertainty', fontsize=12, fontweight='bold')
        ax.set_xlabel('Actual Value', fontsize=10)
        ax.set_ylabel('Predicted Mean', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Remove empty subplots
    for i in range(n_outputs, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions with uncertainty saved to: {save_path}")

    plt.close()


def plot_uncertainty_calibration(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    output_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Plot uncertainty calibration diagnostic.

    Shows how well the predicted uncertainties match actual errors.
    Well-calibrated uncertainties should have errors within predicted bands.

    Args:
        y_true: Actual values
        y_pred_mean: Predicted mean values
        y_pred_std: Predicted standard deviations
        output_names: Names for each output
        save_path: Path to save the plot
    """
    n_outputs = y_true.shape[1] if len(y_true.shape) > 1 else 1

    if n_outputs == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred_mean = y_pred_mean.reshape(-1, 1)
        y_pred_std = y_pred_std.reshape(-1, 1)

    # Calculate errors
    errors = np.abs(y_true - y_pred_mean)

    # Determine subplot layout
    n_cols = min(3, n_outputs)
    n_rows = (n_outputs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_outputs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(n_outputs):
        ax = axes[i]

        # Scatter: predicted std vs actual error
        ax.scatter(y_pred_std[:, i], errors[:, i], alpha=0.5, s=20)

        # Perfect calibration line (error = std)
        max_val = max(y_pred_std[:, i].max(), errors[:, i].max())
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Calibration')

        # 2-sigma line
        ax.plot([0, max_val], [0, 2*max_val], 'g--', linewidth=1.5, alpha=0.7, label='2σ bound')

        # Labels
        title = output_names[i] if output_names else f'Output {i+1}'
        ax.set_title(f'{title} - Calibration', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Std Dev', fontsize=10)
        ax.set_ylabel('Actual |Error|', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add calibration statistics
        within_1sigma = np.sum(errors[:, i] <= y_pred_std[:, i]) / len(errors[:, i]) * 100
        within_2sigma = np.sum(errors[:, i] <= 2*y_pred_std[:, i]) / len(errors[:, i]) * 100
        textstr = f'Within 1σ: {within_1sigma:.1f}%\nWithin 2σ: {within_2sigma:.1f}%'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Remove empty subplots
    for i in range(n_outputs, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Uncertainty calibration plot saved to: {save_path}")

    plt.close()


def plot_epistemic_uncertainty_heatmap(
    y_true: np.ndarray,
    y_pred_std: np.ndarray,
    output_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Plot heatmap of epistemic uncertainty across samples and outputs.

    Args:
        y_true: Actual values (for sample ordering)
        y_pred_std: Predicted standard deviations
        output_names: Names for each output
        save_path: Path to save the plot
    """
    n_outputs = y_pred_std.shape[1] if len(y_pred_std.shape) > 1 else 1

    if n_outputs == 1:
        y_pred_std = y_pred_std.reshape(-1, 1)

    if output_names is None:
        output_names = [f'Output {i+1}' for i in range(n_outputs)]

    # Sort samples by average uncertainty
    avg_uncertainty = y_pred_std.mean(axis=1)
    sort_idx = np.argsort(avg_uncertainty)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(8, n_outputs*1.5), 8))

    im = ax.imshow(
        y_pred_std[sort_idx].T,
        aspect='auto',
        cmap='YlOrRd',
        interpolation='nearest'
    )

    # Labels
    ax.set_title('Epistemic Uncertainty Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Sample (sorted by avg. uncertainty)', fontsize=12)
    ax.set_ylabel('Output Variable', fontsize=12)
    ax.set_yticks(range(n_outputs))
    ax.set_yticklabels(output_names)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Std Dev (Epistemic Uncertainty)', fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Epistemic uncertainty heatmap saved to: {save_path}")

    plt.close()


def plot_confidence_intervals(
    sample_indices: np.ndarray,
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    confidence_intervals: dict,
    output_idx: int = 0,
    output_name: Optional[str] = None,
    save_path: Optional[str] = None
):
    """
    Plot specific samples with confidence intervals.

    Useful for visualizing uncertainty on a subset of predictions.

    Args:
        sample_indices: Indices of samples to plot
        y_true: Actual values
        y_pred_mean: Predicted mean values
        confidence_intervals: Dict with CI bounds (e.g., {'68%': (lower, upper), ...})
        output_idx: Which output to plot (default: 0)
        output_name: Name of the output
        save_path: Path to save the plot
    """
    n_samples = len(sample_indices)

    fig, ax = plt.subplots(figsize=(max(10, n_samples*0.5), 6))

    x_positions = range(n_samples)

    # Extract values for selected samples and output
    y_true_sel = y_true[sample_indices, output_idx] if len(y_true.shape) > 1 else y_true[sample_indices]
    y_pred_sel = y_pred_mean[sample_indices, output_idx] if len(y_pred_mean.shape) > 1 else y_pred_mean[sample_indices]

    # Plot actual values
    ax.scatter(x_positions, y_true_sel, color='red', s=100, marker='o', label='Actual', zorder=3)

    # Plot predicted means
    ax.scatter(x_positions, y_pred_sel, color='blue', s=100, marker='x', label='Predicted Mean', zorder=3)

    # Plot confidence intervals
    colors = {'68%': 'lightblue', '95%': 'lightgreen', '99%': 'lightyellow'}
    for ci_level, (lower, upper) in sorted(confidence_intervals.items(), reverse=True):
        lower_sel = lower[sample_indices, output_idx] if len(lower.shape) > 1 else lower[sample_indices]
        upper_sel = upper[sample_indices, output_idx] if len(upper.shape) > 1 else upper[sample_indices]

        for i, x in enumerate(x_positions):
            ax.plot([x, x], [lower_sel[i], upper_sel[i]], color=colors.get(ci_level, 'gray'),
                   linewidth=8, alpha=0.5, label=ci_level if i == 0 else '')

    # Labels
    title = f'{output_name} - Confidence Intervals' if output_name else 'Confidence Intervals'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(sample_indices)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confidence intervals plot saved to: {save_path}")

    plt.close()
