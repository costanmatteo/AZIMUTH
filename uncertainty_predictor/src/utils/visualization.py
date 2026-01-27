"""
Visualization utilities for uncertainty predictions

This module provides functions to visualize predictions with uncertainty bounds.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path


def plot_training_history(train_losses, val_losses, train_mse=None, val_mse=None, save_path=None):
    """
    Plot training history for uncertainty model.

    Args:
        train_losses (list): Training NLL losses
        val_losses (list): Validation NLL losses
        train_mse (list): Training MSE (optional)
        val_mse (list): Validation MSE (optional)
        save_path (str or Path): Path to save the plot
    """
    n_plots = 2 if train_mse is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    # Plot NLL loss
    axes[0].plot(train_losses, label='Train NLL Loss', linewidth=2)
    axes[0].plot(val_losses, label='Validation NLL Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Negative Log-Likelihood', fontsize=12)
    axes[0].set_title('Training History - NLL Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot MSE if provided
    if train_mse is not None and val_mse is not None:
        axes[1].plot(train_mse, label='Train MSE', linewidth=2)
        axes[1].plot(val_mse, label='Validation MSE', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Mean Squared Error', fontsize=12)
        axes[1].set_title('Training History - MSE', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved: {save_path}")

    plt.close()


def plot_predictions_with_uncertainty(y_true, y_pred_mean, y_pred_variance,
                                     output_names=None, save_path=None,
                                     confidence=0.95, y_pred_aleatoric=None,
                                     y_pred_epistemic=None):
    """
    Plot predictions vs true values with uncertainty bounds.

    If y_pred_aleatoric and y_pred_epistemic are provided (ensemble mode),
    shows stacked bands: inner band for aleatoric, outer for total uncertainty.

    Args:
        y_true (np.ndarray): True values
        y_pred_mean (np.ndarray): Predicted means
        y_pred_variance (np.ndarray): Predicted variances (total)
        output_names (list): Names of outputs
        save_path (str or Path): Path to save the plot
        confidence (float): Confidence level for prediction intervals
        y_pred_aleatoric (np.ndarray): Aleatoric variance (optional, for ensemble)
        y_pred_epistemic (np.ndarray): Epistemic variance (optional, for ensemble)
    """
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred_mean = y_pred_mean.reshape(-1, 1)
        y_pred_variance = y_pred_variance.reshape(-1, 1)
        if y_pred_aleatoric is not None:
            y_pred_aleatoric = y_pred_aleatoric.reshape(-1, 1)
            y_pred_epistemic = y_pred_epistemic.reshape(-1, 1)

    n_outputs = y_true.shape[1]
    use_decomposition = y_pred_aleatoric is not None and y_pred_epistemic is not None

    if output_names is None:
        output_names = [f"Output {i+1}" for i in range(n_outputs)]

    # Calculate number of rows needed
    n_cols = min(3, n_outputs)
    n_rows = (n_outputs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_outputs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    from scipy import stats
    z_score = stats.norm.ppf((1 + confidence) / 2)

    for i, name in enumerate(output_names):
        ax = axes[i]

        y_t = y_true[:, i]
        y_p = y_pred_mean[:, i]
        y_v = y_pred_variance[:, i]
        y_std = np.sqrt(y_v)

        # Sort by predicted mean for better visualization
        sort_idx = np.argsort(y_p)
        y_t_sorted = y_t[sort_idx]
        y_p_sorted = y_p[sort_idx]
        y_std_sorted = y_std[sort_idx]

        x_range = range(len(y_p_sorted))

        if use_decomposition:
            # Ensemble mode: show aleatoric (inner) and epistemic (outer) bands
            y_a = y_pred_aleatoric[:, i]
            y_e = y_pred_epistemic[:, i]
            std_aleatoric = np.sqrt(y_a)
            std_aleatoric_sorted = std_aleatoric[sort_idx]

            # Outer band (total = aleatoric + epistemic) - lighter
            ax.fill_between(x_range,
                            y_p_sorted - z_score * y_std_sorted,
                            y_p_sorted + z_score * y_std_sorted,
                            alpha=0.2, color='orange',
                            label=f'Epistemic ({int(confidence*100)}% CI)')

            # Inner band (aleatoric only) - darker
            ax.fill_between(x_range,
                            y_p_sorted - z_score * std_aleatoric_sorted,
                            y_p_sorted + z_score * std_aleatoric_sorted,
                            alpha=0.4, color='blue',
                            label=f'Aleatoric ({int(confidence*100)}% CI)')
        else:
            # Single model mode: show total uncertainty
            ax.fill_between(x_range,
                            y_p_sorted - z_score * y_std_sorted,
                            y_p_sorted + z_score * y_std_sorted,
                            alpha=0.3, color='blue',
                            label=f'{int(confidence*100)}% Prediction Interval')

        # Plot predictions and true values
        ax.plot(y_p_sorted, 'b-', linewidth=2, label='Predicted Mean', alpha=0.7)
        ax.plot(y_t_sorted, 'ro', markersize=3, label='True Values', alpha=0.6)

        ax.set_xlabel('Sample (sorted by prediction)', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide extra subplots
    for i in range(n_outputs, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions with uncertainty plot saved: {save_path}")

    plt.close()


def plot_scatter_with_uncertainty(y_true, y_pred_mean, y_pred_variance,
                                  output_names=None, save_path=None):
    """
    Create scatter plots of predictions vs true values with uncertainty coloring.

    Points with higher uncertainty are shown in different colors.

    Args:
        y_true (np.ndarray): True values
        y_pred_mean (np.ndarray): Predicted means
        y_pred_variance (np.ndarray): Predicted variances
        output_names (list): Names of outputs
        save_path (str or Path): Path to save the plot
    """
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred_mean = y_pred_mean.reshape(-1, 1)
        y_pred_variance = y_pred_variance.reshape(-1, 1)

    n_outputs = y_true.shape[1]

    if output_names is None:
        output_names = [f"Output {i+1}" for i in range(n_outputs)]

    n_cols = min(3, n_outputs)
    n_rows = (n_outputs + n_cols - 1) // n_cols

    # Compact figure size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5*n_cols, 3.2*n_rows))
    if n_outputs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, name in enumerate(output_names):
        ax = axes[i]

        y_t = y_true[:, i]
        y_p = y_pred_mean[:, i]
        y_v = y_pred_variance[:, i]

        # Create scatter plot with color representing uncertainty
        scatter = ax.scatter(y_t, y_p, c=y_v, cmap='viridis',
                           alpha=0.6, s=20, edgecolors='black', linewidth=0.3)

        # Add perfect prediction line
        min_val = min(y_t.min(), y_p.min())
        max_val = max(y_t.max(), y_p.max())
        ax.plot([min_val, max_val], [min_val, max_val],
               'r--', linewidth=1.5, label='Perfect Prediction', alpha=0.7)

        ax.set_xlabel('True Values', fontsize=9)
        ax.set_ylabel('Predicted Mean', fontsize=9)
        ax.set_title(f'{name}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Predicted Variance', fontsize=8)

        # Calculate and display R²
        from sklearn.metrics import r2_score
        r2 = r2_score(y_t, y_p)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}',
               transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               verticalalignment='top', fontsize=8)

    # Hide extra subplots
    for i in range(n_outputs, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot with uncertainty saved: {save_path}")

    plt.close()


def plot_uncertainty_distribution(y_pred_variance, output_names=None, save_path=None):
    """
    Plot the distribution of predicted uncertainties.

    This helps understand how the model distributes uncertainty across predictions.

    Args:
        y_pred_variance (np.ndarray): Predicted variances
        output_names (list): Names of outputs
        save_path (str or Path): Path to save the plot
    """
    if len(y_pred_variance.shape) == 1:
        y_pred_variance = y_pred_variance.reshape(-1, 1)

    n_outputs = y_pred_variance.shape[1]

    if output_names is None:
        output_names = [f"Output {i+1}" for i in range(n_outputs)]

    n_cols = min(3, n_outputs)
    n_rows = (n_outputs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_outputs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, name in enumerate(output_names):
        ax = axes[i]
        y_v = y_pred_variance[:, i]

        # Plot histogram with KDE
        ax.hist(y_v, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')

        # Add KDE curve
        from scipy import stats
        kde = stats.gaussian_kde(y_v)
        x_range = np.linspace(y_v.min(), y_v.max(), 200)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

        ax.set_xlabel('Predicted Variance', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{name} - Uncertainty Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_var = np.mean(y_v)
        std_var = np.std(y_v)
        ax.axvline(mean_var, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_var:.4f}')
        ax.text(0.6, 0.95, f'Mean: {mean_var:.4f}\nStd: {std_var:.4f}',
               transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               verticalalignment='top', fontsize=9)

    # Hide extra subplots
    for i in range(n_outputs, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Uncertainty distribution plot saved: {save_path}")

    plt.close()


def plot_uncertainty_decomposition(y_pred_aleatoric, y_pred_epistemic, output_names=None, save_path=None):
    """
    Plot the decomposition of uncertainty into aleatoric and epistemic components.

    Creates a visualization showing:
    1. Stacked bar chart of mean aleatoric vs epistemic
    2. Distribution of aleatoric and epistemic across samples

    Args:
        y_pred_aleatoric (np.ndarray): Aleatoric variance per sample
        y_pred_epistemic (np.ndarray): Epistemic variance per sample
        output_names (list): Names of outputs
        save_path (str or Path): Path to save the plot
    """
    if y_pred_aleatoric is None or y_pred_epistemic is None:
        print("Warning: Aleatoric/epistemic data not available for decomposition plot")
        return

    if len(y_pred_aleatoric.shape) == 1:
        y_pred_aleatoric = y_pred_aleatoric.reshape(-1, 1)
        y_pred_epistemic = y_pred_epistemic.reshape(-1, 1)

    n_outputs = y_pred_aleatoric.shape[1]

    if output_names is None:
        output_names = [f"Output {i+1}" for i in range(n_outputs)]

    # Create figure with 2 rows: top for bar chart, bottom for distributions
    fig, axes = plt.subplots(2, n_outputs, figsize=(5*n_outputs, 8))
    if n_outputs == 1:
        axes = axes.reshape(2, 1)

    for i, name in enumerate(output_names):
        aleatoric = y_pred_aleatoric[:, i]
        epistemic = y_pred_epistemic[:, i]
        total = aleatoric + epistemic

        mean_aleatoric = np.mean(aleatoric)
        mean_epistemic = np.mean(epistemic)
        mean_total = mean_aleatoric + mean_epistemic

        # Top: Stacked bar chart showing mean decomposition
        ax1 = axes[0, i]
        bars = ax1.bar(['Uncertainty'], [mean_aleatoric], label='Aleatoric', color='#3498db', alpha=0.8)
        ax1.bar(['Uncertainty'], [mean_epistemic], bottom=[mean_aleatoric], label='Epistemic', color='#e74c3c', alpha=0.8)

        # Add percentage labels
        aleatoric_pct = mean_aleatoric / mean_total * 100 if mean_total > 0 else 0
        epistemic_pct = mean_epistemic / mean_total * 100 if mean_total > 0 else 0

        ax1.text(0, mean_aleatoric/2, f'{aleatoric_pct:.1f}%', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        ax1.text(0, mean_aleatoric + mean_epistemic/2, f'{epistemic_pct:.1f}%', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

        ax1.set_ylabel('Variance', fontsize=10)
        ax1.set_title(f'{name}\nUncertainty Decomposition', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')

        # Add text with values
        ax1.text(0.02, 0.98, f'Aleatoric: {mean_aleatoric:.6f}\nEpistemic: {mean_epistemic:.6f}\nTotal: {mean_total:.6f}',
                transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Bottom: Distribution comparison
        ax2 = axes[1, i]

        # Histogram for both
        bins = 50
        alpha = 0.6
        ax2.hist(aleatoric, bins=bins, alpha=alpha, label='Aleatoric', color='#3498db', density=True)
        ax2.hist(epistemic, bins=bins, alpha=alpha, label='Epistemic', color='#e74c3c', density=True)

        ax2.set_xlabel('Variance', fontsize=10)
        ax2.set_ylabel('Density', fontsize=10)
        ax2.set_title(f'{name}\nDistribution Comparison', fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Add vertical lines for means
        ax2.axvline(mean_aleatoric, color='#2980b9', linestyle='--', linewidth=2, label=f'Mean Aleatoric')
        ax2.axvline(mean_epistemic, color='#c0392b', linestyle='--', linewidth=2, label=f'Mean Epistemic')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Uncertainty decomposition plot saved: {save_path}")

    plt.close()
