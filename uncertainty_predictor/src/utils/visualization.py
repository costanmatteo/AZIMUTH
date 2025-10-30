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
                                     confidence=0.95,
                                     y_true_train=None, y_pred_mean_train=None,
                                     y_pred_variance_train=None):
    """
    Plot predictions vs true values with uncertainty bounds.

    Args:
        y_true (np.ndarray): True values (test set)
        y_pred_mean (np.ndarray): Predicted means (test set)
        y_pred_variance (np.ndarray): Predicted variances (test set)
        output_names (list): Names of outputs
        save_path (str or Path): Path to save the plot
        confidence (float): Confidence level for prediction intervals
        y_true_train (np.ndarray): True values (training set, optional)
        y_pred_mean_train (np.ndarray): Predicted means (training set, optional)
        y_pred_variance_train (np.ndarray): Predicted variances (training set, optional)
    """
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred_mean = y_pred_mean.reshape(-1, 1)
        y_pred_variance = y_pred_variance.reshape(-1, 1)

    # Handle training data reshaping if provided
    has_train_data = y_true_train is not None and y_pred_mean_train is not None and y_pred_variance_train is not None
    if has_train_data:
        if len(y_true_train.shape) == 1:
            y_true_train = y_true_train.reshape(-1, 1)
            y_pred_mean_train = y_pred_mean_train.reshape(-1, 1)
            y_pred_variance_train = y_pred_variance_train.reshape(-1, 1)

    n_outputs = y_true.shape[1]

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

        # Test set data
        y_t = y_true[:, i]
        y_p = y_pred_mean[:, i]
        y_v = y_pred_variance[:, i]
        y_std = np.sqrt(y_v)

        # Sort by predicted mean for better visualization
        sort_idx = np.argsort(y_p)
        y_t_sorted = y_t[sort_idx]
        y_p_sorted = y_p[sort_idx]
        y_std_sorted = y_std[sort_idx]

        # Plot test set uncertainty bounds
        ax.fill_between(range(len(y_p_sorted)),
                        y_p_sorted - z_score * y_std_sorted,
                        y_p_sorted + z_score * y_std_sorted,
                        alpha=0.3, color='blue',
                        label=f'Test {int(confidence*100)}% PI')

        # Plot test set predictions and true values
        ax.plot(y_p_sorted, 'b-', linewidth=2, label='Test Predicted Mean', alpha=0.7)
        ax.plot(y_t_sorted, 'ro', markersize=3, label='Test True Values', alpha=0.6)

        # Add training data if provided
        if has_train_data:
            y_t_train = y_true_train[:, i]
            y_p_train = y_pred_mean_train[:, i]
            y_v_train = y_pred_variance_train[:, i]
            y_std_train = np.sqrt(y_v_train)

            # Sort by predicted mean
            sort_idx_train = np.argsort(y_p_train)
            y_t_train_sorted = y_t_train[sort_idx_train]
            y_p_train_sorted = y_p_train[sort_idx_train]
            y_std_train_sorted = y_std_train[sort_idx_train]

            # Offset training data points to appear after test data
            offset = len(y_p_sorted)
            x_train = range(offset, offset + len(y_p_train_sorted))

            # Plot training set uncertainty bounds
            ax.fill_between(x_train,
                           y_p_train_sorted - z_score * y_std_train_sorted,
                           y_p_train_sorted + z_score * y_std_train_sorted,
                           alpha=0.2, color='green',
                           label=f'Train {int(confidence*100)}% PI')

            # Plot training set predictions and true values
            ax.plot(x_train, y_p_train_sorted, 'g-', linewidth=2,
                   label='Train Predicted Mean', alpha=0.7)
            ax.plot(x_train, y_t_train_sorted, 'go', markersize=2,
                   label='Train True Values', alpha=0.4)

        ax.set_xlabel('Sample (sorted by prediction)', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=7, loc='best')
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

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
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
                           alpha=0.6, s=30, edgecolors='black', linewidth=0.5)

        # Add perfect prediction line
        min_val = min(y_t.min(), y_p.min())
        max_val = max(y_t.max(), y_p.max())
        ax.plot([min_val, max_val], [min_val, max_val],
               'r--', linewidth=2, label='Perfect Prediction', alpha=0.7)

        ax.set_xlabel('True Values', fontsize=10)
        ax.set_ylabel('Predicted Mean', fontsize=10)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Predicted Variance', fontsize=9)

        # Calculate and display R²
        from sklearn.metrics import r2_score
        r2 = r2_score(y_t, y_p)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}',
               transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               verticalalignment='top', fontsize=9)

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
