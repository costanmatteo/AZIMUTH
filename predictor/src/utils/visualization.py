"""
Utilities for results visualization
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(train_losses, val_losses, save_path=None):
    """
    Plot the training history.

    Args:
        train_losses (list): List of training losses
        val_losses (list): List of validation losses
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()


def plot_predictions(y_true, y_pred, output_names=None, save_path=None):
    """
    Plot predictions vs actual values.

    Args:
        y_true (np.ndarray): Actual values
        y_pred (np.ndarray): Predicted values
        output_names (list, optional): Output names
        save_path (str, optional): Path to save the plot
    """
    n_outputs = y_true.shape[1] if len(y_true.shape) > 1 else 1

    if n_outputs == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    # Determine subplot layout
    n_cols = min(3, n_outputs)
    n_rows = (n_outputs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_outputs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(n_outputs):
        ax = axes[i]

        # Scatter plot
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=20)

        # Perfect prediction line
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

        # Labels
        title = output_names[i] if output_names else f'Output {i+1}'
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Actual Value', fontsize=10)
        ax.set_ylabel('Predicted Value', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Remove empty subplots
    for i in range(n_outputs, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()


def plot_error_distribution(y_true, y_pred, output_names=None, save_path=None):
    """
    Plot error distribution.

    Args:
        y_true (np.ndarray): Actual values
        y_pred (np.ndarray): Predicted values
        output_names (list, optional): Output names
        save_path (str, optional): Path to save the plot
    """
    errors = y_pred - y_true
    n_outputs = errors.shape[1] if len(errors.shape) > 1 else 1

    if n_outputs == 1:
        errors = errors.reshape(-1, 1)

    n_cols = min(3, n_outputs)
    n_rows = (n_outputs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_outputs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(n_outputs):
        ax = axes[i]

        # Error histogram
        ax.hist(errors[:, i], bins=50, alpha=0.7, edgecolor='black')

        # Vertical line at zero
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2)

        # Labels
        title = output_names[i] if output_names else f'Output {i+1}'
        ax.set_title(f'Errors - {title}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Error (Predicted - Actual)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    # Remove empty subplots
    for i in range(n_outputs, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()
