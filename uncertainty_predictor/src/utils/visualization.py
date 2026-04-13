"""
Visualization utilities for uncertainty predictions

This module provides functions to visualize predictions with uncertainty bounds.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path


def apply_plot_style():
    plt.rcParams.update({
        'font.family':           'sans-serif',
        'font.sans-serif':       ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size':             11,
        'axes.titlesize':        13,
        'axes.titleweight':      'normal',
        'axes.titlelocation':    'left',
        'axes.labelsize':        12,
        'axes.labelweight':      'normal',
        'axes.linewidth':        0.5,
        'axes.spines.top':       False,
        'axes.spines.right':     False,
        'axes.grid':             True,
        'grid.color':            '#DDDDDD',
        'grid.linewidth':        0.4,
        'grid.alpha':            1.0,
        'xtick.labelsize':       11,
        'ytick.labelsize':       11,
        'xtick.major.width':     0.4,
        'ytick.major.width':     0.4,
        'xtick.major.size':      3,
        'ytick.major.size':      3,
        'legend.fontsize':       10,
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


def _normalize_output_names(output_names, n_outputs, context=""):
    """
    Return a list of exactly ``n_outputs`` labels.

    If ``output_names`` is None, generic labels are generated.
    If its length does not match ``n_outputs``, the list is truncated or
    padded and a warning is printed so the inconsistency is visible — this
    typically indicates a stale on-disk dataset that no longer matches the
    process config (e.g. st_params.p was changed without regenerating the
    per-process dataset file).
    """
    if output_names is None:
        return [f"Output {i + 1}" for i in range(n_outputs)]

    output_names = list(output_names)
    if len(output_names) == n_outputs:
        return output_names

    where = f" ({context})" if context else ""
    print(
        f"  [viz warning{where}] output_names length ({len(output_names)}) "
        f"does not match y_true.shape[1] ({n_outputs}). "
        f"This usually means the on-disk dataset was generated with a "
        f"different output_dim than the current process config. "
        f"Consider regenerating the dataset with generate_dataset.py."
    )

    if len(output_names) > n_outputs:
        return output_names[:n_outputs]
    return output_names + [
        f"Output {i + 1}" for i in range(len(output_names), n_outputs)
    ]


def plot_training_history(train_losses, val_losses, train_mse=None, val_mse=None,
                          save_path=None, swa_start_epoch=None):
    """
    Plot training history for uncertainty model.

    Args:
        train_losses (list): Training NLL losses
        val_losses (list): Validation NLL losses
        train_mse (list): Training MSE (optional)
        val_mse (list): Validation MSE (optional)
        save_path (str or Path): Path to save the plot
        swa_start_epoch (int): Epoch when SWA collection started (optional)
    """
    apply_plot_style()
    n_plots = 2 if train_mse is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 3.5))

    if n_plots == 1:
        axes = [axes]
    else:
        axes = list(axes)

    for _ax in axes:
        _ax.spines['top'].set_visible(False)
        _ax.spines['right'].set_visible(False)

    # Plot NLL loss
    axes[0].plot(train_losses, label='Train NLL Loss', linewidth=2)
    axes[0].plot(val_losses, label='Validation NLL Loss', linewidth=2)

    # Add SWA start marker
    if swa_start_epoch is not None and swa_start_epoch < len(train_losses):
        axes[0].axvline(x=swa_start_epoch, color='green', linestyle='--', linewidth=1.0,
                        label=f'SWA Start (epoch {swa_start_epoch})')

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Negative Log-Likelihood')
    axes[0].set_title('Training History - NLL Loss')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)

    # Plot MSE if provided
    if train_mse is not None and val_mse is not None:
        axes[1].plot(train_mse, label='Train MSE', linewidth=2)
        axes[1].plot(val_mse, label='Validation MSE', linewidth=2)

        # Add SWA start marker to MSE plot too
        if swa_start_epoch is not None and swa_start_epoch < len(train_mse):
            axes[1].axvline(x=swa_start_epoch, color='green', linestyle='--', linewidth=1.0,
                            label=f'SWA Start (epoch {swa_start_epoch})')

        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Mean Squared Error')
        axes[1].set_title('Training History - MSE')
        axes[1].legend(loc='upper left')
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
    apply_plot_style()
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred_mean = y_pred_mean.reshape(-1, 1)
        y_pred_variance = y_pred_variance.reshape(-1, 1)
        if y_pred_aleatoric is not None:
            y_pred_aleatoric = y_pred_aleatoric.reshape(-1, 1)
            y_pred_epistemic = y_pred_epistemic.reshape(-1, 1)

    n_outputs = y_true.shape[1]
    use_decomposition = y_pred_aleatoric is not None and y_pred_epistemic is not None

    output_names = _normalize_output_names(
        output_names, n_outputs, context="plot_predictions_with_uncertainty"
    )

    # Calculate number of rows needed
    n_cols = min(3, n_outputs)
    n_rows = (n_outputs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_outputs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for _ax in axes:
        _ax.spines['top'].set_visible(False)
        _ax.spines['right'].set_visible(False)

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

        ax.set_xlabel('Sample (sorted by prediction)')
        ax.set_ylabel('Value')
        ax.set_title(f'{name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide extra subplots
    for i in range(n_outputs, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions with uncertainty plot saved: {save_path}")

    plt.close()


def plot_combined_predictions_with_uncertainty(
        y_true_val, y_pred_mean_val, y_pred_variance_val,
        y_true_train, y_pred_mean_train, y_pred_variance_train,
        output_names=None, save_path=None, confidence=0.95,
        y_pred_aleatoric_val=None, y_pred_epistemic_val=None,
        y_pred_aleatoric_train=None, y_pred_epistemic_train=None):
    """
    Plot validation and training predictions with uncertainty in a single
    combined figure (left column = validation, right column = training).
    """
    apply_plot_style()

    # Ensure 2-D
    def _to2d(arr):
        return arr.reshape(-1, 1) if arr is not None and len(arr.shape) == 1 else arr

    y_true_val        = _to2d(y_true_val)
    y_pred_mean_val   = _to2d(y_pred_mean_val)
    y_pred_variance_val = _to2d(y_pred_variance_val)
    y_pred_aleatoric_val  = _to2d(y_pred_aleatoric_val)
    y_pred_epistemic_val  = _to2d(y_pred_epistemic_val)
    y_true_train      = _to2d(y_true_train)
    y_pred_mean_train = _to2d(y_pred_mean_train)
    y_pred_variance_train = _to2d(y_pred_variance_train)
    y_pred_aleatoric_train = _to2d(y_pred_aleatoric_train)
    y_pred_epistemic_train = _to2d(y_pred_epistemic_train)

    n_outputs = y_true_val.shape[1]
    output_names = _normalize_output_names(
        output_names, n_outputs, context="plot_combined_predictions_with_uncertainty"
    )

    from scipy import stats
    z_score = stats.norm.ppf((1 + confidence) / 2)

    fig, axes = plt.subplots(n_outputs, 2, figsize=(14, 3.5 * n_outputs),
                             squeeze=False)

    for col_idx, (y_t, y_p, y_v, y_a, y_e, col_title) in enumerate([
        (y_true_train, y_pred_mean_train, y_pred_variance_train,
         y_pred_aleatoric_train, y_pred_epistemic_train, "Training"),
        (y_true_val, y_pred_mean_val, y_pred_variance_val,
         y_pred_aleatoric_val, y_pred_epistemic_val, "Validation"),
    ]):
        use_decomp = y_a is not None and y_e is not None
        for i, name in enumerate(output_names):
            ax = axes[i, col_idx]
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            yt = y_t[:, i]
            yp = y_p[:, i]
            yv = y_v[:, i]
            ystd = np.sqrt(yv)

            sort_idx = np.argsort(yp)
            yt_s = yt[sort_idx]
            yp_s = yp[sort_idx]
            ystd_s = ystd[sort_idx]
            x_range = range(len(yp_s))

            if use_decomp:
                std_a_s = np.sqrt(y_a[:, i])[sort_idx]
                ax.fill_between(x_range,
                                yp_s - z_score * ystd_s,
                                yp_s + z_score * ystd_s,
                                alpha=0.2, color='orange',
                                label=f'Epistemic ({int(confidence*100)}% CI)')
                ax.fill_between(x_range,
                                yp_s - z_score * std_a_s,
                                yp_s + z_score * std_a_s,
                                alpha=0.4, color='blue',
                                label=f'Aleatoric ({int(confidence*100)}% CI)')
            else:
                ax.fill_between(x_range,
                                yp_s - z_score * ystd_s,
                                yp_s + z_score * ystd_s,
                                alpha=0.3, color='blue',
                                label=f'{int(confidence*100)}% Prediction Interval')

            ax.plot(yp_s, 'b-', linewidth=2, label='Predicted Mean', alpha=0.7)
            ax.plot(yt_s, 'ro', markersize=3, label='True Values', alpha=0.6)

            ax.set_xlabel('Sample (sorted by prediction)')
            ax.set_ylabel('Value')
            ax.set_title(f'{col_title} — {name}')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined predictions with uncertainty plot saved: {save_path}")

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
    apply_plot_style()
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred_mean = y_pred_mean.reshape(-1, 1)
        y_pred_variance = y_pred_variance.reshape(-1, 1)

    n_outputs = y_true.shape[1]

    output_names = _normalize_output_names(
        output_names, n_outputs, context="plot_scatter_with_uncertainty"
    )

    n_cols = min(3, n_outputs)
    n_rows = (n_outputs + n_cols - 1) // n_cols

    # Compact figure size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5*n_cols, 3.2*n_rows))
    if n_outputs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for _ax in axes:
        _ax.spines['top'].set_visible(False)
        _ax.spines['right'].set_visible(False)

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

        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Mean')
        ax.set_title(f'{name}')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Predicted Variance')

        # Calculate and display R²
        from sklearn.metrics import r2_score
        r2 = r2_score(y_t, y_p)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}',
               transform=ax.transAxes,
               bbox=dict(boxstyle='square,pad=0.3', facecolor='white',
                         edgecolor='#CCCCCC', linewidth=0.5, alpha=0.9),
               verticalalignment='top', fontsize=11)

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

    output_names = _normalize_output_names(
        output_names, n_outputs, context="plot_uncertainty_distribution"
    )

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

        ax.set_xlabel('Predicted Variance')
        ax.set_ylabel('Density')
        ax.set_title(f'{name} - Uncertainty Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_var = np.mean(y_v)
        std_var = np.std(y_v)
        ax.axvline(mean_var, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_var:.4f}')
        ax.text(0.6, 0.95, f'Mean: {mean_var:.4f}\nStd: {std_var:.4f}',
               transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               verticalalignment='top', fontsize=11)

    # Hide extra subplots
    for i in range(n_outputs, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Uncertainty distribution plot saved: {save_path}")

    plt.close()
