"""
Metrics for evaluating predictions
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(y_true, y_pred, output_names=None):
    """
    Calculate metrics to evaluate predictions.

    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        output_names (list, optional): Names of outputs

    Returns:
        dict: Dictionary with metrics
    """
    n_outputs = y_true.shape[1] if len(y_true.shape) > 1 else 1

    if n_outputs == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    metrics = {}

    for i in range(n_outputs):
        name = output_names[i] if output_names else f'output_{i+1}'

        # Calculate metrics
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])

        # MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        mask = y_true[:, i] != 0
        if mask.any():
            mape = np.mean(np.abs((y_true[:, i][mask] - y_pred[:, i][mask]) / y_true[:, i][mask])) * 100
        else:
            mape = float('inf')

        metrics[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }

    return metrics


def print_metrics(metrics):
    """
    Print metrics in a readable format.

    Args:
        metrics (dict): Dictionary of metrics from print_metrics
    """
    print("\n" + "="*70)
    print("EVALUATION METRICS")
    print("="*70)

    for output_name, metric_dict in metrics.items():
        print(f"\n{output_name}:")
        print(f"  MSE  (Mean Squared Error):          {metric_dict['MSE']:.6f}")
        print(f"  RMSE (Root Mean Squared Error):     {metric_dict['RMSE']:.6f}")
        print(f"  MAE  (Mean Absolute Error):         {metric_dict['MAE']:.6f}")
        print(f"  R²   (Coefficient of Determination): {metric_dict['R2']:.6f}")
        if metric_dict['MAPE'] != float('inf'):
            print(f"  MAPE (Mean Absolute % Error):       {metric_dict['MAPE']:.2f}%")

    print("\n" + "="*70 + "\n")


def calculate_overall_metrics(y_true, y_pred):
    """
    Calculate global metrics across all outputs.

    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values

    Returns:
        dict: Global metrics
    """
    return {
        'overall_MSE': mean_squared_error(y_true, y_pred),
        'overall_RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'overall_MAE': mean_absolute_error(y_true, y_pred),
        'overall_R2': r2_score(y_true, y_pred)
    }
