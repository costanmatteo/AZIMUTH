"""
Metrics calculation for uncertainty predictions

This module provides functions to calculate metrics for models that predict
both mean values and uncertainty (variance).
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(y_true, y_pred_mean, y_pred_variance=None, output_names=None):
    """
    Calculate comprehensive metrics for uncertainty predictions.

    Args:
        y_true (np.ndarray): True values, shape (n_samples, n_outputs)
        y_pred_mean (np.ndarray): Predicted mean values, shape (n_samples, n_outputs)
        y_pred_variance (np.ndarray): Predicted variance, shape (n_samples, n_outputs)
        output_names (list): Names of output variables

    Returns:
        dict: Dictionary with metrics for each output and overall
    """
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
    if len(y_pred_mean.shape) == 1:
        y_pred_mean = y_pred_mean.reshape(-1, 1)

    n_outputs = y_true.shape[1]

    if output_names is None:
        output_names = [f"Output_{i+1}" for i in range(n_outputs)]

    metrics = {}

    # Calculate metrics for each output
    for i, name in enumerate(output_names):
        y_t = y_true[:, i]
        y_p = y_pred_mean[:, i]

        # DEBUG: Check for NaN before computing metrics
        print(f"\n--- DEBUG: Metrics for output '{name}' (index {i}) ---")
        print(f"y_true[:, {i}] shape: {y_t.shape}")
        print(f"y_true[:, {i}] contains NaN: {np.isnan(y_t).any()}")
        print(f"y_true[:, {i}] NaN count: {np.isnan(y_t).sum()}")
        print(f"y_true[:, {i}] min: {np.nanmin(y_t) if not np.all(np.isnan(y_t)) else 'all NaN'}")
        print(f"y_true[:, {i}] max: {np.nanmax(y_t) if not np.all(np.isnan(y_t)) else 'all NaN'}")

        print(f"\ny_pred_mean[:, {i}] shape: {y_p.shape}")
        print(f"y_pred_mean[:, {i}] contains NaN: {np.isnan(y_p).any()}")
        print(f"y_pred_mean[:, {i}] NaN count: {np.isnan(y_p).sum()}")
        print(f"y_pred_mean[:, {i}] min: {np.nanmin(y_p) if not np.all(np.isnan(y_p)) else 'all NaN'}")
        print(f"y_pred_mean[:, {i}] max: {np.nanmax(y_p) if not np.all(np.isnan(y_p)) else 'all NaN'}")

        if np.isnan(y_p).any():
            nan_indices = np.where(np.isnan(y_p))[0]
            print(f"\nNaN found at indices: {nan_indices[:10]}...")  # Show first 10
            print(f"Total NaN count: {len(nan_indices)}")

        output_metrics = {
            'MSE': mean_squared_error(y_t, y_p),
            'RMSE': np.sqrt(mean_squared_error(y_t, y_p)),
            'MAE': mean_absolute_error(y_t, y_p),
            'R2': r2_score(y_t, y_p),
            'MAPE': np.mean(np.abs((y_t - y_p) / (y_t + 1e-8))) * 100
        }

        # If variance is provided, calculate uncertainty metrics
        if y_pred_variance is not None:
            y_v = y_pred_variance[:, i]
            squared_errors = (y_t - y_p) ** 2

            output_metrics.update({
                'Mean_Variance': np.mean(y_v),
                'Std_Variance': np.std(y_v),
                'Calibration_Ratio': np.mean(squared_errors) / np.mean(y_v),
                'NLL': np.mean(0.5 * (np.log(y_v + 1e-8) + squared_errors / (y_v + 1e-8)))
            })

            # Uncertainty quality assessment
            calibration_ratio = output_metrics['Calibration_Ratio']
            if 0.8 <= calibration_ratio <= 1.2:
                output_metrics['Calibration_Status'] = 'Well calibrated'
            elif calibration_ratio < 0.8:
                output_metrics['Calibration_Status'] = 'Under-confident'
            else:
                output_metrics['Calibration_Status'] = 'Over-confident'

        metrics[name] = output_metrics

    # Calculate overall metrics (average across outputs)
    overall = {
        'MSE': np.mean([metrics[name]['MSE'] for name in output_names]),
        'RMSE': np.mean([metrics[name]['RMSE'] for name in output_names]),
        'MAE': np.mean([metrics[name]['MAE'] for name in output_names]),
        'R2': np.mean([metrics[name]['R2'] for name in output_names]),
        'MAPE': np.mean([metrics[name]['MAPE'] for name in output_names])
    }

    if y_pred_variance is not None:
        overall.update({
            'Mean_Variance': np.mean([metrics[name]['Mean_Variance'] for name in output_names]),
            'Calibration_Ratio': np.mean([metrics[name]['Calibration_Ratio'] for name in output_names]),
            'NLL': np.mean([metrics[name]['NLL'] for name in output_names])
        })

    metrics['Overall'] = overall

    return metrics


def print_metrics(metrics):
    """
    Print metrics in a formatted way.

    Args:
        metrics (dict): Dictionary returned by calculate_metrics
    """
    print("\n" + "="*70)
    print("EVALUATION METRICS")

    for name, values in metrics.items():
        print(f"\n{name}:")
        print("-" * 50)
        for metric_name, value in values.items():
            if isinstance(value, (int, float)):
                if metric_name in ['R2', 'Calibration_Ratio']:
                    print(f"  {metric_name:20s}: {value:12.4f}")
                else:
                    print(f"  {metric_name:20s}: {value:12.6f}")
            else:
                print(f"  {metric_name:20s}: {value}")



def compute_prediction_intervals(y_pred_mean, y_pred_variance, confidence=0.95):
    """
    Compute prediction intervals based on predicted mean and variance.

    Args:
        y_pred_mean (np.ndarray): Predicted means
        y_pred_variance (np.ndarray): Predicted variances
        confidence (float): Confidence level (default: 0.95 for 95% interval)

    Returns:
        tuple: (lower_bound, upper_bound)
    """
    from scipy import stats

    # Get z-score for confidence level
    z_score = stats.norm.ppf((1 + confidence) / 2)

    # Compute standard deviation
    std = np.sqrt(y_pred_variance)

    # Compute bounds
    lower = y_pred_mean - z_score * std
    upper = y_pred_mean + z_score * std

    return lower, upper


def evaluate_prediction_intervals(y_true, y_pred_mean, y_pred_variance, confidence=0.95):
    """
    Evaluate if prediction intervals have correct coverage.

    For a well-calibrated model, approximately 'confidence'% of true values
    should fall within the prediction intervals.

    Args:
        y_true (np.ndarray): True values
        y_pred_mean (np.ndarray): Predicted means
        y_pred_variance (np.ndarray): Predicted variances
        confidence (float): Confidence level

    Returns:
        dict: Coverage statistics
    """
    lower, upper = compute_prediction_intervals(y_pred_mean, y_pred_variance, confidence)

    # Check which predictions fall within the interval
    within_interval = (y_true >= lower) & (y_true <= upper)
    coverage = np.mean(within_interval) * 100

    return {
        'expected_coverage': confidence * 100,
        'actual_coverage': coverage,
        'coverage_error': coverage - (confidence * 100),
        'well_calibrated': abs(coverage - confidence * 100) < 5  # within 5% is considered good
    }
