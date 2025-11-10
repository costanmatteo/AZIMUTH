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


def calculate_metrics_per_process(
    y_true,
    y_pred_mean,
    process_ids,
    y_pred_variance=None,
    output_names=None,
    process_names=None
):
    """
    Calculate metrics separately for each process.

    This function computes metrics for each manufacturing process independently,
    as well as overall metrics across all processes.

    Args:
        y_true (np.ndarray): True values, shape (n_samples, n_outputs)
        y_pred_mean (np.ndarray): Predicted mean values, shape (n_samples, n_outputs)
        process_ids (np.ndarray): Process IDs for each sample, shape (n_samples,)
        y_pred_variance (np.ndarray, optional): Predicted variance, shape (n_samples, n_outputs)
        output_names (list, optional): Names of output variables
        process_names (dict, optional): Mapping {process_id: process_name}

    Returns:
        dict: Dictionary with structure:
            {
                'process_0': {...metrics...},
                'process_1': {...metrics...},
                'process_2': {...metrics...},
                'process_3': {...metrics...},
                'overall': {...metrics...},
                'process_counts': {0: count, 1: count, ...}
            }
    """
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
    if len(y_pred_mean.shape) == 1:
        y_pred_mean = y_pred_mean.reshape(-1, 1)
    if y_pred_variance is not None and len(y_pred_variance.shape) == 1:
        y_pred_variance = y_pred_variance.reshape(-1, 1)

    # Default process names
    if process_names is None:
        process_names = {
            0: 'laser',
            1: 'plasma',
            2: 'galvanic',
            3: 'microetch'
        }

    results = {}

    # Get unique process IDs
    unique_processes = np.unique(process_ids)

    # Count samples per process
    process_counts = {}
    for pid in unique_processes:
        process_counts[int(pid)] = int(np.sum(process_ids == pid))

    # Calculate metrics for each process
    for pid in unique_processes:
        # Get mask for this process
        mask = (process_ids == pid)

        # Extract data for this process
        y_true_proc = y_true[mask]
        y_pred_mean_proc = y_pred_mean[mask]

        if y_pred_variance is not None:
            y_pred_variance_proc = y_pred_variance[mask]
        else:
            y_pred_variance_proc = None

        # Calculate metrics
        proc_name = process_names.get(int(pid), f'process_{int(pid)}')
        proc_metrics = calculate_metrics(
            y_true_proc,
            y_pred_mean_proc,
            y_pred_variance_proc,
            output_names
        )

        results[f'process_{int(pid)}'] = proc_metrics
        results[f'process_{int(pid)}_name'] = proc_name

    # Calculate overall metrics (across all processes)
    overall_metrics = calculate_metrics(
        y_true,
        y_pred_mean,
        y_pred_variance,
        output_names
    )

    results['overall'] = overall_metrics
    results['process_counts'] = process_counts

    return results


def print_metrics_per_process(metrics_per_process):
    """
    Print multi-process metrics in a formatted way.

    Args:
        metrics_per_process (dict): Dictionary returned by calculate_metrics_per_process
    """
    print("\n" + "="*80)
    print("MULTI-PROCESS EVALUATION METRICS")
    print("="*80)

    # Print sample counts
    if 'process_counts' in metrics_per_process:
        print("\nSample counts per process:")
        print("-" * 50)
        for pid, count in sorted(metrics_per_process['process_counts'].items()):
            proc_name = metrics_per_process.get(f'process_{pid}_name', f'process_{pid}')
            print(f"  Process {pid} ({proc_name:12s}): {count:5d} samples")

    # Print metrics for each process
    for key in sorted(metrics_per_process.keys()):
        if key.startswith('process_') and not key.endswith('_name'):
            pid = int(key.split('_')[1])
            proc_name = metrics_per_process.get(f'{key}_name', key)

            print(f"\n{'='*80}")
            print(f"Process {pid}: {proc_name.upper()}")
            print(f"{'='*80}")

            proc_metrics = metrics_per_process[key]
            for output_name, values in proc_metrics.items():
                print(f"\n{output_name}:")
                print("-" * 50)
                for metric_name, value in values.items():
                    if isinstance(value, (int, float)):
                        if metric_name in ['R2', 'Calibration_Ratio']:
                            print(f"  {metric_name:20s}: {value:12.4f}")
                        else:
                            print(f"  {metric_name:20s}: {value:12.6f}")
                    else:
                        print(f"  {metric_name:20s}: {value}")

    # Print overall metrics
    if 'overall' in metrics_per_process:
        print(f"\n{'='*80}")
        print("OVERALL (ALL PROCESSES COMBINED)")
        print(f"{'='*80}")

        overall_metrics = metrics_per_process['overall']
        for output_name, values in overall_metrics.items():
            print(f"\n{output_name}:")
            print("-" * 50)
            for metric_name, value in values.items():
                if isinstance(value, (int, float)):
                    if metric_name in ['R2', 'Calibration_Ratio']:
                        print(f"  {metric_name:20s}: {value:12.4f}")
                    else:
                        print(f"  {metric_name:20s}: {value:12.6f}")
                else:
                    print(f"  {metric_name:20s}: {value}")

    print(f"\n{'='*80}\n")
