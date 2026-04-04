"""
Metriche per valutazione controller optimization.
"""

import numpy as np
import torch


def convert_trajectory_to_numpy(trajectory):
    """
    Convert trajectory tensors to numpy arrays.

    Handles two formats:
    - Target/baseline: {'inputs': array, 'outputs': array}
    - Actual: {'inputs': tensor, 'outputs_mean': tensor, 'outputs_var': tensor}

    Args:
        trajectory (dict): Trajectory with torch tensors or numpy arrays

    Returns:
        dict: Trajectory with numpy arrays in consistent format
              (always 'outputs_mean' and 'outputs_var')
    """
    numpy_traj = {}
    for process_name, data in trajectory.items():
        # Convert inputs
        inputs = data['inputs']
        if torch.is_tensor(inputs):
            inputs = inputs.detach().cpu().numpy()

        # Convert outputs (handle both formats)
        if 'outputs_mean' in data:
            # Format: outputs_mean, outputs_var
            outputs_mean = data['outputs_mean']
            if torch.is_tensor(outputs_mean):
                outputs_mean = outputs_mean.detach().cpu().numpy()

            outputs_var = data.get('outputs_var', None)
            if outputs_var is not None:
                if torch.is_tensor(outputs_var):
                    outputs_var = outputs_var.detach().cpu().numpy()
            else:
                outputs_var = np.zeros_like(outputs_mean)
        else:
            # Format: outputs only (target/baseline)
            outputs_mean = data['outputs']
            if torch.is_tensor(outputs_mean):
                outputs_mean = outputs_mean.detach().cpu().numpy()
            outputs_var = np.zeros_like(outputs_mean)

        numpy_traj[process_name] = {
            'inputs': inputs,
            'outputs_mean': outputs_mean,
            'outputs_var': outputs_var
        }
    return numpy_traj


def compute_trajectory_distance(trajectory1, trajectory2):
    """
    Calcola distanza tra due trajectories.

    Args:
        trajectory1 (dict): First trajectory
        trajectory2 (dict): Second trajectory

    Returns:
        dict: {
            'input_distance': float,
            'output_distance': float,
            'total_distance': float,
        }
    """
    # Convert to numpy if needed
    traj1 = convert_trajectory_to_numpy(trajectory1)
    traj2 = convert_trajectory_to_numpy(trajectory2)

    input_distances = []
    output_distances = []

    for process_name in traj1.keys():
        # Input distance (MSE)
        inputs1 = traj1[process_name]['inputs']
        inputs2 = traj2[process_name]['inputs']
        input_dist = np.mean((inputs1 - inputs2) ** 2)
        input_distances.append(input_dist)

        # Output distance (MSE)
        outputs1 = traj1[process_name]['outputs_mean']
        outputs2 = traj2[process_name]['outputs_mean']
        output_dist = np.mean((outputs1 - outputs2) ** 2)
        output_distances.append(output_dist)

    return {
        'input_distance': float(np.mean(input_distances)),
        'output_distance': float(np.mean(output_distances)),
        'total_distance': float(np.mean(input_distances + output_distances)),
    }


def compute_process_wise_metrics(trajectory, target_trajectory):
    """
    Calcola metriche per ogni processo individualmente.

    Args:
        trajectory (dict): Actual trajectory
        target_trajectory (dict): Target trajectory

    Returns:
        dict: {
            'laser': {'input_mse': ..., 'output_mse': ..., ...},
            'plasma': {...},
            ...
        }
    """
    # Convert to numpy
    actual = convert_trajectory_to_numpy(trajectory)
    target = convert_trajectory_to_numpy(target_trajectory)

    process_metrics = {}

    for process_name in actual.keys():
        # Input metrics
        inputs_actual = actual[process_name]['inputs']
        inputs_target = target[process_name]['inputs']

        input_mse = np.mean((inputs_actual - inputs_target) ** 2)
        input_mae = np.mean(np.abs(inputs_actual - inputs_target))
        input_max_error = np.max(np.abs(inputs_actual - inputs_target))

        # Output metrics
        outputs_actual = actual[process_name]['outputs_mean']
        outputs_target = target[process_name]['outputs_mean']

        output_mse = np.mean((outputs_actual - outputs_target) ** 2)
        output_mae = np.mean(np.abs(outputs_actual - outputs_target))
        output_max_error = np.max(np.abs(outputs_actual - outputs_target))

        # Combined metrics
        combined_mse = (input_mse + output_mse) / 2

        process_metrics[process_name] = {
            'input_mse': float(input_mse),
            'input_mae': float(input_mae),
            'input_max_error': float(input_max_error),
            'output_mse': float(output_mse),
            'output_mae': float(output_mae),
            'output_max_error': float(output_max_error),
            'combined_mse': float(combined_mse),
        }

    return process_metrics


def create_metrics_summary(F_star, F_baseline, F_actual, trajectory_metrics):
    """
    Crea summary completo di tutte le metriche.

    Args:
        F_star (float): Target reliability
        F_baseline (float): Baseline reliability
        F_actual (float): Actual reliability with controller
        trajectory_metrics (dict): Process-wise metrics

    Returns:
        dict: Summary completo per report
    """
    # Compute improvements
    baseline_improvement = ((F_actual - F_baseline) / F_baseline) if F_baseline != 0 else 0
    target_gap = abs((F_star - F_actual) / F_star) if F_star != 0 else 0

    summary = {
        'reliability': {
            'F_star': float(F_star),
            'F_baseline': float(F_baseline),
            'F_actual': float(F_actual),
            'baseline_improvement': float(baseline_improvement),
            'baseline_improvement_pct': float(baseline_improvement * 100),
            'target_gap': float(target_gap),
            'target_gap_pct': float(target_gap * 100),
        },
        'process_metrics': trajectory_metrics,
    }

    return summary


def compute_worst_case_gap(F_star_per_scenario, F_actual_per_scenario):
    """
    Compute worst-case gap between target and actual reliability.

    Args:
        F_star_per_scenario (array-like): Target reliability for each scenario
        F_actual_per_scenario (array-like): Actual reliability for each scenario

    Returns:
        dict: {
            'worst_case_gap': Maximum gap (F_star - F_actual),
            'worst_case_scenario_idx': Index of worst scenario,
            'worst_case_F_star': F_star of worst scenario,
            'worst_case_F_actual': F_actual of worst scenario,
        }
    """
    F_star_arr = np.atleast_1d(F_star_per_scenario)
    F_actual_arr = np.atleast_1d(F_actual_per_scenario)

    gaps = F_star_arr - F_actual_arr
    worst_idx = int(np.argmax(gaps))
    worst_gap = float(gaps[worst_idx])

    return {
        'worst_case_gap': worst_gap,
        'worst_case_scenario_idx': worst_idx,
        'worst_case_F_star': float(F_star_arr[worst_idx]),
        'worst_case_F_actual': float(F_actual_arr[worst_idx]),
    }


def compute_gap_closure(F_star_per_scenario, F_baseline_per_scenario, F_actual_per_scenario):
    """
    Compute gap closure: (F_actual - F_baseline) / (F_star - F_baseline) per scenario.

    Measures what fraction of the recoverable gap the controller closes.
    Values: 0 = same as baseline, 1 = reached target, <0 = worse than baseline, >1 = better than target.

    Scenarios where F_star ≈ F_baseline (gap < 1e-6) are excluded as the gap is not meaningful.

    Args:
        F_star_per_scenario (array-like): Target reliability for each scenario
        F_baseline_per_scenario (array-like): Baseline reliability for each scenario
        F_actual_per_scenario (array-like): Actual reliability for each scenario

    Returns:
        dict: {
            'gap_closure_per_scenario': list of gap closure values (NaN for invalid scenarios),
            'gap_closure_mean': float (mean over valid scenarios),
            'gap_closure_std': float,
            'gap_closure_min': float (worst-case gap closure),
            'gap_closure_min_scenario_idx': int,
            'gap_closure_max': float,
            'n_valid': int (scenarios where F_star != F_baseline),
            'n_total': int,
        }
    """
    F_star_arr = np.atleast_1d(F_star_per_scenario).astype(float)
    F_baseline_arr = np.atleast_1d(F_baseline_per_scenario).astype(float)
    F_actual_arr = np.atleast_1d(F_actual_per_scenario).astype(float)

    recoverable_gap = F_star_arr - F_baseline_arr
    valid_mask = np.abs(recoverable_gap) > 1e-6

    gap_closure = np.full_like(F_star_arr, np.nan)
    gap_closure[valid_mask] = (F_actual_arr[valid_mask] - F_baseline_arr[valid_mask]) / recoverable_gap[valid_mask]

    valid_values = gap_closure[valid_mask]
    n_valid = int(np.sum(valid_mask))

    if n_valid > 0:
        gc_mean = float(np.mean(valid_values))
        gc_std = float(np.std(valid_values))
        gc_min = float(np.min(valid_values))
        gc_min_idx = int(np.nanargmin(gap_closure))
        gc_max = float(np.max(valid_values))
    else:
        gc_mean = 0.0
        gc_std = 0.0
        gc_min = 0.0
        gc_min_idx = 0
        gc_max = 0.0

    return {
        'gap_closure_per_scenario': [float(x) if not np.isnan(x) else None for x in gap_closure],
        'gap_closure_mean': gc_mean,
        'gap_closure_std': gc_std,
        'gap_closure_min': gc_min,
        'gap_closure_min_scenario_idx': gc_min_idx,
        'gap_closure_max': gc_max,
        'n_valid': n_valid,
        'n_total': len(F_star_arr),
    }


def compute_success_rate(F_star_per_scenario, F_actual_per_scenario,
                         F_baseline_per_scenario=None, threshold=0.95):
    """
    Compute win rate: percentage of scenarios where the controller's gap
    to the target is smaller than the baseline's gap.

    Win condition per scenario:
        |F_actual - F_star| < |F_baseline - F_star|

    Args:
        F_star_per_scenario (array-like): Target reliability for each scenario
        F_actual_per_scenario (array-like): Controller reliability for each scenario
        F_baseline_per_scenario (array-like): Baseline reliability for each scenario
        threshold (float): Unused, kept for backward compatibility

    Returns:
        dict with success_rate, n_successful, n_total, threshold
    """
    F_star_arr = np.atleast_1d(F_star_per_scenario)
    F_actual_arr = np.atleast_1d(F_actual_per_scenario)

    if F_baseline_per_scenario is not None:
        F_baseline_arr = np.atleast_1d(F_baseline_per_scenario)
        gap_actual = np.abs(F_actual_arr - F_star_arr)
        gap_baseline = np.abs(F_baseline_arr - F_star_arr)
        success_mask = gap_actual < gap_baseline
    else:
        # Fallback: old behaviour (F_actual >= threshold * F_star)
        success_mask = F_actual_arr >= (threshold * F_star_arr)

    n_successful = int(np.sum(success_mask))
    n_total = len(F_star_arr)
    success_rate = n_successful / n_total if n_total > 0 else 0.0

    return {
        'success_rate': float(success_rate),
        'success_rate_pct': float(success_rate * 100),
        'n_successful': n_successful,
        'n_total': n_total,
        'threshold': float(threshold),
    }


def compute_train_test_gap(F_star_train, F_actual_train, F_star_test, F_actual_test):
    """
    Compute train-test gap: difference between mean gaps on train and test sets.
    Gap = F_star - F_actual
    Train-test gap = mean(gap_train) - mean(gap_test)

    Positive value means the controller performs better on test (smaller gap).
    Negative value means the controller performs worse on test (larger gap).

    Args:
        F_star_train (array-like): Target reliability for train scenarios
        F_actual_train (array-like): Actual reliability for train scenarios
        F_star_test (array-like): Target reliability for test scenarios
        F_actual_test (array-like): Actual reliability for test scenarios

    Returns:
        dict: {
            'train_test_gap': mean(gap_train) - mean(gap_test),
            'mean_gap_train': Mean gap on train set,
            'mean_gap_test': Mean gap on test set,
        }
    """
    # Convert to arrays
    F_star_train_arr = np.atleast_1d(F_star_train)
    F_actual_train_arr = np.atleast_1d(F_actual_train)
    F_star_test_arr = np.atleast_1d(F_star_test)
    F_actual_test_arr = np.atleast_1d(F_actual_test)

    # Compute gaps
    gaps_train = F_star_train_arr - F_actual_train_arr
    gaps_test = F_star_test_arr - F_actual_test_arr

    # Compute means
    mean_gap_train = float(np.mean(gaps_train))
    mean_gap_test = float(np.mean(gaps_test))

    # Train-test gap
    train_test_gap = mean_gap_train - mean_gap_test

    return {
        'train_test_gap': float(train_test_gap),
        'mean_gap_train': mean_gap_train,
        'mean_gap_test': mean_gap_test,
    }


def compute_scenario_diversity(structural_conditions):
    """
    Compute scenario diversity score based on structural conditions variation.

    Uses coefficient of variation (CV = std/mean) averaged across all conditions.
    Higher CV means more diverse scenarios.

    Args:
        structural_conditions (dict): Dictionary mapping condition names to arrays
                                     e.g., {'AmbientTemp': [20, 25, 30], 'Humidity': [0.4, 0.5, 0.6]}

    Returns:
        dict: {
            'diversity_score': Overall diversity score (mean CV across all conditions),
            'per_condition_cv': CV for each condition,
            'per_condition_stats': {mean, std, min, max} for each condition,
        }
    """
    if not structural_conditions:
        return {
            'diversity_score': 0.0,
            'per_condition_cv': {},
            'per_condition_stats': {},
        }

    cvs = []
    per_condition_cv = {}
    per_condition_stats = {}

    for condition_name, values in structural_conditions.items():
        values_arr = np.atleast_1d(values)

        mean_val = float(np.mean(values_arr))
        std_val = float(np.std(values_arr))
        min_val = float(np.min(values_arr))
        max_val = float(np.max(values_arr))

        # Coefficient of variation (avoid division by zero)
        cv = (std_val / abs(mean_val)) if abs(mean_val) > 1e-10 else 0.0

        cvs.append(cv)
        per_condition_cv[condition_name] = float(cv)
        per_condition_stats[condition_name] = {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'range': max_val - min_val,
        }

    diversity_score = float(np.mean(cvs)) if cvs else 0.0

    return {
        'diversity_score': diversity_score,
        'per_condition_cv': per_condition_cv,
        'per_condition_stats': per_condition_stats,
    }


def compute_final_metrics(target_trajectory, baseline_trajectory, actual_trajectory, F_star, F_baseline, F_actual):
    """
    Compute comprehensive final metrics.

    Args:
        target_trajectory: Target trajectory (a*)
        baseline_trajectory: Baseline trajectory (a')
        actual_trajectory: Actual trajectory with controller (a)
        F_star: Target reliability
        F_baseline: Baseline reliability
        F_actual: Actual reliability

    Returns:
        dict: Complete metrics
    """
    # Distance metrics
    baseline_vs_target = compute_trajectory_distance(baseline_trajectory, target_trajectory)
    actual_vs_target = compute_trajectory_distance(actual_trajectory, target_trajectory)
    actual_vs_baseline = compute_trajectory_distance(actual_trajectory, baseline_trajectory)

    # Process-wise metrics
    process_metrics_baseline = compute_process_wise_metrics(baseline_trajectory, target_trajectory)
    process_metrics_actual = compute_process_wise_metrics(actual_trajectory, target_trajectory)

    # Improvement metrics
    improvement = ((F_actual - F_baseline) / abs(F_baseline)) if F_baseline != 0 else 0
    target_gap = abs((F_star - F_actual) / F_star) if F_star != 0 else 0

    return {
        'F_star': float(F_star),
        'F_baseline': float(F_baseline),
        'F_actual': float(F_actual),
        'improvement': float(improvement),
        'improvement_pct': float(improvement * 100),
        'target_gap': float(target_gap),
        'target_gap_pct': float(target_gap * 100),
        'distances': {
            'baseline_vs_target': baseline_vs_target,
            'actual_vs_target': actual_vs_target,
            'actual_vs_baseline': actual_vs_baseline,
        },
        'process_metrics': {
            'baseline': process_metrics_baseline,
            'actual': process_metrics_actual,
        }
    }


def detect_overfitting(history, train_test_gap_metrics=None, tolerance=0.005,
                       trend_window=20, severity_thresholds=None):
    """
    Comprehensive overfitting diagnosis from training history.

    Combines multiple signals to determine whether the controller is overfitting:
      1. Within-scenario divergence: train-split F vs val-split F
      2. Cross-scenario divergence: train loss vs cross-scenario validation loss
      3. Train-test gap: final evaluation F on train vs test scenarios
      4. Validation loss trend: is val loss rising while train loss falls?

    Args:
        history (dict): Training history from ControllerTrainer (keys like
            'total_loss', 'F_values', 'val_within_total_loss', 'val_within_F_values',
            'val_total_loss', 'val_F_values', 'reliability_weight', etc.)
        train_test_gap_metrics (dict or None): Output of compute_train_test_gap().
            If provided, the train_test_gap signal is included.
        tolerance (float): Absolute gap below which train/val differences are
            considered negligible (default 0.005).
        trend_window (int): Number of trailing epochs used to detect a rising
            validation-loss trend (default 20).
        severity_thresholds (dict or None): Override default severity bands.
            Keys: 'mild', 'moderate' (floats). Gaps above 'moderate' are 'severe'.

    Returns:
        dict with keys:
            'overfitting_detected' (bool): True if at least one signal fires.
            'severity' (str): 'none' | 'mild' | 'moderate' | 'severe'
            'signals' (list[dict]): Individual signal results, each with:
                'name', 'fired' (bool), 'detail' (str), 'value' (float or None)
            'summary' (str): Human-readable one-paragraph diagnosis.
            'recommendation' (str): Actionable suggestion.
            'epoch_of_divergence' (int or None): Earliest epoch where divergence
                was detected across all signals.
    """
    if severity_thresholds is None:
        severity_thresholds = {'mild': 0.005, 'moderate': 0.02}

    signals = []
    divergence_epochs = []

    # ------------------------------------------------------------------
    # Signal 1 — Within-scenario: train-split F vs val-split F
    # ------------------------------------------------------------------
    within_F_train = np.array(history.get('F_values', []))
    within_F_val = np.array(history.get('val_within_F_values', []))

    if len(within_F_train) > 0 and len(within_F_val) > 0:
        n = min(len(within_F_train), len(within_F_val))
        # Use last `trend_window` epochs for a stable estimate
        tail = min(trend_window, n)
        gap = float(np.mean(within_F_train[-tail:]) - np.mean(within_F_val[-tail:]))

        # Find first epoch where gap exceeds tolerance persistently
        gaps_full = within_F_train[:n] - within_F_val[:n]
        div_mask = gaps_full > tolerance
        first_div = int(np.argmax(div_mask)) + 1 if np.any(div_mask) else None

        fired = gap > tolerance
        signals.append({
            'name': 'within_scenario_F_gap',
            'fired': fired,
            'value': round(gap, 6),
            'detail': (f"Train-split F exceeds val-split F by {gap:.6f} "
                       f"(last {tail} epochs). "
                       f"{'Divergence at epoch ' + str(first_div) + '.' if first_div else 'No persistent divergence.'}"),
        })
        if first_div is not None and fired:
            divergence_epochs.append(first_div)
    else:
        signals.append({
            'name': 'within_scenario_F_gap',
            'fired': False,
            'value': None,
            'detail': 'Within-scenario validation not available.',
        })

    # ------------------------------------------------------------------
    # Signal 2 — Within-scenario loss divergence (val loss > train loss)
    # ------------------------------------------------------------------
    train_loss = np.array(history.get('total_loss', []))
    val_within_loss = np.array(history.get('val_within_total_loss', []))

    if len(train_loss) > 0 and len(val_within_loss) > 0:
        n = min(len(train_loss), len(val_within_loss))
        tail = min(trend_window, n)
        gap_loss = float(np.mean(val_within_loss[-tail:]) - np.mean(train_loss[-tail:]))

        div_mask = val_within_loss[:n] > train_loss[:n]
        first_div = int(np.argmax(div_mask)) + 1 if np.any(div_mask) else None

        fired = gap_loss > tolerance
        signals.append({
            'name': 'within_scenario_loss_gap',
            'fired': fired,
            'value': round(gap_loss, 6),
            'detail': (f"Val loss exceeds train loss by {gap_loss:.6f} "
                       f"(last {tail} epochs). "
                       f"{'Divergence at epoch ' + str(first_div) + '.' if first_div else ''}"),
        })
        if first_div is not None and fired:
            divergence_epochs.append(first_div)
    else:
        signals.append({
            'name': 'within_scenario_loss_gap',
            'fired': False,
            'value': None,
            'detail': 'Within-scenario validation loss not available.',
        })

    # ------------------------------------------------------------------
    # Signal 3 — Cross-scenario loss divergence
    # ------------------------------------------------------------------
    val_cross_loss = np.array(history.get('val_total_loss', []))

    if len(train_loss) > 0 and len(val_cross_loss) > 0:
        n = min(len(train_loss), len(val_cross_loss))
        tail = min(trend_window, n)
        gap_cross = float(np.mean(val_cross_loss[-tail:]) - np.mean(train_loss[-tail:]))

        div_mask = val_cross_loss[:n] > train_loss[:n]
        first_div = int(np.argmax(div_mask)) + 1 if np.any(div_mask) else None

        fired = gap_cross > tolerance
        signals.append({
            'name': 'cross_scenario_loss_gap',
            'fired': fired,
            'value': round(gap_cross, 6),
            'detail': (f"Cross-scenario val loss exceeds train loss by {gap_cross:.6f} "
                       f"(last {tail} epochs). "
                       f"{'Divergence at epoch ' + str(first_div) + '.' if first_div else ''}"),
        })
        if first_div is not None and fired:
            divergence_epochs.append(first_div)
    else:
        signals.append({
            'name': 'cross_scenario_loss_gap',
            'fired': False,
            'value': None,
            'detail': 'Cross-scenario validation not available.',
        })

    # ------------------------------------------------------------------
    # Signal 4 — Validation loss rising trend (late training)
    # ------------------------------------------------------------------
    # Pick the best available validation loss series
    val_series = None
    val_label = ''
    if len(val_within_loss) > 0:
        val_series = val_within_loss
        val_label = 'within-scenario'
    elif len(val_cross_loss) > 0:
        val_series = val_cross_loss
        val_label = 'cross-scenario'

    if val_series is not None and len(val_series) >= trend_window:
        tail_vals = val_series[-trend_window:]
        # Simple linear regression slope
        x = np.arange(trend_window, dtype=float)
        slope = float(np.polyfit(x, tail_vals, 1)[0])

        fired = slope > 0
        signals.append({
            'name': 'val_loss_rising_trend',
            'fired': fired,
            'value': round(slope, 8),
            'detail': (f"Validation loss ({val_label}) slope over last {trend_window} epochs: "
                       f"{slope:+.8f} per epoch. "
                       f"{'Rising — possible late overfitting.' if fired else 'Stable or decreasing.'}"),
        })
    else:
        signals.append({
            'name': 'val_loss_rising_trend',
            'fired': False,
            'value': None,
            'detail': f'Not enough epochs for trend analysis (need >= {trend_window}).',
        })

    # ------------------------------------------------------------------
    # Signal 5 — Train-test gap from final evaluation
    # ------------------------------------------------------------------
    if train_test_gap_metrics is not None:
        ttg = train_test_gap_metrics.get('train_test_gap', 0.0)
        # Negative ttg means controller is worse on test → overfitting
        fired = ttg < -tolerance
        signals.append({
            'name': 'train_test_gap',
            'fired': fired,
            'value': round(ttg, 6),
            'detail': (f"Train-test gap = {ttg:+.6f}. "
                       f"{'Controller performs worse on unseen test scenarios.' if fired else 'Consistent across train/test.'}"),
        })
    else:
        signals.append({
            'name': 'train_test_gap',
            'fired': False,
            'value': None,
            'detail': 'Train-test gap metrics not provided.',
        })

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    fired_signals = [s for s in signals if s['fired']]
    overfitting_detected = len(fired_signals) > 0
    n_fired = len(fired_signals)

    # Compute max absolute gap across fired numeric signals
    max_gap = 0.0
    for s in fired_signals:
        v = s['value']
        if v is not None:
            max_gap = max(max_gap, abs(v))

    # Severity: based on number of signals AND magnitude
    if not overfitting_detected:
        severity = 'none'
    elif max_gap > severity_thresholds['moderate'] or n_fired >= 3:
        severity = 'severe'
    elif max_gap > severity_thresholds['mild'] or n_fired >= 2:
        severity = 'moderate'
    else:
        severity = 'mild'

    epoch_of_divergence = min(divergence_epochs) if divergence_epochs else None

    # Build human-readable summary
    if not overfitting_detected:
        summary = ("No overfitting detected. Train and validation metrics are consistent "
                   "across all available signals.")
        recommendation = "Training looks healthy. No changes needed."
    else:
        fired_names = [s['name'] for s in fired_signals]
        summary = (f"Overfitting detected ({severity}): {n_fired}/{len(signals)} signals fired "
                   f"({', '.join(fired_names)}). "
                   f"Max gap magnitude: {max_gap:.6f}."
                   + (f" First divergence at epoch {epoch_of_divergence}."
                      if epoch_of_divergence else ""))

        if severity == 'mild':
            recommendation = ("Mild overfitting — consider increasing dropout, reducing model "
                              "capacity (hidden_sizes), or adding more training scenarios (n_train).")
        elif severity == 'moderate':
            recommendation = ("Moderate overfitting — try: (1) increase n_train for more scenario "
                              "diversity, (2) increase dropout or weight_decay, (3) reduce epochs "
                              "or lower patience, (4) enable cross_scenario validation for early "
                              "stopping on unseen conditions.")
        else:
            recommendation = ("Severe overfitting — the controller memorises training conditions. "
                              "Strongly recommended: (1) increase n_train significantly, "
                              "(2) reduce model capacity (smaller hidden_sizes), "
                              "(3) increase weight_decay and dropout, "
                              "(4) enable cross_scenario_enabled for early stopping, "
                              "(5) consider reducing batch_size or epochs.")

    return {
        'overfitting_detected': overfitting_detected,
        'severity': severity,
        'signals': signals,
        'summary': summary,
        'recommendation': recommendation,
        'epoch_of_divergence': epoch_of_divergence,
        'n_signals_fired': n_fired,
        'n_signals_total': len(signals),
        'max_gap': round(max_gap, 6),
    }
