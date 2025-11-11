"""
Metriche per valutazione controller optimization.
"""

import numpy as np
import torch


def convert_trajectory_to_numpy(trajectory):
    """
    Convert trajectory tensors to numpy arrays.

    Args:
        trajectory (dict): Trajectory with torch tensors

    Returns:
        dict: Trajectory with numpy arrays
    """
    numpy_traj = {}
    for process_name, data in trajectory.items():
        numpy_traj[process_name] = {
            'inputs': data['inputs'].detach().cpu().numpy() if torch.is_tensor(data['inputs']) else data['inputs'],
            'outputs_mean': data['outputs_mean'].detach().cpu().numpy() if torch.is_tensor(data['outputs_mean']) else data.get('outputs', data.get('outputs_mean')),
            'outputs_var': data.get('outputs_var', np.zeros_like(data.get('outputs', data.get('outputs_mean')))).detach().cpu().numpy() if torch.is_tensor(data.get('outputs_var', 0)) else data.get('outputs_var', np.zeros_like(data.get('outputs', data.get('outputs_mean'))))
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
    target_gap = ((F_star - F_actual) / F_star) if F_star != 0 else 0

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
