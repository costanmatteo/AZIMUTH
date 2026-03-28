"""
Metriche per valutazione controller optimization.
"""

import logging
import warnings

import numpy as np
import torch

logger = logging.getLogger(__name__)


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


def compute_empirical_lmin(uncertainty_predictor, surrogate, target_trajectory,
                           F_star, N=500, device='cpu'):
    """
    Empirical minimum achievable loss estimator L̂_min (Eq. 3.46–3.48).

    Estimates the irreducible loss floor for a dataset by sampling N complete
    trajectories from the uncertainty predictor's predictive distributions
    along the optimal target trajectory, scoring each with the surrogate,
    and computing the mean squared deviation from F*.

    Args:
        uncertainty_predictor: Callable a*_t → (µ_t, σ²_t) as tensors.
        surrogate: Callable trajectory tensor (P, d_out) → scalar F̂.
        target_trajectory: Sequence of P action tensors {a*_t}.
        F_star (float): Target reliability score.
        N (int): Number of Monte Carlo trajectories to sample (default: 500).
        device (str): Torch device (default: 'cpu').

    Returns:
        float: Estimated L̂_min ≥ 0.
    """
    if N < 100:
        warnings.warn(
            f"N={N} is small; the L̂_min estimate may have high variance. "
            "Consider N >= 100 for stable estimates.",
            stacklevel=2,
        )

    P = len(target_trajectory)
    F_star_tensor = torch.tensor(F_star, dtype=torch.float32, device=device)

    with torch.no_grad():
        # Step 2: get predictive distributions for each step
        mus = []
        vars_ = []
        for t in range(P):
            a_t = target_trajectory[t]
            if not torch.is_tensor(a_t):
                a_t = torch.tensor(a_t, dtype=torch.float32, device=device)
            else:
                a_t = a_t.to(device)
            mu_t, var_t = uncertainty_predictor(a_t)
            mus.append(mu_t)
            vars_.append(var_t)

        # Stack: (P, d_out) each
        mus = torch.stack(mus)     # (P, d_out)
        vars_ = torch.stack(vars_) # (P, d_out)
        stds = torch.sqrt(vars_ + 1e-8)

        # Step 3–4: sample N trajectories and score each
        # Use fixed seed for reproducibility in eval mode
        generator = torch.Generator(device=device)
        generator.manual_seed(42)

        F_samples = torch.empty(N, dtype=torch.float32, device=device)
        for k in range(N):
            # Sample o_{t,k} ~ N(mu_t, sigma_t^2) for all t at once
            eps = torch.randn(mus.shape, generator=generator,
                              dtype=torch.float32, device=device)
            trajectory_k = mus + eps * stds  # (P, d_out)
            F_samples[k] = surrogate(trajectory_k)

        # Step 5: L̂_min = (1/N) * Σ (F̂_k - F*)²
        l_hat_min = torch.mean((F_samples - F_star_tensor) ** 2).item()

    # Validity check: warn if suspiciously close to 0
    if l_hat_min < 1e-7:
        logger.warning(
            "L̂_min = %.2e is implausibly close to 0. This may indicate "
            "surrogate overfitting near the target trajectory τ*.",
            l_hat_min,
        )

    return l_hat_min


def compute_training_efficiency_emp(L_phi, L_hat_min):
    """
    Empirical training efficiency η_emp = L̂_min / L(Φ), clipped to [0, 1].

    Values near 1 indicate the controller loss is close to the irreducible
    minimum; values near 0 indicate large controllable suboptimality.

    Args:
        L_phi (float): Current controller loss L(Φ).
        L_hat_min (float): Empirical minimum achievable loss L̂_min.

    Returns:
        float: η_emp ∈ [0, 1].
    """
    if L_phi <= 0:
        raise ValueError(f"L_phi must be positive, got {L_phi}")
    ratio = L_hat_min / L_phi
    return float(max(0.0, min(1.0, ratio)))
