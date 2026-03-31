"""
Monte Carlo estimation of L_min using the analytical reliability formula.

Unlike montecarlo_lmin.py (Method 2) which uses the frozen surrogate f_Theta,
this estimator uses the closed-form reliability function F_analytical:

    F = sum_i  w_bar_i * Q_i(o_i, o_{<i})

where:
    Q_i = exp( -(o_i - tau_i(o_{<i}))^2 / s_i )
    tau_i(o_{<i}) = base_target_i + sum_{j in up(i)} alpha_ij * (o_j - b_ij)

This allows an apples-to-apples comparison:
    L_min_MC_analytical  ~=  L_min_Bellman   (both use analytical F)
    L_min_MC_surrogate   ~=  observed_loss   (both use f_Theta)

Formula:
    L_min_MC_analytical = (1/N) * sum_i  (1/(S-1)) * sum_s (F^(i,s) - F_bar^(i))^2
"""

import numpy as np
import torch
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path
import json


@dataclass
class MonteCarloAnalyticalLminResult:
    """Results from analytical Monte Carlo L_min estimation."""
    L_min_mc_analytical: float          # Unscaled variance estimate
    L_min_mc_analytical_scaled: float   # Scaled by loss_scale
    std_error: float                    # std(var_i) / sqrt(N)
    std_error_scaled: float             # std_error * loss_scale
    N_trajectories: int                 # Number of trajectories used
    S_samples: int                      # MC samples per trajectory
    per_trajectory_var: np.ndarray      # var^(i) for each trajectory (unscaled)
    computation_time_s: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'L_min_mc_analytical': self.L_min_mc_analytical,
            'L_min_mc_analytical_scaled': self.L_min_mc_analytical_scaled,
            'std_error': self.std_error,
            'std_error_scaled': self.std_error_scaled,
            'N_trajectories': self.N_trajectories,
            'S_samples': self.S_samples,
            'per_trajectory_var_mean': float(np.mean(self.per_trajectory_var)),
            'per_trajectory_var_std': float(np.std(self.per_trajectory_var)),
            'computation_time_s': self.computation_time_s,
        }

    def save(self, path: Path):
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def _get_process_config(surrogate, process_name):
    """
    Extract (base_target, scale, weight, adaptive_coefficients, adaptive_baselines)
    for a given process from the surrogate's config.

    Supports both _dynamic_configs (ST processes) and PROCESS_CONFIGS (legacy).
    """
    if (hasattr(surrogate, '_dynamic_configs')
            and surrogate._dynamic_configs is not None
            and process_name in surrogate._dynamic_configs):
        cfg = surrogate._dynamic_configs[process_name]
        return {
            'base_target': cfg.get('base_target', 0.0),
            'scale': cfg.get('scale', 1.0),
            'weight': cfg.get('weight', 1.0),
            'adaptive_coefficients': cfg.get('adaptive_coefficients', {}),
            'adaptive_baselines': cfg.get('adaptive_baselines', {}),
        }

    # Legacy hardcoded configs
    from controller_optimization.src.models.surrogate import ProTSurrogate
    pc = ProTSurrogate.PROCESS_CONFIGS.get(process_name, {})

    # Legacy adaptive coefficients (hardcoded relationships)
    adaptive_coefficients = {}
    adaptive_baselines = {}
    if process_name == 'plasma':
        adaptive_coefficients = {'laser': 0.2}
        adaptive_baselines = {'laser': 0.8}
    elif process_name == 'galvanic':
        adaptive_coefficients = {'plasma': 0.5, 'laser': 0.4}
        adaptive_baselines = {'plasma': 5.0, 'laser': 0.5}
    elif process_name == 'microetch':
        adaptive_coefficients = {'laser': 1.5, 'plasma': 0.3, 'galvanic': -0.15}
        adaptive_baselines = {'laser': 0.5, 'plasma': 5.0, 'galvanic': 10.0}

    return {
        'base_target': pc.get('target', 0.0),
        'scale': pc.get('scale', 1.0),
        'weight': pc.get('weight', 1.0),
        'adaptive_coefficients': adaptive_coefficients,
        'adaptive_baselines': adaptive_baselines,
    }


def _compute_F_analytical(outputs, process_names, configs):
    """
    Compute the closed-form analytical reliability F from sampled outputs.

    F = sum_i  w_bar_i * Q_i
    Q_i = exp( -(o_i - tau_i)^2 / s_i )
    tau_i = base_target + sum_j alpha_j * (o_j - b_j)

    Args:
        outputs: dict {process_name: tensor (S,)} of sampled outputs
        process_names: ordered list of process names
        configs: dict {process_name: config_dict} from _get_process_config

    Returns:
        F_analytical: tensor (S,)
    """
    Q_values = []
    weights = []

    for proc_name in process_names:
        cfg = configs[proc_name]
        o_i = outputs[proc_name]  # (S,)

        # Compute adaptive target tau_i
        tau_i = cfg['base_target']
        for upstream_name, coeff in cfg['adaptive_coefficients'].items():
            if upstream_name in outputs:
                baseline = cfg['adaptive_baselines'][upstream_name]
                tau_i = tau_i + coeff * (outputs[upstream_name] - baseline)

        # Quality score Q_i = exp(-(o_i - tau_i)^2 / s_i)
        scale = max(cfg['scale'], 1e-8)
        Q_i = torch.exp(-((o_i - tau_i) ** 2) / scale)

        Q_values.append(Q_i)
        weights.append(cfg['weight'])

    # Weighted average: F = sum(w_bar_i * Q_i)
    w = torch.tensor(weights, dtype=Q_values[0].dtype, device=Q_values[0].device)
    w_bar = w / w.sum()

    F = sum(w_bar[i] * Q_values[i] for i in range(len(Q_values)))
    return F  # (S,)


def compute_montecarlo_lmin_analytical(
    process_chain,
    surrogate,
    n_scenarios: int,
    S: int = 50,
    loss_scale: float = 100.0,
    verbose: bool = True,
) -> MonteCarloAnalyticalLminResult:
    """
    Compute L_min_MC_analytical — Monte Carlo estimate of Var[F_analytical].

    Uses the closed-form F = sum w_bar_i * Q_i instead of the frozen surrogate
    f_Theta. This allows direct comparison with L_min_Bellman (both use the
    analytical SCM formula).

    For each scenario i = 1..N:
      1. Run one forward pass through the trained controller to get the
         actions {a^(i)_t} and the frozen UP predictions {mu_t, sigma^2_t}.
      2. Keep a^(i)_t FIXED, resample S times: o^(i,s)_t ~ N(mu_t, sigma^2_t).
      3. Compute F_analytical for each of the S perturbations using the
         closed-form quality formula (NOT f_Theta).
      4. Compute unbiased sample variance of F_analytical across S samples.

    L_min = mean of per-trajectory variances.

    Args:
        process_chain: ProcessChain with frozen UPs and trained policies.
        surrogate: ProTSurrogate (used only for SCM config — NOT called as f_Theta).
        n_scenarios: Number of training scenarios (N trajectories).
        S: MC perturbation samples per trajectory (default 50). Must be >= 2.
        loss_scale: Scale factor matching training (default 100.0).
        verbose: Print progress.

    Returns:
        MonteCarloAnalyticalLminResult with L_min_MC_analytical and diagnostics.
    """
    if S < 2:
        raise ValueError(
            f"S must be >= 2 for unbiased variance estimation (got S={S}). "
            f"With S=1 the (S-1) denominator is zero."
        )

    t0 = time.time()

    process_chain.eval()
    process_names = process_chain.process_names
    device = process_chain.device
    N = n_scenarios

    # Pre-extract process configs from surrogate (closed-form parameters)
    configs = {}
    for proc_name in process_names:
        configs[proc_name] = _get_process_config(surrogate, proc_name)

    per_trajectory_var = np.zeros(N)

    with torch.no_grad():
        for i in range(N):
            # Step 1: forward pass with trained controller
            ref_traj = process_chain.forward(batch_size=1, scenario_idx=i)

            # Collect per-stage UP predictions (mu, sigma^2)
            stage_mu = []
            stage_var = []

            for proc_name in process_names:
                data = ref_traj[proc_name]
                stage_mu.append(data['outputs_mean'])    # (1, output_dim)
                stage_var.append(data['outputs_var'])     # (1, output_dim)

            # Step 2: sample S perturbed output vectors
            outputs = {}
            for t, proc_name in enumerate(process_names):
                mu_t = stage_mu[t]   # (1, output_dim)
                var_t = stage_var[t]  # (1, output_dim)

                # Expand to (S, output_dim)
                mu_expanded = mu_t.expand(S, -1)
                var_expanded = var_t.expand(S, -1)

                # Sample: o^(i,s)_t ~ N(mu_t, sigma^2_t)
                std_t = torch.sqrt(var_expanded + 1e-8)
                epsilon = torch.randn_like(mu_expanded)
                o_sampled = mu_expanded + epsilon * std_t

                # Squeeze to (S,) since each process has 1 output
                outputs[proc_name] = o_sampled.squeeze(-1)

            # Step 3: compute F_analytical for all S samples (closed-form)
            F_analytical = _compute_F_analytical(outputs, process_names, configs)
            F_np = F_analytical.detach().cpu().numpy()  # (S,)

            # Step 4: unbiased sample variance with (S-1) denominator
            F_bar_i = np.mean(F_np)
            var_i = np.sum((F_np - F_bar_i) ** 2) / (S - 1)
            per_trajectory_var[i] = var_i

            if verbose and (i + 1) % max(1, N // 5) == 0:
                print(f"    MC L_min analytical: trajectory {i+1}/{N}, "
                      f"var^(i)={var_i:.8f}, F_bar={F_bar_i:.6f}")

    # Aggregate: L_min = mean of per-trajectory variances
    L_min = float(np.mean(per_trajectory_var))
    L_min_scaled = L_min * loss_scale
    std_var = float(np.std(per_trajectory_var, ddof=1)) if N > 1 else 0.0
    std_error = std_var / np.sqrt(N) if N > 1 else 0.0
    std_error_scaled = std_error * loss_scale

    elapsed = time.time() - t0

    result = MonteCarloAnalyticalLminResult(
        L_min_mc_analytical=L_min,
        L_min_mc_analytical_scaled=L_min_scaled,
        std_error=std_error,
        std_error_scaled=std_error_scaled,
        N_trajectories=N,
        S_samples=S,
        per_trajectory_var=per_trajectory_var,
        computation_time_s=elapsed,
    )

    if verbose:
        print(f"\n  Monte Carlo L_min (Analytical) results:")
        print(f"    L_min_MC_analytical (unscaled):  {L_min:.10f}")
        print(f"    L_min_MC_analytical (x{loss_scale:.0f}):     {L_min_scaled:.6f}")
        print(f"    Std error (scaled):              +/-{std_error_scaled:.6f}")
        print(f"    N trajectories:                  {N}")
        print(f"    S samples/trajectory:            {S}")
        print(f"    Computation time:                {elapsed:.1f}s")

    # Sanity check
    if L_min <= 0 and verbose:
        print(f"  WARNING: L_min_MC_analytical <= 0 ({L_min:.10f}). "
              f"This should not happen if any sigma_i > 0.")

    return result
