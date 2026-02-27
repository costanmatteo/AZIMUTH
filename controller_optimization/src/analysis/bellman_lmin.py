"""
Bellman backward-induction computation of L_min for the AZIMUTH pipeline.

Computes the theoretical lower bound of the loss L = E[(F - F*)^2] for a
manufacturing pipeline with n=4 processes (Laser -> Plasma -> Galvanic -> Microetch).

The computation accounts for:
- Variance-dependent controller actions (manifold M_i)
- Correlated noise between processes (non-diagonal Sigma)
- Optimal reactive controller exploiting upstream observations

Three phases:
1. Input estimation (Sigma, M_i) from trained models and data
2. Backward induction on discretised state grid (i=4 to i=1)
3. Forward simulation validation

Result: L_min = V_1(F*, empty) > 0.
"""

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import time


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BellmanConfig:
    """Configuration for backward induction grid."""
    # Grid sizes
    N_R: int = 200          # Grid points for remaining reliability R
    N_eps: int = 30         # Grid points per noise dimension
    eps_range: float = 3.0  # Noise range: [-eps_range, +eps_range]
    R_min: float = -0.1     # Lower bound of R grid
    # R_max is set to F* dynamically

    # Manifold resolution
    M_actions: int = 100    # Number of action candidates per process

    # Monte Carlo
    K_mc: int = 1000        # MC samples for non-terminal steps
    use_antithetic: bool = True  # Antithetic variates

    # Forward validation
    N_forward: int = 10000  # Forward simulation trajectories

    # Regularisation
    sigma_shrinkage: float = 0.01  # Shrinkage for Sigma if not PD


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BellmanLminResult:
    """Results from backward induction."""
    L_min_bellman: float        # Bellman backward induction result
    L_min_forward: float        # Forward simulation estimate
    L_min_forward_se: float     # Standard error of forward estimate
    L_min_naive: float          # Naive (non-reactive) lower bound
    F_star: float               # Target reliability
    Sigma: np.ndarray           # Noise covariance matrix (4x4)
    w_bar: np.ndarray           # Normalised weights
    scales: np.ndarray          # Scale parameters
    n_manifold_points: List[int]  # Number of M_i points per process
    computation_time_s: float   # Wall-clock time

    def to_dict(self) -> Dict[str, Any]:
        return {
            'L_min_bellman': self.L_min_bellman,
            'L_min_forward': self.L_min_forward,
            'L_min_forward_se': self.L_min_forward_se,
            'L_min_naive': self.L_min_naive,
            'F_star': self.F_star,
            'Sigma': self.Sigma.tolist(),
            'w_bar': self.w_bar.tolist(),
            'scales': self.scales.tolist(),
            'n_manifold_points': self.n_manifold_points,
            'computation_time_s': self.computation_time_s,
        }

    def save(self, path: Path):
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Input estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_noise_covariance(
    process_chain,
    n_samples: int = 2000,
    scenario_idx: int = 0,
    shrinkage: float = 0.01,
) -> np.ndarray:
    """
    Estimate the 4x4 noise covariance matrix Sigma from standardised residuals.

    For each forward pass, compute eps_i = (o_sampled_i - mu_i) / sigma_i
    for each process i. Then compute the sample covariance of these residuals.

    Args:
        process_chain: ProcessChain with trained UPs
        n_samples: Number of forward passes to collect
        scenario_idx: Scenario to use
        shrinkage: Ledoit-Wolf shrinkage parameter

    Returns:
        Sigma: (4, 4) positive-definite covariance matrix
    """
    n_processes = len(process_chain.process_names)
    residuals = np.zeros((n_samples, n_processes))

    process_chain.eval()
    with torch.no_grad():
        for k in range(n_samples):
            trajectory = process_chain.forward(batch_size=1, scenario_idx=scenario_idx)

            for i, proc_name in enumerate(process_chain.process_names):
                data = trajectory[proc_name]
                mu = data['outputs_mean'].item()
                var = data['outputs_var'].item()
                sigma = np.sqrt(var + 1e-8)
                o_sampled = data['outputs_sampled'].item()
                residuals[k, i] = (o_sampled - mu) / sigma

    # Sample covariance
    Sigma_hat = np.cov(residuals.T)

    # Regularise to ensure positive definiteness
    eigvals = np.linalg.eigvalsh(Sigma_hat)
    if eigvals.min() <= 0:
        Sigma = (1 - shrinkage) * Sigma_hat + shrinkage * np.eye(n_processes)
    else:
        Sigma = Sigma_hat

    # Verify PD
    eigvals_final = np.linalg.eigvalsh(Sigma)
    if eigvals_final.min() <= 0:
        # Force PD via eigenvalue clipping
        eigvals_clip, eigvecs = np.linalg.eigh(Sigma)
        eigvals_clip = np.maximum(eigvals_clip, 1e-6)
        Sigma = eigvecs @ np.diag(eigvals_clip) @ eigvecs.T

    return Sigma


def compute_manifold(
    process_chain,
    process_idx: int,
    scenario_idx: int = 0,
    n_actions: int = 100,
) -> np.ndarray:
    """
    Compute the achievable manifold M_i = {(mu_i, sigma2_i)} for process i.

    Sweeps controllable inputs over their range while fixing non-controllable
    inputs from the target trajectory.

    Args:
        process_chain: ProcessChain
        process_idx: Index of the process (0-3)
        scenario_idx: Scenario to use for context
        n_actions: Number of grid points for the sweep

    Returns:
        manifold: (M, 2) array where columns are [mu_i, sigma2_i]
    """
    proc_config = process_chain.processes_config[process_idx]
    proc_name = proc_config['name']
    info = process_chain.controllable_info_per_process[process_idx]

    # Get bounds for controllable inputs from preprocessor
    preprocessor = process_chain.preprocessors[process_idx]
    ctrl_indices = info['controllable_indices']
    n_ctrl = info['n_controllable']

    if preprocessor.input_min is not None:
        ctrl_min = np.array([preprocessor.input_min[idx] for idx in ctrl_indices])
        ctrl_max = np.array([preprocessor.input_max[idx] for idx in ctrl_indices])
    else:
        # Fallback: use a wide range
        ctrl_min = np.zeros(n_ctrl)
        ctrl_max = np.ones(n_ctrl)

    # Generate grid over controllable inputs
    if n_ctrl == 1:
        grid_values = np.linspace(ctrl_min[0], ctrl_max[0], n_actions).reshape(-1, 1)
    elif n_ctrl == 2:
        # 2D grid: sqrt(n_actions) points per dimension
        n_per_dim = max(int(np.sqrt(n_actions)), 5)
        g0 = np.linspace(ctrl_min[0], ctrl_max[0], n_per_dim)
        g1 = np.linspace(ctrl_min[1], ctrl_max[1], n_per_dim)
        g0v, g1v = np.meshgrid(g0, g1)
        grid_values = np.column_stack([g0v.ravel(), g1v.ravel()])
    else:
        # For 3+ dims, use random sampling
        grid_values = np.random.uniform(ctrl_min, ctrl_max, size=(n_actions, n_ctrl))

    # Get target inputs for non-controllable values
    target_inputs = process_chain.target_trajectory[proc_name]['inputs'][scenario_idx]

    results = []
    process_chain.eval()
    with torch.no_grad():
        for action in grid_values:
            # Build full input: merge controllable action with non-controllable from target
            full_input = np.array(target_inputs, dtype=np.float64).copy()
            for out_idx, input_idx in enumerate(ctrl_indices):
                full_input[input_idx] = action[out_idx]

            input_tensor = torch.tensor(full_input, dtype=torch.float32).unsqueeze(0).to(
                process_chain.device
            )

            # Scale, predict, unscale
            scaled = process_chain.scale_inputs(input_tensor, process_idx)
            mu_scaled, var_scaled = process_chain.uncertainty_predictors[process_idx](scaled)
            mu = process_chain.unscale_outputs(mu_scaled, process_idx)
            var = process_chain.unscale_variance(var_scaled, process_idx)

            results.append([mu.item(), var.item()])

    manifold = np.array(results)  # (M, 2): [mu, sigma2]
    return manifold


def get_adaptive_target(process_idx: int, upstream_outputs: Optional[Dict[str, float]] = None) -> float:
    """
    Get the adaptive target for process i given upstream outputs.

    Mirrors the logic in ProTSurrogate.compute_reliability().

    Args:
        process_idx: 0=laser, 1=plasma, 2=galvanic, 3=microetch
        upstream_outputs: dict {proc_name: output_value} for completed processes

    Returns:
        target value tau_i
    """
    if upstream_outputs is None:
        upstream_outputs = {}

    if process_idx == 0:  # laser
        return 0.8
    elif process_idx == 1:  # plasma
        target = 3.0
        if 'laser' in upstream_outputs:
            target += 0.2 * (upstream_outputs['laser'] - 0.8)
        return target
    elif process_idx == 2:  # galvanic
        target = 10.0
        if 'plasma' in upstream_outputs:
            target += 0.5 * (upstream_outputs['plasma'] - 5.0)
        if 'laser' in upstream_outputs:
            target += 0.4 * (upstream_outputs['laser'] - 0.5)
        return target
    elif process_idx == 3:  # microetch
        target = 20.0
        if 'laser' in upstream_outputs:
            target += 1.5 * (upstream_outputs['laser'] - 0.5)
        if 'plasma' in upstream_outputs:
            target += 0.3 * (upstream_outputs['plasma'] - 5.0)
        if 'galvanic' in upstream_outputs:
            target -= 0.15 * (upstream_outputs['galvanic'] - 10.0)
        return target
    else:
        raise ValueError(f"Invalid process_idx: {process_idx}")


def get_target_outputs(process_chain, scenario_idx: int = 0) -> Dict[str, float]:
    """
    Get the target trajectory outputs for a given scenario.

    Returns:
        dict mapping process_name to target output value
    """
    target_outputs = {}
    for proc_name in process_chain.process_names:
        target_out = process_chain.target_trajectory[proc_name]['outputs'][scenario_idx]
        target_outputs[proc_name] = float(target_out[0]) if len(target_out.shape) > 0 else float(target_out)
    return target_outputs


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Backward induction helpers
# ─────────────────────────────────────────────────────────────────────────────

def precompute_conditional_params(Sigma: np.ndarray, n: int = 4):
    """
    Precompute conditional Gaussian parameters from Sigma.

    For process i (0-indexed), given eps_{<i}, the conditional distribution is:
        eps_i | eps_{<i} ~ N(cond_weights[i] @ eps_{<i}, sigma2_cond[i])

    For i=0: unconditional, eps_0 ~ N(0, Sigma[0,0])

    Args:
        Sigma: (n, n) covariance matrix
        n: number of processes

    Returns:
        sigma2_cond: dict {i: conditional variance} for i=0,...,n-1
        cond_weights: dict {i: weight vector of length i} for i=1,...,n-1
    """
    sigma2_cond = {}
    cond_weights = {}

    # i=0: unconditional
    sigma2_cond[0] = Sigma[0, 0]

    for i in range(1, n):
        Sigma_past = Sigma[:i, :i]  # (i, i)
        Sigma_cross = Sigma[i, :i]  # (i,)
        Sigma_ii = Sigma[i, i]      # scalar

        Sigma_past_inv = np.linalg.inv(Sigma_past)
        cond_weights[i] = Sigma_cross @ Sigma_past_inv  # (i,)
        sigma2_cond[i] = Sigma_ii - Sigma_cross @ Sigma_past_inv @ Sigma_cross  # scalar
        sigma2_cond[i] = max(sigma2_cond[i], 1e-10)  # Numerical safety

    return sigma2_cond, cond_weights


def build_grids(F_star: float, cfg: BellmanConfig):
    """
    Build discretisation grids for R and eps.

    Returns:
        grid_R: (N_R,) array
        grid_eps: (N_eps,) array
    """
    grid_R = np.linspace(cfg.R_min, F_star, cfg.N_R)
    grid_eps = np.linspace(-cfg.eps_range, cfg.eps_range, cfg.N_eps)
    return grid_R, grid_eps


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Terminal step (i=3, 0-indexed -> process 4)
# ─────────────────────────────────────────────────────────────────────────────

def bellman_terminal(
    grid_R: np.ndarray,
    grid_eps: np.ndarray,
    manifold: np.ndarray,
    w_bar_i: float,
    s_i: float,
    tau_i: float,
    sigma2_cond_i: float,
    cond_weights_i: np.ndarray,
) -> np.ndarray:
    """
    Closed-form Bellman for the terminal step (process index 3, last process).

    V_4(R_4, eps_{<4}) = min_{(mu,sigma2) in M_4} E[ (R_4 - w_bar_4 * Q_4)^2 | eps_{<4} ]

    Uses the Gaussian integral identity:
        E[exp(-((alpha + beta*z)^2) / gamma)] = exp(-alpha^2/(gamma+2*beta^2)) / sqrt(1+2*beta^2/gamma)

    Args:
        grid_R: (N_R,) R values
        grid_eps: (N_eps,) eps values per dimension
        manifold: (M, 2) array [mu, sigma2]
        w_bar_i: normalised weight for this process
        s_i: scale parameter
        tau_i: adaptive target (at target trajectory operating point)
        sigma2_cond_i: conditional variance of eps_i | eps_{<i}
        cond_weights_i: (3,) conditional mean weights

    Returns:
        V: array of shape (N_R, N_eps, N_eps, N_eps)
    """
    N_R = len(grid_R)
    N_eps = len(grid_eps)
    M = manifold.shape[0]

    # Extract manifold columns
    mu_m = manifold[:, 0]       # (M,)
    sigma2_m = manifold[:, 1]   # (M,)
    sigma_m = np.sqrt(np.maximum(sigma2_m, 1e-10))  # (M,)

    # Compute delta_m = mu_m - tau_i for each action
    delta_m = mu_m - tau_i  # (M,)

    # Build meshgrid for eps_{<4} = (eps_0, eps_1, eps_2)
    # cond_weights_i has shape (3,)
    # eps_hat_4 = cond_weights_i[0]*e0 + cond_weights_i[1]*e1 + cond_weights_i[2]*e2

    # Precompute eps_hat for all grid points
    # Shape: (N_eps, N_eps, N_eps)
    e0, e1, e2 = np.meshgrid(grid_eps, grid_eps, grid_eps, indexing='ij')
    eps_hat = cond_weights_i[0] * e0 + cond_weights_i[1] * e1 + cond_weights_i[2] * e2
    # Shape: (N_eps, N_eps, N_eps)

    sqrt_cond = np.sqrt(sigma2_cond_i)

    # For each action m and grid point, compute:
    # d_4 = delta_m + sigma_m * eps_hat
    # beta = sigma_m * sqrt_cond
    # Then:
    # E[Q_4] = exp(-d^2 / (s + 2*beta^2)) / sqrt(1 + 2*beta^2/s)
    # E[Q_4^2] = exp(-2*d^2 / (s + 4*beta^2)) / sqrt(1 + 4*beta^2/s)
    # cost = R^2 - 2*R*w*E[Q] + w^2*E[Q^2]

    # We want to vectorise over (R, e0, e1, e2, m)
    # Shape strategy: R -> (N_R,1,1,1,1), eps_hat -> (1,Ne,Ne,Ne,1), m -> (1,1,1,1,M)

    R = grid_R[:, None, None, None, None]       # (N_R,1,1,1,1)
    eps_hat_5d = eps_hat[None, :, :, :, None]    # (1,Ne,Ne,Ne,1)
    delta_5d = delta_m[None, None, None, None, :]  # (1,1,1,1,M)
    sigma_5d = sigma_m[None, None, None, None, :]  # (1,1,1,1,M)

    d = delta_5d + sigma_5d * eps_hat_5d    # (N_R,Ne,Ne,Ne,M) — broadcast on R trivially
    # d actually doesn't depend on R, but keeping 5 dims for broadcasting

    beta = sigma_5d * sqrt_cond  # (1,1,1,1,M)
    beta2 = beta ** 2

    # E[Q]
    denom1 = s_i + 2 * beta2
    EQ = np.exp(-d ** 2 / denom1) / np.sqrt(1 + 2 * beta2 / s_i)
    # (N_R, Ne, Ne, Ne, M) — still no true R dependence in EQ

    # E[Q^2]
    denom2 = s_i + 4 * beta2
    EQ2 = np.exp(-2 * d ** 2 / denom2) / np.sqrt(1 + 4 * beta2 / s_i)

    # cost(R, eps, m) = R^2 - 2*R*w*EQ + w^2*EQ2
    cost = R ** 2 - 2 * R * w_bar_i * EQ + w_bar_i ** 2 * EQ2
    # (N_R, Ne, Ne, Ne, M)

    # Minimise over actions
    V = np.min(cost, axis=-1)  # (N_R, Ne, Ne, Ne)

    return V


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Non-terminal steps (i=2,1,0 in 0-indexed)
# ─────────────────────────────────────────────────────────────────────────────

def bellman_non_terminal(
    grid_R: np.ndarray,
    grid_eps: np.ndarray,
    V_next: np.ndarray,
    manifold: np.ndarray,
    w_bar_i: float,
    s_i: float,
    tau_i: float,
    sigma2_cond_i: float,
    cond_weights_i: Optional[np.ndarray],
    process_idx: int,
    K: int = 1000,
    use_antithetic: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Monte Carlo Bellman for non-terminal step (process_idx in {0,1,2}).

    V_i(R_i, eps_{<i}) = min_{(mu,sigma2) in M_i} E[ V_{i+1}(R_{i+1}, eps_{<=i}) | eps_{<i} ]

    Args:
        grid_R: (N_R,)
        grid_eps: (N_eps,)
        V_next: value function for step i+1
            - If process_idx=2: shape (N_R, N_eps, N_eps, N_eps) for step 3
            - If process_idx=1: shape (N_R, N_eps, N_eps) for step 2
            - If process_idx=0: shape (N_R, N_eps) for step 1
        manifold: (M, 2) array [mu, sigma2]
        w_bar_i: normalised weight
        s_i: scale parameter
        tau_i: adaptive target
        sigma2_cond_i: conditional variance of eps_i | eps_{<i}
        cond_weights_i: conditional mean weights (None for i=0)
        process_idx: 0-indexed process index
        K: MC samples
        use_antithetic: use antithetic variates
        rng: random generator

    Returns:
        V: value function for step i
            - process_idx=2: (N_R, N_eps, N_eps)
            - process_idx=1: (N_R, N_eps)
            - process_idx=0: scalar
    """
    if rng is None:
        rng = np.random.default_rng(42)

    N_R = len(grid_R)
    N_eps = len(grid_eps)
    M = manifold.shape[0]
    mu_m = manifold[:, 0]
    sigma2_m = manifold[:, 1]
    sigma_m = np.sqrt(np.maximum(sigma2_m, 1e-10))
    delta_m = mu_m - tau_i

    sqrt_cond = np.sqrt(sigma2_cond_i)

    # Build interpolator for V_next
    if process_idx == 2:
        # V_next shape: (N_R, N_eps, N_eps, N_eps)
        interp = RegularGridInterpolator(
            (grid_R, grid_eps, grid_eps, grid_eps),
            V_next,
            method='linear',
            bounds_error=False,
            fill_value=None,  # Extrapolate
        )
    elif process_idx == 1:
        # V_next shape: (N_R, N_eps, N_eps)
        interp = RegularGridInterpolator(
            (grid_R, grid_eps, grid_eps),
            V_next,
            method='linear',
            bounds_error=False,
            fill_value=None,
        )
    elif process_idx == 0:
        # V_next shape: (N_R, N_eps)
        interp = RegularGridInterpolator(
            (grid_R, grid_eps),
            V_next,
            method='linear',
            bounds_error=False,
            fill_value=None,
        )
    else:
        raise ValueError(f"Invalid process_idx: {process_idx}")

    # Generate MC samples (common random numbers)
    if use_antithetic:
        K_half = K // 2
        z_pos = rng.standard_normal(K_half)
        z_all = np.concatenate([z_pos, -z_pos])  # (K,)
    else:
        z_all = rng.standard_normal(K)

    K_actual = len(z_all)

    # Grid bounds for clamping
    R_lo, R_hi = grid_R[0], grid_R[-1]
    eps_lo, eps_hi = grid_eps[0], grid_eps[-1]

    if process_idx == 0:
        # i=0: state is (F*, empty) — single point
        # eps_0 ~ N(0, Sigma[0,0]) => eps_0 = sqrt(Sigma[0,0]) * z
        # Here sigma2_cond_i = Sigma[0,0] for i=0
        eps_i_samples = sqrt_cond * z_all  # (K,)

        # For each action m, for each sample k:
        # Q_0_k = exp(-(delta_m + sigma_m * eps_i_k)^2 / s_i)
        # R_1_k = F* - w_bar_i * Q_0_k (F* = grid_R[-1])
        # V_next expects (R_1, eps_0): shape (K*M, 2)
        F_star = grid_R[-1]  # F* is the max R value

        best_cost = np.inf
        for m_idx in range(M):
            d_km = delta_m[m_idx] + sigma_m[m_idx] * eps_i_samples  # (K,)
            Q_km = np.exp(-(d_km ** 2) / s_i)  # (K,)
            R_next = F_star - w_bar_i * Q_km  # (K,)

            # Clamp to grid bounds
            R_next_c = np.clip(R_next, R_lo, R_hi)
            eps_c = np.clip(eps_i_samples, eps_lo, eps_hi)

            # Interpolate V_next(R_next, eps_0)
            points = np.column_stack([R_next_c, eps_c])  # (K, 2)
            V_vals = interp(points)  # (K,)
            cost_m = np.mean(V_vals)

            if cost_m < best_cost:
                best_cost = cost_m

        return best_cost  # scalar

    elif process_idx == 1:
        # i=1: state is (R_1, eps_0) -> output V shape (N_R, N_eps)
        V_out = np.zeros((N_R, N_eps))

        for r_idx in range(N_R):
            R_val = grid_R[r_idx]
            for e0_idx in range(N_eps):
                eps_0_val = grid_eps[e0_idx]

                # eps_hat_1 = cond_weights_i[0] * eps_0
                eps_hat = cond_weights_i[0] * eps_0_val
                eps_i_samples = eps_hat + sqrt_cond * z_all  # (K,)

                best_cost = np.inf
                for m_idx in range(M):
                    d_km = delta_m[m_idx] + sigma_m[m_idx] * eps_i_samples  # (K,)
                    Q_km = np.exp(-(d_km ** 2) / s_i)  # (K,)
                    R_next = R_val - w_bar_i * Q_km  # (K,)

                    R_next_c = np.clip(R_next, R_lo, R_hi)
                    eps_0_c = np.full(K_actual, np.clip(eps_0_val, eps_lo, eps_hi))
                    eps_i_c = np.clip(eps_i_samples, eps_lo, eps_hi)

                    points = np.column_stack([R_next_c, eps_0_c, eps_i_c])  # (K, 3)
                    V_vals = interp(points)  # (K,)
                    cost_m = np.mean(V_vals)

                    if cost_m < best_cost:
                        best_cost = cost_m

                V_out[r_idx, e0_idx] = best_cost

        return V_out

    elif process_idx == 2:
        # i=2: state is (R_2, eps_0, eps_1) -> output V shape (N_R, N_eps, N_eps)
        V_out = np.zeros((N_R, N_eps, N_eps))

        for r_idx in range(N_R):
            R_val = grid_R[r_idx]
            for e0_idx in range(N_eps):
                eps_0_val = grid_eps[e0_idx]
                for e1_idx in range(N_eps):
                    eps_1_val = grid_eps[e1_idx]

                    eps_less = np.array([eps_0_val, eps_1_val])
                    eps_hat = cond_weights_i @ eps_less
                    eps_i_samples = eps_hat + sqrt_cond * z_all  # (K,)

                    best_cost = np.inf
                    for m_idx in range(M):
                        d_km = delta_m[m_idx] + sigma_m[m_idx] * eps_i_samples
                        Q_km = np.exp(-(d_km ** 2) / s_i)
                        R_next = R_val - w_bar_i * Q_km

                        R_next_c = np.clip(R_next, R_lo, R_hi)
                        eps_0_c = np.full(K_actual, np.clip(eps_0_val, eps_lo, eps_hi))
                        eps_1_c = np.full(K_actual, np.clip(eps_1_val, eps_lo, eps_hi))
                        eps_i_c = np.clip(eps_i_samples, eps_lo, eps_hi)

                        points = np.column_stack([R_next_c, eps_0_c, eps_1_c, eps_i_c])
                        V_vals = interp(points)
                        cost_m = np.mean(V_vals)

                        if cost_m < best_cost:
                            best_cost = cost_m

                    V_out[r_idx, e0_idx, e1_idx] = best_cost

        return V_out


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Main backward induction
# ─────────────────────────────────────────────────────────────────────────────

def backward_induction(
    F_star: float,
    Sigma: np.ndarray,
    manifolds: List[np.ndarray],
    w_bar: np.ndarray,
    scales: np.ndarray,
    target_outputs: Dict[str, float],
    process_names: List[str],
    cfg: BellmanConfig,
    verbose: bool = True,
) -> float:
    """
    Run backward induction from i=3 (terminal) to i=0 (initial).

    Args:
        F_star: target reliability
        Sigma: (4, 4) noise covariance
        manifolds: list of 4 manifold arrays, each (M_i, 2)
        w_bar: (4,) normalised weights
        scales: (4,) scale parameters
        target_outputs: {proc_name: target_output} for adaptive targets
        process_names: ['laser', 'plasma', 'galvanic', 'microetch']
        cfg: BellmanConfig
        verbose: print progress

    Returns:
        L_min: scalar
    """
    n = 4
    grid_R, grid_eps = build_grids(F_star, cfg)

    # Precompute conditional parameters
    sigma2_cond, cond_weights = precompute_conditional_params(Sigma, n)

    if verbose:
        print(f"  Conditional variances: {[sigma2_cond[i] for i in range(n)]}")
        for i in range(1, n):
            print(f"  Conditional weights[{i}]: {cond_weights[i]}")

    # Compute adaptive targets at the operating point
    taus = []
    upstream = {}
    for i in range(n):
        tau_i = get_adaptive_target(i, upstream)
        taus.append(tau_i)
        upstream[process_names[i]] = target_outputs[process_names[i]]
    if verbose:
        print(f"  Adaptive targets: {taus}")

    rng = np.random.default_rng(42)

    # ── Step i=3 (terminal, 0-indexed): closed-form ──
    if verbose:
        print(f"\n  [Step 3/3] Terminal step (process '{process_names[3]}') — closed-form...")
    t0 = time.time()
    V3 = bellman_terminal(
        grid_R, grid_eps, manifolds[3],
        w_bar[3], scales[3], taus[3],
        sigma2_cond[3], cond_weights[3],
    )
    if verbose:
        print(f"    Done in {time.time()-t0:.1f}s. V3 shape: {V3.shape}, range: [{V3.min():.6f}, {V3.max():.6f}]")

    # ── Step i=2: MC ──
    if verbose:
        print(f"\n  [Step 2/3] Process '{process_names[2]}' — Monte Carlo (K={cfg.K_mc})...")
    t0 = time.time()
    V2 = bellman_non_terminal(
        grid_R, grid_eps, V3, manifolds[2],
        w_bar[2], scales[2], taus[2],
        sigma2_cond[2], cond_weights[2],
        process_idx=2, K=cfg.K_mc,
        use_antithetic=cfg.use_antithetic, rng=rng,
    )
    if verbose:
        print(f"    Done in {time.time()-t0:.1f}s. V2 shape: {V2.shape}, range: [{V2.min():.6f}, {V2.max():.6f}]")

    # ── Step i=1: MC ──
    if verbose:
        print(f"\n  [Step 1/3] Process '{process_names[1]}' — Monte Carlo (K={cfg.K_mc})...")
    t0 = time.time()
    V1 = bellman_non_terminal(
        grid_R, grid_eps, V2, manifolds[1],
        w_bar[1], scales[1], taus[1],
        sigma2_cond[1], cond_weights[1],
        process_idx=1, K=cfg.K_mc,
        use_antithetic=cfg.use_antithetic, rng=rng,
    )
    if verbose:
        print(f"    Done in {time.time()-t0:.1f}s. V1 shape: {V1.shape}, range: [{V1.min():.6f}, {V1.max():.6f}]")

    # ── Step i=0: MC (returns scalar) ──
    if verbose:
        print(f"\n  [Step 0/3] Process '{process_names[0]}' — Monte Carlo (K={cfg.K_mc})...")
    t0 = time.time()
    L_min = bellman_non_terminal(
        grid_R, grid_eps, V1, manifolds[0],
        w_bar[0], scales[0], taus[0],
        sigma2_cond[0], None,
        process_idx=0, K=cfg.K_mc,
        use_antithetic=cfg.use_antithetic, rng=rng,
    )
    if verbose:
        print(f"    Done in {time.time()-t0:.1f}s. L_min = {L_min:.8f}")

    return float(L_min)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2b: Naive (non-reactive) L_min
# ─────────────────────────────────────────────────────────────────────────────

def compute_naive_lmin(
    F_star: float,
    Sigma: np.ndarray,
    manifolds: List[np.ndarray],
    w_bar: np.ndarray,
    scales: np.ndarray,
    taus: List[float],
    N_mc: int = 100000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Compute naive L_min where the controller picks actions before observing noise.

    For each process, the controller chooses the action that minimises
    E[(F-F*)^2] without observing upstream outputs.

    This is an upper bound on L_min_bellman.

    Args:
        F_star: target reliability
        Sigma: (4, 4) noise covariance
        manifolds: list of 4 manifold arrays
        w_bar: (4,) normalised weights
        scales: (4,) scale parameters
        taus: list of 4 adaptive targets at operating point
        N_mc: number of MC samples
        rng: random generator

    Returns:
        L_min_naive: scalar
    """
    if rng is None:
        rng = np.random.default_rng(123)

    n = 4
    L = np.linalg.cholesky(Sigma)

    best_loss = np.inf
    best_actions = None

    # Try all combinations of actions (or a random subset if too many)
    # For efficiency, find best action per-process first, then do joint MC
    # This is a greedy heuristic but works well for the naive case

    # First: find the action per process that makes delta_i closest to 0
    # and sigma2_i smallest
    best_m_per_process = []
    for i in range(n):
        m_arr = manifolds[i]
        # Score: prefer small |delta| and small sigma2
        deltas = m_arr[:, 0] - taus[i]
        sigma2s = m_arr[:, 1]
        # Heuristic: minimise delta^2 + w_bar^2 * sigma2 / s
        scores = deltas ** 2 + w_bar[i] ** 2 * sigma2s / scales[i]
        best_idx = np.argmin(scores)
        best_m_per_process.append(best_idx)

    # MC simulation with best naive actions
    eps_samples = rng.multivariate_normal(np.zeros(n), Sigma, size=N_mc)  # (N_mc, 4)

    F_samples = np.zeros(N_mc)
    for k in range(N_mc):
        F_val = 0.0
        for i in range(n):
            m_idx = best_m_per_process[i]
            mu_i = manifolds[i][m_idx, 0]
            sigma2_i = manifolds[i][m_idx, 1]
            sigma_i = np.sqrt(max(sigma2_i, 1e-10))
            delta_i = mu_i - taus[i]
            d_i = delta_i + sigma_i * eps_samples[k, i]
            Q_i = np.exp(-(d_i ** 2) / scales[i])
            F_val += w_bar[i] * Q_i
        F_samples[k] = F_val

    loss_samples = (F_samples - F_star) ** 2
    return float(np.mean(loss_samples))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Forward simulation validation
# ─────────────────────────────────────────────────────────────────────────────

def forward_simulation(
    F_star: float,
    Sigma: np.ndarray,
    manifolds: List[np.ndarray],
    w_bar: np.ndarray,
    scales: np.ndarray,
    taus: List[float],
    grid_R: np.ndarray,
    grid_eps: np.ndarray,
    V_functions: Dict[int, np.ndarray],
    sigma2_cond: Dict[int, float],
    cond_weights: Dict[int, np.ndarray],
    N: int = 10000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """
    Forward simulation using the optimal policy from backward induction.

    For each trajectory:
    1. Sample eps ~ N(0, Sigma) (full 4-vector)
    2. At each step i, find the optimal action by looking up the policy
    3. Compute Q_i and update R
    4. Loss = R_5^2

    Args:
        F_star: target reliability
        Sigma: (4, 4) noise covariance
        manifolds: list of 4 manifold arrays
        w_bar: (4,) normalised weights
        scales: (4,) scale parameters
        taus: (4,) adaptive targets
        grid_R, grid_eps: grids from backward induction
        V_functions: {i: V_i array} for i=1,2,3
        sigma2_cond, cond_weights: conditional parameters
        N: number of trajectories
        rng: random generator

    Returns:
        L_min_forward: mean loss
        se: standard error
    """
    if rng is None:
        rng = np.random.default_rng(999)

    n = 4
    L_chol = np.linalg.cholesky(Sigma)
    R_lo, R_hi = grid_R[0], grid_R[-1]
    eps_lo, eps_hi = grid_eps[0], grid_eps[-1]

    # Build interpolators for each V function (for policy lookup)
    # V[1]: (N_R, N_eps) — used at i=1
    # V[2]: (N_R, N_eps, N_eps) — used at i=2
    # V[3]: (N_R, N_eps, N_eps, N_eps) — used at i=3
    interp_V = {}
    interp_V[1] = RegularGridInterpolator(
        (grid_R, grid_eps), V_functions[1],
        method='linear', bounds_error=False, fill_value=None,
    )
    interp_V[2] = RegularGridInterpolator(
        (grid_R, grid_eps, grid_eps), V_functions[2],
        method='linear', bounds_error=False, fill_value=None,
    )
    interp_V[3] = RegularGridInterpolator(
        (grid_R, grid_eps, grid_eps, grid_eps), V_functions[3],
        method='linear', bounds_error=False, fill_value=None,
    )

    losses = np.zeros(N)

    for j in range(N):
        # Sample noise vector
        z = rng.standard_normal(n)
        eps = L_chol @ z  # (4,)

        R = F_star
        eps_history = []

        for i in range(n):
            eps_i = eps[i]

            # Find optimal action: try all actions, pick best
            best_cost = np.inf
            best_m = 0

            mu_m = manifolds[i][:, 0]
            sigma2_m = manifolds[i][:, 1]
            sigma_m_arr = np.sqrt(np.maximum(sigma2_m, 1e-10))
            delta_m = mu_m - taus[i]

            for m_idx in range(len(mu_m)):
                d_i = delta_m[m_idx] + sigma_m_arr[m_idx] * eps_i
                Q_i = np.exp(-(d_i ** 2) / scales[i])
                R_next = R - w_bar[i] * Q_i

                if i < n - 1:
                    # Look up continuation value
                    R_c = np.clip(R_next, R_lo, R_hi)
                    eps_hist_c = [np.clip(e, eps_lo, eps_hi) for e in eps_history]
                    eps_i_c = np.clip(eps_i, eps_lo, eps_hi)

                    if i == 0:
                        point = np.array([R_c, eps_i_c])
                    elif i == 1:
                        point = np.array([R_c, eps_hist_c[0], eps_i_c])
                    elif i == 2:
                        point = np.array([R_c, eps_hist_c[0], eps_hist_c[1], eps_i_c])

                    v_next = interp_V[i + 1](point.reshape(1, -1))[0]
                else:
                    # Terminal: cost is R_next^2
                    v_next = R_next ** 2

                if v_next < best_cost:
                    best_cost = v_next
                    best_m = m_idx

            # Apply best action
            d_i = delta_m[best_m] + sigma_m_arr[best_m] * eps_i
            Q_i = np.exp(-(d_i ** 2) / scales[i])
            R = R - w_bar[i] * Q_i
            eps_history.append(eps_i)

        losses[j] = R ** 2

    L_min_forward = float(np.mean(losses))
    se = float(np.std(losses) / np.sqrt(N))
    return L_min_forward, se


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def compute_bellman_lmin(
    process_chain,
    surrogate,
    cfg: Optional[BellmanConfig] = None,
    loss_scale: float = 1.0,
    scenario_idx: int = 0,
    verbose: bool = True,
) -> BellmanLminResult:
    """
    Compute L_min via Bellman backward induction.

    This is the main entry point. It:
    1. Estimates Sigma from the process chain
    2. Computes manifolds M_i for each process
    3. Runs backward induction
    4. Validates with forward simulation
    5. Computes naive L_min for comparison

    Args:
        process_chain: Trained ProcessChain
        surrogate: ProTSurrogate with F_star
        cfg: BellmanConfig (uses defaults if None)
        loss_scale: Scale factor (default 1.0, set to reliability_loss_scale for comparison)
        scenario_idx: Which scenario to use
        verbose: Print progress

    Returns:
        BellmanLminResult with all computed values
    """
    if cfg is None:
        cfg = BellmanConfig()

    t_start = time.time()
    n = 4
    process_names = process_chain.process_names

    F_star = surrogate.F_star

    # ── Extract weights and scales ──
    from controller_optimization.src.models.surrogate import ProTSurrogate
    proc_configs = ProTSurrogate.PROCESS_CONFIGS

    weights = np.array([proc_configs[name]['weight'] for name in process_names])
    w_bar = weights / weights.sum()
    scales = np.array([proc_configs[name]['scale'] for name in process_names])

    if verbose:
        print(f"\n{'='*70}")
        print(f"BELLMAN L_min COMPUTATION")
        print(f"{'='*70}")
        print(f"  F* = {F_star:.6f}")
        print(f"  Weights (raw):  {weights}")
        print(f"  Weights (norm): {w_bar}")
        print(f"  Scales:         {scales}")
        print(f"  Grid: N_R={cfg.N_R}, N_eps={cfg.N_eps}, K_mc={cfg.K_mc}")

    # ── Phase 1: Estimate Sigma ──
    if verbose:
        print(f"\n[Phase 1] Estimating noise covariance Sigma...")
    Sigma = estimate_noise_covariance(
        process_chain, n_samples=2000, scenario_idx=scenario_idx,
        shrinkage=cfg.sigma_shrinkage,
    )
    if verbose:
        print(f"  Sigma diagonal: {np.diag(Sigma)}")
        print(f"  Sigma off-diag max: {np.max(np.abs(Sigma - np.diag(np.diag(Sigma)))):.4f}")
        print(f"  Sigma eigenvalues: {np.linalg.eigvalsh(Sigma)}")

    # ── Phase 1: Compute manifolds ──
    if verbose:
        print(f"\n[Phase 1] Computing achievable manifolds M_i...")
    manifolds = []
    for i in range(n):
        M_i = compute_manifold(
            process_chain, i, scenario_idx=scenario_idx, n_actions=cfg.M_actions,
        )
        manifolds.append(M_i)
        if verbose:
            print(f"  {process_names[i]}: {M_i.shape[0]} points, "
                  f"mu range [{M_i[:,0].min():.3f}, {M_i[:,0].max():.3f}], "
                  f"sigma2 range [{M_i[:,1].min():.6f}, {M_i[:,1].max():.6f}]")

    # ── Get target outputs for adaptive targets ──
    target_outputs = get_target_outputs(process_chain, scenario_idx)
    if verbose:
        print(f"  Target outputs: {target_outputs}")

    # ── Phase 2: Backward induction ──
    if verbose:
        print(f"\n[Phase 2] Running backward induction...")
    L_min_bellman = backward_induction(
        F_star, Sigma, manifolds, w_bar, scales,
        target_outputs, process_names, cfg, verbose,
    )
    L_min_bellman_scaled = L_min_bellman * loss_scale

    if verbose:
        print(f"\n  L_min (Bellman) = {L_min_bellman:.8f}")
        if loss_scale != 1.0:
            print(f"  L_min (scaled) = {L_min_bellman_scaled:.6f}")

    # ── Phase 2b: Naive L_min ──
    if verbose:
        print(f"\n[Phase 2b] Computing naive (non-reactive) L_min...")

    # Compute taus at operating point
    taus = []
    upstream = {}
    for i in range(n):
        tau_i = get_adaptive_target(i, upstream)
        taus.append(tau_i)
        upstream[process_names[i]] = target_outputs[process_names[i]]

    L_min_naive = compute_naive_lmin(
        F_star, Sigma, manifolds, w_bar, scales, taus,
    )
    L_min_naive_scaled = L_min_naive * loss_scale

    if verbose:
        print(f"  L_min (naive) = {L_min_naive:.8f}")
        print(f"  Bellman advantage: {(L_min_naive - L_min_bellman) / L_min_naive * 100:.1f}%")

    # ── Phase 3: Forward validation ──
    if verbose:
        print(f"\n[Phase 3] Forward simulation validation (N={cfg.N_forward})...")

    grid_R, grid_eps = build_grids(F_star, cfg)
    sigma2_cond, cond_wts = precompute_conditional_params(Sigma, n)

    # Reconstruct V functions for forward simulation
    # Re-run terminal step (fast, closed-form)
    V3 = bellman_terminal(
        grid_R, grid_eps, manifolds[3],
        w_bar[3], scales[3], taus[3],
        sigma2_cond[3], cond_wts[3],
    )

    # For forward sim, we need V1, V2, V3
    # V2 and V1 were computed during backward_induction but not saved
    # We reconstruct V2 by re-running step i=2
    rng_fwd = np.random.default_rng(42)
    V2 = bellman_non_terminal(
        grid_R, grid_eps, V3, manifolds[2],
        w_bar[2], scales[2], taus[2],
        sigma2_cond[2], cond_wts[2],
        process_idx=2, K=cfg.K_mc,
        use_antithetic=cfg.use_antithetic, rng=rng_fwd,
    )
    V1 = bellman_non_terminal(
        grid_R, grid_eps, V2, manifolds[1],
        w_bar[1], scales[1], taus[1],
        sigma2_cond[1], cond_wts[1],
        process_idx=1, K=cfg.K_mc,
        use_antithetic=cfg.use_antithetic, rng=rng_fwd,
    )

    V_functions = {1: V1, 2: V2, 3: V3}

    L_min_forward, L_min_forward_se = forward_simulation(
        F_star, Sigma, manifolds, w_bar, scales, taus,
        grid_R, grid_eps, V_functions,
        sigma2_cond, cond_wts,
        N=cfg.N_forward,
    )
    L_min_forward_scaled = L_min_forward * loss_scale
    L_min_forward_se_scaled = L_min_forward_se * loss_scale

    if verbose:
        print(f"  L_min (forward) = {L_min_forward:.8f} ± {L_min_forward_se:.8f}")
        rel_diff = abs(L_min_forward - L_min_bellman) / max(abs(L_min_bellman), 1e-10)
        print(f"  Relative difference Bellman vs Forward: {rel_diff*100:.2f}%")

    # ── Summary ──
    computation_time = time.time() - t_start
    if verbose:
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"  L_min (Bellman):  {L_min_bellman:.8f}" +
              (f"  (scaled: {L_min_bellman_scaled:.6f})" if loss_scale != 1.0 else ""))
        print(f"  L_min (forward):  {L_min_forward:.8f} ± {L_min_forward_se:.8f}" +
              (f"  (scaled: {L_min_forward_scaled:.6f})" if loss_scale != 1.0 else ""))
        print(f"  L_min (naive):    {L_min_naive:.8f}" +
              (f"  (scaled: {L_min_naive_scaled:.6f})" if loss_scale != 1.0 else ""))
        print(f"  Bellman <= Naive? {L_min_bellman <= L_min_naive + 1e-8}")
        print(f"  Forward ≈ Bellman? |diff| = {abs(L_min_forward - L_min_bellman):.8f}")
        print(f"  Computation time: {computation_time:.1f}s")
        print(f"{'='*70}\n")

    return BellmanLminResult(
        L_min_bellman=L_min_bellman_scaled if loss_scale != 1.0 else L_min_bellman,
        L_min_forward=L_min_forward_scaled if loss_scale != 1.0 else L_min_forward,
        L_min_forward_se=L_min_forward_se_scaled if loss_scale != 1.0 else L_min_forward_se,
        L_min_naive=L_min_naive_scaled if loss_scale != 1.0 else L_min_naive,
        F_star=F_star,
        Sigma=Sigma,
        w_bar=w_bar,
        scales=scales,
        n_manifold_points=[m.shape[0] for m in manifolds],
        computation_time_s=computation_time,
    )
