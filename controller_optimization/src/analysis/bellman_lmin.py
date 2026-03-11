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
    # Grid sizes — memory budget for terminal step:
    #   (N_R × N_eps³ × M) elements × 8 bytes
    #   N_R=50, N_eps=16, M=50 → 102.4M elements ≈ 820 MB (peak, transient)
    #   N_R=50, N_eps=20, M=50 → 200M elements ≈ 1.6 GB (approaching limit)
    N_R: int = 50           # Grid points for remaining reliability R
    N_eps: int = 16         # Grid points per noise dimension (8 too coarse)
    eps_range: float = 3.0  # Noise range: [-eps_range, +eps_range]
    R_min: float = -0.1     # Lower bound of R grid
    # R_max is set to F* dynamically

    # Manifold resolution
    M_actions: int = 50     # Number of action candidates per process

    # Monte Carlo
    K_mc: int = 500         # MC samples for non-terminal steps
    use_antithetic: bool = True  # Antithetic variates

    # Forward validation
    N_forward: int = 5000   # Forward simulation trajectories

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

    Uses a single batched forward pass for efficiency.

    Args:
        process_chain: ProcessChain with trained UPs
        n_samples: Number of samples to collect
        scenario_idx: Scenario to use
        shrinkage: Ledoit-Wolf shrinkage parameter

    Returns:
        Sigma: (4, 4) positive-definite covariance matrix
    """
    n_processes = len(process_chain.process_names)
    residuals = np.zeros((n_samples, n_processes))

    process_chain.eval()
    with torch.no_grad():
        # Single batched forward pass (much faster than n_samples individual calls)
        trajectory = process_chain.forward(batch_size=n_samples, scenario_idx=scenario_idx)

        for i, proc_name in enumerate(process_chain.process_names):
            data = trajectory[proc_name]
            mu = data['outputs_mean'].squeeze(-1).cpu().numpy()        # (n_samples,)
            var = data['outputs_var'].squeeze(-1).cpu().numpy()        # (n_samples,)
            sigma = np.sqrt(var + 1e-8)                                # (n_samples,)
            o_sampled = data['outputs_sampled'].squeeze(-1).cpu().numpy()  # (n_samples,)
            residuals[:, i] = (o_sampled - mu) / sigma

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

    # Build all full inputs at once (batched, much faster than looping)
    n_grid = len(grid_values)
    all_inputs = np.tile(np.array(target_inputs, dtype=np.float64), (n_grid, 1))
    for out_idx, input_idx in enumerate(ctrl_indices):
        all_inputs[:, input_idx] = grid_values[:, out_idx]

    process_chain.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(all_inputs, dtype=torch.float32).to(process_chain.device)

        # Single batched UP call
        scaled = process_chain.scale_inputs(input_tensor, process_idx)
        mu_scaled, var_scaled = process_chain.uncertainty_predictors[process_idx](scaled)
        mu = process_chain.unscale_outputs(mu_scaled, process_idx)
        var = process_chain.unscale_variance(var_scaled, process_idx)

    manifold = np.column_stack([
        mu.squeeze(-1).cpu().numpy(),
        var.squeeze(-1).cpu().numpy(),
    ])  # (M, 2): [mu, sigma2]
    return manifold


def get_adaptive_target(process_idx: int, upstream_outputs: Optional[Dict[str, float]] = None,
                        surrogate=None) -> float:
    """
    Get the adaptive target for process i given upstream outputs.

    τ_i = base_target_i + β × (Y_{i-1} - τ_{i-1})

    Args:
        process_idx: indice del processo nella catena
        upstream_outputs: dict {proc_name: output_value} for completed processes
        surrogate: ProTSurrogate instance (opzionale, per accedere a _dynamic_configs)

    Returns:
        target value tau_i
    """
    if upstream_outputs is None:
        upstream_outputs = {}

    # GENERIC PATH: usa target calibrati dal surrogate (processi ST)
    if surrogate is not None and hasattr(surrogate, '_dynamic_configs') and surrogate._dynamic_configs is not None:
        proc_names = list(surrogate._dynamic_configs.keys()) if surrogate._process_order is None else surrogate._process_order
        beta = getattr(surrogate, '_beta', 0.0)

        # Compute targets sequentially up to process_idx
        prev_target = None
        prev_output = None
        for idx, pname in enumerate(proc_names):
            cfg = surrogate._dynamic_configs[pname]
            base_target = cfg['target']

            if prev_output is not None and prev_target is not None and beta != 0.0:
                target = base_target + beta * (prev_output - prev_target)
            else:
                target = base_target

            if idx == process_idx:
                return target

            prev_target = target
            prev_output = upstream_outputs.get(pname, base_target)

        return 0.0

    # LEGACY PATH: processi fisici con target fissi (no adattamento)
    legacy_targets = [0.8, 3.0, 10.0, 20.0]
    if process_idx < len(legacy_targets):
        return legacy_targets[process_idx]
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
    Closed-form Bellman for the terminal step (last process).

    V_T(R_T, eps_{<T}) = min_{(mu,sigma2) in M_T} E[ (R_T - w_bar_T * Q_T)^2 | eps_{<T} ]

    Generic for any number of preceding processes (n_prev = len(cond_weights_i)).

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
        cond_weights_i: (n_prev,) conditional mean weights

    Returns:
        V: array of shape (N_R, N_eps, ..., N_eps) with n_prev eps dimensions
    """
    N_R = len(grid_R)
    N_eps = len(grid_eps)
    M = manifold.shape[0]
    n_prev = len(cond_weights_i)

    # Extract manifold columns
    mu_m = manifold[:, 0]       # (M,)
    sigma2_m = manifold[:, 1]   # (M,)
    sigma_m = np.sqrt(np.maximum(sigma2_m, 1e-10))  # (M,)

    # Compute delta_m = mu_m - tau_i for each action
    delta_m = mu_m - tau_i  # (M,)

    # Build meshgrid for eps_{<T} with n_prev dimensions
    # eps_hat = sum_j cond_weights_i[j] * e_j
    grids = np.meshgrid(*([grid_eps] * n_prev), indexing='ij')  # list of n_prev arrays, each (N_eps,)*n_prev
    eps_hat = sum(cond_weights_i[j] * grids[j] for j in range(n_prev))
    # Shape: (N_eps,) * n_prev

    sqrt_cond = np.sqrt(sigma2_cond_i)

    # Vectorise over (R, eps_dims..., m)
    # n_total_dims = 1 (R) + n_prev (eps) + 1 (m) = n_prev + 2
    # R:       (N_R, 1, 1, ..., 1)  — n_prev+1 trailing dims
    # eps_hat: (1, N_eps, ..., N_eps, 1)  — leading 1 + n_prev eps dims + trailing 1
    # delta_m: (1, 1, ..., 1, M)  — n_prev+1 leading dims

    R_shape = (N_R,) + (1,) * (n_prev + 1)
    R = grid_R.reshape(R_shape)

    eps_hat_shape = (1,) + eps_hat.shape + (1,)
    eps_hat_nd = eps_hat.reshape(eps_hat_shape)

    trailing_shape = (1,) * (n_prev + 1) + (M,)
    delta_nd = delta_m.reshape(trailing_shape)
    sigma_nd = sigma_m.reshape(trailing_shape)

    d = delta_nd + sigma_nd * eps_hat_nd  # (N_R, Ne, ..., Ne, M)

    beta = sigma_nd * sqrt_cond
    beta2 = beta ** 2

    # E[Q]
    denom1 = s_i + 2 * beta2
    EQ = np.exp(-d ** 2 / denom1) / np.sqrt(1 + 2 * beta2 / s_i)

    # E[Q^2]
    denom2 = s_i + 4 * beta2
    EQ2 = np.exp(-2 * d ** 2 / denom2) / np.sqrt(1 + 4 * beta2 / s_i)

    # cost(R, eps, m) = R^2 - 2*R*w*EQ + w^2*EQ2
    cost = R ** 2 - 2 * R * w_bar_i * EQ + w_bar_i ** 2 * EQ2

    # Minimise over actions (last axis)
    V = np.min(cost, axis=-1)  # (N_R, N_eps, ..., N_eps)

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

    # Number of preceding eps dimensions in V_next
    # V_next has shape (N_R, N_eps, ..., N_eps) with (process_idx + 1) eps dims
    n_eps_dims_next = len(V_next.shape) - 1  # subtract R dimension

    # Build interpolator for V_next — generic for any number of eps dims
    interp_axes = (grid_R,) + (grid_eps,) * n_eps_dims_next
    interp = RegularGridInterpolator(
        interp_axes, V_next,
        method='linear', bounds_error=False, fill_value=None,
    )

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
        # i=0: state is (F*, empty) — single point, returns scalar
        eps_i_samples = sqrt_cond * z_all  # (K,)
        F_star = grid_R[-1]

        d_all = delta_m[:, None] + sigma_m[:, None] * eps_i_samples[None, :]  # (M, K)
        Q_all = np.exp(-(d_all ** 2) / s_i)
        R_next_all = F_star - w_bar_i * Q_all

        R_next_flat = np.clip(R_next_all.ravel(), R_lo, R_hi)
        eps_flat = np.tile(np.clip(eps_i_samples, eps_lo, eps_hi), M)

        points = np.column_stack([R_next_flat, eps_flat])
        V_vals = interp(points).reshape(M, K_actual)
        costs = V_vals.mean(axis=1)

        return float(costs.min())

    else:
        # Generic non-terminal step for process_idx >= 1
        # State: (R, eps_0, ..., eps_{i-1}) → output shape (N_R, N_eps, ..., N_eps) with i eps dims
        n_prev = process_idx  # number of preceding eps dimensions
        V_out_shape = (N_R,) + (N_eps,) * n_prev
        V_out = np.zeros(V_out_shape)

        # Iterate over all combinations of preceding eps values
        for multi_idx in np.ndindex(*([N_eps] * n_prev)):
            eps_vals = np.array([grid_eps[idx] for idx in multi_idx])

            # Conditional mean: eps_hat = cond_weights @ eps_vals
            eps_hat = cond_weights_i @ eps_vals
            eps_i_samples = eps_hat + sqrt_cond * z_all  # (K,)

            # d, Q independent of R: (M, K)
            d_all = delta_m[:, None] + sigma_m[:, None] * eps_i_samples[None, :]
            Q_all = np.exp(-(d_all ** 2) / s_i)

            # Vectorize over all R grid points: (N_R, M, K)
            R_next = grid_R[:, None, None] - w_bar_i * Q_all[None, :, :]

            R_flat = np.clip(R_next.ravel(), R_lo, R_hi)  # (N_R*M*K,)
            # Build interpolation points: [R, eps_0, ..., eps_{i-1}, eps_i]
            n_points = N_R * M * K_actual
            cols = [R_flat]
            for j in range(n_prev):
                cols.append(np.full(n_points, np.clip(eps_vals[j], eps_lo, eps_hi)))
            cols.append(np.tile(np.clip(eps_i_samples, eps_lo, eps_hi), N_R * M))

            points = np.column_stack(cols)
            V_vals = interp(points).reshape(N_R, M, K_actual)
            costs = V_vals.mean(axis=2)  # (N_R, M)

            V_out[(slice(None),) + multi_idx] = costs.min(axis=1)  # (N_R,)

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
    **kwargs,
) -> Tuple[float, Dict[int, np.ndarray], np.ndarray, np.ndarray, Dict[int, float], Dict[int, np.ndarray], List[float]]:
    """
    Run backward induction from terminal process to initial.

    Args:
        F_star: target reliability
        Sigma: (n, n) noise covariance
        manifolds: list of n manifold arrays, each (M_i, 2)
        w_bar: (n,) normalised weights
        scales: (n,) scale parameters
        target_outputs: {proc_name: target_output} for adaptive targets
        process_names: list of process names
        cfg: BellmanConfig
        verbose: print progress
        **kwargs: optional 'surrogate' for dynamic target configs (ST processes)

    Returns:
        Tuple of (L_min, V_functions, grid_R, grid_eps, sigma2_cond, cond_weights, taus)
    """
    n = len(process_names)
    grid_R, grid_eps = build_grids(F_star, cfg)

    # Precompute conditional parameters
    sigma2_cond, cond_weights = precompute_conditional_params(Sigma, n)

    if verbose:
        print(f"  Conditional variances: {[sigma2_cond[i] for i in range(n)]}")
        for i in range(1, n):
            print(f"  Conditional weights[{i}]: {cond_weights[i]}")

    # Compute adaptive targets at the operating point
    # surrogate viene passato via kwargs per supportare target dinamici (ST)
    surrogate = kwargs.get('surrogate', None)
    taus = []
    upstream = {}
    for i in range(n):
        tau_i = get_adaptive_target(i, upstream, surrogate=surrogate)
        taus.append(tau_i)
        upstream[process_names[i]] = target_outputs[process_names[i]]
    if verbose:
        print(f"  Adaptive targets: {taus}")

    rng = np.random.default_rng(42)

    # ── Backward induction: da terminal (i=n-1) a i=0 ──
    V_functions = {}

    # Step terminale (i = n-1): closed-form
    terminal_idx = n - 1
    if verbose:
        print(f"\n  [Step {n-1}/{n-1}] Terminal step (process '{process_names[terminal_idx]}') — closed-form...")
    t0 = time.time()
    V_prev = bellman_terminal(
        grid_R, grid_eps, manifolds[terminal_idx],
        w_bar[terminal_idx], scales[terminal_idx], taus[terminal_idx],
        sigma2_cond[terminal_idx], cond_weights[terminal_idx],
    )
    V_functions[terminal_idx] = V_prev
    if verbose:
        print(f"    Done in {time.time()-t0:.1f}s. V shape: {V_prev.shape}, range: [{V_prev.min():.6f}, {V_prev.max():.6f}]")

    # Steps intermedi (i = n-2 ... 1): MC
    for i in range(n - 2, 0, -1):
        if verbose:
            print(f"\n  [Step {i}/{n-1}] Process '{process_names[i]}' — Monte Carlo (K={cfg.K_mc})...")
        t0 = time.time()
        V_prev = bellman_non_terminal(
            grid_R, grid_eps, V_prev, manifolds[i],
            w_bar[i], scales[i], taus[i],
            sigma2_cond[i], cond_weights[i],
            process_idx=i, K=cfg.K_mc,
            use_antithetic=cfg.use_antithetic, rng=rng,
        )
        V_functions[i] = V_prev
        if verbose:
            print(f"    Done in {time.time()-t0:.1f}s. V shape: {V_prev.shape}, range: [{V_prev.min():.6f}, {V_prev.max():.6f}]")

    # Step iniziale (i = 0): MC, ritorna scalare
    if verbose:
        print(f"\n  [Step 0/{n-1}] Process '{process_names[0]}' — Monte Carlo (K={cfg.K_mc})...")
    t0 = time.time()
    L_min = bellman_non_terminal(
        grid_R, grid_eps, V_prev, manifolds[0],
        w_bar[0], scales[0], taus[0],
        sigma2_cond[0], None,
        process_idx=0, K=cfg.K_mc,
        use_antithetic=cfg.use_antithetic, rng=rng,
    )
    if verbose:
        print(f"    Done in {time.time()-t0:.1f}s. L_min = {L_min:.8f}")

    return float(L_min), V_functions, grid_R, grid_eps, sigma2_cond, cond_weights, taus



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

    Vectorized over all N trajectories: for each step i, we evaluate all M
    actions for all N trajectories in one batch interpolation call, then
    pick the best action per trajectory.

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

    n = len(manifolds)
    L_chol = np.linalg.cholesky(Sigma)
    R_lo, R_hi = grid_R[0], grid_R[-1]
    eps_lo, eps_hi = grid_eps[0], grid_eps[-1]

    # Build interpolators for each V function (for policy lookup)
    interp_V = {}
    for vi_idx, V_arr in V_functions.items():
        # V_i ha dimensione (N_R, N_eps, ..., N_eps) con vi_idx assi eps
        grid_axes = [grid_R] + [grid_eps] * vi_idx
        interp_V[vi_idx] = RegularGridInterpolator(
            tuple(grid_axes), V_arr,
            method='linear', bounds_error=False, fill_value=None,
        )

    # Sample all noise vectors at once: (N, n)
    Z = rng.standard_normal((N, n))
    EPS = Z @ L_chol.T  # (N, n)

    # State vectors — all trajectories in parallel
    R = np.full(N, F_star)                   # (N,)
    eps_hist = np.empty((N, 0))              # grows to (N, i) at step i

    for i in range(n):
        eps_i = EPS[:, i]                    # (N,)
        M_i = manifolds[i].shape[0]
        mu_m = manifolds[i][:, 0]            # (M,)
        sigma2_m = manifolds[i][:, 1]
        sigma_m = np.sqrt(np.maximum(sigma2_m, 1e-10))
        delta_m = mu_m - taus[i]

        # d[m, j] = delta_m[m] + sigma_m[m] * eps_i[j]  — shape (M, N)
        d_all = delta_m[:, None] + sigma_m[:, None] * eps_i[None, :]
        Q_all = np.exp(-(d_all ** 2) / scales[i])       # (M, N)
        R_next_all = R[None, :] - w_bar[i] * Q_all      # (M, N)

        if i < n - 1:
            # Build interpolation points for all M actions × N trajectories
            R_c = np.clip(R_next_all, R_lo, R_hi)        # (M, N)
            eps_i_c = np.clip(eps_i, eps_lo, eps_hi)      # (N,)

            R_flat = R_c.ravel()                          # (M*N,)
            eps_i_flat = np.tile(eps_i_c, M_i)            # (M*N,)

            # Costruisci colonne interpolazione: [R, eps_0, eps_1, ..., eps_i]
            cols = [R_flat]
            for h in range(eps_hist.shape[1]):
                cols.append(np.tile(np.clip(eps_hist[:, h], eps_lo, eps_hi), M_i))
            cols.append(eps_i_flat)
            points = np.column_stack(cols)

            v_next = interp_V[i + 1](points).reshape(M_i, N)  # (M, N)
        else:
            # Terminal: cost is R_next^2
            v_next = R_next_all ** 2                      # (M, N)

        # Pick best action per trajectory
        best_m = np.argmin(v_next, axis=0)                # (N,)

        # Apply chosen actions
        d_best = d_all[best_m, np.arange(N)]
        Q_best = np.exp(-(d_best ** 2) / scales[i])
        R = R - w_bar[i] * Q_best

        eps_hist = np.column_stack([eps_hist, eps_i]) if eps_hist.size > 0 else eps_i[:, None]

    losses = R ** 2
    L_min_forward = float(np.mean(losses))
    se = float(np.std(losses) / np.sqrt(N))
    return L_min_forward, se


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3b: Baseline forward simulations (policy discrimination)
# ─────────────────────────────────────────────────────────────────────────────

def forward_simulation_greedy(
    F_star: float,
    Sigma: np.ndarray,
    manifolds: List[np.ndarray],
    w_bar: np.ndarray,
    scales: np.ndarray,
    taus: List[float],
    N: int = 10000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """
    Forward simulation with greedy (myopic) policy: minimise R_next² at each step.

    No V-function lookup — just picks the action that minimises immediate R².
    If this matches the V-policy result, then V functions are not adding value
    (the problem has excess control authority).
    """
    if rng is None:
        rng = np.random.default_rng(777)

    n = len(manifolds)
    L_chol = np.linalg.cholesky(Sigma)
    Z = rng.standard_normal((N, n))
    EPS = Z @ L_chol.T  # (N, n)

    R = np.full(N, F_star)

    for i in range(n):
        eps_i = EPS[:, i]
        mu_m = manifolds[i][:, 0]
        sigma2_m = manifolds[i][:, 1]
        sigma_m = np.sqrt(np.maximum(sigma2_m, 1e-10))
        delta_m = mu_m - taus[i]

        d_all = delta_m[:, None] + sigma_m[:, None] * eps_i[None, :]  # (M, N)
        Q_all = np.exp(-(d_all ** 2) / scales[i])
        R_next_all = R[None, :] - w_bar[i] * Q_all  # (M, N)

        # Greedy: pick action minimising R_next²
        best_m = np.argmin(R_next_all ** 2, axis=0)  # (N,)

        d_best = d_all[best_m, np.arange(N)]
        Q_best = np.exp(-(d_best ** 2) / scales[i])
        R = R - w_bar[i] * Q_best

    losses = R ** 2
    return float(np.mean(losses)), float(np.std(losses) / np.sqrt(N))


def forward_simulation_random(
    F_star: float,
    Sigma: np.ndarray,
    manifolds: List[np.ndarray],
    w_bar: np.ndarray,
    scales: np.ndarray,
    taus: List[float],
    N: int = 10000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """
    Forward simulation with uniformly random action selection.

    Baseline: if this also gives loss ≈ 0, the problem has so much excess
    control authority that no policy can fail.
    """
    if rng is None:
        rng = np.random.default_rng(888)

    n = len(manifolds)
    L_chol = np.linalg.cholesky(Sigma)
    Z = rng.standard_normal((N, n))
    EPS = Z @ L_chol.T  # (N, n)

    R = np.full(N, F_star)

    for i in range(n):
        eps_i = EPS[:, i]
        M_i = manifolds[i].shape[0]
        mu_m = manifolds[i][:, 0]
        sigma2_m = manifolds[i][:, 1]
        sigma_m = np.sqrt(np.maximum(sigma2_m, 1e-10))
        delta_m = mu_m - taus[i]

        # Random action per trajectory
        random_m = rng.integers(0, M_i, size=N)
        d = delta_m[random_m] + sigma_m[random_m] * eps_i
        Q = np.exp(-(d ** 2) / scales[i])
        R = R - w_bar[i] * Q

    losses = R ** 2
    return float(np.mean(losses)), float(np.std(losses) / np.sqrt(N))


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
    process_names = process_chain.process_names
    n = len(process_names)

    F_star = surrogate.F_star

    # ── Extract weights and scales ──
    # Usa _dynamic_configs dal surrogate se disponibili (ST), altrimenti PROCESS_CONFIGS
    if hasattr(surrogate, '_dynamic_configs') and surrogate._dynamic_configs is not None:
        proc_configs = surrogate._dynamic_configs
    else:
        from controller_optimization.src.models.surrogate import ProTSurrogate
        proc_configs = {name: {'target': t, 'scale': s, 'weight': 1.0}
                        for name, t, s in ProTSurrogate.LEGACY_CONFIGS}

    weights = np.array([proc_configs[name].get('weight', 1.0) for name in process_names])
    w_bar = weights / weights.sum()
    scales = np.array([proc_configs[name]['scale'] for name in process_names])

    budget_target_ratio = w_bar.sum() / F_star

    if verbose:
        print(f"\n{'='*70}")
        print(f"BELLMAN L_min COMPUTATION")
        print(f"{'='*70}")
        print(f"  F* = {F_star:.6f}")
        print(f"  Weights (raw):  {weights}")
        print(f"  Weights (norm): {w_bar}")
        print(f"  Scales:         {scales}")
        print(f"  Budget/Target ratio: {budget_target_ratio:.2f}x "
              f"(Σw̄={w_bar.sum():.3f} / F*={F_star:.3f})")
        if budget_target_ratio > 2.0:
            print(f"  ⚠ High ratio — reactive controller has excess authority; "
                  f"forward validation may be uninformative")
        print(f"  Grid: N_R={cfg.N_R}, N_eps={cfg.N_eps}, K_mc={cfg.K_mc}")

    # ── Phase 1: Estimate Sigma ──
    if verbose:
        print(f"\n[Phase 1a] Estimating noise covariance Sigma (n_samples=1000)...")
    t_phase = time.time()
    Sigma = estimate_noise_covariance(
        process_chain, n_samples=1000, scenario_idx=scenario_idx,
        shrinkage=cfg.sigma_shrinkage,
    )
    if verbose:
        print(f"  Done in {time.time()-t_phase:.1f}s")
        print(f"  Sigma diagonal: {np.diag(Sigma)}")
        print(f"  Sigma off-diag max: {np.max(np.abs(Sigma - np.diag(np.diag(Sigma)))):.4f}")
        print(f"  Sigma eigenvalues: {np.linalg.eigvalsh(Sigma)}")

    # ── Phase 1b: Compute manifolds ──
    if verbose:
        print(f"\n[Phase 1b] Computing achievable manifolds M_i (n_actions={cfg.M_actions})...")
    t_phase = time.time()
    manifolds = []
    for i in range(n):
        M_i = compute_manifold(
            process_chain, i, scenario_idx=scenario_idx, n_actions=cfg.M_actions,
        )
        manifolds.append(M_i)
        if verbose:
            print(f"  {process_names[i]:12s}: {M_i.shape[0]:3d} points, "
                  f"mu=[{M_i[:,0].min():.3f}, {M_i[:,0].max():.3f}], "
                  f"sigma2=[{M_i[:,1].min():.6f}, {M_i[:,1].max():.6f}]")
    if verbose:
        print(f"  Done in {time.time()-t_phase:.1f}s")

    # ── Get target outputs for adaptive targets ──
    target_outputs = get_target_outputs(process_chain, scenario_idx)
    if verbose:
        print(f"  Target outputs: {target_outputs}")

    # ── Phase 2: Backward induction ──
    if verbose:
        print(f"\n[Phase 2] Running backward induction...")
    (L_min_bellman, V_functions, grid_R, grid_eps,
     sigma2_cond, cond_wts, taus) = backward_induction(
        F_star, Sigma, manifolds, w_bar, scales,
        target_outputs, process_names, cfg, verbose,
        surrogate=surrogate,
    )
    L_min_bellman_scaled = L_min_bellman * loss_scale

    if verbose:
        print(f"\n  L_min (Bellman) = {L_min_bellman:.8f}")
        if loss_scale != 1.0:
            print(f"  L_min (scaled) = {L_min_bellman_scaled:.6f}")

    # ── Phase 3: Forward validation ──
    # V_functions, grid_R, grid_eps, sigma2_cond, cond_wts already
    # returned from backward_induction — no recomputation needed.
    if verbose:
        print(f"\n[Phase 3] Forward simulation validation (N={cfg.N_forward})...")
    t_phase = time.time()

    L_min_forward, L_min_forward_se = forward_simulation(
        F_star, Sigma, manifolds, w_bar, scales, taus,
        grid_R, grid_eps, V_functions,
        sigma2_cond, cond_wts,
        N=cfg.N_forward,
    )
    L_min_forward_scaled = L_min_forward * loss_scale
    L_min_forward_se_scaled = L_min_forward_se * loss_scale

    if verbose:
        print(f"  Done in {time.time()-t_phase:.1f}s")
        print(f"  L_min (forward) = {L_min_forward:.8f} ± {L_min_forward_se:.8f}")
        rel_diff = abs(L_min_forward - L_min_bellman) / max(abs(L_min_bellman), 1e-10)
        print(f"  Relative difference Bellman vs Forward: {rel_diff*100:.2f}%")
        if rel_diff > 0.5:
            print(f"  NOTE: Large forward/Bellman gap — may indicate coarse grid discretisation")

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
        print(f"  Forward ≈ Bellman? |diff| = {abs(L_min_forward - L_min_bellman):.8f}")
        print(f"  Computation time: {computation_time:.1f}s")
        print(f"{'='*70}\n")

    return BellmanLminResult(
        L_min_bellman=L_min_bellman_scaled if loss_scale != 1.0 else L_min_bellman,
        L_min_forward=L_min_forward_scaled if loss_scale != 1.0 else L_min_forward,
        L_min_forward_se=L_min_forward_se_scaled if loss_scale != 1.0 else L_min_forward_se,
        F_star=F_star,
        Sigma=Sigma,
        w_bar=w_bar,
        scales=scales,
        n_manifold_points=[m.shape[0] for m in manifolds],
        computation_time_s=computation_time,
    )
