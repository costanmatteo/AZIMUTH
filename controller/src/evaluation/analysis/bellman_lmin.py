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
from concurrent.futures import ThreadPoolExecutor
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
    #   N_R=200, N_eps=30, M=100 → 540M elements ≈ 4.3 GB (peak, transient)
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
    N_forward: int = 5000   # Forward simulation trajectories

    # Modelling fidelity level
    level: int = 3              # 1 = fixed σ², Σ=I
                                # 2 = action-dependent σ², Σ=I
                                # 3 = action-dependent σ², full Σ (default)
    parallel_levels: bool = False  # If True, compute all 3 levels in parallel

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
    level: int = 3              # Modelling fidelity level used
    level_results: Optional[Dict[int, Dict[str, Any]]] = None  # Per-level results when parallel_levels=True

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'L_min_bellman': self.L_min_bellman,
            'L_min_forward': self.L_min_forward,
            'L_min_forward_se': self.L_min_forward_se,
            'F_star': self.F_star,
            'Sigma': self.Sigma.tolist(),
            'w_bar': self.w_bar.tolist(),
            'scales': self.scales.tolist(),
            'n_manifold_points': self.n_manifold_points,
            'computation_time_s': self.computation_time_s,
            'level': self.level,
        }
        if self.level_results is not None:
            d['level_results'] = self.level_results
        return d

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

    Uses **fixed actions from the target trajectory** (a*_i) for each process,
    so that residual variation reflects only intrinsic process noise — not
    controller reactivity. This is what the Bellman model assumes.

    The forward pass propagates sampled outputs downstream (to capture any
    correlation induced by shared upstream noise), but controllable inputs
    are always fixed to a*_i rather than generated by the policy.

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
        # Sequential forward with fixed target actions but propagated sampled outputs.
        # For process 0: use target inputs directly.
        # For process i>0: non-controllable inputs may depend on upstream sampled
        # outputs (if the chain feeds them through), but controllable inputs are
        # fixed to a*_i from the target trajectory.

        prev_outputs_sampled = None

        for i, proc_name in enumerate(process_chain.process_names):
            info = process_chain.controllable_info_per_process[i]

            # Build fixed inputs: controllable from target, non-controllable from baseline/target
            if process_chain.baseline_trajectories is not None:
                target_inputs = process_chain.target_trajectory[proc_name]['inputs'][0].copy()
                baseline_inputs = process_chain.baseline_trajectories[proc_name]['inputs'][scenario_idx]
                for idx in info['non_controllable_indices']:
                    target_inputs[idx] = baseline_inputs[idx]
                fixed_inputs_np = target_inputs
            else:
                target_inputs_all = process_chain.target_trajectory[proc_name]['inputs']
                target_idx = min(scenario_idx, target_inputs_all.shape[0] - 1)
                fixed_inputs_np = target_inputs_all[target_idx]

            # Tile to batch — all samples get the same fixed action a*_i
            batch_inputs = np.tile(fixed_inputs_np, (n_samples, 1))
            input_tensor = torch.tensor(batch_inputs, dtype=torch.float32, device=process_chain.device)

            # Pass through UP
            scaled = process_chain.scale_inputs(input_tensor, i)
            mu_scaled, var_scaled = process_chain.uncertainty_predictors[i](scaled)
            mu_t = process_chain.unscale_outputs(mu_scaled, i)
            var_t = process_chain.unscale_variance(var_scaled, i)

            mu = mu_t.squeeze(-1).cpu().numpy()           # (n_samples,)
            var = var_t.squeeze(-1).cpu().numpy()          # (n_samples,)
            sigma = np.sqrt(var + 1e-8)                    # (n_samples,)

            # Sample and compute standardised residual
            epsilon = torch.randn(n_samples).numpy()
            o_sampled = mu + epsilon * sigma
            residuals[:, i] = (o_sampled - mu) / sigma

            # Diagnostic: verify that mu/sigma are constant (fixed actions → deterministic UP)
            print(f"  [Sigma diag] process {i} ({proc_name}): "
                  f"mu range=[{mu.min():.6f}, {mu.max():.6f}], "
                  f"sigma range=[{sigma.min():.6f}, {sigma.max():.6f}], "
                  f"residual std={residuals[:, i].std():.4f}")

    # Sample covariance
    Sigma_hat = np.cov(residuals.T)
    print(f"  [Sigma diag] Sigma_hat diagonal: {np.diag(Sigma_hat)}")
    print(f"  [Sigma diag] Sigma_hat off-diag:\n{Sigma_hat - np.diag(np.diag(Sigma_hat))}")

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

    # Get bounds for controllable inputs
    preprocessor = process_chain.preprocessors[process_idx]
    ctrl_indices = info['controllable_indices']
    n_ctrl = info['n_controllable']

    # Check for explicit action_domain override (e.g. [-1, 1] for sinusoidal SCM)
    action_domain = proc_config.get('action_domain')
    if action_domain is not None:
        a_lo, a_hi = action_domain
        ctrl_min = np.full(n_ctrl, a_lo)
        ctrl_max = np.full(n_ctrl, a_hi)
    elif preprocessor.input_min is not None:
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

    Se il surrogate ha _dynamic_configs (processi ST), usa il target calibrato.
    Altrimenti usa la logica hardcoded per i processi fisici (legacy).

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
        proc_names = surrogate._process_order if surrogate._process_order else list(surrogate._dynamic_configs.keys())
        if process_idx < len(proc_names):
            name = proc_names[process_idx]
            cfg = surrogate._dynamic_configs[name]
            target = cfg['base_target']
            for upstream, coeff in cfg.get('adaptive_coefficients', {}).items():
                if upstream in upstream_outputs:
                    baseline = cfg['adaptive_baselines'][upstream]
                    upstream_val = upstream_outputs[upstream]
                    if isinstance(baseline, (list, tuple)):
                        # Per-dim: mean of coeff * (upstream_val - baseline_k) over all k
                        target += coeff * sum(upstream_val - b for b in baseline) / len(baseline)
                    else:
                        target += coeff * (upstream_val - baseline)
            return target
        return 0.0

    # LEGACY PATH: logica hardcoded per processi fisici
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


def build_grids(F_star: float, cfg: BellmanConfig,
                 w_bar: np.ndarray = None, margin: float = 0.05):
    """
    Build discretisation grids for R and eps.

    If w_bar is provided, R_min is computed dynamically as
        R_min = F* - sum(w_bar) - margin
    so the grid always covers the full range reachable during
    forward simulation.  Falls back to cfg.R_min otherwise.

    Returns:
        grid_R: (N_R,) array
        grid_eps: (N_eps,) array
    """
    if w_bar is not None:
        R_min = F_star - float(np.sum(w_bar)) - margin
    else:
        R_min = cfg.R_min
    grid_R = np.linspace(R_min, F_star, cfg.N_R)
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

    # Minimise over actions (last axis) and extract optimal policy
    best_a = np.argmin(cost, axis=-1)  # (N_R, N_eps, ..., N_eps)
    V = np.min(cost, axis=-1)          # (N_R, N_eps, ..., N_eps)

    # Store optimal (mu, sigma2) at each grid point for policy extraction
    mu_opt = mu_m[best_a]
    sigma2_opt = sigma2_m[best_a]

    return V, mu_opt, sigma2_opt


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

        best_a = int(costs.argmin())
        return float(costs.min()), float(mu_m[best_a]), float(sigma2_m[best_a])

    else:
        # Generic non-terminal step for process_idx >= 1
        # State: (R, eps_0, ..., eps_{i-1}) → output shape (N_R, N_eps, ..., N_eps) with i eps dims
        n_prev = process_idx  # number of preceding eps dimensions
        V_out_shape = (N_R,) + (N_eps,) * n_prev
        V_out = np.zeros(V_out_shape)
        mu_opt_out = np.zeros(V_out_shape)
        sigma2_opt_out = np.zeros(V_out_shape)

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

            best_a = costs.argmin(axis=1)  # (N_R,)
            V_out[(slice(None),) + multi_idx] = costs.min(axis=1)  # (N_R,)
            mu_opt_out[(slice(None),) + multi_idx] = mu_m[best_a]
            sigma2_opt_out[(slice(None),) + multi_idx] = sigma2_m[best_a]

        return V_out, mu_opt_out, sigma2_opt_out


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
) -> Tuple[float, Dict[int, np.ndarray], np.ndarray, np.ndarray,
           Dict[int, float], Dict[int, np.ndarray], List[float],
           Dict[int, np.ndarray], Dict[int, np.ndarray],
           float, float]:
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
        Tuple of (L_min, V_functions, grid_R, grid_eps, sigma2_cond,
                  cond_weights, taus, policy_mu, policy_sigma2,
                  mu0_opt, sigma2_0_opt)
        policy_mu[i]: array with same shape as V_i, storing optimal mu at each grid point
        policy_sigma2[i]: array with same shape as V_i, storing optimal sigma2
        mu0_opt, sigma2_0_opt: optimal action for initial step (scalars)
    """
    n = len(process_names)

    # NOTE: level-dependent transforms (Sigma, manifolds) are applied by the
    # caller before passing to this function. The manifolds and Sigma received
    # here are already effective values for the requested level.
    manifolds_effective = manifolds

    grid_R, grid_eps = build_grids(F_star, cfg, w_bar=w_bar)

    if verbose:
        print(f"  R grid: [{grid_R[0]:.4f}, {grid_R[-1]:.4f}]  "
              f"(R_min = F*-sum(w̄)-margin = {F_star:.4f}-{float(np.sum(w_bar)):.4f}-0.05)")

    # Precompute conditional parameters (uses possibly-overridden Sigma)
    sigma2_cond, cond_weights = precompute_conditional_params(Sigma, n)

    if verbose:
        print(f"  Conditional variances: {[sigma2_cond[i] for i in range(n)]}")
        for i in range(1, n):
            print(f"  Conditional weights[{i}]: {cond_weights[i]}")

    # Compute adaptive targets at the operating point
    surrogate = kwargs.get('surrogate', None)
    taus = []
    upstream = {}
    for i in range(n):
        tau_i = get_adaptive_target(i, upstream, surrogate=surrogate)
        taus.append(tau_i)
        upstream[process_names[i]] = target_outputs[process_names[i]]
    if verbose:
        print(f"  Adaptive targets: {taus}")
        print(f"  w_bar: {w_bar}")
        print(f"  scales: {scales}")

    rng = np.random.default_rng(42)

    # ── Backward induction: da terminal (i=n-1) a i=0 ──
    V_functions = {}
    policy_mu = {}     # optimal mu at each grid point
    policy_sigma2 = {} # optimal sigma2 at each grid point

    # Step terminale (i = n-1): closed-form
    terminal_idx = n - 1
    if verbose:
        print(f"\n  [Step {n-1}/{n-1}] Terminal step (process '{process_names[terminal_idx]}') — closed-form...")
        print(f"    w̄={w_bar[terminal_idx]:.4f}, s={scales[terminal_idx]:.4f}, "
              f"τ={taus[terminal_idx]:.4f}, σ²_cond={sigma2_cond[terminal_idx]:.4f}")
        mu_range = manifolds_effective[terminal_idx][:, 0]
        sigma2_range = manifolds_effective[terminal_idx][:, 1]
        print(f"    Manifold: mu=[{mu_range.min():.4f}, {mu_range.max():.4f}], "
              f"sigma2=[{sigma2_range.min():.6f}, {sigma2_range.max():.6f}]")
    t0 = time.time()
    V_prev, mu_opt_prev, sigma2_opt_prev = bellman_terminal(
        grid_R, grid_eps, manifolds_effective[terminal_idx],
        w_bar[terminal_idx], scales[terminal_idx], taus[terminal_idx],
        sigma2_cond[terminal_idx], cond_weights[terminal_idx],
    )
    V_functions[terminal_idx] = V_prev
    policy_mu[terminal_idx] = mu_opt_prev
    policy_sigma2[terminal_idx] = sigma2_opt_prev
    if verbose:
        print(f"    Done in {time.time()-t0:.1f}s. V shape: {V_prev.shape}")
        print(f"    V range: [{V_prev.min():.8f}, {V_prev.max():.8f}]")
        # Check V at the F* boundary (R = F*, all eps = 0)
        r_idx_fstar = len(grid_R) - 1
        eps_mid_idx = len(grid_eps) // 2
        center_idx = (r_idx_fstar,) + (eps_mid_idx,) * (len(V_prev.shape) - 1)
        print(f"    V at (R=F*, eps=0,...,0) = {V_prev[center_idx]:.8f}")
        if V_prev.max() > F_star ** 2 + 0.01:
            print(f"    ⚠ V_max ({V_prev.max():.6f}) exceeds F*²={F_star**2:.6f} — check grid")

    # Steps intermedi (i = n-2 ... 1): MC
    for i in range(n - 2, 0, -1):
        if verbose:
            print(f"\n  [Step {i}/{n-1}] Process '{process_names[i]}' — Monte Carlo (K={cfg.K_mc})...")
            print(f"    w̄={w_bar[i]:.4f}, s={scales[i]:.4f}, "
                  f"τ={taus[i]:.4f}, σ²_cond={sigma2_cond[i]:.4f}")
        t0 = time.time()
        V_prev, mu_opt_prev, sigma2_opt_prev = bellman_non_terminal(
            grid_R, grid_eps, V_prev, manifolds_effective[i],
            w_bar[i], scales[i], taus[i],
            sigma2_cond[i], cond_weights[i],
            process_idx=i, K=cfg.K_mc,
            use_antithetic=cfg.use_antithetic, rng=rng,
        )
        V_functions[i] = V_prev
        policy_mu[i] = mu_opt_prev
        policy_sigma2[i] = sigma2_opt_prev
        if verbose:
            print(f"    Done in {time.time()-t0:.1f}s. V shape: {V_prev.shape}")
            print(f"    V range: [{V_prev.min():.8f}, {V_prev.max():.8f}]")
            if V_prev.max() > F_star ** 2 + 0.01:
                print(f"    ⚠ V_max ({V_prev.max():.6f}) exceeds F*²={F_star**2:.6f}")

    # Step iniziale (i = 0): MC, ritorna scalare
    if verbose:
        print(f"\n  [Step 0/{n-1}] Process '{process_names[0]}' — Monte Carlo (K={cfg.K_mc})...")
        print(f"    w̄={w_bar[0]:.4f}, s={scales[0]:.4f}, "
              f"τ={taus[0]:.4f}, σ²_cond={sigma2_cond[0]:.4f}")
    t0 = time.time()
    L_min, mu0_opt, sigma2_0_opt = bellman_non_terminal(
        grid_R, grid_eps, V_prev, manifolds_effective[0],
        w_bar[0], scales[0], taus[0],
        sigma2_cond[0], None,
        process_idx=0, K=cfg.K_mc,
        use_antithetic=cfg.use_antithetic, rng=rng,
    )
    if verbose:
        print(f"    Done in {time.time()-t0:.1f}s. L_min = {L_min:.8f}")
        print(f"    Optimal action for process 0: mu={mu0_opt:.4f}, sigma2={sigma2_0_opt:.6f}")
        if L_min > F_star ** 2:
            print(f"    ⚠ L_min ({L_min:.6f}) > F*² ({F_star**2:.6f}) — this is impossible, check for bugs")

    return (float(L_min), V_functions, grid_R, grid_eps, sigma2_cond, cond_weights, taus,
            policy_mu, policy_sigma2, mu0_opt, sigma2_0_opt)



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
    policy_mu: Optional[Dict[int, np.ndarray]] = None,
    policy_sigma2: Optional[Dict[int, np.ndarray]] = None,
    mu0_opt: Optional[float] = None,
    sigma2_0_opt: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[float, float]:
    """
    Forward simulation using the optimal policy from backward induction.

    IMPORTANT: The policy is extracted from the backward induction and applied
    WITHOUT conditioning on the current step's noise eps_i. The controller at
    step i observes only (R_i, eps_{<i}) and commits to action a_i before
    eps_i is realized — matching the Bellman information structure.

    Args:
        F_star: target reliability
        Sigma: (n, n) noise covariance
        manifolds: list of n manifold arrays
        w_bar: (n,) normalised weights
        scales: (n,) scale parameters
        taus: list of adaptive targets
        grid_R, grid_eps: grids from backward induction
        V_functions: {i: V_i array} for i=1,...,n-1
        sigma2_cond, cond_weights: conditional parameters
        N: number of trajectories
        rng: random generator
        policy_mu: {i: mu_opt array} for i=1,...,n-1 (from backward induction)
        policy_sigma2: {i: sigma2_opt array} for i=1,...,n-1
        mu0_opt: optimal mu for process 0 (scalar)
        sigma2_0_opt: optimal sigma2 for process 0 (scalar)
        verbose: print per-step diagnostics

    Returns:
        L_min_forward: mean loss
        se: standard error
    """
    if rng is None:
        rng = np.random.default_rng(999)

    use_policy = (policy_mu is not None and policy_sigma2 is not None
                  and mu0_opt is not None and sigma2_0_opt is not None)

    n = len(manifolds)
    L_chol = np.linalg.cholesky(Sigma)
    R_lo, R_hi = grid_R[0], grid_R[-1]
    eps_lo, eps_hi = grid_eps[0], grid_eps[-1]

    if use_policy:
        # Build interpolators for policy (mu*, sigma2*) at each step
        interp_mu = {}
        interp_sigma2 = {}
        for vi_idx in policy_mu:
            grid_axes = tuple([grid_R] + [grid_eps] * vi_idx)
            interp_mu[vi_idx] = RegularGridInterpolator(
                grid_axes, policy_mu[vi_idx],
                method='linear', bounds_error=False, fill_value=None,
            )
            interp_sigma2[vi_idx] = RegularGridInterpolator(
                grid_axes, policy_sigma2[vi_idx],
                method='linear', bounds_error=False, fill_value=None,
            )
    else:
        # Fallback: old clairvoyant method (picks action after seeing eps_i)
        interp_V = {}
        for vi_idx, V_arr in V_functions.items():
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
        eps_i = EPS[:, i]                    # (N,) — realized noise

        if use_policy:
            # ── CORRECT: choose action from precomputed policy ──
            # Policy depends only on (R, eps_{<i}), NOT on eps_i
            if i == 0:
                # Process 0: single optimal action (no state dependence)
                mu_star = mu0_opt
                sigma_star = np.sqrt(max(sigma2_0_opt, 1e-10))
                delta_star = mu_star - taus[i]
                d = delta_star + sigma_star * eps_i          # (N,)
            else:
                # Process i >= 1: interpolate policy at (R, eps_{<i})
                cols = [np.clip(R, R_lo, R_hi)]
                for h in range(eps_hist.shape[1]):
                    cols.append(np.clip(eps_hist[:, h], eps_lo, eps_hi))
                points = np.column_stack(cols)

                mu_star = interp_mu[i](points)               # (N,)
                sigma2_star = interp_sigma2[i](points)        # (N,)
                sigma_star = np.sqrt(np.maximum(sigma2_star, 1e-10))
                delta_star = mu_star - taus[i]
                d = delta_star + sigma_star * eps_i           # (N,)

            Q = np.exp(-(d ** 2) / scales[i])
            R = R - w_bar[i] * Q

        else:
            # ── CLAIRVOYANT fallback (old behaviour, biased low) ──
            M_i = manifolds[i].shape[0]
            mu_m = manifolds[i][:, 0]
            sigma2_m = manifolds[i][:, 1]
            sigma_m = np.sqrt(np.maximum(sigma2_m, 1e-10))
            delta_m = mu_m - taus[i]

            d_all = delta_m[:, None] + sigma_m[:, None] * eps_i[None, :]
            Q_all = np.exp(-(d_all ** 2) / scales[i])
            R_next_all = R[None, :] - w_bar[i] * Q_all

            if i < n - 1:
                R_c = np.clip(R_next_all, R_lo, R_hi)
                eps_i_c = np.clip(eps_i, eps_lo, eps_hi)
                R_flat = R_c.ravel()
                eps_i_flat = np.tile(eps_i_c, M_i)
                cols = [R_flat]
                for h in range(eps_hist.shape[1]):
                    cols.append(np.tile(np.clip(eps_hist[:, h], eps_lo, eps_hi), M_i))
                cols.append(eps_i_flat)
                points = np.column_stack(cols)
                v_next = interp_V[i + 1](points).reshape(M_i, N)
            else:
                v_next = R_next_all ** 2

            best_m = np.argmin(v_next, axis=0)
            d_best = d_all[best_m, np.arange(N)]
            Q_best = np.exp(-(d_best ** 2) / scales[i])
            R = R - w_bar[i] * Q_best

        eps_hist = np.column_stack([eps_hist, eps_i]) if eps_hist.size > 0 else eps_i[:, None]

        if verbose:
            mode_label = 'policy' if use_policy else 'clairvoyant'
            print(f"    Step {i} ({mode_label}): "
                  f"R mean={R.mean():.6f}, std={R.std():.6f}, "
                  f"range=[{R.min():.6f}, {R.max():.6f}]")

    losses = R ** 2
    L_min_forward = float(np.mean(losses))
    se = float(np.std(losses) / np.sqrt(N))

    if verbose:
        print(f"    Final: L_min={L_min_forward:.8f} ± {se:.8f}")
        pct_R_neg = float(np.mean(R < 0) * 100)
        pct_R_out = float(np.mean(R < R_lo) * 100)
        print(f"    R<0: {pct_R_neg:.1f}%, R<R_min({R_lo:.2f}): {pct_R_out:.1f}%")

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
    # Usa _dynamic_configs dal surrogate se disponibili (ST), altrimenti vuoto
    if hasattr(surrogate, '_dynamic_configs') and surrogate._dynamic_configs is not None:
        proc_configs = surrogate._dynamic_configs
    else:
        proc_configs = {}

    def _scalar(val, default=1.0):
        """Convert list config value to scalar (mean) for theoretical analysis."""
        if isinstance(val, list):
            return sum(val) / len(val)
        return val

    weights = np.array([_scalar(proc_configs[name]['weight']) for name in process_names])
    w_bar = weights / weights.sum()
    scales = np.array([_scalar(proc_configs[name]['scale']) for name in process_names])

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
        print(f"\n[Phase 1a] Estimating noise covariance Sigma (n_samples=1000, fixed actions a*)...")
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
            sigma2_min = M_i[:, 1].min()
            sigma2_max = M_i[:, 1].max()
            print(f"  {process_names[i]:12s}: {M_i.shape[0]:3d} points, "
                  f"mu=[{M_i[:,0].min():.3f}, {M_i[:,0].max():.3f}], "
                  f"sigma2=[{sigma2_min:.6f}, {sigma2_max:.6f}], "
                  f"sigma2_min/scale={sigma2_min/scales[i]:.6f}")
    if verbose:
        print(f"  Done in {time.time()-t_phase:.1f}s")

    # ── Get target outputs for adaptive targets ──
    target_outputs = get_target_outputs(process_chain, scenario_idx)
    if verbose:
        print(f"  Target outputs: {target_outputs}")

    # ── Helper: apply level-dependent transformations ──
    def _apply_level_transforms(lvl, Sigma_full, manifolds_raw):
        if lvl == 1:
            Sigma_eff = np.eye(n)
            manifolds_eff = []
            for m in manifolds_raw:
                sigma2_mean = float(np.mean(m[:, 1]))
                m_new = m.copy()
                m_new[:, 1] = sigma2_mean
                manifolds_eff.append(m_new)
        elif lvl == 2:
            Sigma_eff = np.eye(n)
            manifolds_eff = manifolds_raw
        else:
            Sigma_eff = Sigma_full
            manifolds_eff = manifolds_raw
        return Sigma_eff, manifolds_eff

    # ── Helper: run backward induction + forward validation for one level ──
    def _run_single_level(lvl, Sigma_full, manifolds_raw, verbose_level):
        level_desc = {1: 'fixed σ², Σ=I', 2: 'free σ², Σ=I', 3: 'free σ², full Σ'}
        Sigma_eff, manifolds_eff = _apply_level_transforms(lvl, Sigma_full, manifolds_raw)

        if verbose_level:
            print(f"\n  [Level {lvl}] {level_desc.get(lvl, 'unknown')}")

        # Phase 2: Backward induction
        (L_bwd, V_funcs, gr_R, gr_eps,
         s2_cond, c_wts, taus_lvl,
         p_mu, p_sigma2, mu0, sigma2_0) = backward_induction(
            F_star, Sigma_eff, manifolds_eff, w_bar, scales,
            target_outputs, process_names, cfg, verbose_level,
            surrogate=surrogate,
        )

        # Phase 3a: Forward validation (policy-based)
        L_fwd, L_fwd_se = forward_simulation(
            F_star, Sigma_eff, manifolds_eff, w_bar, scales, taus_lvl,
            gr_R, gr_eps, V_funcs, s2_cond, c_wts,
            N=cfg.N_forward,
            policy_mu=p_mu, policy_sigma2=p_sigma2,
            mu0_opt=mu0, sigma2_0_opt=sigma2_0,
            verbose=verbose_level,
        )

        # Phase 3b: Clairvoyant forward (diagnostic)
        L_clairv, L_clairv_se = forward_simulation(
            F_star, Sigma_eff, manifolds_eff, w_bar, scales, taus_lvl,
            gr_R, gr_eps, V_funcs, s2_cond, c_wts,
            N=cfg.N_forward,
        )

        return {
            'level': lvl,
            'L_min_bellman': L_bwd,
            'L_min_forward': L_fwd,
            'L_min_forward_se': L_fwd_se,
            'L_clairvoyant': L_clairv,
            'L_clairvoyant_se': L_clairv_se,
            'manifolds_effective': manifolds_eff,
            'V_functions': V_funcs,
            'grid_R': gr_R,
            'grid_eps': gr_eps,
            'sigma2_cond': s2_cond,
            'cond_weights': c_wts,
            'taus': taus_lvl,
            'policy_mu': p_mu,
            'policy_sigma2': p_sigma2,
            'mu0_opt': mu0,
            'sigma2_0_opt': sigma2_0,
        }

    # ── Determine which levels to run ──
    if cfg.parallel_levels:
        levels_to_run = [1, 2, 3]
    else:
        levels_to_run = [cfg.level]

    # ── Run level(s) ──
    if len(levels_to_run) == 1:
        # Single level — run directly (original path)
        lvl = levels_to_run[0]
        if verbose:
            level_desc = {1: 'fixed σ², Σ=I', 2: 'free σ², Σ=I', 3: 'free σ², full Σ'}
            print(f"\n  Bellman level: {lvl}  ({level_desc.get(lvl, 'unknown')})")
            print(f"\n[Phase 2] Running backward induction...")
        res = _run_single_level(lvl, Sigma, manifolds, verbose)
        all_results = {lvl: res}
    else:
        # Parallel levels — verbose only for primary (level 3)
        if verbose:
            print(f"\n[Phase 2+3] Running all 3 Bellman levels in parallel...")
        t_par = time.time()
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                lvl: executor.submit(
                    _run_single_level, lvl, Sigma, manifolds,
                    verbose and (lvl == cfg.level),  # verbose only for primary level
                )
                for lvl in levels_to_run
            }
            all_results = {lvl: fut.result() for lvl, fut in futures.items()}
        if verbose:
            print(f"  All 3 levels completed in {time.time()-t_par:.1f}s")

    # ── Primary level results (for backward-compatible return) ──
    primary = all_results[cfg.level]
    L_min_bellman = primary['L_min_bellman']
    L_min_forward = primary['L_min_forward']
    L_min_forward_se = primary['L_min_forward_se']
    L_clairv = primary['L_clairvoyant']
    L_clairv_se = primary['L_clairvoyant_se']
    manifolds_effective = primary['manifolds_effective']

    L_min_bellman_scaled = L_min_bellman * loss_scale
    L_min_forward_scaled = L_min_forward * loss_scale
    L_min_forward_se_scaled = L_min_forward_se * loss_scale

    if verbose:
        print(f"\n  L_min (Bellman) = {L_min_bellman:.8f}")
        if loss_scale != 1.0:
            print(f"  L_min (scaled) = {L_min_bellman_scaled:.6f}")
        if L_min_bellman > F_star ** 2 + 1e-6:
            print(f"  ⚠ WARNING: L_min ({L_min_bellman:.6f}) > F*² ({F_star**2:.6f}) "
                  f"— this exceeds the theoretical maximum, check for numerical issues")
        print(f"  L_min (forward, policy) = {L_min_forward:.8f} ± {L_min_forward_se:.8f}")
        rel_diff = abs(L_min_forward - L_min_bellman) / max(abs(L_min_bellman), 1e-10)
        print(f"  Relative difference Bellman vs Forward: {rel_diff*100:.2f}%")
        if rel_diff > 0.3:
            print(f"  NOTE: >30% gap — may indicate coarse grid discretisation")
        print(f"  L_min (clairvoyant) = {L_clairv:.8f} ± {L_clairv_se:.8f}")
        print(f"  Clairvoyant/Policy ratio: {L_clairv / max(L_min_forward, 1e-10):.2f}x "
              f"(should be ≤ 1.0; clairvoyant has information advantage)")

    # ── Summary ──
    computation_time = time.time() - t_start
    if verbose:
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"  F*            = {F_star:.6f}")
        print(f"  F*²           = {F_star**2:.6f}  (theoretical max loss)")

        if cfg.parallel_levels:
            # Multi-level comparison table
            print(f"\n  {'Level':<8} {'Bellman':>14} {'Forward':>14} {'Clairvoyant':>14}")
            print(f"  {'─'*52}")
            for lvl in [1, 2, 3]:
                r = all_results[lvl]
                Lb = r['L_min_bellman'] * loss_scale if loss_scale != 1.0 else r['L_min_bellman']
                Lf = r['L_min_forward'] * loss_scale if loss_scale != 1.0 else r['L_min_forward']
                Lc = r['L_clairvoyant'] * loss_scale if loss_scale != 1.0 else r['L_clairvoyant']
                marker = " ←" if lvl == cfg.level else ""
                print(f"  {lvl:<8} {Lb:>14.8f} {Lf:>14.8f} {Lc:>14.8f}{marker}")
            print(f"  {'─'*52}")
            print(f"  L1 ≤ L2 ≤ L3 hierarchy: ", end="")
            L_fwd = {lvl: all_results[lvl]['L_min_forward'] for lvl in [1, 2, 3]}
            if L_fwd[1] <= L_fwd[2] * 1.01 and L_fwd[2] <= L_fwd[3] * 1.01:
                print("✓ OK (less noise info → lower L_min)")
            else:
                print(f"✗ UNEXPECTED (L1={L_fwd[1]:.8f}, L2={L_fwd[2]:.8f}, L3={L_fwd[3]:.8f})")
        else:
            print(f"  L_min (Bellman):     {L_min_bellman:.8f}" +
                  (f"  (scaled: {L_min_bellman_scaled:.6f})" if loss_scale != 1.0 else ""))
            print(f"  L_min (fwd policy):  {L_min_forward:.8f} ± {L_min_forward_se:.8f}" +
                  (f"  (scaled: {L_min_forward_scaled:.6f})" if loss_scale != 1.0 else ""))
            print(f"  L_min (fwd clairv):  {L_clairv:.8f} ± {L_clairv_se:.8f}" +
                  (f"  (scaled: {L_clairv * loss_scale:.6f})" if loss_scale != 1.0 else ""))
            print(f"  Bellman ≈ Policy?  |diff| = {abs(L_min_forward - L_min_bellman):.8f} "
                  f"({abs(L_min_forward - L_min_bellman) / max(abs(L_min_bellman), 1e-10) * 100:.1f}%)")
            print(f"  Clairvoyant < Policy? {L_clairv:.8f} < {L_min_forward:.8f} → "
                  f"{'yes (expected)' if L_clairv < L_min_forward else 'no (unexpected)'}")

        print(f"  Computation time: {computation_time:.1f}s")
        print(f"{'='*70}\n")

    # ── Build per-level results dict (for parallel mode) ──
    level_results_dict = None
    if cfg.parallel_levels:
        level_results_dict = {}
        for lvl in [1, 2, 3]:
            r = all_results[lvl]
            level_results_dict[lvl] = {
                'L_min_bellman': r['L_min_bellman'] * loss_scale if loss_scale != 1.0 else r['L_min_bellman'],
                'L_min_forward': r['L_min_forward'] * loss_scale if loss_scale != 1.0 else r['L_min_forward'],
                'L_min_forward_se': r['L_min_forward_se'] * loss_scale if loss_scale != 1.0 else r['L_min_forward_se'],
                'L_clairvoyant': r['L_clairvoyant'] * loss_scale if loss_scale != 1.0 else r['L_clairvoyant'],
                'L_clairvoyant_se': r['L_clairvoyant_se'] * loss_scale if loss_scale != 1.0 else r['L_clairvoyant_se'],
            }

    return BellmanLminResult(
        L_min_bellman=L_min_bellman_scaled if loss_scale != 1.0 else L_min_bellman,
        L_min_forward=L_min_forward_scaled if loss_scale != 1.0 else L_min_forward,
        L_min_forward_se=L_min_forward_se_scaled if loss_scale != 1.0 else L_min_forward_se,
        F_star=F_star,
        Sigma=Sigma,
        w_bar=w_bar,
        scales=scales,
        n_manifold_points=[m.shape[0] for m in manifolds_effective],
        computation_time_s=computation_time,
        level=cfg.level,
        level_results=level_results_dict,
    )
