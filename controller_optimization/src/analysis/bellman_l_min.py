"""
Bellman Backward Induction for L_min Computation.

Computes the true minimum achievable loss L_min via dynamic programming,
accounting for an optimal reactive controller that observes upstream noise
realizations and adapts downstream actions accordingly.

The pipeline has n=4 sequential processes (Laser -> Plasma -> Galvanic -> Microetch).
The loss is L = (F - F*)^2 where:
  - F = sum_i w_bar_i * Q_i  (weighted average of quality scores)
  - Q_i = exp(-(delta_i + sigma_i * eps_i)^2 / s_i)
  - F* is the deterministic target reliability

L_min = V_1(F*) where V_i is the Bellman value function.

Level 1: Independent noise (Sigma = I), variance fixed per process.
         State is scalar R_i (remaining reliability).

Level 3: Correlated noise (general Sigma), variance fixed per process.
         State is (R_i, eps_{<i}) -- grows with process index.

References:
    Bellman, R. (1957). Dynamic Programming. Princeton University Press.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class BellmanLMinResult:
    """Results from Bellman backward induction."""
    L_min: float                          # Minimum achievable loss (unscaled)
    L_min_scaled: float                   # L_min * loss_scale
    level: int                            # 1 or 3
    F_star: float                         # Target reliability
    n_processes: int                      # Number of processes
    w_bar: np.ndarray                     # Normalized weights
    sigma: np.ndarray                     # Per-process noise std
    s: np.ndarray                         # Scale parameters
    optimal_policy: Optional[List[np.ndarray]] = None  # delta*(R) per process
    value_functions: Optional[List[np.ndarray]] = None  # V_i(R) per process
    R_grid: Optional[np.ndarray] = None   # Grid used for R
    validation_L_min: Optional[float] = None  # Forward simulation estimate
    validation_std: Optional[float] = None    # Std of forward simulation

    def to_dict(self) -> Dict:
        return {
            'L_min': self.L_min,
            'L_min_scaled': self.L_min_scaled,
            'level': self.level,
            'F_star': self.F_star,
            'n_processes': self.n_processes,
            'w_bar': self.w_bar.tolist(),
            'sigma': self.sigma.tolist(),
            's': self.s.tolist(),
            'validation_L_min': self.validation_L_min,
            'validation_std': self.validation_std,
        }


def _closed_form_terminal(R_n, w_bar_n, sigma_n, s_n, delta_grid):
    """
    Closed-form expected cost at the terminal (last) process.

    E[(R_n - w_bar_n * Q_n)^2] = R_n^2 - 2*R_n*w_bar_n*E[Q_n] + w_bar_n^2*E[Q_n^2]

    where:
      E[Q_n] = exp(-delta^2 / (s + 2*sigma^2)) / sqrt(1 + 2*sigma^2/s)
      E[Q_n^2] = exp(-2*delta^2 / (s + 4*sigma^2)) / sqrt(1 + 4*sigma^2/s)

    Args:
        R_n: array of shape (N_R,), remaining reliability values
        w_bar_n: scalar, normalized weight for process n
        sigma_n: scalar, noise std for process n
        s_n: scalar, scale parameter for process n
        delta_grid: array of shape (N_delta,), candidate deviations

    Returns:
        V_n: array of shape (N_R,), optimal value at each R_n
        best_delta: array of shape (N_R,), optimal delta at each R_n
    """
    sigma2 = sigma_n ** 2

    # E[Q] for each delta: shape (N_delta,)
    denom1 = s_n + 2 * sigma2
    E_Q = np.exp(-delta_grid ** 2 / denom1) / np.sqrt(1 + 2 * sigma2 / s_n)

    # E[Q^2] for each delta: shape (N_delta,)
    denom2 = s_n + 4 * sigma2
    E_Q2 = np.exp(-2 * delta_grid ** 2 / denom2) / np.sqrt(1 + 4 * sigma2 / s_n)

    # For each (R, delta): E[(R - w*Q)^2] = R^2 - 2*R*w*E[Q] + w^2*E[Q^2]
    # Shape: (N_R, N_delta)
    R_col = R_n[:, np.newaxis]  # (N_R, 1)
    cost = R_col ** 2 - 2 * R_col * w_bar_n * E_Q[np.newaxis, :] + w_bar_n ** 2 * E_Q2[np.newaxis, :]

    # Minimize over delta
    best_idx = np.argmin(cost, axis=1)
    V_n = cost[np.arange(len(R_n)), best_idx]
    best_delta = delta_grid[best_idx]

    return V_n, best_delta


def compute_l_min_level1(
    F_star: float,
    w_bar: np.ndarray,
    sigma: np.ndarray,
    s: np.ndarray,
    loss_scale: float = 1.0,
    N_R: int = 200,
    N_delta: int = 100,
    N_quad: int = 80,
    delta_range: Tuple[float, float] = (-3.0, 3.0),
    **kwargs,
) -> BellmanLMinResult:
    """
    Compute L_min via Bellman backward induction (Level 1: independent noise).

    State is scalar R_i = F* - sum_{j<i} w_bar_j * Q_j (remaining reliability).
    Noise eps_i ~ N(0,1) iid across processes.

    Uses closed-form expectations at the terminal process and Gauss-Hermite
    quadrature for exact numerical integration at earlier processes.

    Args:
        F_star: Target reliability (scalar).
        w_bar: Normalized weights, shape (n,). Must sum to 1.
        sigma: Per-process noise std, shape (n,).
        s: Scale parameters, shape (n,).
        loss_scale: Multiplier for the loss (typically 100.0).
        N_R: Number of grid points for R.
        N_delta: Number of grid points for delta.
        N_quad: Number of Gauss-Hermite quadrature points (default 80).
        delta_range: Range for delta grid.

    Returns:
        BellmanLMinResult with L_min and optional policy/value functions.
    """
    w_bar = np.asarray(w_bar, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    n = len(w_bar)

    # R grid: the remaining reliability can range roughly in [-margin, F* + margin]
    # where margin accounts for the fact that Q_i can overshoot.
    margin = 0.15 * F_star
    R_lo = -margin
    R_hi = F_star + margin
    R_grid = np.linspace(R_lo, R_hi, N_R)

    delta_grid = np.linspace(delta_range[0], delta_range[1], N_delta)
    # Ensure delta=0 is always on the grid (critical: it's the "hit target" action)
    if 0.0 not in delta_grid:
        delta_grid = np.sort(np.append(delta_grid, 0.0))

    # Gauss-Hermite quadrature for E[f(eps)] where eps ~ N(0,1).
    # hermgauss gives nodes t_k, weights w_k for integral of f(t)*exp(-t^2) dt.
    # E[f(eps)] = (1/sqrt(pi)) * sum_k w_k * f(sqrt(2)*t_k)
    gh_nodes, gh_weights = np.polynomial.hermite.hermgauss(N_quad)
    # Transform to N(0,1) quadrature
    quad_points = np.sqrt(2.0) * gh_nodes   # eps values
    quad_weights = gh_weights / np.sqrt(np.pi)  # normalized weights (sum to 1)

    # Storage for value functions and policies
    value_functions = [None] * (n + 1)
    optimal_policies = [None] * n

    # Terminal condition: V_{n+1}(R) = R^2
    value_functions[n] = R_grid ** 2

    # Backward induction
    for i in range(n - 1, -1, -1):
        w_i = w_bar[i]
        sigma_i = sigma[i]
        s_i = s[i]

        if i == n - 1:
            # Last process: use closed-form
            V_i, best_delta_i = _closed_form_terminal(
                R_grid, w_i, sigma_i, s_i, delta_grid
            )
        else:
            # Earlier processes: Gauss-Hermite quadrature
            V_next = value_functions[i + 1]  # shape (N_R,)

            # Precompute Q for all (delta, quadrature_point) pairs
            # delta_grid: (N_delta,), quad_points: (N_quad,)
            # arg: (N_delta, N_quad)
            arg = (delta_grid[:, np.newaxis] + sigma_i * quad_points[np.newaxis, :])
            Q_all = np.exp(-(arg ** 2) / s_i)  # (N_delta, N_quad)

            V_i = np.full(N_R, np.inf)
            best_delta_i = np.zeros(N_R)

            for r_idx, R_val in enumerate(R_grid):
                # R_next for all (delta, quad): (N_delta, N_quad)
                R_next = R_val - w_i * Q_all

                # Interpolate V_{i+1} at R_next
                R_next_flat = R_next.ravel()
                V_next_interp = np.interp(R_next_flat, R_grid, V_next)
                V_next_interp = V_next_interp.reshape(Q_all.shape)  # (N_delta, N_quad)

                # Weighted sum (exact Gaussian expectation): (N_delta,)
                E_V = V_next_interp @ quad_weights

                # Minimize over delta
                best_d_idx = np.argmin(E_V)
                V_i[r_idx] = E_V[best_d_idx]
                best_delta_i[r_idx] = delta_grid[best_d_idx]

        value_functions[i] = V_i
        optimal_policies[i] = best_delta_i

    # L_min = V_1(F*), interpolated from the grid
    L_min = float(np.interp(F_star, R_grid, value_functions[0]))

    return BellmanLMinResult(
        L_min=L_min,
        L_min_scaled=L_min * loss_scale,
        level=1,
        F_star=F_star,
        n_processes=n,
        w_bar=w_bar,
        sigma=sigma,
        s=s,
        optimal_policy=optimal_policies,
        value_functions=value_functions,
        R_grid=R_grid,
    )


def compute_l_min_level3(
    F_star: float,
    w_bar: np.ndarray,
    sigma: np.ndarray,
    s: np.ndarray,
    Sigma: np.ndarray,
    loss_scale: float = 1.0,
    N_R: int = 200,
    N_eps: int = 30,
    N_delta: int = 80,
    K: int = 1000,
    delta_range: Tuple[float, float] = (-3.0, 3.0),
    eps_range: Tuple[float, float] = (-3.0, 3.0),
    seed: int = 42,
) -> BellmanLMinResult:
    """
    Compute L_min via Bellman backward induction (Level 3: correlated noise).

    State at step i is (R_i, eps_{<i}) where eps_{<i} is the vector of
    previously observed standardized residuals.

    The conditional distribution of eps_i given eps_{<i} is:
      eps_i | eps_{<i} ~ N(mu_cond, sigma2_cond)
    where:
      mu_cond = Sigma_{i,<i} @ Sigma_{<i,<i}^{-1} @ eps_{<i}
      sigma2_cond = Sigma_{ii} - Sigma_{i,<i} @ Sigma_{<i,<i}^{-1} @ Sigma_{<i,i}

    This is computationally intensive: at step i the state grid has dimension
    1 + (i-1) = i, so the last step has a 4D grid.

    Args:
        F_star: Target reliability (scalar).
        w_bar: Normalized weights, shape (n,).
        sigma: Per-process noise std, shape (n,).
        s: Scale parameters, shape (n,).
        Sigma: Noise covariance matrix, shape (n, n).
        loss_scale: Multiplier for the loss.
        N_R: Grid points for R dimension.
        N_eps: Grid points per epsilon dimension.
        N_delta: Grid points for delta optimization.
        K: Monte Carlo samples for conditional expectation.
        delta_range: Range for delta grid.
        eps_range: Range for epsilon grid.
        seed: Random seed.

    Returns:
        BellmanLMinResult with L_min.
    """
    w_bar = np.asarray(w_bar, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    n = len(w_bar)

    rng = np.random.default_rng(seed)

    # Precompute conditional distribution parameters for each process
    # For process i, given eps_{<i} = (eps_0, ..., eps_{i-1}):
    #   mu_cond = A_i @ eps_{<i}
    #   sigma2_cond = scalar (constant, independent of eps_{<i})
    cond_params = []
    for i in range(n):
        if i == 0:
            # First process: no conditioning, eps_0 ~ N(0, Sigma_{00})
            cond_params.append({
                'A': np.zeros((0,)),  # no conditioning vector
                'sigma2_cond': Sigma[i, i],
            })
        else:
            Sigma_i_prev = Sigma[i, :i]  # (i,)
            Sigma_prev_prev = Sigma[:i, :i]  # (i, i)
            # Regularize if needed
            try:
                Sigma_prev_inv = np.linalg.inv(Sigma_prev_prev)
            except np.linalg.LinAlgError:
                reg = 0.01 * np.eye(i)
                Sigma_prev_inv = np.linalg.inv(Sigma_prev_prev + reg)

            A = Sigma_i_prev @ Sigma_prev_inv  # (i,)
            sigma2_cond = Sigma[i, i] - Sigma_i_prev @ Sigma_prev_inv @ Sigma[: i, i]
            sigma2_cond = max(sigma2_cond, 1e-10)  # numerical safety

            cond_params.append({
                'A': A,
                'sigma2_cond': sigma2_cond,
            })

    margin = 0.15 * F_star
    R_lo = -margin
    R_hi = F_star + margin
    R_grid = np.linspace(R_lo, R_hi, N_R)
    delta_grid = np.linspace(delta_range[0], delta_range[1], N_delta)
    eps_grid = np.linspace(eps_range[0], eps_range[1], N_eps)

    # We work backward. At each step i, the value function is defined over
    # (R, eps_0, ..., eps_{i-1}), i.e. dimension 1 + i.
    # We store the value function as an ndarray over a meshgrid.

    # Terminal: V_{n+1}(R) = R^2 (no eps dimensions since all are observed)
    # At step n (last process), state is (R_n, eps_0, ..., eps_{n-1}).
    # But we only need V_{n+1}(R_{n+1}) = R_{n+1}^2, which depends only on R.

    # Start from the terminal
    # V_{n}(R, eps_{<n}) = min_delta E[V_{n+1}(R - w_n Q_n) | eps_{<n}]
    # where eps_n | eps_{<n} ~ N(mu_cond, sigma2_cond)

    # We'll compute iteratively from i = n-1 down to 0.
    # At each step, store the value function on the grid.

    # For process i=n-1 (last, index n-1), use closed-form since
    # V_{n}(R_{n+1}) = R_{n+1}^2 regardless of eps history.
    # But the conditional distribution of eps_{n-1} depends on eps_{<n-1}.

    # We'll use a recursive approach with interpolation.
    # To keep memory manageable, we process one step at a time.

    # Step n-1 (terminal process): value function on (R, eps_0, ..., eps_{n-2})
    # For n=4, at i=3 (microetch): state = (R, eps_0, eps_1, eps_2) -> 4D grid
    # Grid size: N_R * N_eps^3 ~ 200 * 30^3 = 5.4M points -- feasible

    # We process backward: i = n-1, n-2, ..., 0
    # After processing step i, we have V_i on grid (R, eps_0, ..., eps_{i-1})
    # For i: dimension = 1 + i (R plus i epsilon dimensions)

    # At the terminal step (i = n-1), V_{n} depends only on R, so
    # V_n(R) = R^2 is 1D.

    V_next = R_grid ** 2  # shape (N_R,), only depends on R

    for i in range(n - 1, -1, -1):
        w_i = w_bar[i]
        sigma_i = sigma[i]
        s_i = s[i]

        cp = cond_params[i]
        sigma2_cond_i = cp['sigma2_cond']
        sigma_cond_i = np.sqrt(sigma2_cond_i)
        A_i = cp['A']

        # State dimensions at step i: (R, eps_0, ..., eps_{i-1})
        # After minimization, V_i has shape (N_R,) if i==0, or (N_R, N_eps, ..., N_eps) with i eps dims

        if i == 0:
            # No epsilon history. eps_0 ~ N(0, Sigma_{00}).
            # V_0(R) = min_delta E_{eps_0}[V_1(R - w_0 Q_0)]
            # V_1 depends only on R (if n==1) or on (R, eps_0) (if n>1)

            std_0 = np.sqrt(Sigma[0, 0])
            z_samples_local = rng.standard_normal(K)

            if i == n - 1:
                # Single process: use closed-form
                V_i_arr, _ = _closed_form_terminal(R_grid, w_i, sigma_i * std_0, s_i, delta_grid)
            else:
                # V_next shape: (N_R, N_eps) for i=0, next is step 1 with state (R, eps_0)
                V_i_arr = np.full(N_R, np.inf)

                for r_idx, R_val in enumerate(R_grid):
                    best_cost = np.inf
                    for d_idx, delta_val in enumerate(delta_grid):
                        # Sample eps_0 ~ N(0, Sigma_{00})
                        eps_0_samples = std_0 * z_samples_local  # (K,)
                        Q_samples = np.exp(-((delta_val + sigma_i * eps_0_samples) ** 2) / s_i)
                        R_next_samples = R_val - w_i * Q_samples  # (K,)

                        # Interpolate V_next(R_next, eps_0) for each sample
                        cost_samples = np.zeros(K)
                        for k in range(K):
                            cost_samples[k] = _interp_nd(
                                V_next, [R_grid, eps_grid],
                                [R_next_samples[k], eps_0_samples[k]]
                            )
                        avg_cost = cost_samples.mean()
                        if avg_cost < best_cost:
                            best_cost = avg_cost
                    V_i_arr[r_idx] = best_cost

            V_current = V_i_arr  # shape (N_R,)

        else:
            # State: (R, eps_0, ..., eps_{i-1}) -- (1 + i) dimensions
            # For each state point, compute conditional mean/var of eps_i,
            # then MC integrate over eps_i.

            # Build the state grid shape
            eps_dims = tuple([N_eps] * i)
            state_shape = (N_R,) + eps_dims
            V_i_arr = np.full(state_shape, np.inf)

            # Generate eps grid points for each dimension
            eps_grids = [eps_grid] * i  # list of i arrays, each (N_eps,)

            # Create meshgrid indices for iterating over epsilon dimensions
            eps_indices = np.array(np.meshgrid(
                *[np.arange(N_eps)] * i, indexing='ij'
            )).reshape(i, -1).T  # (N_eps^i, i)

            z_samples_local = rng.standard_normal(K)

            for r_idx, R_val in enumerate(R_grid):
                for eps_idx_tuple in eps_indices:
                    eps_vals = np.array([eps_grid[j] for j in eps_idx_tuple])  # (i,)

                    # Conditional distribution: eps_i | eps_{<i}
                    mu_cond = A_i @ eps_vals
                    # sigma_cond_i already computed

                    # Sample eps_i from conditional
                    eps_i_samples = mu_cond + sigma_cond_i * z_samples_local  # (K,)

                    best_cost = np.inf
                    for d_idx, delta_val in enumerate(delta_grid):
                        Q_samples = np.exp(-((delta_val + sigma_i * eps_i_samples) ** 2) / s_i)
                        R_next_samples = R_val - w_i * Q_samples

                        if i == n - 1:
                            # V_next(R) = R^2 -- only depends on R
                            cost_samples = R_next_samples ** 2
                        else:
                            # V_next has shape (N_R, N_eps, ..., N_eps) with (i+1) eps dims
                            # Interpolate V_next(R_next, eps_0, ..., eps_{i-1}, eps_i)
                            cost_samples = np.zeros(K)
                            for k in range(K):
                                point = [R_next_samples[k]] + eps_vals.tolist() + [eps_i_samples[k]]
                                grids = [R_grid] + eps_grids + [eps_grid]
                                cost_samples[k] = _interp_nd(V_next, grids, point)

                        avg_cost = cost_samples.mean()
                        if avg_cost < best_cost:
                            best_cost = avg_cost

                    idx = (r_idx,) + tuple(eps_idx_tuple)
                    V_i_arr[idx] = best_cost

            V_current = V_i_arr

        V_next = V_current

    # L_min = V_0(F*)
    # V_0 has shape (N_R,) -- scalar state (just R, no epsilon history for process 0)
    L_min = float(np.interp(F_star, R_grid, V_next.ravel()[:N_R] if V_next.ndim > 1 else V_next))

    return BellmanLMinResult(
        L_min=L_min,
        L_min_scaled=L_min * loss_scale,
        level=3,
        F_star=F_star,
        n_processes=n,
        w_bar=w_bar,
        sigma=sigma,
        s=s,
        R_grid=R_grid,
    )


def _interp_nd(V, grids, point):
    """
    N-dimensional linear interpolation on a regular grid.

    Simple recursive implementation for small dimensionality (up to 4-5D).

    Args:
        V: ndarray, values on the grid.
        grids: list of 1D arrays, one per dimension.
        point: list of floats, coordinates to interpolate at.

    Returns:
        Interpolated value (float).
    """
    ndim = len(grids)
    if ndim == 1:
        return float(np.interp(point[0], grids[0], V))

    # Find bracketing indices in first dimension
    grid0 = grids[0]
    x = point[0]

    # Clip to grid range
    x = np.clip(x, grid0[0], grid0[-1])

    # Find index
    idx = np.searchsorted(grid0, x) - 1
    idx = np.clip(idx, 0, len(grid0) - 2)

    x0, x1 = grid0[idx], grid0[idx + 1]
    if x1 == x0:
        t = 0.0
    else:
        t = (x - x0) / (x1 - x0)

    # Recursively interpolate the two slices
    v0 = _interp_nd(V[idx], grids[1:], point[1:])
    v1 = _interp_nd(V[idx + 1], grids[1:], point[1:])

    return (1 - t) * v0 + t * v1


def validate_bellman_policy(
    result: BellmanLMinResult,
    N_simulations: int = 10000,
    seed: int = 123,
) -> Tuple[float, float]:
    """
    Validate Bellman L_min via forward Monte Carlo simulation.

    Simulates N complete pipeline trajectories under the optimal policy
    extracted from backward induction, and computes the sample mean of (F - F*)^2.

    Uses nearest-neighbor policy lookup (not interpolation) to avoid
    artifacts from interpolating a potentially non-smooth policy function.

    For each state R encountered, the policy is evaluated by finding the
    nearest R grid point and re-optimizing delta using the stored value
    function V_{i+1}.

    Only works for Level 1 results (independent noise, scalar state).

    Args:
        result: BellmanLMinResult from compute_l_min_level1.
        N_simulations: Number of forward trajectories.
        seed: Random seed.

    Returns:
        (mean_loss, std_loss): Sample mean and std of (F - F*)^2.
    """
    if result.level != 1 or result.optimal_policy is None:
        raise ValueError("Forward validation requires Level 1 result with optimal_policy.")

    rng = np.random.default_rng(seed)
    n = result.n_processes
    F_star = result.F_star
    w_bar = result.w_bar
    sigma = result.sigma
    s = result.s
    R_grid = result.R_grid
    policies = result.optimal_policy

    losses = np.zeros(N_simulations)

    for sim in range(N_simulations):
        R = F_star  # Start with full remaining reliability
        for i in range(n):
            # Nearest-neighbor policy lookup (avoids interpolation artifacts)
            idx = np.searchsorted(R_grid, R)
            idx = np.clip(idx, 0, len(R_grid) - 1)
            # Check if the previous index is closer
            if idx > 0 and abs(R - R_grid[idx - 1]) < abs(R - R_grid[idx]):
                idx = idx - 1
            delta_opt = policies[i][idx]

            # Sample noise
            eps_i = rng.standard_normal()

            # Compute quality
            Q_i = np.exp(-((delta_opt + sigma[i] * eps_i) ** 2) / s[i])

            # Update remaining reliability
            R = R - w_bar[i] * Q_i

        # R is now F* - F, so loss = R^2
        losses[sim] = R ** 2

    mean_loss = float(losses.mean())
    std_loss = float(losses.std() / np.sqrt(N_simulations))

    result.validation_L_min = mean_loss
    result.validation_std = std_loss

    return mean_loss, std_loss


def extract_process_sigma_from_surrogate(surrogate, process_chain, scenario_idx=0, n_samples=500):
    """
    Extract per-process noise std (sigma_i) from the uncertainty predictor.

    For each process, runs forward passes through the process chain and
    collects sqrt(predicted_variance) as the noise scale.

    Args:
        surrogate: ProTSurrogate instance.
        process_chain: ProcessChain instance.
        scenario_idx: Scenario to evaluate on (default 0 = target).
        n_samples: Number of forward passes.

    Returns:
        Dict mapping process_name -> mean sigma (std of predicted distribution).
    """
    import torch

    sigma_per_process = {}

    with torch.no_grad():
        process_chain.eval()
        sigma_accum = {}

        for _ in range(n_samples):
            trajectory = process_chain.forward(batch_size=1, scenario_idx=scenario_idx)
            for proc_name, data in trajectory.items():
                var = data['outputs_var'].mean().item()
                sigma_val = np.sqrt(max(var, 1e-10))
                if proc_name not in sigma_accum:
                    sigma_accum[proc_name] = []
                sigma_accum[proc_name].append(sigma_val)

        for proc_name, vals in sigma_accum.items():
            sigma_per_process[proc_name] = float(np.mean(vals))

    return sigma_per_process


def compute_bellman_l_min_from_surrogate(
    surrogate,
    process_chain=None,
    loss_scale: float = 100.0,
    level: int = 1,
    N_R: int = 200,
    N_delta: int = 100,
    K: int = 2000,
    validate: bool = True,
    process_names: Optional[List[str]] = None,
    sigma_override: Optional[Dict[str, float]] = None,
) -> BellmanLMinResult:
    """
    Compute Bellman L_min using parameters extracted from the surrogate.

    This is the main entry point that extracts w_bar, sigma, s from the
    surrogate's PROCESS_CONFIGS and optional process_chain, then calls
    compute_l_min_level1 or compute_l_min_level3.

    Args:
        surrogate: ProTSurrogate instance (must have F_star and PROCESS_CONFIGS).
        process_chain: ProcessChain instance (used to estimate sigma if sigma_override not given).
        loss_scale: Loss scale factor (typically 100.0).
        level: 1 for independent noise, 3 for correlated noise.
        N_R: Grid points for R.
        N_delta: Grid points for delta.
        K: Monte Carlo samples (used for Level 3 and forward validation).
        validate: If True, run forward simulation to validate.
        process_names: Ordered list of process names. If None, uses default order.
        sigma_override: Dict of process_name -> sigma to override automatic estimation.

    Returns:
        BellmanLMinResult.
    """
    if process_names is None:
        process_names = ['laser', 'plasma', 'galvanic', 'microetch']

    configs = surrogate.PROCESS_CONFIGS
    F_star = surrogate.F_star

    # Extract weights and normalize
    weights = np.array([configs[p]['weight'] for p in process_names])
    total_weight = weights.sum()
    w_bar = weights / total_weight

    # Extract scale parameters
    s = np.array([configs[p]['scale'] for p in process_names])

    # Extract or estimate sigma per process
    if sigma_override is not None:
        sigma_arr = np.array([sigma_override.get(p, 0.1) for p in process_names])
    elif process_chain is not None:
        sigma_dict = extract_process_sigma_from_surrogate(surrogate, process_chain)
        sigma_arr = np.array([sigma_dict.get(p, 0.1) for p in process_names])
    else:
        # Fallback: small default sigma
        sigma_arr = np.full(len(process_names), 0.1)

    print(f"\n  Bellman L_min (Level {level}):")
    print(f"    F* = {F_star:.6f}")
    print(f"    Processes: {process_names}")
    print(f"    w_bar = {w_bar}")
    print(f"    sigma = {sigma_arr}")
    print(f"    s     = {s}")

    if level == 1:
        result = compute_l_min_level1(
            F_star=F_star,
            w_bar=w_bar,
            sigma=sigma_arr,
            s=s,
            loss_scale=loss_scale,
            N_R=N_R,
            N_delta=N_delta,
        )
    elif level == 3:
        # For level 3, we need the covariance matrix
        # Default to identity if not available
        n = len(process_names)
        Sigma = np.eye(n)
        print(f"    Sigma = identity (Level 3 default)")

        result = compute_l_min_level3(
            F_star=F_star,
            w_bar=w_bar,
            sigma=sigma_arr,
            s=s,
            Sigma=Sigma,
            loss_scale=loss_scale,
            N_R=N_R,
            N_delta=min(N_delta, 80),
            K=min(K, 1000),
        )
    else:
        raise ValueError(f"level must be 1 or 3, got {level}")

    print(f"    L_min (unscaled) = {result.L_min:.8f}")
    print(f"    L_min (scaled)   = {result.L_min_scaled:.6f}")

    if validate and level == 1:
        print(f"    Running forward validation (10000 simulations)...")
        val_mean, val_std = validate_bellman_policy(result)
        print(f"    Validation L_min = {val_mean:.8f} +/- {val_std:.8f}")
        rel_err = abs(val_mean - result.L_min) / max(result.L_min, 1e-12)
        print(f"    Relative error   = {rel_err:.4%}")

    return result
