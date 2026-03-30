"""
Empirical Loss Analysis for Reliability-based Controller Optimization.

Computes the minimum achievable loss (L_min) empirically from stochastic
forward passes through the process chain.

L_min = (Var[F] + (E[F] - F*)²) × loss_scale

Where F samples are collected by running N forward passes with the
reparameterization trick (o = μ + ε·σ, ε ~ N(0,1)).
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json


@dataclass
class TheoreticalLossComponents:
    """Components of L_min analysis."""
    L_min: float           # Minimum achievable loss
    E_F: float             # Expected value of F
    E_F2: float            # Expected value of F^2
    Var_F: float           # Variance of F
    Bias2: float           # Bias squared (E[F] - F*)^2
    F_star: float          # Target reliability
    sigma2: float          # Predicted variance (informational)
    delta: float           # Not used (kept for interface compatibility)
    s: float               # Not used (kept for interface compatibility)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'L_min': self.L_min,
            'E_F': self.E_F,
            'E_F2': self.E_F2,
            'Var_F': self.Var_F,
            'Bias2': self.Bias2,
            'F_star': self.F_star,
            'sigma2': self.sigma2,
            'delta': self.delta,
            's': self.s
        }


def compute_empirical_L_min(
    F_samples: np.ndarray,
    F_star: float,
    loss_scale: float = 1.0
) -> TheoreticalLossComponents:
    """
    Compute L_min empirically from forward-pass samples.

        L_min = (Var(F_samples) + (mean(F_samples) - F*)²) × loss_scale

    Args:
        F_samples: Array of F values from N stochastic forward passes.
                   Should contain at least ~100 samples for stable estimates.
        F_star: Target reliability (deterministic F*).
        loss_scale: Scale factor for the loss (training typically uses 100.0).

    Returns:
        TheoreticalLossComponents populated with empirical values.
    """
    F_samples = np.asarray(F_samples)

    if len(F_samples) == 0:
        return TheoreticalLossComponents(
            L_min=0.0, E_F=F_star, E_F2=F_star**2,
            Var_F=0.0, Bias2=0.0, F_star=F_star,
            sigma2=0.0, delta=0.0, s=0.0
        )

    E_F = float(np.mean(F_samples))
    E_F2 = float(np.mean(F_samples**2))
    Var_F = float(np.var(F_samples))
    Bias2 = (E_F - F_star) ** 2
    L_min = (Var_F + Bias2) * loss_scale

    return TheoreticalLossComponents(
        L_min=L_min,
        E_F=E_F,
        E_F2=E_F2,
        Var_F=Var_F * loss_scale,
        Bias2=Bias2 * loss_scale,
        F_star=F_star,
        sigma2=0.0,
        delta=0.0,
        s=0.0
    )


@dataclass
class TheoreticalLossTracker:
    """
    Tracks loss analysis throughout training.

    Collects observed loss per epoch. L_min is computed empirically
    at the end of training and backfilled.
    """

    # Process parameters (informational, from surrogate.PROCESS_CONFIGS)
    process_configs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    process_weights: Dict[str, float] = field(default_factory=dict)
    loss_scale: float = 100.0

    # History tracking
    epochs: List[int] = field(default_factory=list)
    observed_loss: List[float] = field(default_factory=list)
    theoretical_L_min: List[float] = field(default_factory=list)
    gap: List[float] = field(default_factory=list)
    efficiency: List[float] = field(default_factory=list)

    # Empirical statistics
    empirical_E_F: List[float] = field(default_factory=list)
    empirical_Var_F: List[float] = field(default_factory=list)
    empirical_Bias2: List[float] = field(default_factory=list)

    # Kept for interface compatibility (populated by backfill)
    theoretical_E_F: List[float] = field(default_factory=list)
    theoretical_Var_F: List[float] = field(default_factory=list)
    theoretical_Bias2: List[float] = field(default_factory=list)

    # Per-epoch sigma2 (informational)
    sigma2_per_epoch: List[float] = field(default_factory=list)

    # Validation counters
    n_violations: int = 0

    def update(
        self,
        epoch: int,
        observed_loss_value: float,
        F_star: float,
        F_samples: np.ndarray,
        sigma2_mean: float,
        **kwargs
    ):
        """
        Record observed loss for this epoch.

        L_min, gap, and efficiency are placeholder zeros here.
        They get backfilled after training with the empirical L_min.

        Args:
            epoch: Current epoch number
            observed_loss_value: Observed reliability loss
            F_star: Target reliability
            F_samples: Array of F values (used for empirical E[F])
            sigma2_mean: Mean predicted variance (informational)
        """
        self.epochs.append(epoch)
        self.observed_loss.append(observed_loss_value)
        self.sigma2_per_epoch.append(sigma2_mean)

        # Placeholder — will be overwritten by backfill
        self.theoretical_L_min.append(0.0)
        self.theoretical_E_F.append(0.0)
        self.theoretical_Var_F.append(0.0)
        self.theoretical_Bias2.append(0.0)

        # Empirical stats from F_samples
        if len(F_samples) > 0:
            empirical_mean = np.mean(F_samples)
            empirical_var = np.var(F_samples)
            empirical_bias2 = (empirical_mean - F_star)**2
        else:
            empirical_mean = F_star
            empirical_var = 0.0
            empirical_bias2 = 0.0

        self.empirical_E_F.append(empirical_mean)
        self.empirical_Var_F.append(empirical_var * self.loss_scale)
        self.empirical_Bias2.append(empirical_bias2 * self.loss_scale)

        # Placeholder
        self.gap.append(0.0)
        self.efficiency.append(0.0)

    def get_final_summary(self) -> Dict[str, Any]:
        """Get summary statistics at end of training."""
        if len(self.epochs) == 0:
            return {}

        final_idx = -1
        best_idx = int(np.argmin(self.observed_loss)) if self.observed_loss else -1

        return {
            'final_loss': self.observed_loss[final_idx],
            'best_loss': self.observed_loss[best_idx] if best_idx >= 0 else 0.0,
            'final_L_min': self.theoretical_L_min[final_idx],
            'final_gap': self.gap[final_idx],
            'final_efficiency': self.efficiency[final_idx],
            'best_efficiency': max(self.efficiency) if self.efficiency else 0.0,
            'mean_efficiency': float(np.mean(self.efficiency)) if self.efficiency else 0.0,
            'empirical_E_F_final': self.empirical_E_F[final_idx],
            'theoretical_E_F_final': self.theoretical_E_F[final_idx],
            'empirical_Var_F_final': self.empirical_Var_F[final_idx],
            'theoretical_Var_F_final': self.theoretical_Var_F[final_idx],
            'n_violations': self.n_violations,
            'total_epochs': len(self.epochs),
            'violation_rate': self.n_violations / len(self.epochs) if self.epochs else 0.0,
            'epoch_90_efficiency': self._find_efficiency_epoch(0.9),
            'epoch_95_efficiency': self._find_efficiency_epoch(0.95),
        }

    def _find_efficiency_epoch(self, threshold: float) -> Optional[int]:
        """Find first epoch where efficiency >= threshold."""
        for i, eff in enumerate(self.efficiency):
            if eff >= threshold:
                return self.epochs[i]
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire tracker to dictionary for serialization."""
        return {
            'process_configs': self.process_configs,
            'process_weights': self.process_weights,
            'loss_scale': self.loss_scale,
            'epochs': self.epochs,
            'observed_loss': self.observed_loss,
            'theoretical_L_min': self.theoretical_L_min,
            'gap': self.gap,
            'efficiency': self.efficiency,
            'empirical_E_F': self.empirical_E_F,
            'empirical_Var_F': self.empirical_Var_F,
            'empirical_Bias2': self.empirical_Bias2,
            'theoretical_E_F': self.theoretical_E_F,
            'theoretical_Var_F': self.theoretical_Var_F,
            'theoretical_Bias2': self.theoretical_Bias2,
            'sigma2_per_epoch': self.sigma2_per_epoch,
            'n_violations': self.n_violations,
            'summary': self.get_final_summary()
        }

    def save(self, path: Path):
        """Save tracker to JSON file."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'TheoreticalLossTracker':
        """Load tracker from JSON file."""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)

        tracker = cls(
            process_configs=data.get('process_configs', {}),
            process_weights=data.get('process_weights', {}),
            loss_scale=data.get('loss_scale', 100.0)
        )
        tracker.epochs = data.get('epochs', [])
        tracker.observed_loss = data.get('observed_loss', [])
        tracker.theoretical_L_min = data.get('theoretical_L_min', [])
        tracker.gap = data.get('gap', [])
        tracker.efficiency = data.get('efficiency', [])
        tracker.empirical_E_F = data.get('empirical_E_F', [])
        tracker.empirical_Var_F = data.get('empirical_Var_F', [])
        tracker.empirical_Bias2 = data.get('empirical_Bias2', [])
        tracker.theoretical_E_F = data.get('theoretical_E_F', [])
        tracker.theoretical_Var_F = data.get('theoretical_Var_F', [])
        tracker.theoretical_Bias2 = data.get('theoretical_Bias2', [])
        tracker.sigma2_per_epoch = data.get('sigma2_per_epoch', [])
        tracker.n_violations = data.get('n_violations', 0)

        return tracker


def run_validation_sampling(
    process_chain,
    surrogate,
    scenario_idx: int,
    n_samples: int = 100,
    batch_size: int = 1
) -> Tuple[np.ndarray, float, float]:
    """
    Run multiple forward passes to estimate empirical E[F] and Var[F].

    Args:
        process_chain: ProcessChain instance
        surrogate: ProTSurrogate instance
        scenario_idx: Which scenario to evaluate
        n_samples: Number of forward passes
        batch_size: Batch size per pass

    Returns:
        Tuple of (F_samples array, mean_sigma2, F_star)
    """
    F_samples = []
    sigma2_samples = []

    with torch.no_grad():
        process_chain.eval()

        for _ in range(n_samples):
            trajectory = process_chain.forward(batch_size=batch_size, scenario_idx=scenario_idx)
            F = surrogate.compute_reliability(trajectory).item()
            F_samples.append(F)

            for proc_name, data in trajectory.items():
                sigma2 = data['outputs_var'].mean().item()
                sigma2_samples.append(sigma2)

    F_star = surrogate.F_star
    mean_sigma2 = np.mean(sigma2_samples)

    return np.array(F_samples), mean_sigma2, F_star


@dataclass
class LambdaMCResult:
    """Results from Monte Carlo Lambda estimation (per-trajectory variance method).

    Decomposes the irreducible loss floor into variance and bias components
    by fixing controller actions and re-sampling outputs S times per trajectory.

    Lambda_MC = E_i[Var^(i)] + E_i[Bias^(i)²]
    where:
        Var^(i)  = Var_s[F(tau_tilde^(i,s))]           — per-trajectory variance
        Bias^(i) = E_s[F(tau_tilde^(i,s))] - F*        — per-trajectory bias
    """
    lambda_mc: float            # Total: E[Var] + E[Bias²], scaled
    lambda_mc_var: float        # Variance component: E[Var^(i)], scaled
    lambda_mc_bias: float       # Bias component: E[Bias^(i)²], scaled
    n_trajectories: int         # Number of observed trajectories (N)
    n_perturbations: int        # Perturbations per trajectory (S)
    per_trajectory_var: np.ndarray   # (N,) unscaled per-trajectory variances
    per_trajectory_bias2: np.ndarray # (N,) unscaled per-trajectory bias²
    per_trajectory_E_F: np.ndarray   # (N,) per-trajectory E[F]
    F_star: float
    loss_scale: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'lambda_mc': self.lambda_mc,
            'lambda_mc_var': self.lambda_mc_var,
            'lambda_mc_bias': self.lambda_mc_bias,
            'n_trajectories': self.n_trajectories,
            'n_perturbations': self.n_perturbations,
            'per_trajectory_var': self.per_trajectory_var.tolist(),
            'per_trajectory_bias2': self.per_trajectory_bias2.tolist(),
            'per_trajectory_E_F': self.per_trajectory_E_F.tolist(),
            'F_star': self.F_star,
            'loss_scale': self.loss_scale,
        }

    def save(self, path: Path):
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def compute_lambda_mc(
    process_chain,
    surrogate,
    n_scenarios: int,
    S: int = 100,
    loss_scale: float = 100.0,
    verbose: bool = False,
) -> LambdaMCResult:
    """
    Estimate L_min via Monte Carlo per-trajectory variance (Lambda method).

    For each observed trajectory tau^(i) (one per scenario):
      1. Forward pass through the chain to obtain (mu_t, sigma2_t) per stage.
      2. Draw S perturbed trajectories by re-sampling outputs from N(mu_t, sigma2_t),
         keeping actions fixed.
      3. Evaluate F on each perturbed trajectory.
      4. Compute per-trajectory Var^(i) and Bias^(i).

    Lambda_MC = ( E_i[Var^(i)] + E_i[(E_i[F] - F*)²] ) × loss_scale

    This decomposes the irreducible floor into:
      - Variance component (pure stochastic noise)
      - Bias component (systematic offset from F*)

    Args:
        process_chain: ProcessChain with frozen uncertainty predictors.
        surrogate: Surrogate model (ProTSurrogate or CasualiTSurrogate).
        n_scenarios: Number of scenarios (trajectories) to evaluate.
        S: Number of perturbations per trajectory (50-200 recommended).
        loss_scale: Scale factor matching training loss (default 100.0).
        verbose: Print progress.

    Returns:
        LambdaMCResult with decomposed Lambda_MC estimate.
    """
    F_star = surrogate.F_star
    per_traj_var = []
    per_traj_bias2 = []
    per_traj_E_F = []

    with torch.no_grad():
        process_chain.eval()

        for i in range(n_scenarios):
            # Step 1: single forward pass to get (mu, sigma2) per stage
            trajectory = process_chain.forward(batch_size=1, scenario_idx=i)

            # Extract per-process (mu, sigma2) — these are fixed for all S perturbations
            frozen_params = {}
            for proc_name, data in trajectory.items():
                frozen_params[proc_name] = {
                    'inputs': data['inputs'],               # (1, input_dim)
                    'outputs_mean': data['outputs_mean'],    # (1, output_dim)
                    'outputs_var': data['outputs_var'],      # (1, output_dim)
                }

            # Step 2-3: draw S perturbed trajectories and evaluate F
            F_samples = []
            for _ in range(S):
                perturbed_traj = {}
                for proc_name, params in frozen_params.items():
                    mu = params['outputs_mean']
                    var = params['outputs_var']
                    std = torch.sqrt(var + 1e-8)
                    eps = torch.randn_like(mu)
                    o_tilde = mu + eps * std

                    perturbed_traj[proc_name] = {
                        'inputs': params['inputs'],
                        'outputs_mean': mu,
                        'outputs_var': var,
                        'outputs_sampled': o_tilde,
                    }

                F_val = surrogate.compute_reliability(perturbed_traj).item()
                F_samples.append(F_val)

            F_arr = np.array(F_samples)

            # Step 4: per-trajectory statistics
            E_F_i = float(np.mean(F_arr))
            Var_i = float(np.var(F_arr, ddof=1)) if S > 1 else 0.0
            Bias2_i = (E_F_i - F_star) ** 2

            per_traj_var.append(Var_i)
            per_traj_bias2.append(Bias2_i)
            per_traj_E_F.append(E_F_i)

            if verbose and (i + 1) % max(1, n_scenarios // 5) == 0:
                print(f"    Lambda_MC: scenario {i+1}/{n_scenarios}  "
                      f"Var={Var_i:.8f}  Bias²={(Bias2_i):.8f}  E[F]={E_F_i:.6f}")

    per_traj_var = np.array(per_traj_var)
    per_traj_bias2 = np.array(per_traj_bias2)
    per_traj_E_F = np.array(per_traj_E_F)

    # Aggregate: average over trajectories, then scale
    mean_var = float(np.mean(per_traj_var))
    mean_bias2 = float(np.mean(per_traj_bias2))

    lambda_mc_var = mean_var * loss_scale
    lambda_mc_bias = mean_bias2 * loss_scale
    lambda_mc = lambda_mc_var + lambda_mc_bias

    if verbose:
        print(f"    Lambda_MC total:  {lambda_mc:.6f}")
        print(f"      Var component:  {lambda_mc_var:.6f}")
        print(f"      Bias component: {lambda_mc_bias:.6f}")
        print(f"    ({len(per_traj_var)} trajectories × {S} perturbations)")

    return LambdaMCResult(
        lambda_mc=lambda_mc,
        lambda_mc_var=lambda_mc_var,
        lambda_mc_bias=lambda_mc_bias,
        n_trajectories=n_scenarios,
        n_perturbations=S,
        per_trajectory_var=per_traj_var,
        per_trajectory_bias2=per_traj_bias2,
        per_trajectory_E_F=per_traj_E_F,
        F_star=F_star,
        loss_scale=loss_scale,
    )


def compute_z_score(empirical: float, theoretical: float, std: float, n_samples: int) -> float:
    """
    Compute z-score for comparing empirical vs theoretical values.

    z = (empirical - theoretical) / (std / sqrt(n))
    """
    if std <= 0 or n_samples <= 0:
        return 0.0

    se = std / np.sqrt(n_samples)
    if se <= 0:
        return 0.0

    return (empirical - theoretical) / se


def format_status(value: float, thresholds: Tuple[float, float] = (0.05, 0.20)) -> str:
    """Format status based on relative difference."""
    good_thresh, warn_thresh = thresholds
    if abs(value) < good_thresh:
        return "OK"
    elif abs(value) < warn_thresh:
        return "WARN"
    else:
        return "MISMATCH"
