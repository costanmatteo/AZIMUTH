"""
Loss Analysis for Reliability-based Controller Optimization.

Computes the minimum achievable loss (L_min) via two methods:

1. **Bellman backward induction** (primary): True L_min accounting for an
   optimal reactive controller using dynamic programming.
   L_min = V_1(F*) where V_i is the Bellman value function.

2. **Empirical feedforward** (naive upper bound): L_min from stochastic
   forward passes with the current (non-reactive) controller.
   L_min_naive = (Var[F] + (E[F] - F*)^2) * loss_scale

The Bellman L_min is always <= the naive empirical L_min.
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json

from .bellman_l_min import (
    compute_l_min_level1,
    compute_bellman_l_min_from_surrogate,
    validate_bellman_policy,
    BellmanLMinResult,
)


@dataclass
class TheoreticalLossComponents:
    """Components of L_min analysis."""
    L_min: float           # Minimum achievable loss (Bellman)
    L_min_naive: float     # Naive empirical L_min (feedforward)
    E_F: float             # Expected value of F (from empirical samples)
    E_F2: float            # Expected value of F^2
    Var_F: float           # Variance of F
    Bias2: float           # Bias squared (E[F] - F*)^2
    F_star: float          # Target reliability
    sigma2: float          # Predicted variance (informational)
    delta: float           # Not used (kept for interface compatibility)
    s: float               # Not used (kept for interface compatibility)
    bellman_result: Optional[BellmanLMinResult] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        d = {
            'L_min': self.L_min,
            'L_min_naive': self.L_min_naive,
            'E_F': self.E_F,
            'E_F2': self.E_F2,
            'Var_F': self.Var_F,
            'Bias2': self.Bias2,
            'F_star': self.F_star,
            'sigma2': self.sigma2,
            'delta': self.delta,
            's': self.s,
        }
        if self.bellman_result is not None:
            d['bellman'] = self.bellman_result.to_dict()
        return d


def compute_empirical_L_min(
    F_samples: np.ndarray,
    F_star: float,
    loss_scale: float = 1.0,
    surrogate=None,
    process_chain=None,
    sigma_override: Optional[Dict[str, float]] = None,
) -> TheoreticalLossComponents:
    """
    Compute L_min using Bellman backward induction + empirical statistics.

    The primary L_min is computed via Bellman dynamic programming (Level 1:
    independent noise, fixed variance). The naive empirical L_min from
    forward-pass samples is also computed for comparison.

    Args:
        F_samples: Array of F values from N stochastic forward passes.
        F_star: Target reliability (deterministic F*).
        loss_scale: Scale factor for the loss (training typically uses 100.0).
        surrogate: ProTSurrogate instance (for Bellman computation).
                   If None, falls back to empirical-only.
        process_chain: ProcessChain instance (for sigma estimation).
        sigma_override: Dict of process_name -> sigma to override estimation.

    Returns:
        TheoreticalLossComponents with Bellman L_min as primary and naive as reference.
    """
    F_samples = np.asarray(F_samples)

    # Empirical statistics (always computed)
    if len(F_samples) == 0:
        E_F, E_F2, Var_F, Bias2 = F_star, F_star ** 2, 0.0, 0.0
        L_min_naive = 0.0
    else:
        E_F = float(np.mean(F_samples))
        E_F2 = float(np.mean(F_samples ** 2))
        Var_F = float(np.var(F_samples))
        Bias2 = (E_F - F_star) ** 2
        L_min_naive = (Var_F + Bias2) * loss_scale

    # Bellman L_min (primary)
    bellman_result = None
    L_min_bellman = L_min_naive  # fallback

    if surrogate is not None:
        try:
            bellman_result = compute_bellman_l_min_from_surrogate(
                surrogate=surrogate,
                process_chain=process_chain,
                loss_scale=loss_scale,
                level=1,
                N_R=200,
                N_delta=100,
                K=2000,
                validate=True,
                sigma_override=sigma_override,
            )
            L_min_bellman = bellman_result.L_min_scaled

            # Sanity checks
            if L_min_bellman > L_min_naive * 1.01:
                print(f"  WARNING: Bellman L_min ({L_min_bellman:.6f}) > naive L_min ({L_min_naive:.6f}).")
                print(f"           This may indicate grid resolution issues. Using naive as fallback.")
                L_min_bellman = L_min_naive
            if L_min_bellman < 0:
                print(f"  WARNING: Bellman L_min is negative ({L_min_bellman:.6f}). Clamping to 0.")
                L_min_bellman = 0.0

            print(f"\n  Bellman vs Naive comparison:")
            print(f"    L_min (Bellman):  {L_min_bellman:.6f}")
            print(f"    L_min (naive):    {L_min_naive:.6f}")
            if L_min_naive > 0:
                print(f"    Ratio:            {L_min_bellman / L_min_naive:.4f}")
        except Exception as e:
            print(f"  WARNING: Bellman L_min computation failed: {e}")
            print(f"           Falling back to naive empirical L_min.")
            L_min_bellman = L_min_naive

    return TheoreticalLossComponents(
        L_min=L_min_bellman,
        L_min_naive=L_min_naive,
        E_F=E_F,
        E_F2=E_F2,
        Var_F=Var_F * loss_scale,
        Bias2=Bias2 * loss_scale,
        F_star=F_star,
        sigma2=0.0,
        delta=0.0,
        s=0.0,
        bellman_result=bellman_result,
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
