"""
Empirical Loss Analysis for Reliability-based Controller Optimization.

Collects empirical statistics (E[F], Var[F], Bias²) from stochastic
forward passes through the process chain.

F samples are collected by running N forward passes with the
reparameterization trick (o = μ + ε·σ, ε ~ N(0,1)).
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json


@dataclass
class EmpiricalStats:
    """Empirical statistics from stochastic forward passes."""
    E_F: float             # Expected value of F
    E_F2: float            # Expected value of F^2
    Var_F: float           # Variance of F
    Bias2: float           # Bias squared (E[F] - F*)^2
    F_star: float          # Target reliability

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'E_F': self.E_F,
            'E_F2': self.E_F2,
            'Var_F': self.Var_F,
            'Bias2': self.Bias2,
            'F_star': self.F_star,
        }


def compute_empirical_stats(
    F_samples: np.ndarray,
    F_star: float,
    loss_scale: float = 1.0
) -> EmpiricalStats:
    """
    Compute empirical statistics from forward-pass samples.

    Args:
        F_samples: Array of F values from N stochastic forward passes.
                   Should contain at least ~100 samples for stable estimates.
        F_star: Target reliability (deterministic F*).
        loss_scale: Scale factor (training typically uses 100.0).

    Returns:
        EmpiricalStats populated with empirical values.
    """
    F_samples = np.asarray(F_samples)

    if len(F_samples) == 0:
        return EmpiricalStats(
            E_F=F_star, E_F2=F_star**2,
            Var_F=0.0, Bias2=0.0, F_star=F_star,
        )

    E_F = float(np.mean(F_samples))
    E_F2 = float(np.mean(F_samples**2))
    Var_F = float(np.var(F_samples))
    Bias2 = (E_F - F_star) ** 2

    return EmpiricalStats(
        E_F=E_F,
        E_F2=E_F2,
        Var_F=Var_F * loss_scale,
        Bias2=Bias2 * loss_scale,
        F_star=F_star,
    )


@dataclass
class TheoreticalLossTracker:
    """
    Tracks empirical statistics throughout training.

    Collects observed loss and empirical E[F], Var[F], Bias² per epoch.
    """

    # Process parameters (informational, from surrogate.PROCESS_CONFIGS)
    process_configs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    process_weights: Dict[str, float] = field(default_factory=dict)
    loss_scale: float = 100.0

    # History tracking
    epochs: List[int] = field(default_factory=list)
    observed_loss: List[float] = field(default_factory=list)

    # Empirical statistics
    empirical_E_F: List[float] = field(default_factory=list)
    empirical_Var_F: List[float] = field(default_factory=list)
    empirical_Bias2: List[float] = field(default_factory=list)

    # Per-epoch sigma2 (informational)
    sigma2_per_epoch: List[float] = field(default_factory=list)

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
        Record observed loss and empirical stats for this epoch.

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

    def get_final_summary(self) -> Dict[str, Any]:
        """Get summary statistics at end of training."""
        if len(self.epochs) == 0:
            return {}

        final_idx = -1
        best_idx = int(np.argmin(self.observed_loss)) if self.observed_loss else -1

        return {
            'final_loss': self.observed_loss[final_idx],
            'best_loss': self.observed_loss[best_idx] if best_idx >= 0 else 0.0,
            'empirical_E_F_final': self.empirical_E_F[final_idx],
            'empirical_Var_F_final': self.empirical_Var_F[final_idx],
            'total_epochs': len(self.epochs),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire tracker to dictionary for serialization."""
        return {
            'process_configs': self.process_configs,
            'process_weights': self.process_weights,
            'loss_scale': self.loss_scale,
            'epochs': self.epochs,
            'observed_loss': self.observed_loss,
            'empirical_E_F': self.empirical_E_F,
            'empirical_Var_F': self.empirical_Var_F,
            'empirical_Bias2': self.empirical_Bias2,
            'sigma2_per_epoch': self.sigma2_per_epoch,
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
        tracker.empirical_E_F = data.get('empirical_E_F', [])
        tracker.empirical_Var_F = data.get('empirical_Var_F', [])
        tracker.empirical_Bias2 = data.get('empirical_Bias2', [])
        tracker.sigma2_per_epoch = data.get('sigma2_per_epoch', [])

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
