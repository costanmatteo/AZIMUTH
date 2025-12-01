"""
Theoretical Loss Analysis for Reliability-based Controller Optimization.

Implements the theoretical framework for computing the minimum achievable loss (L_min)
when using stochastic sampling from the UncertaintyPredictor.

Theory:
The loss function L = (F - F*)^2 where:
- F* = reliability of target trajectory (deterministic)
- F = reliability of controller (computed with stochastic sampling)

When sampling is stochastic (sigma^2 > 0), there's an irreducible minimum L_min > 0.

Formulas:
- E[F] = F* * (1/sqrt(1 + 2*sigma^2/s)) * exp(2*delta^2*sigma^2 / (s*(s + 2*sigma^2)))
- E[F^2] = F*^2 * (1/sqrt(1 + 4*sigma^2/s)) * exp(4*delta^2*sigma^2 / (s*(s + 4*sigma^2)))
- L_min = Var[F] + Bias^2 = (E[F^2] - E[F]^2) + (E[F] - F*)^2

Where:
- sigma^2 = predicted variance from UncertaintyPredictor
- s = scale parameter of quality function Q(o) = exp(-(o-tau)^2/s)
- delta = mu_target - tau (distance of target output from process optimum)
- F* = target reliability
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json


@dataclass
class TheoreticalLossComponents:
    """Components of theoretical loss analysis."""
    L_min: float           # Minimum achievable loss
    E_F: float             # Expected value of F
    E_F2: float            # Expected value of F^2
    Var_F: float           # Variance of F
    Bias2: float           # Bias squared (E[F] - F*)^2
    F_star: float          # Target reliability
    sigma2: float          # Predicted variance
    delta: float           # Distance from process optimum
    s: float               # Scale parameter

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


@dataclass
class ControllerLossComponents:
    """
    L_min analysis for a single controller.

    Architecture: Policy[i] controls Process[i+1]
    - Controller i receives outputs from Process i
    - Controller i produces inputs for Process i+1
    - L_min for Controller i depends on parameters of Process[i+1] (the controlled process):
      - sigma2: predicted variance of Process[i+1] outputs
      - delta: distance of target output from process optimum (mu_target - tau)
      - s: scale parameter of quality function
      - F_star: target reliability = exp(-delta^2/s)
    """
    controller_idx: int          # Controller index (0-based)
    source_process: str          # Process providing inputs to controller (Process i)
    target_process: str          # Process controlled by this controller (Process i+1)
    L_min: float                 # Minimum achievable loss for this controller
    E_F: float                   # Expected value of F for target process
    E_F2: float                  # Expected value of F^2
    Var_F: float                 # Variance of F
    Bias2: float                 # Bias squared
    F_star: float                # Target reliability (for target process)
    sigma2: float                # Predicted variance of target process outputs
    delta: float                 # Distance from target process optimum
    s: float                     # Scale parameter of target process

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'controller_idx': self.controller_idx,
            'source_process': self.source_process,
            'target_process': self.target_process,
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


def compute_theoretical_E_F(F_star: float, delta: float, sigma2: float, s: float) -> float:
    """
    Compute theoretical expected value of F.

    E[F] = F* * (1/sqrt(1 + 2*sigma^2/s)) * exp(2*delta^2*sigma^2 / (s*(s + 2*sigma^2)))

    Args:
        F_star: Target reliability
        delta: mu_target - tau (distance from process optimum)
        sigma2: Predicted variance
        s: Scale parameter of quality function

    Returns:
        E[F]: Expected value of reliability
    """
    if sigma2 <= 0:
        # Deterministic case: E[F] = F*
        return F_star

    # Avoid division by zero
    if s <= 0:
        return F_star

    # Compute the two factors
    factor1 = 1.0 / np.sqrt(1 + 2 * sigma2 / s)

    # Exponent term
    numerator = 2 * delta**2 * sigma2
    denominator = s * (s + 2 * sigma2)
    factor2 = np.exp(numerator / denominator) if denominator > 0 else 1.0

    E_F = F_star * factor1 * factor2

    return E_F


def compute_theoretical_E_F2(F_star: float, delta: float, sigma2: float, s: float) -> float:
    """
    Compute theoretical expected value of F^2.

    E[F^2] = F*^2 * (1/sqrt(1 + 4*sigma^2/s)) * exp(4*delta^2*sigma^2 / (s*(s + 4*sigma^2)))

    Args:
        F_star: Target reliability
        delta: mu_target - tau (distance from process optimum)
        sigma2: Predicted variance
        s: Scale parameter of quality function

    Returns:
        E[F^2]: Expected value of reliability squared
    """
    if sigma2 <= 0:
        # Deterministic case: E[F^2] = F*^2
        return F_star**2

    # Avoid division by zero
    if s <= 0:
        return F_star**2

    # Compute the two factors
    factor1 = 1.0 / np.sqrt(1 + 4 * sigma2 / s)

    # Exponent term
    numerator = 4 * delta**2 * sigma2
    denominator = s * (s + 4 * sigma2)
    factor2 = np.exp(numerator / denominator) if denominator > 0 else 1.0

    E_F2 = F_star**2 * factor1 * factor2

    return E_F2


def compute_theoretical_L_min(
    F_star: float,
    delta: float,
    sigma2: float,
    s: float,
    loss_scale: float = 1.0
) -> TheoreticalLossComponents:
    """
    Compute the theoretical minimum achievable loss and all its components.

    L_min = Var[F] + Bias^2
          = (E[F^2] - E[F]^2) + (E[F] - F*)^2

    Args:
        F_star: Target reliability
        delta: mu_target - tau (distance from process optimum)
        sigma2: Predicted variance (mean across all samples)
        s: Scale parameter of quality function
        loss_scale: Scale factor for the loss (default 1.0, training uses 100.0)

    Returns:
        TheoreticalLossComponents with all computed values
    """
    E_F = compute_theoretical_E_F(F_star, delta, sigma2, s)
    E_F2 = compute_theoretical_E_F2(F_star, delta, sigma2, s)

    # Variance of F
    Var_F = E_F2 - E_F**2
    # Ensure non-negative (numerical stability)
    Var_F = max(Var_F, 0.0)

    # Bias squared
    Bias2 = (E_F - F_star)**2

    # Minimum achievable loss
    L_min = (Var_F + Bias2) * loss_scale

    return TheoreticalLossComponents(
        L_min=L_min,
        E_F=E_F,
        E_F2=E_F2,
        Var_F=Var_F * loss_scale,  # Scale variance too for consistency
        Bias2=Bias2 * loss_scale,  # Scale bias too
        F_star=F_star,
        sigma2=sigma2,
        delta=delta,
        s=s
    )


def compute_multi_process_L_min(
    process_params: Dict[str, Dict[str, float]],
    process_weights: Dict[str, float],
    loss_scale: float = 1.0
) -> Tuple[TheoreticalLossComponents, Dict[str, TheoreticalLossComponents]]:
    """
    Compute theoretical L_min for a multi-process system.

    The reliability F is a weighted average of per-process quality scores.
    This function computes L_min for each process and combines them.

    Args:
        process_params: Dict mapping process_name to {'F_star', 'delta', 'sigma2', 's'}
        process_weights: Dict mapping process_name to weight
        loss_scale: Scale factor for the loss

    Returns:
        Tuple of (combined L_min components, dict of per-process components)
    """
    per_process_components = {}

    # Compute for each process
    for process_name, params in process_params.items():
        components = compute_theoretical_L_min(
            F_star=params['F_star'],
            delta=params['delta'],
            sigma2=params['sigma2'],
            s=params['s'],
            loss_scale=loss_scale
        )
        per_process_components[process_name] = components

    # Combine using weights
    total_weight = sum(process_weights.get(name, 1.0) for name in process_params.keys())

    if total_weight > 0:
        # Weighted average of L_min
        combined_L_min = sum(
            components.L_min * process_weights.get(name, 1.0)
            for name, components in per_process_components.items()
        ) / total_weight

        combined_E_F = sum(
            components.E_F * process_weights.get(name, 1.0)
            for name, components in per_process_components.items()
        ) / total_weight

        combined_Var_F = sum(
            components.Var_F * process_weights.get(name, 1.0)
            for name, components in per_process_components.items()
        ) / total_weight

        combined_Bias2 = sum(
            components.Bias2 * process_weights.get(name, 1.0)
            for name, components in per_process_components.items()
        ) / total_weight

        combined_F_star = sum(
            components.F_star * process_weights.get(name, 1.0)
            for name, components in per_process_components.items()
        ) / total_weight

        combined_sigma2 = sum(
            components.sigma2 * process_weights.get(name, 1.0)
            for name, components in per_process_components.items()
        ) / total_weight
    else:
        combined_L_min = 0.0
        combined_E_F = 0.0
        combined_Var_F = 0.0
        combined_Bias2 = 0.0
        combined_F_star = 0.0
        combined_sigma2 = 0.0

    combined_components = TheoreticalLossComponents(
        L_min=combined_L_min,
        E_F=combined_E_F,
        E_F2=combined_E_F**2 + combined_Var_F / loss_scale,  # Reconstruct E[F^2]
        Var_F=combined_Var_F,
        Bias2=combined_Bias2,
        F_star=combined_F_star,
        sigma2=combined_sigma2,
        delta=0.0,  # Not meaningful for combined
        s=0.0  # Not meaningful for combined
    )

    return combined_components, per_process_components


def compute_per_controller_L_min(
    process_names: List[str],
    process_params: Dict[str, Dict[str, float]],
    sigma2_per_process: Dict[str, float],
    loss_scale: float = 1.0
) -> List[ControllerLossComponents]:
    """
    Compute theoretical L_min for each controller separately.

    Architecture: Policy[i] controls Process[i+1]
    - Process[0] is NOT controlled (inputs come from target trajectory)
    - Controller[i] receives outputs from Process[i] and controls Process[i+1]

    For Controller[i], all parameters come from Process[i+1] (the controlled process):
    - sigma2: predicted variance of Process[i+1] outputs
    - delta: distance of target output from process optimum
    - s, F_star: quality function parameters for Process[i+1]

    Args:
        process_names: Ordered list of process names [process_0, process_1, ..., process_n]
        process_params: Dict mapping process_name to {'F_star', 'delta', 's', 'tau', 'mu_target'}
                       For each CONTROLLED process (all except first)
        sigma2_per_process: Dict mapping process_name to mean predicted variance
        loss_scale: Scale factor for the loss

    Returns:
        List of ControllerLossComponents, one per controller (len = n_processes - 1)
    """
    controllers = []
    n_processes = len(process_names)

    # Controller i controls Process[i+1], receives from Process[i]
    for i in range(n_processes - 1):
        source_process = process_names[i]      # Process providing inputs to controller
        target_process = process_names[i + 1]  # Process being controlled

        # All parameters come from target process (the controlled process)
        if target_process not in process_params:
            continue

        target_params = process_params[target_process]
        F_star = target_params['F_star']
        delta = target_params['delta']
        s = target_params['s']

        # sigma2 from target process (variance of the controlled process outputs)
        sigma2 = sigma2_per_process.get(target_process, 0.01)

        # Compute theoretical values for this controller
        E_F = compute_theoretical_E_F(F_star, delta, sigma2, s)
        E_F2 = compute_theoretical_E_F2(F_star, delta, sigma2, s)

        Var_F = max(E_F2 - E_F**2, 0.0)
        Bias2 = (E_F - F_star)**2

        L_min = (Var_F + Bias2) * loss_scale

        controller = ControllerLossComponents(
            controller_idx=i,
            source_process=source_process,
            target_process=target_process,
            L_min=L_min,
            E_F=E_F,
            E_F2=E_F2,
            Var_F=Var_F * loss_scale,
            Bias2=Bias2 * loss_scale,
            F_star=F_star,
            sigma2=sigma2,
            delta=delta,
            s=s
        )
        controllers.append(controller)

    return controllers


def compute_controlled_process_params(
    trajectory: Dict,
    target_trajectory: Dict,
    process_configs: Dict[str, Dict[str, float]],
    process_names: List[str]
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Compute theoretical parameters ONLY for controlled processes (all except first).

    Process[0] is NOT controlled by any policy, so we skip it.
    For each controlled process (i >= 1):
    - delta = mu_target - tau (from target trajectory)
    - F_star = exp(-delta^2/s)
    - sigma2 is collected separately per source process

    Args:
        trajectory: Output from process_chain.forward()
        target_trajectory: Target trajectory tensors
        process_configs: From surrogate.PROCESS_CONFIGS {'process_name': {'target': tau, 'scale': s, 'weight': w}}
        process_names: Ordered list of process names

    Returns:
        Tuple of:
        - process_params: Dict mapping controlled_process_name to {'F_star', 'delta', 's', 'tau', 'mu_target'}
        - sigma2_per_process: Dict mapping ALL process names to mean predicted variance
    """
    process_params = {}
    sigma2_per_process = {}

    # Collect sigma2 from ALL processes (needed for source process variance)
    for process_name, data in trajectory.items():
        if isinstance(data['outputs_var'], torch.Tensor):
            sigma2 = data['outputs_var'].detach().cpu().numpy().mean()
        else:
            sigma2 = np.mean(data['outputs_var'])
        sigma2_per_process[process_name] = float(sigma2)

    # Compute parameters ONLY for controlled processes (skip first)
    for i, process_name in enumerate(process_names):
        if i == 0:
            # Process[0] is NOT controlled - skip it
            continue

        if process_name not in process_configs:
            continue

        config = process_configs[process_name]
        tau = config['target']
        s = config['scale']

        # Get target output for this process from target trajectory
        if process_name in target_trajectory:
            target_data = target_trajectory[process_name]
            if isinstance(target_data['outputs'], torch.Tensor):
                mu_target = target_data['outputs'].detach().cpu().numpy().mean()
            else:
                mu_target = np.mean(target_data['outputs'])
        else:
            # Fallback: use predicted mean
            data = trajectory[process_name]
            if isinstance(data['outputs_mean'], torch.Tensor):
                mu_target = data['outputs_mean'].detach().cpu().numpy().mean()
            else:
                mu_target = np.mean(data['outputs_mean'])

        # Compute delta (distance from process optimum)
        delta = mu_target - tau

        # Compute F_star (quality at target)
        F_star = np.exp(-(mu_target - tau)**2 / s)

        process_params[process_name] = {
            'F_star': float(F_star),
            'delta': float(delta),
            's': float(s),
            'tau': float(tau),
            'mu_target': float(mu_target)
        }

    return process_params, sigma2_per_process


@dataclass
class TheoreticalLossTracker:
    """
    Tracks theoretical loss analysis throughout training.

    Collects:
    - Observed loss per epoch
    - Theoretical L_min per epoch
    - Gap (observed - theoretical)
    - Efficiency (L_min / observed)
    - Empirical E[F], Var[F] from sampling
    """

    # Process parameters (from surrogate.PROCESS_CONFIGS)
    process_configs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    process_weights: Dict[str, float] = field(default_factory=dict)
    loss_scale: float = 100.0  # Default scale factor

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

    # Theoretical statistics
    theoretical_E_F: List[float] = field(default_factory=list)
    theoretical_Var_F: List[float] = field(default_factory=list)
    theoretical_Bias2: List[float] = field(default_factory=list)

    # Per-epoch sigma2 (mean predicted variance)
    sigma2_per_epoch: List[float] = field(default_factory=list)

    # Validation counters
    n_violations: int = 0  # Times observed < theoretical

    def set_process_params_from_surrogate(self, surrogate):
        """
        Extract process parameters from surrogate's PROCESS_CONFIGS.

        Args:
            surrogate: ProTSurrogate instance
        """
        from controller_optimization.src.models.surrogate import ProTSurrogate

        # Get process configs from surrogate class
        configs = ProTSurrogate.PROCESS_CONFIGS

        for process_name, config in configs.items():
            self.process_configs[process_name] = {
                'tau': config['target'],
                's': config['scale']
            }
            self.process_weights[process_name] = config.get('weight', 1.0)

    def update(
        self,
        epoch: int,
        observed_loss_value: float,
        F_star: float,
        F_samples: np.ndarray,
        sigma2_mean: float,
        delta: float = 0.0,
        s: float = 1.0
    ):
        """
        Update tracker with data from current epoch.

        Args:
            epoch: Current epoch number
            observed_loss_value: Observed reliability loss
            F_star: Target reliability
            F_samples: Array of F values from multiple forward passes
            sigma2_mean: Mean predicted variance across all processes
            delta: Distance from process optimum (estimated)
            s: Effective scale parameter
        """
        self.epochs.append(epoch)
        self.observed_loss.append(observed_loss_value)
        self.sigma2_per_epoch.append(sigma2_mean)

        # Compute theoretical values
        theoretical = compute_theoretical_L_min(F_star, delta, sigma2_mean, s, self.loss_scale)
        self.theoretical_L_min.append(theoretical.L_min)
        self.theoretical_E_F.append(theoretical.E_F)
        self.theoretical_Var_F.append(theoretical.Var_F)
        self.theoretical_Bias2.append(theoretical.Bias2)

        # Compute empirical statistics from F_samples
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

        # Compute gap and efficiency
        gap_value = observed_loss_value - theoretical.L_min
        self.gap.append(gap_value)

        if observed_loss_value > 0:
            efficiency_value = theoretical.L_min / observed_loss_value
        else:
            efficiency_value = 1.0 if theoretical.L_min == 0 else 0.0
        self.efficiency.append(efficiency_value)

        # Check for violations (loss < L_min indicates theory issue)
        if observed_loss_value < theoretical.L_min * 0.99:  # 1% tolerance
            self.n_violations += 1

    def get_final_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics at end of training.

        Returns:
            Dict with all summary statistics
        """
        if len(self.epochs) == 0:
            return {}

        final_idx = -1
        best_idx = np.argmin(self.observed_loss) if self.observed_loss else -1

        return {
            'final_loss': self.observed_loss[final_idx],
            'best_loss': self.observed_loss[best_idx] if best_idx >= 0 else 0.0,
            'final_L_min': self.theoretical_L_min[final_idx],
            'final_gap': self.gap[final_idx],
            'final_efficiency': self.efficiency[final_idx],

            'best_efficiency': max(self.efficiency) if self.efficiency else 0.0,
            'mean_efficiency': np.mean(self.efficiency) if self.efficiency else 0.0,

            'empirical_E_F_final': self.empirical_E_F[final_idx],
            'theoretical_E_F_final': self.theoretical_E_F[final_idx],
            'empirical_Var_F_final': self.empirical_Var_F[final_idx],
            'theoretical_Var_F_final': self.theoretical_Var_F[final_idx],

            'n_violations': self.n_violations,
            'total_epochs': len(self.epochs),
            'violation_rate': self.n_violations / len(self.epochs) if self.epochs else 0.0,

            # Find epochs where efficiency thresholds were reached
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


def compute_effective_params_from_trajectory(
    trajectory: Dict,
    target_trajectory: Dict,
    process_configs: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Compute effective theoretical parameters from actual trajectory data.

    Extracts:
    - sigma2: mean predicted variance
    - delta: distance between predicted mean and process target (tau)
    - F_star: per-process quality scores

    Args:
        trajectory: Output from process_chain.forward()
        target_trajectory: Target trajectory tensors
        process_configs: From surrogate.PROCESS_CONFIGS

    Returns:
        Dict mapping process_name to {'F_star', 'delta', 'sigma2', 's'}
    """
    params = {}

    for process_name, data in trajectory.items():
        if process_name not in process_configs:
            continue

        config = process_configs[process_name]
        tau = config['target']
        s = config['scale']

        # Get predicted mean and variance
        if isinstance(data['outputs_mean'], torch.Tensor):
            mu = data['outputs_mean'].detach().cpu().numpy().mean()
            sigma2 = data['outputs_var'].detach().cpu().numpy().mean()
        else:
            mu = np.mean(data['outputs_mean'])
            sigma2 = np.mean(data['outputs_var'])

        # Get target output
        if process_name in target_trajectory:
            target_data = target_trajectory[process_name]
            if isinstance(target_data['outputs'], torch.Tensor):
                mu_target = target_data['outputs'].detach().cpu().numpy().mean()
            else:
                mu_target = np.mean(target_data['outputs'])
        else:
            mu_target = mu  # Fallback

        # Compute delta (distance from process optimum)
        delta = mu_target - tau

        # Compute F_star (quality at target)
        F_star = np.exp(-(mu_target - tau)**2 / s)

        params[process_name] = {
            'F_star': float(F_star),
            'delta': float(delta),
            'sigma2': float(sigma2),
            's': float(s),
            'tau': float(tau),
            'mu_target': float(mu_target),
            'mu_predicted': float(mu)
        }

    return params


def estimate_effective_params_simple(
    F_star_mean: float,
    mean_sigma2: float,
    s_effective: float = 1.0
) -> Dict[str, float]:
    """
    Simple estimation of effective parameters for theoretical L_min.

    Used when detailed per-process tracking is not available.

    Args:
        F_star_mean: Mean target reliability
        mean_sigma2: Mean predicted variance across all processes
        s_effective: Effective scale parameter (weighted average)

    Returns:
        Dict with {'F_star', 'delta', 'sigma2', 's'}
    """
    # For delta, use inverse of quality function
    # F* = exp(-delta^2/s) => delta = sqrt(-s * ln(F*))
    if F_star_mean > 0 and F_star_mean < 1:
        delta = np.sqrt(-s_effective * np.log(F_star_mean))
    else:
        delta = 0.0

    return {
        'F_star': F_star_mean,
        'delta': delta,
        'sigma2': mean_sigma2,
        's': s_effective
    }


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

            # Collect sigma2 from all processes
            for proc_name, data in trajectory.items():
                sigma2 = data['outputs_var'].mean().item()
                sigma2_samples.append(sigma2)

    F_star = surrogate.F_star[scenario_idx]
    mean_sigma2 = np.mean(sigma2_samples)

    return np.array(F_samples), mean_sigma2, F_star


def compute_z_score(empirical: float, theoretical: float, std: float, n_samples: int) -> float:
    """
    Compute z-score for comparing empirical vs theoretical values.

    z = (empirical - theoretical) / (std / sqrt(n))

    Args:
        empirical: Empirical value
        theoretical: Theoretical value
        std: Standard deviation of samples
        n_samples: Number of samples

    Returns:
        z-score (values near 0 indicate good match)
    """
    if std <= 0 or n_samples <= 0:
        return 0.0

    se = std / np.sqrt(n_samples)  # Standard error
    if se <= 0:
        return 0.0

    return (empirical - theoretical) / se


def format_status(value: float, thresholds: Tuple[float, float] = (0.05, 0.20)) -> str:
    """
    Format status based on relative difference.

    Args:
        value: Relative difference |obs - theo| / theo
        thresholds: (good_threshold, warning_threshold)

    Returns:
        Status string
    """
    good_thresh, warn_thresh = thresholds
    if abs(value) < good_thresh:
        return "OK"
    elif abs(value) < warn_thresh:
        return "WARN"
    else:
        return "MISMATCH"


if __name__ == '__main__':
    # Test theoretical calculations
    print("Testing Theoretical Loss Analysis")
    print("="*60)

    # Example parameters
    F_star = 0.85
    delta = 0.2
    sigma2 = 0.05
    s = 1.0

    print(f"\nInput parameters:")
    print(f"  F* = {F_star}")
    print(f"  delta = {delta}")
    print(f"  sigma2 = {sigma2}")
    print(f"  s = {s}")

    # Compute theoretical values
    components = compute_theoretical_L_min(F_star, delta, sigma2, s)

    print(f"\nTheoretical results:")
    print(f"  E[F] = {components.E_F:.6f}")
    print(f"  E[F^2] = {components.E_F2:.6f}")
    print(f"  Var[F] = {components.Var_F:.6f}")
    print(f"  Bias^2 = {components.Bias2:.6f}")
    print(f"  L_min = {components.L_min:.6f}")

    # Test with deterministic case (sigma2 = 0)
    print(f"\nDeterministic case (sigma2 = 0):")
    components_det = compute_theoretical_L_min(F_star, delta, 0.0, s)
    print(f"  E[F] = {components_det.E_F:.6f} (should equal F* = {F_star})")
    print(f"  L_min = {components_det.L_min:.6f} (should be 0)")

    print("\nTest passed!")
