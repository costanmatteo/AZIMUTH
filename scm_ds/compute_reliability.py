"""
Mathematical computation of process chain reliability F.

This module computes reliability F from process chain trajectories using
adaptive targets that account for process dependencies.

Formula:
    For each process i:
        Q_i = exp(-(output_i - τ_i)² / s_i)

    Where τ_i is an adaptive target that depends on upstream process outputs.

    Overall reliability:
        F = Σ(w_i × Q_i) / Σ(w_i)
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, Union

from .process_targets import PROCESS_CONFIGS, PROCESS_ORDER


def _apply_adaptive_mode(delta, coeff, mode, mode_params):
    """
    Compute the adaptive target adjustment for a single upstream variable.

    Args:
        delta: torch.Tensor — (upstream_out - baseline)
        coeff: float — linear coefficient
        mode: str — one of 'linear', 'polynomial', 'power', 'softplus', 'deadband', 'tanh'
        mode_params: dict — mode-specific params for this upstream variable

    Returns:
        torch.Tensor — adjustment to add to target
    """
    if mode == 'linear':
        return coeff * delta

    if mode == 'polynomial':
        coeff2 = mode_params.get('coeff2', 0.0)
        return coeff * delta + coeff2 * delta ** 2

    if mode == 'power':
        alpha = mode_params.get('alpha', 0.5)
        return coeff * torch.sign(delta) * (torch.abs(delta) + 1e-8) ** alpha

    if mode == 'softplus':
        k = mode_params.get('k', 2.0)
        return (1.0 / k) * torch.log(1.0 + torch.exp(k * coeff * delta))

    if mode == 'deadband':
        band = mode_params.get('band', 0.0)
        abs_delta = torch.abs(delta)
        return coeff * torch.clamp(abs_delta - band, min=0.0) * torch.sign(delta)

    if mode == 'tanh':
        max_shift = mode_params.get('max_shift', 1.0)
        return max_shift * torch.tanh(coeff * delta / max_shift)

    raise ValueError(f"Unknown adaptive_mode: '{mode}'. "
                     f"Expected one of: linear, polynomial, power, softplus, deadband, tanh")


class ReliabilityFunction:
    """
    Computes reliability F from process chain trajectories.

    Supports both torch tensors (for differentiable training) and numpy arrays.
    """

    def __init__(self,
                 process_configs: Dict = None,
                 process_order: list = None,
                 device: str = 'cpu'):
        """
        Args:
            process_configs: Process-specific targets and weights.
                            If None, uses default PROCESS_CONFIGS.
            process_order: Order of processes for dependency resolution.
                          If None, uses default PROCESS_ORDER.
            device: Torch device for computations.
        """
        self.process_configs = process_configs or PROCESS_CONFIGS
        self.process_order = process_order or PROCESS_ORDER
        self.device = device

    def compute_reliability(self,
                           trajectory: Dict,
                           return_quality_scores: bool = False,
                           use_sampled_outputs: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Compute reliability F for a trajectory.

        Args:
            trajectory: Dict mapping process_name to:
                {
                    'inputs': tensor (batch, input_dim),
                    'outputs_mean': tensor (batch, output_dim),
                    'outputs_var': tensor (batch, output_dim),  # optional
                    'outputs_sampled': tensor (batch, output_dim)  # optional
                }
            return_quality_scores: If True, also return per-process Q_i scores.
            use_sampled_outputs: If True, use 'outputs_sampled' if available,
                                otherwise use 'outputs_mean'.

        Returns:
            F: Reliability score (batch,) or scalar
            quality_scores: Dict of per-process Q_i (if return_quality_scores=True)
        """
        # Extract outputs from trajectory
        outputs = {}
        for process_name, data in trajectory.items():
            if use_sampled_outputs and 'outputs_sampled' in data:
                output = data['outputs_sampled']
            else:
                output = data['outputs_mean']

            # Ensure tensor
            if isinstance(output, np.ndarray):
                output = torch.tensor(output, dtype=torch.float32, device=self.device)

            # Keep multi-dimensional output as-is; ensure at least 2D for uniform handling
            if output.dim() == 1:
                output = output.unsqueeze(-1)  # (batch,) → (batch, 1)
            outputs[process_name] = output  # (batch, output_dim)

        # Compute adaptive targets and quality scores
        adaptive_targets = {}
        quality_scores = {}

        for process_name in self.process_order:
            if process_name not in outputs:
                continue

            config = self.process_configs.get(process_name, {})
            output = outputs[process_name]

            # Compute adaptive target (broadcastable to (batch, output_dim))
            target = self._compute_adaptive_target(process_name, outputs, config)
            adaptive_targets[process_name] = target

            # Get per-dimension scale; wrap scalar in list for uniform handling
            scales = config.get('scale', 1.0)
            if not isinstance(scales, list):
                scales = [scales]
            scale_t = torch.tensor(scales, dtype=torch.float32, device=self.device)  # (output_dim,)

            # Compute per-dimension quality and average across output dims
            per_dim_q = torch.exp(-((output - target) ** 2) / scale_t)  # (batch, output_dim)
            quality = per_dim_q.mean(dim=-1)  # (batch,)

            quality_scores[process_name] = quality

        # Compute weighted average reliability
        F = self._compute_weighted_average(quality_scores)

        if return_quality_scores:
            return F, quality_scores
        return F

    def _compute_adaptive_target(self,
                                 process_name: str,
                                 outputs: Dict[str, torch.Tensor],
                                 config: Dict) -> torch.Tensor:
        """
        Compute adaptive target for a process based on upstream outputs.

        Returns a tensor broadcastable to (batch, output_dim):
        - No adaptive coefficients: (output_dim,) from base_target.
        - With adaptive coefficients: (1, output_dim) or (batch, output_dim),
          where the per-dimension base_target is preserved and each upstream's
          scalar adjustment is broadcast uniformly across output dimensions.
        """
        base_target = config.get('base_target', 0.0)

        # Normalize base_target to a tensor of shape (output_dim,)
        if isinstance(base_target, list):
            base_target_t = torch.tensor(base_target, dtype=torch.float32, device=self.device)
        else:
            base_target_t = torch.tensor([base_target], dtype=torch.float32, device=self.device)

        adaptive_coeffs = config.get('adaptive_coefficients', {})
        adaptive_baselines = config.get('adaptive_baselines', {})

        if not adaptive_coeffs:
            return base_target_t  # (output_dim,) — broadcasts to (batch, output_dim)

        # Read adaptive mode and per-upstream mode params
        mode = config.get('adaptive_mode', 'linear')
        adaptive_coefficients2 = config.get('adaptive_coefficients2', {})
        adaptive_power = config.get('adaptive_power', {})
        adaptive_sharpness = config.get('adaptive_sharpness', {})
        adaptive_band = config.get('adaptive_band', {})
        adaptive_max_shift = config.get('adaptive_max_shift', {})

        # Adaptive case: preserve per-dimension base_target and broadcast each
        # upstream's scalar adjustment uniformly across output dimensions.
        # Start with shape (1, output_dim) so batch-broadcasting works cleanly.
        target = base_target_t.unsqueeze(0)

        for upstream_name, coeff in adaptive_coeffs.items():
            if upstream_name in outputs:
                upstream_out = outputs[upstream_name]
                # Average across output dims if multi-dimensional → (batch,)
                if upstream_out.dim() > 1:
                    upstream_out = upstream_out.mean(dim=-1)  # (batch,)

                baseline = adaptive_baselines.get(upstream_name, 0.0)
                delta = upstream_out - baseline  # (batch,)

                # Build mode_params for this upstream variable
                mode_params = {}
                if mode == 'polynomial':
                    mode_params['coeff2'] = adaptive_coefficients2.get(upstream_name, 0.0)
                elif mode == 'power':
                    mode_params['alpha'] = adaptive_power.get(upstream_name, 0.5)
                elif mode == 'softplus':
                    mode_params['k'] = adaptive_sharpness.get(upstream_name, 2.0)
                elif mode == 'deadband':
                    mode_params['band'] = adaptive_band.get(upstream_name, 0.0)
                elif mode == 'tanh':
                    mode_params['max_shift'] = adaptive_max_shift.get(upstream_name, 1.0)

                adjustment = _apply_adaptive_mode(delta, coeff, mode, mode_params)  # (batch,)
                # Broadcast uniformly across output dims:
                # (1, output_dim) + (batch, 1) → (batch, output_dim)
                target = target + adjustment.unsqueeze(-1)

        return target

    def _compute_weighted_average(self, quality_scores: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted average of quality scores.
        """
        total_weighted_quality = 0.0
        total_weight = 0.0

        for process_name, quality in quality_scores.items():
            weight = self.process_configs.get(process_name, {}).get('weight', 1.0)
            # Handle list weights (multi-output): average to get per-process scalar weight
            if isinstance(weight, list):
                weight = sum(weight) / len(weight)
            total_weighted_quality = total_weighted_quality + quality * weight
            total_weight += weight

        if total_weight > 0:
            F = total_weighted_quality / total_weight
        else:
            F = torch.tensor(0.0, device=self.device)

        return F

    def compute_target_reliability(self, target_trajectory: Dict) -> torch.Tensor:
        """
        Compute F* (target reliability) for a target trajectory.
        """
        trajectory = {}
        for process_name, data in target_trajectory.items():
            outputs = data.get('outputs', data.get('outputs_mean'))
            if isinstance(outputs, np.ndarray):
                outputs = torch.tensor(outputs, dtype=torch.float32, device=self.device)

            trajectory[process_name] = {
                'outputs_mean': outputs,
                'outputs_sampled': outputs,
            }

        return self.compute_reliability(trajectory)


def compute_reliability(trajectory: Dict,
                       return_quality_scores: bool = False,
                       device: str = 'cpu') -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
    """
    Convenience function to compute reliability F.
    """
    rf = ReliabilityFunction(device=device)
    return rf.compute_reliability(trajectory, return_quality_scores=return_quality_scores)
