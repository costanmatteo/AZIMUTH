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

            # Squeeze to get scalar per sample
            outputs[process_name] = output.squeeze(-1) if output.dim() > 1 else output

        # Compute adaptive targets and quality scores
        adaptive_targets = {}
        quality_scores = {}

        for process_name in self.process_order:
            if process_name not in outputs:
                continue

            config = self.process_configs.get(process_name, {})
            output = outputs[process_name]

            # Compute adaptive target
            target = self._compute_adaptive_target(process_name, outputs, config)
            adaptive_targets[process_name] = target

            # Compute quality score: Q = exp(-(output - target)² / scale)
            scale = config.get('scale', 1.0)

            if isinstance(target, (int, float)):
                quality = torch.exp(-((output - target) ** 2) / scale)
            else:
                quality = torch.exp(-((output - target) ** 2) / scale)

            quality_scores[process_name] = quality

        # Compute weighted average reliability
        F = self._compute_weighted_average(quality_scores)

        if return_quality_scores:
            return F, quality_scores
        return F

    def _compute_adaptive_target(self,
                                 process_name: str,
                                 outputs: Dict[str, torch.Tensor],
                                 config: Dict) -> Union[float, torch.Tensor]:
        """
        Compute adaptive target for a process based on upstream outputs.
        """
        base_target = config.get('base_target', 0.0)

        adaptive_coeffs = config.get('adaptive_coefficients', {})
        adaptive_baselines = config.get('adaptive_baselines', {})

        if not adaptive_coeffs:
            return base_target

        target = base_target

        for upstream_name, coeff in adaptive_coeffs.items():
            if upstream_name in outputs:
                baseline = adaptive_baselines.get(upstream_name, 0.0)
                adjustment = coeff * (outputs[upstream_name] - baseline)
                target = target + adjustment

        return target

    def _compute_weighted_average(self, quality_scores: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted average of quality scores.
        """
        total_weighted_quality = 0.0
        total_weight = 0.0

        for process_name, quality in quality_scores.items():
            weight = self.process_configs.get(process_name, {}).get('weight', 1.0)
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
