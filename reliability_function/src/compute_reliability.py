"""
Mathematical computation of process chain reliability F.

Formula:
    For each process i:
        τ_i = base_target_i + β × (Y_{i-1} - τ_{i-1})   (τ_1 = base_target_1)
        Q_i = exp(-(Y_i - τ_i)² / scale_i)

    Overall reliability:
        F = (Q_1 + Q_2 + ... + Q_n) / n
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, Union

from ..configs.process_targets import PROCESS_CONFIGS, PROCESS_ORDER


class ReliabilityFunction:
    """
    Computes reliability F from process chain trajectories.

    Supports both torch tensors (for differentiable training) and numpy arrays.
    """

    def __init__(self,
                 process_configs: Dict = None,
                 process_order: list = None,
                 beta: float = 0.0,
                 device: str = 'cpu'):
        """
        Args:
            process_configs: Process-specific targets (base_target, scale).
                            If None, uses default PROCESS_CONFIGS.
            process_order: Order of processes for sequential target computation.
                          If None, uses default PROCESS_ORDER.
            beta: Adaptive target coefficient.
                  τ_i = base_target_i + β × (Y_{i-1} - τ_{i-1})
            device: Torch device for computations.
        """
        self.process_configs = process_configs or PROCESS_CONFIGS
        self.process_order = process_order or PROCESS_ORDER
        self.beta = beta
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
                    'outputs_mean': tensor (batch, output_dim),
                    'outputs_sampled': tensor (batch, output_dim)  # optional
                }
            return_quality_scores: If True, also return per-process Q_i scores.
            use_sampled_outputs: If True, use 'outputs_sampled' if available.

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

            if isinstance(output, np.ndarray):
                output = torch.tensor(output, dtype=torch.float32, device=self.device)

            outputs[process_name] = output.squeeze(-1) if output.dim() > 1 else output

        # Compute Q_i for each process with adaptive targets
        quality_scores = {}
        prev_output = None
        prev_target = None

        for process_name in self.process_order:
            if process_name not in outputs:
                continue

            config = self.process_configs.get(process_name, {})
            base_target = config.get('base_target', 0.0)
            scale = config.get('scale', 1.0)
            output = outputs[process_name]

            # τ_i = base_target_i + β × (Y_{i-1} - τ_{i-1})
            if prev_output is not None and prev_target is not None and self.beta != 0.0:
                target = base_target + self.beta * (prev_output - prev_target)
            else:
                target = base_target

            quality_scores[process_name] = torch.exp(
                -((output - target) ** 2) / max(scale, 1e-8)
            )

            prev_output = output
            prev_target = target

        # F = (Q_1 + Q_2 + ... + Q_n) / n
        if quality_scores:
            F = sum(quality_scores.values()) / len(quality_scores)
        else:
            F = torch.tensor(0.0, device=self.device)

        if return_quality_scores:
            return F, quality_scores
        return F

    def compute_target_reliability(self, target_trajectory: Dict) -> torch.Tensor:
        """
        Compute F* (target reliability) for a target trajectory.

        Args:
            target_trajectory: Target trajectory with 'inputs' and 'outputs' keys

        Returns:
            F_star: Target reliability value
        """
        trajectory = {}
        for process_name, data in target_trajectory.items():
            out = data.get('outputs', data.get('outputs_mean'))
            if isinstance(out, np.ndarray):
                out = torch.tensor(out, dtype=torch.float32, device=self.device)

            trajectory[process_name] = {
                'outputs_mean': out,
                'outputs_sampled': out,
            }

        return self.compute_reliability(trajectory)


def compute_reliability(trajectory: Dict,
                       return_quality_scores: bool = False,
                       beta: float = 0.0,
                       device: str = 'cpu') -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
    """
    Convenience function to compute reliability F.

    Args:
        trajectory: Process chain trajectory
        return_quality_scores: If True, also return per-process scores
        beta: Adaptive target coefficient
        device: Torch device

    Returns:
        F: Reliability score
        quality_scores: Per-process Q_i (if return_quality_scores=True)
    """
    rf = ReliabilityFunction(beta=beta, device=device)
    return rf.compute_reliability(trajectory, return_quality_scores=return_quality_scores)


if __name__ == '__main__':
    # Test the reliability function
    print("Testing ReliabilityFunction...")

    rf = ReliabilityFunction()

    # Create test trajectory
    trajectory = {
        'laser': {
            'outputs_mean': torch.tensor([[0.8]]),
        },
        'plasma': {
            'outputs_mean': torch.tensor([[3.0]]),
        },
        'galvanic': {
            'outputs_mean': torch.tensor([[10.0]]),
        },
        'microetch': {
            'outputs_mean': torch.tensor([[20.0]]),
        },
    }

    F, scores = rf.compute_reliability(trajectory, return_quality_scores=True)
    print(f"Reliability F: {F.item():.6f}")
    print("Quality scores:")
    for name, score in scores.items():
        print(f"  {name}: {score.item():.6f}")

    # Test gradient flow
    trajectory_grad = {
        'laser': {'outputs_mean': torch.tensor([[0.8]], requires_grad=True)},
        'plasma': {'outputs_mean': torch.tensor([[3.0]], requires_grad=True)},
    }

    F_grad = rf.compute_reliability(trajectory_grad)
    F_grad.backward()

    print(f"\nGradient test:")
    print(f"  laser grad: {trajectory_grad['laser']['outputs_mean'].grad}")
    print(f"  plasma grad: {trajectory_grad['plasma']['outputs_mean'].grad}")

    print("\n✓ ReliabilityFunction test passed!")
