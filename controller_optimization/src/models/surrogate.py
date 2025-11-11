"""
Surrogate Model (ProT) - Placeholder.

Questo è un PLACEHOLDER che sarà sostituito dal vero transformer.
Per ora: metrica semplice basata su distanza da target.
"""

import torch
import torch.nn as nn
import numpy as np


class ProTSurrogate:
    """
    Placeholder per surrogate model ProT.

    Valuta reliability di una trajectory completa.
    """

    def __init__(self, target_trajectory, device='cpu'):
        """
        Args:
            target_trajectory (dict): Target trajectory da target_generation
            device (str): Device for computations
        """
        self.device = device
        self.F_star = None  # Calcolato una volta all'inizio

        # Convert target trajectory to tensors
        self.target_trajectory_tensors = {}
        for process_name, data in target_trajectory.items():
            self.target_trajectory_tensors[process_name] = {
                'inputs': torch.tensor(data['inputs'], dtype=torch.float32, device=device),
                'outputs': torch.tensor(data['outputs'], dtype=torch.float32, device=device)
            }

        # Compute target reliability once
        self.F_star = self.compute_target_reliability()

    def compute_reliability(self, trajectory):
        """
        Calcola reliability F per una trajectory attuale.

        Args:
            trajectory (dict): {
                'laser': {
                    'inputs': tensor (batch, input_dim),
                    'outputs_mean': tensor (batch, output_dim),
                    'outputs_var': tensor (batch, output_dim)
                },
                'plasma': {...},
                ...
            }

        Returns:
            torch.Tensor: Reliability score F (scalar, differentiable)
        """
        total_distance = 0.0
        num_components = 0

        for process_name, data in trajectory.items():
            # Get target data
            target_inputs = self.target_trajectory_tensors[process_name]['inputs']
            target_outputs = self.target_trajectory_tensors[process_name]['outputs']

            # Get actual data
            actual_inputs = data['inputs']
            actual_outputs_mean = data['outputs_mean']

            # Compute MSE distance for inputs
            input_distance = torch.mean((actual_inputs - target_inputs) ** 2)
            total_distance = total_distance + input_distance
            num_components += 1

            # Compute MSE distance for outputs
            output_distance = torch.mean((actual_outputs_mean - target_outputs) ** 2)
            total_distance = total_distance + output_distance
            num_components += 1

        # Average distance across all components
        avg_distance = total_distance / num_components

        # Convert distance to reliability score
        # F = exp(-distance), so smaller distance → higher reliability
        F = torch.exp(-avg_distance)

        return F

    def compute_target_reliability(self):
        """
        Calcola F* (reliability target, fisso).

        Returns:
            float: Target reliability F*
        """
        # For target trajectory with itself, distance = 0, so F* = exp(0) = 1.0
        # We compute it explicitly for consistency

        # Create dummy trajectory from target
        target_traj_for_eval = {}
        for process_name, data in self.target_trajectory_tensors.items():
            target_traj_for_eval[process_name] = {
                'inputs': data['inputs'],
                'outputs_mean': data['outputs'],
                'outputs_var': torch.zeros_like(data['outputs'])
            }

        with torch.no_grad():
            F_star = self.compute_reliability(target_traj_for_eval)

        return F_star.item()


if __name__ == '__main__':
    # Test ProTSurrogate
    print("Testing ProTSurrogate...")

    # Create dummy target trajectory
    target_trajectory = {
        'laser': {
            'inputs': np.array([[0.5, 25.0]]),  # PowerTarget, AmbientTemp
            'outputs': np.array([[0.45]])        # ActualPower
        },
        'plasma': {
            'inputs': np.array([[200.0, 30.0]]), # RF_Power, Duration
            'outputs': np.array([[5.0]])         # RemovalRate
        }
    }

    # Create surrogate
    surrogate = ProTSurrogate(target_trajectory)
    print(f"F* (target reliability): {surrogate.F_star:.6f}")

    # Create test trajectory (same as target - should give F ≈ F*)
    test_traj_same = {
        'laser': {
            'inputs': torch.tensor([[0.5, 25.0]]),
            'outputs_mean': torch.tensor([[0.45]]),
            'outputs_var': torch.tensor([[0.01]])
        },
        'plasma': {
            'inputs': torch.tensor([[200.0, 30.0]]),
            'outputs_mean': torch.tensor([[5.0]]),
            'outputs_var': torch.tensor([[0.02]])
        }
    }

    F_same = surrogate.compute_reliability(test_traj_same)
    print(f"F (same as target): {F_same.item():.6f}")

    # Create test trajectory (different from target - should give F < F*)
    test_traj_different = {
        'laser': {
            'inputs': torch.tensor([[0.6, 26.0]]),  # Slightly different
            'outputs_mean': torch.tensor([[0.50]]),  # Slightly different
            'outputs_var': torch.tensor([[0.01]])
        },
        'plasma': {
            'inputs': torch.tensor([[220.0, 32.0]]),  # Slightly different
            'outputs_mean': torch.tensor([[5.5]]),    # Slightly different
            'outputs_var': torch.tensor([[0.02]])
        }
    }

    F_different = surrogate.compute_reliability(test_traj_different)
    print(f"F (different from target): {F_different.item():.6f}")

    # Test differentiability
    test_traj_grad = {
        'laser': {
            'inputs': torch.tensor([[0.5, 25.0]], requires_grad=True),
            'outputs_mean': torch.tensor([[0.45]], requires_grad=True),
            'outputs_var': torch.tensor([[0.01]])
        },
        'plasma': {
            'inputs': torch.tensor([[200.0, 30.0]], requires_grad=True),
            'outputs_mean': torch.tensor([[5.0]], requires_grad=True),
            'outputs_var': torch.tensor([[0.02]])
        }
    }

    F_grad = surrogate.compute_reliability(test_traj_grad)
    F_grad.backward()

    print(f"\nGradient test:")
    print(f"  Inputs grad exists: {test_traj_grad['laser']['inputs'].grad is not None}")
    print(f"  Outputs grad exists: {test_traj_grad['laser']['outputs_mean'].grad is not None}")

    print("\n✓ ProTSurrogate test passed!")
