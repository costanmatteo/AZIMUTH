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
    Supporta multi-scenario training con F_star calcolato per ogni scenario.
    """

    def __init__(self, target_trajectory, device='cpu'):
        """
        Args:
            target_trajectory (dict): Target trajectory da target_generation
                                     Ora contiene n_samples scenarios
            device (str): Device for computations
        """
        self.device = device
        self.n_scenarios = None  # Will be inferred from data

        # Convert target trajectory to tensors (all scenarios)
        self.target_trajectory_tensors = {}
        for process_name, data in target_trajectory.items():
            self.target_trajectory_tensors[process_name] = {
                'inputs': torch.tensor(data['inputs'], dtype=torch.float32, device=device),
                'outputs': torch.tensor(data['outputs'], dtype=torch.float32, device=device)
            }

            # Infer number of scenarios
            if self.n_scenarios is None:
                self.n_scenarios = data['inputs'].shape[0]

        # Compute F_star for each scenario
        self.F_star = self.compute_all_target_reliabilities()

    def compute_reliability(self, trajectory):
        """
        Calcola reliability F per una trajectory.

        Fa sampling dalle distribuzioni degli outputs e calcola una metrica
        fisica combinata che rappresenta la qualità del processo.

        Args:
            trajectory (dict): {
                'laser': {
                    'inputs': tensor (batch, input_dim),
                    'outputs_mean': tensor (batch, output_dim),
                    'outputs_var': tensor (batch, output_dim)
                },
                'plasma': {...},
                'galvanic': {...}
            }

        Returns:
            torch.Tensor: Reliability score F (scalar, differentiable)
        """
        # Sample outputs da distribuzioni N(mean, var)
        sampled_outputs = {}

        for process_name, data in trajectory.items():
            mean = data['outputs_mean']
            var = data['outputs_var']

            # Sample: x ~ N(mean, var)
            # Usa reparameterization trick per differenziabilità
            std = torch.sqrt(var + 1e-8)
            epsilon = torch.randn_like(mean)
            sample = mean + epsilon * std

            sampled_outputs[process_name] = sample

        # Formula fisica inventata per reliability CON RELAZIONI TRA PROCESSI
        # F combina gli outputs di tutti i processi in una metrica di qualità
        # con target adattivi basati sui processi precedenti

        # Estrai i valori (assumo 1 output per processo come da config)
        laser_power = sampled_outputs['laser'].squeeze()      # ActualPower
        plasma_rate = sampled_outputs['plasma'].squeeze()     # RemovalRate
        galvanic_thick = sampled_outputs['galvanic'].squeeze() # Thickness
        microetch_depth = sampled_outputs['microetch'].squeeze() # Depth

        # TARGET ADATTIVI: i processi successivi dipendono dai precedenti
        # Valori base:
        # - Laser ActualPower: ~0.4-0.6 → target base 0.5
        # - Plasma RemovalRate: ~3-7 → target base 5.0
        # - Galvanic Thickness: ~8-12 μm → target base 10.0
        # - Microetch Depth: ~15-25 → target base 20.0

        # Laser ha target fisso (è il primo processo)
        laser_target = 0.5

        # Plasma target dipende da Laser
        # Se laser è troppo forte → plasma deve compensare aumentando removal rate
        plasma_target = 5.0 + 3.0 * (laser_power - 0.5)

        # Galvanic target dipende da Plasma
        # Se plasma ha rimosso troppo → galvanic deve depositare più spessore
        galvanic_target = 10.0 + 2.0 * (plasma_rate - 5.0)

        # Microetch target dipende da Laser
        # Se laser è troppo forte → microetch deve essere più profondo per compensare
        microetch_target = 20.0 + 10.0 * (laser_power - 0.5)

        # Ogni componente: quanto è vicino al valore ottimale ADATTIVO
        laser_quality = torch.exp(-((laser_power - laser_target) ** 2) / 0.1)
        plasma_quality = torch.exp(-((plasma_rate - plasma_target) ** 2) / 2.0)
        galvanic_quality = torch.exp(-((galvanic_thick - galvanic_target) ** 2) / 4.0)
        microetch_quality = torch.exp(-((microetch_depth - microetch_target) ** 2) / 4.0)

        # Combinazione pesata (galvanic più importante = prodotto finale)
        F = 0.2 * laser_quality + 0.15 * plasma_quality + 0.5 * galvanic_quality + 0.15 * microetch_quality

        return F

    def compute_all_target_reliabilities(self):
        """
        Calcola F* (reliability target, fisso) per tutti gli n_scenarios.

        Returns:
            np.array: F_star values, shape (n_scenarios,)
        """
        F_star_values = []

        with torch.no_grad():
            for scenario_idx in range(self.n_scenarios):
                # Create trajectory for this specific scenario
                scenario_traj = {}
                for process_name, data in self.target_trajectory_tensors.items():
                    scenario_traj[process_name] = {
                        'inputs': data['inputs'][scenario_idx:scenario_idx+1],  # Keep batch dim
                        'outputs_mean': data['outputs'][scenario_idx:scenario_idx+1],
                        'outputs_var': torch.zeros_like(data['outputs'][scenario_idx:scenario_idx+1])
                    }

                F_star = self.compute_reliability(scenario_traj)
                F_star_values.append(F_star.item())

        return np.array(F_star_values)

    def compute_target_reliability(self):
        """
        Backward compatibility: returns mean of all F_star values.

        Returns:
            float: Mean target reliability across all scenarios
        """
        return float(np.mean(self.F_star))


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
