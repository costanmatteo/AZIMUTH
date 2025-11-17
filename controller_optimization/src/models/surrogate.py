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

        # Formula fisica inventata per reliability
        # F combina gli outputs di tutti i processi in una metrica di qualità

        # Estrai i valori (assumo 1 output per processo come da config)
        laser_power = sampled_outputs['laser'].squeeze()      # ActualPower
        plasma_rate = sampled_outputs['plasma'].squeeze()     # RemovalRate
        galvanic_thick = sampled_outputs['galvanic'].squeeze() # Thickness
        microetch_depth = sampled_outputs['microetch'].squeeze() # Depth

        # Formula inventata: F = weighted combination di qualità processo
        # Normalizzazione basata su valori tipici:
        # - Laser ActualPower: ~0.4-0.6 → target 0.5
        # - Plasma RemovalRate: ~3-7 → target 5.0
        # - Galvanic Thickness: ~8-12 μm → target 10.0
        # - Microetch Depth: ~18-22 → target 20.0

        # ==========================================
        # 1. QUALITY METRICS INDIVIDUALI
        # ==========================================
        # Ogni componente: quanto è vicino al valore ottimale
        laser_quality = torch.exp(-((laser_power - 0.5) ** 2) / 0.1)
        plasma_quality = torch.exp(-((plasma_rate - 5.0) ** 2) / 2.0)
        galvanic_quality = torch.exp(-((galvanic_thick - 10.0) ** 2) / 4.0)
        microetch_quality = torch.exp(-((microetch_depth - 20.0) ** 2) / 4.0)

        # ==========================================
        # 2. TERMINI DI INTERAZIONE TRA PROCESSI
        # ==========================================
        # Modellano come i processi si compensano a vicenda

        # Deviazioni dai target (valori positivi = sopra target, negativi = sotto)
        laser_dev = laser_power - 0.5
        plasma_dev = plasma_rate - 5.0
        galvanic_dev = galvanic_thick - 10.0
        microetch_dev = microetch_depth - 20.0

        # Interazione 1: Laser → Plasma
        # Se laser troppo forte (+), plasma deve essere delicato (-)
        # Premiamo quando laser_dev + plasma_dev ≈ 0 (si compensano)
        laser_plasma_interaction = torch.exp(-((laser_dev + plasma_dev) ** 2) / 1.0)

        # Interazione 2: Plasma → Galvanic
        # Se plasma ha rimosso molto (+), galvanic deve depositare di più (+)
        # Premiamo quando plasma_dev ≈ galvanic_dev (correlazione positiva)
        plasma_galvanic_interaction = torch.exp(-((plasma_dev - galvanic_dev) ** 2) / 2.0)

        # Interazione 3: Galvanic → Microetch
        # Se galvanic troppo spesso (+), microetch deve rimuovere di più (+)
        # Premiamo quando galvanic_dev ≈ microetch_dev (correlazione positiva)
        galvanic_microetch_interaction = torch.exp(-((galvanic_dev - microetch_dev) ** 2) / 4.0)

        # Interazione 4: Laser → Microetch
        # Se laser troppo forte (+), crea imperfezioni → microetch deve pulire di più (+)
        # Premiamo quando laser_dev ≈ microetch_dev (correlazione positiva)
        laser_microetch_interaction = torch.exp(-((laser_dev - microetch_dev) ** 2) / 2.0)

        # ==========================================
        # 3. COMBINAZIONE FINALE
        # ==========================================
        # Pesi: 0.75 per quality individuali, 0.25 per interazioni
        # Galvanic rimane il più importante (prodotto finale)
        F = (
            # Quality individuali (75%)
            0.15 * laser_quality +
            0.10 * plasma_quality +
            0.40 * galvanic_quality +
            0.10 * microetch_quality +
            # Interazioni tra processi (25%)
            0.10 * laser_plasma_interaction +
            0.05 * plasma_galvanic_interaction +
            0.05 * galvanic_microetch_interaction +
            0.05 * laser_microetch_interaction
        )

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
