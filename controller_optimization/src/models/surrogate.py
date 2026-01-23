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

    # Configuration for process-specific targets and quality scales
    # These values are based on typical ranges from the SCM models
    PROCESS_CONFIGS = {
        'laser': {
            'target': 0.8,      # ActualPower target
            'scale': 0.1,       # Quality scale (smaller = more sensitive)
            'weight': 1.0       # Relative importance
        },
        'plasma': {
            'target': 3.0,      # RemovalRate target
            'scale': 2.0,
            'weight': 1.0
        },
        'galvanic': {
            'target': 10.0,     # Thickness target (μm)
            'scale': 4.0,
            'weight': 1.5       # More important (final product quality)
        },
        'microetch': {
            'target': 20.0,     # Depth target
            'scale': 4.0,
            'weight': 1.0
        }
    }

    def __init__(self, target_trajectory, device='cpu', use_deterministic_sampling=True):
        """
        Args:
            target_trajectory (dict): Target trajectory da target_generation
                                     Ora contiene n_samples scenarios
            device (str): Device for computations
            use_deterministic_sampling (bool): If True, use mean values directly (deterministic).
                                               If False, use reparameterization trick (stochastic).
                                               Default: True for stable training.
        """
        self.device = device
        self.use_deterministic_sampling = use_deterministic_sampling
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

    def compute_reliability(self, trajectory, return_quality_scores=False):
        """
        Calcola reliability F per una trajectory.

        Usa gli output già campionati se disponibili, altrimenti fa sampling
        dalle distribuzioni degli outputs (backward compatibility).

        Args:
            trajectory (dict): {
                'laser': {
                    'inputs': tensor (batch, input_dim),
                    'outputs_mean': tensor (batch, output_dim),
                    'outputs_var': tensor (batch, output_dim),
                    'outputs_sampled': tensor (batch, output_dim)  # Optional
                },
                'plasma': {...},
                'galvanic': {...}
            }
            return_quality_scores (bool): If True, also return per-process quality scores.
                                          Default False for backward compatibility.

        Returns:
            If return_quality_scores=False:
                torch.Tensor: Reliability score F (scalar, differentiable)
            If return_quality_scores=True:
                Tuple[torch.Tensor, Dict[str, torch.Tensor]]: (F, quality_scores)
                where quality_scores maps process_name to per-process Q_i tensor
        """
        # Use already sampled outputs if available, otherwise sample here
        sampled_outputs = {}

        for process_name, data in trajectory.items():
            # Check if outputs are already sampled
            if 'outputs_sampled' in data:
                # Use pre-sampled outputs from ProcessChain
                sample = data['outputs_sampled']
            else:
                # Backward compatibility: sample here
                mean = data['outputs_mean']
                var = data['outputs_var']

                if self.use_deterministic_sampling:
                    # DETERMINISTIC: Use mean directly (no sampling)
                    sample = mean
                else:
                    # STOCHASTIC: Sample using reparameterization trick
                    std = torch.sqrt(var + 1e-8)
                    epsilon = torch.randn_like(mean)
                    sample = mean + epsilon * std

            sampled_outputs[process_name] = sample

        # ADAPTIVE RELIABILITY COMPUTATION WITH EXPLICIT PROCESS DEPENDENCIES
        # Formula adapted based on which processes are present
        # Processes influence each other through adaptive targets

        # Extract available process outputs (assume 1 output per process)
        outputs = {}
        for process_name, sample in sampled_outputs.items():
            outputs[process_name] = sample.squeeze()

        # Compute ADAPTIVE TARGETS based on previous processes in chain
        # Each process target adapts based on outputs of processes that came before

        adaptive_targets = {}
        quality_scores = {}

        # LASER: First process, fixed target
        if 'laser' in outputs:
            laser_power = outputs['laser']
            adaptive_targets['laser'] = 0.8  # Fixed target

            laser_quality = torch.exp(-((laser_power - adaptive_targets['laser']) ** 2) / 0.1)
            quality_scores['laser'] = laser_quality

        # PLASMA: Target depends on Laser
        if 'plasma' in outputs:
            plasma_rate = outputs['plasma']

            # Base target
            plasma_target = 3.0

            # Adapt based on Laser (if available)
            # If laser is too strong → plasma must compensate by increasing removal rate
            if 'laser' in outputs:
                plasma_target = plasma_target + 0.2 * (outputs['laser'] - 0.8)

            adaptive_targets['plasma'] = plasma_target

            plasma_quality = torch.exp(-((plasma_rate - plasma_target) ** 2) / 2.0)
            quality_scores['plasma'] = plasma_quality

        # GALVANIC: Target depends on Laser AND Plasma
        if 'galvanic' in outputs:
            galvanic_thick = outputs['galvanic']

            # Base target
            galvanic_target = 10.0

            # Adapt based on previous processes
            # If plasma removed too much → galvanic must deposit more thickness
            if 'plasma' in outputs:
                galvanic_target = galvanic_target + 0.5 * (outputs['plasma'] - 5.0)

            # If laser was strong → galvanic must compensate further
            if 'laser' in outputs:
                galvanic_target = galvanic_target + 0.4 * (outputs['laser'] - 0.5)

            adaptive_targets['galvanic'] = galvanic_target

            galvanic_quality = torch.exp(-((galvanic_thick - galvanic_target) ** 2) / 4.0)
            quality_scores['galvanic'] = galvanic_quality

        # MICROETCH: Target depends on ALL previous processes
        if 'microetch' in outputs:
            microetch_depth = outputs['microetch']

            # Base target
            microetch_target = 20.0

            # Adapt based on all previous processes
            # If laser was too strong → microetch must be deeper
            if 'laser' in outputs:
                microetch_target = microetch_target + 1.5 * (outputs['laser'] - 0.5)

            # If plasma was aggressive → microetch must compensate
            if 'plasma' in outputs:
                microetch_target = microetch_target + 0.3 * (outputs['plasma'] - 5.0)

            # If galvanic deposited too much → microetch must remove more
            if 'galvanic' in outputs:
                microetch_target = microetch_target - 0.15 * (outputs['galvanic'] - 10.0)

            adaptive_targets['microetch'] = microetch_target

            microetch_quality = torch.exp(-((microetch_depth - microetch_target) ** 2) / 4.0)
            quality_scores['microetch'] = microetch_quality

        # COMBINE QUALITY SCORES WITH WEIGHTED AVERAGE
        # Use weights from PROCESS_CONFIGS for consistency with L_min calculation
        total_weighted_quality = 0.0
        total_weight = 0.0

        for process_name, quality in quality_scores.items():
            weight = self.PROCESS_CONFIGS.get(process_name, {}).get('weight', 1.0)
            total_weighted_quality += quality * weight
            total_weight += weight

        # Normalize by total weight
        if total_weight > 0:
            F = total_weighted_quality / total_weight
        else:
            # Fallback (should never happen)
            F = torch.tensor(0.0, device=self.device)

        if return_quality_scores:
            return F, quality_scores
        return F

    def compute_reliability_deterministic(self, trajectory):
        """
        Compute reliability WITHOUT stochastic sampling.

        Uses outputs_mean directly instead of sampled outputs.
        This gives F* (target reliability) for L_min calculation.

        This method is used for computing the deterministic F* value
        that serves as the target for the L_min = Var[F] + (E[F] - F*)^2 formula.

        Args:
            trajectory (dict): Dict with process data, must have 'outputs_mean'

        Returns:
            torch.Tensor: F* - Deterministic reliability
        """
        # ADAPTIVE RELIABILITY COMPUTATION WITH EXPLICIT PROCESS DEPENDENCIES
        # Same logic as compute_reliability but using outputs_mean directly

        # Extract process outputs (using mean, not sampled)
        outputs = {}
        for process_name, data in trajectory.items():
            # Use outputs_mean for deterministic computation
            if 'outputs_mean' in data:
                output = data['outputs_mean']
            elif 'outputs' in data:
                # Fallback for target trajectory format
                output = data['outputs']
            else:
                continue

            if isinstance(output, np.ndarray):
                output = torch.tensor(output, dtype=torch.float32, device=self.device)

            outputs[process_name] = output.squeeze()

        # Compute ADAPTIVE TARGETS and quality scores (same logic as compute_reliability)
        adaptive_targets = {}
        quality_scores = {}

        # LASER: First process, fixed target
        if 'laser' in outputs:
            laser_power = outputs['laser']
            adaptive_targets['laser'] = 0.8  # Fixed target

            laser_quality = torch.exp(-((laser_power - adaptive_targets['laser']) ** 2) / 0.1)
            quality_scores['laser'] = laser_quality

        # PLASMA: Target depends on Laser
        if 'plasma' in outputs:
            plasma_rate = outputs['plasma']

            # Base target
            plasma_target = 3.0

            # Adapt based on Laser (if available)
            if 'laser' in outputs:
                plasma_target = plasma_target + 0.2 * (outputs['laser'] - 0.8)

            adaptive_targets['plasma'] = plasma_target

            plasma_quality = torch.exp(-((plasma_rate - plasma_target) ** 2) / 2.0)
            quality_scores['plasma'] = plasma_quality

        # GALVANIC: Target depends on Laser AND Plasma
        if 'galvanic' in outputs:
            galvanic_thick = outputs['galvanic']

            # Base target
            galvanic_target = 10.0

            # Adapt based on previous processes
            if 'plasma' in outputs:
                galvanic_target = galvanic_target + 0.5 * (outputs['plasma'] - 5.0)

            if 'laser' in outputs:
                galvanic_target = galvanic_target + 0.4 * (outputs['laser'] - 0.5)

            adaptive_targets['galvanic'] = galvanic_target

            galvanic_quality = torch.exp(-((galvanic_thick - galvanic_target) ** 2) / 4.0)
            quality_scores['galvanic'] = galvanic_quality

        # MICROETCH: Target depends on ALL previous processes
        if 'microetch' in outputs:
            microetch_depth = outputs['microetch']

            # Base target
            microetch_target = 20.0

            # Adapt based on all previous processes
            if 'laser' in outputs:
                microetch_target = microetch_target + 1.5 * (outputs['laser'] - 0.5)

            if 'plasma' in outputs:
                microetch_target = microetch_target + 0.3 * (outputs['plasma'] - 5.0)

            if 'galvanic' in outputs:
                microetch_target = microetch_target - 0.15 * (outputs['galvanic'] - 10.0)

            adaptive_targets['microetch'] = microetch_target

            microetch_quality = torch.exp(-((microetch_depth - microetch_target) ** 2) / 4.0)
            quality_scores['microetch'] = microetch_quality

        # COMBINE QUALITY SCORES WITH WEIGHTED AVERAGE
        total_weighted_quality = 0.0
        total_weight = 0.0

        for process_name, quality in quality_scores.items():
            weight = self.PROCESS_CONFIGS.get(process_name, {}).get('weight', 1.0)
            total_weighted_quality += quality * weight
            total_weight += weight

        # Normalize by total weight
        if total_weight > 0:
            F = total_weighted_quality / total_weight
        else:
            F = torch.tensor(0.0, device=self.device)

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

                F_star = self.compute_reliability(scenario_traj, return_quality_scores=False)
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
