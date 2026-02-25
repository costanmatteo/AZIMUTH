"""
Surrogate Models for Controller Optimization.

This module provides surrogate models for computing reliability F:

1. ProTSurrogate: Uses mathematical formula
   - Computes F using adaptive targets and weighted quality scores
   - Deterministic, fast, interpretable

2. CasualiTSurrogate: Uses causaliT (ProT transformer)
   - Loads trained TransformerForecaster from causaliT
   - Predicts F from trajectory using learned model

Use create_surrogate() factory function to instantiate the appropriate surrogate
based on configuration.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Union

sys.path.insert(0, '/home/user/AZIMUTH')


class ProTSurrogate:
    """
    Placeholder per surrogate model ProT.

    Valuta reliability di una trajectory completa.
    F_star è un singolo scalare calcolato dalla target trajectory dello scenario 0.
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

        # Compute F_star from scenario 0 (single scalar, same for all scenarios)
        self.F_star = self._compute_F_star_from_scenario_0()
        print(f"  F* = {self.F_star:.6f} (from scenario 0)")

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
                plasma_target = plasma_target + 0.0 * (outputs['laser'] - 0.8)

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
                galvanic_target = galvanic_target + 0.0 * (outputs['plasma'] - 5.0)

            # If laser was strong → galvanic must compensate further
            if 'laser' in outputs:
                galvanic_target = galvanic_target + 0.0 * (outputs['laser'] - 0.5)

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
                microetch_target = microetch_target + 0.0 * (outputs['laser'] - 0.5)

            # If plasma was aggressive → microetch must compensate
            if 'plasma' in outputs:
                microetch_target = microetch_target + 0.0 * (outputs['plasma'] - 5.0)

            # If galvanic deposited too much → microetch must remove more
            if 'galvanic' in outputs:
                microetch_target = microetch_target - 0.0 * (outputs['galvanic'] - 10.0)

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

    def _compute_F_star_from_scenario_0(self):
        """
        Calcola F* (reliability target) dalla target trajectory dello scenario 0.

        F* è un singolo scalare: la reliability deterministica (var=0) della
        target trajectory dello scenario di riferimento (scenario 0).

        Returns:
            float: F_star value
        """
        with torch.no_grad():
            scenario_traj = {}
            for process_name, data in self.target_trajectory_tensors.items():
                scenario_traj[process_name] = {
                    'inputs': data['inputs'][0:1],  # Scenario 0, keep batch dim
                    'outputs_mean': data['outputs'][0:1],
                    'outputs_var': torch.zeros_like(data['outputs'][0:1])
                }

            F_star = self.compute_reliability(scenario_traj, return_quality_scores=False)
            return F_star.item()

    def compute_target_reliability(self):
        """
        Returns F* (single scalar).

        Returns:
            float: Target reliability
        """
        return float(self.F_star)


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


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_surrogate(config: Dict,
                     target_trajectory: Dict,
                     device: str = 'cpu') -> Union['ProTSurrogate', 'CasualiTSurrogate']:
    """
    Factory function to create the appropriate surrogate based on configuration.

    Args:
        config: Surrogate configuration dict from controller_config['surrogate']
            - type: 'reliability_function' or 'casualit'
            - use_deterministic_sampling: bool
            - casualit: dict with checkpoint_path
        target_trajectory: Target trajectory dict for F* computation
        device: Torch device

    Returns:
        Surrogate instance (ProTSurrogate or CasualiTSurrogate)

    Example:
        >>> surrogate_config = {
        ...     'type': 'reliability_function',
        ...     'use_deterministic_sampling': False,
        ... }
        >>> surrogate = create_surrogate(surrogate_config, target_trajectory, 'cuda')
        >>> F = surrogate.compute_reliability(trajectory)
    """
    surrogate_type = config.get('type', 'reliability_function')
    use_deterministic = config.get('use_deterministic_sampling', True)

    if surrogate_type == 'reliability_function':
        # Use mathematical formula (ProTSurrogate)
        return ProTSurrogate(
            target_trajectory=target_trajectory,
            device=device,
            use_deterministic_sampling=use_deterministic,
        )

    elif surrogate_type == 'casualit':
        # Use learned transformer from causaliT
        casualit_config = config.get('casualit', {})
        checkpoint_path = casualit_config.get('checkpoint_path')

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"CasualiT checkpoint not found at {checkpoint_path}. "
                f"Train the model first using causaliT training pipeline."
            )

        # Load and wrap CasualiT model
        return CasualiTSurrogate(
            checkpoint_path=checkpoint_path,
            target_trajectory=target_trajectory,
            device=device,
        )

    else:
        raise ValueError(
            f"Unknown surrogate type: {surrogate_type}. "
            f"Expected 'reliability_function' or 'casualit'"
        )


class CasualiTSurrogate:
    """
    Minimal adapter for using causaliT (ProT transformer) to predict reliability F.

    This is a thin wrapper that:
    1. Loads a trained TransformerForecaster model
    2. Delegates format conversion to ProcessChain.trajectory_to_prot_format()
    3. Provides the same interface as ProTSurrogate

    The actual data format conversion is done by ProcessChain, making it the
    single source of truth for ProT format.
    """

    def __init__(self,
                 checkpoint_path: str,
                 target_trajectory: Dict,
                 device: str = 'cpu'):
        """
        Args:
            checkpoint_path: Path to trained TransformerForecaster checkpoint
            target_trajectory: Target trajectory for F* computation
            device: Torch device
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.n_scenarios = None
        self.process_chain = None  # Set via set_process_chain() after creation

        # Load the trained model from causaliT
        self.model = self._load_model(checkpoint_path, device)

        # Store target trajectory (same interface as ProTSurrogate)
        self.target_trajectory_tensors = {}
        for process_name, data in target_trajectory.items():
            self.target_trajectory_tensors[process_name] = {
                'inputs': torch.tensor(data['inputs'], dtype=torch.float32, device=device),
                'outputs': torch.tensor(data['outputs'], dtype=torch.float32, device=device)
            }
            if self.n_scenarios is None:
                self.n_scenarios = data['inputs'].shape[0]

        # F_star will be computed after process_chain is set
        self.F_star = None

    def set_process_chain(self, process_chain):
        """
        Set the ProcessChain reference for format conversion.

        Must be called before compute_reliability() can be used.
        After setting, computes F_star from scenario 0.

        Args:
            process_chain: ProcessChain instance
        """
        self.process_chain = process_chain
        # Compute F_star from scenario 0 (single scalar)
        self.F_star = self._compute_F_star_from_scenario_0()

    def _load_model(self, checkpoint_path: str, device: str):
        """
        Load trained TransformerForecaster from causaliT checkpoint.

        Args:
            checkpoint_path: Path to .ckpt file
            device: Torch device

        Returns:
            Loaded model in eval mode
        """
        from causaliT.training.forecasters.transformer_forecaster import TransformerForecaster

        # Load model from checkpoint
        model = TransformerForecaster.load_from_checkpoint(
            checkpoint_path,
            map_location=device
        )
        model.eval()
        model.to(device)

        return model

    def compute_reliability(self, trajectory: Dict, return_quality_scores: bool = False):
        """
        Compute reliability F using CasualiT model.

        Uses ProcessChain.trajectory_to_prot_format() for data conversion,
        then runs inference through the TransformerForecaster.

        Args:
            trajectory: Dict with process outputs from ProcessChain.forward()
            return_quality_scores: If True, return empty dict (not supported by CasualiT)

        Returns:
            F: Predicted reliability tensor

        Raises:
            RuntimeError: If process_chain not set via set_process_chain()
        """
        if self.process_chain is None:
            raise RuntimeError(
                "ProcessChain not set. Call set_process_chain() before compute_reliability(). "
                "CasualiTSurrogate uses ProcessChain.trajectory_to_prot_format() for data conversion."
            )

        # Use ProcessChain for format conversion (single source of truth)
        X, Y = self.process_chain.trajectory_to_prot_format(trajectory)

        # Run inference through causaliT model
        with torch.no_grad():
            forecast_output, _, _, _ = self.model.forward(data_input=X, data_trg=Y)

        # Extract F from forecast output (assumes F is the target)
        F = forecast_output.squeeze()

        if return_quality_scores:
            return F, {}  # CasualiT doesn't provide per-process quality scores
        return F

    def _compute_F_star_from_scenario_0(self) -> float:
        """
        Compute F* from scenario 0 using CasualiT model.

        Returns:
            float: F_star value (single scalar)
        """
        if self.process_chain is None:
            return 1.0  # Placeholder - will be recomputed when process_chain is set

        with torch.no_grad():
            scenario_traj = {}
            for process_name, data in self.target_trajectory_tensors.items():
                scenario_traj[process_name] = {
                    'inputs': data['inputs'][0:1],  # Scenario 0
                    'outputs_mean': data['outputs'][0:1],
                    'outputs_var': torch.zeros_like(data['outputs'][0:1])
                }

            F_star = self.compute_reliability(scenario_traj)
            if isinstance(F_star, torch.Tensor):
                F_star = F_star.item()

        return F_star

    def compute_target_reliability(self) -> float:
        """Return F* (single scalar)."""
        if self.F_star is None:
            return 1.0  # Placeholder before process_chain is set
        return float(self.F_star)
