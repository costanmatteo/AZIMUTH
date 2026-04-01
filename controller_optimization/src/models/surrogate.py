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

from scm_ds.compute_reliability import ReliabilityFunction


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

    def __init__(self, target_trajectory, device='cpu', use_deterministic_sampling=True,
                 process_configs=None, n_scenarios=None):
        """
        Args:
            target_trajectory (dict): Target trajectory da target_generation.
                                     Can be a single sample (1 row) — F* computed from it.
            device (str): Device for computations
            use_deterministic_sampling (bool): If True, use mean values directly (deterministic).
                                               If False, use reparameterization trick (stochastic).
                                               Default: True for stable training.
            process_configs (list, optional): Lista di configurazioni processo (da PROCESSES).
                Se fornita e i processi hanno 'surrogate_target'/'surrogate_scale',
                usa quelli al posto dei PROCESS_CONFIGS hardcoded.
            n_scenarios (int, optional): Override number of scenarios. If None, inferred
                from target_trajectory shape. Use this when target has 1 sample but
                training has n_train scenarios (from baseline trajectories).
        """
        self.device = device
        self.use_deterministic_sampling = use_deterministic_sampling
        self.n_scenarios = n_scenarios  # May be overridden below if None

        # Se process_configs contiene target calibrati, costruisci i config per processo
        self._dynamic_configs = None
        self._process_order = None
        if process_configs is not None:
            dynamic = {}
            order = []
            for pc in process_configs:
                if 'surrogate_target' in pc:
                    name = pc['name']
                    order.append(name)
                    entry = {
                        'base_target': pc['surrogate_target'],
                        'scale': pc['surrogate_scale'],
                        'weight': pc.get('surrogate_weight', 1.0),
                    }
                    # Target adattivi inter-processo
                    if 'surrogate_adaptive_coefficients' in pc:
                        entry['adaptive_coefficients'] = pc['surrogate_adaptive_coefficients']
                        entry['adaptive_baselines'] = pc['surrogate_adaptive_baselines']
                    dynamic[name] = entry
            if dynamic:
                self._dynamic_configs = dynamic
                self._process_order = order
                self._reliability_fn = ReliabilityFunction(
                    process_configs=dynamic,
                    process_order=order,
                    device=device,
                )

        # Convert target trajectory to tensors (all scenarios)
        self.target_trajectory_tensors = {}
        for process_name, data in target_trajectory.items():
            self.target_trajectory_tensors[process_name] = {
                'inputs': torch.tensor(data['inputs'], dtype=torch.float32, device=device),
                'outputs': torch.tensor(data['outputs'], dtype=torch.float32, device=device)
            }

            # Infer number of scenarios from target if not overridden
            if self.n_scenarios is None:
                self.n_scenarios = data['inputs'].shape[0]

        # Compute F_star from scenario 0 (single scalar, same for all scenarios)
        self.F_star = self._compute_F_star_from_scenario_0()
        print(f"  F* = {self.F_star:.6f} (from target trajectory, scenario 0)")

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

        # Extract available process outputs (assume 1 output per process)
        outputs = {}
        for process_name, sample in sampled_outputs.items():
            outputs[process_name] = sample.squeeze()

        quality_scores = {}

        if self._dynamic_configs is not None:
            # DYNAMIC PATH: delegate to ReliabilityFunction from scm_ds
            rf_trajectory = {}
            for process_name, sample in sampled_outputs.items():
                rf_trajectory[process_name] = {'outputs_sampled': sample}
            return self._reliability_fn.compute_reliability(
                rf_trajectory, return_quality_scores=return_quality_scores)
        else:
            # LEGACY PATH: logica hardcoded per processi fisici
            adaptive_targets = {}

            # LASER: First process, fixed target
            if 'laser' in outputs:
                laser_power = outputs['laser']
                adaptive_targets['laser'] = 0.8

                laser_quality = torch.exp(-((laser_power - adaptive_targets['laser']) ** 2) / 0.1)
                quality_scores['laser'] = laser_quality

            # PLASMA: Target depends on Laser
            if 'plasma' in outputs:
                plasma_rate = outputs['plasma']
                plasma_target = 3.0
                if 'laser' in outputs:
                    plasma_target = plasma_target + 0.2 * (outputs['laser'] - 0.8)
                adaptive_targets['plasma'] = plasma_target
                plasma_quality = torch.exp(-((plasma_rate - plasma_target) ** 2) / 2.0)
                quality_scores['plasma'] = plasma_quality

            # GALVANIC: Target depends on Laser AND Plasma
            if 'galvanic' in outputs:
                galvanic_thick = outputs['galvanic']
                galvanic_target = 10.0
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
                microetch_target = 20.0
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
            # Fallback (should never happen)
            F = torch.tensor(0.0, device=self.device)

        if return_quality_scores:
            return F, quality_scores
        return F

    def _compute_F_star_from_scenario_0(self):
        """
        Calcola F* (reliability target) dalla target trajectory dello scenario 0.

        F* è un singolo scalare: la reliability deterministica (var=0) della
        target trajectory dello scenario 0 (calibration row, sempre presente).

        Returns:
            float: F_star value
        """
        with torch.no_grad():
            scenario_traj = {}
            for process_name, data in self.target_trajectory_tensors.items():
                scenario_traj[process_name] = {
                    'inputs': data['inputs'][0:1],  # Scenario 0 (always present)
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
                     device: str = 'cpu',
                     process_configs: list = None,
                     n_scenarios: int = None) -> Union['ProTSurrogate', 'CasualiTSurrogate']:
    """
    Factory function to create the appropriate surrogate based on configuration.

    Args:
        config: Surrogate configuration dict from controller_config['surrogate']
            - type: 'reliability_function' or 'casualit'
            - use_deterministic_sampling: bool
            - casualit: dict with checkpoint_path
        target_trajectory: Target trajectory dict for F* computation
        device: Torch device
        process_configs: Lista configurazioni processo (da PROCESSES).
            Se i processi hanno 'surrogate_target'/'surrogate_scale',
            ProTSurrogate li usa al posto dei target hardcoded.
        n_scenarios: Override number of scenarios. If None, inferred from
            target_trajectory shape.

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
            process_configs=process_configs,
            n_scenarios=n_scenarios,
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

        # Load and wrap CasualiT model.
        # process_configs and use_deterministic_sampling are forwarded so
        # the internal ProTSurrogate (for F* and F_baseline) is configured
        # identically to the reliability_function path.
        return CasualiTSurrogate(
            checkpoint_path=checkpoint_path,
            target_trajectory=target_trajectory,
            device=device,
            use_deterministic_sampling=use_deterministic,
            process_configs=process_configs,
            n_scenarios=n_scenarios,
        )

    else:
        raise ValueError(
            f"Unknown surrogate type: {surrogate_type}. "
            f"Expected 'reliability_function' or 'casualit'"
        )


class CasualiTSurrogate:
    """
    Adapter for using a CausalIT model to predict reliability F.

    Supports multiple model architectures:
    - proT (TransformerForecaster): sequence-to-scalar prediction
    - StageCausaliT (StageCausalForecaster): dual decoder S->X, X->Y
    - SingleCausalLayer (SingleCausalForecaster): single decoder S->X

    The model_type is read from the checkpoint metadata, so the controller
    does not need to know which architecture was trained.
    """

    def __init__(self,
                 checkpoint_path: str,
                 target_trajectory: Dict,
                 device: str = 'cpu',
                 use_deterministic_sampling: bool = True,
                 process_configs: list = None,
                 n_scenarios: int = None):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            target_trajectory: Target trajectory for F* computation
            device: Torch device
            use_deterministic_sampling: Passed to internal ProTSurrogate
            process_configs: Process configs, passed to internal ProTSurrogate
            n_scenarios: Override number of scenarios
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.n_scenarios = n_scenarios
        self.process_chain = None  # Set via set_process_chain() after creation

        # Load the trained model from causaliT
        self.model, self.model_type = self._load_model(checkpoint_path, device)

        # Store target trajectory (same interface as ProTSurrogate)
        self.target_trajectory_tensors = {}
        for process_name, data in target_trajectory.items():
            self.target_trajectory_tensors[process_name] = {
                'inputs': torch.tensor(data['inputs'], dtype=torch.float32, device=device),
                'outputs': torch.tensor(data['outputs'], dtype=torch.float32, device=device)
            }
            if self.n_scenarios is None:
                self.n_scenarios = data['inputs'].shape[0]

        # Internal ProTSurrogate for F_star and F_baseline.
        # Only F_actual uses the CausaliT model; F_star (noise=0) and
        # F_baseline are always computed with the mathematical formula,
        # exactly as when type='reliability_function'.
        self._formula_surrogate = ProTSurrogate(
            target_trajectory=target_trajectory,
            device=device,
            use_deterministic_sampling=use_deterministic_sampling,
            process_configs=process_configs,
            n_scenarios=self.n_scenarios,
        )
        self.F_star = self._formula_surrogate.F_star

    def set_process_chain(self, process_chain):
        """
        Set the ProcessChain reference for format conversion.

        Must be called before compute_reliability() can be used.
        F_star is NOT recomputed here — it always comes from the internal
        ProTSurrogate (mathematical formula, noise=0).

        Args:
            process_chain: ProcessChain instance
        """
        self.process_chain = process_chain

    def _load_model(self, checkpoint_path: str, device: str):
        """
        Load a trained CausalIT model from checkpoint.

        Loads PyTorch Lightning checkpoints produced by the causaliT training
        pipeline (trainer.save_checkpoint).

        Args:
            checkpoint_path: Path to .ckpt file
            device: Torch device

        Returns:
            Tuple of (model, model_type)
        """
        checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_type = checkpoint_data.get('model_type', 'proT')

        is_pl_checkpoint = 'pytorch-lightning_version' in checkpoint_data

        # CausaliT Lightning forecaster classes
        if model_type == 'proT':
            from causaliT.training.forecasters.transformer_forecaster import TransformerForecaster
            forecaster_cls = TransformerForecaster
        elif model_type == 'StageCausaliT':
            from causaliT.training.forecasters.stage_causal_forecaster import StageCausalForecaster
            forecaster_cls = StageCausalForecaster
        elif model_type == 'SingleCausalLayer':
            from causaliT.training.forecasters.single_causal_forecaster import SingleCausalForecaster
            forecaster_cls = SingleCausalForecaster
        else:
            raise ValueError(
                f"Unknown model_type '{model_type}' in checkpoint. "
                f"Expected 'proT', 'StageCausaliT', or 'SingleCausalLayer'.")

        if is_pl_checkpoint:
            model = forecaster_cls.load_from_checkpoint(
                checkpoint_path, map_location=device, weights_only=False)
        else:
            # Manual load from non-PL checkpoint with CausaliT config
            hparams = checkpoint_data.get('hyper_parameters', checkpoint_data.get('hparams', {}))
            config = hparams if hparams else checkpoint_data.get('config', {})
            if not config:
                raise ValueError(
                    f"Cannot find hyperparameters/config in checkpoint {checkpoint_path}. "
                    f"Available keys: {list(checkpoint_data.keys())}")
            state_dict = checkpoint_data.get('state_dict', checkpoint_data)
            model = forecaster_cls(config)
            model.load_state_dict(state_dict, strict=False)

        model.eval()
        # NOTE: Do NOT use requires_grad_(False) here. Freezing parameters
        # blocks gradient flow through the model (including w.r.t. inputs).
        # The surrogate parameters are not in the controller's optimizer,
        # so they won't be updated. But we need the computation graph intact
        # for gradients to flow from F back to the controller.
        model.to(device)
        print(f"  CasualiTSurrogate loaded model_type='{model_type}' from {checkpoint_path}")

        return model, model_type

    def compute_reliability(self, trajectory: Dict, return_quality_scores: bool = False):
        """
        Compute reliability F using the loaded CausalIT model.

        The input conversion adapts to the model type:
        - proT: uses ProcessChain.trajectory_to_prot_format()
        - StageCausaliT / SingleCausalLayer: extracts s (inputs) and x (outputs) separately

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
                "CasualiTSurrogate uses ProcessChain for data conversion."
            )

        if self.model_type == 'proT':
            F = self._inference_prot(trajectory)
        elif self.model_type in ('StageCausaliT', 'SingleCausalLayer'):
            F = self._inference_stage_causal(trajectory)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        if return_quality_scores:
            return F, {}  # CasualiT doesn't provide per-process quality scores
        return F

    def _inference_prot(self, trajectory: Dict) -> torch.Tensor:
        """Run inference for ProT (TransformerForecaster)."""
        X, Y = self.process_chain.trajectory_to_prot_format(trajectory)
        forecast_output, _, _, _ = self.model.forward(data_input=X, data_trg=Y)
        return forecast_output.squeeze()

    def _inference_stage_causal(self, trajectory: Dict) -> torch.Tensor:
        """
        Run inference for StageCausaliT or SingleCausalLayer.

        Extracts separate S (inputs) and X (outputs) tensors from the trajectory.
        """
        process_names = self.process_chain.process_names

        # Build S and X tensors: (batch, n_processes, features)
        s_list = []
        x_list = []
        batch_size = None

        for pname in process_names:
            data = trajectory[pname]
            inputs = data['inputs']  # (batch, input_dim)
            # Use sampled outputs if available, else mean
            outputs = data.get('outputs_sampled', data.get('outputs_mean'))  # (batch, output_dim)

            if batch_size is None:
                batch_size = inputs.shape[0]

            s_list.append(inputs)
            x_list.append(outputs)

        # Pad to same feature dim and stack
        s_max = max(s.shape[1] for s in s_list)
        x_max = max(x.shape[1] for x in x_list)

        S = torch.zeros(batch_size, len(process_names), s_max,
                        device=self.device, dtype=torch.float32)
        X = torch.zeros(batch_size, len(process_names), x_max,
                        device=self.device, dtype=torch.float32)

        for j, (s, x) in enumerate(zip(s_list, x_list)):
            S[:, j, :s.shape[1]] = s
            X[:, j, :x.shape[1]] = x

        if self.model_type == 'StageCausaliT':
            # StageCausalForecaster.forward(S, X, Y) - Y is the target placeholder
            Y_placeholder = torch.zeros(batch_size, 1, 1,
                                        device=self.device, dtype=torch.float32)
            pred_x, pred_y, _, _, _ = self.model.forward(
                data_source=S, data_intermediate=X, data_target=Y_placeholder)
            return pred_y.squeeze()
        else:
            # SingleCausalForecaster.forward(S, X) - only predicts X, no Y
            # For SingleCausalLayer used as surrogate, the predicted X is fed through
            # a simple aggregation. However, this model type doesn't directly predict F.
            # In practice, StageCausaliT should be preferred for F prediction.
            pred_x, _, _, _ = self.model.forward(
                data_source=S, data_intermediate=X)
            # SingleCausalLayer predicts X, not Y directly - return mean as proxy
            return pred_x.mean(dim=(1, 2))

    def compute_formula_reliability(self, trajectory, return_quality_scores=False):
        """
        Compute reliability using the mathematical formula (ProTSurrogate).

        Used for F_baseline. F_star and F_baseline always use the formula,
        only F_actual uses the CausaliT model.

        Args:
            trajectory: Same format as compute_reliability()
            return_quality_scores: If True, also return per-process quality scores

        Returns:
            Same as ProTSurrogate.compute_reliability()
        """
        return self._formula_surrogate.compute_reliability(
            trajectory, return_quality_scores=return_quality_scores
        )

    def compute_target_reliability(self) -> float:
        """Return F* (single scalar, always from mathematical formula)."""
        return float(self.F_star)
