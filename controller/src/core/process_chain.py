"""
Process Chain: orchestrazione della sequenza di processi.

Gestisce:
- Uncertainty predictors (frozen)
- Policy generators (trainable)
- Preprocessors (scaling/unscaling)
- Forward pass attraverso tutta la catena
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# Add paths
REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from controller.src.models.uncertainty_predictor.loader import (
    load_uncertainty_predictor,
    load_preprocessor
)
from controller.src.models.policy_generator.policy_generator import PolicyGenerator
from controller.src.models.policy_generator.scenario_encoder import ScenarioEncoder


class ProcessChain(nn.Module):
    """
    Catena di processi con uncertainty predictors frozen e policy generators trainable.

    Sequenza:
    a1 (fisso) → UncertPred1 → (o1, σ1²) → Policy1 → a2 → UncertPred2 → (o2, σ2²) → ...
    """

    def enable_debug(self, enable: bool = True):
        """Enable/disable debug mode for ProcessChain and all PolicyGenerators."""
        self.debug = enable
        for policy in self.policy_generators:
            policy.debug = enable

    @staticmethod
    def _get_controllable_info(process_config):
        """
        Get information about controllable and non-controllable inputs for a process.

        Args:
            process_config (dict): Process configuration

        Returns:
            dict: {
                'controllable_indices': list of indices for controllable inputs,
                'non_controllable_indices': list of indices for non-controllable inputs,
                'controllable_labels': list of controllable input names,
                'non_controllable_labels': list of non-controllable input names,
                'n_controllable': number of controllable inputs,
                'n_non_controllable': number of non-controllable inputs,
            }
        """
        from configs.processes_config import get_controllable_inputs

        input_labels = process_config['input_labels']
        controllable = get_controllable_inputs(process_config)

        controllable_indices = []
        non_controllable_indices = []
        controllable_labels = []
        non_controllable_labels = []

        for idx, label in enumerate(input_labels):
            if label in controllable:
                controllable_indices.append(idx)
                controllable_labels.append(label)
            else:
                non_controllable_indices.append(idx)
                non_controllable_labels.append(label)

        return {
            'controllable_indices': controllable_indices,
            'non_controllable_indices': non_controllable_indices,
            'controllable_labels': controllable_labels,
            'non_controllable_labels': non_controllable_labels,
            'n_controllable': len(controllable_indices),
            'n_non_controllable': len(non_controllable_indices),
        }

    @staticmethod
    def _count_structural_params(processes_config):
        """
        Conta il numero totale di parametri strutturali (non-controllabili) in tutti i processi.

        Args:
            processes_config (list): Lista di configurazioni dei processi

        Returns:
            int: Numero totale di parametri strutturali
        """
        total = 0
        for process_config in processes_config:
            info = ProcessChain._get_controllable_info(process_config)
            total += info['n_non_controllable']

        return total

    def __init__(self, processes_config, target_trajectory, policy_config=None, device='cpu',
                 baseline_trajectories=None, debug: bool = False):
        """
        Args:
            processes_config (list): Lista da PROCESSES
            target_trajectory (dict): Da generate_target_trajectory() — single target (1 sample)
            policy_config (dict): Config for policy generators
            device (str): Device
            baseline_trajectories (dict, optional): Baseline trajectories with different env params.
                If provided, non-controllable inputs (env params) are sourced from baselines
                instead of the target trajectory. Shape: (n_baselines, input_dim) per process.
            debug (bool): Enable debug mode
        """
        super(ProcessChain, self).__init__()

        self.device = device
        self.debug = debug
        self.process_names = [p['name'] for p in processes_config]

        self.processes_config = processes_config
        self.target_trajectory = target_trajectory
        self.baseline_trajectories = baseline_trajectories

        # Default policy config
        if policy_config is None:
            policy_config = {
                'architecture': 'medium',
                'hidden_sizes': [64, 32],
                'dropout': 0.1,
                'use_batchnorm': False,
                'use_scenario_encoder': True,  # Enable scenario encoding
                'scenario_embedding_dim': 16,  # Dimension of scenario embedding
            }
        self.policy_config = policy_config

        # Use scenario encoder if enabled
        self.use_scenario_encoder = policy_config.get('use_scenario_encoder', True)
        self.scenario_embedding_dim = policy_config.get('scenario_embedding_dim', 16)
        self.observation_mode = policy_config.get('observation_mode', 'mean_var')

        # Load uncertainty predictors (frozen)
        self.uncertainty_predictors = nn.ModuleList()
        self.preprocessors = []

        for process_config in processes_config:
            checkpoint_dir = Path(process_config['checkpoint_dir'])
            model_path = checkpoint_dir / 'uncertainty_predictor.pth'
            scaler_path = checkpoint_dir / 'scalers.pkl'

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Uncertainty predictor not found for process '{process_config['name']}'. "
                    f"Run train_processes.py first."
                )

            # Load model
            model = load_uncertainty_predictor(
                checkpoint_path=model_path,
                input_dim=process_config['input_dim'],
                output_dim=process_config['output_dim'],
                model_config=process_config['uncertainty_predictor']['model'],
                device=device
            )
            self.uncertainty_predictors.append(model)

            # Load preprocessor
            preprocessor = load_preprocessor(scaler_path)
            self.preprocessors.append(preprocessor)

        # Create scenario encoder (if enabled)
        if self.use_scenario_encoder:
            n_structural_params = self._count_structural_params(processes_config)
            self.scenario_encoder = ScenarioEncoder(
                n_structural_params=max(n_structural_params, 1),  # At least 1 for dummy case
                embedding_dim=self.scenario_embedding_dim,
                hidden_dim=32
            ).to(device)
            print(f"  Scenario encoder created:")
            print(f"    Structural params: {n_structural_params}")
            print(f"    Embedding dim: {self.scenario_embedding_dim}")
            print(f"    Parameters: {sum(p.numel() for p in self.scenario_encoder.parameters()):,}")
        else:
            self.scenario_encoder = None
            print(f"  Scenario encoder: disabled")

        # Create policy generators (trainable)
        # Policy i generates ONLY CONTROLLABLE inputs for process i+1 based on outputs of process i
        self.policy_generators = nn.ModuleList()
        self.controllable_info_per_process = []  # Store controllable info for each process

        # Pre-compute controllable info for all processes
        for process_config in processes_config:
            info = self._get_controllable_info(process_config)
            self.controllable_info_per_process.append(info)

        for i in range(len(processes_config) - 1):
            # Input to policy: [prev_outputs_mean, prev_outputs_var, non_controllable_inputs, scenario_embedding]
            prev_output_dim = processes_config[i]['output_dim']
            next_process_info = self.controllable_info_per_process[i + 1]
            n_non_controllable = next_process_info['n_non_controllable']

            # policy_input_size depends on observation_mode
            if self.observation_mode == 'mean_var':
                policy_input_size = prev_output_dim + prev_output_dim + n_non_controllable
            elif self.observation_mode == 'sample':
                policy_input_size = prev_output_dim + n_non_controllable
            elif self.observation_mode == 'residual':
                # residual mode concatenates [o_sampled, ε] → 2 × prev_output_dim
                policy_input_size = 2 * prev_output_dim + n_non_controllable
            else:
                raise ValueError(f"Unknown observation_mode: '{self.observation_mode}'")


            # Add scenario embedding dimension if encoder is enabled
            if self.use_scenario_encoder:
                policy_input_size += self.scenario_embedding_dim

            # Output from policy: ONLY controllable inputs for next process
            n_controllable = next_process_info['n_controllable']

            if n_controllable == 0:
                raise ValueError(
                    f"Process '{processes_config[i + 1]['name']}' has no controllable inputs. "
                    f"Policy generator cannot be created for this process."
                )

            # Get output bounds ONLY for controllable inputs from next process's preprocessor
            next_preprocessor = self.preprocessors[i + 1]
            controllable_indices = next_process_info['controllable_indices']

            if next_preprocessor.input_min is not None and next_preprocessor.input_max is not None:
                # Check for explicit action_domain override (e.g. [-1, 1] for sinusoidal SCM)
                action_domain = processes_config[i + 1].get('action_domain')
                if action_domain is not None:
                    a_lo, a_hi = action_domain
                    output_min = torch.full((len(controllable_indices),), a_lo, dtype=torch.float32)
                    output_max = torch.full((len(controllable_indices),), a_hi, dtype=torch.float32)
                else:
                    # Extract bounds only for controllable inputs
                    output_min = torch.tensor(
                        [next_preprocessor.input_min[idx] for idx in controllable_indices],
                        dtype=torch.float32
                    )
                    output_max = torch.tensor(
                        [next_preprocessor.input_max[idx] for idx in controllable_indices],
                        dtype=torch.float32
                    )
                print(f"  Policy {i} -> Process '{processes_config[i + 1]['name']}' (observation_mode='{self.observation_mode}'):")
                if self.observation_mode == 'mean_var':
                    print(f"    Input dim: {policy_input_size} (prev_mean={prev_output_dim} + prev_var={prev_output_dim} + non_controllable={n_non_controllable})")
                elif self.observation_mode == 'sample':
                    print(f"    Input dim: {policy_input_size} (prev_sampled={prev_output_dim} + non_controllable={n_non_controllable})")
                elif self.observation_mode == 'residual':
                    print(f"    Input dim: {policy_input_size} (prev_sampled={prev_output_dim} + prev_residual={prev_output_dim} + non_controllable={n_non_controllable})")
                print(f"    Output dim: {n_controllable} (controllable only)")
                print(f"    Controllable inputs: {next_process_info['controllable_labels']}")
                print(f"    Non-controllable inputs (env conditions): {next_process_info['non_controllable_labels']}")
                print(f"    Output bounds - Min: {output_min.numpy()}, Max: {output_max.numpy()}")
            else:
                output_min = None
                output_max = None
                print(f"  Policy {i} -> Process '{processes_config[i + 1]['name']}': No bounds (unbounded output)")
                print(f"    Controllable inputs: {next_process_info['controllable_labels']}")

            # Create policy generator with bounds - output size is n_controllable, not full input_dim
            if policy_config['architecture'] == 'small':
                policy = PolicyGenerator(
                    input_size=policy_input_size,
                    output_size=n_controllable,
                    hidden_sizes=[32, 16],
                    dropout_rate=0.05,
                    output_min=output_min,
                    output_max=output_max
                )
            elif policy_config['architecture'] == 'medium':
                policy = PolicyGenerator(
                    input_size=policy_input_size,
                    output_size=n_controllable,
                    hidden_sizes=[64, 32],
                    dropout_rate=0.1,
                    output_min=output_min,
                    output_max=output_max
                )
            elif policy_config['architecture'] == 'large':
                policy = PolicyGenerator(
                    input_size=policy_input_size,
                    output_size=n_controllable,
                    hidden_sizes=[128, 64, 32],
                    dropout_rate=0.15,
                    output_min=output_min,
                    output_max=output_max
                )
            elif policy_config['architecture'] == 'custom':
                policy = PolicyGenerator(
                    input_size=policy_input_size,
                    output_size=n_controllable,
                    hidden_sizes=policy_config['hidden_sizes'],
                    dropout_rate=policy_config['dropout'],
                    use_batchnorm=policy_config['use_batchnorm'],
                    output_min=output_min,
                    output_max=output_max
                )
            else:
                raise ValueError(f"Unknown policy architecture: {policy_config['architecture']}")

            policy = policy.to(device)
            # Set debug name for this policy generator
            policy.debug_name = f"{processes_config[i]['name']}->{processes_config[i + 1]['name']}"
            self.policy_generators.append(policy)

    def train(self, mode=True):
        """Override train() to keep frozen UncertaintyPredictors in eval() mode.

        nn.Module.train() recursively sets ALL submodules to training mode.
        This reactivates dropout in the frozen UncertaintyPredictors (dropout_rate=0.2),
        causing their mean/variance predictions to fluctuate randomly during training.
        Since L_min is computed in eval() mode (deterministic predictions), the dropout
        noise can produce training loss values below L_min — a theoretical impossibility.

        Fix: only set PolicyGenerators (and ScenarioEncoder) to train mode;
        UncertaintyPredictors always stay in eval() mode.
        """
        # Set self.training flag without recursion
        self.training = mode

        # PolicyGenerators: respect train/eval mode (they have trainable dropout)
        for policy in self.policy_generators:
            policy.train(mode)

        # ScenarioEncoder: respect train/eval mode (if present)
        if hasattr(self, 'scenario_encoder') and self.scenario_encoder is not None:
            self.scenario_encoder.train(mode)

        # UncertaintyPredictors: ALWAYS eval (frozen, dropout must stay off)
        for predictor in self.uncertainty_predictors:
            predictor.eval()

        return self

    def get_initial_inputs(self, batch_size=1, scenario_idx=None):
        """
        Get initial inputs a1 for the first process.

        Controllable inputs come from the target trajectory (always sample 0).
        Non-controllable inputs (env params) come from baseline_trajectories[scenario_idx]
        if available, otherwise from target_trajectory[scenario_idx].

        Args:
            batch_size (int): Number of parallel samples
            scenario_idx (int, optional): Which scenario's env params to use.
                                         If None, uses scenario 0.

        Returns:
            torch.Tensor: Initial inputs for first process
        """
        if scenario_idx is None:
            scenario_idx = 0

        first_process_name = self.process_names[0]
        info = self.controllable_info_per_process[0]

        if self.baseline_trajectories is not None:
            # Target always has 1 sample — use sample 0 for controllable inputs
            target_inputs = self.target_trajectory[first_process_name]['inputs'][0].copy()  # (input_dim,)
            # Non-controllable inputs from baseline scenario
            baseline_inputs = self.baseline_trajectories[first_process_name]['inputs'][scenario_idx]

            # Override non-controllable with baseline env params
            for idx in info['non_controllable_indices']:
                target_inputs[idx] = baseline_inputs[idx]

            initial_inputs = np.tile(target_inputs, (batch_size, 1))
        else:
            # Legacy: use target trajectory directly
            target_inputs_all = self.target_trajectory[first_process_name]['inputs']
            target_idx = min(scenario_idx, target_inputs_all.shape[0] - 1)
            target_inputs = target_inputs_all[target_idx]  # (input_dim,)
            initial_inputs = np.tile(target_inputs, (batch_size, 1))

        return torch.tensor(initial_inputs, dtype=torch.float32, device=self.device)

    def scale_inputs(self, inputs, process_idx):
        """
        Scale inputs using preprocessor (DIFFERENTIABLE VERSION).

        Uses PyTorch operations to maintain gradient flow.
        For StandardScaler: scaled = (x - mean) / scale
        """
        scaler = self.preprocessors[process_idx].input_scaler

        # Get scaler parameters as tensors (on same device as inputs)
        mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=self.device)
        scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=self.device)

        # Differentiable scaling
        inputs_scaled = (inputs - mean) / scale

        return inputs_scaled

    def unscale_outputs(self, outputs, process_idx):
        """
        Unscale outputs using preprocessor (DIFFERENTIABLE VERSION).

        Uses PyTorch operations to maintain gradient flow.
        For StandardScaler: unscaled = x * scale + mean
        """
        scaler = self.preprocessors[process_idx].output_scaler

        # Get scaler parameters as tensors (on same device as outputs)
        mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=self.device)
        scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=self.device)

        # Differentiable unscaling
        outputs_unscaled = outputs * scale + mean

        return outputs_unscaled

    def unscale_variance(self, variance, process_idx):
        """Unscale variance (variance scales with scale^2)."""
        output_scale = self.preprocessors[process_idx].output_scaler.scale_
        scale_squared = torch.tensor(output_scale ** 2, dtype=torch.float32, device=self.device)
        return variance * scale_squared

    def _merge_controllable_inputs(self, controllable_outputs, process_idx, scenario_idx, batch_size):
        """
        Merges controllable inputs from policy with non-controllable inputs.

        Non-controllable values come from baseline_trajectories[scenario_idx] if
        available, otherwise from target_trajectory.

        Args:
            controllable_outputs: Tensor with ONLY controllable inputs from policy generator
                                 Shape: (batch_size, n_controllable)
            process_idx: Index of the current process
            scenario_idx: Index of the scenario (for retrieving env params)
            batch_size: Batch size

        Returns:
            Tensor with all inputs in correct order
            Shape: (batch_size, input_dim)
        """
        process_name = self.process_names[process_idx]
        process_config = self.processes_config[process_idx]
        info = self.controllable_info_per_process[process_idx]

        input_dim = process_config['input_dim']
        controllable_indices = info['controllable_indices']
        non_controllable_indices = info['non_controllable_indices']

        # If all inputs are controllable, return as-is (no merging needed)
        if info['n_non_controllable'] == 0:
            return controllable_outputs

        # Source non-controllable values from baseline (env params) or target
        if self.baseline_trajectories is not None:
            source_inputs = self.baseline_trajectories[process_name]['inputs'][scenario_idx]
        else:
            target_inputs_all = self.target_trajectory[process_name]['inputs']
            target_idx = min(scenario_idx, target_inputs_all.shape[0] - 1)
            source_inputs = target_inputs_all[target_idx]

        # Create full input tensor
        full_inputs = torch.zeros(batch_size, input_dim, dtype=torch.float32, device=self.device)

        # Place controllable outputs in their correct positions
        for out_idx, input_idx in enumerate(controllable_indices):
            full_inputs[:, input_idx] = controllable_outputs[:, out_idx]

        # Place non-controllable values from source in their positions
        for input_idx in non_controllable_indices:
            full_inputs[:, input_idx] = source_inputs[input_idx]

        return full_inputs

    def _get_non_controllable_inputs(self, process_idx, scenario_idx, batch_size):
        """
        Get non-controllable inputs (env params) for a process.

        Sources from baseline_trajectories[scenario_idx] if available,
        otherwise from target_trajectory.

        Args:
            process_idx: Index of the process
            scenario_idx: Index of the scenario (for retrieving env params)
            batch_size: Batch size

        Returns:
            Tensor with non-controllable inputs, shape: (batch_size, n_non_controllable)
            Returns None if there are no non-controllable inputs
        """
        process_name = self.process_names[process_idx]
        info = self.controllable_info_per_process[process_idx]

        n_non_controllable = info['n_non_controllable']

        if n_non_controllable == 0:
            return None

        non_controllable_indices = info['non_controllable_indices']

        # Source from baseline (env params) or target
        if self.baseline_trajectories is not None:
            source_inputs = self.baseline_trajectories[process_name]['inputs'][scenario_idx]
        else:
            target_inputs_all = self.target_trajectory[process_name]['inputs']
            target_idx = min(scenario_idx, target_inputs_all.shape[0] - 1)
            source_inputs = target_inputs_all[target_idx]

        non_controllable_values = torch.tensor(
            [source_inputs[idx] for idx in non_controllable_indices],
            dtype=torch.float32,
            device=self.device
        )

        non_controllable_batch = non_controllable_values.unsqueeze(0).expand(batch_size, -1)

        return non_controllable_batch

    def _extract_structural_params(self, scenario_idx):
        """
        Estrae parametri strutturali (non-controllabili) per uno scenario specifico.

        Sources from baseline_trajectories if available, otherwise from target.

        Args:
            scenario_idx (int): Index dello scenario

        Returns:
            torch.Tensor: Parametri strutturali, shape (n_structural_params,)
        """
        structural_values = []

        for i, process_config in enumerate(self.processes_config):
            process_name = process_config['name']
            info = self._get_controllable_info(process_config)

            # Source from baseline (env params) or target
            if self.baseline_trajectories is not None:
                source_inputs = self.baseline_trajectories[process_name]['inputs'][scenario_idx]
            else:
                target_inputs_all = self.target_trajectory[process_name]['inputs']
                target_idx = min(scenario_idx, target_inputs_all.shape[0] - 1)
                source_inputs = target_inputs_all[target_idx]

            for idx in info['non_controllable_indices']:
                structural_values.append(source_inputs[idx])

        if len(structural_values) == 0:
            return torch.tensor([0.0], dtype=torch.float32, device=self.device)

        return torch.tensor(structural_values, dtype=torch.float32, device=self.device)

    def _debug_forward_step(self, phase: int, **kwargs):
        """Print debug info for a single forward step."""
        print(f"\n[{phase}] " + " | ".join(f"{k}={v}" for k, v in kwargs.items()))

    def forward(self, batch_size=1, scenario_idx=0):
        """
        Forward pass attraverso tutta la catena per uno scenario specifico.

        Args:
            batch_size (int): Number of parallel samples
            scenario_idx (int): Which scenario's structural conditions to use (default: 0)

        Returns:
            trajectory (dict): {
                'laser': {
                    'inputs': tensor,
                    'outputs_mean': tensor (predicted mean),
                    'outputs_var': tensor (predicted variance),
                    'outputs_sampled': tensor (sampled from N(mean, var))
                },
                'plasma': {...},
                ...
            }
        """

        trajectory = {}

        # a1 è fisso dalla target trajectory (per lo scenario specifico)
        current_inputs = self.get_initial_inputs(batch_size, scenario_idx)

        if self.debug:
            self._debug_forward_step(0,
                scenario=scenario_idx,
                batch=batch_size,
                initial_inputs=current_inputs[0].tolist())

        # Extract and encode scenario structural parameters (if encoder is enabled)
        if self.use_scenario_encoder:
            structural_params = self._extract_structural_params(scenario_idx)  # Shape: (n_params,)
            # Add batch dimension and replicate for batch
            structural_params = structural_params.unsqueeze(0).repeat(batch_size, 1)  # (batch_size, n_params)
            # Encode to embedding
            scenario_embedding = self.scenario_encoder(structural_params)  # (batch_size, embedding_dim)
            if self.debug:
                self._debug_forward_step(0,
                    scenario_embedding_mean=f"{scenario_embedding.mean().item():.4f}",
                    scenario_embedding_std=f"{scenario_embedding.std().item():.4f}")
        else:
            scenario_embedding = None

        for i, process_name in enumerate(self.process_names):
            if self.debug:
                self._debug_forward_step(1,
                    step=i,
                    process=process_name)

            # 1. Se i > 0: policy generator produce inputs
            if i > 0:
                # Get non-controllable inputs for the current process (environmental conditions)
                non_controllable_inputs = self._get_non_controllable_inputs(i, scenario_idx, batch_size)

                if self.debug:
                    nc_info = self.controllable_info_per_process[i]
                    nc_shape = non_controllable_inputs.shape if non_controllable_inputs is not None else None
                    nc_labels = nc_info['non_controllable_labels'] if non_controllable_inputs is not None else None
                    self._debug_forward_step(1,
                        policy=f"{self.process_names[i-1]}->{process_name}",
                        prev_mean_shape=str(prev_outputs_mean.shape),
                        prev_mean_mean=f"{prev_outputs_mean.mean().item():.6f}",
                        prev_var_mean=f"{prev_outputs_var.mean().item():.6f}",
                        non_controllable_shape=str(nc_shape),
                        non_controllable_labels=str(nc_labels))

                # Build policy input based on observation mode
                if self.observation_mode == 'mean_var':
                    policy_input_parts = [prev_outputs_mean, prev_outputs_var]
                elif self.observation_mode == 'sample':
                    policy_input_parts = [prev_outputs_sampled]
                elif self.observation_mode == 'residual':
                    # ε = (o_sampled - mean) / sqrt(var + 1e-8)
                    # Pass BOTH the sampled value and the standardized residual so the
                    # policy has access to the absolute level AND the realized noise.
                    eps = (prev_outputs_sampled - prev_outputs_mean) / \
                          torch.sqrt(prev_outputs_var + 1e-8)
                    policy_input_parts = [prev_outputs_sampled, eps]

                # Add non-controllable inputs (environmental conditions) if present
                if non_controllable_inputs is not None:
                    policy_input_parts.append(non_controllable_inputs)

                # Add scenario embedding if encoder is enabled
                if self.use_scenario_encoder:
                    policy_input_parts.append(scenario_embedding)

                policy_input = torch.cat(policy_input_parts, dim=1)

                if self.debug:
                    self._debug_forward_step(2,
                        policy_input_shape=str(policy_input.shape),
                        mean=f"{policy_input.mean().item():.6f}",
                        std=f"{policy_input.std().item():.6f}",
                        min=f"{policy_input.min().item():.6f}",
                        max=f"{policy_input.max().item():.6f}")

                # Policy outputs ONLY controllable inputs
                controllable_outputs = self.policy_generators[i - 1](policy_input)
                process_info = self.controllable_info_per_process[i]

                if self.debug:
                    self._debug_forward_step(2,
                        controllable_labels=str(process_info['controllable_labels']),
                        sample=controllable_outputs[0].tolist())

                # Merge controllable outputs with non-controllable values from target
                current_inputs = self._merge_controllable_inputs(
                    controllable_outputs, i, scenario_idx, batch_size
                )

                if self.debug:
                    self._debug_forward_step(3,
                        merged_shape=str(current_inputs.shape),
                        sample=current_inputs[0].tolist())

            # 2. Scale inputs
            scaled_inputs = self.scale_inputs(current_inputs, i)

            if self.debug:
                self._debug_forward_step(4,
                    process=process_name,
                    scaled_mean=f"{scaled_inputs.mean().item():.6f}",
                    scaled_std=f"{scaled_inputs.std().item():.6f}")

            # 3. Uncertainty predictor (frozen)
            outputs_mean_scaled, outputs_var_scaled = self.uncertainty_predictors[i](scaled_inputs)

            if self.debug:
                self._debug_forward_step(5,
                    mean_scaled=f"{outputs_mean_scaled.mean().item():.6f}",
                    var_scaled=f"{outputs_var_scaled.mean().item():.6f}")

            # 4. Unscale outputs
            outputs_mean = self.unscale_outputs(outputs_mean_scaled, i)
            outputs_var = self.unscale_variance(outputs_var_scaled, i)

            if self.debug:
                self._debug_forward_step(6,
                    outputs_mean=f"{outputs_mean.mean().item():.6f}",
                    outputs_var=f"{outputs_var.mean().item():.6f}",
                    sample_mean=outputs_mean[0].tolist(),
                    sample_var=outputs_var[0].tolist())

            # 5. Sample from distribution using reparameterization trick
            # This makes the actual trajectory stochastic based on predicted uncertainty
            std = torch.sqrt(outputs_var + 1e-8)
            epsilon = torch.randn_like(outputs_mean)
            outputs_sampled = outputs_mean + epsilon * std

            if self.debug:
                self._debug_forward_step(7,
                    std_mean=f"{std.mean().item():.6f}",
                    epsilon_mean=f"{epsilon.mean().item():.6f}",
                    sampled_mean=f"{outputs_sampled.mean().item():.6f}",
                    sample=outputs_sampled[0].tolist())

            # 6. Store in trajectory
            trajectory[process_name] = {
                'inputs': current_inputs,
                'outputs_mean': outputs_mean,
                'outputs_var': outputs_var,
                'outputs_sampled': outputs_sampled  # Actual sampled outputs
            }

            # 7. Update per prossima iterazione
            # Keep all three separate — needed for 'residual' mode (ε computation)
            prev_inputs = current_inputs
            prev_outputs_mean = outputs_mean       # μ_{i}
            prev_outputs_var = outputs_var          # σ²_{i}
            prev_outputs_sampled = outputs_sampled  # o_{i} ~ N(μ, σ²)

            if self.debug:
                self._debug_forward_step(8,
                    prev_outputs_mean=prev_outputs_mean[0].tolist(),
                    prev_outputs_var=prev_outputs_var[0].tolist(),
                    prev_outputs_sampled=prev_outputs_sampled[0].tolist())

        if self.debug:
            self._debug_forward_step(9, status="FORWARD PASS COMPLETE")

        return trajectory

    def trajectory_to_prot_format(self, trajectory: dict) -> tuple:
        """
        Convert trajectory dict to causaliT input format matching build_arrays().

        Produces the same [value, var_id] token format used during surrogate
        training (see scm_ds/convert_azimuth_trajectories.py::build_arrays).

        Each scalar output of each process becomes one token in X:
            X[:, k, :] = [output_value, var_id]   (var_id is 1-based, local to X)

        Falls back to 'outputs_mean' when 'outputs_sampled' is not present.

        Args:
            trajectory: Dict from forward() with structure:
                {process_name: {'inputs', 'outputs_mean', 'outputs_var', 'outputs_sampled'}}

        Returns:
            X: (batch, n_output_vars, 2) - one [value, var_id] per output variable
            Y: (batch, 1, 2) - decoder target placeholder [0.0, var_id]

        Example:
            >>> trajectory = process_chain.forward(batch_size=32, scenario_idx=0)
            >>> X, Y = process_chain.trajectory_to_prot_format(trajectory)
            >>> # 3 processes, 1 output each → X shape: (32, 3, 2)
            >>> # Y shape: (32, 1, 2)
        """
        tokens = []
        batch_size = None
        var_id = 1  # 1-based, local to X group

        for process_name in self.process_names:
            if process_name not in trajectory:
                continue

            data = trajectory[process_name]
            if batch_size is None:
                batch_size = data['inputs'].shape[0]

            # Get sampled outputs (fallback to mean if not available)
            outputs = data.get('outputs_sampled', data['outputs_mean'])  # (batch, output_dim)
            output_dim = outputs.shape[-1]

            # One token per scalar output
            for d in range(output_dim):
                value = outputs[:, d:d+1]  # (batch, 1) — keeps grad
                vid = torch.full((batch_size, 1), var_id, dtype=torch.float32,
                                 device=self.device)
                tokens.append(torch.cat([value, vid], dim=-1))  # (batch, 2)
                var_id += 1

        # Stack tokens: (batch, n_output_vars, 2)
        X = torch.stack(tokens, dim=1)

        # Decoder target placeholder: [0.0, var_id] (same convention as build_arrays y_ids)
        Y = torch.zeros(batch_size, 1, 2, dtype=torch.float32, device=self.device)
        Y[:, 0, 1] = 1.0  # y_ids = [1.0]

        return X, Y

    def forward_prot(self, batch_size: int = 1, scenario_idx: int = 0) -> tuple:
        """
        Forward pass that directly outputs ProT format for causaliT.

        Combines forward() + trajectory_to_prot_format() for convenience.

        Args:
            batch_size: Number of parallel samples
            scenario_idx: Which scenario's structural conditions to use

        Returns:
            X: (batch, n_processes, features) - encoder input for ProT
            Y: (batch, 1, 1) - decoder input placeholder
            trajectory: Original trajectory dict (for debugging/other uses)

        Example:
            >>> X, Y, traj = process_chain.forward_prot(batch_size=32, scenario_idx=0)
            >>> # Feed X, Y directly to TransformerForecaster
            >>> forecast_output, _, _, _ = model.forward(data_input=X, data_trg=Y)
        """
        trajectory = self.forward(batch_size=batch_size, scenario_idx=scenario_idx)
        X, Y = self.trajectory_to_prot_format(trajectory)
        return X, Y, trajectory

    def debug_all_gradients(self):
        """Print gradient statistics for all policy generators."""
        for i, policy in enumerate(self.policy_generators):
            policy.debug_gradients()

    def debug_all_weights(self):
        """Print weight statistics for all policy generators."""
        for i, policy in enumerate(self.policy_generators):
            policy.debug_weights()
