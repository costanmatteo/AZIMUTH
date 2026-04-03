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
        from controller_optimization.configs.processes_config import get_controllable_inputs

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

            # policy_input_size = prev_mean + prev_var + non_controllable_inputs
            policy_input_size = prev_output_dim + prev_output_dim + n_non_controllable

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
                # Extract bounds only for controllable inputs
                output_min = torch.tensor(
                    [next_preprocessor.input_min[idx] for idx in controllable_indices],
                    dtype=torch.float32
                )
                output_max = torch.tensor(
                    [next_preprocessor.input_max[idx] for idx in controllable_indices],
                    dtype=torch.float32
                )
                print(f"  Policy {i} -> Process '{processes_config[i + 1]['name']}':")
                print(f"    Input dim: {policy_input_size} (prev_mean={prev_output_dim} + prev_var={prev_output_dim} + non_controllable={n_non_controllable})")
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
