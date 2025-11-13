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

from controller_optimization.src.utils.model_utils import (
    load_uncertainty_predictor,
    load_preprocessor
)
from controller_optimization.src.models.policy_generator import (
    create_small_policy_generator,
    create_medium_policy_generator,
    create_large_policy_generator
)


class ProcessChain(nn.Module):
    """
    Catena di processi con uncertainty predictors frozen e policy generators trainable.

    Sequenza:
    a1 (fisso) → UncertPred1 → (o1, σ1²) → Policy1 → a2 → UncertPred2 → (o2, σ2²) → ...
    """

    def __init__(self, processes_config, target_trajectory, policy_config=None, device='cpu'):
        """
        Args:
            processes_config (list): Lista da PROCESSES
            target_trajectory (dict): Da generate_target_trajectory()
            policy_config (dict): Config for policy generators
            device (str): Device
        """
        super(ProcessChain, self).__init__()

        self.device = device
        self.process_names = [p['name'] for p in processes_config]




        self.processes_config = processes_config
        self.target_trajectory = target_trajectory

        # Default policy config
        if policy_config is None:
            policy_config = {
                'architecture': 'medium',
                'hidden_sizes': [64, 32],
                'dropout': 0.1,
                'use_batchnorm': False
            }
        self.policy_config = policy_config

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

        # Create policy generators (trainable)
        # Policy i generates inputs for process i+1 based on outputs of process i
        self.policy_generators = nn.ModuleList()

        for i in range(len(processes_config) - 1):
            # Input to policy: [prev_inputs, prev_outputs_mean, prev_outputs_var]
            prev_input_dim = processes_config[i]['input_dim']
            prev_output_dim = processes_config[i]['output_dim']
            policy_input_size = prev_input_dim + prev_output_dim + prev_output_dim

            # Output from policy: next process inputs
            next_input_dim = processes_config[i + 1]['input_dim']

            # Create policy generator
            if policy_config['architecture'] == 'small':
                policy = create_small_policy_generator(policy_input_size, next_input_dim)
            elif policy_config['architecture'] == 'medium':
                policy = create_medium_policy_generator(policy_input_size, next_input_dim)
            elif policy_config['architecture'] == 'large':
                policy = create_large_policy_generator(policy_input_size, next_input_dim)
            elif policy_config['architecture'] == 'custom':
                from controller_optimization.src.models.policy_generator import PolicyGenerator
                policy = PolicyGenerator(
                    input_size=policy_input_size,
                    output_size=next_input_dim,
                    hidden_sizes=policy_config['hidden_sizes'],
                    dropout_rate=policy_config['dropout'],
                    use_batchnorm=policy_config['use_batchnorm']
                )
            else:
                raise ValueError(f"Unknown policy architecture: {policy_config['architecture']}")

            policy = policy.to(device)
            self.policy_generators.append(policy)

    def get_initial_inputs(self, batch_size=1, scenario_idx=None):
        """
        Get initial inputs a1 from target trajectory.

        Args:
            batch_size (int): Number of parallel samples
            scenario_idx (int, optional): Which scenario's structural conditions to use.
                                         If None, uses scenario 0.

        Returns:
            torch.Tensor: Initial inputs for first process
        """
        if scenario_idx is None:
            scenario_idx = 0

        first_process_name = self.process_names[0]
        target_inputs_all = self.target_trajectory[first_process_name]['inputs']

        # Select specific scenario
        target_inputs = target_inputs_all[scenario_idx]  # Shape: (input_dim,)

        # Replicate for batch
        initial_inputs = np.tile(target_inputs, (batch_size, 1))
        return torch.tensor(initial_inputs, dtype=torch.float32, device=self.device)

    def scale_inputs(self, inputs, process_idx):
        """Scale inputs using preprocessor."""
        inputs_np = inputs.detach().cpu().numpy()
        inputs_scaled = self.preprocessors[process_idx].input_scaler.transform(inputs_np)
        return torch.tensor(inputs_scaled, dtype=torch.float32, device=self.device)

    def unscale_outputs(self, outputs, process_idx):
        """Unscale outputs using preprocessor."""
        outputs_np = outputs.detach().cpu().numpy()
        outputs_unscaled = self.preprocessors[process_idx].output_scaler.inverse_transform(outputs_np)

        


        # DEBUG
        if process_idx == 'microetch':
            print(f"\n=== UNSCALE OUTPUTS: {process_idx} ===")
            print(f"Scaled outputs (from model): {outputs_np}")
            print(f"Scaler type: {type(self.preprocessors[process_idx].output_scaler)}")
            print(f"Scaler mean_: {self.preprocessors[process_idx].output_scaler.mean_}")
            print(f"Scaler scale_: {self.preprocessors[process_idx].output_scaler.scale_}")
    
        outputs_unscaled = self.preprocessors[process_idx].output_scaler.inverse_transform(outputs_np)
    
        if process_idx == 'microetch':
            print(f"Unscaled outputs: {outputs_unscaled}")
            print(f"Expected range: 0-40")
            print("=" * 50)









        return torch.tensor(outputs_unscaled, dtype=torch.float32, device=self.device)

    def unscale_variance(self, variance, process_idx):
        """Unscale variance (variance scales with scale^2)."""
        output_scale = self.preprocessors[process_idx].output_scaler.scale_
        scale_squared = torch.tensor(output_scale ** 2, dtype=torch.float32, device=self.device)
        return variance * scale_squared

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
                    'outputs_mean': tensor,
                    'outputs_var': tensor
                },
                'plasma': {...},
                ...
            }
        """




        trajectory = {}

        # a1 è fisso dalla target trajectory (per lo scenario specifico)
        current_inputs = self.get_initial_inputs(batch_size, scenario_idx)

        for i, process_name in enumerate(self.process_names):
            # 1. Se i > 0: policy generator produce inputs
            if i > 0:
                # Concatenate: [prev_inputs, prev_outputs_mean, prev_outputs_var]
                policy_input = torch.cat([
                    prev_inputs,
                    prev_outputs_mean,
                    prev_outputs_var
                ], dim=1)

                current_inputs = self.policy_generators[i - 1](policy_input)

            # 2. Scale inputs
            scaled_inputs = self.scale_inputs(current_inputs, i)

            # 3. Uncertainty predictor (frozen)
            outputs_mean_scaled, outputs_var_scaled = self.uncertainty_predictors[i](scaled_inputs)

            # 4. Unscale outputs
            outputs_mean = self.unscale_outputs(outputs_mean_scaled, i)
            outputs_var = self.unscale_variance(outputs_var_scaled, i)

            # 5. Store in trajectory
            trajectory[process_name] = {
                'inputs': current_inputs,
                'outputs_mean': outputs_mean,
                'outputs_var': outputs_var
            }

            # 6. Update per prossima iterazione
            prev_inputs = current_inputs
            prev_outputs_mean = outputs_mean
            prev_outputs_var = outputs_var

        return trajectory

    def evaluate_trajectory(self, trajectory_type='target'):
        """
        Genera trajectory senza training (per evaluation).

        Args:
            trajectory_type (str): 'target' (usa a*) o 'baseline' (usa a' con noise)

        Returns:
            trajectory (dict): Trajectory completa
        """
        # This is used during evaluation, not training
        # Simply use fixed inputs from target trajectory
        with torch.no_grad():
            trajectory = {}

            for i, process_name in enumerate(self.process_names):
                # Use target inputs
                target_inputs = self.target_trajectory[process_name]['inputs']
                current_inputs = torch.tensor(target_inputs, dtype=torch.float32, device=self.device)

                # Scale inputs
                scaled_inputs = self.scale_inputs(current_inputs, i)

                # Uncertainty predictor
                outputs_mean_scaled, outputs_var_scaled = self.uncertainty_predictors[i](scaled_inputs)

                # Unscale outputs
                outputs_mean = self.unscale_outputs(outputs_mean_scaled, i)
                outputs_var = self.unscale_variance(outputs_var_scaled, i)

                trajectory[process_name] = {
                    'inputs': current_inputs,
                    'outputs_mean': outputs_mean,
                    'outputs_var': outputs_var
                }

        return trajectory


if __name__ == '__main__':
    # Test ProcessChain (requires trained models)
    print("Testing ProcessChain...")
    print("Note: This requires trained uncertainty predictors.")
    print("Run train_processes.py first if not already done.")

    from controller_optimization.configs.processes_config import PROCESSES
    from controller_optimization.src.utils.target_generation import generate_target_trajectory

    # Generate target trajectory
    target_traj = generate_target_trajectory(PROCESSES, n_samples=1, seed=42)

    try:
        # Create process chain
        chain = ProcessChain(
            processes_config=PROCESSES,
            target_trajectory=target_traj,
            device='cpu'
        )

        print(f"\nProcessChain created successfully!")
        print(f"  Processes: {chain.process_names}")
        print(f"  Uncertainty predictors: {len(chain.uncertainty_predictors)}")
        print(f"  Policy generators: {len(chain.policy_generators)}")

        # Test forward pass
        trajectory = chain.forward(batch_size=4)

        print(f"\nForward pass test:")
        for process_name, data in trajectory.items():
            print(f"  {process_name}:")
            print(f"    Inputs shape: {data['inputs'].shape}")
            print(f"    Outputs mean shape: {data['outputs_mean'].shape}")
            print(f"    Outputs var shape: {data['outputs_var'].shape}")

        print("\n✓ ProcessChain test passed!")

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("Please run train_processes.py first to train uncertainty predictors.")
