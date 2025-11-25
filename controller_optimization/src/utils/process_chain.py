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
    PolicyGenerator,
    create_small_policy_generator,
    create_medium_policy_generator,
    create_large_policy_generator
)
from controller_optimization.src.models.scenario_encoder import ScenarioEncoder


class ProcessChain(nn.Module):
    """
    Catena di processi con uncertainty predictors frozen e policy generators trainable.

    Sequenza:
    a1 (fisso) → UncertPred1 → (o1, σ1²) → Policy1 → a2 → UncertPred2 → (o2, σ2²) → ...
    """

    # Class-level debug flag
    debug = False

    @classmethod
    def enable_debug(cls, enable=True):
        """Enable/disable debug mode for ProcessChain and all PolicyGenerators."""
        cls.debug = enable
        PolicyGenerator.debug = enable
        print(f"Debug mode {'ENABLED' if enable else 'DISABLED'} for ProcessChain and PolicyGenerator")

    def debug_all_gradients(self):
        """Print gradient statistics for all policy generators."""
        for i, policy in enumerate(self.policy_generators):
            policy.debug_gradients()

    def debug_all_weights(self):
        """Print weight statistics for all policy generators."""
        for i, policy in enumerate(self.policy_generators):
            policy.debug_weights()

    @staticmethod
    def _count_structural_params(processes_config):
        """
        Conta il numero totale di parametri strutturali (non-controllabili) in tutti i processi.

        Args:
            processes_config (list): Lista di configurazioni dei processi

        Returns:
            int: Numero totale di parametri strutturali
        """
        from controller_optimization.configs.processes_config import get_controllable_inputs

        total = 0
        for process_config in processes_config:
            input_labels = process_config['input_labels']
            controllable = get_controllable_inputs(process_config)
            # Count non-controllable inputs
            n_non_controllable = len([label for label in input_labels if label not in controllable])
            total += n_non_controllable

        return total

    def _extract_structural_params(self, scenario_idx):
        """
        Estrae parametri strutturali (non-controllabili) per uno scenario specifico.

        Args:
            scenario_idx (int): Index dello scenario

        Returns:
            torch.Tensor: Parametri strutturali, shape (n_structural_params,)
        """
        from controller_optimization.configs.processes_config import get_controllable_inputs

        structural_values = []

        for i, process_config in enumerate(self.processes_config):
            process_name = process_config['name']
            input_labels = process_config['input_labels']
            controllable = get_controllable_inputs(process_config)

            # Get target inputs for this scenario
            target_inputs = self.target_trajectory[process_name]['inputs'][scenario_idx]

            # Extract non-controllable values
            for idx, label in enumerate(input_labels):
                if label not in controllable:
                    structural_values.append(target_inputs[idx])

        if len(structural_values) == 0:
            # No structural params → return dummy zero tensor
            return torch.tensor([0.0], dtype=torch.float32, device=self.device)

        return torch.tensor(structural_values, dtype=torch.float32, device=self.device)

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
        # Policy i generates inputs for process i+1 based on outputs of process i
        self.policy_generators = nn.ModuleList()

        for i in range(len(processes_config) - 1):
            # Input to policy: [prev_inputs, prev_outputs_mean, prev_outputs_var, scenario_embedding]
            prev_input_dim = processes_config[i]['input_dim']
            prev_output_dim = processes_config[i]['output_dim']
            policy_input_size = prev_output_dim + prev_output_dim #+ prev_input_dim

            # Add scenario embedding dimension if encoder is enabled
            if self.use_scenario_encoder:
                policy_input_size += self.scenario_embedding_dim

            # Output from policy: next process inputs
            next_input_dim = processes_config[i + 1]['input_dim']

            # Get output bounds from next process's preprocessor (derived from UP training data)
            next_preprocessor = self.preprocessors[i + 1]
            if next_preprocessor.input_min is not None and next_preprocessor.input_max is not None:
                output_min = torch.tensor(next_preprocessor.input_min, dtype=torch.float32)
                output_max = torch.tensor(next_preprocessor.input_max, dtype=torch.float32)
                print(f"  Policy {i} -> Process '{processes_config[i + 1]['name']}' bounds:")
                print(f"    Min: {output_min.numpy()}")
                print(f"    Max: {output_max.numpy()}")
            else:
                output_min = None
                output_max = None
                print(f"  Policy {i} -> Process '{processes_config[i + 1]['name']}': No bounds (unbounded output)")

            # Create policy generator with bounds
            if policy_config['architecture'] == 'small':
                policy = create_small_policy_generator(
                    policy_input_size, next_input_dim,
                    output_min=output_min, output_max=output_max
                )
            elif policy_config['architecture'] == 'medium':
                policy = create_medium_policy_generator(
                    policy_input_size, next_input_dim,
                    output_min=output_min, output_max=output_max
                )
            elif policy_config['architecture'] == 'large':
                policy = create_large_policy_generator(
                    policy_input_size, next_input_dim,
                    output_min=output_min, output_max=output_max
                )
            elif policy_config['architecture'] == 'custom':
                from controller_optimization.src.models.policy_generator import PolicyGenerator
                policy = PolicyGenerator(
                    input_size=policy_input_size,
                    output_size=next_input_dim,
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

    def _apply_non_controllable_constraints(self, generated_inputs, process_idx, scenario_idx, batch_size):
        """
        Replaces non-controllable inputs with values from target trajectory.

        This ensures that environmental/fixed conditions (e.g., ambient temperature)
        are inherited from the target scenario, while controllable inputs
        are generated by the policy network.

        Args:
            generated_inputs: Tensor with all inputs generated by policy generator
                            Shape: (batch_size, input_dim)
            process_idx: Index of the current process
            scenario_idx: Index of the scenario (for retrieving target values)
            batch_size: Batch size

        Returns:
            Tensor with controllable inputs from policy, non-controllable from target
            Shape: (batch_size, input_dim)
        """
        from controller_optimization.configs.processes_config import get_controllable_inputs

        process_name = self.process_names[process_idx]
        process_config = self.processes_config[process_idx]
        input_labels = process_config['input_labels']
        controllable = get_controllable_inputs(process_config)

        # If all inputs are controllable, return as-is (optimization)
        if len(controllable) == len(input_labels):
            return generated_inputs

        # Clone tensor to avoid modifying the original
        constrained_inputs = generated_inputs.clone()

        # Get target inputs for this scenario
        target_inputs_all = self.target_trajectory[process_name]['inputs']
        target_inputs = target_inputs_all[scenario_idx]  # Shape: (input_dim,)

        # Replace non-controllable inputs with target values
        for idx, label in enumerate(input_labels):
            if label not in controllable:
                # This input is NOT controllable -> use value from target
                target_value = target_inputs[idx]
                constrained_inputs[:, idx] = target_value  # Broadcast to batch

        return constrained_inputs

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

        # Extract and encode scenario structural parameters (if encoder is enabled)
        if self.use_scenario_encoder:
            structural_params = self._extract_structural_params(scenario_idx)  # Shape: (n_params,)
            # Add batch dimension and replicate for batch
            structural_params = structural_params.unsqueeze(0).repeat(batch_size, 1)  # (batch_size, n_params)
            # Encode to embedding
            scenario_embedding = self.scenario_encoder(structural_params)  # (batch_size, embedding_dim)
        else:
            scenario_embedding = None

        for i, process_name in enumerate(self.process_names):
            # 1. Se i > 0: policy generator produce inputs
            if i > 0:
                # Concatenate: [prev_inputs, prev_outputs_mean, prev_outputs_var, scenario_embedding]
                policy_input_parts = [
                   # prev_inputs,
                    prev_outputs_mean,
                    prev_outputs_var
                ]

                # Add scenario embedding if encoder is enabled
                if self.use_scenario_encoder:
                    policy_input_parts.append(scenario_embedding)

                policy_input = torch.cat(policy_input_parts, dim=1)

                generated_inputs = self.policy_generators[i - 1](policy_input)

                # Debug: print policy generator input/output
                if ProcessChain.debug:
                    with torch.no_grad():
                        print(f"\n{'#'*70}")
                        print(f"# PROCESS CHAIN DEBUG: {self.process_names[i-1]} -> {process_name}")
                        print(f"{'#'*70}")
                        print(f"Policy input (prev_outputs_mean):")
                        print(f"  mean={prev_outputs_mean.mean().item():.4f}, std={prev_outputs_mean.std().item():.4f}")
                        print(f"  min={prev_outputs_mean.min().item():.4f}, max={prev_outputs_mean.max().item():.4f}")
                        print(f"Policy input (prev_outputs_var):")
                        print(f"  mean={prev_outputs_var.mean().item():.4f}, std={prev_outputs_var.std().item():.4f}")
                        print(f"Generated inputs (before constraints):")
                        for dim_idx in range(generated_inputs.shape[1]):
                            label = self.processes_config[i]['input_labels'][dim_idx]
                            print(f"  {label}: mean={generated_inputs[:, dim_idx].mean().item():.4f}, "
                                  f"min={generated_inputs[:, dim_idx].min().item():.4f}, "
                                  f"max={generated_inputs[:, dim_idx].max().item():.4f}")

                # Apply non-controllable constraints: replace non-controllable inputs
                # with values from target trajectory (e.g., Temperature for microetch)
                current_inputs = self._apply_non_controllable_constraints(
                    generated_inputs, i, scenario_idx, batch_size
                )

                # Debug: print after constraints
                if ProcessChain.debug:
                    with torch.no_grad():
                        print(f"Current inputs (after constraints):")
                        for dim_idx in range(current_inputs.shape[1]):
                            label = self.processes_config[i]['input_labels'][dim_idx]
                            print(f"  {label}: mean={current_inputs[:, dim_idx].mean().item():.4f}, "
                                  f"min={current_inputs[:, dim_idx].min().item():.4f}, "
                                  f"max={current_inputs[:, dim_idx].max().item():.4f}")

            # 2. Scale inputs
            scaled_inputs = self.scale_inputs(current_inputs, i)

            # 3. Uncertainty predictor (frozen)
            outputs_mean_scaled, outputs_var_scaled = self.uncertainty_predictors[i](scaled_inputs)

            # 4. Unscale outputs
            outputs_mean = self.unscale_outputs(outputs_mean_scaled, i)
            outputs_var = self.unscale_variance(outputs_var_scaled, i)

            # 5. Sample from distribution using reparameterization trick
            # This makes the actual trajectory stochastic based on predicted uncertainty
            std = torch.sqrt(outputs_var + 1e-8)
            epsilon = torch.randn_like(outputs_mean)
            outputs_sampled = outputs_mean + epsilon * std

            # 6. Store in trajectory
            trajectory[process_name] = {
                'inputs': current_inputs,
                'outputs_mean': outputs_mean,
                'outputs_var': outputs_var,
                'outputs_sampled': outputs_sampled  # Actual sampled outputs
            }

            # 7. Update per prossima iterazione
            # Use sampled outputs as feedback for next policy generator
            # This propagates uncertainty through the chain
            prev_inputs = current_inputs
            prev_outputs_mean = outputs_sampled  # Use sampled outputs instead of mean
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

                # Sample from distribution
                std = torch.sqrt(outputs_var + 1e-8)
                epsilon = torch.randn_like(outputs_mean)
                outputs_sampled = outputs_mean + epsilon * std

                trajectory[process_name] = {
                    'inputs': current_inputs,
                    'outputs_mean': outputs_mean,
                    'outputs_var': outputs_var,
                    'outputs_sampled': outputs_sampled
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
