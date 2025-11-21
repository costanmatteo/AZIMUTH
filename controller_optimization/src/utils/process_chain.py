"""
Process Chain: orchestrazione della sequenza di processi.

Gestisce:
- Funzioni SCM realistiche (structural noise=0, process noise=attivo)
- Policy generators (trainable)
- Preprocessors (scaling/unscaling)
- Forward pass attraverso tutta la catena

Il sistema ora simula condizioni realistiche come baseline:
- Structural conditions fisse per scenario (es. AmbientTemp)
- Process noise attivo (es. shot noise, drift)
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
    load_scm,
    load_preprocessor
)
from controller_optimization.src.models.policy_generator import (
    create_small_policy_generator,
    create_medium_policy_generator,
    create_large_policy_generator
)
from controller_optimization.src.models.scenario_encoder import ScenarioEncoder


class ProcessChain(nn.Module):
    """
    Catena di processi con funzioni SCM realistiche e policy generators trainable.

    Sequenza:
    a1 (fisso) → SCM1 → o1 → Policy1 → a2 → SCM2 → o2 → ...

    Noise model:
    - Structural noise (es. AmbientTemp): Fisso per scenario (da target)
    - Process noise (es. shot noise, drift): Attivo (variabilità realistica)

    Questo simula il comportamento baseline: stessi input, ma con variabilità del processo.
    """

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

        # Load SCM functions with noise models
        self.scm_functions = []
        self.scm_input_labels = []
        self.scm_output_labels = []
        self.scm_datasets = []  # Complete datasets for noise classification
        self.preprocessors = []

        for process_config in processes_config:
            checkpoint_dir = Path(process_config['checkpoint_dir'])
            scaler_path = checkpoint_dir / 'scalers.pkl'

            if not scaler_path.exists():
                raise FileNotFoundError(
                    f"Preprocessor not found for process '{process_config['name']}'. "
                    f"Run train_processes.py first to generate scalers."
                )

            # Load SCM with complete dataset
            scm, input_labels, output_labels, ds_scm = load_scm(process_config)
            self.scm_functions.append(scm)
            self.scm_input_labels.append(input_labels)
            self.scm_output_labels.append(output_labels)
            self.scm_datasets.append(ds_scm)

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
            # Input to policy: [prev_outputs, scenario_embedding]
            # Note: No uncertainty/variance, just deterministic outputs
            prev_output_dim = processes_config[i]['output_dim']
            policy_input_size = prev_output_dim

            # Add scenario embedding dimension if encoder is enabled
            if self.use_scenario_encoder:
                policy_input_size += self.scenario_embedding_dim

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

    def compute_scm_outputs(self, inputs, process_idx, seed=None):
        """
        Calcola output usando funzioni SCM con process noise attivo.

        Noise model:
        - Structural noise (es. AmbientTemp): ZERO (già fissato negli input)
        - Process noise (es. shot noise, drift): ATTIVO (campionato)

        Args:
            inputs (torch.Tensor): Input tensor, shape (batch_size, input_dim)
            process_idx (int): Index del processo
            seed (int, optional): Random seed per reproducibilità

        Returns:
            torch.Tensor: Output con process noise, shape (batch_size, output_dim)
        """
        batch_size = inputs.shape[0]
        scm = self.scm_functions[process_idx]
        input_labels = self.scm_input_labels[process_idx]
        output_labels = self.scm_output_labels[process_idx]
        ds_scm = self.scm_datasets[process_idx]

        # Convert to numpy for SCM evaluation
        inputs_np = inputs.detach().cpu().numpy()

        # Evaluate SCM for each sample in batch
        outputs_list = []
        for i in range(batch_size):
            # Create RNG with unique seed for each batch item
            if seed is not None:
                rng = np.random.default_rng(seed + i)
            else:
                rng = np.random.default_rng()

            # Create context with input values
            context = {}
            for j, label in enumerate(input_labels):
                context[label] = np.array([inputs_np[i, j]])

            # Create noise draws: structural=0, process=sampled
            eps_draws = {}
            for node in scm.specs.keys():
                if node in ds_scm.structural_noise_vars:
                    # Structural noise: zero (deterministic for this scenario)
                    eps_draws[node] = np.zeros(1)
                elif node in ds_scm.process_noise_vars:
                    # Process noise: ACTIVE (realistic variability)
                    eps_draws[node] = rng.standard_normal(1)
                else:
                    # Other nodes (inputs, constants): zero
                    eps_draws[node] = np.zeros(1)

            # Forward pass through SCM
            scm.forward(context, eps_draws)

            # Extract outputs
            outputs = np.array([context[label][0] for label in output_labels])
            outputs_list.append(outputs)

        # Convert back to torch tensor
        outputs_np = np.array(outputs_list)  # Shape: (batch_size, output_dim)
        return torch.tensor(outputs_np, dtype=torch.float32, device=self.device)

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
                    'outputs': tensor (outputs from SCM with process noise)
                },
                'plasma': {...},
                ...
            }

        Note: Gli output includono variabilità realistica da process noise,
              come nel comportamento baseline.
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
                # Concatenate: [prev_outputs, scenario_embedding]
                policy_input_parts = [prev_outputs]

                # Add scenario embedding if encoder is enabled
                if self.use_scenario_encoder:
                    policy_input_parts.append(scenario_embedding)

                policy_input = torch.cat(policy_input_parts, dim=1)

                generated_inputs = self.policy_generators[i - 1](policy_input)

                # Apply non-controllable constraints: replace non-controllable inputs
                # with values from target trajectory (e.g., Temperature for microetch)
                current_inputs = self._apply_non_controllable_constraints(
                    generated_inputs, i, scenario_idx, batch_size
                )

            # 2. Scale inputs
            scaled_inputs = self.scale_inputs(current_inputs, i)

            # 3. Compute deterministic outputs using SCM
            outputs_scaled = self.compute_scm_outputs(scaled_inputs, i)

            # 4. Unscale outputs
            outputs = self.unscale_outputs(outputs_scaled, i)

            # 5. Store in trajectory
            trajectory[process_name] = {
                'inputs': current_inputs,
                'outputs': outputs
            }

            # 6. Update per prossima iterazione
            prev_outputs = outputs

        return trajectory

    def evaluate_trajectory(self, trajectory_type='target'):
        """
        Genera trajectory senza training (per evaluation).

        Args:
            trajectory_type (str): 'target' (usa a*) - baseline non più necessario

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

                # Compute deterministic outputs using SCM
                outputs_scaled = self.compute_scm_outputs(scaled_inputs, i)

                # Unscale outputs
                outputs = self.unscale_outputs(outputs_scaled, i)

                trajectory[process_name] = {
                    'inputs': current_inputs,
                    'outputs': outputs
                }

        return trajectory


if __name__ == '__main__':
    # Test ProcessChain
    print("Testing ProcessChain...")
    print("Note: This requires preprocessor scalers.")
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
        print(f"  SCM functions: {len(chain.scm_functions)}")
        print(f"  Policy generators: {len(chain.policy_generators)}")

        # Test forward pass
        trajectory = chain.forward(batch_size=4)

        print(f"\nForward pass test:")
        for process_name, data in trajectory.items():
            print(f"  {process_name}:")
            print(f"    Inputs shape: {data['inputs'].shape}")
            print(f"    Outputs shape: {data['outputs'].shape}")

        print("\n✓ ProcessChain test passed!")

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("Please run train_processes.py first to generate preprocessor scalers.")
