"""
Trajectory Generator for CasualiT Surrogate Training.

Generates trajectories with reliability labels:
1. Sample random controllable parameters within scm_ds boundaries
2. Run through ProcessChain to get trajectory (inputs → outputs)
3. Compute F using reliability_function (ground truth label)
4. Format as CasualiT input (sequence of 4 process steps)
"""

import sys
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add paths for imports
sys.path.insert(0, '/home/user/AZIMUTH')

from casualit_surrogate.configs.surrogate_config import (
    INPUT_BOUNDARIES,
    CONTROLLABLE_INPUTS,
    STRUCTURAL_INPUTS,
)


@dataclass
class ScenarioConfig:
    """Configuration for a single scenario."""
    scenario_idx: int
    structural_values: Dict[str, Dict[str, float]]  # process -> input -> value
    seed: int


class TrajectoryGenerator:
    """
    Generate trajectories with reliability labels for CasualiT training.

    Each trajectory is a sequence of (inputs, outputs) for each process,
    labeled with the final reliability F computed by reliability_function.
    """

    def __init__(self,
                 process_chain,
                 reliability_fn,
                 config: Dict,
                 device: str = 'cpu'):
        """
        Args:
            process_chain: ProcessChain instance for running trajectories
            reliability_fn: ReliabilityFunction instance for computing F labels
            config: Configuration dict (from SURROGATE_CONFIG)
            device: Torch device
        """
        self.process_chain = process_chain
        self.reliability_fn = reliability_fn
        self.config = config
        self.device = device

        # Get process order from process_chain
        self.process_names = config.get('process_names', ['laser', 'plasma', 'galvanic', 'microetch'])

        # Input boundaries for sampling
        self.input_boundaries = INPUT_BOUNDARIES
        self.controllable_inputs = CONTROLLABLE_INPUTS
        self.structural_inputs = STRUCTURAL_INPUTS

    def generate_scenarios(self, n_scenarios: int, seed: int, seed_offset: int = 0) -> List[ScenarioConfig]:
        """
        Generate scenario configurations with diverse structural parameters.

        Args:
            n_scenarios: Number of scenarios to generate
            seed: Base random seed
            seed_offset: Offset for test scenarios

        Returns:
            List of ScenarioConfig objects
        """
        rng = np.random.default_rng(seed + seed_offset)
        scenarios = []

        for i in range(n_scenarios):
            # Sample structural/environmental parameters for this scenario
            structural_values = {}

            for process_name in self.process_names:
                structural_values[process_name] = {}
                structural_vars = self.structural_inputs.get(process_name, [])

                for var_name in structural_vars:
                    bounds = self.input_boundaries.get(process_name, {}).get(var_name)
                    if bounds:
                        value = rng.uniform(bounds[0], bounds[1])
                        structural_values[process_name][var_name] = value

            scenarios.append(ScenarioConfig(
                scenario_idx=i,
                structural_values=structural_values,
                seed=seed + seed_offset + i * 1000,
            ))

        return scenarios

    def generate_trajectories_for_scenario(self,
                                           scenario: ScenarioConfig,
                                           n_trajectories: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate trajectories for a single scenario.

        Args:
            scenario: ScenarioConfig with structural parameters fixed
            n_trajectories: Number of trajectories to generate

        Returns:
            X: Trajectory features (n_trajectories, n_processes, features_per_process)
            Y: Reliability labels (n_trajectories, 1)
        """
        rng = np.random.default_rng(scenario.seed)

        trajectories_X = []
        trajectories_Y = []

        for traj_idx in range(n_trajectories):
            # Sample controllable parameters
            controllable_values = self._sample_controllable_params(rng)

            # Combine with structural parameters from scenario
            full_inputs = self._combine_inputs(controllable_values, scenario.structural_values)

            # Run through process chain
            trajectory = self._run_process_chain(full_inputs)

            # Compute reliability F using reliability_function
            F = self._compute_reliability(trajectory)

            # Convert trajectory to feature sequence
            X_seq = self._trajectory_to_sequence(trajectory, full_inputs)

            trajectories_X.append(X_seq)
            trajectories_Y.append(F)

        X = np.stack(trajectories_X, axis=0)
        Y = np.array(trajectories_Y).reshape(-1, 1)

        return X, Y

    def generate_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate full training and test datasets.

        Returns:
            X_train, Y_train, X_test, Y_test
        """
        scenarios_cfg = self.config.get('scenarios', {})
        n_train = scenarios_cfg.get('n_train', 50)
        n_test = scenarios_cfg.get('n_test', 10)
        n_trajectories = scenarios_cfg.get('n_trajectories_per_scenario', 200)
        seed = scenarios_cfg.get('seed', 42)
        seed_offset_test = scenarios_cfg.get('seed_offset_test', 1000)

        # Generate train scenarios
        train_scenarios = self.generate_scenarios(n_train, seed, seed_offset=0)

        # Generate test scenarios (different structural conditions)
        test_scenarios = self.generate_scenarios(n_test, seed, seed_offset=seed_offset_test)

        # Generate trajectories for each scenario
        X_train_list, Y_train_list = [], []
        for scenario in train_scenarios:
            X, Y = self.generate_trajectories_for_scenario(scenario, n_trajectories)
            X_train_list.append(X)
            Y_train_list.append(Y)

        X_test_list, Y_test_list = [], []
        for scenario in test_scenarios:
            X, Y = self.generate_trajectories_for_scenario(scenario, n_trajectories)
            X_test_list.append(X)
            Y_test_list.append(Y)

        # Concatenate
        X_train = np.concatenate(X_train_list, axis=0)
        Y_train = np.concatenate(Y_train_list, axis=0)
        X_test = np.concatenate(X_test_list, axis=0)
        Y_test = np.concatenate(Y_test_list, axis=0)

        return X_train, Y_train, X_test, Y_test

    def _sample_controllable_params(self, rng: np.random.Generator) -> Dict[str, Dict[str, float]]:
        """
        Sample random controllable parameters within boundaries.

        Args:
            rng: NumPy random generator

        Returns:
            Dict mapping process_name -> input_name -> value
        """
        controllable_values = {}

        for process_name in self.process_names:
            controllable_values[process_name] = {}
            controllable_vars = self.controllable_inputs.get(process_name, [])

            for var_name in controllable_vars:
                bounds = self.input_boundaries.get(process_name, {}).get(var_name)
                if bounds:
                    value = rng.uniform(bounds[0], bounds[1])
                    controllable_values[process_name][var_name] = value

        return controllable_values

    def _combine_inputs(self,
                       controllable: Dict[str, Dict[str, float]],
                       structural: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Combine controllable and structural inputs."""
        combined = {}
        for process_name in self.process_names:
            combined[process_name] = {}
            combined[process_name].update(controllable.get(process_name, {}))
            combined[process_name].update(structural.get(process_name, {}))
        return combined

    def _run_process_chain(self, inputs: Dict[str, Dict[str, float]]) -> Dict:
        """
        Run inputs through process chain to get trajectory.

        Args:
            inputs: Full inputs for each process

        Returns:
            Trajectory dict with inputs/outputs for each process
        """
        # Convert inputs dict to tensor format expected by process chain
        # This needs to match the ProcessChain interface
        trajectory = {}

        # Build input tensors for each process
        for process_name in self.process_names:
            process_inputs = inputs.get(process_name, {})

            # Get input order from process config
            # For now, use alphabetical order within boundaries
            input_names = sorted(self.input_boundaries.get(process_name, {}).keys())

            input_values = []
            for name in input_names:
                if name in process_inputs:
                    input_values.append(process_inputs[name])
                else:
                    # Use middle of range as default
                    bounds = self.input_boundaries[process_name].get(name, (0, 1))
                    input_values.append((bounds[0] + bounds[1]) / 2)

            input_tensor = torch.tensor([input_values], dtype=torch.float32, device=self.device)

            # Run through process model
            if hasattr(self.process_chain, 'process_models') and process_name in self.process_chain.process_models:
                model = self.process_chain.process_models[process_name]

                # Handle sequential dependency: previous outputs become part of next inputs
                # This is simplified - actual process chain may have different logic
                with torch.no_grad():
                    output_mean, output_var = model(input_tensor)

                trajectory[process_name] = {
                    'inputs': input_tensor.cpu().numpy(),
                    'outputs_mean': output_mean.cpu().numpy(),
                    'outputs_var': output_var.cpu().numpy(),
                    'outputs_sampled': output_mean.cpu().numpy(),  # Use mean for now
                }
            else:
                # Fallback: just store inputs with dummy outputs
                trajectory[process_name] = {
                    'inputs': input_tensor.cpu().numpy(),
                    'outputs_mean': np.zeros((1, 1)),
                    'outputs_var': np.zeros((1, 1)),
                    'outputs_sampled': np.zeros((1, 1)),
                }

        return trajectory

    def _compute_reliability(self, trajectory: Dict) -> float:
        """
        Compute reliability F using reliability_function.

        Args:
            trajectory: Trajectory dict with outputs for each process

        Returns:
            Reliability F value
        """
        # Convert to tensor format
        traj_tensors = {}
        for process_name, data in trajectory.items():
            traj_tensors[process_name] = {
                'outputs_mean': torch.tensor(data['outputs_mean'], dtype=torch.float32),
                'outputs_sampled': torch.tensor(data['outputs_sampled'], dtype=torch.float32),
            }

        F = self.reliability_fn.compute_reliability(traj_tensors)

        if isinstance(F, torch.Tensor):
            return F.item()
        return float(F)

    def _trajectory_to_sequence(self,
                                trajectory: Dict,
                                inputs: Dict[str, Dict[str, float]]) -> np.ndarray:
        """
        Convert trajectory to sequence format for CasualiT.

        Args:
            trajectory: Trajectory dict with inputs/outputs
            inputs: Original input values

        Returns:
            Sequence array (n_processes, features_per_process)
        """
        sequence = []

        for process_name in self.process_names:
            proc_data = trajectory.get(process_name, {})

            # Features: [inputs..., output_mean, output_var]
            input_arr = proc_data.get('inputs', np.zeros((1, 2)))
            output_mean = proc_data.get('outputs_mean', np.zeros((1, 1)))
            output_var = proc_data.get('outputs_var', np.zeros((1, 1)))

            # Flatten and concatenate
            features = np.concatenate([
                input_arr.flatten(),
                output_mean.flatten(),
                output_var.flatten(),
            ])

            sequence.append(features)

        # Pad sequences to same length if needed
        max_len = max(len(s) for s in sequence)
        padded_sequence = []
        for s in sequence:
            if len(s) < max_len:
                s = np.pad(s, (0, max_len - len(s)), mode='constant')
            padded_sequence.append(s)

        return np.stack(padded_sequence, axis=0)


def create_trajectory_generator(config: Dict, device: str = 'cpu') -> TrajectoryGenerator:
    """
    Factory function to create TrajectoryGenerator with dependencies.

    Args:
        config: SURROGATE_CONFIG
        device: Torch device

    Returns:
        TrajectoryGenerator instance
    """
    from reliability_function import ReliabilityFunction

    # Import process chain (will need actual implementation)
    try:
        from controller_optimization.src.utils.process_chain import ProcessChain
        from controller_optimization.configs.processes_config import get_filtered_processes

        process_configs = get_filtered_processes(config.get('process_names'))
        process_chain = ProcessChain(process_configs, device=device)
    except ImportError:
        # Fallback: create dummy process chain
        process_chain = None

    reliability_fn = ReliabilityFunction(device=device)

    return TrajectoryGenerator(
        process_chain=process_chain,
        reliability_fn=reliability_fn,
        config=config,
        device=device,
    )
