"""
Data Generator for CasualiT Surrogate Training.

Generates trajectory data with reliability F labels for training the
TransformerForecaster to predict F from process chain trajectories.

The generator:
1. Creates random controllable parameters within uncertainty predictor bounds
2. Runs trajectories through ProcessChain
3. Computes F using ProTSurrogate (mathematical formula)
4. Converts data to ProT format using ProcessChain.trajectory_to_prot_format()
"""

import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# Add paths
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from controller_optimization.configs.processes_config import PROCESSES, get_filtered_processes
from controller_optimization.src.utils.target_generation import generate_target_trajectory
from controller_optimization.src.utils.process_chain import ProcessChain
from controller_optimization.src.models.surrogate import ProTSurrogate


class TrajectoryDataGenerator:
    """
    Generates trajectory data with F labels for surrogate training.

    Produces data in ProT format:
        X: (n_samples, n_processes, features) - encoder input
        Y: (n_samples, 1, 1) - reliability F
    """

    def __init__(self, config: dict, device: str = 'cpu'):
        """
        Args:
            config: Data generation configuration
            device: Torch device
        """
        self.config = config
        self.device = device

        # Get process configuration
        process_names = config.get('process_names', None)
        self.processes_config = get_filtered_processes(PROCESSES, process_names)
        self.process_names = [p['name'] for p in self.processes_config]

        print(f"TrajectoryDataGenerator initialized:")
        print(f"  Processes: {self.process_names}")
        print(f"  Device: {device}")

    def generate_dataset(self,
                        n_trajectories: int,
                        n_scenarios: int = 10,
                        seed: int = 42,
                        batch_size: int = 100,
                        verbose: bool = True) -> dict:
        """
        Generate a dataset of trajectories with F labels.

        Args:
            n_trajectories: Total number of trajectories to generate
            n_scenarios: Number of different scenario conditions
            seed: Random seed
            batch_size: Batch size for generation
            verbose: Print progress

        Returns:
            dict: {
                'X': np.array (n_samples, n_processes, features),
                'Y': np.array (n_samples, 1, 1) - F values,
                'metadata': {scenario_indices, seeds, ...}
            }
        """
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Calculate trajectories per scenario
        trajectories_per_scenario = n_trajectories // n_scenarios
        remaining = n_trajectories % n_scenarios

        all_X = []
        all_Y = []
        all_scenario_idx = []
        all_F_star = []

        if verbose:
            print(f"\nGenerating {n_trajectories} trajectories across {n_scenarios} scenarios...")
            scenario_iter = tqdm(range(n_scenarios), desc="Scenarios")
        else:
            scenario_iter = range(n_scenarios)

        for scenario_idx in scenario_iter:
            # Generate target trajectory for this scenario
            scenario_seed = seed + scenario_idx * self.config.get('scenario_seed_offset', 1000)
            target_trajectory = generate_target_trajectory(
                self.processes_config,
                n_samples=1,
                seed=scenario_seed
            )

            # Create ProcessChain for this scenario
            process_chain = ProcessChain(
                processes_config=self.processes_config,
                target_trajectory=target_trajectory,
                policy_config={'architecture': 'medium', 'use_scenario_encoder': False},
                device=self.device
            )

            # Create ProTSurrogate for computing F
            surrogate = ProTSurrogate(
                target_trajectory=target_trajectory,
                device=self.device,
                use_deterministic_sampling=False  # Use stochastic sampling for diversity
            )

            # Determine number of trajectories for this scenario
            n_traj_scenario = trajectories_per_scenario + (1 if scenario_idx < remaining else 0)

            # Generate trajectories in batches
            n_batches = (n_traj_scenario + batch_size - 1) // batch_size

            for batch_idx in range(n_batches):
                current_batch_size = min(batch_size, n_traj_scenario - batch_idx * batch_size)

                # Generate random trajectories
                with torch.no_grad():
                    trajectory = process_chain.forward(
                        batch_size=current_batch_size,
                        scenario_idx=0
                    )

                    # Convert to ProT format
                    X, _ = process_chain.trajectory_to_prot_format(trajectory)

                    # Compute F for each trajectory
                    F_values = []
                    for i in range(current_batch_size):
                        # Extract single trajectory
                        single_traj = {}
                        for process_name, data in trajectory.items():
                            single_traj[process_name] = {
                                'inputs': data['inputs'][i:i+1],
                                'outputs_mean': data['outputs_mean'][i:i+1],
                                'outputs_var': data['outputs_var'][i:i+1],
                                'outputs_sampled': data['outputs_sampled'][i:i+1],
                            }
                        F = surrogate.compute_reliability(single_traj)
                        F_values.append(F.item())

                    F_tensor = torch.tensor(F_values, dtype=torch.float32).view(-1, 1, 1)

                # Store results
                all_X.append(X.cpu().numpy())
                all_Y.append(F_tensor.numpy())
                all_scenario_idx.extend([scenario_idx] * current_batch_size)
                all_F_star.extend([surrogate.F_star[0]] * current_batch_size)

        # Concatenate all batches
        X = np.concatenate(all_X, axis=0)
        Y = np.concatenate(all_Y, axis=0)

        if verbose:
            print(f"\nDataset generated:")
            print(f"  X shape: {X.shape}")
            print(f"  Y shape: {Y.shape}")
            print(f"  F range: [{Y.min():.4f}, {Y.max():.4f}]")
            print(f"  F mean: {Y.mean():.4f} +/- {Y.std():.4f}")

        return {
            'X': X,
            'Y': Y,
            'metadata': {
                'scenario_indices': np.array(all_scenario_idx),
                'F_star': np.array(all_F_star),
                'n_scenarios': n_scenarios,
                'seed': seed,
                'process_names': self.process_names,
                'n_processes': len(self.process_names),
                'features_per_process': X.shape[2],
            }
        }

    def save_dataset(self, data: dict, output_dir: str, prefix: str = 'train'):
        """
        Save dataset to disk.

        Args:
            data: Dataset dict from generate_dataset()
            output_dir: Output directory
            prefix: Filename prefix ('train', 'val', 'test')
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save data
        np.save(output_path / f'{prefix}_X.npy', data['X'])
        np.save(output_path / f'{prefix}_Y.npy', data['Y'])

        # Save metadata
        np.savez(output_path / f'{prefix}_metadata.npz', **data['metadata'])

        print(f"Saved {prefix} dataset to {output_path}")


def generate_all_datasets(config: dict, output_dir: str, device: str = 'cpu'):
    """
    Generate train, validation, and test datasets.

    Args:
        config: Configuration dict
        output_dir: Output directory
        device: Torch device

    Returns:
        dict: Statistics about generated datasets
    """
    generator = TrajectoryDataGenerator(config, device)

    data_config = config['data']

    # Generate training data
    print("\n" + "="*60)
    print("Generating TRAINING data")
    print("="*60)
    train_data = generator.generate_dataset(
        n_trajectories=data_config['n_trajectories'],
        n_scenarios=data_config['n_scenarios'],
        seed=data_config['random_seed'],
        batch_size=data_config['batch_size_generation'],
    )
    generator.save_dataset(train_data, output_dir, 'train')

    # Generate validation data
    print("\n" + "="*60)
    print("Generating VALIDATION data")
    print("="*60)
    val_data = generator.generate_dataset(
        n_trajectories=data_config['n_val_trajectories'],
        n_scenarios=data_config['n_scenarios'] // 2,
        seed=data_config['random_seed'] + 10000,
        batch_size=data_config['batch_size_generation'],
    )
    generator.save_dataset(val_data, output_dir, 'val')

    # Generate test data
    print("\n" + "="*60)
    print("Generating TEST data")
    print("="*60)
    test_data = generator.generate_dataset(
        n_trajectories=data_config['n_test_trajectories'],
        n_scenarios=data_config['n_scenarios'] // 2,
        seed=data_config['random_seed'] + 20000,
        batch_size=data_config['batch_size_generation'],
    )
    generator.save_dataset(test_data, output_dir, 'test')

    stats = {
        'train': {'n_samples': len(train_data['X']), 'F_mean': train_data['Y'].mean(), 'F_std': train_data['Y'].std()},
        'val': {'n_samples': len(val_data['X']), 'F_mean': val_data['Y'].mean(), 'F_std': val_data['Y'].std()},
        'test': {'n_samples': len(test_data['X']), 'F_mean': test_data['Y'].mean(), 'F_std': test_data['Y'].std()},
    }

    return stats


if __name__ == '__main__':
    from causaliT.surrogate_training.configs.surrogate_config import SURROGATE_CONFIG

    output_dir = 'causaliT/data/surrogate_training'
    stats = generate_all_datasets(SURROGATE_CONFIG, output_dir, device='cpu')

    print("\n" + "="*60)
    print("Dataset Generation Complete")
    print("="*60)
    for split, s in stats.items():
        print(f"  {split}: {s['n_samples']} samples, F = {s['F_mean']:.4f} +/- {s['F_std']:.4f}")
