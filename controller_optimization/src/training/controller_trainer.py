"""
Trainer per policy generators.

Loss: L = (F - F*)^2 + λ_BC * Σ ||a_t - a_t*||^2
"""

import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
from datetime import datetime


class ControllerTrainer:
    """
    Trainer per policy generators.

    Args:
        process_chain (ProcessChain): Catena di processi
        surrogate (ProTSurrogate): Surrogate model
        lambda_bc (float): Behavior cloning weight
        learning_rate (float): Learning rate
        device (str): Device
    """

    def __init__(self, process_chain, surrogate, lambda_bc=0.1,
                 learning_rate=0.001, weight_decay=0.01, device='cpu'):

        self.process_chain = process_chain
        self.surrogate = surrogate
        self.lambda_bc = lambda_bc
        self.device = device

        # Optimizer SOLO per policy generators (uncertainty predictors sono frozen)
        trainable_params = [p for p in process_chain.parameters() if p.requires_grad]

        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found in process chain!")

        self.optimizer = optim.Adam(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Tracking
        self.history = {
            'total_loss': [],
            'reliability_loss': [],
            'bc_loss': [],
            'F_values': [],  # Reliability values durante training
        }

        # Best model tracking
        self.best_loss = float('inf')
        self.best_F = -float('inf')
        self.epochs_without_improvement = 0

        # Compute normalization statistics from target trajectories
        self._compute_normalization_stats()

        print(f"ControllerTrainer initialized:")
        print(f"  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Lambda BC: {lambda_bc}")
        print(f"  Device: {device}")

    def _compute_normalization_stats(self):
        """
        Compute normalization statistics (min, max) for each process's inputs
        from the target trajectories to scale BC loss to [0,1] range.
        """
        self.input_stats = {}

        for process_name, data in self.surrogate.target_trajectory_tensors.items():
            inputs = data['inputs']  # Shape: (n_scenarios, seq_len, input_dim)

            # Compute min and max across all scenarios and timesteps
            # Add small epsilon to avoid division by zero
            input_min = inputs.min(dim=0)[0].min(dim=0)[0]  # Shape: (input_dim,)
            input_max = inputs.max(dim=0)[0].max(dim=0)[0]  # Shape: (input_dim,)
            input_range = input_max - input_min + 1e-8  # Avoid division by zero

            self.input_stats[process_name] = {
                'min': input_min.to(self.device),
                'range': input_range.to(self.device)
            }

        print(f"  Input normalization stats computed for {len(self.input_stats)} processes")

    def compute_loss(self, trajectory, scenario_idx):
        """
        Calcola loss totale per uno specifico scenario.

        Args:
            trajectory (dict): Output trajectory from process_chain.forward()
            scenario_idx (int): Index of the scenario being evaluated

        Returns:
            total_loss, reliability_loss, bc_loss, F
        """
        # Reliability loss: (F - F*)^2
        F = self.surrogate.compute_reliability(trajectory)

        # Get F_star for this specific scenario
        F_star_value = self.surrogate.F_star[scenario_idx]
        F_star_tensor = torch.tensor(F_star_value, dtype=torch.float32, device=self.device)
        reliability_loss = (F - F_star_tensor) ** 2

        # Behavior cloning loss: mean( ||a_t - a_t*||^2 ) across all processes
        # Compare to the specific scenario's target inputs
        # Inputs are normalized to [0,1] range to match reliability loss scale
        bc_loss = torch.tensor(0.0, device=self.device)
        n_processes = len(trajectory.keys())

        for process_name in trajectory.keys():
            actual_inputs = trajectory[process_name]['inputs']
            # Select target inputs for this specific scenario
            target_inputs_scenario = self.surrogate.target_trajectory_tensors[process_name]['inputs'][scenario_idx:scenario_idx+1]

            # Normalize inputs to [0,1] using precomputed stats
            stats = self.input_stats[process_name]
            actual_inputs_norm = (actual_inputs - stats['min']) / stats['range']
            target_inputs_norm = (target_inputs_scenario - stats['min']) / stats['range']

            # Compute MSE on normalized inputs
            bc_loss = bc_loss + torch.mean((actual_inputs_norm - target_inputs_norm) ** 2)

        # Average BC loss across processes (not sum!)
        bc_loss = bc_loss / n_processes

        # Total loss (mean over batch)
        total_loss = torch.mean(reliability_loss) + self.lambda_bc * bc_loss

        return total_loss, torch.mean(reliability_loss).item(), bc_loss.item(), torch.mean(F).item()

    def train_epoch(self, batch_size=32):
        """
        Training per un epoch ACROSS ALL SCENARIOS.

        Each epoch cycles through ALL scenarios exactly once in shuffled order.
        This ensures:
        - Equal coverage: every scenario trained once per epoch
        - Diversity: shuffled order prevents overfitting patterns
        - Balanced generalization: no scenario over/under-represented

        Per ogni scenario:
        1. Forward pass → trajectory for that scenario
        2. Compute F (surrogate) using scenario-specific F_star
        3. Loss = (F - F*_scenario)^2 + λ_BC * BC_loss
        4. Backward + optimizer step

        Args:
            batch_size: Number of samples per scenario (default 32)

        Returns:
            Tuple of (avg_total_loss, avg_reliability_loss, avg_bc_loss, avg_F)
        """
        self.process_chain.train()

        epoch_total_loss = 0.0
        epoch_reliability_loss = 0.0
        epoch_bc_loss = 0.0
        epoch_F_values = []

        n_scenarios = len(self.surrogate.F_star)

        # Shuffle scenario order each epoch for diversity
        scenario_order = np.random.permutation(n_scenarios)

        # Cycle through all scenarios exactly once
        for scenario_idx in scenario_order:
            # Forward pass through process chain for this scenario
            trajectory = self.process_chain.forward(
                batch_size=batch_size,
                scenario_idx=scenario_idx
            )

            # Compute loss using scenario-specific F_star
            total_loss, rel_loss, bc_loss, F = self.compute_loss(trajectory, scenario_idx)

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Track metrics
            epoch_total_loss += total_loss.item()
            epoch_reliability_loss += rel_loss
            epoch_bc_loss += bc_loss
            epoch_F_values.append(F)

        # Average over all scenarios
        avg_total_loss = epoch_total_loss / n_scenarios
        avg_reliability_loss = epoch_reliability_loss / n_scenarios
        avg_bc_loss = epoch_bc_loss / n_scenarios
        avg_F = np.mean(epoch_F_values)

        return avg_total_loss, avg_reliability_loss, avg_bc_loss, avg_F

    def train(self, epochs=100, batch_size=32,
              patience=20, save_dir='checkpoints/controller', verbose=True):
        """
        Training loop completo con early stopping.

        Each epoch cycles through all scenarios exactly once in shuffled order.

        Args:
            epochs: Number of training epochs
            batch_size: Number of samples per scenario
            patience: Early stopping patience
            save_dir: Directory to save checkpoints
            verbose: Print training progress

        Returns:
            history (dict): Training history con tutte le metriche
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        n_scenarios = len(self.surrogate.F_star)

        if verbose:
            print(f"\n{'='*70}")
            print("STARTING CONTROLLER TRAINING")
            print(f"{'='*70}")
            print(f"  Epochs: {epochs}")
            print(f"  Scenarios per epoch: {n_scenarios} (all scenarios)")
            print(f"  Batch size per scenario: {batch_size}")
            print(f"  Total batches: {epochs * n_scenarios}")
            print(f"  Patience: {patience}")
            print(f"  Save dir: {save_dir}")
            print(f"  F* (target, mean): {np.mean(self.surrogate.F_star):.6f} ± {np.std(self.surrogate.F_star):.6f}")

        for epoch in range(1, epochs + 1):
            # Train epoch (cycles through all scenarios once)
            avg_total_loss, avg_rel_loss, avg_bc_loss, avg_F = self.train_epoch(
                batch_size=batch_size
            )

            # Track history
            self.history['total_loss'].append(avg_total_loss)
            self.history['reliability_loss'].append(avg_rel_loss)
            self.history['bc_loss'].append(avg_bc_loss)
            self.history['F_values'].append(avg_F)

            # Print progress
            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"\nEpoch {epoch}/{epochs}:")
                print(f"  Total Loss:       {avg_total_loss:.6f}")
                print(f"  Reliability Loss: {avg_rel_loss:.6f}")
                print(f"  BC Loss:          {avg_bc_loss:.6f}")
                print(f"  F (actual):       {avg_F:.6f}")
                print(f"  F* (target, mean):{np.mean(self.surrogate.F_star):.6f}")

            # Check for improvement
            if avg_F > self.best_F:
                self.best_F = avg_F
                self.best_loss = avg_total_loss
                self.epochs_without_improvement = 0

                # Save best model
                self.save_checkpoint(save_dir / 'best_model.pt', epoch)

                if verbose:
                    print(f"  ✓ New best F: {self.best_F:.6f}")

            else:
                self.epochs_without_improvement += 1

                if verbose and self.epochs_without_improvement >= patience:
                    print(f"\n  Early stopping triggered (patience={patience})")
                    break

        # Save final model
        self.save_checkpoint(save_dir / 'final_model.pt', epochs)

        # Save training history
        history_path = save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        if verbose:
            print(f"\n{'='*70}")
            print("TRAINING COMPLETED")
            print(f"{'='*70}")
            print(f"  Best F: {self.best_F:.6f}")
            print(f"  Final F: {self.history['F_values'][-1]:.6f}")
            print(f"  Target F* (mean): {np.mean(self.surrogate.F_star):.6f}")

        return self.history

    def evaluate_all_scenarios(self):
        """
        Evaluate controller performance on ALL scenarios.

        Returns:
            dict: {
                'F_actual_per_scenario': np.array of shape (n_scenarios,),
                'F_actual_mean': float,
                'F_actual_std': float,
                'F_star_mean': float,
                'F_star_std': float,
                'trajectories': list of trajectories for each scenario
            }
        """
        self.process_chain.eval()

        n_scenarios = len(self.surrogate.F_star)
        F_actual_values = []
        trajectories = []

        with torch.no_grad():
            for scenario_idx in range(n_scenarios):
                # Run forward pass for this scenario
                trajectory = self.process_chain.forward(
                    batch_size=1,
                    scenario_idx=scenario_idx
                )

                # Compute reliability
                F_actual = self.surrogate.compute_reliability(trajectory).item()
                F_actual_values.append(F_actual)
                trajectories.append(trajectory)

        F_actual_array = np.array(F_actual_values)

        return {
            'F_actual_per_scenario': F_actual_array,
            'F_actual_mean': np.mean(F_actual_array),
            'F_actual_std': np.std(F_actual_array),
            'F_star_mean': np.mean(self.surrogate.F_star),
            'F_star_std': np.std(self.surrogate.F_star),
            'trajectories': trajectories
        }

    def save_checkpoint(self, path, epoch):
        """Save model checkpoint."""
        path = Path(path)

        # Save each policy generator separately
        for i, policy in enumerate(self.process_chain.policy_generators):
            policy_path = path.parent / f'policy_{i}.pth'
            torch.save(policy.state_dict(), policy_path)

        # Save training state
        state = {
            'epoch': epoch,
            'best_F': self.best_F,
            'best_loss': self.best_loss,
            'history': self.history,
        }

        state_path = path.parent / 'training_state.json'
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

    def compute_final_metrics(self, baseline_trajectory):
        """
        Calcola metriche finali confrontando:
        - a* (target trajectory): F*
        - a' (baseline trajectory): F'
        - a (actual trajectory con controller): F

        Args:
            baseline_trajectory (dict): Baseline trajectory da generate_baseline_trajectory()

        Returns:
            dict: {
                'F_star': float,         # Reliability target
                'F_baseline': float,     # Reliability baseline (no controller)
                'F_actual': float,       # Reliability con controller
                'improvement': float,    # (F_actual - F_baseline) / F_baseline
                'target_gap': float,     # (F_star - F_actual) / F_star
            }
        """
        from controller_optimization.src.utils.model_utils import convert_numpy_to_tensor

        # F* già calcolato
        F_star = self.surrogate.F_star

        # F' = evaluate baseline trajectory
        with torch.no_grad():
            self.process_chain.eval()
            baseline_tensor = convert_numpy_to_tensor(baseline_trajectory, device=self.device)
            F_baseline = self.surrogate.compute_reliability(baseline_tensor).item()

        # F = evaluate actual trajectory (con policy generators)
        with torch.no_grad():
            self.process_chain.eval()
            actual_trajectory = self.process_chain.forward(batch_size=1)
            F_actual = self.surrogate.compute_reliability(actual_trajectory).item()

        # Calcola metriche comparative
        improvement = (F_actual - F_baseline) / abs(F_baseline) if F_baseline != 0 else 0
        target_gap = abs(F_star - F_actual) / F_star if F_star != 0 else 0

        return {
            'F_star': F_star,
            'F_baseline': F_baseline,
            'F_actual': F_actual,
            'improvement': improvement,
            'target_gap': target_gap,
        }
