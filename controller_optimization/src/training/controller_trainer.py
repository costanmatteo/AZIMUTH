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
                 learning_rate=0.001, weight_decay=0.01, device='cpu',
                 train_scenario_indices=None, val_scenario_indices=None):

        self.process_chain = process_chain
        self.surrogate = surrogate
        self.lambda_bc = lambda_bc
        self.device = device

        # Train/validation split
        n_scenarios = len(self.surrogate.F_star)
        if train_scenario_indices is None:
            # Use all scenarios for training (backward compatibility)
            self.train_scenario_indices = list(range(n_scenarios))
            self.val_scenario_indices = []
        else:
            self.train_scenario_indices = train_scenario_indices
            self.val_scenario_indices = val_scenario_indices if val_scenario_indices is not None else []

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
            'train_total_loss': [],
            'train_reliability_loss': [],
            'train_bc_loss': [],
            'train_F_values': [],
            'val_total_loss': [],
            'val_reliability_loss': [],
            'val_bc_loss': [],
            'val_F_values': [],
        }

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_F = -float('inf')
        self.epochs_without_improvement = 0

        # Compute normalization statistics from target trajectories
        self._compute_normalization_stats()

        print(f"ControllerTrainer initialized:")
        print(f"  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Lambda BC: {lambda_bc}")
        print(f"  Device: {device}")
        print(f"  Training scenarios: {len(self.train_scenario_indices)}")
        print(f"  Validation scenarios: {len(self.val_scenario_indices)}")

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
        Training per un epoch usando solo i TRAINING SCENARIOS.

        Each epoch cycles through all TRAINING scenarios exactly once in shuffled order.
        This ensures:
        - Equal coverage: every training scenario trained once per epoch
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

        # Use only training scenarios
        n_train_scenarios = len(self.train_scenario_indices)

        # Shuffle training scenario order each epoch for diversity
        train_scenario_order = np.random.permutation(self.train_scenario_indices)

        # Cycle through all training scenarios exactly once
        for scenario_idx in train_scenario_order:
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

        # Average over all training scenarios
        avg_total_loss = epoch_total_loss / n_train_scenarios
        avg_reliability_loss = epoch_reliability_loss / n_train_scenarios
        avg_bc_loss = epoch_bc_loss / n_train_scenarios
        avg_F = np.mean(epoch_F_values)

        return avg_total_loss, avg_reliability_loss, avg_bc_loss, avg_F

    def validate_epoch(self, batch_size=32):
        """
        Validation per un epoch usando solo i VALIDATION SCENARIOS.

        Evaluates the model on validation scenarios without updating weights.

        Args:
            batch_size: Number of samples per scenario (default 32)

        Returns:
            Tuple of (avg_total_loss, avg_reliability_loss, avg_bc_loss, avg_F)
        """
        self.process_chain.eval()

        epoch_total_loss = 0.0
        epoch_reliability_loss = 0.0
        epoch_bc_loss = 0.0
        epoch_F_values = []

        # Use only validation scenarios
        n_val_scenarios = len(self.val_scenario_indices)

        if n_val_scenarios == 0:
            # No validation set, return zeros
            return 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            # No shuffling for validation - consistent order
            for scenario_idx in self.val_scenario_indices:
                # Forward pass through process chain for this scenario
                trajectory = self.process_chain.forward(
                    batch_size=batch_size,
                    scenario_idx=scenario_idx
                )

                # Compute loss using scenario-specific F_star
                total_loss, rel_loss, bc_loss, F = self.compute_loss(trajectory, scenario_idx)

                # Track metrics
                epoch_total_loss += total_loss.item()
                epoch_reliability_loss += rel_loss
                epoch_bc_loss += bc_loss
                epoch_F_values.append(F)

        # Average over all validation scenarios
        avg_total_loss = epoch_total_loss / n_val_scenarios
        avg_reliability_loss = epoch_reliability_loss / n_val_scenarios
        avg_bc_loss = epoch_bc_loss / n_val_scenarios
        avg_F = np.mean(epoch_F_values)

        return avg_total_loss, avg_reliability_loss, avg_bc_loss, avg_F

    def train(self, epochs=100, batch_size=32,
              patience=20, save_dir='checkpoints/controller', verbose=True):
        """
        Training loop completo con early stopping basato su validation set.

        Each epoch:
        - Cycles through all TRAINING scenarios once in shuffled order
        - Evaluates on VALIDATION scenarios (no weight updates)

        Early stopping is based on validation loss (or validation F if no validation set).

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

        n_train_scenarios = len(self.train_scenario_indices)
        n_val_scenarios = len(self.val_scenario_indices)
        has_validation = n_val_scenarios > 0

        if verbose:
            print(f"\n{'='*70}")
            print("STARTING CONTROLLER TRAINING")
            print(f"{'='*70}")
            print(f"  Epochs: {epochs}")
            print(f"  Training scenarios: {n_train_scenarios}")
            print(f"  Validation scenarios: {n_val_scenarios}")
            print(f"  Batch size per scenario: {batch_size}")
            print(f"  Total training batches per epoch: {n_train_scenarios}")
            print(f"  Patience: {patience}")
            print(f"  Save dir: {save_dir}")
            F_star_train_mean = np.mean(self.surrogate.F_star[self.train_scenario_indices])
            print(f"  F* (training scenarios, mean): {F_star_train_mean:.6f}")
            if has_validation:
                F_star_val_mean = np.mean(self.surrogate.F_star[self.val_scenario_indices])
                print(f"  F* (validation scenarios, mean): {F_star_val_mean:.6f}")

        for epoch in range(1, epochs + 1):
            # Train epoch (cycles through all training scenarios once)
            train_total_loss, train_rel_loss, train_bc_loss, train_F = self.train_epoch(
                batch_size=batch_size
            )

            # Track training history
            self.history['train_total_loss'].append(train_total_loss)
            self.history['train_reliability_loss'].append(train_rel_loss)
            self.history['train_bc_loss'].append(train_bc_loss)
            self.history['train_F_values'].append(train_F)

            # Validation epoch (if validation set exists)
            if has_validation:
                val_total_loss, val_rel_loss, val_bc_loss, val_F = self.validate_epoch(
                    batch_size=batch_size
                )

                # Track validation history
                self.history['val_total_loss'].append(val_total_loss)
                self.history['val_reliability_loss'].append(val_rel_loss)
                self.history['val_bc_loss'].append(val_bc_loss)
                self.history['val_F_values'].append(val_F)

            # Print progress
            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"\nEpoch {epoch}/{epochs}:")
                print(f"  Train - Total Loss: {train_total_loss:.6f} | Rel Loss: {train_rel_loss:.6f} | BC Loss: {train_bc_loss:.6f} | F: {train_F:.6f}")
                if has_validation:
                    print(f"  Val   - Total Loss: {val_total_loss:.6f} | Rel Loss: {val_rel_loss:.6f} | BC Loss: {val_bc_loss:.6f} | F: {val_F:.6f}")

            # Check for improvement (use validation metrics if available, otherwise training metrics)
            if has_validation:
                current_loss = val_total_loss
                current_F = val_F
            else:
                current_loss = train_total_loss
                current_F = train_F

            # Early stopping based on validation F (higher is better)
            if current_F > self.best_val_F:
                self.best_val_F = current_F
                self.best_val_loss = current_loss
                self.epochs_without_improvement = 0

                # Save best model
                self.save_checkpoint(save_dir / 'best_model.pt', epoch)

                if verbose:
                    metric_type = "Val" if has_validation else "Train"
                    print(f"  ✓ New best {metric_type} F: {self.best_val_F:.6f}")

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
            print(f"  Best Val F: {self.best_val_F:.6f}")
            if has_validation:
                print(f"  Final Train F: {self.history['train_F_values'][-1]:.6f}")
                print(f"  Final Val F: {self.history['val_F_values'][-1]:.6f}")
            else:
                print(f"  Final Train F: {self.history['train_F_values'][-1]:.6f}")

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
            'best_val_F': self.best_val_F,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'train_scenario_indices': self.train_scenario_indices,
            'val_scenario_indices': self.val_scenario_indices,
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
