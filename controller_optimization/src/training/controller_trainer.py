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

        print(f"ControllerTrainer initialized:")
        print(f"  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Lambda BC: {lambda_bc}")
        print(f"  Device: {device}")

    def compute_loss(self, trajectory):
        """
        Calcola loss totale.

        Returns:
            total_loss, reliability_loss, bc_loss, F
        """
        # Reliability loss: (F - F*)^2
        F = self.surrogate.compute_reliability(trajectory)
        F_star_tensor = torch.tensor(self.surrogate.F_star, dtype=torch.float32, device=self.device)
        reliability_loss = (F - F_star_tensor) ** 2

        # Behavior cloning loss: Σ ||a_t - a_t*||^2
        bc_loss = torch.tensor(0.0, device=self.device)

        for process_name in trajectory.keys():
            actual_inputs = trajectory[process_name]['inputs']
            target_inputs_np = self.surrogate.target_trajectory_tensors[process_name]['inputs']
            bc_loss = bc_loss + torch.mean((actual_inputs - target_inputs_np) ** 2)

        # Total loss
        total_loss = reliability_loss + self.lambda_bc * bc_loss

        return total_loss, reliability_loss.item(), bc_loss.item(), F.item()

    def train_epoch(self, n_batches=100, batch_size=32):
        """
        Training per un epoch.

        Per ogni batch:
        1. Forward pass → trajectory
        2. Compute F (surrogate)
        3. Loss = (F - F*)^2 + λ_BC * BC_loss
        4. Backward + optimizer step
        """
        self.process_chain.train()

        epoch_total_loss = 0.0
        epoch_reliability_loss = 0.0
        epoch_bc_loss = 0.0
        epoch_F_values = []

        for batch_idx in range(n_batches):
            # Forward pass through process chain
            trajectory = self.process_chain.forward(batch_size=batch_size)

            # Compute loss
            total_loss, rel_loss, bc_loss, F = self.compute_loss(trajectory)

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Track metrics
            epoch_total_loss += total_loss.item()
            epoch_reliability_loss += rel_loss
            epoch_bc_loss += bc_loss
            epoch_F_values.append(F)

        # Average over batches
        avg_total_loss = epoch_total_loss / n_batches
        avg_reliability_loss = epoch_reliability_loss / n_batches
        avg_bc_loss = epoch_bc_loss / n_batches
        avg_F = np.mean(epoch_F_values)

        return avg_total_loss, avg_reliability_loss, avg_bc_loss, avg_F

    def train(self, epochs=100, n_batches_per_epoch=100, batch_size=32,
              patience=20, save_dir='checkpoints/controller', verbose=True):
        """
        Training loop completo con early stopping.

        Returns:
            history (dict): Training history con tutte le metriche
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"\n{'='*70}")
            print("STARTING CONTROLLER TRAINING")
            print(f"{'='*70}")
            print(f"  Epochs: {epochs}")
            print(f"  Batches per epoch: {n_batches_per_epoch}")
            print(f"  Batch size: {batch_size}")
            print(f"  Patience: {patience}")
            print(f"  Save dir: {save_dir}")
            print(f"  F* (target): {self.surrogate.F_star:.6f}")

        for epoch in range(1, epochs + 1):
            # Train epoch
            avg_total_loss, avg_rel_loss, avg_bc_loss, avg_F = self.train_epoch(
                n_batches=n_batches_per_epoch,
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
                print(f"  F* (target):      {self.surrogate.F_star:.6f}")

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
            print(f"  Target F*: {self.surrogate.F_star:.6f}")

        return self.history

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
