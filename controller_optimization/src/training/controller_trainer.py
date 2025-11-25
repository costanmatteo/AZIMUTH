"""
Trainer per policy generators.

Loss: L = scale * (F - F*)^2 + λ_BC * Σ ||a_t - a_t*||^2

Where:
- scale: Reliability loss scale factor (prevents vanishing gradients)
- F: Actual reliability
- F*: Target reliability
- λ_BC: Behavior cloning weight
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
        weight_decay (float): L2 regularization weight
        reliability_loss_scale (float): Scale factor for reliability loss (default: 100.0)
        device (str): Device
    """

    def __init__(self, process_chain, surrogate, lambda_bc=0.1,
                 learning_rate=0.001, weight_decay=0.01,
                 reliability_loss_scale=100.0, device='cpu',
                 curriculum_config=None):

        self.process_chain = process_chain
        self.surrogate = surrogate
        self.lambda_bc = lambda_bc
        self.reliability_loss_scale = reliability_loss_scale
        self.device = device

        # Curriculum learning configuration
        self.curriculum_config = curriculum_config or {
            'enabled': False,
            'warmup_fraction': 0.1,
            'lambda_bc_start': 10.0,
            'lambda_bc_end': 0.001,
            'reliability_weight_curve': 'exponential'
        }

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
            'lambda_bc': [],  # Track lambda_bc per epoch
            'reliability_weight': [],  # Track reliability_weight per epoch
        }

        # Embedding tracking (if scenario encoder is enabled)
        self.embedding_history = {}  # Dict: {epoch: embeddings_array}

        # Training progression tracking (for visualization)
        self.training_progression = []  # List of snapshots at key epochs

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
        print(f"  Reliability loss scale: {reliability_loss_scale}")
        print(f"  Device: {device}")
        if self.curriculum_config['enabled']:
            print(f"  Curriculum Learning: ENABLED")
            print(f"    Warm-up fraction: {self.curriculum_config['warmup_fraction']}")
            print(f"    Lambda BC: {self.curriculum_config['lambda_bc_start']} → {self.curriculum_config['lambda_bc_end']}")
            print(f"    Reliability weight curve: {self.curriculum_config['reliability_weight_curve']}")

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

    def get_loss_weights(self, epoch, total_epochs):
        """
        Calculate dynamic loss weights for curriculum learning.

        Args:
            epoch (int): Current epoch (1-indexed)
            total_epochs (int): Total number of epochs

        Returns:
            tuple: (lambda_bc, reliability_weight, phase)
                - lambda_bc: Behavior cloning weight
                - reliability_weight: Reliability loss weight
                - phase: 'warmup' or 'curriculum'
        """
        if not self.curriculum_config['enabled']:
            # Curriculum learning disabled: use fixed weights
            return self.lambda_bc, 1.0, 'standard'

        # Calculate warmup epoch threshold
        warmup_epochs = int(total_epochs * self.curriculum_config['warmup_fraction'])

        if epoch <= warmup_epochs:
            # WARM-UP PHASE: BC only, no reliability loss
            lambda_bc = self.curriculum_config['lambda_bc_start']
            reliability_weight = 0.0
            phase = 'warmup'
        else:
            # CURRICULUM LEARNING PHASE: gradual transition
            # Progress within curriculum phase (0.0 at start, 1.0 at end)
            curriculum_start_epoch = warmup_epochs + 1
            curriculum_epochs = total_epochs - warmup_epochs
            progress = (epoch - curriculum_start_epoch) / curriculum_epochs

            # Lambda BC: exponential decay from start to end
            lambda_bc_start = self.curriculum_config['lambda_bc_start']
            lambda_bc_end = self.curriculum_config['lambda_bc_end']
            lambda_bc = lambda_bc_start * np.exp(np.log(lambda_bc_end / lambda_bc_start) * progress)

            # Reliability weight: S-curve based on selected curve type
            curve_type = self.curriculum_config['reliability_weight_curve']

            if curve_type == 'exponential':
                # S-curve: 1 - exp(-5 * progress)
                # Fast growth at start, slows down later
                reliability_weight = 1.0 - np.exp(-5.0 * progress)
            elif curve_type == 'linear':
                # Linear growth from 0 to 1
                reliability_weight = progress
            elif curve_type == 'sigmoid':
                # Sigmoid curve: smooth S-shape
                # Maps progress [0,1] to sigmoid range
                x = 10 * (progress - 0.5)  # Center at 0.5, range ±5
                reliability_weight = 1.0 / (1.0 + np.exp(-x))
            else:
                raise ValueError(f"Unknown reliability_weight_curve: {curve_type}")

            phase = 'curriculum'

        return lambda_bc, reliability_weight, phase

    def _extract_all_embeddings(self):
        """
        Extract embeddings for all scenarios using the scenario encoder.

        Returns:
            tuple: (embeddings, structural_params, scenario_indices)
                - embeddings: np.array of shape (n_scenarios, embedding_dim)
                - structural_params: np.array of shape (n_scenarios, n_structural_params)
                - scenario_indices: np.array of scenario indices
        """
        if not hasattr(self.process_chain, 'scenario_encoder') or self.process_chain.scenario_encoder is None:
            return None, None, None

        n_scenarios = len(self.surrogate.F_star)
        embedding_dim = self.process_chain.scenario_embedding_dim

        embeddings_list = []
        structural_params_list = []

        with torch.no_grad():
            for scenario_idx in range(n_scenarios):
                # Extract structural params
                structural_params = self.process_chain._extract_structural_params(scenario_idx)
                structural_params_list.append(structural_params.cpu().numpy())

                # Encode to embedding
                structural_params_batch = structural_params.unsqueeze(0)  # Add batch dim
                embedding = self.process_chain.scenario_encoder(structural_params_batch)
                embeddings_list.append(embedding.squeeze(0).cpu().numpy())

        embeddings = np.array(embeddings_list)  # Shape: (n_scenarios, embedding_dim)
        structural_params = np.array(structural_params_list)  # Shape: (n_scenarios, n_params)
        scenario_indices = np.arange(n_scenarios)

        return embeddings, structural_params, scenario_indices

    def _save_embedding_snapshot(self, epoch):
        """
        Save embedding snapshot for a specific epoch.

        Args:
            epoch (int): Current epoch number
        """
        if not hasattr(self.process_chain, 'scenario_encoder') or self.process_chain.scenario_encoder is None:
            return

        embeddings, _, _ = self._extract_all_embeddings()
        if embeddings is not None:
            self.embedding_history[epoch] = embeddings

    def _save_training_progression_snapshot(self, epoch, lambda_bc, reliability_weight, phase):
        """
        Save snapshot of generated inputs/outputs at key epochs for progression visualization.

        Args:
            epoch (int): Current epoch number
            lambda_bc (float): Current lambda_bc value
            reliability_weight (float): Current reliability_weight value
            phase (str): Current training phase
        """
        self.process_chain.eval()

        # Use first scenario as representative
        representative_scenario_idx = 0

        with torch.no_grad():
            # Generate trajectory for representative scenario
            trajectory = self.process_chain.forward(
                batch_size=1,
                scenario_idx=representative_scenario_idx
            )

            # Convert to numpy for storage
            from controller_optimization.src.utils.metrics import convert_trajectory_to_numpy
            trajectory_np = convert_trajectory_to_numpy(trajectory)

            # Also get target trajectory for this scenario
            target_trajectory_np = {}
            for process_name, data in self.surrogate.target_trajectory_tensors.items():
                target_trajectory_np[process_name] = {
                    'inputs': data['inputs'][representative_scenario_idx:representative_scenario_idx+1].cpu().numpy(),
                    'outputs': data['outputs'][representative_scenario_idx:representative_scenario_idx+1].cpu().numpy()
                }

            # Compute F for this snapshot
            F_actual = self.surrogate.compute_reliability(trajectory).item()
            F_star = self.surrogate.F_star[representative_scenario_idx]

            # Save snapshot
            snapshot = {
                'epoch': epoch,
                'phase': phase,
                'lambda_bc': lambda_bc,
                'reliability_weight': reliability_weight,
                'trajectory': trajectory_np,
                'target_trajectory': target_trajectory_np,
                'F_actual': F_actual,
                'F_star': F_star,
                'scenario_idx': representative_scenario_idx
            }

            self.training_progression.append(snapshot)

        self.process_chain.train()

    def compute_loss(self, trajectory, scenario_idx, reliability_weight=1.0, lambda_bc=None):
        """
        Calcola loss totale per uno specifico scenario.

        Args:
            trajectory (dict): Output trajectory from process_chain.forward()
            scenario_idx (int): Index of the scenario being evaluated
            reliability_weight (float): Weight for reliability loss (0.0 = ignore, 1.0 = full)
            lambda_bc (float): Behavior cloning weight (if None, uses self.lambda_bc)

        Returns:
            total_loss, reliability_loss, bc_loss, F
        """
        # Use provided lambda_bc or fall back to instance value
        if lambda_bc is None:
            lambda_bc = self.lambda_bc

        # Reliability loss: scale * (F - F*)^2
        # Scale prevents vanishing gradients when delta F is small (~0.1)
        F = self.surrogate.compute_reliability(trajectory)

        # Get F_star for this specific scenario
        F_star_value = self.surrogate.F_star[scenario_idx]
        F_star_tensor = torch.tensor(F_star_value, dtype=torch.float32, device=self.device)
        reliability_loss = self.reliability_loss_scale * (F - F_star_tensor) ** 2

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

        # Total loss with dynamic weights
        # reliability_weight controls if reliability loss is active (0.0 during warm-up)
        total_loss = reliability_weight * torch.mean(reliability_loss) + lambda_bc * bc_loss

        return total_loss, torch.mean(reliability_loss).item(), bc_loss.item(), torch.mean(F).item()

    def train_epoch(self, batch_size=32, reliability_weight=1.0, lambda_bc=None):
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
        3. Loss = reliability_weight * (F - F*_scenario)^2 + λ_BC * BC_loss
        4. Backward + optimizer step

        Args:
            batch_size: Number of samples per scenario (default 32)
            reliability_weight: Weight for reliability loss (for curriculum learning)
            lambda_bc: BC weight (if None, uses self.lambda_bc)

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

            # Compute loss using scenario-specific F_star and dynamic weights
            total_loss, rel_loss, bc_loss, F = self.compute_loss(
                trajectory, scenario_idx,
                reliability_weight=reliability_weight,
                lambda_bc=lambda_bc
            )

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()

            # DEBUG: Check gradient flow on first scenario of first few epochs
            if hasattr(self, '_debug_gradients') and self._debug_gradients and scenario_idx == scenario_order[0]:
                print(f"\n{'='*60}")
                print("GRADIENT DEBUG - After backward pass")
                print(f"{'='*60}")

                # Check gradients on policy generators
                for i, policy in enumerate(self.process_chain.policy_generators):
                    print(f"\nPolicy Generator {i}:")
                    total_grad_norm = 0.0
                    has_any_grad = False
                    for name, param in policy.named_parameters():
                        if param.grad is not None:
                            has_any_grad = True
                            grad_norm = param.grad.norm().item()
                            total_grad_norm += grad_norm ** 2
                            if 'weight' in name or 'bias' in name:
                                print(f"  {name}: grad_norm={grad_norm:.8f}, param_norm={param.norm().item():.4f}")
                        else:
                            print(f"  {name}: NO GRADIENT!")

                    if has_any_grad:
                        print(f"  Total grad norm: {total_grad_norm**0.5:.8f}")
                    else:
                        print(f"  WARNING: No gradients at all!")

                # Check if trajectory inputs have gradients
                print(f"\nTrajectory Gradient Check:")
                for proc_name, data in trajectory.items():
                    inputs = data['inputs']
                    outputs_mean = data['outputs_mean']
                    print(f"  {proc_name}:")
                    print(f"    inputs.requires_grad: {inputs.requires_grad}, inputs.grad_fn: {inputs.grad_fn}")
                    print(f"    outputs_mean.requires_grad: {outputs_mean.requires_grad}, outputs_mean.grad_fn: {outputs_mean.grad_fn}")

                print(f"{'='*60}\n")

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

        # Calculate warmup epochs for curriculum learning
        warmup_epochs = 0
        if self.curriculum_config['enabled']:
            warmup_epochs = int(epochs * self.curriculum_config['warmup_fraction'])

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

            if self.curriculum_config['enabled']:
                print(f"\n  CURRICULUM LEARNING STRATEGY:")
                print(f"    Phase 1 - Warm-up: Epochs 1-{warmup_epochs} (BC only)")
                print(f"    Phase 2 - Curriculum: Epochs {warmup_epochs + 1}-{epochs} (gradual reliability)")
                print(f"    Lambda BC schedule: {self.curriculum_config['lambda_bc_start']:.3f} → {self.curriculum_config['lambda_bc_end']:.6f}")
                print(f"    Reliability weight curve: {self.curriculum_config['reliability_weight_curve']}")

        # Save embedding snapshot at epoch 1 (initial state)
        if hasattr(self.process_chain, 'scenario_encoder') and self.process_chain.scenario_encoder is not None:
            self._save_embedding_snapshot(epoch=1)
            if verbose:
                print(f"  Saved initial embedding snapshot")

        for epoch in range(1, epochs + 1):
            # Disable debug after first epoch to avoid flooding output
            if epoch == 2:
                from controller_optimization.src.utils.process_chain import ProcessChain
                ProcessChain.enable_debug(False)
                self._debug_gradients = False  # Also disable gradient debug

            # Get dynamic loss weights for curriculum learning
            lambda_bc, reliability_weight, phase = self.get_loss_weights(epoch, epochs)

            # Train epoch with dynamic weights
            avg_total_loss, avg_rel_loss, avg_bc_loss, avg_F = self.train_epoch(
                batch_size=batch_size,
                reliability_weight=reliability_weight,
                lambda_bc=lambda_bc
            )

            # Track history
            self.history['total_loss'].append(avg_total_loss)
            self.history['reliability_loss'].append(avg_rel_loss)
            self.history['bc_loss'].append(avg_bc_loss)
            self.history['F_values'].append(avg_F)
            self.history['lambda_bc'].append(lambda_bc)
            self.history['reliability_weight'].append(reliability_weight)

            # Save training progression snapshots every epoch
            self._save_training_progression_snapshot(epoch, lambda_bc, reliability_weight, phase)

            # Save embedding snapshot periodically
            if hasattr(self.process_chain, 'scenario_encoder') and self.process_chain.scenario_encoder is not None:
                if epoch % 20 == 0 or epoch == 1:  # Save every 20 epochs and at epoch 1
                    self._save_embedding_snapshot(epoch)

            # Print progress with curriculum learning info
            if verbose and (epoch % 10 == 0 or epoch == 1):
                phase_label = "[WARM-UP]" if phase == 'warmup' else "[CURRICULUM]" if phase == 'curriculum' else ""
                print(f"\nEpoch {epoch}/{epochs} {phase_label}")
                print(f"  λ_BC: {lambda_bc:.6f}, Rel Weight: {reliability_weight:.3f}")
                print(f"  Total Loss:       {avg_total_loss:.6f}")
                print(f"  Reliability Loss: {avg_rel_loss:.6f} {'(ignored)' if reliability_weight == 0 else ''}")
                print(f"  BC Loss:          {avg_bc_loss:.6f}")
                print(f"  F (actual):       {avg_F:.6f}")
                print(f"  F* (target, mean):{np.mean(self.surrogate.F_star):.6f}")

            # Print warm-up completion message
            if verbose and self.curriculum_config['enabled'] and epoch == warmup_epochs:
                print(f"\n{'='*70}")
                print("WARM-UP COMPLETED! Starting curriculum learning phase...")
                print(f"{'='*70}")

            # Check for improvement (based on LOSS, not F value)
            # We want F to become more similar to F*, i.e., loss to decrease
            # IMPORTANT: During warm-up, reliability loss is random, so we skip early stopping
            if avg_total_loss < self.best_loss:
                self.best_loss = avg_total_loss
                self.best_F = avg_F  # Track best F for information
                self.epochs_without_improvement = 0

                # Save best model
                self.save_checkpoint(save_dir / 'best_model.pt', epoch)

                if verbose:
                    print(f"  ✓ New best loss: {self.best_loss:.6f} (F: {self.best_F:.6f})")

            else:
                self.epochs_without_improvement += 1

                # Early stopping: ONLY after warm-up period
                # During warm-up, reliability is random and shouldn't trigger early stopping
                if epoch > warmup_epochs and self.epochs_without_improvement >= patience:
                    if verbose:
                        print(f"\n  Early stopping triggered (patience={patience})")
                        print(f"  Loss has not improved for {patience} epochs")
                    break

        # Save final model
        self.save_checkpoint(save_dir / 'final_model.pt', epochs)

        # Save final embedding snapshot
        if hasattr(self.process_chain, 'scenario_encoder') and self.process_chain.scenario_encoder is not None:
            final_epoch = epoch  # Use actual final epoch (may have stopped early)
            self._save_embedding_snapshot(final_epoch)

            # Save embedding data
            embeddings, structural_params, scenario_indices = self._extract_all_embeddings()
            if embeddings is not None:
                embedding_data = {
                    'embeddings': embeddings.tolist(),
                    'structural_params': structural_params.tolist(),
                    'scenario_indices': scenario_indices.tolist(),
                    'embedding_history_epochs': list(self.embedding_history.keys()),
                }
                embedding_path = save_dir / 'embeddings.json'
                with open(embedding_path, 'w') as f:
                    json.dump(embedding_data, f, indent=2)

                # Save embedding history as numpy
                embedding_history_path = save_dir / 'embedding_history.npz'
                np.savez(embedding_history_path, **{f'epoch_{k}': v for k, v in self.embedding_history.items()})

                if verbose:
                    print(f"  Saved embedding data to {save_dir}")

        # Save training history
        history_path = save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        # Save training progression snapshots
        if len(self.training_progression) > 0:
            progression_path = save_dir / 'training_progression.npz'
            progression_data = {}

            for i, snapshot in enumerate(self.training_progression):
                prefix = f'snapshot_{i}_epoch_{snapshot["epoch"]}'
                progression_data[f'{prefix}_epoch'] = snapshot['epoch']
                progression_data[f'{prefix}_phase'] = snapshot['phase']
                progression_data[f'{prefix}_lambda_bc'] = snapshot['lambda_bc']
                progression_data[f'{prefix}_reliability_weight'] = snapshot['reliability_weight']
                progression_data[f'{prefix}_F_actual'] = snapshot['F_actual']
                progression_data[f'{prefix}_F_star'] = snapshot['F_star']

                # Save inputs and outputs for each process
                for process_name in snapshot['trajectory'].keys():
                    progression_data[f'{prefix}_{process_name}_inputs'] = snapshot['trajectory'][process_name]['inputs']
                    progression_data[f'{prefix}_{process_name}_outputs'] = snapshot['trajectory'][process_name]['outputs_mean']
                    progression_data[f'{prefix}_{process_name}_target_inputs'] = snapshot['target_trajectory'][process_name]['inputs']
                    progression_data[f'{prefix}_{process_name}_target_outputs'] = snapshot['target_trajectory'][process_name]['outputs']

            np.savez(progression_path, **progression_data)

            if verbose:
                print(f"  Saved {len(self.training_progression)} training progression snapshots to {progression_path}")

        if verbose:
            print(f"\n{'='*70}")
            print("TRAINING COMPLETED")
            print(f"{'='*70}")
            print(f"  Best F: {self.best_F:.6f}")
            print(f"  Final F: {self.history['F_values'][-1]:.6f}")
            print(f"  Target F* (mean): {np.mean(self.surrogate.F_star):.6f}")

        return self.history

    def evaluate_all_scenarios(self, batch_size=32, per_sample=False):
        """
        Evaluate controller performance on ALL scenarios.

        Args:
            batch_size (int): Number of samples per scenario
            per_sample (bool): If True, compute F for each sample individually.
                              If False, compute F for aggregated batch (old behavior).

        Returns:
            dict: {
                'F_actual_per_sample': np.array of shape (n_scenarios * batch_size,) if per_sample=True
                                       or (n_scenarios,) if per_sample=False,
                'F_actual_mean': float,
                'F_actual_std': float,
                'F_star_mean': float,
                'F_star_std': float,
                'trajectories': list of trajectories (representative, one per scenario)
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
                    batch_size=batch_size,
                    scenario_idx=scenario_idx
                )

                if per_sample:
                    # Compute reliability for each sample individually
                    for sample_idx in range(batch_size):
                        # Extract single sample from trajectory
                        sample_trajectory = {}
                        for process_name, data in trajectory.items():
                            sample_trajectory[process_name] = {
                                'inputs': data['inputs'][sample_idx:sample_idx+1],
                                'outputs_mean': data['outputs_mean'][sample_idx:sample_idx+1],
                                'outputs_var': data['outputs_var'][sample_idx:sample_idx+1]
                            }

                        # Compute reliability for this single sample
                        F_actual = self.surrogate.compute_reliability(sample_trajectory).item()
                        F_actual_values.append(F_actual)
                else:
                    # Compute reliability for aggregated batch (take mean across batch)
                    F_actual_batch = self.surrogate.compute_reliability(trajectory)
                    F_actual = F_actual_batch.mean().item()
                    F_actual_values.append(F_actual)

                # Save trajectory for each scenario (needed for representative trajectory selection)
                trajectories.append(trajectory)

        F_actual_array = np.array(F_actual_values)

        return {
            'F_actual_per_sample': F_actual_array,
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

    def load_checkpoint(self, checkpoint_dir):
        """
        Load best model checkpoint.

        Args:
            checkpoint_dir (Path or str): Directory containing policy checkpoints
        """
        checkpoint_dir = Path(checkpoint_dir)

        # Load each policy generator
        for i, policy in enumerate(self.process_chain.policy_generators):
            policy_path = checkpoint_dir / f'policy_{i}.pth'
            if not policy_path.exists():
                raise FileNotFoundError(f"Policy checkpoint not found: {policy_path}")

            policy.load_state_dict(torch.load(policy_path, map_location=self.device))

        print(f"  ✓ Loaded best model from {checkpoint_dir}")

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
