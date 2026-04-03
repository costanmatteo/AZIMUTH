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
from torch.optim import lr_scheduler as torch_lr_scheduler
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
                 curriculum_config=None, lr_scheduler_config=None):

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

        # Learning rate scheduler configuration
        self.lr_scheduler_config = lr_scheduler_config

        # Optimizer SOLO per policy generators (uncertainty predictors sono frozen)
        trainable_params = [p for p in process_chain.parameters() if p.requires_grad]

        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found in process chain!")

        self.optimizer = optim.Adam(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Create learning rate scheduler if configured
        self.scheduler = self._create_scheduler()

        # Tracking
        self.history = {
            'total_loss': [],
            'reliability_loss': [],
            'bc_loss': [],
            'F_values': [],  # Reliability values durante training
            'F_formula_values': [],  # F from mathematical formula (when using CasualiT surrogate)
            'lambda_bc': [],  # Track lambda_bc per epoch
            'reliability_weight': [],  # Track reliability_weight per epoch
            'learning_rate': [],  # Track learning rate per epoch
            # Cross-scenario validation metrics (on test scenarios with different conditions)
            'val_total_loss': [],
            'val_reliability_loss': [],
            'val_bc_loss': [],
            'val_F_values': [],
            # Within-scenario validation metrics (held-out samples from training scenarios)
            'val_within_total_loss': [],
            'val_within_reliability_loss': [],
            'val_within_bc_loss': [],
            'val_within_F_values': [],
            'gap_closure': [],
        }

        # Formula surrogate for comparison (set via set_formula_surrogate when using CasualiT)
        self.formula_surrogate = None

        # Cross-scenario validation data (set externally via set_validation_data)
        self.validation_surrogate = None
        self.validation_process_chain = None

        # Within-scenario validation config
        self.within_scenario_validation_enabled = False
        self.within_scenario_split = 0.0  # Fraction of samples for validation

        # Per-process quality score history for correlation estimation
        # Dict: {process_name: [Q_values per epoch/scenario]}
        self.Q_history = {}

        # Baseline reliabilities per scenario (for gap closure computation during training)
        self.F_baseline_per_scenario = None

        # Embedding tracking (if scenario encoder is enabled)
        self.embedding_history = {}  # Dict: {epoch: embeddings_array}

        # Training progression tracking (for visualization)
        self.training_progression = []  # List of snapshots at key epochs

        # Best model tracking
        self.best_loss = float('inf')
        self.best_F = -float('inf')
        self.epochs_without_improvement = 0

        # Build mapping: process_name → controllable input indices
        # BC loss should only compare controllable inputs (non-controllable can't be changed)
        self.controllable_indices = {}
        for i, process_name in enumerate(self.process_chain.process_names):
            info = self.process_chain.controllable_info_per_process[i]
            self.controllable_indices[process_name] = info['controllable_indices']

        # Compute normalization statistics from target trajectories
        self._compute_normalization_stats()

        # Debug flags — settati dall'esterno in train_controller.py
        # _debug_gradients, _debug_bc_loss, _debug_F_graph

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

    def set_formula_surrogate(self, formula_surrogate):
        """
        Set a ProTSurrogate (mathematical formula) for comparison during training.

        When the main surrogate is CasualiT, this allows tracking the formula-based F
        alongside the surrogate F for comparison purposes.

        Args:
            formula_surrogate (ProTSurrogate): Formula-based surrogate for F computation
        """
        self.formula_surrogate = formula_surrogate
        print(f"  Formula surrogate set for comparison (F* formula = {formula_surrogate.F_star:.6f})")

    def get_current_lr(self):
        """Get current learning rate from optimizer."""
        return self.optimizer.param_groups[0]['lr']

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
            # decay_speed controls how fast λ_BC decreases:
            #   1.0 = normal, 2.0 = twice as fast, 3.0 = three times as fast
            lambda_bc_start = self.curriculum_config['lambda_bc_start']
            lambda_bc_end = self.curriculum_config['lambda_bc_end']
            decay_speed = self.curriculum_config.get('decay_speed', 1.0)
            lambda_bc = lambda_bc_start * np.exp(decay_speed * np.log(lambda_bc_end / lambda_bc_start) * progress)
            # Clamp to end value (in case decay_speed > 1 causes undershoot)
            lambda_bc = max(lambda_bc, lambda_bc_end)

            # Reliability weight: S-curve based on selected curve type
            # reliability_speed controls how fast reliability weight increases:
            #   1.0 = normal, 2.0 = twice as fast (reaches 1.0 at half training)
            curve_type = self.curriculum_config['reliability_weight_curve']
            reliability_speed = self.curriculum_config.get('reliability_speed', 1.0)
            accelerated_progress = min(1.0, reliability_speed * progress)

            if curve_type == 'exponential':
                # S-curve: 1 - exp(-5 * progress)
                # Fast growth at start, slows down later
                reliability_weight = 1.0 - np.exp(-5.0 * accelerated_progress)
            elif curve_type == 'linear':
                # Linear growth from 0 to 1
                reliability_weight = accelerated_progress
            elif curve_type == 'sigmoid':
                # Sigmoid curve: smooth S-shape
                # Maps progress [0,1] to sigmoid range
                x = 10 * (accelerated_progress - 0.5)  # Center at 0.5, range ±5
                reliability_weight = 1.0 / (1.0 + np.exp(-x))
            else:
                raise ValueError(f"Unknown reliability_weight_curve: {curve_type}")

            phase = 'curriculum'

        return lambda_bc, reliability_weight, phase

    def train_epoch(self, batch_size=32, reliability_weight=1.0, lambda_bc=None):
        """
        Training per un epoch ACROSS ALL SCENARIOS with gradient accumulation.

        Each epoch cycles through ALL scenarios exactly once in shuffled order.
        Gradients are accumulated across all scenarios and a single optimizer
        step is performed at the end of the epoch. This ensures the NN learns
        to satisfy all scenarios simultaneously, avoiding catastrophic forgetting
        where the last scenario's update partially undoes earlier ones.

        The batch_size parameter represents the TOTAL number of samples across
        all scenarios. Samples are split equally: samples_per_scenario = batch_size // n_scenarios.
        This keeps total computation per epoch constant regardless of n_scenarios.

        Pattern:
            optimizer.zero_grad()                        # once per epoch
            for scenario_idx in scenario_order:
                trajectory = forward(samples_per_scenario, scenario_idx)
                loss = compute_loss(trajectory, scenario_idx)
                (loss / n_scenarios).backward()           # accumulate gradients
            optimizer.step()                              # single step per epoch

        Args:
            batch_size: Total number of samples across all scenarios (default 32).
                        Divided equally among scenarios: samples_per_scenario = batch_size // n_scenarios.
            reliability_weight: Weight for reliability loss (for curriculum learning)
            lambda_bc: BC weight (if None, uses self.lambda_bc)

        Returns:
            Tuple of (avg_total_loss, avg_reliability_loss, avg_bc_loss, avg_F)
            Also stores within-scenario validation metrics in self._within_val_metrics if enabled.
        """
        self.process_chain.train()

        epoch_total_loss = 0.0
        epoch_reliability_loss = 0.0
        epoch_bc_loss = 0.0
        epoch_F_values = []
        epoch_F_formula_values = []  # F from mathematical formula (for comparison)
        epoch_F_per_scenario = {}  # {scenario_idx: F_value} for gap closure

        # Within-scenario validation tracking
        epoch_val_total_loss = 0.0
        epoch_val_reliability_loss = 0.0
        epoch_val_bc_loss = 0.0
        epoch_val_F_values = []

        n_scenarios = self.surrogate.n_scenarios

        # Split total samples equally across scenarios
        samples_per_scenario = max(1, batch_size // n_scenarios)

        # Calculate train/val batch sizes if within-scenario validation is enabled
        if self.within_scenario_validation_enabled and self.within_scenario_split > 0:
            # Generate more samples to split into train and val
            val_samples = max(1, int(samples_per_scenario * self.within_scenario_split / (1 - self.within_scenario_split)))
            total_samples_per_scenario = samples_per_scenario + val_samples
            train_samples_per_scenario = samples_per_scenario
        else:
            total_samples_per_scenario = samples_per_scenario
            train_samples_per_scenario = samples_per_scenario
            val_samples = 0

        # Shuffle scenario order each epoch for diversity
        scenario_order = np.random.permutation(n_scenarios)

        # Zero gradients once per epoch (gradient accumulation across scenarios)
        self.optimizer.zero_grad()

        # Cycle through all scenarios exactly once
        for scenario_idx in scenario_order:
            # Forward pass through process chain for this scenario
            # Generate enough samples for both train and val
            trajectory = self.process_chain.forward(
                batch_size=total_samples_per_scenario,
                scenario_idx=scenario_idx
            )

            # Split into train and val trajectories if within-scenario validation is enabled
            if self.within_scenario_validation_enabled and val_samples > 0:
                train_trajectory, val_trajectory = self._split_trajectory(
                    trajectory, train_samples_per_scenario, val_samples
                )
            else:
                train_trajectory = trajectory
                val_trajectory = None

            # Compute loss using scenario-specific F_star and dynamic weights
            # Use ONLY train_trajectory for gradient computation
            total_loss, rel_loss, bc_loss, F, quality_scores = self.compute_loss(
                train_trajectory, scenario_idx,
                reliability_weight=reliability_weight,
                lambda_bc=lambda_bc,
                return_quality_scores=True
            )

            # Compute within-scenario validation loss (no gradient)
            if val_trajectory is not None:
                with torch.no_grad():
                    val_total, val_rel, val_bc, val_F = self.compute_loss(
                        val_trajectory, scenario_idx,
                        reliability_weight=reliability_weight,
                        lambda_bc=lambda_bc,
                        return_quality_scores=False
                    )
                    epoch_val_total_loss += val_total.item() if torch.is_tensor(val_total) else val_total
                    epoch_val_reliability_loss += val_rel
                    epoch_val_bc_loss += val_bc
                    epoch_val_F_values.append(val_F)

            # Collect quality scores for correlation estimation
            for proc_name, Q in quality_scores.items():
                if proc_name not in self.Q_history:
                    self.Q_history[proc_name] = []
                # Store mean Q value (detached from computation graph)
                Q_mean = Q.detach().mean().item()
                self.Q_history[proc_name].append(Q_mean)

            # Backward pass: accumulate gradients (scaled by 1/n_scenarios)
            (total_loss / n_scenarios).backward()

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

                # Check if trajectory inputs/outputs are in the computation graph
                print(f"\nTrajectory Gradient Check:")
                for proc_name, data in trajectory.items():
                    inputs = data['inputs']
                    outputs_mean = data['outputs_mean']
                    outputs_sampled = data.get('outputs_sampled')
                    print(f"  {proc_name}:")
                    print(f"    inputs:          requires_grad={inputs.requires_grad}, grad_fn={inputs.grad_fn}")
                    print(f"    outputs_mean:    requires_grad={outputs_mean.requires_grad}, grad_fn={outputs_mean.grad_fn}")
                    if outputs_sampled is not None:
                        print(f"    outputs_sampled: requires_grad={outputs_sampled.requires_grad}, grad_fn={outputs_sampled.grad_fn}")

                # Check F tensor properties
                print(f"\nF Tensor Check:")
                F_tensor = F if torch.is_tensor(F) else torch.tensor(F)
                print(f"  F.requires_grad={F_tensor.requires_grad}, F.grad_fn={F_tensor.grad_fn}")
                print(f"  F value={F_tensor.mean().item():.6f}, F*={self.surrogate.F_star:.6f}")
                if F_tensor.requires_grad:
                    print(f"  -> Reliability loss WILL produce gradients for controller")
                else:
                    print(f"  -> WARNING: F is detached! Reliability loss has NO gradient effect!")

                print(f"{'='*60}\n")

            # Track metrics
            epoch_total_loss += total_loss.item()
            epoch_reliability_loss += rel_loss
            epoch_bc_loss += bc_loss
            epoch_F_values.append(F)
            epoch_F_per_scenario[int(scenario_idx)] = F

            # Compute F from mathematical formula for comparison (if formula surrogate is set)
            if self.formula_surrogate is not None:
                with torch.no_grad():
                    F_formula = self.formula_surrogate.compute_reliability(train_trajectory)
                    epoch_F_formula_values.append(torch.mean(F_formula).item())

        # Single optimizer step after accumulating gradients from all scenarios
        self.optimizer.step()

        # Average over all scenarios
        avg_total_loss = epoch_total_loss / n_scenarios
        avg_reliability_loss = epoch_reliability_loss / n_scenarios
        avg_bc_loss = epoch_bc_loss / n_scenarios
        avg_F = np.mean(epoch_F_values)

        # Average F_formula (if computed)
        self._epoch_avg_F_formula = np.mean(epoch_F_formula_values) if epoch_F_formula_values else None

        # Compute gap closure if baseline reliabilities are available
        if self.F_baseline_per_scenario is not None:
            gc_values = []
            for s_idx, F_i in epoch_F_per_scenario.items():
                denom = self.surrogate.F_star - self.F_baseline_per_scenario[s_idx]
                if abs(denom) > 1e-6:
                    gc_values.append((F_i - self.F_baseline_per_scenario[s_idx]) / denom)
            self._epoch_gap_closure = float(np.mean(gc_values)) if gc_values else 0.0
        else:
            self._epoch_gap_closure = None

        # Store within-scenario validation metrics (if enabled)
        if self.within_scenario_validation_enabled and len(epoch_val_F_values) > 0:
            self._within_val_metrics = {
                'total_loss': epoch_val_total_loss / n_scenarios,
                'reliability_loss': epoch_val_reliability_loss / n_scenarios,
                'bc_loss': epoch_val_bc_loss / n_scenarios,
                'F': np.mean(epoch_val_F_values),
            }
        else:
            self._within_val_metrics = None

        return avg_total_loss, avg_reliability_loss, avg_bc_loss, avg_F

    def compute_validation_loss(self, batch_size=32, reliability_weight=1.0, lambda_bc=None):
        """
        Compute validation loss on test scenarios (no gradient computation).

        Uses the same loss function as training but on held-out test scenarios.

        Args:
            batch_size (int): Number of samples per scenario
            reliability_weight (float): Weight for reliability loss
            lambda_bc (float): Behavior cloning weight

        Returns:
            tuple: (avg_total_loss, avg_reliability_loss, avg_bc_loss, avg_F)
            Returns (None, None, None, None) if validation data not set
        """
        if self.validation_surrogate is None or self.validation_process_chain is None:
            return None, None, None, None

        # Copy current policy weights to validation process chain
        for i, policy in enumerate(self.process_chain.policy_generators):
            self.validation_process_chain.policy_generators[i].load_state_dict(
                policy.state_dict()
            )

        self.validation_process_chain.eval()

        if lambda_bc is None:
            lambda_bc = self.lambda_bc

        n_val_scenarios = self.validation_surrogate.n_scenarios

        val_total_loss = 0.0
        val_reliability_loss = 0.0
        val_bc_loss = 0.0
        val_F_values = []

        with torch.no_grad():
            for scenario_idx in range(n_val_scenarios):
                # Forward pass through validation process chain
                trajectory = self.validation_process_chain.forward(
                    batch_size=batch_size,
                    scenario_idx=scenario_idx
                )

                # Compute reliability
                F = self.validation_surrogate.compute_reliability(trajectory)

                F_star_tensor = torch.tensor(self.validation_surrogate.F_star, dtype=torch.float32, device=self.device)

                # Reliability loss
                rel_loss = self.reliability_loss_scale * (F - F_star_tensor) ** 2

                # BC loss (compare to validation target inputs)
                bc_loss = torch.tensor(0.0, device=self.device)
                n_processes = len(trajectory.keys())

                for process_name in trajectory.keys():
                    ctrl_idx = self.controllable_indices[process_name]
                    if len(ctrl_idx) == 0:
                        continue

                    actual_inputs = trajectory[process_name]['inputs'][..., ctrl_idx]
                    val_target_all = self.validation_surrogate.target_trajectory_tensors[process_name]['inputs']
                    val_target_idx = min(scenario_idx, val_target_all.shape[0] - 1)
                    target_inputs = val_target_all[val_target_idx:val_target_idx+1][..., ctrl_idx]

                    # Use same normalization stats as training (with fallback scale for constant dims)
                    stats = self.input_stats[process_name]
                    actual_norm = (actual_inputs - stats['min']) / stats['range']
                    target_norm = (target_inputs - stats['min']) / stats['range']

                    bc_loss = bc_loss + torch.mean((actual_norm - target_norm) ** 2)

                bc_loss = bc_loss / n_processes

                # Total loss
                total_loss = reliability_weight * torch.mean(rel_loss) + lambda_bc * bc_loss

                val_total_loss += total_loss.item()
                val_reliability_loss += torch.mean(rel_loss).item()
                val_bc_loss += bc_loss.item()
                val_F_values.append(torch.mean(F).item())

        # Average over validation scenarios
        avg_val_total_loss = val_total_loss / n_val_scenarios
        avg_val_reliability_loss = val_reliability_loss / n_val_scenarios
        avg_val_bc_loss = val_bc_loss / n_val_scenarios
        avg_val_F = np.mean(val_F_values)

        return avg_val_total_loss, avg_val_reliability_loss, avg_val_bc_loss, avg_val_F
