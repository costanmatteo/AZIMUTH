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
