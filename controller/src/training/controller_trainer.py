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

    def _create_scheduler(self):
        """Create learning rate scheduler from config dict.

        Supported types: step, exponential, cosine, reduce_on_plateau.
        Returns None if no scheduler is configured.
        """
        cfg = self.lr_scheduler_config
        if cfg is None:
            return None

        sched_type = cfg.get('type', 'step')

        if sched_type == 'step':
            return torch_lr_scheduler.StepLR(
                self.optimizer,
                step_size=cfg.get('step_size', 50),
                gamma=cfg.get('gamma', 0.5)
            )
        elif sched_type == 'exponential':
            return torch_lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=cfg.get('gamma', 0.99)
            )
        elif sched_type == 'cosine':
            return torch_lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.get('T_max', 1500),
                eta_min=cfg.get('eta_min', 0)
            )
        elif sched_type == 'reduce_on_plateau':
            return torch_lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=cfg.get('factor', 0.5),
                patience=cfg.get('patience', 10)
            )
        else:
            raise ValueError(f"Unknown lr_scheduler type: {sched_type}")

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

    def _compute_normalization_stats(self):
        """
        Compute normalization statistics (min, max) for each process's CONTROLLABLE inputs
        from the target trajectories to scale BC loss to [0,1] range.

        Only controllable inputs are included since BC loss should not penalize
        non-controllable inputs (the controller cannot change them).
        """
        self.input_stats = {}

        for process_name, data in self.surrogate.target_trajectory_tensors.items():
            inputs = data['inputs']  # Shape: (n_scenarios, seq_len, input_dim)

            # Filter to controllable inputs only
            ctrl_idx = self.controllable_indices[process_name]
            if len(ctrl_idx) > 0:
                inputs = inputs[..., ctrl_idx]

            # Compute min and max across all scenarios and timesteps
            # reshape(-1) ensures 1D even with a single controllable dimension
            input_min = inputs.min(dim=0)[0].min(dim=0)[0].reshape(-1)  # Shape: (n_controllable,)
            input_max = inputs.max(dim=0)[0].max(dim=0)[0].reshape(-1)  # Shape: (n_controllable,)
            input_range = input_max - input_min

            # For constant dimensions (range ≈ 0), use |target_value| as scale
            # instead of near-zero range which would cause normalization to explode.
            # This gives a relative-deviation penalty: loss ∝ (Δ / |target|)²
            RANGE_THRESHOLD = 1e-4
            is_constant = input_range < RANGE_THRESHOLD

            if is_constant.any():
                # Fallback scale: max(|target_value|, 1.0) to avoid division issues near zero
                fallback_scale = torch.clamp(torch.abs(input_min), min=1.0)
                input_range = torch.where(is_constant, fallback_scale, input_range)

                const_indices = [i for i, c in enumerate(is_constant.tolist()) if c]
                const_values = [input_min[i].item() for i in const_indices]
                const_scales = [fallback_scale[i].item() for i in const_indices]
                print(f"  INFO [{process_name}]: constant controllable dims {const_indices} "
                      f"(values={const_values}) — using fallback scale {const_scales}")

            # Small epsilon for float precision safety
            input_range = input_range + 1e-8

            self.input_stats[process_name] = {
                'min': input_min.to(self.device),
                'range': input_range.to(self.device),
            }

        print(f"  Input normalization stats computed for {len(self.input_stats)} processes (controllable inputs only)")

    def compute_loss(self, trajectory, scenario_idx, reliability_weight=1.0, lambda_bc=None,
                     return_quality_scores=False):
        """
        Calcola loss totale per uno specifico scenario.

        Args:
            trajectory (dict): Output trajectory from process_chain.forward()
            scenario_idx (int): Index of the scenario being evaluated
            reliability_weight (float): Weight for reliability loss (0.0 = ignore, 1.0 = full)
            lambda_bc (float): Behavior cloning weight (if None, uses self.lambda_bc)
            return_quality_scores (bool): If True, also return per-process quality scores

        Returns:
            If return_quality_scores=False:
                total_loss, reliability_loss, bc_loss, F
            If return_quality_scores=True:
                total_loss, reliability_loss, bc_loss, F, quality_scores
        """
        # Use provided lambda_bc or fall back to instance value
        if lambda_bc is None:
            lambda_bc = self.lambda_bc

        # Reliability loss: scale * (F - F*)^2
        # Scale prevents vanishing gradients when delta F is small (~0.1)
        F, quality_scores = self.surrogate.compute_reliability(trajectory, return_quality_scores=True)

        # Debug: verify F is in the computation graph (first call only)
        if hasattr(self, '_debug_F_graph') and self._debug_F_graph:
            self._debug_F_graph = False
            print(f"\n{'='*60}", flush=True)
            print(f"SURROGATE GRADIENT DEBUG", flush=True)
            print(f"{'='*60}", flush=True)
            print(f"  F.requires_grad = {F.requires_grad}", flush=True)
            print(f"  F.grad_fn       = {F.grad_fn}", flush=True)
            print(f"  F.shape         = {F.shape}", flush=True)
            print(f"  F.mean()        = {F.mean().item():.6f}", flush=True)
            _f_star_debug = self.formula_surrogate.F_star if self.formula_surrogate is not None else self.surrogate.F_star
            print(f"  F*              = {_f_star_debug:.6f} ({'formula' if self.formula_surrogate is not None else 'surrogate'})", flush=True)
            if F.requires_grad and F.grad_fn is not None:
                print(f"  STATUS: OK - Gradients WILL flow to controller", flush=True)
            else:
                print(f"  STATUS: BROKEN - F is detached, reliability loss has NO gradient effect!", flush=True)
            print(f"{'='*60}\n", flush=True)

        # Use formula surrogate F* when available (CasualiT mode)
        F_star_val = self.formula_surrogate.F_star if self.formula_surrogate is not None else self.surrogate.F_star
        F_star_tensor = torch.tensor(F_star_val, dtype=torch.float32, device=self.device)
        reliability_loss = self.reliability_loss_scale * (F - F_star_tensor) ** 2

        # Behavior cloning loss: mean( ||a_ctrl - a_ctrl*||^2 ) across all processes
        # Only compares CONTROLLABLE inputs (non-controllable can't be changed by the controller)
        # Inputs are normalized using precomputed stats (with fallback scale for constant dims)
        bc_loss = torch.tensor(0.0, device=self.device)
        n_processes = len(trajectory.keys())

        for process_name in trajectory.keys():
            ctrl_idx = self.controllable_indices[process_name]
            if len(ctrl_idx) == 0:
                continue

            actual_inputs = trajectory[process_name]['inputs'][..., ctrl_idx]
            # Target trajectory may have only 1 sample (adaptive calibration targets).
            # Always use sample 0 for BC reference; clamp scenario_idx for legacy cases.
            target_all = self.surrogate.target_trajectory_tensors[process_name]['inputs']
            target_idx = min(scenario_idx, target_all.shape[0] - 1)
            target_inputs_scenario = target_all[target_idx:target_idx+1][..., ctrl_idx]

            # Normalize inputs using precomputed stats
            # Non-constant dims: normalized by actual range → [0,1]
            # Constant dims: normalized by max(|target|, 1.0) → relative deviation penalty
            stats = self.input_stats[process_name]
            actual_inputs_norm = (actual_inputs - stats['min']) / stats['range']
            target_inputs_norm = (target_inputs_scenario - stats['min']) / stats['range']

            # DEBUG: Print BC loss details on first call
            if hasattr(self, '_debug_bc_loss') and self._debug_bc_loss:
                print(f"\n  BC Loss Debug [{process_name}] (controllable only):")
                print(f"    controllable_indices: {ctrl_idx}")
                print(f"    actual_inputs (raw): {actual_inputs[0].tolist()}")
                print(f"    target_inputs (raw): {target_inputs_scenario[0].tolist()}")
                print(f"    stats['min']: {stats['min'].tolist()}")
                print(f"    stats['range']: {stats['range'].tolist()}")
                print(f"    actual_inputs_norm: {actual_inputs_norm[0].tolist()}")
                print(f"    target_inputs_norm: {target_inputs_norm[0].tolist()}")
                print(f"    MSE per dim: {((actual_inputs_norm - target_inputs_norm) ** 2).mean(dim=0).tolist()}")

            # Compute MSE on normalized controllable inputs
            bc_loss = bc_loss + torch.mean((actual_inputs_norm - target_inputs_norm) ** 2)

        # Average BC loss across processes (not sum!)
        bc_loss = bc_loss / n_processes

        # Total loss with dynamic weights
        # reliability_weight controls if reliability loss is active (0.0 during warm-up)
        total_loss = reliability_weight * torch.mean(reliability_loss) + lambda_bc * bc_loss

        if return_quality_scores:
            return total_loss, torch.mean(reliability_loss).item(), bc_loss.item(), torch.mean(F).item(), quality_scores
        return total_loss, torch.mean(reliability_loss).item(), bc_loss.item(), torch.mean(F).item()

    def _split_trajectory(self, trajectory, train_size, val_size):
        """
        Split a trajectory into training and validation portions.

        Args:
            trajectory (dict): Trajectory from process_chain.forward()
            train_size (int): Number of samples for training
            val_size (int): Number of samples for validation

        Returns:
            tuple: (train_trajectory, val_trajectory)
        """
        train_trajectory = {}
        val_trajectory = {}

        for process_name, data in trajectory.items():
            train_trajectory[process_name] = {
                'inputs': data['inputs'][:train_size],
                'outputs_mean': data['outputs_mean'][:train_size],
                'outputs_var': data['outputs_var'][:train_size],
                'outputs_sampled': data['outputs_sampled'][:train_size],
            }
            val_trajectory[process_name] = {
                'inputs': data['inputs'][train_size:train_size+val_size],
                'outputs_mean': data['outputs_mean'][train_size:train_size+val_size],
                'outputs_var': data['outputs_var'][train_size:train_size+val_size],
                'outputs_sampled': data['outputs_sampled'][train_size:train_size+val_size],
            }

        return train_trajectory, val_trajectory

    def set_validation_data(self, validation_surrogate, validation_process_chain):
        """
        Set cross-scenario validation data for computing validation loss during training.

        This allows tracking train vs validation loss to detect overfitting
        across different operating conditions.

        Args:
            validation_surrogate (ProTSurrogate): Surrogate for test scenarios
            validation_process_chain (ProcessChain): Process chain for test scenarios
                (will have trained policy weights copied from main chain)
        """
        self.validation_surrogate = validation_surrogate
        self.validation_process_chain = validation_process_chain
        print(f"  Cross-scenario validation data set: {validation_surrogate.n_scenarios} test scenarios")

    def set_within_scenario_validation(self, enabled=True, split_fraction=0.2):
        """
        Configure within-scenario validation (train/val split within same scenarios).

        This splits each batch of samples into training and validation portions.
        The validation samples are NEVER used for gradient updates, preventing data leakage.

        Args:
            enabled (bool): Enable within-scenario validation
            split_fraction (float): Fraction of samples to use as validation (0.0 to 0.5)
                                   E.g., 0.2 = 20% validation, 80% training
        """
        self.within_scenario_validation_enabled = enabled
        self.within_scenario_split = max(0.0, min(0.5, split_fraction))  # Clamp to [0, 0.5]

        if enabled:
            print(f"  Within-scenario validation: ENABLED")
            print(f"    Split: {self.within_scenario_split*100:.0f}% validation, {(1-self.within_scenario_split)*100:.0f}% training")
        else:
            print(f"  Within-scenario validation: DISABLED")

    def set_baseline_reliabilities(self, F_baseline_per_scenario):
        """
        Store precomputed F_baseline per scenario for gap closure computation during training.

        Args:
            F_baseline_per_scenario (array-like): F' for each scenario, shape (n_scenarios,)
        """
        self.F_baseline_per_scenario = np.array(F_baseline_per_scenario)
        print(f"  Baseline reliabilities set for {len(self.F_baseline_per_scenario)} scenarios")
        print(f"    F' range: [{self.F_baseline_per_scenario.min():.6f}, {self.F_baseline_per_scenario.max():.6f}]")

    def compute_correlation_matrix(self):
        """
        Compute correlation matrix between process quality scores from Q_history.

        Uses the quality scores (Qᵢ) collected during training to estimate
        empirical correlations ρᵢⱼ between processes. These correlations
        are used to compute a more accurate L_min that accounts for
        correlated process errors.

        Formula:
            ρᵢⱼ = Cov(Qᵢ, Qⱼ) / (σᵢ × σⱼ)

        Returns:
            Dict[Tuple[str, str], float]: Correlation matrix as dict mapping
                (process_i, process_j) to correlation coefficient ρᵢⱼ.
                Returns empty dict if not enough data.
        """
        if not self.Q_history:
            return {}

        process_names = list(self.Q_history.keys())

        # Need at least 2 processes and some data points
        if len(process_names) < 2:
            return {}

        # Check all processes have same length
        n_samples = len(self.Q_history[process_names[0]])
        if n_samples < 10:  # Need minimum samples for meaningful correlation
            return {}

        for name in process_names:
            if len(self.Q_history[name]) != n_samples:
                print(f"  Warning: Inconsistent Q_history lengths, skipping correlation")
                return {}

        # Compute correlation matrix
        correlation_matrix = {}

        for i, name_i in enumerate(process_names):
            for j, name_j in enumerate(process_names):
                if i >= j:  # Only upper triangle, correlation is symmetric
                    continue

                Q_i = np.array(self.Q_history[name_i])
                Q_j = np.array(self.Q_history[name_j])

                # Compute Pearson correlation
                # Handle edge case where one array has no variance
                std_i = np.std(Q_i)
                std_j = np.std(Q_j)

                if std_i < 1e-10 or std_j < 1e-10:
                    rho = 0.0  # No correlation if no variance
                else:
                    rho = np.corrcoef(Q_i, Q_j)[0, 1]
                    # Handle NaN from corrcoef
                    if np.isnan(rho):
                        rho = 0.0

                # Store both directions (symmetric)
                correlation_matrix[(name_i, name_j)] = rho
                correlation_matrix[(name_j, name_i)] = rho

        return correlation_matrix

    def get_correlation_summary(self):
        """
        Get a summary of the correlation matrix for logging.

        Returns:
            str: Formatted string with correlation summary
        """
        corr_matrix = self.compute_correlation_matrix()

        if not corr_matrix:
            return "  No correlation data available"

        lines = ["  Process Correlation Matrix:"]
        process_names = list(self.Q_history.keys())

        # Header
        header = "    " + " ".join(f"{name[:6]:>8}" for name in process_names)
        lines.append(header)

        # Matrix rows
        for name_i in process_names:
            row = f"    {name_i[:6]:>6}"
            for name_j in process_names:
                if name_i == name_j:
                    row += f"{'1.000':>8}"
                else:
                    rho = corr_matrix.get((name_i, name_j), 0.0)
                    row += f"{rho:>8.3f}"
            lines.append(row)

        return "\n".join(lines)

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
            json.dump(state, f, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else str(o))

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
        from controller.src.io.utils import convert_numpy_to_tensor

        # F* and F_baseline from formula surrogate when available (CasualiT mode)
        baseline_surrogate = self.formula_surrogate if self.formula_surrogate is not None else self.surrogate
        F_star = baseline_surrogate.F_star

        # F' = evaluate baseline trajectory
        with torch.no_grad():
            self.process_chain.eval()
            baseline_tensor = convert_numpy_to_tensor(baseline_trajectory, device=self.device)
            F_baseline = baseline_surrogate.compute_reliability(baseline_tensor).item()

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
                _f_star_ref = self.formula_surrogate.F_star if self.formula_surrogate is not None else self.surrogate.F_star
                print(f"  F value={F_tensor.mean().item():.6f}, F*={_f_star_ref:.6f}")
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
                _f_star = self.formula_surrogate.F_star if self.formula_surrogate is not None else self.surrogate.F_star
                denom = _f_star - self.F_baseline_per_scenario[s_idx]
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

    def train(self, epochs=100, batch_size=32,
              patience=20, save_dir='checkpoints/controller', verbose=True):
        """
        Training loop completo con early stopping.

        Each epoch cycles through all scenarios exactly once in shuffled order.
        Gradients are accumulated across all scenarios with a single optimizer
        step per epoch to avoid catastrophic forgetting.

        Args:
            epochs: Number of training epochs
            batch_size: Total number of samples per epoch (split equally across scenarios).
                        Each scenario gets samples_per_scenario = batch_size // n_scenarios.
            patience: Early stopping patience
            save_dir: Directory to save checkpoints
            verbose: Print training progress

        Returns:
            history (dict): Training history con tutte le metriche
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        n_scenarios = self.surrogate.n_scenarios

        # Calculate warmup epochs for curriculum learning
        warmup_epochs = 0
        if self.curriculum_config['enabled']:
            warmup_epochs = int(epochs * self.curriculum_config['warmup_fraction'])

        # Reset patience activation flag (for cases where train() is called multiple times)
        self._patience_activated = False

        if verbose:
            print(f"\n{'='*70}")
            print("STARTING CONTROLLER TRAINING")
            print(f"{'='*70}")
            print(f"  Epochs: {epochs}")
            print(f"  Scenarios per epoch: {n_scenarios} (all scenarios)")
            print(f"  Total batch size: {batch_size}")
            print(f"  Samples per scenario: {max(1, batch_size // n_scenarios)}")
            print(f"  Patience: {patience}")
            print(f"  Save dir: {save_dir}")
            _f_star = self.formula_surrogate.F_star if self.formula_surrogate is not None else self.surrogate.F_star
            print(f"  F* (target): {_f_star:.6f}")

            if self.curriculum_config['enabled']:
                print(f"\n  CURRICULUM LEARNING STRATEGY:")
                print(f"    Phase 1 - Warm-up: Epochs 1-{warmup_epochs} (BC only)")
                print(f"    Phase 2 - Curriculum: Epochs {warmup_epochs + 1}-{epochs} (gradual reliability)")
                print(f"    Lambda BC schedule: {self.curriculum_config['lambda_bc_start']:.3f} → {self.curriculum_config['lambda_bc_end']:.6f}")
                print(f"    Reliability weight curve: {self.curriculum_config['reliability_weight_curve']}")
                print(f"    Patience: starts after warm-up AND reliability_weight >= 0.9")

        # Save embedding snapshot at epoch 1 (initial state)
        if hasattr(self.process_chain, 'scenario_encoder') and self.process_chain.scenario_encoder is not None:
            self._save_embedding_snapshot(epoch=1)
            if verbose:
                print(f"  Saved initial embedding snapshot")

        for epoch in range(1, epochs + 1):
            # Disable debug after first epoch to avoid flooding output
            if epoch == 2:
                self.process_chain.enable_debug(False)
                self._debug_gradients = False  # Also disable gradient debug
                self._debug_bc_loss = False  # Also disable BC loss debug
                self._debug_F_graph = False  # Also disable F graph debug

            # Get dynamic loss weights for curriculum learning
            lambda_bc, reliability_weight, phase = self.get_loss_weights(epoch, epochs)

            # Train epoch with dynamic weights
            avg_total_loss, avg_rel_loss, avg_bc_loss, avg_F = self.train_epoch(
                batch_size=batch_size,
                reliability_weight=reliability_weight,
                lambda_bc=lambda_bc
            )

            # Track history
            current_lr = self.get_current_lr()
            self.history['total_loss'].append(avg_total_loss)
            self.history['reliability_loss'].append(avg_rel_loss)
            self.history['bc_loss'].append(avg_bc_loss)
            self.history['F_values'].append(avg_F)
            self.history['lambda_bc'].append(lambda_bc)
            self.history['reliability_weight'].append(reliability_weight)
            if self._epoch_gap_closure is not None:
                self.history['gap_closure'].append(self._epoch_gap_closure)
            self.history['learning_rate'].append(current_lr)
            if self._epoch_avg_F_formula is not None:
                self.history['F_formula_values'].append(self._epoch_avg_F_formula)

            # Compute cross-scenario validation loss (on test scenarios) if validation data is set
            if self.validation_surrogate is not None:
                val_total, val_rel, val_bc, val_F = self.compute_validation_loss(
                    batch_size=batch_size,
                    reliability_weight=reliability_weight,
                    lambda_bc=lambda_bc
                )
                self.history['val_total_loss'].append(val_total)
                self.history['val_reliability_loss'].append(val_rel)
                self.history['val_bc_loss'].append(val_bc)
                self.history['val_F_values'].append(val_F)

            # Track within-scenario validation metrics (already computed in train_epoch)
            if self._within_val_metrics is not None:
                self.history['val_within_total_loss'].append(self._within_val_metrics['total_loss'])
                self.history['val_within_reliability_loss'].append(self._within_val_metrics['reliability_loss'])
                self.history['val_within_bc_loss'].append(self._within_val_metrics['bc_loss'])
                self.history['val_within_F_values'].append(self._within_val_metrics['F'])

            # Step the learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch_lr_scheduler.ReduceLROnPlateau):
                    # ReduceLROnPlateau needs the metric value
                    self.scheduler.step(avg_total_loss)
                else:
                    # All other schedulers just step
                    self.scheduler.step()

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
                print(f"  λ_BC: {lambda_bc:.6f}, Rel Weight: {reliability_weight:.3f}, LR: {current_lr:.2e}")
                print(f"  Train Loss:       {avg_total_loss:.6f}")
                print(f"  Reliability Loss: {avg_rel_loss:.6f} {'(ignored)' if reliability_weight == 0 else ''}")
                print(f"  BC Loss:          {avg_bc_loss:.6f}")
                print(f"  F (actual):       {avg_F:.6f}")
                if self._epoch_avg_F_formula is not None:
                    print(f"  F (formula):      {self._epoch_avg_F_formula:.6f}")
                _f_star = self.formula_surrogate.F_star if self.formula_surrogate is not None else self.surrogate.F_star
                print(f"  F* (target):      {_f_star:.6f}")
                if self._epoch_gap_closure is not None:
                    print(f"  Gap Closure:      {self._epoch_gap_closure:.4f}")
                # Print cross-scenario validation metrics if available
                if self.validation_surrogate is not None and len(self.history['val_total_loss']) > 0:
                    val_loss = self.history['val_total_loss'][-1]
                    val_F = self.history['val_F_values'][-1]
                    print(f"  Val Loss (cross): {val_loss:.6f}")
                    print(f"  Val F (cross):    {val_F:.6f}")
                # Print within-scenario validation metrics if available
                if self.within_scenario_validation_enabled and len(self.history['val_within_total_loss']) > 0:
                    val_within_loss = self.history['val_within_total_loss'][-1]
                    val_within_F = self.history['val_within_F_values'][-1]
                    print(f"  Val Loss (within):{val_within_loss:.6f}")
                    print(f"  Val F (within):   {val_within_F:.6f}")

            # Print warm-up completion message
            if verbose and self.curriculum_config['enabled'] and epoch == warmup_epochs:
                print(f"\n{'='*70}")
                print("WARM-UP COMPLETED! Starting curriculum learning phase...")
                print(f"{'='*70}")

            # Print message when patience becomes active (reliability_weight >= 0.9)
            if verbose and self.curriculum_config['enabled'] and reliability_weight >= 0.9:
                if not hasattr(self, '_patience_activated') or not self._patience_activated:
                    self._patience_activated = True
                    print(f"\n{'='*70}")
                    print(f"PATIENCE ACTIVATED! (reliability_weight={reliability_weight:.3f} >= 0.9)")
                    print(f"Early stopping will now monitor loss improvements (patience={patience})")
                    print(f"{'='*70}")

            # Check for improvement (based on LOSS, not F value)
            # We want F to become more similar to F*, i.e., loss to decrease
            # IMPORTANT: During warm-up and early curriculum phase, reliability loss is
            # not yet dominant, so we skip early stopping until reliability_weight >= 0.9

            # Patience only starts counting when:
            # 1. Warm-up is finished (epoch > warmup_epochs)
            # 2. Reliability weight has reached 0.9 (loss is now reliability-dominated)
            patience_active = epoch > warmup_epochs and reliability_weight >= 0.9

            # Best model saving only when patience is active
            # (avoids saving models optimized for BC loss instead of reliability)
            if patience_active and avg_total_loss < self.best_loss:
                self.best_loss = avg_total_loss
                self.best_F = avg_F  # Track best F for information
                self.epochs_without_improvement = 0

                # Save best model
                self.save_checkpoint(save_dir / 'best_model.pt', epoch)

                if verbose:
                    print(f"  ✓ New best loss: {self.best_loss:.6f} (F: {self.best_F:.6f})")

            else:
                # Only count epochs without improvement when patience is active
                if patience_active:
                    self.epochs_without_improvement += 1

                # Early stopping: ONLY when patience is active and threshold reached
                if patience_active and self.epochs_without_improvement >= patience:
                    if verbose:
                        print(f"\n  Early stopping triggered (patience={patience})")
                        print(f"  Loss has not improved for {patience} epochs")
                        print(f"  (Patience was active since reliability_weight >= 0.9)")
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
            json.dump(self.history, f, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else str(o))

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
                progression_data[f'{prefix}_F_star'] = snapshot['F_star']

                # Save all 3 samples if available, otherwise fallback to single trajectory
                trajectories = snapshot.get('trajectories', [snapshot['trajectory']])
                F_actuals = snapshot.get('F_actuals', [snapshot['F_actual']])
                seeds = snapshot.get('seeds', [0])

                progression_data[f'{prefix}_n_samples'] = len(trajectories)
                progression_data[f'{prefix}_seeds'] = np.array(seeds)
                progression_data[f'{prefix}_F_actuals'] = np.array(F_actuals)
                progression_data[f'{prefix}_F_actual'] = F_actuals[0]  # Backward compatibility

                # Save inputs and outputs for each sample and process
                for sample_idx, traj in enumerate(trajectories):
                    for process_name in traj.keys():
                        sample_prefix = f'{prefix}_sample{sample_idx}_{process_name}'
                        progression_data[f'{sample_prefix}_inputs'] = traj[process_name]['inputs']
                        progression_data[f'{sample_prefix}_outputs'] = traj[process_name]['outputs_mean']

                # Save target (same for all samples) - backward compatible keys
                for process_name in snapshot['target_trajectory'].keys():
                    progression_data[f'{prefix}_{process_name}_target_inputs'] = snapshot['target_trajectory'][process_name]['inputs']
                    progression_data[f'{prefix}_{process_name}_target_outputs'] = snapshot['target_trajectory'][process_name]['outputs']
                    # Also save with sample0 prefix for new format
                    progression_data[f'{prefix}_sample0_{process_name}_inputs'] = trajectories[0][process_name]['inputs']
                    progression_data[f'{prefix}_sample0_{process_name}_outputs'] = trajectories[0][process_name]['outputs_mean']

            np.savez(progression_path, **progression_data)

            if verbose:
                print(f"  Saved {len(self.training_progression)} training progression snapshots to {progression_path}")

        if verbose:
            print(f"\n{'='*70}")
            print("TRAINING COMPLETED")
            print(f"{'='*70}")
            print(f"  Best F: {self.best_F:.6f}")
            print(f"  Final F: {self.history['F_values'][-1]:.6f}")
            _f_star = self.formula_surrogate.F_star if self.formula_surrogate is not None else self.surrogate.F_star
            print(f"  Target F*: {_f_star:.6f}")

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
                'trajectories': list of trajectories (representative, one per scenario),
                'F_formula_per_sample': np.array (only when formula_surrogate is set),
                'F_formula_mean': float (only when formula_surrogate is set),
                'F_formula_std': float (only when formula_surrogate is set),
            }
        """
        self.process_chain.eval()

        n_scenarios = self.surrogate.n_scenarios
        F_actual_values = []
        F_formula_values = []
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

                        # Compute F from formula for comparison
                        if self.formula_surrogate is not None:
                            F_formula = self.formula_surrogate.compute_reliability(sample_trajectory).item()
                            F_formula_values.append(F_formula)
                else:
                    # Compute reliability for aggregated batch (take mean across batch)
                    F_actual_batch = self.surrogate.compute_reliability(trajectory)
                    F_actual = F_actual_batch.mean().item()
                    F_actual_values.append(F_actual)

                    # Compute F from formula for comparison
                    if self.formula_surrogate is not None:
                        F_formula_batch = self.formula_surrogate.compute_reliability(trajectory)
                        F_formula = F_formula_batch.mean().item()
                        F_formula_values.append(F_formula)

                # Save trajectory for each scenario (needed for representative trajectory selection)
                trajectories.append(trajectory)

        F_actual_array = np.array(F_actual_values)

        result = {
            'F_actual_per_sample': F_actual_array,
            'F_actual_mean': np.mean(F_actual_array),
            'F_actual_std': np.std(F_actual_array),
            'F_star_mean': self.formula_surrogate.F_star if self.formula_surrogate is not None else self.surrogate.F_star,
            'F_star_std': 0.0,
            'trajectories': trajectories
        }

        # Add formula-based F if available
        if F_formula_values:
            F_formula_array = np.array(F_formula_values)
            result['F_formula_per_sample'] = F_formula_array
            result['F_formula_mean'] = np.mean(F_formula_array)
            result['F_formula_std'] = np.std(F_formula_array)

        return result

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

        n_scenarios = self.surrogate.n_scenarios
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

        Generates one trajectory per scenario (fixed seed for reproducibility),
        allowing tracking of how controller outputs evolve during training.

        Args:
            epoch (int): Current epoch number
            lambda_bc (float): Current lambda_bc value
            reliability_weight (float): Current reliability_weight value
            phase (str): Current training phase
        """
        self.process_chain.eval()

        n_scenarios = self.surrogate.n_scenarios
        fixed_seed = 42  # Single fixed seed for reproducibility across epochs

        with torch.no_grad():
            from controller.src.evaluation.metrics import convert_trajectory_to_numpy

            rng_state = torch.get_rng_state()

            # Generate one trajectory per scenario
            per_scenario = {}
            F_actuals_per_scenario = {}
            for scenario_idx in range(n_scenarios):
                torch.manual_seed(fixed_seed)
                trajectory = self.process_chain.forward(
                    batch_size=1, scenario_idx=scenario_idx
                )
                per_scenario[scenario_idx] = convert_trajectory_to_numpy(trajectory)
                F_actuals_per_scenario[scenario_idx] = self.surrogate.compute_reliability(trajectory).item()

            torch.set_rng_state(rng_state)

            # Target trajectory (single sample, same for all scenarios)
            target_trajectory_np = {}
            for process_name, data in self.surrogate.target_trajectory_tensors.items():
                t_idx = min(0, data['inputs'].shape[0] - 1)
                target_trajectory_np[process_name] = {
                    'inputs': data['inputs'][t_idx:t_idx+1].cpu().numpy(),
                    'outputs': data['outputs'][t_idx:t_idx+1].cpu().numpy()
                }

            F_star = self.formula_surrogate.F_star if self.formula_surrogate is not None else self.surrogate.F_star

            # Backward-compatible fields (scenario 0)
            trajectories_0 = [per_scenario[0]]
            F_actuals_0 = [F_actuals_per_scenario[0]]

            snapshot = {
                'epoch': epoch,
                'phase': phase,
                'lambda_bc': lambda_bc,
                'reliability_weight': reliability_weight,
                # Per-scenario data (new)
                'per_scenario': per_scenario,
                'F_per_scenario': F_actuals_per_scenario,
                # Backward compatibility (scenario 0)
                'trajectories': trajectories_0,
                'trajectory': trajectories_0[0],
                'target_trajectory': target_trajectory_np,
                'F_actuals': F_actuals_0,
                'F_actual': F_actuals_0[0],
                'F_star': F_star,
                'scenario_idx': 0,
                'seeds': [fixed_seed]
            }

            self.training_progression.append(snapshot)

        self.process_chain.train()
