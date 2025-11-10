"""
Trainer for Uncertainty Quantification Model with Conditional Process Support

This trainer handles training loop for models that predict both mean and variance,
with optional conditioning on process ID and environment variables for multi-process learning.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Optional, Dict, List
from collections import defaultdict


class UncertaintyTrainer:
    """
    Trainer for uncertainty quantification models with optional process conditioning.

    Features:
    - Training loop with validation
    - Early stopping
    - Checkpoint saving
    - Metrics logging for both mean predictions and uncertainty
    - Per-process metrics (when conditioning is enabled)
    - Backward compatible with non-conditional models

    Args:
        model (nn.Module): Uncertainty model to train
        criterion (nn.Module): Loss function (typically GaussianNLLLoss or EnergyScoreLoss)
        device (str): 'cuda' or 'cpu' (default: auto-detect)
        learning_rate (float): Learning rate (default: 0.001)
        weight_decay (float): L2 regularization strength (default: 0.0)
        conditioning_enabled (bool): Whether model uses conditional processing (default: False)
    """

    def __init__(
        self,
        model,
        criterion,
        device=None,
        learning_rate=0.001,
        weight_decay=0.0,
        conditioning_enabled=False
    ):
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.conditioning_enabled = conditioning_enabled

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Tracking - overall metrics
        self.train_losses = []
        self.val_losses = []
        self.train_mse = []
        self.val_mse = []
        self.best_val_loss = float('inf')

        # Tracking - per-process metrics (when conditioning enabled)
        self.per_process_metrics = defaultdict(lambda: {'val_mse': [], 'val_variance': []})

        print(f"UncertaintyTrainer initialized on device: {self.device}")
        print(f"Optimizer: Adam (lr={learning_rate}, weight_decay={weight_decay})")
        print(f"Loss function: {criterion.__class__.__name__}")
        print(f"Conditioning: {'Enabled' if conditioning_enabled else 'Disabled'}")

    def _extract_batch_data(self, batch):
        """
        Extract data from batch dict and move to device.

        Handles both formats:
        - Standard: (X, y) tuple or {'X': X, 'y': y}
        - Conditional: {'X': X, 'y': y, 'process_id': ..., 'env_continuous': ..., ...}

        Returns:
            tuple: (X, y, conditioning_kwargs)
        """
        # Handle old tuple format for backward compatibility
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            X, y = batch
            X = X.to(self.device)
            y = y.to(self.device)
            return X, y, {}

        # Handle dict format
        X = batch['X'].to(self.device)
        y = batch['y'].to(self.device)

        # Extract conditioning arguments if present
        conditioning_kwargs = {}
        if self.conditioning_enabled:
            if 'process_id' in batch:
                conditioning_kwargs['process_id'] = batch['process_id'].to(self.device)
            if 'env_continuous' in batch:
                conditioning_kwargs['env_continuous'] = batch['env_continuous'].to(self.device)
            if 'env_continuous_masks' in batch:
                conditioning_kwargs['env_masks'] = batch['env_continuous_masks'].to(self.device)
            if 'env_categorical' in batch:
                env_cat = {k: v.to(self.device) for k, v in batch['env_categorical'].items()}
                conditioning_kwargs['env_categorical'] = env_cat
            if 'timestamp' in batch:
                conditioning_kwargs['timestamp'] = batch['timestamp'].to(self.device)

        return X, y, conditioning_kwargs

    def train_epoch(self, train_loader):
        """
        Training for a single epoch.

        Args:
            train_loader (DataLoader): DataLoader for training set

        Returns:
            tuple: (avg_loss, avg_mse)
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0

        for batch in train_loader:
            # Extract data
            X, y, conditioning_kwargs = self._extract_batch_data(batch)

            # Forward pass
            mean, variance = self.model(X, **conditioning_kwargs)

            # Compute loss
            loss = self.criterion(mean, variance, y)

            # Compute MSE for monitoring
            with torch.no_grad():
                mse = torch.mean((mean - y) ** 2)
                epoch_mse += mse.item()

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        avg_mse = epoch_mse / len(train_loader)
        return avg_loss, avg_mse

    def validate(self, val_loader, compute_per_process_metrics=True):
        """
        Model validation.

        Args:
            val_loader (DataLoader): DataLoader for validation set
            compute_per_process_metrics (bool): If True and conditioning enabled,
                                               compute metrics per process

        Returns:
            tuple: (avg_loss, avg_mse, avg_variance, per_process_metrics)
                - avg_loss: Average loss
                - avg_mse: Average MSE of mean predictions
                - avg_variance: Average predicted variance
                - per_process_metrics: Dict {process_id: {'mse': float, 'variance': float}}
                                     or None if not computed
        """
        self.model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_variance = 0.0

        # For per-process metrics
        per_process_stats = defaultdict(lambda: {'squared_errors': [], 'variances': []})

        with torch.no_grad():
            for batch in val_loader:
                X, y, conditioning_kwargs = self._extract_batch_data(batch)

                # Forward pass
                mean, variance = self.model(X, **conditioning_kwargs)

                # Compute losses
                loss = self.criterion(mean, variance, y)
                mse = torch.mean((mean - y) ** 2)

                val_loss += loss.item()
                val_mse += mse.item()
                val_variance += torch.mean(variance).item()

                # Collect per-process statistics
                if compute_per_process_metrics and self.conditioning_enabled and 'process_id' in conditioning_kwargs:
                    process_ids = conditioning_kwargs['process_id'].cpu().numpy()
                    squared_errors = ((mean - y) ** 2).cpu().numpy()
                    variances_np = variance.cpu().numpy()

                    for i, pid in enumerate(process_ids):
                        per_process_stats[pid]['squared_errors'].append(squared_errors[i])
                        per_process_stats[pid]['variances'].append(variances_np[i])

        n_batches = len(val_loader)
        avg_loss = val_loss / n_batches
        avg_mse = val_mse / n_batches
        avg_variance = val_variance / n_batches

        # Compute per-process metrics
        per_process_metrics = None
        if compute_per_process_metrics and per_process_stats:
            per_process_metrics = {}
            for pid, stats in per_process_stats.items():
                sq_errors = np.concatenate(stats['squared_errors'], axis=0)
                vars_np = np.concatenate(stats['variances'], axis=0)
                per_process_metrics[int(pid)] = {
                    'mse': float(np.mean(sq_errors)),
                    'variance': float(np.mean(vars_np)),
                    'n_samples': len(sq_errors)
                }

        return avg_loss, avg_mse, avg_variance, per_process_metrics

    def train(
        self,
        train_loader,
        val_loader,
        epochs=100,
        patience=10,
        save_dir='checkpoints_uncertainty',
        compute_per_process_metrics=None
    ):
        """
        Complete training with early stopping.

        Args:
            train_loader (DataLoader): DataLoader for training
            val_loader (DataLoader): DataLoader for validation
            epochs (int): Maximum number of epochs
            patience (int): Epochs to wait before early stopping
            save_dir (str): Directory to save checkpoints
            compute_per_process_metrics (bool, optional): If True, compute per-process metrics.
                If None (default), uses self.conditioning_enabled.

        Returns:
            dict: Dictionary with training history
        """
        # Default to conditioning_enabled if not specified
        if compute_per_process_metrics is None:
            compute_per_process_metrics = self.conditioning_enabled

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"START UNCERTAINTY QUANTIFICATION TRAINING")
        if self.conditioning_enabled:
            print(f"Mode: MULTI-PROCESS CONDITIONAL TRAINING")
        else:
            print(f"Mode: STANDARD TRAINING")
        print(f"{'='*70}")
        print(f"Epochs: {epochs}")
        print(f"Early stopping patience: {patience}")
        print(f"Checkpoint directory: {save_path}")
        print(f"{'='*70}\n")

        epochs_without_improvement = 0

        for epoch in range(epochs):
            # Training
            train_loss, train_mse = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_mse.append(train_mse)

            # Validation
            val_loss, val_mse, val_variance, per_process_metrics = self.validate(
                val_loader,
                compute_per_process_metrics=compute_per_process_metrics
            )
            self.val_losses.append(val_loss)
            self.val_mse.append(val_mse)

            # Logging
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train - NLL Loss: {train_loss:.6f}, MSE: {train_mse:.6f}")
            print(f"  Val   - NLL Loss: {val_loss:.6f}, MSE: {val_mse:.6f}, "
                  f"Avg Variance: {val_variance:.6f}")

            # Per-process metrics logging
            if per_process_metrics is not None:
                print(f"  Per-Process Val Metrics:")
                process_names = {0: 'Laser', 1: 'Plasma', 2: 'Galvanic', 3: 'Microetch'}
                for pid in sorted(per_process_metrics.keys()):
                    metrics = per_process_metrics[pid]
                    proc_name = process_names.get(pid, f'Process {pid}')
                    print(f"    {proc_name:12s} - MSE: {metrics['mse']:.6f}, "
                          f"Variance: {metrics['variance']:.6f}, "
                          f"Samples: {metrics['n_samples']}")

                    # Store for history
                    self.per_process_metrics[pid]['val_mse'].append(metrics['mse'])
                    self.per_process_metrics[pid]['val_variance'].append(metrics['variance'])

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(save_path / 'best_model.pth', epoch, val_loss)
                print(f"  → New best model saved! (Val NLL Loss: {val_loss:.6f})")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best Val NLL Loss: {self.best_val_loss:.6f}")
                break

        # Save training history
        self.save_training_history(save_path / 'training_history.json')

        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETED")
        print(f"Best Val NLL Loss: {self.best_val_loss:.6f}")
        print(f"{'='*70}\n")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_mse': self.train_mse,
            'val_mse': self.val_mse,
            'best_val_loss': self.best_val_loss,
            'total_epochs': len(self.train_losses),
            'per_process_metrics': dict(self.per_process_metrics) if self.conditioning_enabled else None
        }

    def save_checkpoint(self, filepath, epoch, val_loss):
        """Save a model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_mse': self.train_mse,
            'val_mse': self.val_mse,
            'conditioning_enabled': self.conditioning_enabled,
            'per_process_metrics': dict(self.per_process_metrics)
        }, filepath)

    def load_checkpoint(self, filepath):
        """Load a model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_mse = checkpoint.get('train_mse', [])
        self.val_mse = checkpoint.get('val_mse', [])
        self.conditioning_enabled = checkpoint.get('conditioning_enabled', False)
        self.per_process_metrics = defaultdict(lambda: {'val_mse': [], 'val_variance': []},
                                               checkpoint.get('per_process_metrics', {}))
        print(f"Checkpoint loaded from: {filepath}")
        return checkpoint

    def save_training_history(self, filepath):
        """Save training history to JSON"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_mse': self.train_mse,
            'val_mse': self.val_mse,
            'best_val_loss': float(self.best_val_loss),
            'conditioning_enabled': self.conditioning_enabled,
            'timestamp': datetime.now().isoformat()
        }

        # Add per-process metrics if available
        if self.conditioning_enabled and self.per_process_metrics:
            history['per_process_metrics'] = dict(self.per_process_metrics)

        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)

    def predict(
        self,
        X,
        return_uncertainty=True,
        process_id=None,
        env_continuous=None,
        env_categorical=None,
        timestamp=None,
        env_masks=None
    ):
        """
        Make predictions on new data.

        Args:
            X (np.ndarray or torch.Tensor): Input data
            return_uncertainty (bool): If True, return both mean and variance
            process_id: Optional process IDs for conditional prediction
            env_continuous: Optional continuous environment variables
            env_categorical: Optional categorical environment variables
            timestamp: Optional timestamps
            env_masks: Optional missing value masks

        Returns:
            If return_uncertainty=True:
                tuple: (mean_predictions, variance_predictions)
            If return_uncertainty=False:
                np.ndarray: Mean predictions only
        """
        self.model.eval()

        # Convert to tensor if necessary
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        X = X.to(self.device)

        # Prepare conditioning kwargs
        conditioning_kwargs = {}
        if self.conditioning_enabled:
            if process_id is not None:
                if isinstance(process_id, np.ndarray):
                    process_id = torch.LongTensor(process_id)
                conditioning_kwargs['process_id'] = process_id.to(self.device)

            if env_continuous is not None:
                if isinstance(env_continuous, np.ndarray):
                    env_continuous = torch.FloatTensor(env_continuous)
                conditioning_kwargs['env_continuous'] = env_continuous.to(self.device)

            if env_categorical is not None:
                env_cat_tensors = {}
                for k, v in env_categorical.items():
                    if isinstance(v, np.ndarray):
                        v = torch.LongTensor(v)
                    env_cat_tensors[k] = v.to(self.device)
                conditioning_kwargs['env_categorical'] = env_cat_tensors

            if timestamp is not None:
                if isinstance(timestamp, np.ndarray):
                    timestamp = torch.FloatTensor(timestamp)
                conditioning_kwargs['timestamp'] = timestamp.to(self.device)

            if env_masks is not None:
                if isinstance(env_masks, np.ndarray):
                    env_masks = torch.BoolTensor(env_masks)
                conditioning_kwargs['env_masks'] = env_masks.to(self.device)

        with torch.no_grad():
            mean, variance = self.model(X, **conditioning_kwargs)

        if return_uncertainty:
            return mean.cpu().numpy(), variance.cpu().numpy()
        else:
            return mean.cpu().numpy()

    def compute_calibration_metrics(self, val_loader):
        """
        Compute calibration metrics to assess uncertainty quality.

        Checks if predicted uncertainties are well-calibrated:
        - MSE vs mean variance (calibration ratio should be close to 1)
        - Per-process calibration (if conditioning enabled)

        Args:
            val_loader (DataLoader): Validation data loader

        Returns:
            dict: Dictionary with calibration metrics
        """
        self.model.eval()
        squared_errors = []
        variances = []

        # Per-process tracking
        per_process_cal = defaultdict(lambda: {'sq_errors': [], 'variances': []})

        with torch.no_grad():
            for batch in val_loader:
                X, y, conditioning_kwargs = self._extract_batch_data(batch)
                mean, variance = self.model(X, **conditioning_kwargs)

                # Compute squared errors
                sq_error = (mean - y) ** 2
                squared_errors.append(sq_error.cpu().numpy())
                variances.append(variance.cpu().numpy())

                # Per-process calibration
                if self.conditioning_enabled and 'process_id' in conditioning_kwargs:
                    process_ids = conditioning_kwargs['process_id'].cpu().numpy()
                    sq_err_np = sq_error.cpu().numpy()
                    var_np = variance.cpu().numpy()

                    for i, pid in enumerate(process_ids):
                        per_process_cal[pid]['sq_errors'].append(sq_err_np[i])
                        per_process_cal[pid]['variances'].append(var_np[i])

        squared_errors = np.concatenate(squared_errors, axis=0)
        variances = np.concatenate(variances, axis=0)

        mse = np.mean(squared_errors)
        mean_variance = np.mean(variances)
        calibration_ratio = mse / mean_variance if mean_variance > 0 else float('inf')

        result = {
            'mse': float(mse),
            'mean_predicted_variance': float(mean_variance),
            'calibration_ratio': float(calibration_ratio),
            'interpretation': 'Well calibrated!' if 0.8 <= calibration_ratio <= 1.2 else
                            'Under-confident (predicts too much uncertainty)' if calibration_ratio < 0.8 else
                            'Over-confident (predicts too little uncertainty)'
        }

        # Per-process calibration
        if per_process_cal:
            per_process_results = {}
            process_names = {0: 'Laser', 1: 'Plasma', 2: 'Galvanic', 3: 'Microetch'}

            for pid, data in per_process_cal.items():
                sq_errs = np.concatenate(data['sq_errors'], axis=0)
                vars_np = np.concatenate(data['variances'], axis=0)

                proc_mse = np.mean(sq_errs)
                proc_var = np.mean(vars_np)
                proc_cal_ratio = proc_mse / proc_var if proc_var > 0 else float('inf')

                per_process_results[process_names.get(pid, f'Process {pid}')] = {
                    'mse': float(proc_mse),
                    'mean_variance': float(proc_var),
                    'calibration_ratio': float(proc_cal_ratio)
                }

            result['per_process_calibration'] = per_process_results

        return result
