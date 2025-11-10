"""
Trainer for Uncertainty Quantification Model

This trainer handles the training loop for models that predict both
mean and variance (uncertainty).
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from datetime import datetime


class UncertaintyTrainer:
    """
    Trainer for uncertainty quantification models.

    This trainer manages models that output both mean (μ) and variance (σ²),
    using Gaussian Negative Log-Likelihood as the loss function.

    Features:
    - Training loop with validation
    - Early stopping
    - Checkpoint saving
    - Metrics logging for both mean predictions and uncertainty
    - Separate tracking of prediction error and uncertainty calibration
    - Per-process metrics tracking for conditional multi-process training

    Args:
        model (nn.Module): Uncertainty model to train
        criterion (nn.Module): Loss function (typically GaussianNLLLoss)
        device (str): 'cuda' or 'cpu' (default: auto-detect)
        learning_rate (float): Learning rate (default: 0.001)
        weight_decay (float): L2 regularization strength (default: 0.0)
        conditioning_enabled (bool): If True, expect dict batch format (default: False)
        early_stopping_metric (str): Metric to use for early stopping (default: 'val_loss')
            Options:
            - 'val_loss': Aggregate validation NLL loss (default)
            - 'val_mse': Aggregate validation MSE
            - 'worst_process_mse': Worst-case MSE across all processes (recommended for multi-process)
            - 'mean_process_mse': Average MSE across all processes
    """

    def __init__(self, model, criterion, device=None, learning_rate=0.001, weight_decay=0.0,
                 conditioning_enabled=False, early_stopping_metric='val_loss'):
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
        self.early_stopping_metric = early_stopping_metric

        # Validate early stopping metric
        valid_metrics = ['val_loss', 'val_mse', 'worst_process_mse', 'mean_process_mse']
        if early_stopping_metric not in valid_metrics:
            raise ValueError(f"Invalid early_stopping_metric '{early_stopping_metric}'. "
                           f"Must be one of: {valid_metrics}")

        # Warn if using per-process metric without conditioning
        if 'process' in early_stopping_metric and not conditioning_enabled:
            print(f"WARNING: early_stopping_metric='{early_stopping_metric}' requires "
                  f"conditioning_enabled=True. Falling back to 'val_loss'.")
            self.early_stopping_metric = 'val_loss'

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.train_mse = []  # Track MSE separately for mean predictions
        self.val_mse = []
        self.best_val_loss = float('inf')
        self.best_metric_value = float('inf')  # For early stopping metric
        self.per_process_metrics = {}  # Track metrics per process_id

        print(f"UncertaintyTrainer initialized on device: {self.device}")
        print(f"Optimizer: Adam (lr={learning_rate}, weight_decay={weight_decay})")
        print(f"Loss function: {criterion.__class__.__name__}")
        print(f"Early stopping metric: {self.early_stopping_metric}")
        if conditioning_enabled:
            print(f"Conditional training enabled: will compute per-process metrics")

    def _extract_batch_data(self, batch):
        """
        Extract data from batch (supports both tuple and dict formats).

        Args:
            batch: Either (X, y) tuple or dict with keys: X, y, process_id, etc.

        Returns:
            dict: {
                'X': tensor,
                'y': tensor,
                'process_id': tensor or None,
                'timestamp': tensor or None,
                'env_continuous': dict or None,
                'env_categorical': dict or None,
                'env_masks': dict or None
            }
        """
        if isinstance(batch, dict):
            # Dict format (conditional training)
            return {
                'X': batch['X'].to(self.device),
                'y': batch['y'].to(self.device),
                'process_id': batch.get('process_id').to(self.device) if batch.get('process_id') is not None else None,
                'timestamp': batch.get('timestamp').to(self.device) if batch.get('timestamp') is not None else None,
                'env_continuous': {k: v.to(self.device) for k, v in batch['env_continuous'].items()}
                                  if batch.get('env_continuous') is not None else None,
                'env_categorical': {k: v.to(self.device) for k, v in batch['env_categorical'].items()}
                                   if batch.get('env_categorical') is not None else None,
                'env_masks': {k: v.to(self.device) for k, v in batch['env_masks'].items()}
                             if batch.get('env_masks') is not None else None,
            }
        else:
            # Tuple format (standard training)
            batch_X, batch_y = batch
            return {
                'X': batch_X.to(self.device),
                'y': batch_y.to(self.device),
                'process_id': None,
                'timestamp': None,
                'env_continuous': None,
                'env_categorical': None,
                'env_masks': None,
            }

    def train_epoch(self, train_loader):
        """
        Training for a single epoch.

        Args:
            train_loader (DataLoader): DataLoader for training set

        Returns:
            tuple: (avg_loss, avg_mse)
                - avg_loss: Average Gaussian NLL loss
                - avg_mse: Average MSE of mean predictions (for monitoring)
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0

        for batch in train_loader:
            # Extract batch data (supports both tuple and dict formats)
            data = self._extract_batch_data(batch)

            # Forward pass (with or without conditioning)
            mean, variance = self.model(
                data['X'],
                process_id=data['process_id'],
                env_continuous=data['env_continuous'],
                env_categorical=data['env_categorical'],
                timestamp=data['timestamp'],
                env_masks=data['env_masks']
            )

            # Compute Gaussian NLL loss
            loss = self.criterion(mean, variance, data['y'])

            # Compute MSE for monitoring (not used for backprop)
            with torch.no_grad():
                mse = torch.mean((mean - data['y']) ** 2)
                epoch_mse += mse.item()

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        avg_mse = epoch_mse / len(train_loader)
        return avg_loss, avg_mse

    def validate(self, val_loader, compute_per_process_metrics=None):
        """
        Model validation with optional per-process metrics tracking.

        Args:
            val_loader (DataLoader): DataLoader for validation set
            compute_per_process_metrics (bool, optional): If True, compute metrics per process.
                                                          Defaults to self.conditioning_enabled.

        Returns:
            tuple: (avg_loss, avg_mse, avg_variance, per_process_metrics)
                - avg_loss: Average Gaussian NLL loss
                - avg_mse: Average MSE of mean predictions
                - avg_variance: Average predicted variance
                - per_process_metrics: Dict {process_id: {'mse': float, 'variance': float}} or None
        """
        if compute_per_process_metrics is None:
            compute_per_process_metrics = self.conditioning_enabled

        self.model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_variance = 0.0

        # Per-process tracking
        if compute_per_process_metrics:
            process_metrics = {}  # {process_id: {'mse': [], 'variance': []}}

        with torch.no_grad():
            for batch in val_loader:
                # Extract batch data
                data = self._extract_batch_data(batch)

                # Forward pass
                mean, variance = self.model(
                    data['X'],
                    process_id=data['process_id'],
                    env_continuous=data['env_continuous'],
                    env_categorical=data['env_categorical'],
                    timestamp=data['timestamp'],
                    env_masks=data['env_masks']
                )

                # Compute losses
                loss = self.criterion(mean, variance, data['y'])
                mse = torch.mean((mean - data['y']) ** 2)

                val_loss += loss.item()
                val_mse += mse.item()
                val_variance += torch.mean(variance).item()

                # Per-process metrics
                if compute_per_process_metrics and data['process_id'] is not None:
                    process_ids = data['process_id'].cpu().numpy()
                    batch_mse = ((mean - data['y']) ** 2).cpu().numpy()
                    batch_variance = variance.cpu().numpy()

                    for i, pid in enumerate(process_ids):
                        pid = int(pid)
                        if pid not in process_metrics:
                            process_metrics[pid] = {'mse': [], 'variance': []}
                        process_metrics[pid]['mse'].append(float(batch_mse[i].mean()))
                        process_metrics[pid]['variance'].append(float(batch_variance[i].mean()))

        n_batches = len(val_loader)

        # Calculate per-process metrics for this epoch
        epoch_per_process = None
        if compute_per_process_metrics and process_metrics:
            epoch_per_process = {}
            for pid, metrics in process_metrics.items():
                avg_mse = np.mean(metrics['mse'])
                avg_var = np.mean(metrics['variance'])
                epoch_per_process[pid] = {'mse': avg_mse, 'variance': avg_var}

                # Store in history
                if pid not in self.per_process_metrics:
                    self.per_process_metrics[pid] = {'val_mse': [], 'val_variance': []}
                self.per_process_metrics[pid]['val_mse'].append(avg_mse)
                self.per_process_metrics[pid]['val_variance'].append(avg_var)

        return val_loss / n_batches, val_mse / n_batches, val_variance / n_batches, epoch_per_process

    def train(self, train_loader, val_loader, epochs=100, patience=10, save_dir='checkpoints',
              compute_per_process_metrics=None):
        """
        Complete training with early stopping.

        Args:
            train_loader (DataLoader): DataLoader for training
            val_loader (DataLoader): DataLoader for validation
            epochs (int): Maximum number of epochs
            patience (int): Epochs to wait before early stopping
            save_dir (str): Directory to save checkpoints
            compute_per_process_metrics (bool, optional): Compute per-process metrics.
                                                          Defaults to self.conditioning_enabled.

        Returns:
            dict: Dictionary with training history
        """
        if compute_per_process_metrics is None:
            compute_per_process_metrics = self.conditioning_enabled
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"START UNCERTAINTY QUANTIFICATION TRAINING")
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
            val_loss, val_mse, val_variance, per_process = self.validate(val_loader, compute_per_process_metrics)
            self.val_losses.append(val_loss)
            self.val_mse.append(val_mse)

            # Logging
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train - NLL Loss: {train_loss:.6f}, MSE: {train_mse:.6f}")
            print(f"  Val   - NLL Loss: {val_loss:.6f}, MSE: {val_mse:.6f}, "
                  f"Avg Variance: {val_variance:.6f}")

            # Print per-process metrics if available
            if per_process is not None:
                process_names = ['Laser', 'Plasma', 'Galvanic', 'Microetch']
                print(f"  Per-Process Metrics:")
                for pid in sorted(per_process.keys()):
                    pname = process_names[pid] if pid < len(process_names) else f"Process_{pid}"
                    mse = per_process[pid]['mse']
                    var = per_process[pid]['variance']
                    print(f"    {pname:10s} - MSE: {mse:.6f}, Variance: {var:.6f}")

            # Compute early stopping metric
            if self.early_stopping_metric == 'val_loss':
                current_metric = val_loss
                metric_display = f"Val NLL Loss: {current_metric:.6f}"
            elif self.early_stopping_metric == 'val_mse':
                current_metric = val_mse
                metric_display = f"Val MSE: {current_metric:.6f}"
            elif self.early_stopping_metric == 'worst_process_mse':
                if per_process is not None:
                    # Worst-case MSE across all processes
                    process_mses = [per_process[pid]['mse'] for pid in per_process.keys()]
                    current_metric = max(process_mses)
                    worst_pid = max(per_process.keys(), key=lambda pid: per_process[pid]['mse'])
                    process_names = ['Laser', 'Plasma', 'Galvanic', 'Microetch']
                    worst_name = process_names[worst_pid] if worst_pid < len(process_names) else f"Process_{worst_pid}"
                    metric_display = f"Worst Process MSE: {current_metric:.6f} ({worst_name})"
                else:
                    current_metric = val_mse
                    metric_display = f"Val MSE: {current_metric:.6f} (no per-process data)"
            elif self.early_stopping_metric == 'mean_process_mse':
                if per_process is not None:
                    # Mean MSE across all processes
                    process_mses = [per_process[pid]['mse'] for pid in per_process.keys()]
                    current_metric = np.mean(process_mses)
                    metric_display = f"Mean Process MSE: {current_metric:.6f}"
                else:
                    current_metric = val_mse
                    metric_display = f"Val MSE: {current_metric:.6f} (no per-process data)"

            # Save best model based on early stopping metric
            if current_metric < self.best_metric_value:
                self.best_metric_value = current_metric
                self.best_val_loss = val_loss  # Keep for backward compatibility
                self.save_checkpoint(save_path / 'best_model.pth', epoch, val_loss)
                print(f"  → New best model saved! ({metric_display})")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best {metric_display.split(':')[0]}: {self.best_metric_value:.6f}")
                break

        # Save training history
        self.save_training_history(save_path / 'training_history.json')

        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETED")
        print(f"Best Metric ({self.early_stopping_metric}): {self.best_metric_value:.6f}")
        print(f"Best Val NLL Loss: {self.best_val_loss:.6f}")
        print(f"{'='*70}\n")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_mse': self.train_mse,
            'val_mse': self.val_mse,
            'best_val_loss': self.best_val_loss,
            'total_epochs': len(self.train_losses)
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
            'timestamp': datetime.now().isoformat()
        }
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)

    def predict(self, X, return_uncertainty=True):
        """
        Make predictions on new data.

        Args:
            X (np.ndarray or torch.Tensor): Input data
            return_uncertainty (bool): If True, return both mean and variance

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

        with torch.no_grad():
            mean, variance = self.model(X)

        if return_uncertainty:
            return mean.cpu().numpy(), variance.cpu().numpy()
        else:
            return mean.cpu().numpy()

    def compute_calibration_metrics(self, val_loader):
        """
        Compute calibration metrics to assess uncertainty quality.

        Checks if the predicted uncertainties are well-calibrated by computing:
        - Mean squared error
        - Mean predicted variance
        - Calibration ratio (MSE / mean_variance)

        A well-calibrated model should have calibration ratio close to 1.

        Args:
            val_loader (DataLoader): Validation data loader

        Returns:
            dict: Dictionary with calibration metrics
        """
        self.model.eval()
        squared_errors = []
        variances = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                mean, variance = self.model(batch_X)

                # Compute squared errors
                sq_error = (mean - batch_y) ** 2
                squared_errors.append(sq_error.cpu().numpy())
                variances.append(variance.cpu().numpy())

        squared_errors = np.concatenate(squared_errors, axis=0)
        variances = np.concatenate(variances, axis=0)

        mse = np.mean(squared_errors)
        mean_variance = np.mean(variances)
        calibration_ratio = mse / mean_variance if mean_variance > 0 else float('inf')

        return {
            'mse': mse,
            'mean_predicted_variance': mean_variance,
            'calibration_ratio': calibration_ratio,
            'interpretation': 'Well calibrated!' if 0.8 <= calibration_ratio <= 1.2 else
                            'Under-confident (predicts too much uncertainty)' if calibration_ratio < 0.8 else
                            'Over-confident (predicts too little uncertainty)'
        }
