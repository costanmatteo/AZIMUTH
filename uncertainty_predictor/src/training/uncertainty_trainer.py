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

    Args:
        model (nn.Module): Uncertainty model to train
        criterion (nn.Module): Loss function (typically GaussianNLLLoss)
        device (str): 'cuda' or 'cpu' (default: auto-detect)
        learning_rate (float): Learning rate (default: 0.001)
        weight_decay (float): L2 regularization strength (default: 0.0)
    """

    def __init__(self, model, criterion, device=None, learning_rate=0.001, weight_decay=0.0):
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

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

        print(f"UncertaintyTrainer initialized on device: {self.device}")
        print(f"Optimizer: Adam (lr={learning_rate}, weight_decay={weight_decay})")
        print(f"Loss function: {criterion.__class__.__name__}")

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

        for batch_X, batch_y in train_loader:
            # Move data to device
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            mean, variance = self.model(batch_X)

            # Compute Gaussian NLL loss
            loss = self.criterion(mean, variance, batch_y)

            # Compute MSE for monitoring (not used for backprop)
            with torch.no_grad():
                mse = torch.mean((mean - batch_y) ** 2)
                epoch_mse += mse.item()

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        avg_mse = epoch_mse / len(train_loader)
        return avg_loss, avg_mse

    def validate(self, val_loader):
        """
        Model validation.

        Args:
            val_loader (DataLoader): DataLoader for validation set

        Returns:
            tuple: (avg_loss, avg_mse, avg_variance)
                - avg_loss: Average Gaussian NLL loss
                - avg_mse: Average MSE of mean predictions
                - avg_variance: Average predicted variance
        """
        self.model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_variance = 0.0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                mean, variance = self.model(batch_X)

                # Compute losses
                loss = self.criterion(mean, variance, batch_y)
                mse = torch.mean((mean - batch_y) ** 2)

                val_loss += loss.item()
                val_mse += mse.item()
                val_variance += torch.mean(variance).item()

        n_batches = len(val_loader)
        return val_loss / n_batches, val_mse / n_batches, val_variance / n_batches

    def train(self, train_loader, val_loader, epochs=100, patience=10, save_dir='checkpoints'):
        """
        Complete training with early stopping.

        Args:
            train_loader (DataLoader): DataLoader for training
            val_loader (DataLoader): DataLoader for validation
            epochs (int): Maximum number of epochs
            patience (int): Epochs to wait before early stopping
            save_dir (str): Directory to save checkpoints

        Returns:
            dict: Dictionary with training history
        """
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
            val_loss, val_mse, val_variance = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_mse.append(val_mse)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(save_path / 'best_model.pth', epoch, val_loss)
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
