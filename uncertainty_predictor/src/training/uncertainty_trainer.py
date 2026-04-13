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

    def __init__(self, model, criterion, device=None, learning_rate=0.001, weight_decay=0.0,
                 grad_clip_max_norm=1.0):
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip_max_norm = grad_clip_max_norm

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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_max_norm)
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

            # Logging removed

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


class EnsembleTrainer:
    """
    Trainer for Deep Ensemble Uncertainty Quantification.

    This trainer handles training of multiple independent models in an ensemble.
    Each model is trained with a different random seed to ensure diversity.

    Features:
    - Independent training of each ensemble member
    - Different random seeds for weight initialization and data shuffling
    - Parallel-friendly design (can train models sequentially or in parallel)
    - Aggregated metrics tracking
    - Checkpoint saving for entire ensemble

    Args:
        ensemble_model (EnsembleUncertaintyPredictor): Ensemble model to train
        criterion (nn.Module): Loss function (typically GaussianNLLLoss)
        device (str): 'cuda' or 'cpu' (default: auto-detect)
        learning_rate (float): Learning rate (default: 0.001)
        weight_decay (float): L2 regularization strength (default: 0.0)
        base_seed (int): Base random seed for reproducibility (default: 42)
    """

    def __init__(self, ensemble_model, criterion, device=None, learning_rate=0.001,
                 weight_decay=0.0, base_seed=42, grad_clip_max_norm=1.0):
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.ensemble_model = ensemble_model.to(self.device)
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.base_seed = base_seed
        self.n_models = ensemble_model.n_models
        self.grad_clip_max_norm = grad_clip_max_norm

        # Create optimizer for each model
        self.optimizers = [
            optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
            for model in self.ensemble_model.models
        ]

        # Tracking for each model
        self.train_losses = [[] for _ in range(self.n_models)]
        self.val_losses = [[] for _ in range(self.n_models)]
        self.train_mse = [[] for _ in range(self.n_models)]
        self.val_mse = [[] for _ in range(self.n_models)]
        self.best_val_losses = [float('inf') for _ in range(self.n_models)]

        # Aggregated tracking
        self.ensemble_train_losses = []
        self.ensemble_val_losses = []
        self.ensemble_train_mse = []
        self.ensemble_val_mse = []

        print(f"EnsembleTrainer initialized on device: {self.device}")
        print(f"Number of models in ensemble: {self.n_models}")
        print(f"Optimizer: Adam (lr={learning_rate}, weight_decay={weight_decay})")
        print(f"Loss function: {criterion.__class__.__name__}")
        print(f"Base seed: {base_seed}")

    def _set_seed(self, model_idx):
        """Set random seed for a specific model"""
        seed = self.base_seed + model_idx * 1000
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return seed

    def train_single_model(self, model_idx, train_loader, val_loader, epochs, patience):
        """
        Train a single model in the ensemble.

        Args:
            model_idx (int): Index of model to train
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            epochs (int): Maximum epochs
            patience (int): Early stopping patience

        Returns:
            dict: Training history for this model
        """
        model = self.ensemble_model.models[model_idx]
        optimizer = self.optimizers[model_idx]

        epochs_without_improvement = 0
        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Training
            model.train()
            epoch_loss = 0.0
            epoch_mse = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                mean, variance = model(batch_X)
                loss = self.criterion(mean, variance, batch_y)

                with torch.no_grad():
                    mse = torch.mean((mean - batch_y) ** 2)
                    epoch_mse += mse.item()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.grad_clip_max_norm)
                optimizer.step()

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            avg_train_mse = epoch_mse / len(train_loader)

            self.train_losses[model_idx].append(avg_train_loss)
            self.train_mse[model_idx].append(avg_train_mse)

            # Validation
            model.eval()
            val_loss = 0.0
            val_mse = 0.0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    mean, variance = model(batch_X)
                    loss = self.criterion(mean, variance, batch_y)
                    mse = torch.mean((mean - batch_y) ** 2)

                    val_loss += loss.item()
                    val_mse += mse.item()

            avg_val_loss = val_loss / len(val_loader)
            avg_val_mse = val_mse / len(val_loader)

            self.val_losses[model_idx].append(avg_val_loss)
            self.val_mse[model_idx].append(avg_val_mse)

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.best_val_losses[model_idx] = best_val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                break

        return {
            'train_losses': self.train_losses[model_idx],
            'val_losses': self.val_losses[model_idx],
            'train_mse': self.train_mse[model_idx],
            'val_mse': self.val_mse[model_idx],
            'best_val_loss': best_val_loss,
            'total_epochs': len(self.train_losses[model_idx])
        }

    def train(self, train_loader, val_loader, epochs=100, patience=10, save_dir='checkpoints'):
        """
        Train all models in the ensemble.

        Each model is trained independently with a different random seed.

        Args:
            train_loader (DataLoader): DataLoader for training
            val_loader (DataLoader): DataLoader for validation
            epochs (int): Maximum number of epochs per model
            patience (int): Epochs to wait before early stopping
            save_dir (str): Directory to save checkpoints

        Returns:
            dict: Dictionary with aggregated training history
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"START DEEP ENSEMBLE TRAINING")
        print(f"{'='*70}")
        print(f"Models in ensemble: {self.n_models}")
        print(f"Max epochs per model: {epochs}")
        print(f"Early stopping patience: {patience}")
        print(f"Checkpoint directory: {save_path}")
        print(f"{'='*70}\n")

        all_histories = []

        for model_idx in range(self.n_models):
            seed = self._set_seed(model_idx)
            print(f"\n{'─'*50}")
            print(f"Training Model {model_idx + 1}/{self.n_models} (seed={seed})")
            print(f"{'─'*50}")

            # Re-initialize model weights with new seed
            self._reinitialize_model(model_idx)

            history = self.train_single_model(
                model_idx, train_loader, val_loader, epochs, patience
            )
            all_histories.append(history)

            print(f"Model {model_idx + 1}: Best Val Loss = {history['best_val_loss']:.6f}, "
                  f"Epochs = {history['total_epochs']}")

        # Compute aggregated metrics (average across models per epoch)
        self._compute_ensemble_metrics()

        # Save ensemble checkpoint
        self.save_checkpoint(save_path / 'best_model.pth')

        # Save training history
        self.save_training_history(save_path / 'training_history.json')

        print(f"\n{'='*70}")
        print(f"ENSEMBLE TRAINING COMPLETED")
        print(f"{'='*70}")
        print(f"Individual model best losses: {[f'{l:.4f}' for l in self.best_val_losses]}")
        print(f"Mean best val loss: {np.mean(self.best_val_losses):.6f}")
        print(f"{'='*70}\n")

        return {
            'train_losses': self.ensemble_train_losses,
            'val_losses': self.ensemble_val_losses,
            'train_mse': self.ensemble_train_mse,
            'val_mse': self.ensemble_val_mse,
            'best_val_loss': float(np.mean(self.best_val_losses)),
            'total_epochs': max(len(h['train_losses']) for h in all_histories),
            'individual_histories': all_histories,
            'individual_best_losses': self.best_val_losses
        }

    def _reinitialize_model(self, model_idx):
        """Reinitialize model weights with current random state"""
        model = self.ensemble_model.models[model_idx]
        for layer in model.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        # Recreate optimizer after reinitialization
        self.optimizers[model_idx] = optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

    def _compute_ensemble_metrics(self):
        """Compute aggregated ensemble metrics by averaging across models"""
        # Find the maximum number of epochs across all models
        max_epochs = max(len(losses) for losses in self.train_losses)

        for epoch in range(max_epochs):
            train_loss_avg = []
            val_loss_avg = []
            train_mse_avg = []
            val_mse_avg = []

            for model_idx in range(self.n_models):
                if epoch < len(self.train_losses[model_idx]):
                    train_loss_avg.append(self.train_losses[model_idx][epoch])
                    val_loss_avg.append(self.val_losses[model_idx][epoch])
                    train_mse_avg.append(self.train_mse[model_idx][epoch])
                    val_mse_avg.append(self.val_mse[model_idx][epoch])

            self.ensemble_train_losses.append(np.mean(train_loss_avg))
            self.ensemble_val_losses.append(np.mean(val_loss_avg))
            self.ensemble_train_mse.append(np.mean(train_mse_avg))
            self.ensemble_val_mse.append(np.mean(val_mse_avg))

    def save_checkpoint(self, filepath):
        """Save ensemble checkpoint"""
        checkpoint = {
            'n_models': self.n_models,
            'ensemble_state_dict': self.ensemble_model.state_dict(),
            'optimizer_state_dicts': [opt.state_dict() for opt in self.optimizers],
            'best_val_losses': self.best_val_losses,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_mse': self.train_mse,
            'val_mse': self.val_mse,
            'ensemble_train_losses': self.ensemble_train_losses,
            'ensemble_val_losses': self.ensemble_val_losses,
            'base_seed': self.base_seed
        }
        torch.save(checkpoint, filepath)
        print(f"Ensemble checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath):
        """Load ensemble checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.ensemble_model.load_state_dict(checkpoint['ensemble_state_dict'])

        for i, opt_state in enumerate(checkpoint['optimizer_state_dicts']):
            self.optimizers[i].load_state_dict(opt_state)

        self.best_val_losses = checkpoint.get('best_val_losses', [float('inf')] * self.n_models)
        self.train_losses = checkpoint.get('train_losses', [[] for _ in range(self.n_models)])
        self.val_losses = checkpoint.get('val_losses', [[] for _ in range(self.n_models)])
        self.train_mse = checkpoint.get('train_mse', [[] for _ in range(self.n_models)])
        self.val_mse = checkpoint.get('val_mse', [[] for _ in range(self.n_models)])
        self.ensemble_train_losses = checkpoint.get('ensemble_train_losses', [])
        self.ensemble_val_losses = checkpoint.get('ensemble_val_losses', [])

        print(f"Ensemble checkpoint loaded from: {filepath}")
        return checkpoint

    def save_training_history(self, filepath):
        """Save training history to JSON"""
        history = {
            'n_models': self.n_models,
            'train_losses': self.ensemble_train_losses,
            'val_losses': self.ensemble_val_losses,
            'train_mse': self.ensemble_train_mse,
            'val_mse': self.ensemble_val_mse,
            'best_val_loss': float(np.mean(self.best_val_losses)),
            'individual_best_losses': [float(l) for l in self.best_val_losses],
            'individual_train_losses': self.train_losses,
            'individual_val_losses': self.val_losses,
            'timestamp': datetime.now().isoformat()
        }
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)

    def predict(self, X, return_uncertainty=True):
        """
        Make predictions using the ensemble.

        Args:
            X (np.ndarray or torch.Tensor): Input data
            return_uncertainty (bool): If True, return both mean and variance

        Returns:
            If return_uncertainty=True:
                tuple: (mean_predictions, total_variance, aleatoric, epistemic)
            If return_uncertainty=False:
                np.ndarray: Mean predictions only
        """
        self.ensemble_model.eval()

        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)

        X = X.to(self.device)

        with torch.no_grad():
            mean, total_var, aleatoric, epistemic = self.ensemble_model.predict_with_decomposition(X)

        if return_uncertainty:
            return (
                mean.cpu().numpy(),
                total_var.cpu().numpy(),
                aleatoric.cpu().numpy(),
                epistemic.cpu().numpy()
            )
        else:
            return mean.cpu().numpy()

    def compute_calibration_metrics(self, val_loader):
        """
        Compute calibration metrics for the ensemble.

        Args:
            val_loader (DataLoader): Validation data loader

        Returns:
            dict: Dictionary with calibration metrics including uncertainty decomposition
        """
        self.ensemble_model.eval()
        squared_errors = []
        total_variances = []
        aleatorics = []
        epistemics = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                mean, total_var, aleatoric, epistemic = \
                    self.ensemble_model.predict_with_decomposition(batch_X)

                sq_error = (mean - batch_y) ** 2
                squared_errors.append(sq_error.cpu().numpy())
                total_variances.append(total_var.cpu().numpy())
                aleatorics.append(aleatoric.cpu().numpy())
                epistemics.append(epistemic.cpu().numpy())

        squared_errors = np.concatenate(squared_errors, axis=0)
        total_variances = np.concatenate(total_variances, axis=0)
        aleatorics = np.concatenate(aleatorics, axis=0)
        epistemics = np.concatenate(epistemics, axis=0)

        mse = np.mean(squared_errors)
        mean_total_var = np.mean(total_variances)
        mean_aleatoric = np.mean(aleatorics)
        mean_epistemic = np.mean(epistemics)
        calibration_ratio = mse / mean_total_var if mean_total_var > 0 else float('inf')

        return {
            'mse': mse,
            'mean_predicted_variance': mean_total_var,
            'mean_aleatoric': mean_aleatoric,
            'mean_epistemic': mean_epistemic,
            'epistemic_ratio': mean_epistemic / mean_total_var if mean_total_var > 0 else 0,
            'calibration_ratio': calibration_ratio,
            'interpretation': 'Well calibrated!' if 0.8 <= calibration_ratio <= 1.2 else
                            'Under-confident (predicts too much uncertainty)' if calibration_ratio < 0.8 else
                            'Over-confident (predicts too little uncertainty)'
        }


class SWAGTrainer:
    """
    Trainer for SWAG (Stochastic Weight Averaging - Gaussian) Uncertainty Quantification.

    SWAG training consists of two phases:
    1. Pre-training phase: Standard training with SGD/Adam
    2. SWA collection phase: Continue training with constant/cyclic LR while
       collecting weight statistics at the end of each epoch

    The collected statistics form a Gaussian approximation to the posterior
    distribution over weights, which enables Bayesian inference at test time.

    Reference:
        Maddox et al. (2019) "A Simple Baseline for Bayesian Uncertainty in Deep Learning"

    Args:
        swag_model (SWAGUncertaintyPredictor): SWAG model to train
        criterion (nn.Module): Loss function (typically GaussianNLLLoss)
        device (str): 'cuda' or 'cpu' (default: auto-detect)
        learning_rate (float): Learning rate for pre-training (default: 0.001)
        swa_learning_rate (float): Learning rate for SWA phase (default: 0.01)
        weight_decay (float): L2 regularization strength (default: 0.0)
        swa_start_epoch (float): Fraction of training to do before SWA (default: 0.5)
        swa_freq (int): Collect weights every swa_freq epochs (default: 1)

    Example:
        >>> trainer = SWAGTrainer(swag_model, criterion, swa_start_epoch=0.5)
        >>> history = trainer.train(train_loader, val_loader, epochs=100)
        >>> mean, var, aleatoric, epistemic = trainer.predict(X, return_uncertainty=True)
    """

    def __init__(self, swag_model, criterion, device=None, learning_rate=0.001,
                 swa_learning_rate=0.01, weight_decay=0.0, swa_start_epoch=0.5, swa_freq=1,
                 min_samples=20, grad_clip_max_norm=1.0):
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.swag_model = swag_model.to(self.device)
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.swa_learning_rate = swa_learning_rate
        self.weight_decay = weight_decay
        self.swa_start_epoch = swa_start_epoch  # Fraction (0.5 = start SWA at 50% of training)
        self.swa_freq = swa_freq
        self.min_samples = min_samples  # Minimum weight samples before training can stop
        self.grad_clip_max_norm = grad_clip_max_norm

        # Setup optimizer for base model
        self.optimizer = optim.Adam(
            self.swag_model.base_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.train_mse = []
        self.val_mse = []
        self.best_val_loss = float('inf')
        self.swa_start_actual = None  # Actual epoch when SWA started

        print(f"SWAGTrainer initialized on device: {self.device}")
        print(f"Pre-training LR: {learning_rate}, SWA LR: {swa_learning_rate}")
        print(f"SWA start: {swa_start_epoch*100:.0f}% of training")
        print(f"Weight collection frequency: every {swa_freq} epoch(s)")
        print(f"Minimum samples before stopping: {min_samples}")
        print(f"Max rank for covariance: {swag_model.max_rank}")
        print(f"Loss function: {criterion.__class__.__name__}")

    def _set_learning_rate(self, lr):
        """Set learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train_epoch(self, train_loader):
        """Training for a single epoch."""
        self.swag_model.base_model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass through base model
            mean, variance = self.swag_model.base_model(batch_X)

            # Compute loss
            loss = self.criterion(mean, variance, batch_y)

            # Compute MSE for monitoring
            with torch.no_grad():
                mse = torch.mean((mean - batch_y) ** 2)
                epoch_mse += mse.item()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.swag_model.base_model.parameters(), max_norm=self.grad_clip_max_norm)
            self.optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        avg_mse = epoch_mse / len(train_loader)
        return avg_loss, avg_mse

    def validate(self, val_loader):
        """Model validation."""
        self.swag_model.base_model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_variance = 0.0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                mean, variance = self.swag_model.base_model(batch_X)

                loss = self.criterion(mean, variance, batch_y)
                mse = torch.mean((mean - batch_y) ** 2)

                val_loss += loss.item()
                val_mse += mse.item()
                val_variance += torch.mean(variance).item()

        n_batches = len(val_loader)
        return val_loss / n_batches, val_mse / n_batches, val_variance / n_batches

    def train(self, train_loader, val_loader, epochs=100, patience=10, save_dir='checkpoints'):
        """
        Complete SWAG training with two phases.

        Phase 1: Pre-training with standard learning rate
        Phase 2: SWA collection with constant learning rate

        Args:
            train_loader (DataLoader): DataLoader for training
            val_loader (DataLoader): DataLoader for validation
            epochs (int): Total number of epochs
            patience (int): Early stopping patience (only for pre-training phase)
            save_dir (str): Directory to save checkpoints

        Returns:
            dict: Dictionary with training history
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Calculate when to start SWA
        swa_start = int(epochs * self.swa_start_epoch)
        self.swa_start_actual = swa_start

        print(f"\n{'='*70}")
        print(f"START SWAG TRAINING")
        print(f"{'='*70}")
        print(f"Total epochs: {epochs}")
        print(f"Pre-training phase: epochs 1-{swa_start}")
        print(f"SWA collection phase: epochs {swa_start+1}-{epochs}")
        print(f"Minimum SWA samples required: {self.min_samples}")
        print(f"Early stopping patience: {patience} (pre-training only)")
        print(f"Checkpoint directory: {save_path}")
        print(f"{'='*70}\n")

        epochs_without_improvement = 0
        in_swa_phase = False
        epoch = 0

        # Continue training until we have collected enough samples
        while True:
            # Check if we should switch to SWA phase (scheduled start)
            if epoch >= swa_start and not in_swa_phase:
                in_swa_phase = True
                self.swa_start_actual = epoch
                self._set_learning_rate(self.swa_learning_rate)
                print(f"\n{'─'*50}")
                print(f"Entering SWA collection phase (epoch {epoch+1})")
                print(f"Learning rate set to: {self.swa_learning_rate}")
                print(f"{'─'*50}\n")

            # Training
            train_loss, train_mse = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_mse.append(train_mse)

            # Validation
            val_loss, val_mse, val_variance = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_mse.append(val_mse)

            # Collect weights during SWA phase
            if in_swa_phase:
                # Use swa_start_actual to track collection frequency from when SWA actually started
                swa_epoch = epoch - self.swa_start_actual if self.swa_start_actual is not None else epoch - swa_start
                if swa_epoch % self.swa_freq == 0:
                    self.swag_model.collect_model()
                    n_collected = self.swag_model.n_models_collected.item()
                    print(f"  [SWA] Epoch {epoch+1}: collected sample {n_collected}/{self.min_samples} (val_loss: {val_loss:.6f})")

            # Early stopping (only during pre-training phase)
            if not in_swa_phase:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(save_path / 'best_pretrain_model.pth', epoch, val_loss)
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    # Instead of stopping, switch to SWA phase
                    in_swa_phase = True
                    self._set_learning_rate(self.swa_learning_rate)
                    self.swa_start_actual = epoch + 1
                    epochs_without_improvement = 0
                    print(f"\n{'─'*50}")
                    print(f"EARLY STOPPING triggered at epoch {epoch+1}")
                    print(f"Switching to SWA collection phase")
                    print(f"Learning rate set to: {self.swa_learning_rate}")
                    print(f"Will collect minimum {self.min_samples} weight samples")
                    print(f"{'─'*50}\n")

            epoch += 1

            # Check if we can stop training
            n_samples_collected = self.swag_model.n_models_collected.item()

            if in_swa_phase and n_samples_collected >= self.min_samples:
                # Enough samples collected — stop training
                if epoch < epochs:
                    print(f"\n{'─'*50}")
                    print(f"Collected {n_samples_collected}/{self.min_samples} SWA samples — stopping early at epoch {epoch}")
                    print(f"{'─'*50}\n")
                break

            if epoch >= epochs and not in_swa_phase:
                # Reached max epochs without entering SWA — force SWA phase
                in_swa_phase = True
                self._set_learning_rate(self.swa_learning_rate)
                self.swa_start_actual = epoch
                print(f"\n{'─'*50}")
                print(f"Reached {epochs} epochs without SWA phase — starting SWA collection now")
                print(f"{'─'*50}\n")

            # Safety check: don't run forever (max 2x original epochs)
            if epoch >= epochs * 2:
                print(f"\nWarning: Reached maximum epochs ({epoch}), stopping with {n_samples_collected} samples")
                break

        # Save final SWAG model
        self.save_checkpoint(save_path / 'best_model.pth', epoch, self.val_losses[-1])

        # Save training history
        self.save_training_history(save_path / 'training_history.json')

        # Print SWAG statistics
        swag_stats = self.swag_model.get_swag_stats()
        print(f"\n{'='*70}")
        print(f"SWAG TRAINING COMPLETED")
        print(f"{'='*70}")
        print(f"Models collected: {swag_stats['n_models_collected']}")
        print(f"Low-rank dimension: {swag_stats['max_rank_used']}")
        print(f"Mean weight norm: {swag_stats['mean_weight_norm']:.4f}")
        print(f"Mean variance: {swag_stats['mean_variance']:.6f}")
        print(f"Best pre-training val loss: {self.best_val_loss:.6f}")
        print(f"{'='*70}\n")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_mse': self.train_mse,
            'val_mse': self.val_mse,
            'best_val_loss': self.best_val_loss,
            'total_epochs': len(self.train_losses),
            'swa_start_epoch': self.swa_start_actual,
            'models_collected': swag_stats['n_models_collected']
        }

    def save_checkpoint(self, filepath, epoch, val_loss):
        """Save a model checkpoint including SWAG statistics."""
        torch.save({
            'epoch': epoch,
            'base_model_state_dict': self.swag_model.base_model.state_dict(),
            'swa_mean': self.swag_model.swa_mean,
            'swa_sq_mean': self.swag_model.swa_sq_mean,
            'deviation_matrix': self.swag_model.deviation_matrix,
            'n_models_collected': self.swag_model.n_models_collected,
            'deviation_idx': self.swag_model.deviation_idx,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_mse': self.train_mse,
            'val_mse': self.val_mse,
        }, filepath)

    def load_checkpoint(self, filepath):
        """Load a model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.swag_model.base_model.load_state_dict(checkpoint['base_model_state_dict'])
        self.swag_model.swa_mean.copy_(checkpoint['swa_mean'])
        self.swag_model.swa_sq_mean.copy_(checkpoint['swa_sq_mean'])
        self.swag_model.deviation_matrix.copy_(checkpoint['deviation_matrix'])
        self.swag_model.n_models_collected.copy_(checkpoint['n_models_collected'])
        self.swag_model.deviation_idx.copy_(checkpoint['deviation_idx'])
        # ``swag_collected`` is now a property derived from ``n_models_collected``,
        # so it is restored automatically by the buffer copy above. Old checkpoints
        # that still carry an explicit 'swag_collected' key are ignored.

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_mse = checkpoint.get('train_mse', [])
        self.val_mse = checkpoint.get('val_mse', [])

        print(f"SWAG checkpoint loaded from: {filepath}")
        return checkpoint

    def save_training_history(self, filepath):
        """Save training history to JSON."""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_mse': self.train_mse,
            'val_mse': self.val_mse,
            'best_val_loss': float(self.best_val_loss),
            'swa_start_epoch': self.swa_start_actual,
            'models_collected': self.swag_model.n_models_collected.item(),
            'timestamp': datetime.now().isoformat()
        }
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)

    def predict(self, X, return_uncertainty=True, n_samples=30):
        """
        Make predictions using SWAG sampling.

        Args:
            X (np.ndarray or torch.Tensor): Input data
            return_uncertainty (bool): If True, return full uncertainty decomposition
            n_samples (int): Number of weight samples for prediction

        Returns:
            If return_uncertainty=True:
                tuple: (mean, total_variance, aleatoric, epistemic)
            If return_uncertainty=False:
                np.ndarray: Mean predictions only
        """
        self.swag_model.eval()

        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)

        X = X.to(self.device)

        with torch.no_grad():
            mean, total_var, aleatoric, epistemic = \
                self.swag_model.predict_with_decomposition(X, n_samples=n_samples)

        if return_uncertainty:
            return (
                mean.cpu().numpy(),
                total_var.cpu().numpy(),
                aleatoric.cpu().numpy(),
                epistemic.cpu().numpy()
            )
        else:
            return mean.cpu().numpy()

    def compute_calibration_metrics(self, val_loader, n_samples=30):
        """
        Compute calibration metrics for the SWAG model.

        Args:
            val_loader (DataLoader): Validation data loader
            n_samples (int): Number of weight samples

        Returns:
            dict: Dictionary with calibration metrics
        """
        self.swag_model.eval()
        squared_errors = []
        total_variances = []
        aleatorics = []
        epistemics = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                mean, total_var, aleatoric, epistemic = \
                    self.swag_model.predict_with_decomposition(batch_X, n_samples=n_samples)

                sq_error = (mean - batch_y) ** 2
                squared_errors.append(sq_error.cpu().numpy())
                total_variances.append(total_var.cpu().numpy())
                aleatorics.append(aleatoric.cpu().numpy())
                epistemics.append(epistemic.cpu().numpy())

        squared_errors = np.concatenate(squared_errors, axis=0)
        total_variances = np.concatenate(total_variances, axis=0)
        aleatorics = np.concatenate(aleatorics, axis=0)
        epistemics = np.concatenate(epistemics, axis=0)

        mse = np.mean(squared_errors)
        mean_total_var = np.mean(total_variances)
        mean_aleatoric = np.mean(aleatorics)
        mean_epistemic = np.mean(epistemics)
        calibration_ratio = mse / mean_total_var if mean_total_var > 0 else float('inf')

        return {
            'mse': mse,
            'mean_predicted_variance': mean_total_var,
            'mean_aleatoric': mean_aleatoric,
            'mean_epistemic': mean_epistemic,
            'epistemic_ratio': mean_epistemic / mean_total_var if mean_total_var > 0 else 0,
            'calibration_ratio': calibration_ratio,
            'interpretation': 'Well calibrated!' if 0.8 <= calibration_ratio <= 1.2 else
                            'Under-confident (predicts too much uncertainty)' if calibration_ratio < 0.8 else
                            'Over-confident (predicts too little uncertainty)'
        }
