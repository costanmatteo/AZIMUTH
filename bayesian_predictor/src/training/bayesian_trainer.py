"""
Trainer for Bayesian Neural Networks

This module implements a specialized trainer for Bayesian neural networks
that handles:
- ELBO loss optimization (NLL + KL divergence)
- KL weight scheduling (annealing)
- Monte Carlo sampling for predictions
- Uncertainty quantification
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Optional, Dict, List, Tuple


class BayesianTrainer:
    """
    Trainer for Bayesian Neural Networks with variational inference.

    This trainer optimizes the ELBO (Evidence Lower Bound):
        ELBO = -log p(y|x,w) + β * KL[q(w|θ)||p(w)]

    Features:
    - ELBO loss with KL weight scheduling
    - Training with weight sampling
    - Validation with multiple Monte Carlo samples
    - Early stopping
    - Checkpoint saving
    - Uncertainty tracking

    Args:
        model: Bayesian neural network model
        loss_fn: ELBO loss function
        device (str): 'cuda' or 'cpu' (default: auto-detect)
        learning_rate (float): Learning rate (default: 0.001)
        weight_decay (float): L2 regularization (default: 1e-5)
        kl_schedule (str): KL weight schedule ('constant', 'linear', 'cyclical')
        kl_warmup_epochs (int): Epochs for KL warmup (default: 10)
        n_train_samples (int): Monte Carlo samples during training (default: 1)
        n_val_samples (int): Monte Carlo samples during validation (default: 10)
    """

    def __init__(
        self,
        model,
        loss_fn,
        device: Optional[str] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        kl_schedule: str = 'linear',
        kl_warmup_epochs: int = 10,
        n_train_samples: int = 1,
        n_val_samples: int = 10
    ):
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # KL scheduling parameters
        self.kl_schedule = kl_schedule
        self.kl_warmup_epochs = kl_warmup_epochs
        self.base_kl_weight = loss_fn.kl_weight

        # Monte Carlo sampling
        self.n_train_samples = n_train_samples
        self.n_val_samples = n_val_samples

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Tracking
        self.train_losses = []
        self.train_nlls = []
        self.train_kls = []
        self.val_losses = []
        self.val_nlls = []
        self.val_kls = []
        self.val_uncertainties = []
        self.best_val_loss = float('inf')
        self.current_kl_weight = 0.0 if kl_schedule != 'constant' else self.base_kl_weight

        print(f"Bayesian Trainer initialized on device: {self.device}")
        print(f"Optimizer: Adam (lr={learning_rate}, weight_decay={weight_decay})")
        print(f"KL schedule: {kl_schedule} (base weight: {self.base_kl_weight:.6f})")
        print(f"KL warmup epochs: {kl_warmup_epochs}")
        print(f"Train samples: {n_train_samples}, Val samples: {n_val_samples}")

    def get_kl_weight(self, epoch: int, total_epochs: int) -> float:
        """
        Compute KL weight for current epoch based on schedule.

        Args:
            epoch (int): Current epoch
            total_epochs (int): Total number of epochs

        Returns:
            float: KL weight for this epoch
        """
        if self.kl_schedule == 'constant':
            return self.base_kl_weight

        elif self.kl_schedule == 'linear':
            # Linear warmup from 0 to base_kl_weight
            if epoch < self.kl_warmup_epochs:
                return self.base_kl_weight * (epoch / self.kl_warmup_epochs)
            else:
                return self.base_kl_weight

        elif self.kl_schedule == 'cyclical':
            # Cyclical annealing (helps explore multiple modes)
            cycle_length = max(10, total_epochs // 4)
            cycle_position = epoch % cycle_length
            return self.base_kl_weight * (cycle_position / cycle_length)

        else:
            return self.base_kl_weight

    def train_epoch(self, train_loader: DataLoader, epoch: int, total_epochs: int) -> Tuple[float, float, float]:
        """
        Training for a single epoch.

        Args:
            train_loader: DataLoader for training set
            epoch: Current epoch number
            total_epochs: Total number of epochs

        Returns:
            tuple: (avg_loss, avg_nll, avg_kl)
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_nll = 0.0
        epoch_kl = 0.0

        # Update KL weight for this epoch
        self.current_kl_weight = self.get_kl_weight(epoch, total_epochs)
        self.loss_fn.kl_weight = self.current_kl_weight

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            # Monte Carlo sampling during training
            batch_loss = 0.0
            batch_nll = 0.0
            batch_kl = 0.0

            for _ in range(self.n_train_samples):
                # Forward pass with weight sampling
                predictions = self.model(batch_X, sample=True)

                # Compute KL divergence
                kl_div = self.model.kl_divergence()

                # Compute ELBO loss
                loss, nll, kl = self.loss_fn(predictions, batch_y, kl_div)

                batch_loss += loss
                batch_nll += nll.item() if torch.is_tensor(nll) else nll
                batch_kl += kl.item() if torch.is_tensor(kl) else kl

            # Average over samples
            batch_loss = batch_loss / self.n_train_samples
            batch_nll = batch_nll / self.n_train_samples
            batch_kl = batch_kl / self.n_train_samples

            # Backward pass and optimization
            self.optimizer.zero_grad()
            batch_loss.backward()

            # Gradient clipping (helps with stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            epoch_loss += batch_loss.item()
            epoch_nll += batch_nll
            epoch_kl += batch_kl

        avg_loss = epoch_loss / len(train_loader)
        avg_nll = epoch_nll / len(train_loader)
        avg_kl = epoch_kl / len(train_loader)

        return avg_loss, avg_nll, avg_kl

    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float, float]:
        """
        Model validation with uncertainty estimation.

        Args:
            val_loader: DataLoader for validation set

        Returns:
            tuple: (avg_loss, avg_nll, avg_kl, avg_uncertainty)
        """
        self.model.eval()
        val_loss = 0.0
        val_nll = 0.0
        val_kl = 0.0
        val_uncertainty = 0.0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Monte Carlo sampling for uncertainty estimation
                predictions_list = []
                for _ in range(self.n_val_samples):
                    pred = self.model(batch_X, sample=True)
                    predictions_list.append(pred)

                predictions = torch.stack(predictions_list)  # (n_samples, batch, output)

                # Use mean prediction for loss
                mean_pred = predictions.mean(dim=0)

                # Compute uncertainty (standard deviation across samples)
                uncertainty = predictions.std(dim=0).mean().item()

                # Compute KL divergence
                kl_div = self.model.kl_divergence()

                # Compute ELBO loss on mean prediction
                loss, nll, kl = self.loss_fn(mean_pred, batch_y, kl_div)

                val_loss += loss.item()
                val_nll += nll.item() if torch.is_tensor(nll) else nll
                val_kl += kl.item() if torch.is_tensor(kl) else kl
                val_uncertainty += uncertainty

        avg_loss = val_loss / len(val_loader)
        avg_nll = val_nll / len(val_loader)
        avg_kl = val_kl / len(val_loader)
        avg_uncertainty = val_uncertainty / len(val_loader)

        return avg_loss, avg_nll, avg_kl, avg_uncertainty

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 15,
        save_dir: str = 'checkpoints'
    ) -> Dict:
        """
        Complete training with early stopping.

        Args:
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
            epochs: Maximum number of epochs
            patience: Epochs to wait before early stopping
            save_dir: Directory to save checkpoints

        Returns:
            dict: Training history
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"START BAYESIAN TRAINING")
        print(f"{'='*70}")
        print(f"Epochs: {epochs}")
        print(f"Early stopping patience: {patience}")
        print(f"Checkpoint directory: {save_path}")
        print(f"{'='*70}\n")

        epochs_without_improvement = 0

        for epoch in range(epochs):
            # Training
            train_loss, train_nll, train_kl = self.train_epoch(train_loader, epoch, epochs)
            self.train_losses.append(train_loss)
            self.train_nlls.append(train_nll)
            self.train_kls.append(train_kl)

            # Validation
            val_loss, val_nll, val_kl, val_uncertainty = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_nlls.append(val_nll)
            self.val_kls.append(val_kl)
            self.val_uncertainties.append(val_uncertainty)

            # Logging
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train - Loss: {train_loss:.6f} | NLL: {train_nll:.6f} | KL: {train_kl:.6f}")
            print(f"  Val   - Loss: {val_loss:.6f} | NLL: {val_nll:.6f} | KL: {val_kl:.6f}")
            print(f"  Val Uncertainty: {val_uncertainty:.6f} | KL Weight: {self.current_kl_weight:.6f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(save_path / 'best_model.pth', epoch, val_loss)
                print(f"  → New best model saved! (Val Loss: {val_loss:.6f})")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best Val Loss: {self.best_val_loss:.6f}")
                break

        # Save training history
        self.save_training_history(save_path / 'training_history.json')

        print(f"\n{'='*70}")
        print(f"BAYESIAN TRAINING COMPLETED")
        print(f"Best Val Loss: {self.best_val_loss:.6f}")
        print(f"{'='*70}\n")

        return {
            'train_losses': self.train_losses,
            'train_nlls': self.train_nlls,
            'train_kls': self.train_kls,
            'val_losses': self.val_losses,
            'val_nlls': self.val_nlls,
            'val_kls': self.val_kls,
            'val_uncertainties': self.val_uncertainties,
            'best_val_loss': self.best_val_loss,
            'total_epochs': len(self.train_losses)
        }

    def save_checkpoint(self, filepath: Path, epoch: int, val_loss: float):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_nlls': self.train_nlls,
            'train_kls': self.train_kls,
            'val_nlls': self.val_nlls,
            'val_kls': self.val_kls,
            'val_uncertainties': self.val_uncertainties,
        }, filepath)

    def load_checkpoint(self, filepath: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_nlls = checkpoint.get('train_nlls', [])
        self.train_kls = checkpoint.get('train_kls', [])
        self.val_nlls = checkpoint.get('val_nlls', [])
        self.val_kls = checkpoint.get('val_kls', [])
        self.val_uncertainties = checkpoint.get('val_uncertainties', [])
        print(f"Checkpoint loaded from: {filepath}")
        return checkpoint

    def save_training_history(self, filepath: Path):
        """Save training history to JSON"""
        history = {
            'train_losses': self.train_losses,
            'train_nlls': self.train_nlls,
            'train_kls': self.train_kls,
            'val_losses': self.val_losses,
            'val_nlls': self.val_nlls,
            'val_kls': self.val_kls,
            'val_uncertainties': self.val_uncertainties,
            'best_val_loss': float(self.best_val_loss),
            'timestamp': datetime.now().isoformat()
        }
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)

    def predict(self, X: np.ndarray, n_samples: int = 100) -> np.ndarray:
        """
        Make point predictions (mean over samples).

        Args:
            X: Input data
            n_samples: Number of Monte Carlo samples

        Returns:
            np.ndarray: Mean predictions
        """
        results = self.predict_with_uncertainty(X, n_samples)
        return results['mean']

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        n_samples: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions with full uncertainty quantification.

        Args:
            X: Input data (numpy array)
            n_samples: Number of Monte Carlo samples

        Returns:
            dict: Dictionary with 'mean', 'std', 'samples', etc.
        """
        self.model.eval()

        # Convert to tensor
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)

        X = X.to(self.device)

        # Use model's built-in uncertainty prediction
        with torch.no_grad():
            results = self.model.predict_with_uncertainty(X, n_samples=n_samples)

        # Convert to numpy
        return {
            'mean': results['mean'].cpu().numpy(),
            'std': results['std'].cpu().numpy(),
            'samples': results['samples'].cpu().numpy(),
            'epistemic_uncertainty': results['epistemic_uncertainty'].cpu().numpy(),
            'confidence_intervals': {
                k: (v[0].cpu().numpy(), v[1].cpu().numpy())
                for k, v in results['confidence_intervals'].items()
            }
        }
