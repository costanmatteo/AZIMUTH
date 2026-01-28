"""
Training Pipeline for CasualiT Surrogate.

Trains the transformer model to predict reliability F from trajectories.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from typing import Dict, Optional, Tuple
from pathlib import Path

sys.path.insert(0, '/home/user/AZIMUTH')


class SurrogateTrainer:
    """
    Trainer for CasualiT surrogate model.

    Handles:
    - Training loop with validation
    - Learning rate scheduling
    - Early stopping
    - Checkpointing
    - Metrics logging
    """

    def __init__(self,
                 model: nn.Module,
                 data_module,
                 config: Dict,
                 device: str = 'cpu'):
        """
        Args:
            model: CasualiTSurrogate model
            data_module: SurrogateDataModule with train/val/test loaders
            config: Training configuration
            device: Torch device
        """
        self.model = model.to(device)
        self.data_module = data_module
        self.config = config
        self.device = device

        # Training config
        train_cfg = config.get('training', {})
        self.max_epochs = train_cfg.get('max_epochs', 200)
        self.patience = train_cfg.get('patience', 30)
        self.checkpoint_dir = Path(train_cfg.get('checkpoint_dir', 'casualit_surrogate/checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Loss function
        loss_fn_name = train_cfg.get('loss_fn', 'mse')
        if loss_fn_name == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_fn_name == 'mae':
            self.criterion = nn.L1Loss()
        elif loss_fn_name == 'huber':
            self.criterion = nn.HuberLoss()
        else:
            self.criterion = nn.MSELoss()

        # Optimizer
        optimizer_name = train_cfg.get('optimizer', 'adamw')
        lr = train_cfg.get('learning_rate', 1e-4)
        weight_decay = train_cfg.get('weight_decay', 1e-5)

        if optimizer_name == 'adam':
            self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Learning rate scheduler
        self.scheduler = self._create_scheduler(train_cfg)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_r2': [],
            'lr': [],
        }

    def _create_scheduler(self, config: Dict):
        """Create learning rate scheduler from config."""
        scheduler_cfg = config.get('lr_scheduler', {})
        scheduler_type = scheduler_cfg.get('type', 'cosine')

        if scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_cfg.get('T_max', self.max_epochs),
                eta_min=scheduler_cfg.get('eta_min', 1e-6),
            )
        elif scheduler_type == 'step':
            return StepLR(
                self.optimizer,
                step_size=scheduler_cfg.get('step_size', 50),
                gamma=scheduler_cfg.get('gamma', 0.5),
            )
        elif scheduler_type == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_cfg.get('factor', 0.5),
                patience=scheduler_cfg.get('patience', 10),
            )
        else:
            return None

    def train(self, verbose: bool = True) -> Dict:
        """
        Run full training loop.

        Args:
            verbose: Print progress

        Returns:
            Training history dict
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training CasualiT Surrogate")
            print(f"{'='*60}")
            stats = self.data_module.get_statistics()
            print(f"Train samples: {stats['train']['n_samples']}")
            print(f"Val samples: {stats['val']['n_samples']}")
            print(f"Test samples: {stats['test']['n_samples']}")
            print(f"Input shape: {stats['train']['input_shape']}")
            print(f"{'='*60}\n")

        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.val_dataloader()

        start_time = time.time()

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch

            # Training epoch
            train_loss = self._train_epoch(train_loader)

            # Validation
            val_metrics = self._validate(val_loader)
            val_loss = val_metrics['loss']

            # Learning rate scheduling
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['val_r2'].append(val_metrics['r2'])
            self.history['lr'].append(current_lr)

            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self._save_checkpoint('best_model.ckpt')
            else:
                self.epochs_without_improvement += 1

            # Logging
            if verbose and (epoch % 10 == 0 or epoch == self.max_epochs - 1):
                print(f"Epoch {epoch+1:4d}/{self.max_epochs} | "
                      f"Train: {train_loss:.6f} | "
                      f"Val: {val_loss:.6f} | "
                      f"MAE: {val_metrics['mae']:.4f} | "
                      f"R²: {val_metrics['r2']:.4f} | "
                      f"LR: {current_lr:.2e}")

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break

        elapsed = time.time() - start_time

        # Final evaluation on test set
        test_loader = self.data_module.test_dataloader()
        test_metrics = self._validate(test_loader)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Training Complete in {elapsed:.1f}s")
            print(f"Best Val Loss: {self.best_val_loss:.6f}")
            print(f"Test Loss: {test_metrics['loss']:.6f}")
            print(f"Test MAE: {test_metrics['mae']:.4f}")
            print(f"Test R²: {test_metrics['r2']:.4f}")
            print(f"{'='*60}\n")

        # Save final checkpoint
        self._save_checkpoint('final_model.ckpt')

        return {
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'test_metrics': test_metrics,
            'elapsed_time': elapsed,
        }

    def _train_epoch(self, dataloader) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for X, Y in dataloader:
            X = X.to(self.device)
            Y = Y.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            Y_pred = self.model(X)

            # Compute loss
            loss = self.criterion(Y_pred, Y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    @torch.no_grad()
    def _validate(self, dataloader) -> Dict:
        """Run validation and compute metrics."""
        self.model.eval()

        all_preds = []
        all_targets = []
        total_loss = 0.0
        n_batches = 0

        for X, Y in dataloader:
            X = X.to(self.device)
            Y = Y.to(self.device)

            Y_pred = self.model(X)
            loss = self.criterion(Y_pred, Y)

            all_preds.append(Y_pred.cpu())
            all_targets.append(Y.cpu())
            total_loss += loss.item()
            n_batches += 1

        # Concatenate all predictions
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()

        # Compute metrics
        mae = np.abs(all_preds - all_targets).mean()

        # R² score
        ss_res = np.sum((all_targets - all_preds) ** 2)
        ss_tot = np.sum((all_targets - all_targets.mean()) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        return {
            'loss': total_loss / n_batches,
            'mae': mae,
            'r2': r2,
        }

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename

        # Get model config
        model_config = {
            'n_processes': self.model.n_processes,
            'features_per_process': self.model.features_per_process,
            'embed_dim': self.model.embed_dim,
            'n_heads': self.model.encoder.layers[0].self_attn.num_heads,
            'n_encoder_layers': len(self.model.encoder.layers),
            'n_decoder_layers': len(self.model.decoder.layers),
            'ff_dim': self.model.encoder.layers[0].linear1.out_features,
            'dropout': self.model.encoder.layers[0].dropout.p,
        }

        # Get normalization stats
        norm_stats = self.data_module.normalization_stats

        self.model.save(
            str(path),
            config=model_config,
            normalization_stats=norm_stats,
            extra_data={
                'epoch': self.current_epoch,
                'history': self.history,
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
        )


def train_casualit_surrogate(config: Optional[Dict] = None,
                             data: Optional[Tuple] = None,
                             verbose: bool = True) -> Dict:
    """
    Main training function for CasualiT surrogate.

    Args:
        config: SURROGATE_CONFIG (if None, uses default)
        data: Pre-generated (X_train, Y_train, X_test, Y_test) data
              If None, generates new data using TrajectoryGenerator
        verbose: Print progress

    Returns:
        Training results dict
    """
    from casualit_surrogate.configs.surrogate_config import SURROGATE_CONFIG
    from casualit_surrogate.src.data.surrogate_dataset import SurrogateDataModule
    from casualit_surrogate.src.models.casualit_surrogate import CasualiTSurrogate

    config = config or SURROGATE_CONFIG

    # Determine device
    device_cfg = config.get('misc', {}).get('device', 'auto')
    if device_cfg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_cfg

    if verbose:
        print(f"Using device: {device}")

    # Generate or use provided data
    if data is None:
        from casualit_surrogate.src.data.trajectory_generator import create_trajectory_generator

        if verbose:
            print("Generating training data...")

        generator = create_trajectory_generator(config, device=device)
        X_train, Y_train, X_test, Y_test = generator.generate_training_data()
    else:
        X_train, Y_train, X_test, Y_test = data

    if verbose:
        print(f"Data shapes: X_train={X_train.shape}, Y_train={Y_train.shape}")
        print(f"             X_test={X_test.shape}, Y_test={Y_test.shape}")

    # Create data module
    train_cfg = config.get('training', {})
    data_cfg = config.get('data', {})

    data_module = SurrogateDataModule(
        X_train, Y_train, X_test, Y_test,
        batch_size=train_cfg.get('batch_size', 64),
        val_split=train_cfg.get('val_split', 0.2),
        normalize_inputs=data_cfg.get('normalize_inputs', True),
    )

    # Create model
    model_cfg = config.get('model', {})
    n_processes, features_per_process = data_module.get_input_dim()

    model = CasualiTSurrogate(
        n_processes=n_processes,
        features_per_process=features_per_process,
        embed_dim=model_cfg.get('embed_dim', 64),
        n_heads=model_cfg.get('n_heads', 4),
        n_encoder_layers=model_cfg.get('n_encoder_layers', 2),
        n_decoder_layers=model_cfg.get('n_decoder_layers', 2),
        ff_dim=model_cfg.get('ff_dim', 128),
        dropout=model_cfg.get('dropout', 0.1),
        device=device,
    )

    # Create trainer and train
    trainer = SurrogateTrainer(model, data_module, config, device)
    results = trainer.train(verbose=verbose)

    return results


if __name__ == '__main__':
    # Test training with dummy data
    print("Testing SurrogateTrainer with dummy data...")

    # Create dummy data
    n_train = 1000
    n_test = 200
    n_processes = 4
    features = 4

    X_train = np.random.randn(n_train, n_processes, features).astype(np.float32)
    Y_train = np.random.rand(n_train, 1).astype(np.float32)
    X_test = np.random.randn(n_test, n_processes, features).astype(np.float32)
    Y_test = np.random.rand(n_test, 1).astype(np.float32)

    # Train
    results = train_casualit_surrogate(
        data=(X_train, Y_train, X_test, Y_test),
        verbose=True,
    )

    print("\n✓ Training test passed!")
