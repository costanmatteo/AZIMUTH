#!/usr/bin/env python3
"""
Train CasualiT as a Surrogate for Reliability F Prediction.

This script trains a model to predict reliability F from process chain
trajectories.  The model architecture is selected via surrogate_config
(casualit_model: proT | StageCausaliT | SingleCausalLayer).

Usage:
    python train_surrogate.py [options]

Options:
    --epochs INT              Max training epochs (default: from config)
    --batch_size INT          Batch size (default: from config)
    --learning_rate FLOAT     Learning rate (default: from config)
    --generate_data           Generate new training data before training
    --data_only               Generate data only (no training)
    --skip_training           Skip training, only generate report from existing results
    --use_existing_dataset    Load data converted by convert_dataset.py instead of generating
    --output_dir PATH         Output directory
    --device STR              Device (cpu/cuda/auto)
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

# Add paths
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

# Check for required dependencies
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install: pip install torch tqdm")
    sys.exit(1)

from configs.surrogate_config import SURROGATE_CONFIG
from causaliT.surrogate_training.data_generator import generate_all_datasets, TrajectoryDataGenerator


class SimpleSurrogateModel(nn.Module):
    """
    Simplified surrogate model for F prediction.

    Takes trajectory sequence (n_processes, features) and outputs scalar F.
    Uses a transformer-inspired architecture with self-attention.
    """

    def __init__(self, config: dict):
        super().__init__()

        model_cfg = config['model']

        # Will be set when data is loaded
        self.n_processes = None
        self.n_features = None

        self.d_model = model_cfg['d_model_enc']
        self.d_ff = model_cfg['d_ff']
        self.n_heads = model_cfg['n_heads']
        self.n_layers = model_cfg['e_layers']
        self.dropout = model_cfg['dropout_emb']

        # Input projection (will be initialized when data shape is known)
        self.input_proj = None

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, 1),
            nn.Sigmoid()  # F is in [0, 1]
        )

    def set_input_dim(self, n_features: int, device: torch.device = None):
        """Set input dimension after data is loaded."""
        self.n_features = n_features
        self.input_proj = nn.Linear(n_features, self.d_model)
        # Move to same device as the rest of the model
        if device is not None:
            self.input_proj = self.input_proj.to(device)
        elif next(self.parameters(), None) is not None:
            self.input_proj = self.input_proj.to(next(self.parameters()).device)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, n_processes, features)

        Returns:
            F: (batch,) reliability predictions
        """
        if self.input_proj is None:
            raise RuntimeError("Call set_input_dim() before forward pass")

        # Project input to model dimension
        x = self.input_proj(x)  # (batch, n_processes, d_model)

        # Transformer encoder
        x = self.encoder(x)  # (batch, n_processes, d_model)

        # Pool over sequence (mean pooling)
        x = x.mean(dim=1)  # (batch, d_model)

        # Output
        F = self.output_head(x).squeeze(-1)  # (batch,)

        return F


class SurrogateTrainer:
    """Trainer for CasualiT surrogate model."""

    def __init__(self, config: dict, device: str = 'auto'):
        self.config = config

        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"SurrogateTrainer initialized on device: {self.device}")

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'learning_rate': [],
        }

        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def load_data(self, data_dir: str):
        """Load training data from disk."""
        data_path = Path(data_dir)

        print(f"\nLoading data from {data_path}...")

        # Load training data
        train_X = np.load(data_path / 'train_X.npy')
        train_Y = np.load(data_path / 'train_Y.npy')

        # Load validation data
        val_X = np.load(data_path / 'val_X.npy')
        val_Y = np.load(data_path / 'val_Y.npy')

        # Load test data
        test_X = np.load(data_path / 'test_X.npy')
        test_Y = np.load(data_path / 'test_Y.npy')

        # Load metadata
        metadata = dict(np.load(data_path / 'train_metadata.npz', allow_pickle=True))

        print(f"  Train: {train_X.shape[0]} samples")
        print(f"  Val: {val_X.shape[0]} samples")
        print(f"  Test: {test_X.shape[0]} samples")
        print(f"  Features: {train_X.shape}")

        # Convert to tensors
        self.train_X = torch.tensor(train_X, dtype=torch.float32)
        self.train_Y = torch.tensor(train_Y, dtype=torch.float32).squeeze()
        self.val_X = torch.tensor(val_X, dtype=torch.float32)
        self.val_Y = torch.tensor(val_Y, dtype=torch.float32).squeeze()
        self.test_X = torch.tensor(test_X, dtype=torch.float32)
        self.test_Y = torch.tensor(test_Y, dtype=torch.float32).squeeze()
        self.metadata = metadata

        return metadata

    def create_dataloaders(self):
        """Create data loaders."""
        batch_size = self.config['training']['batch_size']

        train_dataset = TensorDataset(self.train_X, self.train_Y)
        val_dataset = TensorDataset(self.val_X, self.val_Y)
        test_dataset = TensorDataset(self.test_X, self.test_Y)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    def train(self, save_dir: str):
        """
        Train the surrogate model.

        Args:
            save_dir: Directory to save checkpoints

        Returns:
            dict: Training results
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        training_cfg = self.config['training']

        # Create model
        self.model = SimpleSurrogateModel(self.config).to(self.device)
        self.model.set_input_dim(self.train_X.shape[2])

        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nModel created:")
        print(f"  Total parameters: {n_params:,}")
        print(f"  Trainable parameters: {n_trainable:,}")

        # Optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_cfg['learning_rate'],
            weight_decay=training_cfg['weight_decay']
        )

        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=training_cfg['scheduler_factor'],
            patience=training_cfg['scheduler_patience'],
        ) if training_cfg['use_scheduler'] else None

        # Loss function
        criterion = nn.MSELoss()
        mae_fn = nn.L1Loss()

        # Training loop
        max_epochs = training_cfg['max_epochs']
        patience = training_cfg['patience']
        epochs_without_improvement = 0

        print(f"\nStarting training for {max_epochs} epochs...")
        print(f"  Batch size: {training_cfg['batch_size']}")
        print(f"  Learning rate: {training_cfg['learning_rate']}")
        print(f"  Early stopping patience: {patience}")

        for epoch in range(max_epochs):
            # Training
            self.model.train()
            train_losses = []
            train_maes = []

            for batch_X, batch_Y in self.train_loader:
                batch_X = batch_X.to(self.device)
                batch_Y = batch_Y.to(self.device)

                optimizer.zero_grad()
                pred_F = self.model(batch_X)
                loss = criterion(pred_F, batch_Y)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                train_maes.append(mae_fn(pred_F, batch_Y).item())

            # Validation
            self.model.eval()
            val_losses = []
            val_maes = []

            with torch.no_grad():
                for batch_X, batch_Y in self.val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_Y = batch_Y.to(self.device)

                    pred_F = self.model(batch_X)
                    loss = criterion(pred_F, batch_Y)

                    val_losses.append(loss.item())
                    val_maes.append(mae_fn(pred_F, batch_Y).item())

            # Record history
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            train_mae = np.mean(train_maes)
            val_mae = np.mean(val_maes)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            self.history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            # Scheduler step
            if scheduler:
                scheduler.step(val_loss)

            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                epochs_without_improvement = 0

                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config,
                    'model_type': self.config['model'].get('casualit_model', 'proT'),
                }, save_path / 'best_model.ckpt')
            else:
                epochs_without_improvement += 1

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs - 1:
                print(f"Epoch {epoch:4d}/{max_epochs} | "
                      f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                      f"Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.2e}")

            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch}. Best epoch: {self.best_epoch}")
                break

        # Save final model
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'model_type': self.config['model'].get('casualit_model', 'proT'),
        }, save_path / 'final_model.ckpt')

        # Save training history
        np.savez(save_path / 'training_history.npz', **self.history)

        print(f"\nTraining complete!")
        print(f"  Best epoch: {self.best_epoch}")
        print(f"  Best val loss: {self.best_val_loss:.6f}")
        print(f"  Checkpoints saved to: {save_path}")

        return {
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'final_epoch': epoch,
            'final_val_loss': val_loss,
        }

    def evaluate(self, save_dir: str = None):
        """
        Evaluate on test set.

        Args:
            save_dir: If provided, load best model from this directory

        Returns:
            dict: Evaluation metrics
        """
        if save_dir:
            # Load best model
            checkpoint = torch.load(Path(save_dir) / 'best_model.ckpt', map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.eval()
        criterion = nn.MSELoss()
        mae_fn = nn.L1Loss()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_X, batch_Y in self.test_loader:
                batch_X = batch_X.to(self.device)
                batch_Y = batch_Y.to(self.device)

                pred_F = self.model(batch_X)
                all_preds.append(pred_F.cpu().numpy())
                all_targets.append(batch_Y.cpu().numpy())

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        # Compute metrics
        mse = np.mean((preds - targets) ** 2)
        mae = np.mean(np.abs(preds - targets))
        rmse = np.sqrt(mse)

        # R^2
        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        results = {
            'test_mse': mse,
            'test_mae': mae,
            'test_rmse': rmse,
            'test_r2': r2,
            'predictions': preds,
            'targets': targets,
        }

        print(f"\nTest Evaluation:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")

        return results


def generate_pdf_report(trainer: SurrogateTrainer, eval_results: dict,
                       config: dict, output_dir: str):
    """
    Generate PDF report for surrogate training.

    Args:
        trainer: Trained SurrogateTrainer
        eval_results: Evaluation results dict
        config: Configuration dict
        output_dir: Output directory for report
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        print("Warning: matplotlib not available, skipping PDF report")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = output_path / f'surrogate_training_report_{timestamp}.pdf'

    print(f"\nGenerating PDF report: {pdf_path}")

    with PdfPages(pdf_path) as pdf:
        # Page 1: Title and Summary
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')

        title_text = "CasualiT Surrogate Training Report"
        summary_text = f"""
Training Summary
================

Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Configuration:
- Max Epochs: {config['training']['max_epochs']}
- Batch Size: {config['training']['batch_size']}
- Learning Rate: {config['training']['learning_rate']}
- Weight Decay: {config['training']['weight_decay']}

Model Architecture:
- Encoder Dimension: {config['model']['d_model_enc']}
- Feed-forward Dimension: {config['model']['d_ff']}
- Attention Heads: {config['model']['n_heads']}
- Encoder Layers: {config['model']['e_layers']}
- Dropout: {config['model']['dropout_emb']}

Training Results:
- Best Epoch: {trainer.best_epoch}
- Best Validation Loss: {trainer.best_val_loss:.6f}

Test Evaluation:
- MSE: {eval_results['test_mse']:.6f}
- MAE: {eval_results['test_mae']:.4f}
- RMSE: {eval_results['test_rmse']:.4f}
- R²: {eval_results['test_r2']:.4f}
        """

        ax.text(0.5, 0.95, title_text, transform=ax.transAxes, fontsize=20,
                fontweight='bold', ha='center', va='top')
        ax.text(0.1, 0.85, summary_text, transform=ax.transAxes, fontsize=11,
                fontfamily='monospace', va='top')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 2: Training Curves
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

        # Loss curves
        epochs = range(len(trainer.history['train_loss']))
        axes[0, 0].plot(epochs, trainer.history['train_loss'], label='Train', alpha=0.8)
        axes[0, 0].plot(epochs, trainer.history['val_loss'], label='Validation', alpha=0.8)
        axes[0, 0].axvline(x=trainer.best_epoch, color='r', linestyle='--', alpha=0.5, label='Best')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('MSE Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)

        # MAE curves
        axes[0, 1].plot(epochs, trainer.history['train_mae'], label='Train', alpha=0.8)
        axes[0, 1].plot(epochs, trainer.history['val_mae'], label='Validation', alpha=0.8)
        axes[0, 1].axvline(x=trainer.best_epoch, color='r', linestyle='--', alpha=0.5, label='Best')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('MAE Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Learning rate
        axes[1, 0].plot(epochs, trainer.history['learning_rate'], color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

        # Prediction vs Target
        preds = eval_results['predictions']
        targets = eval_results['targets']
        axes[1, 1].scatter(targets, preds, alpha=0.3, s=10)
        axes[1, 1].plot([0, 1], [0, 1], 'r--', label='Perfect')
        axes[1, 1].set_xlabel('True F')
        axes[1, 1].set_ylabel('Predicted F')
        axes[1, 1].set_title(f'Predictions vs Targets (R²={eval_results["test_r2"]:.3f})')
        axes[1, 1].legend()
        axes[1, 1].set_xlim([0, 1])
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 3: Error Analysis
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

        errors = preds - targets

        # Error distribution
        axes[0, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(x=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Prediction Error (Pred - True)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title(f'Error Distribution (Mean={np.mean(errors):.4f}, Std={np.std(errors):.4f})')
        axes[0, 0].grid(True, alpha=0.3)

        # Absolute error vs target
        abs_errors = np.abs(errors)
        axes[0, 1].scatter(targets, abs_errors, alpha=0.3, s=10)
        axes[0, 1].set_xlabel('True F')
        axes[0, 1].set_ylabel('Absolute Error')
        axes[0, 1].set_title('Absolute Error vs True F')
        axes[0, 1].grid(True, alpha=0.3)

        # F distribution
        axes[1, 0].hist(targets, bins=50, alpha=0.5, label='True F', edgecolor='black')
        axes[1, 0].hist(preds, bins=50, alpha=0.5, label='Predicted F', edgecolor='black')
        axes[1, 0].set_xlabel('F Value')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Distribution of F Values')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Residuals by percentile
        percentiles = np.percentile(targets, np.linspace(0, 100, 11))
        percentile_errors = []
        percentile_labels = []
        for i in range(len(percentiles) - 1):
            mask = (targets >= percentiles[i]) & (targets < percentiles[i+1])
            if mask.sum() > 0:
                percentile_errors.append(np.abs(errors[mask]).mean())
                percentile_labels.append(f'{int(percentiles[i]*100)}-{int(percentiles[i+1]*100)}%')

        axes[1, 1].bar(range(len(percentile_errors)), percentile_errors)
        axes[1, 1].set_xticks(range(len(percentile_labels)))
        axes[1, 1].set_xticklabels(percentile_labels, rotation=45)
        axes[1, 1].set_xlabel('F Percentile')
        axes[1, 1].set_ylabel('Mean Absolute Error')
        axes[1, 1].set_title('Error by F Percentile')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    print(f"PDF report saved to: {pdf_path}")
    return pdf_path


def _train_stage_causal_lightning(casualit_model, config, args):
    """
    Train StageCausaliT or SingleCausalLayer using the PyTorch Lightning pipeline.

    Builds the OmegaConf config expected by causaliT.training.trainer.trainer()
    from the flat SURROGATE_CONFIG and the dataset_metadata.json produced by
    convert_dataset.py, then launches training.
    """
    from omegaconf import OmegaConf
    from causaliT.training.trainer import trainer as lightning_trainer

    if casualit_model == 'SingleCausalLayer':
        print(f"\n[INFO] SingleCausalLayer Lightning training not yet wired.")
        print(f"  Data has been prepared in {args.data_dir}.")
        return

    # ── Load dataset metadata (created by convert_dataset.py) ─────────
    metadata_path = Path(args.data_dir) / 'dataset_metadata.json'
    if not metadata_path.exists():
        print(f"\n[ERROR] dataset_metadata.json not found at {metadata_path}")
        print("  Run with --use_existing_dataset or --generate_data first.")
        return
    with open(metadata_path) as f:
        metadata = json.load(f)

    n_s = metadata['variable_info']['n_source']
    n_x = metadata['variable_info']['n_input']
    n_y = metadata['variable_info']['n_target']
    total_vars = n_s + n_x + n_y + 1  # +1 for padding_idx=0

    # ── Read hyper-parameters from SURROGATE_CONFIG ───────────────────
    d_model = config['model'].get('d_model_enc', 64)
    d_ff    = config['model'].get('d_ff', 128)
    d_qk    = config['model'].get('d_qk', 16)
    n_heads = config['model'].get('n_heads', 4)
    dropout = config['model'].get('dropout_emb', 0.1)

    # ── Path handling: Lightning trainer joins data_dir + dataset ─────
    # data lives directly in args.data_dir, so we split parent / name
    data_parent  = str(Path(args.data_dir).parent)   # e.g. "causaliT/data"
    dataset_name = str(Path(args.data_dir).name)      # e.g. "surrogate_training"

    # ── Build the config dict expected by the Lightning pipeline ──────
    lt_config = OmegaConf.create({
        'model': {
            'model_object': casualit_model,
            'kwargs': {
                'model': casualit_model,

                # Shared embedding (same table for S, X, Y – global variable IDs)
                'use_independent_embeddings': False,
                'ds_embed_shared': {
                    'setting': {'d_model': d_model, 'sparse_grad': False},
                    'modules': [
                        {'idx': 0, 'embed': 'linear', 'label': 'value',
                         'kwargs': {'input_dim': 1, 'embedding_dim': d_model}},
                        {'idx': 1, 'embed': 'nn_embedding', 'label': 'variable',
                         'kwargs': {'num_embeddings': total_vars,
                                    'embedding_dim': d_model,
                                    'padding_idx': 0, 'sparse': False,
                                    'max_norm': 1}},
                        {'idx': 0, 'embed': 'mask', 'label': 'value_missing',
                         'kwargs': {}},
                        {'idx': 1, 'embed': 'pass', 'label': 'order',
                         'kwargs': {}},
                    ],
                },
                'comps_embed_shared': 'summation',
                'val_idx_X': 0,  # value column to blank / predict

                # Attention (plain scaled dot-product, no causal masks)
                'dec1_cross_attention_type': 'ScaledDotProduct',
                'dec1_cross_mask_type':      'Uniform',
                'dec1_self_attention_type':   'ScaledDotProduct',
                'dec1_self_mask_type':        'Uniform',
                'dec2_cross_attention_type':  'ScaledDotProduct',
                'dec2_cross_mask_type':       'Uniform',
                'dec2_self_attention_type':   'ScaledDotProduct',
                'dec2_self_mask_type':        'Uniform',
                'n_heads':          n_heads,
                'dec1_causal_mask': False,
                'dec2_causal_mask': False,

                # Architecture
                'd1_layers':     config['model'].get('d_layers', 1),
                'd2_layers':     config['model'].get('d_layers', 1),
                'activation':    config['model'].get('activation', 'gelu'),
                'norm':          'layer',
                'use_final_norm': True,
                'device':         'cuda' if args.device != 'cpu' else 'cpu',

                # Dimensions
                'out_dim':  1,
                'd_ff':     d_ff,
                'd_model':  d_model,
                'd_qk':     d_qk,
                'S_seq_len': n_s,
                'X_seq_len': n_x,
                'Y_seq_len': n_y,

                # Dropout
                'dropout_emb':                  dropout,
                'dropout_attn_out':             dropout,
                'dropout_ff':                   dropout,
                'dec1_cross_dropout_qkv':       dropout,
                'dec1_cross_attention_dropout':  dropout,
                'dec1_self_dropout_qkv':        dropout,
                'dec1_self_attention_dropout':   dropout,
                'dec2_cross_dropout_qkv':       dropout,
                'dec2_cross_attention_dropout':  dropout,
                'dec2_self_dropout_qkv':        dropout,
                'dec2_self_attention_dropout':   dropout,
            },
        },

        'data': {
            'dataset':        dataset_name,
            'filename_input': 'ds.npz',
            'val_idx':   0,   # value column
            'val_idx_X': 0,
            'val_idx_Y': 0,
            'test_ds_ixd':    None,
            'max_data_size':  None,
            'S_seq_len': n_s,
            'X_seq_len': n_x,
            'Y_seq_len': n_y,
        },

        'training': {
            'optimizer':      'adamw',
            'lr':             config['training'].get('learning_rate', 1e-3),
            'weight_decay':   config['training'].get('weight_decay', 0.01),
            'use_scheduler':  config['training'].get('use_scheduler', True),
            'loss_fn':        'mse',
            'loss_weight_x':  1.0,
            'loss_weight_y':  1.0,
            'teacher_forcing': True,
            'batch_size':     config['training'].get('batch_size', 64),
            'max_epochs':     config['training'].get('max_epochs', 200),
            'k_fold':         config['training'].get('k_fold', 1),
            'seed':           config['training'].get('seed', 42),
            'save_ckpt_every_n_epochs': 50,
            'log_entropy':       False,
            'log_acyclicity':    False,
            'use_hard_masks':    False,
            'use_in_context_masks': False,
        },

        'special':    {'mode': []},
        'evaluation': {'functions': ['eval_train_metrics']},
    })

    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n[2/4] Training {casualit_model} with PyTorch Lightning...")
    print(f"  d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}")
    print(f"  S_seq_len={n_s}, X_seq_len={n_x}, Y_seq_len={n_y}")
    print(f"  num_embeddings={total_vars} (shared)")

    results_df = lightning_trainer(
        config=lt_config,
        data_dir=data_parent,
        save_dir=save_dir,
        cluster=False,
    )

    # Copy best checkpoint to the expected location
    best_ckpt_src = Path(save_dir) / 'k_0' / 'best_checkpoint.ckpt'
    best_ckpt_dst = Path(save_dir) / 'best_model.ckpt'
    if best_ckpt_src.exists():
        import shutil
        shutil.copy2(best_ckpt_src, best_ckpt_dst)
        print(f"  Best checkpoint copied to: {best_ckpt_dst}")

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"  Checkpoints: {save_dir}")
    if results_df is not None:
        print(f"  Results:\n{results_df.to_string()}")


def main():
    parser = argparse.ArgumentParser(description='Train CasualiT Surrogate')
    parser.add_argument('--epochs', type=int, default=None, help='Max epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--generate_data', action='store_true', help='Generate new training data')
    parser.add_argument('--data_only', action='store_true', help='Generate data only (no training)')
    parser.add_argument('--skip_training', action='store_true', help='Skip training')
    parser.add_argument('--use_existing_dataset', action='store_true',
                       help='Use pre-existing full_trajectories.pt (converted by convert_dataset.py)')
    parser.add_argument('--output_dir', type=str, default='causaliT/checkpoints/surrogate',
                       help='Output directory')
    parser.add_argument('--data_dir', type=str, default='causaliT/data/surrogate_training',
                       help='Data directory')
    parser.add_argument('--device', type=str, default='auto', help='Device (cpu/cuda/auto)')
    args = parser.parse_args()

    # Load config
    config = SURROGATE_CONFIG.copy()

    # Override config with command line args
    if args.epochs:
        config['training']['max_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate

    casualit_model = config['model'].get('casualit_model', 'proT')
    use_existing = args.use_existing_dataset or config['data'].get('use_existing_dataset', False)

    print("="*70)
    print("CasualiT Surrogate Training")
    print("="*70)
    print(f"Model type: {casualit_model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Data directory: {args.data_dir}")
    print(f"Use existing dataset: {use_existing}")

    # ── Data preparation ───────────────────────────────────────────────────
    data_path = Path(args.data_dir)

    if use_existing:
        # Convert from full_trajectories.pt if needed
        from causaliT.surrogate_training.convert_dataset import convert_trajectories_to_causalit_format

        dataset_path = config['data'].get('dataset_path',
                                           'data/trajectories/full_trajectories.pt')
        print("\n[1/4] Converting existing dataset...")
        convert_trajectories_to_causalit_format(
            trajectories_path=dataset_path,
            output_dir=args.data_dir,
            model_type=casualit_model,
            train_frac=config['data'].get('train_frac', 0.70),
            val_frac=config['data'].get('val_frac', 0.15),
            test_frac=config['data'].get('test_frac', 0.15),
            seed=config['data'].get('random_seed', 42),
        )

        if args.data_only:
            print("\nData conversion complete.")
            return

    elif args.data_only or args.generate_data or not (data_path / 'train_X.npy').exists():
        print("\n[1/4] Generating training data...")
        stats = generate_all_datasets(config, args.data_dir, device=args.device)

        if args.data_only:
            print("\n" + "="*70)
            print("Data Generation Complete!")
            print("="*70)
            print(f"Data saved to: {args.data_dir}")
            for split, s in stats.items():
                print(f"  {split}: {s['n_samples']} samples, F = {s['F_mean']:.4f} +/- {s['F_std']:.4f}")
            return
    else:
        print("\n[1/4] Using existing training data")

    # ── Load & train ───────────────────────────────────────────────────────
    if casualit_model in ('StageCausaliT', 'SingleCausalLayer'):
        _train_stage_causal_lightning(casualit_model, config, args)
        return

    # Create trainer (proT path)
    trainer = SurrogateTrainer(config, device=args.device)

    # Load data
    print("\n[2/4] Loading data...")
    metadata = trainer.load_data(args.data_dir)
    trainer.create_dataloaders()

    # Train
    if not args.skip_training:
        print("\n[3/4] Training model...")
        train_results = trainer.train(args.output_dir)
    else:
        print("\n[3/4] Skipping training (--skip_training)")
        # Load existing model
        trainer.model = SimpleSurrogateModel(config).to(trainer.device)
        trainer.model.set_input_dim(trainer.train_X.shape[2])

    # Evaluate
    print("\n[4/4] Evaluating model...")
    eval_results = trainer.evaluate(args.output_dir if not args.skip_training else args.output_dir)

    # Generate report
    report_dir = config['report']['output_dir']
    pdf_path = generate_pdf_report(trainer, eval_results, config, report_dir)

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Checkpoints: {args.output_dir}")
    print(f"Report: {pdf_path}")


if __name__ == '__main__':
    main()
