"""
Step 1b: Addestra il surrogato CasualiT per predire F.

Carica le traiettorie complete da data/trajectories/full_trajectories.pt
(prodotte da generate_dataset.py, stesso dataset usato per gli uncertainty predictor)
e addestra un transformer a predire la reliability F.

Per ogni campione, il modello riceve:
    X = (n_processes, features_per_process)
dove features_per_process = cat(inputs, env, outputs) per ogni processo.

Il target è lo scalare F calcolato dalla ReliabilityFunction analitica.

Usa: python train_surrogate.py [--device cuda] [--epochs 200]

Output:
- checkpoints/surrogate/best_model.ckpt
- checkpoints/surrogate/surrogate_training_report_*.pdf
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
import numpy as np

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from configs.surrogate_config import SURROGATE_CONFIG


# ═════════════════════════════════════════════════════════════════════════════
# Model
# ═════════════════════════════════════════════════════════════════════════════

class SurrogateModel(nn.Module):
    """
    Transformer-based surrogate for F prediction.

    Input:  (batch, n_processes, features_per_process)
    Output: (batch,) — reliability F in [0, 1]
    """

    def __init__(self, n_features: int, config: dict):
        super().__init__()

        cfg = config['model']
        d_model = cfg['d_model_enc']

        self.input_proj = nn.Linear(n_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cfg['n_heads'],
            dim_feedforward=cfg['d_ff'],
            dropout=cfg['dropout_emb'],
            activation=cfg['activation'],
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg['e_layers']
        )

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg['dropout_emb']),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        x = x.mean(dim=1)  # pool over processes
        return self.output_head(x).squeeze(-1)


# ═════════════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════════════

def load_trajectories(data_dir: str):
    """
    Carica full_trajectories.pt e converte nel formato surrogate.

    Returns:
        X: (n_samples, n_processes, features) tensor
        Y: (n_samples,) tensor — valori F
        process_names: list of process names
    """
    traj_path = Path(data_dir) / 'trajectories' / 'full_trajectories.pt'
    if not traj_path.exists():
        raise FileNotFoundError(
            f"Dataset non trovato: {traj_path}\n"
            f"Esegui prima generate_dataset.py per generare i dati."
        )

    print(f"  Caricamento da {traj_path}...")
    raw = torch.load(traj_path, weights_only=False)

    # Estrai nomi processi (ordine dal primo campione)
    process_names = list(raw[0]['trajectory'].keys())
    n_samples = len(raw)

    # Costruisci X: per ogni campione e processo, concatena (inputs, env, outputs)
    all_X = []
    all_Y = []

    for sample in raw:
        process_features = []
        for pname in process_names:
            pdata = sample['trajectory'][pname]
            feat = torch.cat([pdata['inputs'], pdata['env'], pdata['outputs']])
            process_features.append(feat)

        all_X.append(torch.stack(process_features))  # (n_processes, features)
        all_Y.append(sample['F'])

    X = torch.stack(all_X)  # (n_samples, n_processes, features)
    Y = torch.tensor(all_Y, dtype=torch.float32)  # (n_samples,)

    print(f"  Campioni: {n_samples}")
    print(f"  Processi: {len(process_names)} ({', '.join(process_names)})")
    print(f"  Features per processo: {X.shape[2]}")
    print(f"  X shape: {X.shape}")
    print(f"  F: mean={Y.mean():.4f}, std={Y.std():.4f}, "
          f"min={Y.min():.4f}, max={Y.max():.4f}")

    return X, Y, process_names


def split_data(X, Y, config, seed=42):
    """Split in train/val/test."""
    n = len(X)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    data_cfg = config['data']
    n_train = int(n * data_cfg['train_size'])
    n_val = int(n * data_cfg['val_size'])

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return (
        X[train_idx], Y[train_idx],
        X[val_idx], Y[val_idx],
        X[test_idx], Y[test_idx],
    )


# ═════════════════════════════════════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════════════════════════════════════

def train_surrogate(config, X_train, Y_train, X_val, Y_val, device, save_dir):
    """Train the surrogate model. Returns (model, history)."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    training_cfg = config['training']
    batch_size = training_cfg['batch_size']

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False
    )

    # Model
    n_features = X_train.shape[2]
    model = SurrogateModel(n_features, config).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parametri trainabili: {n_params:,}")

    # Optimizer & scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_cfg['learning_rate'],
        weight_decay=training_cfg['weight_decay'],
    )
    scheduler = None
    if training_cfg['use_scheduler']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            factor=training_cfg['scheduler_factor'],
            patience=training_cfg['scheduler_patience'],
        )

    criterion = nn.MSELoss()
    mae_fn = nn.L1Loss()

    # Training loop
    max_epochs = training_cfg['max_epochs']
    patience = training_cfg['patience']
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0

    history = {
        'train_loss': [], 'val_loss': [],
        'train_mae': [], 'val_mae': [],
        'learning_rate': [],
    }

    print(f"  Epochs: {max_epochs}, batch_size: {batch_size}, "
          f"lr: {training_cfg['learning_rate']}, patience: {patience}")

    for epoch in range(max_epochs):
        # Train
        model.train()
        losses, maes = [], []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            maes.append(mae_fn(pred, by).item())

        # Validate
        model.eval()
        v_losses, v_maes = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                v_losses.append(criterion(pred, by).item())
                v_maes.append(mae_fn(pred, by).item())

        train_loss = np.mean(losses)
        val_loss = np.mean(v_losses)
        train_mae = np.mean(maes)
        val_mae = np.mean(v_maes)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        if scheduler:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
                'n_features': n_features,
            }, save_path / 'best_model.ckpt')
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == max_epochs - 1:
            print(f"  Epoch {epoch:4d}/{max_epochs} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"MAE: {val_mae:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}. Best: {best_epoch}")
            break

    # Save final
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'config': config,
        'n_features': n_features,
    }, save_path / 'final_model.ckpt')

    np.savez(save_path / 'training_history.npz', **history)

    print(f"\n  Best epoch: {best_epoch}, best val loss: {best_val_loss:.6f}")

    return model, history, best_epoch, best_val_loss


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═════════════════════════════════════════════════════════════════════════════

def evaluate(model, X_test, Y_test, device, batch_size=64):
    """Evaluate on test set. Returns metrics dict."""
    loader = DataLoader(
        TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=False
    )

    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            pred = model(bx)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(by.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(mse)

    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    results = {
        'test_mse': mse,
        'test_mae': mae,
        'test_rmse': rmse,
        'test_r2': r2,
        'predictions': preds,
        'targets': targets,
    }

    print(f"\n  Test Results:")
    print(f"    MSE:  {mse:.6f}")
    print(f"    MAE:  {mae:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    R2:   {r2:.4f}")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# PDF Report
# ═════════════════════════════════════════════════════════════════════════════

def generate_pdf_report(history, eval_results, config, best_epoch, output_dir):
    """Generate PDF report with training curves and error analysis."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        print("  matplotlib non disponibile, skip report PDF")
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = output_path / f'surrogate_training_report_{timestamp}.pdf'

    with PdfPages(pdf_path) as pdf:
        # Page 1: Summary
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')

        summary = f"""CasualiT Surrogate Training Report
{'='*50}

Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Configuration:
  Max Epochs:    {config['training']['max_epochs']}
  Batch Size:    {config['training']['batch_size']}
  Learning Rate: {config['training']['learning_rate']}
  Weight Decay:  {config['training']['weight_decay']}

Model:
  d_model:       {config['model']['d_model_enc']}
  d_ff:          {config['model']['d_ff']}
  Heads:         {config['model']['n_heads']}
  Layers:        {config['model']['e_layers']}
  Dropout:       {config['model']['dropout_emb']}

Results:
  Best Epoch:    {best_epoch}
  Test MSE:      {eval_results['test_mse']:.6f}
  Test MAE:      {eval_results['test_mae']:.4f}
  Test RMSE:     {eval_results['test_rmse']:.4f}
  Test R2:       {eval_results['test_r2']:.4f}
"""
        ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=11,
                fontfamily='monospace', va='top')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 2: Training curves + scatter
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

        epochs = range(len(history['train_loss']))

        axes[0, 0].plot(epochs, history['train_loss'], label='Train', alpha=0.8)
        axes[0, 0].plot(epochs, history['val_loss'], label='Val', alpha=0.8)
        axes[0, 0].axvline(x=best_epoch, color='r', ls='--', alpha=0.5, label='Best')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('MSE Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(epochs, history['train_mae'], label='Train', alpha=0.8)
        axes[0, 1].plot(epochs, history['val_mae'], label='Val', alpha=0.8)
        axes[0, 1].axvline(x=best_epoch, color='r', ls='--', alpha=0.5, label='Best')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('MAE Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(epochs, history['learning_rate'], color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('LR Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

        preds = eval_results['predictions']
        targets = eval_results['targets']
        axes[1, 1].scatter(targets, preds, alpha=0.3, s=10)
        axes[1, 1].plot([0, 1], [0, 1], 'r--', label='Perfect')
        axes[1, 1].set_xlabel('True F')
        axes[1, 1].set_ylabel('Predicted F')
        axes[1, 1].set_title(f'Pred vs True (R2={eval_results["test_r2"]:.3f})')
        axes[1, 1].legend()
        axes[1, 1].set_xlim([0, 1])
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 3: Error analysis
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        errors = preds - targets

        axes[0, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(x=0, color='r', ls='--')
        axes[0, 0].set_xlabel('Error (Pred - True)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title(f'Error Distribution (mean={np.mean(errors):.4f}, std={np.std(errors):.4f})')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].scatter(targets, np.abs(errors), alpha=0.3, s=10)
        axes[0, 1].set_xlabel('True F')
        axes[0, 1].set_ylabel('|Error|')
        axes[0, 1].set_title('Absolute Error vs True F')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].hist(targets, bins=50, alpha=0.5, label='True', edgecolor='black')
        axes[1, 0].hist(preds, bins=50, alpha=0.5, label='Pred', edgecolor='black')
        axes[1, 0].set_xlabel('F')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('F Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Error by percentile
        percentiles = np.percentile(targets, np.linspace(0, 100, 11))
        pct_errors, pct_labels = [], []
        for j in range(len(percentiles) - 1):
            mask = (targets >= percentiles[j]) & (targets < percentiles[j + 1])
            if mask.sum() > 0:
                pct_errors.append(np.abs(errors[mask]).mean())
                pct_labels.append(f'{int(percentiles[j]*100)}-{int(percentiles[j+1]*100)}%')

        axes[1, 1].bar(range(len(pct_errors)), pct_errors)
        axes[1, 1].set_xticks(range(len(pct_labels)))
        axes[1, 1].set_xticklabels(pct_labels, rotation=45)
        axes[1, 1].set_xlabel('F Percentile')
        axes[1, 1].set_ylabel('Mean |Error|')
        axes[1, 1].set_title('Error by Percentile')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    print(f"  Report PDF: {pdf_path}")
    return pdf_path


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Step 1b: Train CasualiT surrogate to predict F'
    )
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_dir', type=str, default='data/',
                        help='Directory dei dati (da generate_dataset.py)')
    parser.add_argument('--skip_training', action='store_true',
                        help='Salta il training, valuta un modello esistente')
    args = parser.parse_args()

    # Config
    import copy
    config = copy.deepcopy(SURROGATE_CONFIG)
    if args.epochs:
        config['training']['max_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    save_dir = config['checkpoints']['save_dir']
    report_dir = config['report']['output_dir']

    print("=" * 70)
    print("AZIMUTH - STEP 1b: TRAIN CAUSALIT SURROGATE")
    print("=" * 70)
    print(f"\nDevice: {device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Checkpoint dir: {save_dir}")

    # ── Step 1: Load data ────────────────────────────────────────────────
    print(f"\n[1/4] Caricamento traiettorie...")
    X, Y, process_names = load_trajectories(args.data_dir)

    # ── Step 2: Split ────────────────────────────────────────────────────
    print(f"\n[2/4] Split train/val/test...")
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(
        X, Y, config, seed=args.seed
    )
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ── Step 3: Train ────────────────────────────────────────────────────
    if not args.skip_training:
        print(f"\n[3/4] Training...")
        model, history, best_epoch, best_val_loss = train_surrogate(
            config, X_train, Y_train, X_val, Y_val, device, save_dir
        )
    else:
        print(f"\n[3/4] Skip training, caricamento modello esistente...")
        ckpt_path = Path(save_dir) / 'best_model.ckpt'
        if not ckpt_path.exists():
            print(f"  Errore: {ckpt_path} non trovato")
            return
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        n_features = ckpt['n_features']
        model = SurrogateModel(n_features, config).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        best_epoch = ckpt['epoch']
        best_val_loss = ckpt['val_loss']

        hist_path = Path(save_dir) / 'training_history.npz'
        if hist_path.exists():
            h = np.load(hist_path)
            history = {k: h[k].tolist() for k in h.files}
        else:
            history = None

    # ── Step 4: Evaluate ─────────────────────────────────────────────────
    print(f"\n[4/4] Valutazione su test set...")
    # Load best model for evaluation
    ckpt_path = Path(save_dir) / 'best_model.ckpt'
    if ckpt_path.exists() and not args.skip_training:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

    eval_results = evaluate(model, X_test, Y_test, device,
                           batch_size=config['training']['batch_size'])

    # Save eval results
    save_path = Path(save_dir)
    eval_save = {k: v for k, v in eval_results.items()
                 if k not in ('predictions', 'targets')}
    eval_save['best_epoch'] = best_epoch
    eval_save['process_names'] = process_names
    eval_save['n_samples'] = len(X)

    with open(save_path / 'eval_results.json', 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v
                   for k, v in eval_save.items()}, f, indent=2)

    # Report PDF
    if config['report']['generate_pdf'] and history is not None:
        generate_pdf_report(history, eval_results, config, best_epoch, report_dir)

    # Summary
    print("\n" + "=" * 70)
    print("STEP 1b COMPLETED!")
    print("=" * 70)
    print(f"  Checkpoint: {save_dir}/best_model.ckpt")
    print(f"  R2: {eval_results['test_r2']:.4f}")
    print(f"  MAE: {eval_results['test_mae']:.4f}")
    print("\nNext step: Run train_controller.py to train policy generators")


if __name__ == '__main__':
    main()
