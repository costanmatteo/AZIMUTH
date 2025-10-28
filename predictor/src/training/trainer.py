"""
Trainer per il modello di predizione del macchinario
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from datetime import datetime


class ModelTrainer:
    """
    Classe per gestire il training del modello.

    Funzionalità:
    - Training loop con validation
    - Early stopping
    - Salvataggio checkpoints
    - Logging delle metriche
    - Supporto per diverse loss functions

    Args:
        model (nn.Module): Il modello da trainare
        device (str): 'cuda' o 'cpu' (default: auto-detect)
        learning_rate (float): Learning rate (default: 0.001)
        weight_decay (float): L2 weight decay (default: 0.0)
        loss_fn (str): Funzione di loss ('mse', 'mae', 'huber') (default: 'mse')
    """

    def __init__(self, model, device=None, learning_rate=0.001, weight_decay=0.0, loss_fn='mse'):
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Setup loss function
        if loss_fn == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_fn == 'mae':
            self.criterion = nn.L1Loss()
        elif loss_fn == 'huber':
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Loss function non supportata: {loss_fn}")

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

        print(f"Trainer inizializzato su device: {self.device}")
        print(f"Optimizer: Adam (lr={learning_rate}, weight_decay={weight_decay})")
        print(f"Loss function: {loss_fn}")

    def train_epoch(self, train_loader):
        """
        Training per una singola epoch.

        Args:
            train_loader (DataLoader): DataLoader per training set

        Returns:
            float: Loss media dell'epoca
        """
        self.model.train()
        epoch_loss = 0.0

        for batch_X, batch_y in train_loader:
            # Sposta i dati sul device
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            predictions = self.model(batch_X)
            loss = self.criterion(predictions, batch_y)

            # Backward pass e ottimizzazione
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        return avg_loss

    def validate(self, val_loader):
        """
        Validazione del modello.

        Args:
            val_loader (DataLoader): DataLoader per validation set

        Returns:
            float: Loss media sulla validation
        """
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                val_loss += loss.item()

        avg_loss = val_loss / len(val_loader)
        return avg_loss

    def train(self, train_loader, val_loader, epochs=100, patience=10, save_dir='checkpoints'):
        """
        Training completo con early stopping.

        Args:
            train_loader (DataLoader): DataLoader per training
            val_loader (DataLoader): DataLoader per validation
            epochs (int): Numero massimo di epoche
            patience (int): Epoche da aspettare prima di early stopping
            save_dir (str): Directory per salvare i checkpoints

        Returns:
            dict: Dizionario con la storia del training
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"INIZIO TRAINING")
        print(f"{'='*60}")
        print(f"Epoche: {epochs}")
        print(f"Early stopping patience: {patience}")
        print(f"Checkpoint directory: {save_path}")
        print(f"{'='*60}\n")

        epochs_without_improvement = 0

        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validation
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)

            # Logging
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.6f} - "
                  f"Val Loss: {val_loss:.6f}")

            # Salvataggio best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(save_path / 'best_model.pth', epoch, val_loss)
                print(f"  → Nuovo miglior modello salvato! (Val Loss: {val_loss:.6f})")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping attivato dopo {epoch+1} epoche")
                print(f"Miglior Val Loss: {self.best_val_loss:.6f}")
                break

        # Salva il training history
        self.save_training_history(save_path / 'training_history.json')

        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETATO")
        print(f"Miglior Val Loss: {self.best_val_loss:.6f}")
        print(f"{'='*60}\n")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'total_epochs': len(self.train_losses)
        }

    def save_checkpoint(self, filepath, epoch, val_loss):
        """Salva un checkpoint del modello"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, filepath)

    def load_checkpoint(self, filepath):
        """Carica un checkpoint del modello"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print(f"Checkpoint caricato da: {filepath}")
        return checkpoint

    def save_training_history(self, filepath):
        """Salva la storia del training in JSON"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': float(self.best_val_loss),
            'timestamp': datetime.now().isoformat()
        }
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)

    def predict(self, X):
        """
        Fa predizioni su nuovi dati.

        Args:
            X (np.ndarray o torch.Tensor): Input data

        Returns:
            np.ndarray: Predizioni
        """
        self.model.eval()

        # Converti in tensor se necessario
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)

        X = X.to(self.device)

        with torch.no_grad():
            predictions = self.model(X)

        return predictions.cpu().numpy()
