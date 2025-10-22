"""
Script per testare il modello con dati semplici

Esegui questo script per verificare che il sistema funzioni correttamente
con una funzione matematica semplice: output = 2*x + 3*y - z
"""

import sys
import torch
from torch.utils.data import DataLoader
from pathlib import Path

# Aggiungi src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data import load_csv_data, DataPreprocessor, MachineryDataset
from models import MachineryPredictor
from training import ModelTrainer
from utils import plot_training_history, plot_predictions, calculate_metrics, print_metrics

# Importa configurazione DI TEST
from configs.test_config import CONFIG


def main():
    print("="*70)
    print("TEST DEL MODELLO CON FUNZIONE SEMPLICE")
    print("Funzione: output = 2*x + 3*y - z + rumore")
    print("="*70)

    # 1. CARICA DATI
    print("\n[1/6] Caricamento dati di test...")
    try:
        X, y = load_csv_data(
            CONFIG['data']['csv_path'],
            CONFIG['data']['input_columns'],
            CONFIG['data']['output_columns']
        )
    except FileNotFoundError:
        print(f"\nERRORE: File non trovato: {CONFIG['data']['csv_path']}")
        print("\nEsegui prima: python generate_test_data.py")
        return

    # 2. PREPROCESSING
    print("\n[2/6] Preprocessing dei dati...")
    preprocessor = DataPreprocessor(scaling_method=CONFIG['data']['scaling_method'])

    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        X, y,
        train_size=CONFIG['data']['train_size'],
        val_size=CONFIG['data']['val_size'],
        test_size=CONFIG['data']['test_size'],
        random_state=CONFIG['data']['random_state']
    )

    print(f"  Train set: {len(X_train)} campioni")
    print(f"  Validation set: {len(X_val)} campioni")
    print(f"  Test set: {len(X_test)} campioni")

    # Fit e trasformazione
    X_train_scaled, y_train_scaled = preprocessor.fit_transform(X_train, y_train)
    X_val_scaled, y_val_scaled = preprocessor.transform(X_val, y_val)
    X_test_scaled, y_test_scaled = preprocessor.transform(X_test, y_test)

    # 3. CREA DATASET E DATALOADER
    print("\n[3/6] Creazione dataset PyTorch...")
    train_dataset = MachineryDataset(X_train_scaled, y_train_scaled)
    val_dataset = MachineryDataset(X_val_scaled, y_val_scaled)
    test_dataset = MachineryDataset(X_test_scaled, y_test_scaled)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['training']['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )

    input_dim = train_dataset.get_input_dim()
    output_dim = train_dataset.get_output_dim()
    print(f"  Input dimension: {input_dim}")
    print(f"  Output dimension: {output_dim}")

    # 4. CREA MODELLO
    print("\n[4/6] Creazione del modello...")
    model = MachineryPredictor(
        input_size=input_dim,
        hidden_sizes=CONFIG['model']['hidden_sizes'],
        output_size=output_dim,
        dropout_rate=CONFIG['model']['dropout_rate']
    )

    print(f"  Architettura: {CONFIG['model']['hidden_sizes']}")
    print(f"  Parametri totali: {sum(p.numel() for p in model.parameters())}")

    # 5. TRAINING
    print("\n[5/6] Inizio training...")
    print("\nSe il modello funziona, dovresti vedere:")
    print("  - Loss che scende rapidamente")
    print("  - Validation loss che segue la training loss")
    print("  - Convergenza in poche epoche (< 50)")
    print()

    device = CONFIG['training']['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer = ModelTrainer(
        model,
        device=device,
        learning_rate=CONFIG['training']['learning_rate'],
        loss_fn=CONFIG['training']['loss_function']
    )

    history = trainer.train(
        train_loader,
        val_loader,
        epochs=CONFIG['training']['epochs'],
        patience=CONFIG['training']['patience'],
        save_dir=CONFIG['training']['checkpoint_dir']
    )

    # 6. VALUTAZIONE
    print("\n[6/6] Valutazione sul test set...")

    y_pred_scaled = trainer.predict(X_test_scaled)
    y_pred = preprocessor.inverse_transform_output(y_pred_scaled)

    # Calcola metriche
    metrics = calculate_metrics(
        y_test,
        y_pred,
        output_names=['output']
    )
    print_metrics(metrics)

    # Salva scaler
    scaler_path = Path(CONFIG['training']['checkpoint_dir']) / 'scalers.pkl'
    preprocessor.save_scalers(scaler_path)

    # Visualizzazioni
    print("\nGenerazione visualizzazioni...")

    plot_training_history(
        history['train_losses'],
        history['val_losses'],
        save_path=Path(CONFIG['training']['checkpoint_dir']) / 'training_history.png'
    )

    plot_predictions(
        y_test,
        y_pred,
        output_names=['output = 2x + 3y - z'],
        save_path=Path(CONFIG['training']['checkpoint_dir']) / 'predictions.png'
    )

    # VERIFICA RISULTATI
    print("\n" + "="*70)
    print("VERIFICA RISULTATI")
    print("="*70)

    r2 = metrics['output']['R2']
    rmse = metrics['output']['RMSE']

    print(f"\nR² Score: {r2:.6f}")
    print(f"RMSE: {rmse:.6f}")

    if r2 > 0.99:
        print("\n✓ ECCELLENTE! Il modello ha imparato perfettamente!")
        print("  R² > 0.99 significa che il modello predice quasi perfettamente.")
    elif r2 > 0.95:
        print("\n✓ OTTIMO! Il modello funziona molto bene!")
    elif r2 > 0.90:
        print("\n⚠ BUONO, ma potrebbe migliorare.")
        print("  Prova ad aumentare epoche o cambiare architettura.")
    else:
        print("\n❌ PROBLEMA! Il modello non impara bene.")
        print("  Verifica i dati o i parametri di training.")

    print("\n" + "="*70)
    print("TRAINING TEST COMPLETATO!")
    print("="*70)
    print(f"\nFile salvati in: {CONFIG['training']['checkpoint_dir']}/")
    print("  - best_model.pth          : Miglior modello")
    print("  - scalers.pkl             : Scaler per preprocessing")
    print("  - training_history.png    : Grafico loss")
    print("  - predictions.png         : Grafico predizioni vs reali")

    # Test di predizione
    print("\n" + "="*70)
    print("TEST DI PREDIZIONE")
    print("="*70)

    # Caso di test: x=1, y=1, z=1 → output = 2*1 + 3*1 - 1 = 4
    test_input = [[1.0, 1.0, 1.0]]
    expected_output = 2*1 + 3*1 - 1  # = 4

    print(f"\nInput di test: x=1, y=1, z=1")
    print(f"Output atteso (2*1 + 3*1 - 1): {expected_output}")

    # Preprocessa e predici
    test_input_scaled = preprocessor.transform(test_input)
    pred_scaled = trainer.predict(test_input_scaled)
    pred = preprocessor.inverse_transform_output(pred_scaled)

    print(f"Output predetto dal modello: {pred[0][0]:.4f}")
    print(f"Errore: {abs(pred[0][0] - expected_output):.4f}")

    if abs(pred[0][0] - expected_output) < 0.5:
        print("\n✓ Predizione eccellente!")
    else:
        print("\n⚠ Predizione non precisa, il modello potrebbe migliorare.")


if __name__ == "__main__":
    # Setup seed per riproducibilità
    torch.manual_seed(CONFIG['misc']['random_seed'])
    import numpy as np
    np.random.seed(CONFIG['misc']['random_seed'])

    main()
