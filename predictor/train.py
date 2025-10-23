"""
Script principale per il training del modello

Questo script carica i dati, preprocessa, crea il modello e lo traína.
Personalizza le configurazioni nel file configs/example_config.py
"""

import sys
import torch
from torch.utils.data import DataLoader
from pathlib import Path

# Aggiungi src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data import load_csv_data, DataPreprocessor, MachineryDataset
from models import MachineryPredictor, create_small_model, create_medium_model, create_large_model
from training import ModelTrainer
from utils import plot_training_history, plot_predictions, calculate_metrics, print_metrics

# Importa configurazione
from configs.example_config import CONFIG


def main():
    print("="*70)
    print("TRAINING RETE NEURALE PER PREDIZIONE MACCHINARIO")
    print("="*70)

    # 1. CARICA DATI
    print("\n[1/6] Caricamento dati...")
    try:
        X, y = load_csv_data(
            CONFIG['data']['csv_path'],
            CONFIG['data']['input_columns'],
            CONFIG['data']['output_columns']
        )
    except FileNotFoundError:
        print(f"\nERRORE: File non trovato: {CONFIG['data']['csv_path']}")
        print("\nPer testare il sistema, crea un file CSV di esempio con:")
        print("  - Colonne di input: " + ", ".join(CONFIG['data']['input_columns']))
        print("  - Colonne di output: " + ", ".join(CONFIG['data']['output_columns']))
        print("\nOppure modifica configs/example_config.py con i tuoi dati.")
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
        num_workers=0  # Usa 0 per evitare problemi su alcuni sistemi
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
    model_type = CONFIG['model']['model_type']

    if model_type == 'small':
        model = create_small_model(input_dim, output_dim)
    elif model_type == 'medium':
        model = create_medium_model(input_dim, output_dim)
    elif model_type == 'large':
        model = create_large_model(input_dim, output_dim)
    else:  # custom
        model = MachineryPredictor(
            input_size=input_dim,
            hidden_sizes=CONFIG['model']['hidden_sizes'],
            output_size=output_dim,
            dropout_rate=CONFIG['model']['dropout_rate'],
            use_batchnorm=CONFIG['model']['use_batchnorm']
        )

    print(f"  Tipo modello: {model_type}")
    print(f"  Parametri totali: {sum(p.numel() for p in model.parameters())}")

    # 5. TRAINING
    print("\n[5/6] Inizio training...")

    # Setup device
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
        output_names=CONFIG['data']['output_columns']
    )
    print_metrics(metrics)

    # Salva scaler per uso futuro
    scaler_path = Path(CONFIG['training']['checkpoint_dir']) / 'scalers.pkl'
    preprocessor.save_scalers(scaler_path)

    # 7. VISUALIZZAZIONI
    print("\nGenerazione visualizzazioni...")

    # Plot training history
    plot_training_history(
        history['train_losses'],
        history['val_losses'],
        save_path=Path(CONFIG['training']['checkpoint_dir']) / 'training_history.png'
    )

    # Plot predizioni
    plot_predictions(
        y_test,
        y_pred,
        output_names=CONFIG['data']['output_columns'],
        save_path=Path(CONFIG['training']['checkpoint_dir']) / 'predictions.png'
    )

    print("\n" + "="*70)
    print("TRAINING COMPLETATO CON SUCCESSO!")
    print("="*70)
    print(f"\nFile salvati in: {CONFIG['training']['checkpoint_dir']}/")
    print("  - best_model.pth          : Miglior modello")
    print("  - scalers.pkl             : Scaler per preprocessing")
    print("  - training_history.json   : Storia del training")
    print("  - training_history.png    : Grafico loss")
    print("  - predictions.png         : Grafico predizioni vs reali")


if __name__ == "__main__":
    # Setup seed per riproducibilità
    torch.manual_seed(CONFIG['misc']['random_seed'])
    import numpy as np
    np.random.seed(CONFIG['misc']['random_seed'])

    main()
