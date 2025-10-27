"""
Main script for model training

This script loads data, preprocesses, creates the model and trains it.
Customize configurations in the configs/example_config.py file
"""

import sys
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data import load_csv_data, DataPreprocessor, MachineryDataset
from models import MachineryPredictor, create_small_model, create_medium_model, create_large_model
from training import ModelTrainer
from utils import plot_training_history, plot_predictions, calculate_metrics, print_metrics, generate_training_report

# Import configuration
from configs.example_config import CONFIG


def main():
    # Capture training start timestamp
    training_start_time = datetime.now()

    print("="*70)
    print("NEURAL NETWORK TRAINING FOR MACHINERY PREDICTION")
    print("="*70)
    print(f"Training start: {training_start_time.strftime('%d/%m/%Y %H:%M:%S')}")
    print("="*70)

    # 1. LOAD DATA
    print("\n[1/6] Loading data...")
    try:
        X, y = load_csv_data(
            CONFIG['data']['csv_path'],
            CONFIG['data']['input_columns'],
            CONFIG['data']['output_columns']
        )
    except FileNotFoundError:
        print(f"\nERROR: File not found: {CONFIG['data']['csv_path']}")
        print("\nTo test the system, create a sample CSV file with:")
        print("  - Input columns: " + ", ".join(CONFIG['data']['input_columns']))
        print("  - Output columns: " + ", ".join(CONFIG['data']['output_columns']))
        print("\nOr modify configs/example_config.py with your data.")
        return

    # 2. PREPROCESSING
    print("\n[2/6] Preprocessing data...")
    preprocessor = DataPreprocessor(scaling_method=CONFIG['data']['scaling_method'])

    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        X, y,
        train_size=CONFIG['data']['train_size'],
        val_size=CONFIG['data']['val_size'],
        test_size=CONFIG['data']['test_size'],
        random_state=CONFIG['data']['random_state']
    )

    print(f"  Train set: {len(X_train)} samples")
    print(f"  Validation set: {len(X_val)} samples")
    print(f"  Test set: {len(X_test)} samples")

    # Fit and transform
    X_train_scaled, y_train_scaled = preprocessor.fit_transform(X_train, y_train)
    X_val_scaled, y_val_scaled = preprocessor.transform(X_val, y_val)
    X_test_scaled, y_test_scaled = preprocessor.transform(X_test, y_test)

    # 3. CREATE DATASET AND DATALOADER
    print("\n[3/6] Creating PyTorch dataset...")
    train_dataset = MachineryDataset(X_train_scaled, y_train_scaled)
    val_dataset = MachineryDataset(X_val_scaled, y_val_scaled)
    test_dataset = MachineryDataset(X_test_scaled, y_test_scaled)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['training']['batch_size'],
        shuffle=True,
        num_workers=0  # Use 0 to avoid issues on some systems
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

    # 4. CREATE MODEL
    print("\n[4/6] Creating model...")
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

    print(f"  Model type: {model_type}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")

    # 5. TRAINING
    print("\n[5/6] Starting training...")

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

    # 6. EVALUATION
    print("\n[6/6] Evaluation on test set...")

    y_pred_scaled = trainer.predict(X_test_scaled)
    y_pred = preprocessor.inverse_transform_output(y_pred_scaled)

    # Calculate metrics
    metrics = calculate_metrics(
        y_test,
        y_pred,
        output_names=CONFIG['data']['output_columns']
    )
    print_metrics(metrics)

    # Save scaler for future use
    scaler_path = Path(CONFIG['training']['checkpoint_dir']) / 'scalers.pkl'
    preprocessor.save_scalers(scaler_path)

    # 7. VISUALIZATIONS
    print("\nGenerating visualizations...")

    # Plot training history
    plot_training_history(
        history['train_losses'],
        history['val_losses'],
        save_path=Path(CONFIG['training']['checkpoint_dir']) / 'training_history.png'
    )

    # Plot predictions
    plot_predictions(
        y_test,
        y_pred,
        output_names=CONFIG['data']['output_columns'],
        save_path=Path(CONFIG['training']['checkpoint_dir']) / 'predictions.png'
    )

    # 8. GENERATE PDF REPORT
    print("\nGenerating PDF report...")
    try:
        total_params = sum(p.numel() for p in model.parameters())
        report_path = generate_training_report(
            config=CONFIG,
            history=history,
            metrics=metrics,
            input_dim=input_dim,
            output_dim=output_dim,
            total_params=total_params,
            n_train=len(X_train),
            n_val=len(X_val),
            n_test=len(X_test),
            checkpoint_dir=CONFIG['training']['checkpoint_dir'],
            timestamp=training_start_time
        )
        print(f"PDF report saved: {report_path}")
    except Exception as e:
        print(f"Error generating PDF: {e}")
        print("Training completed successfully anyway.")

    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nFiles saved in: {CONFIG['training']['checkpoint_dir']}/")
    print("  - best_model.pth          : Best model")
    print("  - scalers.pkl             : Scaler for preprocessing")
    print("  - training_history.json   : Training history")
    print("  - training_history.png    : Loss plot")
    print("  - predictions.png         : Predictions vs actual plot")
    print("  - training_report.pdf     : Complete training report")


if __name__ == "__main__":
    # Setup seed for reproducibility
    torch.manual_seed(CONFIG['misc']['random_seed'])
    import numpy as np
    np.random.seed(CONFIG['misc']['random_seed'])

    main()
