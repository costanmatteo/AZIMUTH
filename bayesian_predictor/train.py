"""
Training script for Bayesian Neural Network

This script trains a Bayesian neural network for machinery prediction.
The model learns distributions over weights and provides uncertainty estimates.

Usage:
    python train.py

Configure training parameters in configs/example_config.py
"""

import sys
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data import load_csv_data, DataPreprocessor, MachineryDataset
from models import (
    BayesianPredictor,
    BayesianELBOLoss,
    create_small_bayesian_model,
    create_medium_bayesian_model,
    create_large_bayesian_model
)
from training import BayesianTrainer
from utils import (
    calculate_metrics,
    print_metrics,
    plot_bayesian_training_history,
    plot_predictions_with_uncertainty,
    plot_uncertainty_calibration,
    plot_epistemic_uncertainty_heatmap
)

# Import configuration
from configs.example_config import CONFIG


def main():
    """Main training function"""
    # Capture training start timestamp
    training_start_time = datetime.now()

    print("="*70)
    print("BAYESIAN NEURAL NETWORK TRAINING")
    print("="*70)
    print(f"Training start: {training_start_time.strftime('%d/%m/%Y %H:%M:%S')}")
    print("="*70)

    # 1. LOAD DATA
    print("\n[1/7] Loading data...")
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

    print(f"  Loaded {len(X)} samples")
    print(f"  Input features: {X.shape[1]}")
    print(f"  Output features: {y.shape[1]}")

    # 2. PREPROCESSING
    print("\n[2/7] Preprocessing data...")
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
    print("\n[3/7] Creating PyTorch datasets...")
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

    # 4. CREATE BAYESIAN MODEL
    print("\n[4/7] Creating Bayesian model...")
    model_type = CONFIG['model']['model_type']
    prior_std = CONFIG['model'].get('prior_std', 1.0)

    if model_type == 'small':
        model = create_small_bayesian_model(input_dim, output_dim, prior_std)
    elif model_type == 'medium':
        model = create_medium_bayesian_model(input_dim, output_dim, prior_std)
    elif model_type == 'large':
        model = create_large_bayesian_model(input_dim, output_dim, prior_std)
    else:  # custom
        model = BayesianPredictor(
            input_size=input_dim,
            hidden_sizes=CONFIG['model']['hidden_sizes'],
            output_size=output_dim,
            prior_std=prior_std,
            dropout_rate=CONFIG['model']['dropout_rate']
        )

    # Count parameters (note: Bayesian models have 2x parameters - mean and variance)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model type: {model_type}")
    print(f"  Prior std: {prior_std}")
    print(f"  Total parameters: {total_params}")
    print(f"  (Note: Bayesian models store both mean and variance for each weight)")

    # 5. CREATE LOSS FUNCTION
    print("\n[5/7] Setting up Bayesian loss function...")
    # KL weight: typically 1/N where N is the training set size
    kl_weight = CONFIG['training'].get('kl_weight', 1.0 / len(train_dataset))
    loss_fn = BayesianELBOLoss(kl_weight=kl_weight, reduction='mean')
    print(f"  KL weight: {kl_weight:.6f}")

    # 6. TRAINING
    print("\n[6/7] Starting Bayesian training...")

    # Setup device
    device = CONFIG['training']['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer = BayesianTrainer(
        model,
        loss_fn,
        device=device,
        learning_rate=CONFIG['training']['learning_rate'],
        weight_decay=CONFIG['training']['weight_decay'],
        kl_schedule=CONFIG['training'].get('kl_schedule', 'linear'),
        kl_warmup_epochs=CONFIG['training'].get('kl_warmup_epochs', 10),
        n_train_samples=CONFIG['training'].get('n_train_samples', 1),
        n_val_samples=CONFIG['training'].get('n_val_samples', 10)
    )

    history = trainer.train(
        train_loader,
        val_loader,
        epochs=CONFIG['training']['epochs'],
        patience=CONFIG['training']['patience'],
        save_dir=CONFIG['training']['checkpoint_dir']
    )

    # 7. EVALUATION WITH UNCERTAINTY
    print("\n[7/7] Evaluation on test set with uncertainty quantification...")

    n_test_samples = CONFIG['evaluation'].get('n_samples', 100)
    print(f"  Using {n_test_samples} Monte Carlo samples for uncertainty estimation")

    results = trainer.predict_with_uncertainty(X_test_scaled, n_samples=n_test_samples)
    y_pred_mean = preprocessor.inverse_transform_output(results['mean'])
    y_pred_std_scaled = results['std']

    # Calculate metrics on mean predictions
    metrics = calculate_metrics(
        y_test,
        y_pred_mean,
        output_names=CONFIG['data']['output_columns']
    )
    print_metrics(metrics)

    # Calculate average uncertainty
    avg_uncertainty = y_pred_std_scaled.mean()
    print(f"\nAverage prediction uncertainty (scaled): {avg_uncertainty:.6f}")

    # Save scaler for future use
    scaler_path = Path(CONFIG['training']['checkpoint_dir']) / 'scalers.pkl'
    preprocessor.save_scalers(scaler_path)

    # 8. VISUALIZATIONS
    print("\nGenerating Bayesian visualizations...")

    checkpoint_dir = Path(CONFIG['training']['checkpoint_dir'])

    # Plot training history
    plot_bayesian_training_history(
        history['train_losses'],
        history['val_losses'],
        history.get('train_nlls'),
        history.get('train_kls'),
        history.get('val_nlls'),
        history.get('val_kls'),
        history.get('val_uncertainties'),
        save_path=checkpoint_dir / 'bayesian_training_history.png'
    )

    # Plot predictions with uncertainty
    plot_predictions_with_uncertainty(
        y_test,
        y_pred_mean,
        preprocessor.inverse_transform_output(y_pred_std_scaled),
        output_names=CONFIG['data']['output_columns'],
        confidence_level=0.95,
        save_path=checkpoint_dir / 'predictions_with_uncertainty.png'
    )

    # Plot uncertainty calibration
    plot_uncertainty_calibration(
        y_test,
        y_pred_mean,
        preprocessor.inverse_transform_output(y_pred_std_scaled),
        output_names=CONFIG['data']['output_columns'],
        save_path=checkpoint_dir / 'uncertainty_calibration.png'
    )

    # Plot epistemic uncertainty heatmap
    plot_epistemic_uncertainty_heatmap(
        y_test,
        preprocessor.inverse_transform_output(y_pred_std_scaled),
        output_names=CONFIG['data']['output_columns'],
        save_path=checkpoint_dir / 'epistemic_uncertainty_heatmap.png'
    )

    print("\n" + "="*70)
    print("BAYESIAN TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nFiles saved in: {CONFIG['training']['checkpoint_dir']}/")
    print("  - best_model.pth                      : Best Bayesian model")
    print("  - scalers.pkl                         : Preprocessing scalers")
    print("  - training_history.json               : Training history")
    print("  - bayesian_training_history.png       : Loss plots (ELBO, NLL, KL)")
    print("  - predictions_with_uncertainty.png    : Predictions with uncertainty bands")
    print("  - uncertainty_calibration.png         : Calibration diagnostic")
    print("  - epistemic_uncertainty_heatmap.png   : Uncertainty heatmap")
    print("\n" + "="*70)
    print("Bayesian Neural Network provides:")
    print("  ✓ Point predictions (mean)")
    print("  ✓ Uncertainty estimates (std dev)")
    print("  ✓ Confidence intervals (68%, 95%, 99%)")
    print("  ✓ Epistemic uncertainty quantification")
    print("="*70)


if __name__ == "__main__":
    # Setup seed for reproducibility
    torch.manual_seed(CONFIG['misc']['random_seed'])
    import numpy as np
    import random
    np.random.seed(CONFIG['misc']['random_seed'])
    random.seed(CONFIG['misc']['random_seed'])

    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main()
