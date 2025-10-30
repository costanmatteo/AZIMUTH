"""
Main script for Uncertainty Quantification model training

This script loads data, preprocesses it, creates the uncertainty model,
and trains it to predict both mean values and uncertainties.
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
    UncertaintyPredictor,
    GaussianNLLLoss,
    create_small_uncertainty_model,
    create_medium_uncertainty_model,
    create_large_uncertainty_model
)
from training import UncertaintyTrainer
from utils import (
    calculate_metrics,
    print_metrics,
    plot_training_history,
    plot_predictions_with_uncertainty,
    plot_scatter_with_uncertainty,
    plot_uncertainty_distribution,
    evaluate_prediction_intervals,
    generate_uncertainty_training_report
)

# Import configuration
from configs.example_config import CONFIG


def main():
    # Capture training start timestamp
    training_start_time = datetime.now()

    print("="*70)
    print("UNCERTAINTY QUANTIFICATION TRAINING")
    print("="*70)
    print(f"Training start: {training_start_time.strftime('%d/%m/%Y %H:%M:%S')}")
    print("="*70)
    print("\nThis model predicts both:")
    print("  • μ(x): Mean prediction")
    print("  • σ²(x): Uncertainty (variance)")
    print("\nThe model learns to increase uncertainty in noisy regions")
    print("and decrease it where predictions are more reliable.")
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
        num_workers=0,
        drop_last=True  # Drop last incomplete batch to avoid BatchNorm error
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
    print("\n[4/7] Creating uncertainty model...")
    model_type = CONFIG['model']['model_type']

    if model_type == 'small':
        model = create_small_uncertainty_model(input_dim, output_dim)
    elif model_type == 'medium':
        model = create_medium_uncertainty_model(input_dim, output_dim)
    elif model_type == 'large':
        model = create_large_uncertainty_model(input_dim, output_dim)
    else:  # custom
        model = UncertaintyPredictor(
            input_size=input_dim,
            hidden_sizes=CONFIG['model']['hidden_sizes'],
            output_size=output_dim,
            dropout_rate=CONFIG['model']['dropout_rate'],
            use_batchnorm=CONFIG['model']['use_batchnorm'],
            min_variance=CONFIG['model']['min_variance']
        )

    print(f"  Model type: {model_type}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"  Output: Mean (μ) and Variance (σ²) for each target")

    # 5. CREATE LOSS FUNCTION
    print("\n[5/7] Setting up Gaussian NLL Loss...")
    alpha = CONFIG['training'].get('variance_penalty_alpha', 1.0)
    criterion = GaussianNLLLoss(alpha=alpha, reduction='mean')
    print(f"  Loss: L = 0.5 * ((y - μ)² / σ² + α * log(σ²)) with α={alpha:.3f}")
    print("  This penalizes large errors but accounts for predicted uncertainty")
    if alpha < 1.0:
        print(f"  Note: α={alpha:.3f} < 1 allows model to be more honest about uncertainty")

    # 6. TRAINING
    print("\n[6/7] Starting training...")

    # Setup device
    device = CONFIG['training']['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer = UncertaintyTrainer(
        model,
        criterion,
        device=device,
        learning_rate=CONFIG['training']['learning_rate'],
        weight_decay=CONFIG['training']['weight_decay']
    )

    history = trainer.train(
        train_loader,
        val_loader,
        epochs=CONFIG['training']['epochs'],
        patience=CONFIG['training']['patience'],
        save_dir=CONFIG['training']['checkpoint_dir']
    )

    # 7. EVALUATION
    print("\n[7/7] Evaluation on test set...")

    y_pred_mean, y_pred_variance = trainer.predict(X_test_scaled, return_uncertainty=True)

    # Inverse transform to original scale
    y_pred_mean_orig = preprocessor.inverse_transform_output(y_pred_mean)
    y_test_orig = y_test

    # Need to scale variance appropriately
    # Variance scales with the square of the scaling factor
    if hasattr(preprocessor, 'y_scaler') and preprocessor.y_scaler is not None:
        if hasattr(preprocessor.y_scaler, 'scale_'):
            scale_factors = preprocessor.y_scaler.scale_
            y_pred_variance_orig = y_pred_variance * (scale_factors ** 2)
        else:
            y_pred_variance_orig = y_pred_variance
    else:
        y_pred_variance_orig = y_pred_variance

    # Calculate metrics
    metrics = calculate_metrics(
        y_test_orig,
        y_pred_mean_orig,
        y_pred_variance_orig,
        output_names=CONFIG['data']['output_columns']
    )
    print_metrics(metrics)

    # Evaluate prediction intervals
    print("\n" + "="*70)
    print("PREDICTION INTERVAL COVERAGE")

    coverage_results = evaluate_prediction_intervals(
        y_test_orig,
        y_pred_mean_orig,
        y_pred_variance_orig,
        confidence=CONFIG['uncertainty']['confidence_level']
    )
    print(f"\nExpected coverage: {coverage_results['expected_coverage']:.1f}%")
    print(f"Actual coverage:   {coverage_results['actual_coverage']:.1f}%")
    print(f"Coverage error:    {coverage_results['coverage_error']:.1f}%")
    print(f"Well calibrated:   {'Yes ✓' if coverage_results['well_calibrated'] else 'No ✗'}")
    print("="*70)

    # Save scaler for future use
    scaler_path = Path(CONFIG['training']['checkpoint_dir']) / 'scalers.pkl'
    preprocessor.save_scalers(scaler_path)

    # 8. VISUALIZATIONS
    print("\nGenerating visualizations...")
    checkpoint_dir = Path(CONFIG['training']['checkpoint_dir'])

    # Plot training history
    plot_training_history(
        history['train_losses'],
        history['val_losses'],
        history['train_mse'],
        history['val_mse'],
        save_path=checkpoint_dir / 'training_history.png'
    )

    # Plot predictions with uncertainty bounds
    plot_predictions_with_uncertainty(
        y_test_orig,
        y_pred_mean_orig,
        y_pred_variance_orig,
        output_names=CONFIG['data']['output_columns'],
        save_path=checkpoint_dir / 'predictions_with_uncertainty.png',
        confidence=CONFIG['uncertainty']['confidence_level']
    )

    # Plot scatter with uncertainty coloring
    plot_scatter_with_uncertainty(
        y_test_orig,
        y_pred_mean_orig,
        y_pred_variance_orig,
        output_names=CONFIG['data']['output_columns'],
        save_path=checkpoint_dir / 'scatter_with_uncertainty.png'
    )

    # Plot uncertainty distribution
    plot_uncertainty_distribution(
        y_pred_variance_orig,
        output_names=CONFIG['data']['output_columns'],
        save_path=checkpoint_dir / 'uncertainty_distribution.png'
    )

    # 9. GENERATE PDF REPORT
    print("\nGenerating PDF report...")
    try:
        total_params = sum(p.numel() for p in model.parameters())
        report_path = generate_uncertainty_training_report(
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
            timestamp=training_start_time,
            coverage_results=coverage_results
        )
        print(f"PDF report saved: {report_path}")
    except Exception as e:
        print(f"Error generating PDF: {e}")
        print("Training completed successfully anyway.")

    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nFiles saved in: {CONFIG['training']['checkpoint_dir']}/")
    print("  - best_model.pth                    : Best model checkpoint")
    print("  - scalers.pkl                       : Scaler for preprocessing")
    print("  - training_history.json             : Training history")
    print("  - training_history.png              : Loss plots")
    print("  - predictions_with_uncertainty.png  : Predictions with bounds")
    print("  - scatter_with_uncertainty.png      : Scatter plot colored by uncertainty")
    print("  - uncertainty_distribution.png      : Distribution of uncertainties")
    print("  - training_report.pdf               : Comprehensive PDF report")
    print("\n" + "="*70)



if __name__ == "__main__":
    # Setup seed for reproducibility
    torch.manual_seed(CONFIG['misc']['random_seed'])
    import numpy as np
    np.random.seed(CONFIG['misc']['random_seed'])

    main()
