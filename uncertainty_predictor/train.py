"""
Main script for Uncertainty Quantification model training

This script loads data, preprocesses it, creates the uncertainty model,
and trains it to predict both mean values and uncertainties.
"""

import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data import load_csv_data, generate_scm_data, DataPreprocessor, MachineryDataset
from models import (
    UncertaintyPredictor,
    GaussianNLLLoss,
    EnergyScoreLoss,
    create_small_uncertainty_model,
    create_medium_uncertainty_model,
    create_large_uncertainty_model,
    EnsembleUncertaintyPredictor,
    create_ensemble_model,
    SWAGUncertaintyPredictor,
    create_swag_model
)
from training import UncertaintyTrainer, EnsembleTrainer, SWAGTrainer
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

    # Determine data source: CSV file or SCM synthetic data
    csv_path = CONFIG['data'].get('csv_path')
    use_scm = CONFIG['data'].get('use_scm', False)

    # Check if we should use CSV or SCM
    use_csv = csv_path is not None and Path(csv_path).exists()

    if use_csv:
        # Load from CSV file
        print(f"Using CSV file: {csv_path}")
        try:
            X, y = load_csv_data(
                csv_path,
                CONFIG['data']['input_columns'],
                CONFIG['data']['output_columns']
            )
            input_columns = CONFIG['data']['input_columns']
            output_columns = CONFIG['data']['output_columns']
            env_columns = CONFIG['data'].get('env_columns', [])
            control_columns = [c for c in input_columns if c not in env_columns]
        except FileNotFoundError:
            print(f"\nERROR: File not found: {csv_path}")
            print("\nTo test the system, create a sample CSV file with:")
            print("  - Input columns: " + ", ".join(CONFIG['data']['input_columns']))
            print("  - Output columns: " + ", ".join(CONFIG['data']['output_columns']))
            print("\nOr set csv_path to None in configs/example_config.py to use SCM synthetic data.")
            return
    elif use_scm:
        # Generate synthetic data using SCM
        print("CSV file not specified or not found. Using SCM synthetic data generation...")
        scm_config = CONFIG['data'].get('scm', {})
        n_samples = scm_config.get('n_samples', 5000)
        seed = scm_config.get('seed', 42)
        dataset_type = scm_config.get('dataset_type', 'one_to_one_ct')

        X, y, input_columns, output_columns, E, env_columns = generate_scm_data(
            n_samples=n_samples,
            seed=seed,
            dataset_type=dataset_type,
            save_graph_to=CONFIG['training']['checkpoint_dir']
        )
        # Identify which input columns are controllable vs environmental
        control_columns = [c for c in input_columns if c not in env_columns]
    else:
        print("\nERROR: No data source specified.")
        print("Either provide a valid csv_path or enable use_scm in configs/example_config.py")
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

    # Determine uncertainty method (new config style)
    uncertainty_method = CONFIG['model'].get('uncertainty_method', 'single')

    # Backward compatibility: check old use_ensemble flag
    if CONFIG['model'].get('use_ensemble', False) and uncertainty_method == 'single':
        uncertainty_method = 'ensemble'

    use_ensemble = (uncertainty_method == 'ensemble')
    use_swag = (uncertainty_method == 'swag')

    if use_ensemble:
        # Create Deep Ensemble model
        n_ensemble_models = CONFIG['model'].get('n_ensemble_models', 5)
        print(f"  Using Deep Ensemble with {n_ensemble_models} models")

        model = EnsembleUncertaintyPredictor(
            input_size=input_dim,
            hidden_sizes=CONFIG['model']['hidden_sizes'],
            output_size=output_dim,
            n_models=n_ensemble_models,
            dropout_rate=CONFIG['model']['dropout_rate'],
            use_batchnorm=CONFIG['model']['use_batchnorm'],
            min_variance=CONFIG['model']['min_variance']
        )
        print(f"  Model type: ensemble ({model_type} base architecture)")
        print(f"  Models in ensemble: {n_ensemble_models}")
        print(f"  Parameters per model: {sum(p.numel() for p in model.models[0].parameters())}")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"  Output: Mean (μ), Aleatoric (σ²_a), Epistemic (σ²_e) uncertainty")

    elif use_swag:
        # Create SWAG model (wraps a base UncertaintyPredictor)
        swag_max_rank = CONFIG['model'].get('swag_max_rank', 20)
        print(f"  Using SWAG (Stochastic Weight Averaging - Gaussian)")

        # Create base model
        if model_type == 'small':
            base_model = create_small_uncertainty_model(input_dim, output_dim)
        elif model_type == 'medium':
            base_model = create_medium_uncertainty_model(input_dim, output_dim)
        elif model_type == 'large':
            base_model = create_large_uncertainty_model(input_dim, output_dim)
        else:  # custom
            base_model = UncertaintyPredictor(
                input_size=input_dim,
                hidden_sizes=CONFIG['model']['hidden_sizes'],
                output_size=output_dim,
                dropout_rate=CONFIG['model']['dropout_rate'],
                use_batchnorm=CONFIG['model']['use_batchnorm'],
                min_variance=CONFIG['model']['min_variance']
            )

        # Wrap with SWAG
        model = SWAGUncertaintyPredictor(base_model, max_rank=swag_max_rank)

        print(f"  Model type: swag ({model_type} base architecture)")
        print(f"  Base model parameters: {sum(p.numel() for p in base_model.parameters())}")
        print(f"  Low-rank covariance dimension: {swag_max_rank}")
        print(f"  SWA start: {CONFIG['model'].get('swag_start_epoch', 0.5)*100:.0f}% of training")
        print(f"  Output: Mean (μ), Aleatoric (σ²_a), Epistemic (σ²_e) uncertainty")

    else:
        # Create single model
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
    print("\n[5/7] Setting up loss function...")
    loss_type = CONFIG['training'].get('loss_type', 'gaussian_nll')

    if loss_type == 'gaussian_nll':
        print("Using Gaussian NLL Loss...")
        alpha = CONFIG['training'].get('variance_penalty_alpha', 1.0)
        criterion = GaussianNLLLoss(alpha=alpha, reduction='mean')
        print(f"  Loss: L = 0.5 * ((y - μ)² / σ² + α * log(σ²)) with α={alpha:.3f}")
        print("  This penalizes large errors but accounts for predicted uncertainty")
        if alpha < 1.0:
            print(f"  Note: α={alpha:.3f} < 1 allows model to be more honest about uncertainty")
    elif loss_type == 'energy_score':
        print("Using Energy Score Loss...")
        n_samples = CONFIG['training'].get('energy_score_samples', 50)
        beta = CONFIG['training'].get('energy_score_beta', 1.0)
        criterion = EnergyScoreLoss(n_samples=n_samples, beta=beta, reduction='mean')
        print(f"  Loss: ES = E[|X - y|] - β/2 * E[|X - X'|] with β={beta:.3f}")
        print(f"  Using {n_samples} Monte Carlo samples for estimation")
        print("  Energy Score is a proper scoring rule without distribution assumptions")
        if beta < 1.0:
            print(f"  Note: β={beta:.3f} < 1 allows wider uncertainty distributions")
        elif beta > 1.0:
            print(f"  Note: β={beta:.3f} > 1 encourages tighter uncertainty distributions")
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Choose 'gaussian_nll' or 'energy_score'")

    # 6. TRAINING
    print("\n[6/7] Starting training...")

    # Setup device
    device = CONFIG['training']['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if use_ensemble:
        # Use EnsembleTrainer for Deep Ensemble
        ensemble_base_seed = CONFIG['model'].get('ensemble_base_seed', 42)
        trainer = EnsembleTrainer(
            model,
            criterion,
            device=device,
            learning_rate=CONFIG['training']['learning_rate'],
            weight_decay=CONFIG['training']['weight_decay'],
            base_seed=ensemble_base_seed
        )
    elif use_swag:
        # Use SWAGTrainer for SWAG
        trainer = SWAGTrainer(
            model,
            criterion,
            device=device,
            learning_rate=CONFIG['training']['learning_rate'],
            swa_learning_rate=CONFIG['model'].get('swag_learning_rate', 0.01),
            weight_decay=CONFIG['training']['weight_decay'],
            swa_start_epoch=CONFIG['model'].get('swag_start_epoch', 0.5),
            swa_freq=CONFIG['model'].get('swag_collection_freq', 1),
            min_samples=CONFIG['model'].get('swag_min_samples', 20)
        )
    else:
        # Use standard UncertaintyTrainer
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

    # Get predictions (different return format for ensemble/swag vs single model)
    if use_ensemble or use_swag:
        n_samples = CONFIG['model'].get('swag_n_samples', 30) if use_swag else None
        if use_swag:
            y_pred_mean, y_pred_variance, y_pred_aleatoric, y_pred_epistemic = \
                trainer.predict(X_test_scaled, return_uncertainty=True, n_samples=n_samples)
        else:
            y_pred_mean, y_pred_variance, y_pred_aleatoric, y_pred_epistemic = \
                trainer.predict(X_test_scaled, return_uncertainty=True)
    else:
        y_pred_mean, y_pred_variance = trainer.predict(X_test_scaled, return_uncertainty=True)
        y_pred_aleatoric = None
        y_pred_epistemic = None

    # Inverse transform to original scale
    y_pred_mean_orig = preprocessor.inverse_transform_output(y_pred_mean)
    y_test_orig = y_test

    # Need to scale variance appropriately
    # Variance scales with the square of the scaling factor
    if hasattr(preprocessor, 'y_scaler') and preprocessor.y_scaler is not None:
        if hasattr(preprocessor.y_scaler, 'scale_'):
            scale_factors = preprocessor.y_scaler.scale_
            y_pred_variance_orig = y_pred_variance * (scale_factors ** 2)
            if use_ensemble or use_swag:
                y_pred_aleatoric_orig = y_pred_aleatoric * (scale_factors ** 2)
                y_pred_epistemic_orig = y_pred_epistemic * (scale_factors ** 2)
        else:
            y_pred_variance_orig = y_pred_variance
            if use_ensemble or use_swag:
                y_pred_aleatoric_orig = y_pred_aleatoric
                y_pred_epistemic_orig = y_pred_epistemic
    else:
        y_pred_variance_orig = y_pred_variance
        if use_ensemble or use_swag:
            y_pred_aleatoric_orig = y_pred_aleatoric
            y_pred_epistemic_orig = y_pred_epistemic

    # Also get predictions on training set for visualization
    if use_ensemble or use_swag:
        if use_swag:
            y_train_pred_mean, y_train_pred_variance, y_train_aleatoric, y_train_epistemic = \
                trainer.predict(X_train_scaled, return_uncertainty=True, n_samples=n_samples)
        else:
            y_train_pred_mean, y_train_pred_variance, y_train_aleatoric, y_train_epistemic = \
                trainer.predict(X_train_scaled, return_uncertainty=True)
    else:
        y_train_pred_mean, y_train_pred_variance = trainer.predict(X_train_scaled, return_uncertainty=True)

    y_train_pred_mean_orig = preprocessor.inverse_transform_output(y_train_pred_mean)
    y_train_orig = y_train
    y_train_aleatoric_orig = None
    y_train_epistemic_orig = None

    if hasattr(preprocessor, 'y_scaler') and preprocessor.y_scaler is not None:
        if hasattr(preprocessor.y_scaler, 'scale_'):
            scale_factors = preprocessor.y_scaler.scale_
            y_train_pred_variance_orig = y_train_pred_variance * (scale_factors ** 2)
            # Also scale aleatoric and epistemic variances
            if use_ensemble or use_swag:
                y_train_aleatoric_orig = y_train_aleatoric * (scale_factors ** 2)
                y_train_epistemic_orig = y_train_epistemic * (scale_factors ** 2)
        else:
            y_train_pred_variance_orig = y_train_pred_variance
            if use_ensemble or use_swag:
                y_train_aleatoric_orig = y_train_aleatoric
                y_train_epistemic_orig = y_train_epistemic
    else:
        y_train_pred_variance_orig = y_train_pred_variance
        if use_ensemble or use_swag:
            y_train_aleatoric_orig = y_train_aleatoric
            y_train_epistemic_orig = y_train_epistemic

    # Calculate metrics
    metrics = calculate_metrics(
        y_test_orig,
        y_pred_mean_orig,
        y_pred_variance_orig,
        output_names=output_columns
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

    # Print ensemble/SWAG-specific uncertainty decomposition
    if use_ensemble or use_swag:
        method_name = "Ensemble" if use_ensemble else "SWAG"
        print("\n" + "-"*50)
        print(f"UNCERTAINTY DECOMPOSITION ({method_name})")
        print("-"*50)
        mean_aleatoric = np.mean(y_pred_aleatoric_orig)
        mean_epistemic = np.mean(y_pred_epistemic_orig)
        mean_total = np.mean(y_pred_variance_orig)
        epistemic_ratio = mean_epistemic / mean_total * 100 if mean_total > 0 else 0

        print(f"Mean Aleatoric (data noise):     {mean_aleatoric:.6f}")
        print(f"Mean Epistemic (model uncertainty): {mean_epistemic:.6f}")
        print(f"Mean Total Variance:             {mean_total:.6f}")
        print(f"Epistemic Ratio:                 {epistemic_ratio:.1f}%")

        # Add to coverage_results for report
        coverage_results['mean_aleatoric'] = float(mean_aleatoric)
        coverage_results['mean_epistemic'] = float(mean_epistemic)
        coverage_results['epistemic_ratio'] = float(epistemic_ratio)

    print("="*70)

    # Save scaler for future use
    scaler_path = Path(CONFIG['training']['checkpoint_dir']) / 'scalers.pkl'
    preprocessor.save_scalers(scaler_path)

    # Save column metadata: which inputs are controllable vs environmental
    import json
    column_meta = {
        'input_columns': input_columns,
        'output_columns': output_columns,
        'control_columns': control_columns,
        'env_columns': env_columns,
    }
    meta_path = Path(CONFIG['training']['checkpoint_dir']) / 'column_metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(column_meta, f, indent=2)
    print(f"  Column metadata saved: {meta_path}")
    if column_meta['env_columns']:
        print(f"    Control columns: {column_meta['control_columns']}")
        print(f"    Environmental columns (not controllable): {column_meta['env_columns']}")

    # 8. VISUALIZATIONS
    print("\nGenerating visualizations...")
    checkpoint_dir = Path(CONFIG['training']['checkpoint_dir'])

    # Plot training history (with SWA start marker if using SWAG)
    plot_training_history(
        history['train_losses'],
        history['val_losses'],
        history['train_mse'],
        history['val_mse'],
        save_path=checkpoint_dir / 'training_history.png',
        swa_start_epoch=history.get('swa_start_epoch') if use_swag else None
    )

    # Plot predictions with uncertainty bounds (test set)
    # Pass aleatoric/epistemic for ensemble mode to show decomposition
    plot_predictions_with_uncertainty(
        y_test_orig,
        y_pred_mean_orig,
        y_pred_variance_orig,
        output_names=output_columns,
        save_path=checkpoint_dir / 'predictions_with_uncertainty.png',
        confidence=CONFIG['uncertainty']['confidence_level'],
        y_pred_aleatoric=y_pred_aleatoric_orig if (use_ensemble or use_swag) else None,
        y_pred_epistemic=y_pred_epistemic_orig if (use_ensemble or use_swag) else None
    )

    # Plot predictions with uncertainty bounds (training set)
    plot_predictions_with_uncertainty(
        y_train_orig,
        y_train_pred_mean_orig,
        y_train_pred_variance_orig,
        output_names=output_columns,
        save_path=checkpoint_dir / 'training_predictions_with_uncertainty.png',
        confidence=CONFIG['uncertainty']['confidence_level'],
        y_pred_aleatoric=y_train_aleatoric_orig if (use_ensemble or use_swag) else None,
        y_pred_epistemic=y_train_epistemic_orig if (use_ensemble or use_swag) else None
    )

    # Plot scatter with uncertainty coloring
    plot_scatter_with_uncertainty(
        y_test_orig,
        y_pred_mean_orig,
        y_pred_variance_orig,
        output_names=output_columns,
        save_path=checkpoint_dir / 'scatter_with_uncertainty.png'
    )

    # Plot uncertainty distribution
    plot_uncertainty_distribution(
        y_pred_variance_orig,
        output_names=output_columns,
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
