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

from data import (
    load_csv_data,
    generate_scm_data,
    generate_conditional_scm_data,
    prepare_conditional_tensors,
    create_conditional_collate_fn,
    DataPreprocessor,
    MachineryDataset,
    ConditionalMachineryDataset
)
from models import (
    UncertaintyPredictor,
    GaussianNLLLoss,
    EnergyScoreLoss,
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

    # Check if conditioning is enabled
    conditioning_enabled = CONFIG.get('conditioning', {}).get('enable', False)

    # Determine data source: CSV file or SCM synthetic data
    csv_path = CONFIG['data'].get('csv_path')
    use_scm = CONFIG['data'].get('use_scm', False)

    # Check if we should use CSV or SCM
    use_csv = csv_path is not None and Path(csv_path).exists()

    # For CSV data with conditioning, we'll need to load conditioning columns separately
    # For now, we focus on SCM data which has built-in support for conditioning
    df = None
    conditioning_columns = None

    if use_csv:
        # Load from CSV file (standard mode, no conditioning support yet)
        print(f"Using CSV file: {csv_path}")
        if conditioning_enabled:
            print("WARNING: Conditioning is enabled but CSV mode doesn't support it yet.")
            print("         Disabling conditioning for this run.")
            conditioning_enabled = False

        try:
            X, y = load_csv_data(
                csv_path,
                CONFIG['data']['input_columns'],
                CONFIG['data']['output_columns']
            )
            input_columns = CONFIG['data']['input_columns']
            output_columns = CONFIG['data']['output_columns']
        except FileNotFoundError:
            print(f"\nERROR: File not found: {csv_path}")
            print("\nTo test the system, create a sample CSV file with:")
            print("  - Input columns: " + ", ".join(CONFIG['data']['input_columns']))
            print("  - Output columns: " + ", ".join(CONFIG['data']['output_columns']))
            print("\nOr set csv_path to None in configs/example_config.py to use SCM synthetic data.")
            return
    elif use_scm:
        # Generate synthetic data using SCM
        scm_config = CONFIG['data'].get('scm', {})
        n_samples = scm_config.get('n_samples', 5000)
        seed = scm_config.get('seed', 42)
        process_selection = scm_config.get('process_selection', 'all')
        add_env_vars = scm_config.get('add_env_vars', True)

        # Validate configuration
        if process_selection == 'all' and not conditioning_enabled:
            print("\nERROR: process_selection='all' requires conditioning.enable=True")
            print("Either set conditioning.enable=True or choose a single process")
            return

        if conditioning_enabled and not add_env_vars:
            print("WARNING: Conditioning is enabled but add_env_vars=False")
            print("         Environment variables will not be available for conditioning")

        if conditioning_enabled:
            # Use conditional data generation
            print("CSV file not specified. Using conditional SCM synthetic data generation...")
            df, input_columns, output_columns, conditioning_columns = generate_conditional_scm_data(
                process_selection=process_selection,
                n_samples=n_samples,
                add_env_vars=add_env_vars,
                seed=seed
            )
            X = None  # Will be extracted later with conditioning vars
            y = None
        else:
            # Use standard data generation (backward compatible)
            print("CSV file not specified. Using SCM synthetic data generation (standard mode)...")
            dataset_type = scm_config.get('dataset_type', 'one_to_one_ct')
            X, y, input_columns, output_columns = generate_scm_data(
                n_samples=n_samples,
                seed=seed,
                dataset_type=dataset_type,
                save_graph_to=CONFIG['training']['checkpoint_dir']
            )
    else:
        print("\nERROR: No data source specified.")
        print("Either provide a valid csv_path or enable use_scm in configs/example_config.py")
        return

    if df is not None:
        print(f"  Loaded {len(df)} samples")
    else:
        print(f"  Loaded {len(X)} samples")

    # 2. PREPROCESSING
    print("\n[2/7] Preprocessing data...")
    preprocessor = DataPreprocessor(scaling_method=CONFIG['data']['scaling_method'])

    if conditioning_enabled and df is not None:
        # Conditional mode: split DataFrame then extract tensors
        from sklearn.model_selection import train_test_split
        import pandas as pd

        # First split
        df_temp, df_test = train_test_split(
            df,
            test_size=CONFIG['data']['test_size'],
            random_state=CONFIG['data']['random_state']
        )

        # Second split (train/val from remaining data)
        val_ratio = CONFIG['data']['val_size'] / (1 - CONFIG['data']['test_size'])
        df_train, df_val = train_test_split(
            df_temp,
            test_size=val_ratio,
            random_state=CONFIG['data']['random_state']
        )

        print(f"  Train set: {len(df_train)} samples")
        print(f"  Validation set: {len(df_val)} samples")
        print(f"  Test set: {len(df_test)} samples")

        # Extract tensors from DataFrames
        data_train = prepare_conditional_tensors(df_train, input_columns, output_columns, conditioning_columns)
        data_val = prepare_conditional_tensors(df_val, input_columns, output_columns, conditioning_columns)
        data_test = prepare_conditional_tensors(df_test, input_columns, output_columns, conditioning_columns)

        # Fit and transform physical features (X, y) only
        X_train_scaled, y_train_scaled = preprocessor.fit_transform(data_train['X'], data_train['y'])
        X_val_scaled, y_val_scaled = preprocessor.transform(data_val['X'], data_val['y'])
        X_test_scaled, y_test_scaled = preprocessor.transform(data_test['X'], data_test['y'])

        # Update data dicts with scaled values
        data_train['X'] = X_train_scaled
        data_train['y'] = y_train_scaled
        data_val['X'] = X_val_scaled
        data_val['y'] = y_val_scaled
        data_test['X'] = X_test_scaled
        data_test['y'] = y_test_scaled

        # Keep unscaled test data for final evaluation
        X_test_unscaled = data_test['X']
        y_test_unscaled = data_val['y']  # Will use original from df_test later

    else:
        # Standard mode: split arrays directly
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

        # For standard mode, create simple data dicts
        data_train = {'X': X_train_scaled, 'y': y_train_scaled}
        data_val = {'X': X_val_scaled, 'y': y_val_scaled}
        data_test = {'X': X_test_scaled, 'y': y_test_scaled}

    # 3. CREATE DATASET AND DATALOADER
    print("\n[3/7] Creating PyTorch datasets...")

    if conditioning_enabled:
        # Create conditional datasets
        train_dataset = ConditionalMachineryDataset(data_train)
        val_dataset = ConditionalMachineryDataset(data_val)
        test_dataset = ConditionalMachineryDataset(data_test)

        # Create conditional collate function
        collate_fn = create_conditional_collate_fn(conditioning_enabled=True)

        # Print conditioning statistics
        train_stats = train_dataset.get_statistics()
        print(f"  Conditional training enabled:")
        if 'num_processes' in train_stats:
            print(f"    - Processes: {train_stats['num_processes']}")
            print(f"    - Distribution: {train_stats['process_distribution']}")
        if 'env_continuous_dim' in train_stats:
            print(f"    - Continuous env vars: {train_stats['env_continuous_dim']}")
        if 'env_categorical_vars' in train_stats:
            print(f"    - Categorical env vars: {train_stats['env_categorical_vars']}")
    else:
        # Create standard datasets
        train_dataset = MachineryDataset(data_train['X'], data_train['y'])
        val_dataset = MachineryDataset(data_val['X'], data_val['y'])
        test_dataset = MachineryDataset(data_test['X'], data_test['y'])
        collate_fn = None  # Use default collate

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['training']['batch_size'],
        shuffle=True,
        num_workers=0,
        drop_last=True,  # Drop last incomplete batch to avoid BatchNorm error
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    input_dim = train_dataset.get_input_dim()
    output_dim = train_dataset.get_output_dim()
    print(f"  Input dimension: {input_dim}")
    print(f"  Output dimension: {output_dim}")

    # 4. CREATE MODEL
    print("\n[4/7] Creating uncertainty model...")
    model_type = CONFIG['model']['model_type']

    # Prepare conditioning config if enabled
    conditioning_config = None
    if conditioning_enabled:
        conditioning_config = CONFIG['conditioning'].copy()
        print(f"  Conditional model configuration:")
        print(f"    - Processes: {conditioning_config.get('num_processes', 4)}")
        print(f"    - Process embedding dim: {conditioning_config.get('d_proc', 16)}")
        print(f"    - Context vector dim: {conditioning_config.get('d_context', 64)}")
        print(f"    - Normalization type: {conditioning_config.get('norm_type', 'conditional_layer_norm')}")

    if model_type == 'small':
        model = create_small_uncertainty_model(input_dim, output_dim, conditioning_config=conditioning_config)
    elif model_type == 'medium':
        model = create_medium_uncertainty_model(input_dim, output_dim, conditioning_config=conditioning_config)
    elif model_type == 'large':
        model = create_large_uncertainty_model(input_dim, output_dim, conditioning_config=conditioning_config)
    else:  # custom
        model = UncertaintyPredictor(
            input_size=input_dim,
            hidden_sizes=CONFIG['model']['hidden_sizes'],
            output_size=output_dim,
            dropout_rate=CONFIG['model']['dropout_rate'],
            use_batchnorm=CONFIG['model']['use_batchnorm'],
            min_variance=CONFIG['model']['min_variance'],
            conditioning_config=conditioning_config
        )

    print(f"  Model type: {model_type}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    if conditioning_enabled:
        print(f"  Mode: Conditional (multi-process)")
    else:
        print(f"  Mode: Standard (single-process)")
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

    trainer = UncertaintyTrainer(
        model,
        criterion,
        device=device,
        learning_rate=CONFIG['training']['learning_rate'],
        weight_decay=CONFIG['training']['weight_decay'],
        conditioning_enabled=conditioning_enabled
    )

    history = trainer.train(
        train_loader,
        val_loader,
        epochs=CONFIG['training']['epochs'],
        patience=CONFIG['training']['patience'],
        save_dir=CONFIG['training']['checkpoint_dir'],
        compute_per_process_metrics=conditioning_enabled  # Enable per-process metrics if conditioning
    )

    # Print per-process metrics summary if available
    if conditioning_enabled and hasattr(trainer, 'per_process_metrics') and trainer.per_process_metrics:
        print("\n" + "="*70)
        print("PER-PROCESS VALIDATION METRICS (Final Epoch)")
        print("="*70)
        process_names = ['Laser', 'Plasma', 'Galvanic', 'Microetch']
        for process_id in range(4):
            if process_id in trainer.per_process_metrics:
                metrics = trainer.per_process_metrics[process_id]
                if metrics['val_mse']:
                    final_mse = metrics['val_mse'][-1]
                    final_var = metrics['val_variance'][-1]
                    print(f"  {process_names[process_id]:12s} (ID={process_id}): "
                          f"MSE={final_mse:.6f}, Var={final_var:.6f}")
        print("="*70)

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

    # Also get predictions on training set for visualization
    y_train_pred_mean, y_train_pred_variance = trainer.predict(X_train_scaled, return_uncertainty=True)
    y_train_pred_mean_orig = preprocessor.inverse_transform_output(y_train_pred_mean)
    y_train_orig = y_train

    if hasattr(preprocessor, 'y_scaler') and preprocessor.y_scaler is not None:
        if hasattr(preprocessor.y_scaler, 'scale_'):
            scale_factors = preprocessor.y_scaler.scale_
            y_train_pred_variance_orig = y_train_pred_variance * (scale_factors ** 2)
        else:
            y_train_pred_variance_orig = y_train_pred_variance
    else:
        y_train_pred_variance_orig = y_train_pred_variance

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

    # Plot predictions with uncertainty bounds (test set)
    plot_predictions_with_uncertainty(
        y_test_orig,
        y_pred_mean_orig,
        y_pred_variance_orig,
        output_names=output_columns,
        save_path=checkpoint_dir / 'predictions_with_uncertainty.png',
        confidence=CONFIG['uncertainty']['confidence_level']
    )

    # Plot predictions with uncertainty bounds (training set)
    plot_predictions_with_uncertainty(
        y_train_orig,
        y_train_pred_mean_orig,
        y_train_pred_variance_orig,
        output_names=output_columns,
        save_path=checkpoint_dir / 'training_predictions_with_uncertainty.png',
        confidence=CONFIG['uncertainty']['confidence_level']
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
