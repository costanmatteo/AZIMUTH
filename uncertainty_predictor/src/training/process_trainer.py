"""
Trainer per uncertainty predictor di un singolo processo.

Importa e usa la classe UncertaintyPredictor esistente da uncertainty_predictor/
GENERA REPORT PDF con metriche e visualizzazioni.

Riceve DataLoader già pronti (costruiti dal chiamante, es. train_predictor.py).
"""

import sys
from pathlib import Path
import random
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
from datetime import datetime
import pickle
import importlib.util

# Add uncertainty_predictor to path
REPO_ROOT = Path(__file__).parent.parent.parent.parent
UNCERTAINTY_PREDICTOR_PATH = REPO_ROOT / 'uncertainty_predictor'

# CRITICAL: Add uncertainty_predictor to sys.path FIRST
# This allows nested imports (like scm_ds.datasets) to work when calling
# functions from the loaded modules
if str(UNCERTAINTY_PREDICTOR_PATH) not in sys.path:
    sys.path.insert(0, str(UNCERTAINTY_PREDICTOR_PATH))

# Load modules from uncertainty_predictor explicitly
# Register them in sys.modules for pickle compatibility
spec_nn = importlib.util.spec_from_file_location(
    "uncertainty_nn",
    UNCERTAINTY_PREDICTOR_PATH / "src" / "models" / "uncertainty_nn.py"
)
uncertainty_nn = importlib.util.module_from_spec(spec_nn)
sys.modules['uncertainty_nn'] = uncertainty_nn  # Register for pickle
spec_nn.loader.exec_module(uncertainty_nn)
UncertaintyPredictor = uncertainty_nn.UncertaintyPredictor
GaussianNLLLoss = uncertainty_nn.GaussianNLLLoss
EnsembleUncertaintyPredictor = uncertainty_nn.EnsembleUncertaintyPredictor
SWAGUncertaintyPredictor = uncertainty_nn.SWAGUncertaintyPredictor

spec_trainer = importlib.util.spec_from_file_location(
    "uncertainty_trainer",
    UNCERTAINTY_PREDICTOR_PATH / "src" / "training" / "uncertainty_trainer.py"
)
uncertainty_trainer = importlib.util.module_from_spec(spec_trainer)
sys.modules['uncertainty_trainer'] = uncertainty_trainer  # Register for pickle
spec_trainer.loader.exec_module(uncertainty_trainer)
UncertaintyTrainer = uncertainty_trainer.UncertaintyTrainer
EnsembleTrainer = uncertainty_trainer.EnsembleTrainer
SWAGTrainer = uncertainty_trainer.SWAGTrainer

spec_preprocessing = importlib.util.spec_from_file_location(
    "preprocessing",
    UNCERTAINTY_PREDICTOR_PATH / "src" / "data" / "preprocessing.py"
)
preprocessing = importlib.util.module_from_spec(spec_preprocessing)
sys.modules['preprocessing'] = preprocessing  # Register for pickle
spec_preprocessing.loader.exec_module(preprocessing)
DataPreprocessor = preprocessing.DataPreprocessor

spec_report = importlib.util.spec_from_file_location(
    "report_generator",
    UNCERTAINTY_PREDICTOR_PATH / "src" / "utils" / "report_generator.py"
)
report_gen = importlib.util.module_from_spec(spec_report)
spec_report.loader.exec_module(report_gen)
generate_uncertainty_training_report = report_gen.generate_uncertainty_training_report

spec_metrics = importlib.util.spec_from_file_location(
    "uq_metrics",
    UNCERTAINTY_PREDICTOR_PATH / "src" / "utils" / "metrics.py"
)
uq_metrics = importlib.util.module_from_spec(spec_metrics)
spec_metrics.loader.exec_module(uq_metrics)

spec_viz = importlib.util.spec_from_file_location(
    "uq_viz",
    UNCERTAINTY_PREDICTOR_PATH / "src" / "utils" / "visualization.py"
)
uq_viz = importlib.util.module_from_spec(spec_viz)
spec_viz.loader.exec_module(uq_viz)


def train_single_process(process_config, train_loader, val_loader, test_loader,
                         preprocessor, device='auto', verbose=True, seed=42):
    """
    Addestra uncertainty predictor per un processo e genera report.

    Riceve DataLoader già pronti (costruiti dal chiamante).

    Args:
        process_config (dict): Config da PROCESSES
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        test_loader (DataLoader): Test data loader
        preprocessor (DataPreprocessor): Fitted preprocessor (per unscaling)
        device (str): 'cuda', 'cpu', o 'auto'
        verbose (bool): Print progress
        seed (int): Random seed

    Returns:
        dict: {
            'model_path': str,
            'scaler_path': str,
            'metrics': dict,
            'history': dict,
            'report_path': str  # Path al PDF report generato
        }
    """
    # Setup device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set random seeds and enforce deterministic behavior
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Import global uncertainty config
    from configs.uncertainty_config import GLOBAL_UNCERTAINTY_CONFIG

    # Extract config
    process_name = process_config['name']
    scm_dataset_type = process_config['scm_dataset_type']
    input_dim = process_config['input_dim']
    output_dim = process_config['output_dim']
    input_labels = process_config['input_labels']
    output_labels = process_config['output_labels']

    # Merge global uncertainty config with process-specific config
    # Process-specific values override global defaults
    model_config = {**GLOBAL_UNCERTAINTY_CONFIG, **process_config['uncertainty_predictor']['model']}
    training_config = process_config['uncertainty_predictor']['training']
    checkpoint_dir = Path(process_config['checkpoint_dir'])

    # Create checkpoint directory
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Training Uncertainty Predictor for Process: {process_name.upper()}")
        print(f"{'='*70}")

    # Compute dataset sizes from loaders
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)

    if verbose:
        print(f"\n[1/7] Data received from caller:")
        print(f"  Train: {n_train} samples")
        print(f"  Val:   {n_val} samples")
        print(f"  Test:  {n_test} samples")

    # Extract scaled training data for visualization later (single pass to avoid shuffle mismatch)
    _train_batches = list(train_loader)
    X_train_scaled = torch.cat([batch[0] for batch in _train_batches]).numpy()
    y_train = preprocessor.output_scaler.inverse_transform(
        torch.cat([batch[1] for batch in _train_batches]).numpy()
    )

    # Sanity check: data shape must agree with process_config metadata.
    # Catching mismatches here (before training) gives a much clearer error
    # than a downstream IndexError in the visualization code.
    actual_output_dim = y_train.shape[1] if y_train.ndim == 2 else 1
    if actual_output_dim != output_dim or actual_output_dim != len(output_labels):
        raise ValueError(
            f"Shape mismatch for process '{process_name}': loaded dataset has "
            f"output_dim={actual_output_dim}, but process config declares "
            f"output_dim={output_dim} with output_labels={output_labels} "
            f"(len={len(output_labels)}). The on-disk dataset is likely stale. "
            f"Regenerate it with generate_dataset.py."
        )

    # Create UncertaintyPredictor (Ensemble, SWAG, or Single)
    # Determine uncertainty method
    uncertainty_method = model_config.get('uncertainty_method', 'single')

    # Backward compatibility: check old use_ensemble flag
    if model_config.get('use_ensemble', False) and uncertainty_method == 'single':
        uncertainty_method = 'ensemble'

    use_ensemble = (uncertainty_method == 'ensemble')
    use_swag = (uncertainty_method == 'swag')

    if use_ensemble:
        n_ensemble_models = model_config.get('n_ensemble_models', 5)
        if verbose:
            print(f"\n[2/7] Creating EnsembleUncertaintyPredictor with {n_ensemble_models} models...")

        model = EnsembleUncertaintyPredictor(
            input_size=input_dim,
            output_size=output_dim,
            hidden_sizes=model_config['hidden_sizes'],
            n_models=n_ensemble_models,
            dropout_rate=model_config['dropout_rate'],
            use_batchnorm=model_config['use_batchnorm'],
            min_variance=model_config['min_variance']
        )

        total_params = sum(p.numel() for p in model.parameters())
        params_per_model = total_params // n_ensemble_models
        if verbose:
            print(f"  Models in ensemble: {n_ensemble_models}")
            print(f"  Parameters per model: {params_per_model:,}")
            print(f"  Total parameters: {total_params:,}")

    elif use_swag:
        swag_max_rank = model_config.get('swag_max_rank', 20)
        if verbose:
            print(f"\n[2/7] Creating SWAGUncertaintyPredictor...")

        # Create base model
        base_model = UncertaintyPredictor(
            input_size=input_dim,
            output_size=output_dim,
            hidden_sizes=model_config['hidden_sizes'],
            dropout_rate=model_config['dropout_rate'],
            use_batchnorm=model_config['use_batchnorm'],
            min_variance=model_config['min_variance']
        )

        # Wrap with SWAG
        model = SWAGUncertaintyPredictor(base_model, max_rank=swag_max_rank)

        total_params = sum(p.numel() for p in base_model.parameters())
        if verbose:
            print(f"  Base model parameters: {total_params:,}")
            print(f"  Low-rank covariance dimension: {swag_max_rank}")
            print(f"  SWA start: {model_config.get('swag_start_epoch', 0.5)*100:.0f}% of training")

    else:
        if verbose:
            print(f"\n[2/7] Creating UncertaintyPredictor model...")

        model = UncertaintyPredictor(
            input_size=input_dim,
            output_size=output_dim,
            hidden_sizes=model_config['hidden_sizes'],
            dropout_rate=model_config['dropout_rate'],
            use_batchnorm=model_config['use_batchnorm'],
            min_variance=model_config['min_variance']
        )

        total_params = sum(p.numel() for p in model.parameters())
        if verbose:
            print(f"  Total parameters: {total_params:,}")

    # Create loss function
    criterion = GaussianNLLLoss(alpha=training_config['variance_penalty_alpha'])

    # Create trainer
    if verbose:
        print(f"\n[3/7] Creating trainer...")

    if use_ensemble:
        ensemble_base_seed = model_config.get('ensemble_base_seed', 42)
        trainer = EnsembleTrainer(
            ensemble_model=model,
            criterion=criterion,
            device=device,
            learning_rate=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            base_seed=ensemble_base_seed
        )
    elif use_swag:
        trainer = SWAGTrainer(
            swag_model=model,
            criterion=criterion,
            device=device,
            learning_rate=training_config['learning_rate'],
            swa_learning_rate=model_config.get('swag_learning_rate', 0.01),
            weight_decay=training_config['weight_decay'],
            swa_start_epoch=model_config.get('swag_start_epoch', 0.5),
            swa_freq=model_config.get('swag_collection_freq', 1),
            min_samples=model_config.get('swag_min_samples', 20)
        )
    else:
        trainer = UncertaintyTrainer(
            model=model,
            criterion=criterion,
            device=device,
            learning_rate=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )

    # Train
    if verbose:
        print(f"\n[4/7] Training model...")

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=training_config['epochs'],
        patience=training_config['patience'],
        save_dir=str(checkpoint_dir)
    )

    # Evaluate on test set
    if verbose:
        print(f"\n[5/7] Evaluating on test set...")

    model.eval()
    all_means = []
    all_vars = []
    all_targets = []
    all_aleatorics = []
    all_epistemics = []

    swag_n_samples = model_config.get('swag_n_samples', 30)

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            if use_ensemble:
                mean, variance, aleatoric, epistemic = model.predict_with_decomposition(batch_X)
                all_aleatorics.append(aleatoric.cpu().numpy())
                all_epistemics.append(epistemic.cpu().numpy())
            elif use_swag:
                mean, variance, aleatoric, epistemic = model.predict_with_decomposition(
                    batch_X, n_samples=swag_n_samples
                )
                all_aleatorics.append(aleatoric.cpu().numpy())
                all_epistemics.append(epistemic.cpu().numpy())
            else:
                mean, variance = model(batch_X)

            all_means.append(mean.cpu().numpy())
            all_vars.append(variance.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    means_scaled = np.vstack(all_means)
    vars_scaled = np.vstack(all_vars)
    targets_scaled = np.vstack(all_targets)

    if use_ensemble or use_swag:
        aleatorics_scaled = np.vstack(all_aleatorics)
        epistemics_scaled = np.vstack(all_epistemics)

    # Unscale predictions
    means = preprocessor.output_scaler.inverse_transform(means_scaled)
    targets = preprocessor.output_scaler.inverse_transform(targets_scaled)

    # Unscale variances (variance scales with scale^2)
    output_scale = preprocessor.output_scaler.scale_
    variances = vars_scaled * (output_scale ** 2)

    # Unscale ensemble uncertainty decomposition if available
    if use_ensemble or use_swag:
        aleatorics = aleatorics_scaled * (output_scale ** 2)
        epistemics = epistemics_scaled * (output_scale ** 2)

    # Compute metrics (note: calculate_metrics expects y_true, y_pred_mean, y_pred_variance)
    metrics_full = uq_metrics.calculate_metrics(targets, means, variances)
    test_metrics = metrics_full['Overall']  # Get overall metrics

    # Compute coverage
    coverage_results = uq_metrics.evaluate_prediction_intervals(
        targets, means, variances, confidence=0.95
    )

    # Add ensemble/SWAG uncertainty decomposition to coverage_results for report
    if use_ensemble or use_swag:
        mean_aleatoric = np.mean(aleatorics)
        mean_epistemic = np.mean(epistemics)
        mean_total = np.mean(variances)
        epistemic_ratio = mean_epistemic / mean_total * 100 if mean_total > 0 else 0

        coverage_results['mean_aleatoric'] = float(mean_aleatoric)
        coverage_results['mean_epistemic'] = float(mean_epistemic)
        coverage_results['epistemic_ratio'] = float(epistemic_ratio)

    if verbose:
        print("\n  Test Metrics:")
        print(f"    MSE:               {test_metrics['MSE']:.6f}")
        print(f"    RMSE:              {test_metrics['RMSE']:.6f}")
        print(f"    MAE:               {test_metrics['MAE']:.6f}")
        print(f"    R²:                {test_metrics['R2']:.6f}")
        print(f"    Mean Variance:     {test_metrics['Mean_Variance']:.6f}")
        print(f"    Calibration Ratio: {test_metrics['Calibration_Ratio']:.6f}")
        print(f"    NLL:               {test_metrics['NLL']:.6f}")
        print(f"\n  Coverage:")
        print(f"    Expected: {coverage_results['expected_coverage']:.1f}%")
        print(f"    Actual:   {coverage_results['actual_coverage']:.1f}%")
        print(f"    Well calibrated: {coverage_results['well_calibrated']}")

        # Print ensemble/SWAG-specific uncertainty decomposition
        if use_ensemble or use_swag:
            method_name = "Ensemble" if use_ensemble else "SWAG"
            print(f"\n  Uncertainty Decomposition ({method_name}):")
            print(f"    Mean Aleatoric:  {mean_aleatoric:.6f}")
            print(f"    Mean Epistemic:  {mean_epistemic:.6f}")
            print(f"    Epistemic Ratio: {epistemic_ratio:.1f}%")

    # Also get predictions on training set for visualization
    y_train_pred_aleatoric = None
    y_train_pred_epistemic = None
    if use_ensemble or use_swag:
        if use_swag:
            y_train_pred_mean, y_train_pred_variance, y_train_pred_aleatoric, y_train_pred_epistemic = trainer.predict(
                X_train_scaled, return_uncertainty=True, n_samples=swag_n_samples
            )
        else:
            y_train_pred_mean, y_train_pred_variance, y_train_pred_aleatoric, y_train_pred_epistemic = trainer.predict(X_train_scaled, return_uncertainty=True)
    else:
        y_train_pred_mean, y_train_pred_variance = trainer.predict(X_train_scaled, return_uncertainty=True)

    y_train_pred_mean_orig = preprocessor.inverse_transform_output(y_train_pred_mean)
    y_train_orig = y_train

    if hasattr(preprocessor, 'output_scaler') and preprocessor.output_scaler is not None:
        if hasattr(preprocessor.output_scaler, 'scale_'):
            scale_factors = preprocessor.output_scaler.scale_
            y_train_pred_variance_orig = y_train_pred_variance * (scale_factors ** 2)
            # Also scale aleatoric and epistemic variances
            if y_train_pred_aleatoric is not None:
                y_train_pred_aleatoric_orig = y_train_pred_aleatoric * (scale_factors ** 2)
                y_train_pred_epistemic_orig = y_train_pred_epistemic * (scale_factors ** 2)
        else:
            y_train_pred_variance_orig = y_train_pred_variance
            y_train_pred_aleatoric_orig = y_train_pred_aleatoric
            y_train_pred_epistemic_orig = y_train_pred_epistemic
    else:
        y_train_pred_variance_orig = y_train_pred_variance
        y_train_pred_aleatoric_orig = y_train_pred_aleatoric
        y_train_pred_epistemic_orig = y_train_pred_epistemic

    # Generate visualizations
    if verbose:
        print(f"\n[6/7] Generating visualizations...")

    # Training history plot (with SWA start marker if using SWAG)
    uq_viz.plot_training_history(
        train_losses=history['train_losses'],
        val_losses=history['val_losses'],
        train_mse=history.get('train_mse'),
        val_mse=history.get('val_mse'),
        save_path=str(checkpoint_dir / 'training_history.png'),
        swa_start_epoch=history.get('swa_start_epoch') if use_swag else None
    )

    # Predictions plot - test set (with aleatoric/epistemic decomposition for ensemble)
    uq_viz.plot_predictions_with_uncertainty(
        y_true=targets,
        y_pred_mean=means,
        y_pred_variance=variances,
        output_names=output_labels,
        save_path=str(checkpoint_dir / 'predictions_with_uncertainty.png'),
        y_pred_aleatoric=aleatorics if (use_ensemble or use_swag) else None,
        y_pred_epistemic=epistemics if (use_ensemble or use_swag) else None
    )

    # Predictions plot - training set (with aleatoric/epistemic decomposition for ensemble/SWAG)
    uq_viz.plot_predictions_with_uncertainty(
        y_true=y_train_orig,
        y_pred_mean=y_train_pred_mean_orig,
        y_pred_variance=y_train_pred_variance_orig,
        output_names=output_labels,
        save_path=str(checkpoint_dir / 'training_predictions_with_uncertainty.png'),
        y_pred_aleatoric=y_train_pred_aleatoric_orig if (use_ensemble or use_swag) else None,
        y_pred_epistemic=y_train_pred_epistemic_orig if (use_ensemble or use_swag) else None
    )

    # Combined predictions plot (validation + training side by side in one PNG)
    _decomp = use_ensemble or use_swag
    uq_viz.plot_combined_predictions_with_uncertainty(
        y_true_val=targets,
        y_pred_mean_val=means,
        y_pred_variance_val=variances,
        y_true_train=y_train_orig,
        y_pred_mean_train=y_train_pred_mean_orig,
        y_pred_variance_train=y_train_pred_variance_orig,
        output_names=output_labels,
        save_path=str(checkpoint_dir / 'predictions_combined.png'),
        y_pred_aleatoric_val=aleatorics if _decomp else None,
        y_pred_epistemic_val=epistemics if _decomp else None,
        y_pred_aleatoric_train=y_train_pred_aleatoric_orig if _decomp else None,
        y_pred_epistemic_train=y_train_pred_epistemic_orig if _decomp else None
    )

    # Scatter plot with uncertainty
    uq_viz.plot_scatter_with_uncertainty(
        y_true=targets,
        y_pred_mean=means,
        y_pred_variance=variances,
        output_names=output_labels,
        save_path=str(checkpoint_dir / 'scatter_with_uncertainty.png')
    )

    # Uncertainty distribution plot
    uq_viz.plot_uncertainty_distribution(
        y_pred_variance=variances,
        output_names=output_labels,
        save_path=str(checkpoint_dir / 'uncertainty_distribution.png')
    )

    # Generate PDF report
    if verbose:
        print(f"\n[7/7] Generating PDF report and saving artifacts...")

    # Build config dict for report
    config_dict = {
        'model': {
            'model_type': 'EnsembleUncertaintyPredictor' if use_ensemble else ('SWAGUncertaintyPredictor' if use_swag else 'UncertaintyPredictor'),
            'hidden_sizes': model_config['hidden_sizes'],
            'dropout_rate': model_config['dropout_rate'],
            'use_batchnorm': model_config['use_batchnorm'],
            'min_variance': model_config['min_variance'],
            'uncertainty_method': uncertainty_method,
            'use_ensemble': use_ensemble,
            'n_ensemble_models': model_config.get('n_ensemble_models', 5) if use_ensemble else None,
            'use_swag': use_swag,
            'swag_max_rank': model_config.get('swag_max_rank', 20) if use_swag else None,
        },
        'data': {
            'csv_path': f'SCM:{scm_dataset_type}',
            'input_columns': input_labels,
            'output_columns': output_labels,
            'train_size': 0.7,
            'val_size': 0.15,
            'test_size': 0.15,
            'scaling_method': 'standard',
            'random_state': seed,
        },
        'training': {
            'epochs': training_config['epochs'],
            'batch_size': training_config['batch_size'],
            'learning_rate': training_config['learning_rate'],
            'weight_decay': training_config['weight_decay'],
            'loss_type': training_config['loss_type'],
            'variance_penalty_alpha': training_config['variance_penalty_alpha'],
            'patience': training_config['patience'],
            'device': device,
            'checkpoint_dir': str(checkpoint_dir),
        },
        'uncertainty': {
            'confidence_level': 0.95,
        },
        'misc': {
            'random_seed': seed,
            'verbose': verbose,
        }
    }

    report_path = generate_uncertainty_training_report(
        config=config_dict,
        history=history,
        metrics=metrics_full,  # Pass full metrics dict, not just 'Overall'
        input_dim=input_dim,
        output_dim=output_dim,
        total_params=total_params,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        checkpoint_dir=checkpoint_dir,
        timestamp=datetime.now(),
        coverage_results=coverage_results,
        st_params=process_config.get('st_params'),
    )

    # Save model weights
    model_path = checkpoint_dir / 'uncertainty_predictor.pth'
    torch.save(model.state_dict(), model_path)

    # Save preprocessor
    scaler_path = checkpoint_dir / 'scalers.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(preprocessor, f)

    # Helper function to convert numpy types to native Python types for JSON
    def convert_to_native(obj):
        """Recursively convert numpy types to native Python types"""
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        else:
            return obj

    # Save training info
    training_info = {
        'process_name': process_name,
        'scm_dataset_type': scm_dataset_type,
        'input_dim': int(input_dim),
        'output_dim': int(output_dim),
        'input_labels': input_labels,
        'output_labels': output_labels,
        'model_config': model_config,
        'training_config': training_config,
        'total_params': int(total_params),
        'metrics': convert_to_native(test_metrics),
        'coverage': convert_to_native(coverage_results),
        'history': convert_to_native(history),
        'timestamp': datetime.now().isoformat(),
    }

    info_path = checkpoint_dir / 'training_info.json'
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)

    if verbose:
        print(f"\n  Saved:")
        print(f"    Model:       {model_path}")
        print(f"    Scaler:      {scaler_path}")
        print(f"    Info:        {info_path}")
        print(f"    Report:      {report_path}")

    return {
        'model_path': str(model_path),
        'scaler_path': str(scaler_path),
        'info_path': str(info_path),
        'metrics': test_metrics,
        'history': history,
        'report_path': str(report_path),
    }
