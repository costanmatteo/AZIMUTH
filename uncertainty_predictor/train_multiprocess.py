"""
Multi-Process Training Script for Uncertainty Quantification with Conditioning

This script trains a single neural network on multiple manufacturing processes
simultaneously, using process ID, environmental features, and temporal conditioning
to adapt the model to different processes and conditions.
"""

import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models import (
    UncertaintyPredictor,
    GaussianNLLLoss,
    EnergyScoreLoss,
)
from training import UncertaintyTrainer
from utils import (
    calculate_metrics_per_process,
    print_metrics_per_process,
    plot_training_history,
    plot_predictions_per_process,
    plot_scatter_per_process,
    prepare_data_per_process,
    evaluate_prediction_intervals,
)
from data.multi_process_dataset import (
    MultiProcessDataset,
    collate_fn,
    split_multi_process_dataset
)

# Import SCM dataset generator
sys.path.insert(0, str(Path(__file__).parent / 'scm_ds'))
from datasets import sample_multi_process_dataset

# Import configuration
from configs.example_config import CONFIG


def main():
    # Capture training start timestamp
    training_start_time = datetime.now()

    print("="*80)
    print("MULTI-PROCESS UNCERTAINTY QUANTIFICATION WITH CONDITIONING")
    print("="*80)
    print(f"Training start: {training_start_time.strftime('%d/%m/%Y %H:%M:%S')}")
    print("="*80)
    print("\nThis model predicts both μ(x) and σ²(x) for multiple processes")
    print("using conditional normalization to adapt to:")
    print("  • Process ID (laser, plasma, galvanic, microetch)")
    print("  • Environmental conditions (temperature, humidity, load)")
    print("  • Temporal information (timestamps)")
    print("="*80)

    # 1. GENERATE MULTI-PROCESS DATASET
    print("\n[1/8] Generating multi-process SCM dataset...")

    scm_config = CONFIG['data'].get('scm', {})
    n_samples_per_process = scm_config.get('n_samples', 2000)
    seed = scm_config.get('seed', 42)

    # Generate combined dataset with all 4 processes
    combined_df, metadata = sample_multi_process_dataset(
        n_per_process=n_samples_per_process,
        seed=seed,
        missing_rate=0.1  # 10% missing values in environmental features
    )

    print(f"  Total samples: {len(combined_df)}")
    print(f"  Processes: {list(metadata['process_names'].values())}")
    print(f"  Samples per process:")
    for pid, name in metadata['process_names'].items():
        count = np.sum(combined_df['process_id'] == pid)
        print(f"    {name:12s} (ID {pid}): {count:5d} samples")

    # 2. SPLIT DATA
    print("\n[2/8] Splitting data (stratified by process)...")

    train_df, val_df, test_df = split_multi_process_dataset(
        combined_df,
        metadata,
        train_size=CONFIG['data']['train_size'],
        val_size=CONFIG['data']['val_size'],
        test_size=CONFIG['data']['test_size'],
        random_state=seed,
        stratify_by_process=True
    )

    print(f"  Train set: {len(train_df)} samples")
    print(f"  Validation set: {len(val_df)} samples")
    print(f"  Test set: {len(test_df)} samples")

    # 3. CREATE PYTORCH DATASETS
    print("\n[3/8] Creating PyTorch datasets with conditioning features...")

    # Note: We'll handle scaling within the dataset for simplicity
    # In a production setting, you'd fit scalers on train set only
    train_dataset = MultiProcessDataset(train_df, metadata)
    val_dataset = MultiProcessDataset(val_df, metadata)
    test_dataset = MultiProcessDataset(test_df, metadata)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['training']['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    input_dim = train_dataset.get_input_dim()
    output_dim = train_dataset.get_output_dim()

    print(f"  Input dimension: {input_dim}")
    print(f"  Output dimension: {output_dim}")
    print(f"  Process distribution (train): {train_dataset.get_process_distribution()}")

    # 4. CREATE MODEL WITH CONDITIONING
    print("\n[4/8] Creating conditional uncertainty model...")

    conditioning_config = CONFIG.get('conditioning', None)

    if conditioning_config is not None and conditioning_config.get('use_conditional_norm', False):
        print("  Conditioning ENABLED:")
        print(f"    - Process embedding dim: {conditioning_config['d_proc']}")
        print(f"    - Context vector dim: {conditioning_config['d_context']}")
        print(f"    - Environmental features: {conditioning_config['env_continuous']['features']}")
        print(f"    - Categorical features: {list(conditioning_config['env_categorical']['features'].keys())}")
        print(f"    - Time encoding: {conditioning_config['time_encoding']['method']}")
    else:
        print("  Conditioning DISABLED (standard model)")
        conditioning_config = None

    model = UncertaintyPredictor(
        input_size=input_dim,
        hidden_sizes=CONFIG['model']['hidden_sizes'],
        output_size=output_dim,
        dropout_rate=CONFIG['model']['dropout_rate'],
        use_batchnorm=CONFIG['model']['use_batchnorm'],
        min_variance=CONFIG['model']['min_variance'],
        conditioning_config=conditioning_config
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # 5. CREATE LOSS FUNCTION
    print("\n[5/8] Setting up loss function...")
    loss_type = CONFIG['training'].get('loss_type', 'gaussian_nll')

    if loss_type == 'gaussian_nll':
        alpha = CONFIG['training'].get('variance_penalty_alpha', 1.0)
        criterion = GaussianNLLLoss(alpha=alpha, reduction='mean')
        print(f"  Using Gaussian NLL Loss with α={alpha:.3f}")
    elif loss_type == 'energy_score':
        n_samples = CONFIG['training'].get('energy_score_samples', 50)
        beta = CONFIG['training'].get('energy_score_beta', 1.0)
        criterion = EnergyScoreLoss(n_samples=n_samples, beta=beta, reduction='mean')
        print(f"  Using Energy Score Loss with β={beta:.3f}, {n_samples} samples")
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # 6. TRAINING
    print("\n[6/8] Starting training...")

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

    # 7. EVALUATION - COLLECT PREDICTIONS FOR ALL TEST SAMPLES
    print("\n[7/8] Evaluation on test set...")

    model.eval()
    all_y_true = []
    all_y_pred_mean = []
    all_y_pred_variance = []
    all_process_ids = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            process_id = batch['process_id'].to(device)

            # Extract conditioning
            env_cont = batch.get('env_cont')
            if env_cont is not None:
                env_cont = env_cont.to(device)

            env_cont_mask = batch.get('env_cont_mask')
            if env_cont_mask is not None:
                env_cont_mask = env_cont_mask.to(device)

            env_cat = batch.get('env_cat')
            if env_cat is not None:
                env_cat = {k: v.to(device) for k, v in env_cat.items()}

            timestamp = batch.get('timestamp')
            if timestamp is not None:
                timestamp = timestamp.to(device)

            # Predict
            mean, variance = model(
                x,
                process_id=process_id,
                env_cont=env_cont,
                env_cont_mask=env_cont_mask,
                env_cat=env_cat,
                timestamp=timestamp
            )

            all_y_true.append(y.cpu().numpy())
            all_y_pred_mean.append(mean.cpu().numpy())
            all_y_pred_variance.append(variance.cpu().numpy())
            all_process_ids.append(process_id.cpu().numpy())

    # Concatenate all predictions
    y_true = np.concatenate(all_y_true, axis=0)
    y_pred_mean = np.concatenate(all_y_pred_mean, axis=0)
    y_pred_variance = np.concatenate(all_y_pred_variance, axis=0)
    process_ids = np.concatenate(all_process_ids, axis=0)

    print(f"  Collected {len(y_true)} test predictions")

    # 8. CALCULATE METRICS PER PROCESS
    print("\n[8/8] Calculating metrics per process...")

    metrics_per_process = calculate_metrics_per_process(
        y_true,
        y_pred_mean,
        process_ids,
        y_pred_variance=y_pred_variance,
        output_names=['Output'],
        process_names=metadata['process_names']
    )

    print_metrics_per_process(metrics_per_process)

    # 9. VISUALIZATIONS
    print("\nGenerating multi-process visualizations...")
    checkpoint_dir = Path(CONFIG['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Plot training history (same as before)
    plot_training_history(
        history['train_losses'],
        history['val_losses'],
        history['train_mse'],
        history['val_mse'],
        save_path=checkpoint_dir / 'training_history.png'
    )

    # Prepare data per process for plotting
    y_true_dict, y_pred_mean_dict, y_pred_variance_dict = prepare_data_per_process(
        y_true, y_pred_mean, y_pred_variance, process_ids
    )

    # Plot predictions per process
    plot_predictions_per_process(
        y_true_dict,
        y_pred_mean_dict,
        y_pred_variance_dict,
        metadata['process_names'],
        save_path=checkpoint_dir / 'predictions_per_process.png',
        confidence=CONFIG['uncertainty']['confidence_level']
    )

    # Plot scatter per process
    plot_scatter_per_process(
        y_true_dict,
        y_pred_mean_dict,
        y_pred_variance_dict,
        metadata['process_names'],
        save_path=checkpoint_dir / 'scatter_per_process.png'
    )

    print("\n" + "="*80)
    print("MULTI-PROCESS TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nFiles saved in: {checkpoint_dir}/")
    print("  - best_model.pth                : Best model checkpoint")
    print("  - training_history.json         : Training history")
    print("  - training_history.png          : Loss plots")
    print("  - predictions_per_process.png   : Predictions for each process")
    print("  - scatter_per_process.png       : Scatter plots per process")
    print("\n" + "="*80)

    # Print summary table
    print("\nSUMMARY - R² SCORES PER PROCESS:")
    print("-" * 50)
    for pid in sorted(metadata['process_names'].keys()):
        proc_name = metadata['process_names'][pid]
        r2 = metrics_per_process[f'process_{pid}']['Output']['R2']
        mse = metrics_per_process[f'process_{pid}']['Output']['MSE']
        print(f"  Process {pid} ({proc_name:12s}): R² = {r2:6.4f}, MSE = {mse:8.6f}")

    overall_r2 = metrics_per_process['overall']['Output']['R2']
    overall_mse = metrics_per_process['overall']['Output']['MSE']
    print(f"  {'Overall':19s}: R² = {overall_r2:6.4f}, MSE = {overall_mse:8.6f}")
    print("-" * 50)

    return metrics_per_process


if __name__ == "__main__":
    # Setup seed for reproducibility
    torch.manual_seed(CONFIG['misc']['random_seed'])
    np.random.seed(CONFIG['misc']['random_seed'])

    main()
