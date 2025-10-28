"""
Prediction script for Uncertainty Quantification model

Load a trained model and make predictions with uncertainty estimates.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models import UncertaintyPredictor
from data import DataPreprocessor


def load_model_and_preprocessor(checkpoint_dir):
    """
    Load trained model and preprocessor.

    Args:
        checkpoint_dir (str): Directory containing model checkpoint and scalers

    Returns:
        tuple: (model, preprocessor, device)
    """
    checkpoint_path = Path(checkpoint_dir) / 'best_model.pth'
    scaler_path = Path(checkpoint_dir) / 'scalers.pkl'

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    # Load checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load preprocessor
    preprocessor = DataPreprocessor()
    preprocessor.load_scalers(scaler_path)

    print(f"✓ Model and preprocessor loaded from {checkpoint_dir}")
    print(f"✓ Using device: {device}")

    return checkpoint, preprocessor, device


def predict_with_uncertainty(X, model, preprocessor, device, confidence=0.95):
    """
    Make predictions with uncertainty estimates.

    Args:
        X (np.ndarray): Input features
        model (nn.Module): Trained uncertainty model
        preprocessor (DataPreprocessor): Fitted preprocessor
        device (torch.device): Device to use
        confidence (float): Confidence level for prediction intervals

    Returns:
        dict: Dictionary with predictions and uncertainty estimates
    """
    # Preprocess input
    X_scaled, _ = preprocessor.transform(X, None)
    X_tensor = torch.FloatTensor(X_scaled).to(device)

    # Make predictions
    model.eval()
    with torch.no_grad():
        mean, variance = model(X_tensor)

    mean_np = mean.cpu().numpy()
    variance_np = variance.cpu().numpy()

    # Inverse transform to original scale
    mean_orig = preprocessor.inverse_transform_output(mean_np)

    # Scale variance appropriately
    if hasattr(preprocessor, 'y_scaler') and preprocessor.y_scaler is not None:
        if hasattr(preprocessor.y_scaler, 'scale_'):
            scale_factors = preprocessor.y_scaler.scale_
            variance_orig = variance_np * (scale_factors ** 2)
        else:
            variance_orig = variance_np
    else:
        variance_orig = variance_np

    # Compute standard deviation
    std_orig = np.sqrt(variance_orig)

    # Compute prediction intervals
    from scipy import stats
    z_score = stats.norm.ppf((1 + confidence) / 2)
    lower_bound = mean_orig - z_score * std_orig
    upper_bound = mean_orig + z_score * std_orig

    return {
        'mean': mean_orig,
        'variance': variance_orig,
        'std': std_orig,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'confidence': confidence
    }


def print_predictions(predictions, output_names=None):
    """
    Print predictions in a formatted way.

    Args:
        predictions (dict): Dictionary returned by predict_with_uncertainty
        output_names (list): Names of output variables
    """
    n_samples = predictions['mean'].shape[0]
    n_outputs = predictions['mean'].shape[1]

    if output_names is None:
        output_names = [f"Output_{i+1}" for i in range(n_outputs)]

    confidence = predictions['confidence']

    print("\n" + "="*70)
    print("PREDICTIONS WITH UNCERTAINTY")
    print("="*70)

    for sample_idx in range(n_samples):
        print(f"\nSample {sample_idx + 1}:")
        print("-" * 50)

        for out_idx, name in enumerate(output_names):
            mean = predictions['mean'][sample_idx, out_idx]
            std = predictions['std'][sample_idx, out_idx]
            lower = predictions['lower_bound'][sample_idx, out_idx]
            upper = predictions['upper_bound'][sample_idx, out_idx]

            print(f"  {name}:")
            print(f"    Mean (μ):        {mean:12.6f}")
            print(f"    Std Dev (σ):     {std:12.6f}")
            print(f"    {int(confidence*100)}% Interval:  [{lower:12.6f}, {upper:12.6f}]")

    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Make predictions with uncertainty estimates')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_uncertainty',
                       help='Directory containing model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input CSV file or comma-separated values')
    parser.add_argument('--confidence', type=float, default=0.95,
                       help='Confidence level for prediction intervals (default: 0.95)')
    parser.add_argument('--output_names', type=str, nargs='+', default=None,
                       help='Names of output variables')

    args = parser.parse_args()

    # Load model and preprocessor
    checkpoint, preprocessor, device = load_model_and_preprocessor(args.checkpoint_dir)

    # Reconstruct model (need to know architecture)
    # For simplicity, we'll extract it from the checkpoint
    # In practice, you might want to save this in the config
    state_dict = checkpoint['model_state_dict']

    # Infer model architecture from state dict
    # This is a simplified approach - in production, save the config with the model
    print("\nNote: Model architecture must match the saved checkpoint.")
    print("Attempting to reconstruct model from checkpoint...")

    # For now, create a medium model as example
    # You should save/load the model config properly
    input_size = list(state_dict.values())[0].shape[1]  # First layer input size
    output_size = list(state_dict.values())[-2].shape[0]  # Mean head output size

    from models import create_medium_uncertainty_model
    model = create_medium_uncertainty_model(input_size, output_size)
    model.load_state_dict(state_dict)
    model = model.to(device)

    print(f"✓ Model reconstructed with input_size={input_size}, output_size={output_size}")

    # Load input data
    if args.input.endswith('.csv'):
        # Load from CSV
        import pandas as pd
        df = pd.read_csv(args.input)
        X = df.values
        print(f"\n✓ Loaded {len(X)} samples from {args.input}")
    else:
        # Parse comma-separated values
        values = [float(v) for v in args.input.split(',')]
        X = np.array(values).reshape(1, -1)
        print(f"\n✓ Parsed input: {values}")

    # Make predictions
    predictions = predict_with_uncertainty(
        X, model, preprocessor, device,
        confidence=args.confidence
    )

    # Print results
    print_predictions(predictions, args.output_names)


if __name__ == "__main__":
    main()
