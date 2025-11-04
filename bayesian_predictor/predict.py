"""
Prediction script for Bayesian Neural Network

Makes predictions with uncertainty quantification using a trained Bayesian model.

Usage:
    python predict.py --input "1.2,3.4,5.6,7.8"
    python predict.py --input "1.2,3.4,5.6,7.8" --n-samples 200
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data import DataPreprocessor
from models import BayesianPredictor
from configs.example_config import CONFIG


def load_trained_bayesian_model(checkpoint_path, input_size, output_size, hidden_sizes,
                                 dropout_rate, prior_std):
    """
    Load a trained Bayesian model from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        input_size: Number of input features
        output_size: Number of output features
        hidden_sizes: List of hidden layer sizes
        dropout_rate: Dropout rate
        prior_std: Prior standard deviation

    Returns:
        BayesianPredictor: Loaded model in eval mode
    """
    model = BayesianPredictor(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        prior_std=prior_std,
        dropout_rate=dropout_rate
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def predict_with_uncertainty(model, preprocessor, input_data, n_samples=100):
    """
    Make predictions with uncertainty quantification.

    Args:
        model: Trained Bayesian model
        preprocessor: Preprocessor with fitted scalers
        input_data: Numpy array with input parameters
        n_samples: Number of Monte Carlo samples for uncertainty estimation

    Returns:
        dict: Dictionary with:
            - mean: Mean prediction
            - std: Standard deviation (uncertainty)
            - confidence_intervals: Dict with 68%, 95%, 99% intervals
            - samples: All sampled predictions
    """
    # Preprocess input
    input_scaled = preprocessor.transform(input_data)

    # Predict with uncertainty
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_scaled)
        results = model.predict_with_uncertainty(input_tensor, n_samples=n_samples)

    # Transform back to original scale
    mean_pred = preprocessor.inverse_transform_output(results['mean'].numpy())
    std_pred = preprocessor.inverse_transform_output(results['std'].numpy())

    # Transform confidence intervals
    confidence_intervals = {}
    for level, (lower, upper) in results['confidence_intervals'].items():
        lower_orig = preprocessor.inverse_transform_output(lower.numpy())
        upper_orig = preprocessor.inverse_transform_output(upper.numpy())
        confidence_intervals[level] = (lower_orig, upper_orig)

    return {
        'mean': mean_pred,
        'std': std_pred,
        'confidence_intervals': confidence_intervals,
        'samples': results['samples'].numpy()
    }


def main():
    parser = argparse.ArgumentParser(
        description='Predict machinery output with uncertainty quantification'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input parameters separated by commas (e.g., "1.2,3.4,5.6")'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--scaler',
        type=str,
        default='checkpoints/scalers.pkl',
        help='Path to scalers'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=100,
        help='Number of Monte Carlo samples for uncertainty estimation (default: 100)'
    )

    args = parser.parse_args()

    # Parse input
    try:
        input_values = [float(x.strip()) for x in args.input.split(',')]
        input_array = np.array([input_values])
    except ValueError:
        print("Error: Input must be a list of numbers separated by commas")
        return

    # Verify input dimension
    expected_input_size = len(CONFIG['data']['input_columns'])
    if len(input_values) != expected_input_size:
        print(f"Error: Expected {expected_input_size} input parameters, "
              f"received {len(input_values)}")
        print(f"Expected parameters: {CONFIG['data']['input_columns']}")
        return

    # Load scaler
    print("Loading scaler...")
    preprocessor = DataPreprocessor()
    try:
        preprocessor.load_scalers(args.scaler)
    except FileNotFoundError:
        print(f"Error: Scaler not found at {args.scaler}")
        print("Run training first with train.py")
        return

    # Load model
    print("Loading Bayesian model...")
    input_size = len(CONFIG['data']['input_columns'])
    output_size = len(CONFIG['data']['output_columns'])
    prior_std = CONFIG['model'].get('prior_std', 1.0)

    try:
        model = load_trained_bayesian_model(
            args.checkpoint,
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=CONFIG['model']['hidden_sizes'],
            dropout_rate=CONFIG['model']['dropout_rate'],
            prior_std=prior_std
        )
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Run training first with train.py")
        return

    # Predict with uncertainty
    print(f"\nPredicting with {args.n_samples} Monte Carlo samples...")
    results = predict_with_uncertainty(model, preprocessor, input_array, n_samples=args.n_samples)

    # Print results
    print("\n" + "="*70)
    print("BAYESIAN PREDICTION RESULTS")
    print("="*70)

    print("\nInput Parameters:")
    for name, value in zip(CONFIG['data']['input_columns'], input_values):
        print(f"  {name}: {value:.4f}")

    print("\nPredicted Output (with Uncertainty):")
    print("-" * 70)

    output_names = CONFIG['data']['output_columns']
    for i, name in enumerate(output_names):
        mean = results['mean'][0, i]
        std = results['std'][0, i]
        ci_68 = results['confidence_intervals']['68%']
        ci_95 = results['confidence_intervals']['95%']

        print(f"\n{name}:")
        print(f"  Mean (Best Estimate):  {mean:.4f}")
        print(f"  Std Dev (Uncertainty): {std:.4f}")
        print(f"  68% Confidence:        [{ci_68[0][0, i]:.4f}, {ci_68[1][0, i]:.4f}]")
        print(f"  95% Confidence:        [{ci_95[0][0, i]:.4f}, {ci_95[1][0, i]:.4f}]")

        # Relative uncertainty
        relative_unc = (std / abs(mean) * 100) if mean != 0 else 0
        print(f"  Relative Uncertainty:  {relative_unc:.2f}%")

    print("\n" + "="*70)
    print("Interpretation:")
    print("  - Mean: Most likely prediction")
    print("  - Std Dev: Uncertainty in the prediction")
    print("  - 68% CI: There's a 68% chance the true value is in this range")
    print("  - 95% CI: There's a 95% chance the true value is in this range")
    print("  - Relative Uncertainty: Uncertainty as % of predicted value")
    print("="*70)

    # Summary of uncertainty
    avg_relative_unc = np.mean([
        (results['std'][0, i] / abs(results['mean'][0, i]) * 100)
        if results['mean'][0, i] != 0 else 0
        for i in range(len(output_names))
    ])

    print(f"\nAverage Relative Uncertainty: {avg_relative_unc:.2f}%")

    if avg_relative_unc < 5:
        print("→ Model is very confident in this prediction")
    elif avg_relative_unc < 15:
        print("→ Model has moderate confidence in this prediction")
    else:
        print("→ Model has high uncertainty - prediction should be used with caution")

    print("="*70)


if __name__ == "__main__":
    main()
