"""
Script to make predictions with an already trained model

Usage:
    python predict.py --input "1.2,3.4,5.6,7.8"
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data import DataPreprocessor
from models import MachineryPredictor
from configs.example_config import CONFIG


def load_trained_model(checkpoint_path, input_size, output_size, hidden_sizes, dropout_rate):
    """Load a trained model from checkpoint"""
    model = MachineryPredictor(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        dropout_rate=dropout_rate
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def predict(model, preprocessor, input_data):
    """
    Make a prediction.

    Args:
        model: Trained model
        preprocessor: Preprocessor with fitted scalers
        input_data: Numpy array with input parameters

    Returns:
        np.ndarray: Predictions in original scale
    """
    # Preprocess input
    input_scaled = preprocessor.transform(input_data)

    # Predict
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_scaled)
        output_scaled = model(input_tensor)

    # Return to original scale
    output = preprocessor.inverse_transform_output(output_scaled.numpy())

    return output


def main():
    parser = argparse.ArgumentParser(description='Predict machinery output')
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
    print("Loading model...")
    input_size = len(CONFIG['data']['input_columns'])
    output_size = len(CONFIG['data']['output_columns'])

    try:
        model = load_trained_model(
            args.checkpoint,
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=CONFIG['model']['hidden_sizes'],
            dropout_rate=CONFIG['model']['dropout_rate']
        )
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Run training first with train.py")
        return

    # Predict
    print("\nPredicting...")
    predictions = predict(model, preprocessor, input_array)

    # Print results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print("\nInput:")
    for name, value in zip(CONFIG['data']['input_columns'], input_values):
        print(f"  {name}: {value:.4f}")

    print("\nPredicted output:")
    for name, value in zip(CONFIG['data']['output_columns'], predictions[0]):
        print(f"  {name}: {value:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
