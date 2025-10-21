"""
Script per fare predizioni con un modello già trainato

Uso:
    python predict.py --input "1.2,3.4,5.6,7.8"
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Aggiungi src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data import DataPreprocessor
from models import MachineryPredictor
from configs.example_config import CONFIG


def load_trained_model(checkpoint_path, input_size, output_size, hidden_sizes, dropout_rate):
    """Carica un modello trainato da checkpoint"""
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
    Fa una predizione.

    Args:
        model: Modello trainato
        preprocessor: Preprocessor con scaler fittati
        input_data: Array numpy con i parametri di input

    Returns:
        np.ndarray: Predizioni nella scala originale
    """
    # Preprocessa input
    input_scaled = preprocessor.transform(input_data)

    # Predici
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_scaled)
        output_scaled = model(input_tensor)

    # Riporta alla scala originale
    output = preprocessor.inverse_transform_output(output_scaled.numpy())

    return output


def main():
    parser = argparse.ArgumentParser(description='Predici output del macchinario')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Parametri di input separati da virgola (es: "1.2,3.4,5.6")'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pth',
        help='Path al checkpoint del modello'
    )
    parser.add_argument(
        '--scaler',
        type=str,
        default='checkpoints/scalers.pkl',
        help='Path agli scaler'
    )

    args = parser.parse_args()

    # Parse input
    try:
        input_values = [float(x.strip()) for x in args.input.split(',')]
        input_array = np.array([input_values])
    except ValueError:
        print("Errore: Input deve essere una lista di numeri separati da virgola")
        return

    # Verifica dimensione input
    expected_input_size = len(CONFIG['data']['input_columns'])
    if len(input_values) != expected_input_size:
        print(f"Errore: Attesi {expected_input_size} parametri di input, "
              f"ricevuti {len(input_values)}")
        print(f"Parametri attesi: {CONFIG['data']['input_columns']}")
        return

    # Carica scaler
    print("Caricamento scaler...")
    preprocessor = DataPreprocessor()
    try:
        preprocessor.load_scalers(args.scaler)
    except FileNotFoundError:
        print(f"Errore: Scaler non trovato in {args.scaler}")
        print("Esegui prima il training con train.py")
        return

    # Carica modello
    print("Caricamento modello...")
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
        print(f"Errore: Checkpoint non trovato in {args.checkpoint}")
        print("Esegui prima il training con train.py")
        return

    # Predici
    print("\nPredizione in corso...")
    predictions = predict(model, preprocessor, input_array)

    # Stampa risultati
    print("\n" + "="*60)
    print("RISULTATI PREDIZIONE")
    print("="*60)
    print("\nInput:")
    for name, value in zip(CONFIG['data']['input_columns'], input_values):
        print(f"  {name}: {value:.4f}")

    print("\nOutput predetti:")
    for name, value in zip(CONFIG['data']['output_columns'], predictions[0]):
        print(f"  {name}: {value:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
