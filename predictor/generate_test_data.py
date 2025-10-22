"""
Script per generare dati di test con una funzione matematica semplice

Funzione: f(x, y, z) = 2*x + 3*y - z + rumore

Questo permette di testare il modello su un problema semplice dove
conosciamo la soluzione esatta!
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_simple_test_data(n_samples=500, noise_level=0.1, save_path='data/raw/test_data.csv'):
    """
    Genera dati da una funzione matematica semplice.

    Funzione: output = 2*x + 3*y - z + rumore

    Args:
        n_samples (int): Numero di campioni da generare
        noise_level (float): Livello di rumore (0 = nessun rumore)
        save_path (str): Dove salvare il CSV
    """
    print(f"Generazione di {n_samples} campioni di test...")
    print("="*70)

    # Imposta seed per riproducibilità
    np.random.seed(42)

    # Genera input casuali tra -10 e 10
    x = np.random.uniform(-10, 10, n_samples)
    y = np.random.uniform(-10, 10, n_samples)
    z = np.random.uniform(-10, 10, n_samples)

    # Calcola output con la funzione: f(x,y,z) = 2*x + 3*y - z
    # Aggiungi un po' di rumore gaussiano
    noise = np.random.normal(0, noise_level, n_samples)
    output = 2*x + 3*y - z + noise

    # Crea DataFrame
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'z': z,
        'output': output
    })

    # Crea directory se non esiste
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Salva CSV
    df.to_csv(save_path, index=False)

    print(f"\n✓ Dati salvati in: {save_path}")
    print(f"\nFunzione usata: output = 2*x + 3*y - z + rumore")
    print(f"Rumore: ±{noise_level}")
    print(f"\nStatistiche dei dati:")
    print(df.describe())

    print(f"\nPrime 10 righe:")
    print(df.head(10))

    print(f"\n{'='*70}")
    print("COSA DOVREBBE SUCCEDERE:")
    print("="*70)
    print("Se il modello funziona bene, dopo il training dovrebbe:")
    print("  1. Avere R² vicino a 1.0 (idealmente > 0.99)")
    print("  2. Avere RMSE basso (< 1.0)")
    print("  3. Imparare i coefficienti approssimativi: 2, 3, -1")
    print("="*70)

    return df


if __name__ == "__main__":
    # Genera i dati
    df = generate_simple_test_data(
        n_samples=500,      # 500 campioni
        noise_level=0.1,    # Poco rumore
        save_path='data/raw/test_data.csv'
    )

    print("\nProssimo step:")
    print("1. python train.py  ← Esegui il training")
    print("2. Controlla i risultati in checkpoints/")
    print("3. Il modello dovrebbe avere R² > 0.99!")
