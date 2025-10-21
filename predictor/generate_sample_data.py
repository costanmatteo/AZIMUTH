"""
Script per generare dati sintetici di esempio

Questo script crea un file CSV con dati casuali per testare il sistema.
In produzione, sostituisci con i tuoi dati reali del macchinario.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_sample_data(n_samples=1000, save_path='data/raw/machinery_data.csv'):
    """
    Genera dati sintetici del macchinario.

    Il modello simula:
    - Input: parametri operativi (temperatura, pressione, velocità, etc.)
    - Output: valori misurati che dipendono dagli input + rumore

    Args:
        n_samples (int): Numero di campioni da generare
        save_path (str): Path dove salvare il CSV
    """
    np.random.seed(42)

    print(f"Generazione di {n_samples} campioni...")

    # Parametri di input (es. impostazioni della macchina)
    temperatura_input = np.random.uniform(20, 100, n_samples)
    pressione_input = np.random.uniform(1, 10, n_samples)
    velocita_motore = np.random.uniform(1000, 3000, n_samples)
    portata = np.random.uniform(10, 100, n_samples)
    umidita = np.random.uniform(30, 80, n_samples)

    # Output (dipendono dagli input con relazioni simulate)
    # In produzione, questi saranno i valori REALI misurati dal macchinario

    # Pressione output: dipende da pressione input e velocità
    pressione_output = (
        1.5 * pressione_input +
        0.01 * velocita_motore +
        np.random.normal(0, 0.5, n_samples)
    )

    # Temperatura output: dipende da temperatura input e velocità motore
    temperatura_output = (
        1.2 * temperatura_input +
        0.02 * velocita_motore +
        np.random.normal(0, 2, n_samples)
    )

    # Potenza: dipende da velocità e portata
    potenza = (
        0.5 * velocita_motore +
        2 * portata +
        np.random.normal(0, 50, n_samples)
    )

    # Efficienza: dipende da temperatura e umidità
    efficienza = (
        100 - 0.3 * temperatura_input -
        0.2 * umidita +
        np.random.normal(0, 3, n_samples)
    )

    # Crea DataFrame
    df = pd.DataFrame({
        # Input
        'temperatura_input': temperatura_input,
        'pressione_input': pressione_input,
        'velocita_motore': velocita_motore,
        'portata': portata,
        'umidita': umidita,
        # Output
        'pressione_output': pressione_output,
        'temperatura_output': temperatura_output,
        'potenza': potenza,
        'efficienza': efficienza,
    })

    # Crea directory se non esiste
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Salva CSV
    df.to_csv(save_path, index=False)

    print(f"\nDati salvati in: {save_path}")
    print(f"\nColonne generate:")
    print(f"\nINPUT (parametri operativi):")
    print("  - temperatura_input")
    print("  - pressione_input")
    print("  - velocita_motore")
    print("  - portata")
    print("  - umidita")
    print(f"\nOUTPUT (valori misurati):")
    print("  - pressione_output")
    print("  - temperatura_output")
    print("  - potenza")
    print("  - efficienza")

    print(f"\nStatistiche:")
    print(df.describe())

    return df


if __name__ == "__main__":
    df = generate_sample_data(n_samples=1000)
    print("\n✓ Dati generati con successo!")
    print("\nPuoi ora:")
    print("1. Modificare configs/example_config.py con i nomi delle colonne")
    print("2. Eseguire: python train.py")
