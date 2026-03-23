"""
Caricamento o generazione al volo delle traiettorie di training.

Fornisce load_or_generate_trajectories() che:
- Se esiste il file al path configurato → carica da disco
- Altrimenti → chiama generate_training_trajectories al volo
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple


def load_or_generate_trajectories(
    process_configs: list,
    dataset_path: Optional[str] = None,
    n_samples: int = 5000,
    seed: int = 42,
    noise_mode: str = 'active',
) -> pd.DataFrame:
    """
    Carica traiettorie pre-generate da disco, o le genera al volo come fallback.

    Args:
        process_configs: Lista di configurazioni processo (da PROCESSES).
        dataset_path: Path al file parquet/csv pre-generato. Se None o se il file
            non esiste, genera al volo.
        n_samples: Numero di campioni (usato solo se genera al volo).
        seed: Random seed (usato solo se genera al volo).
        noise_mode: 'active' o 'zero' (usato solo se genera al volo).

    Returns:
        pd.DataFrame con traiettorie complete (colonne per ogni processo + F).
    """
    if dataset_path is not None:
        path = Path(dataset_path)
        if path.exists():
            print(f"  Loading pre-generated trajectories from: {path}")
            if path.suffix == '.csv':
                df = pd.read_csv(path)
            else:
                df = pd.read_parquet(path)
            print(f"  Loaded: {df.shape[0]} samples, {df.shape[1]} columns")
            return df

    # Fallback: genera al volo
    print(f"  No pre-generated dataset found. Generating {n_samples} trajectories...")
    from scm_ds.generate_trajectories import generate_training_trajectories
    return generate_training_trajectories(
        process_configs=process_configs,
        n_samples=n_samples,
        seed=seed,
        noise_mode=noise_mode,
    )


def extract_process_data(
    df: pd.DataFrame,
    process_config: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estrae dal DataFrame di traiettorie i dati (X, y) relativi a un singolo processo.

    Args:
        df: DataFrame di traiettorie complete (da load_or_generate_trajectories).
        process_config: Configurazione del processo.

    Returns:
        X: np.ndarray (n_samples, input_dim)
        y: np.ndarray (n_samples, output_dim)
    """
    process_name = process_config['name']
    input_labels = process_config['input_labels']
    output_labels = process_config['output_labels']

    # Le colonne nel DataFrame hanno il prefisso {process_name}_
    input_cols = [f"{process_name}_{label}" for label in input_labels]
    output_cols = [f"{process_name}_{label}" for label in output_labels]

    X = df[input_cols].values
    y = df[output_cols].values

    return X, y
