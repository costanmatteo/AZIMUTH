#!/usr/bin/env python
"""
Genera il dataset di traiettorie di training per l'intera catena di processi.

Uso:
    python generate_dataset.py [--n_samples 5000] [--seed 42] [--output_path data/training_trajectories.parquet]

Questo script genera il dataset senza addestrare nulla. È lo step "genera i dati"
disaccoppiato dagli step "usa i dati".
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller_optimization.configs.processes_config import PROCESSES
from scm_ds.generate_trajectories import generate_training_trajectories


def main():
    parser = argparse.ArgumentParser(
        description='Genera dataset di traiettorie di training per la catena di processi'
    )
    parser.add_argument(
        '--n_samples', type=int, default=5000,
        help='Numero di campioni per processo (default: 5000)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed per riproducibilità (default: 42)'
    )
    parser.add_argument(
        '--output_path', type=str, default='data/training_trajectories.parquet',
        help='Path di output per il dataset (default: data/training_trajectories.parquet)'
    )
    parser.add_argument(
        '--format', type=str, choices=['parquet', 'csv'], default=None,
        help='Formato di output (default: inferito dall\'estensione)'
    )
    parser.add_argument(
        '--noise_mode', type=str, choices=['active', 'zero'], default='active',
        help='Modalità rumore: active (realistico) o zero (default: active)'
    )
    args = parser.parse_args()

    output_path = Path(args.output_path)

    # Determina formato dall'estensione se non specificato
    fmt = args.format
    if fmt is None:
        if output_path.suffix == '.csv':
            fmt = 'csv'
        else:
            fmt = 'parquet'

    print("=" * 70)
    print("GENERAZIONE DATASET DI TRAIETTORIE")
    print("=" * 70)
    print(f"  Processi: {[p['name'] for p in PROCESSES]}")
    print(f"  n_samples: {args.n_samples}")
    print(f"  seed: {args.seed}")
    print(f"  noise_mode: {args.noise_mode}")
    print(f"  output: {output_path} ({fmt})")
    print()

    # Genera traiettorie
    df = generate_training_trajectories(
        process_configs=PROCESSES,
        n_samples=args.n_samples,
        seed=args.seed,
        noise_mode=args.noise_mode,
    )

    # Salva
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == 'csv':
        df.to_csv(output_path, index=False)
    else:
        df.to_parquet(output_path, index=False)

    # Statistiche
    print("\n" + "=" * 70)
    print("STATISTICHE DATASET")
    print("=" * 70)
    print(f"  Shape: {df.shape}")
    print(f"  Colonne ({len(df.columns)}):")
    for col in df.columns:
        print(f"    {col}")

    f_col = df['F']
    print(f"\n  F range: [{f_col.min():.6f}, {f_col.max():.6f}]")
    print(f"  F mean:  {f_col.mean():.6f}")
    print(f"  F std:   {f_col.std():.6f}")

    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"\n  WARNING: {nan_count} valori NaN trovati!")
        for col in df.columns:
            col_nans = df[col].isna().sum()
            if col_nans > 0:
                print(f"    {col}: {col_nans} NaN")
    else:
        print(f"\n  NaN: nessuno")

    print(f"\n  Dataset salvato in: {output_path}")


if __name__ == '__main__':
    main()
