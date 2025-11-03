#!/usr/bin/env python3
"""
SCM Dataset Generation Script

This script generates a complete synthetic Supply Chain Management (SCM)
dataset that can be used to train the uncertainty predictor model.

Usage:
    python generate_dataset.py [--samples N] [--output PATH]
"""

import argparse
from pathlib import Path
from generator import SCMDataGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic SCM dataset for uncertainty prediction"
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=2000,
        help='Number of samples to generate (default: 2000)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/scm_dataset.csv',
        help='Output path for the CSV file (default: data/scm_dataset.csv)'
    )
    parser.add_argument(
        '--noise-level',
        type=float,
        default=0.15,
        help='Base noise level (default: 0.15)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--no-heteroscedastic',
        action='store_true',
        help='Disable heteroscedastic noise (use constant noise)'
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("SCM DATASET GENERATION")
    print("="*70)
    print(f"Number of samples: {args.samples}")
    print(f"Output path: {args.output}")
    print(f"Noise level: {args.noise_level}")
    print(f"Random seed: {args.seed}")
    print(f"Heteroscedastic noise: {not args.no_heteroscedastic}")
    print("="*70)

    # Create generator
    generator = SCMDataGenerator(
        random_seed=args.seed,
        noise_level=args.noise_level,
        heteroscedastic=not args.no_heteroscedastic
    )

    # Generate and save dataset
    print("\nGenerating dataset...")
    df = generator.generate_and_save(
        output_path=str(output_path),
        n_samples=args.samples,
        x_range=(0, 10),
        y_range=(0, 10),
        z_range=(0, 10)
    )

    print("\n" + "="*70)
    print("DATASET GENERATION COMPLETED!")
    print("="*70)
    print(f"\nDataset saved to: {output_path}")
    print(f"You can now use this dataset with uncertainty_predictor")
    print(f"\nTo train the model with this dataset:")
    print(f"  1. Update uncertainty_predictor/configs/example_config.py")
    print(f"  2. Set csv_path to: '../scm/{args.output}'")
    print(f"  3. Run: cd uncertainty_predictor && python train.py")
    print("="*70)


if __name__ == "__main__":
    main()
