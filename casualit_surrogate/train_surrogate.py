#!/usr/bin/env python3
"""
Train CasualiT Surrogate Model.

Usage:
    python -m casualit_surrogate.train_surrogate

This script:
1. Generates trajectories with random controllable parameters
2. Computes reliability F labels using reliability_function
3. Trains CasualiT to predict F from trajectories
4. Saves the best model checkpoint
"""

import sys
import argparse

sys.path.insert(0, '/home/user/AZIMUTH')

from casualit_surrogate.configs.surrogate_config import SURROGATE_CONFIG
from casualit_surrogate.src.training.surrogate_trainer import train_casualit_surrogate


def main():
    parser = argparse.ArgumentParser(description='Train CasualiT Surrogate Model')

    parser.add_argument('--epochs', type=int, default=None,
                       help='Override max_epochs from config')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch_size from config')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning_rate from config')
    parser.add_argument('--n-train-scenarios', type=int, default=None,
                       help='Override n_train scenarios')
    parser.add_argument('--n-trajectories', type=int, default=None,
                       help='Override n_trajectories_per_scenario')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')

    args = parser.parse_args()

    # Copy config and apply overrides
    config = dict(SURROGATE_CONFIG)
    config['training'] = dict(config['training'])
    config['scenarios'] = dict(config['scenarios'])
    config['misc'] = dict(config['misc'])

    if args.epochs is not None:
        config['training']['max_epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.n_train_scenarios is not None:
        config['scenarios']['n_train'] = args.n_train_scenarios
    if args.n_trajectories is not None:
        config['scenarios']['n_trajectories_per_scenario'] = args.n_trajectories

    config['misc']['device'] = args.device

    # Run training
    results = train_casualit_surrogate(
        config=config,
        verbose=not args.quiet,
    )

    print("\nTraining complete!")
    print(f"Best validation loss: {results['best_val_loss']:.6f}")
    print(f"Test R²: {results['test_metrics']['r2']:.4f}")
    print(f"Checkpoint saved to: casualit_surrogate/checkpoints/best_model.ckpt")


if __name__ == '__main__':
    main()
