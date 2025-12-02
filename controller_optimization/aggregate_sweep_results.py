#!/usr/bin/env python3
"""
Aggregate and compare results from parameter sweep runs.

Usage:
    python aggregate_sweep_results.py [--sweep_dir PATH] [--output results_summary.csv]

This script:
1. Scans all subdirectories in the sweep results folder
2. Loads final_results.json from each run
3. Aggregates key metrics into a summary table
4. Identifies best performing configurations
"""

import argparse
import json
from pathlib import Path
import pandas as pd


def load_run_results(run_dir: Path) -> dict:
    """Load results from a single run directory."""
    results_file = run_dir / "final_results.json"
    if not results_file.exists():
        return None

    try:
        with open(results_file, 'r') as f:
            data = json.load(f)

        # Extract key metrics
        return {
            'run_name': run_dir.name,
            # Config
            'learning_rate': data.get('config', {}).get('training', {}).get('learning_rate'),
            'lambda_bc': data.get('config', {}).get('training', {}).get('lambda_bc'),
            'batch_size': data.get('config', {}).get('training', {}).get('batch_size'),
            'epochs': data.get('config', {}).get('training', {}).get('epochs'),
            'weight_decay': data.get('config', {}).get('training', {}).get('weight_decay'),
            'reliability_loss_scale': data.get('config', {}).get('training', {}).get('reliability_loss_scale'),
            'dropout': data.get('config', {}).get('policy_generator', {}).get('dropout'),
            # Train metrics
            'F_star_train': data.get('train', {}).get('F_star_mean'),
            'F_baseline_train': data.get('train', {}).get('F_baseline_mean'),
            'F_actual_train': data.get('train', {}).get('F_actual_mean'),
            'improvement_train': data.get('train', {}).get('improvement_pct'),
            'target_gap_train': data.get('train', {}).get('target_gap_pct'),
            # Test metrics
            'F_star_test': data.get('test', {}).get('F_star_mean'),
            'F_baseline_test': data.get('test', {}).get('F_baseline_mean'),
            'F_actual_test': data.get('test', {}).get('F_actual_mean'),
            'improvement_test': data.get('test', {}).get('improvement_pct'),
            # Advanced metrics
            'worst_case_gap_train': data.get('advanced_metrics', {}).get('worst_case_gap_train', {}).get('worst_case_gap'),
            'worst_case_gap_test': data.get('advanced_metrics', {}).get('worst_case_gap_test', {}).get('worst_case_gap'),
            'success_rate_train': data.get('advanced_metrics', {}).get('success_rate_train', {}).get('success_rate_pct'),
            'success_rate_test': data.get('advanced_metrics', {}).get('success_rate_test', {}).get('success_rate_pct'),
            # Theoretical
            'L_min': data.get('theoretical_analysis', {}).get('final_L_min'),
            'efficiency': data.get('theoretical_analysis', {}).get('final_efficiency'),
        }
    except Exception as e:
        print(f"Error loading {results_file}: {e}")
        return None


def aggregate_results(sweep_dir: Path) -> pd.DataFrame:
    """Aggregate results from all runs in sweep directory."""
    results = []

    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        run_results = load_run_results(run_dir)
        if run_results is not None:
            results.append(run_results)
            print(f"  Loaded: {run_dir.name}")

    if not results:
        print("No results found!")
        return pd.DataFrame()

    return pd.DataFrame(results)


def print_summary(df: pd.DataFrame):
    """Print summary of sweep results."""
    if df.empty:
        return

    print("\n" + "="*80)
    print("PARAMETER SWEEP SUMMARY")
    print("="*80)

    print(f"\nTotal runs: {len(df)}")

    # Best by improvement (train)
    if 'improvement_train' in df.columns:
        best_train = df.loc[df['improvement_train'].idxmax()]
        print(f"\nBest by train improvement:")
        print(f"  Run: {best_train['run_name']}")
        print(f"  Improvement: {best_train['improvement_train']:.2f}%")
        print(f"  LR: {best_train['learning_rate']}, λ_BC: {best_train['lambda_bc']}")

    # Best by improvement (test)
    if 'improvement_test' in df.columns and df['improvement_test'].notna().any():
        best_test = df.loc[df['improvement_test'].idxmax()]
        print(f"\nBest by test improvement:")
        print(f"  Run: {best_test['run_name']}")
        print(f"  Improvement: {best_test['improvement_test']:.2f}%")
        print(f"  LR: {best_test['learning_rate']}, λ_BC: {best_test['lambda_bc']}")

    # Best by success rate
    if 'success_rate_test' in df.columns and df['success_rate_test'].notna().any():
        best_success = df.loc[df['success_rate_test'].idxmax()]
        print(f"\nBest by test success rate:")
        print(f"  Run: {best_success['run_name']}")
        print(f"  Success rate: {best_success['success_rate_test']:.1f}%")

    # Summary table
    print("\n" + "-"*80)
    print("ALL RUNS (sorted by test improvement):")
    print("-"*80)

    display_cols = ['run_name', 'learning_rate', 'lambda_bc', 'improvement_train', 'improvement_test', 'success_rate_test']
    display_cols = [c for c in display_cols if c in df.columns]

    sorted_df = df.sort_values('improvement_test', ascending=False, na_position='last')
    print(sorted_df[display_cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='Aggregate parameter sweep results')
    parser.add_argument('--sweep_dir', type=str,
                        default='controller_optimization/checkpoints/sweep',
                        help='Directory containing sweep run results')
    parser.add_argument('--output', type=str,
                        default='sweep_results_summary.csv',
                        help='Output CSV file for results')
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"Error: Sweep directory not found: {sweep_dir}")
        return

    print(f"Scanning: {sweep_dir}")
    df = aggregate_results(sweep_dir)

    if df.empty:
        return

    # Save to CSV
    output_path = sweep_dir / args.output
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Print summary
    print_summary(df)


if __name__ == '__main__':
    main()
