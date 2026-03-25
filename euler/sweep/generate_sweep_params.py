#!/usr/bin/env python3
"""
Generate sweep_params.txt with seed combinations.

Usage:
    python generate_sweep_params.py [--n_seeds N] [--output FILE]

Generates a grid of seed_target x seed_baseline combinations.
"""

import argparse
from pathlib import Path


def generate_seed_sweep(n_seeds: int = 10, output_file: Path = None):
    """
    Generate sweep_params.txt with seed combinations.

    Args:
        n_seeds: Number of seed values to test (creates n_seeds x n_seeds grid)
        output_file: Output file path
    """
    if output_file is None:
        output_file = Path(__file__).parent / "sweep_params.txt"

    # Generate seed values (evenly spaced from 1 to ~100)
    step = max(1, 100 // n_seeds)
    seed_values = list(range(1, 100, step))[:n_seeds]

    # Ensure we have exactly n_seeds values
    if len(seed_values) < n_seeds:
        seed_values = list(range(1, n_seeds * step + 1, step))[:n_seeds]

    total_combinations = len(seed_values) ** 2

    print(f"Generating {total_combinations} seed combinations...")
    print(f"  seed_target values: {seed_values}")
    print(f"  seed_baseline values: {seed_values}")

    lines = [
        "# Controller Seed Sweep Configuration",
        "# ====================================",
        "#",
        f"# {total_combinations} combinations of seed_target and seed_baseline",
        f"# Grid: {len(seed_values)}x{len(seed_values)}",
        "#",
        "# Format: run_name seed_target=X seed_baseline=Y",
        "#",
        f"# Update sweep.sh: --array=0-{total_combinations - 1}",
        "# ====================================",
        "",
    ]

    for seed_t in seed_values:
        for seed_b in seed_values:
            run_name = f"seed_t{seed_t:02d}_b{seed_b:02d}"
            line = f"{run_name} seed_target={seed_t} seed_baseline={seed_b}"
            lines.append(line)

    with open(output_file, 'w') as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nWritten to: {output_file}")
    print(f"\nRemember to update sweep.sh:")
    print(f"  #SBATCH --array=0-{total_combinations - 1}")


def main():
    parser = argparse.ArgumentParser(description='Generate seed sweep parameters')
    parser.add_argument('--n_seeds', type=int, default=10,
                        help='Number of seed values (creates n_seeds x n_seeds grid)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path')
    args = parser.parse_args()

    output_file = Path(args.output) if args.output else None
    generate_seed_sweep(args.n_seeds, output_file)


if __name__ == '__main__':
    main()
