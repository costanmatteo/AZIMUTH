#!/usr/bin/env python3
"""
Integrated script to generate SCM dataset and train uncertainty predictor

This script automates the workflow:
1. Generate SCM dataset
2. Train uncertainty predictor on the generated data
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a shell command and print output"""
    print(f"\n{'='*70}")
    print(f"Running: {cmd}")
    print(f"{'='*70}\n")

    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        capture_output=False,
        text=True
    )

    if result.returncode != 0:
        print(f"\nError: Command failed with exit code {result.returncode}")
        sys.exit(1)

    return result


def main():
    project_root = Path(__file__).parent
    scm_dir = project_root / 'scm'
    uncertainty_dir = project_root / 'uncertainty_predictor'

    print("="*70)
    print("SCM DATASET GENERATION AND TRAINING PIPELINE")
    print("="*70)

    # Step 1: Generate dataset
    print("\n[Step 1/2] Generating SCM dataset...")
    run_command(
        "python generate_dataset.py --samples 2000",
        cwd=scm_dir
    )

    # Step 2: Train uncertainty predictor
    print("\n[Step 2/2] Training uncertainty predictor...")
    run_command(
        "python train.py",
        cwd=uncertainty_dir
    )

    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nResults:")
    print(f"  - Dataset: {scm_dir / 'data' / 'scm_dataset.csv'}")
    print(f"  - Model checkpoint: {uncertainty_dir / 'checkpoints_uncertainty'}")
    print("="*70)


if __name__ == "__main__":
    main()
