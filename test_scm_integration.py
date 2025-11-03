#!/usr/bin/env python3
"""
Quick test script to verify SCM integration
"""

import sys
from pathlib import Path

# Add uncertainty_predictor to path
sys.path.insert(0, str(Path(__file__).parent / 'uncertainty_predictor' / 'src'))

print("Testing SCM data generation...")
print("-" * 50)

try:
    from data import generate_scm_data

    print("✓ generate_scm_data imported successfully")

    # Test generation
    print("\nGenerating 100 samples from SCM...")
    X, y, input_columns, output_columns = generate_scm_data(
        n_samples=100,
        seed=42,
        dataset_type='one_to_one_ct'
    )

    print(f"\n✓ Data generated successfully!")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Input columns ({len(input_columns)}): {input_columns}")
    print(f"  Output columns ({len(output_columns)}): {output_columns}")

    print("\n✓ All tests passed!")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
