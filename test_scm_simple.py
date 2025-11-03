#!/usr/bin/env python3
"""
Simple test without torch dependency
"""

import sys
from pathlib import Path

# Add scm_ds to path
scm_path = Path('/home/user/AZIMUTH/uncertainty_predictor/scm_ds')
sys.path.insert(0, str(scm_path))

print("Testing SCM module directly...")
print("-" * 50)

try:
    from datasets import ds_scm_1_to_1_ct

    print("✓ SCM dataset imported successfully")

    # Test generation
    print("\nGenerating 100 samples from SCM...")
    df = ds_scm_1_to_1_ct.sample(n=100, seed=42)

    print(f"\n✓ Data generated successfully!")
    print(f"  DataFrame shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Input labels: {ds_scm_1_to_1_ct.input_labels}")
    print(f"  Target labels: {ds_scm_1_to_1_ct.target_labels}")

    # Extract X and y
    X = df[ds_scm_1_to_1_ct.input_labels].values
    y = df[ds_scm_1_to_1_ct.target_labels].values

    print(f"\n✓ Data extraction successful!")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")

    print("\n✓ All tests passed!")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
