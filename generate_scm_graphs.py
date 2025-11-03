"""
Example script to generate SCM graphs and datasets.
This creates visualizations of the causal structure.
"""

import sys
from pathlib import Path

# Add scm_ds to path
sys.path.insert(0, 'uncertainty_predictor/scm_ds')

from datasets import ds_scm_1_to_1_ct, ds_scm_laser

# Generate parent-child dataset with graph
print("Generating parent-child dataset with graph...")
ds_scm_1_to_1_ct.generate_ds(
    mode="flat",
    n=5000,
    save_dir="output/parent_child_dataset",
    seed=42
)
print("  ✓ Saved to output/parent_child_dataset/")
print("  ✓ Graph saved as: output/parent_child_dataset/graph.pdf")

# Generate laser dataset with graph
print("\nGenerating laser dataset with graph...")
ds_scm_laser.generate_ds(
    mode="flat",
    n=5000,
    save_dir="output/laser_dataset",
    seed=42
)
print("  ✓ Saved to output/laser_dataset/")
print("  ✓ Graph saved as: output/laser_dataset/graph.pdf")

print("\n✓ Done! Check the output directories for graph.pdf files.")
