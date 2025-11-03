"""
Quick script to visualize SCM causal structures without generating datasets.
"""

import sys
from pathlib import Path

# Add scm_ds to path
sys.path.insert(0, 'uncertainty_predictor/scm_ds')

from datasets import ds_scm_1_to_1_ct, ds_scm_laser

# Create output directory
output_dir = Path("scm_graphs")
output_dir.mkdir(exist_ok=True)

# 1. Generate parent-child graph
print("Generating parent-child causal graph...")
graph_parent_child = ds_scm_1_to_1_ct.scm.to_graphviz(rankdir="TB")  # Top-to-bottom
graph_parent_child.render(str(output_dir / "parent_child_graph"), format="pdf", cleanup=True)
print(f"  ✓ Saved: {output_dir}/parent_child_graph.pdf")

# 2. Generate laser graph
print("\nGenerating laser causal graph...")
graph_laser = ds_scm_laser.scm.to_graphviz(rankdir="TB")  # Top-to-bottom
graph_laser.render(str(output_dir / "laser_graph"), format="pdf", cleanup=True)
print(f"  ✓ Saved: {output_dir}/laser_graph.pdf")

# 3. Also save as SVG
print("\nGenerating SVG versions...")
graph_parent_child.format = "svg"
graph_parent_child.render(str(output_dir / "parent_child_graph"), cleanup=True)
print(f"  ✓ Saved: {output_dir}/parent_child_graph.svg")

graph_laser.format = "svg"
graph_laser.render(str(output_dir / "laser_graph"), cleanup=True)
print(f"  ✓ Saved: {output_dir}/laser_graph.svg")

print(f"\n✓ All graphs saved to: {output_dir}/")
print("\nGraph options:")
print("  - rankdir='LR': Left-to-right (default)")
print("  - rankdir='TB': Top-to-bottom")
print("  - format='pdf', 'svg', 'png'")
