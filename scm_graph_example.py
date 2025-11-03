"""
Simple example: Generate and view SCM graph interactively.
"""

import sys
sys.path.insert(0, 'uncertainty_predictor/scm_ds')

from datasets import ds_scm_1_to_1_ct, ds_scm_laser

# Method 1: Get the graph object
graph = ds_scm_laser.scm.to_graphviz(rankdir="LR")

# Method 2: View the DOT source code
print("=== DOT Source Code ===")
print(graph.source)

# Method 3: Render to file
graph.render("my_laser_graph", format="pdf", cleanup=True)
print("\n✓ Graph saved as: my_laser_graph.pdf")

# Method 4: Try different formats
graph.render("my_laser_graph", format="png", cleanup=True)
print("✓ Graph saved as: my_laser_graph.png")

# Method 5: Customize appearance
custom_graph = ds_scm_1_to_1_ct.scm.to_graphviz(
    rankdir="TB",  # Top to bottom
    node_attrs={"shape": "box", "style": "filled", "fillcolor": "lightblue"},
    edge_attrs={"color": "darkgreen", "penwidth": "2"}
)
custom_graph.render("custom_styled_graph", format="pdf", cleanup=True)
print("✓ Custom styled graph saved as: custom_styled_graph.pdf")
