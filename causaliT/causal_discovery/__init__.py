"""
causaliT.causal_discovery
=========================

Causal discovery and validation module for the AZIMUTH pipeline.

Replicates the 5 analysis categories from *causal-chamber-paper*
(juangamella/causal-chamber-paper), adapted for the AZIMUTH process chain
(SCM -> Uncertainty Predictor -> CausaliT/ProT -> Controller).

Submodules
----------
ground_truth           Ground truth DAG extraction from SCMDataset chains.
attention_discovery    Attention-based causal graph estimation from CausaliT.
metrics                SHD, precision, recall, F1 for graph comparison.
discovery_validation   End-to-end discovery validation pipeline.
interventional_analysis  do()-based intervention analysis and F validation.
ood_analysis           Out-of-distribution robustness analysis.
symbolic_analysis      Symbolic regression for equation recovery.
report_generator       PDF report generation (ReportLab).
"""

from .ground_truth import (
    build_ground_truth_dag,
    build_ground_truth_from_datasets,
    get_observable_variables,
    dag_to_edge_list,
    PROCESS_ORDER,
)

from .attention_discovery import (
    extract_attention_weights,
    aggregate_attention,
    attention_to_adjacency,
    discover_dag_from_attention,
    discover_full_dag,
    load_vars_map,
)

from .metrics import (
    structural_hamming_distance,
    edge_precision_recall_f1,
    compute_all_metrics,
    compare_graphs,
)

__all__ = [
    # ground_truth
    "build_ground_truth_dag",
    "build_ground_truth_from_datasets",
    "get_observable_variables",
    "dag_to_edge_list",
    "PROCESS_ORDER",
    # attention_discovery
    "extract_attention_weights",
    "aggregate_attention",
    "attention_to_adjacency",
    "discover_dag_from_attention",
    "discover_full_dag",
    "load_vars_map",
    # metrics
    "structural_hamming_distance",
    "edge_precision_recall_f1",
    "compute_all_metrics",
    "compare_graphs",
]
