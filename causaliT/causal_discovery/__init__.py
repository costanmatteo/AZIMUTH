"""
Causal Discovery & Validation Module for the Azimuth Pipeline.

Inspired by the analysis structure of juangamella/causal-chamber-paper,
this module provides five categories of causal analysis adapted
to the Azimuth SCM -> CausaliT pipeline:

1. Ground Truth & Attention-based Discovery
2. Causal Discovery Validation (metrics, classical baselines)
3. Interventional Data Generation & Validation
4. Out-of-Distribution Analysis
5. Symbolic Regression

All analyses operate on synthetic data from the Azimuth SCM datasets
and use CausaliT attention weights as the primary discovery mechanism.
"""

# Core modules (numpy/pandas only — always importable)
from .ground_truth import extract_ground_truth_dag, get_observable_variables
from .metrics import (
    structural_hamming_distance,
    edge_precision_recall_f1,
    compare_graphs,
)

# Lazy imports for modules that depend on torch or heavy optional libs.
# This allows using ground_truth / metrics without torch installed.


def __getattr__(name):
    _lazy = {
        "AttentionGraphExtractor": ".attention_discovery",
        "DiscoveryValidator": ".discovery_validation",
        "InterventionalAnalyzer": ".interventional_analysis",
        "OODAnalyzer": ".ood_analysis",
        "SymbolicAnalyzer": ".symbolic_analysis",
        "CausalAnalysisReportGenerator": ".report_generator",
    }
    if name in _lazy:
        import importlib

        mod = importlib.import_module(_lazy[name], __package__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "extract_ground_truth_dag",
    "get_observable_variables",
    "AttentionGraphExtractor",
    "structural_hamming_distance",
    "edge_precision_recall_f1",
    "compare_graphs",
    "DiscoveryValidator",
    "InterventionalAnalyzer",
    "OODAnalyzer",
    "SymbolicAnalyzer",
    "CausalAnalysisReportGenerator",
]
