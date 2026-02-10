"""
Ground Truth DAG extraction from SCMDataset process chains.

Builds the inter-process ground truth adjacency matrix considering only
observable variables (input_labels + target_labels per process + F).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scm_ds.scm import SCM, SCMDataset, NodeSpec


# ---------------------------------------------------------------------------
# Default process chain definition
# ---------------------------------------------------------------------------

PROCESS_ORDER = ["laser", "plasma", "galvanic", "microetch"]

# Maps process name -> SCMDataset observable labels
_DEFAULT_OBSERVABLE = {
    "laser":     {"inputs": ["PowerTarget", "AmbientTemp"],    "output": "ActualPower"},
    "plasma":    {"inputs": ["RF_Power", "Duration_plasma"],   "output": "RemovalRate"},
    "galvanic":  {"inputs": ["CurrentDensity", "Duration_galvanic"], "output": "Thickness"},
    "microetch": {"inputs": ["Temperature", "Concentration", "Duration_microetch"], "output": "RemovalDepth"},
}

# Inter-process causal links (output of upstream -> downstream).
# Derived from PROCESS_CONFIGS adaptive_coefficients in
# reliability_function/configs/process_targets.py
_DEFAULT_INTERPROCESS_EDGES: List[Tuple[str, str]] = [
    # laser -> plasma
    ("ActualPower", "RemovalRate"),
    # laser, plasma -> galvanic
    ("ActualPower", "Thickness"),
    ("RemovalRate", "Thickness"),
    # laser, plasma, galvanic -> microetch
    ("ActualPower", "RemovalDepth"),
    ("RemovalRate", "RemovalDepth"),
    ("Thickness",   "RemovalDepth"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_observable_variables(
    process_observables: Optional[Dict[str, dict]] = None,
    process_order: Optional[List[str]] = None,
    include_F: bool = True,
) -> List[str]:
    """Return ordered list of observable variable names across the full chain.

    Order: for each process in chain order list inputs then output,
    finally append ``"F"`` if *include_F* is True.
    """
    obs = process_observables or _DEFAULT_OBSERVABLE
    order = process_order or PROCESS_ORDER

    variables: List[str] = []
    for proc in order:
        cfg = obs[proc]
        variables.extend(cfg["inputs"])
        variables.append(cfg["output"])
    if include_F:
        variables.append("F")
    return variables


def build_ground_truth_dag(
    process_observables: Optional[Dict[str, dict]] = None,
    process_order: Optional[List[str]] = None,
    interprocess_edges: Optional[List[Tuple[str, str]]] = None,
    include_F: bool = True,
    as_dataframe: bool = True,
) -> np.ndarray | pd.DataFrame:
    """Build the inter-process ground truth adjacency matrix.

    Convention (same as ``SCM.adjacency``):
        ``A[i, j] = 1``  means  ``variable_j -> variable_i``
        (i.e. *j* is a parent of *i*).

    Parameters
    ----------
    process_observables : dict, optional
        Mapping ``process_name -> {"inputs": [...], "output": "..."}`` for
        each process.  Defaults to the 4-process PCB chain.
    process_order : list[str], optional
        Ordered process names.  Defaults to ``PROCESS_ORDER``.
    interprocess_edges : list[tuple], optional
        ``(parent, child)`` edges between processes.  Defaults to
        ``_DEFAULT_INTERPROCESS_EDGES``.
    include_F : bool
        If True, add ``F`` as a child of every process output.
    as_dataframe : bool
        If True return a labelled ``pd.DataFrame``; else a plain ndarray.

    Returns
    -------
    adjacency : ndarray or DataFrame
        Square matrix of shape ``(n_vars, n_vars)``.
    """
    obs = process_observables or _DEFAULT_OBSERVABLE
    order = process_order or PROCESS_ORDER
    inter_edges = interprocess_edges if interprocess_edges is not None else _DEFAULT_INTERPROCESS_EDGES

    variables = get_observable_variables(obs, order, include_F)
    idx = {v: i for i, v in enumerate(variables)}
    n = len(variables)
    A = np.zeros((n, n), dtype=int)

    # 1) Intra-process: inputs -> output
    for proc in order:
        cfg = obs[proc]
        out = cfg["output"]
        for inp in cfg["inputs"]:
            A[idx[out], idx[inp]] = 1  # inp is parent of out

    # 2) Inter-process links
    for parent, child in inter_edges:
        if parent in idx and child in idx:
            A[idx[child], idx[parent]] = 1

    # 3) All process outputs -> F
    if include_F:
        for proc in order:
            out = obs[proc]["output"]
            A[idx["F"], idx[out]] = 1

    if as_dataframe:
        return pd.DataFrame(A, index=variables, columns=variables)
    return A


def build_ground_truth_from_datasets(
    datasets: Dict[str, SCMDataset],
    process_order: Optional[List[str]] = None,
    interprocess_edges: Optional[List[Tuple[str, str]]] = None,
    include_F: bool = True,
    as_dataframe: bool = True,
) -> np.ndarray | pd.DataFrame:
    """Build ground truth DAG directly from a dict of SCMDataset objects.

    Automatically extracts ``input_labels`` and ``target_labels`` from each
    dataset instead of relying on hardcoded defaults.

    Parameters
    ----------
    datasets : dict
        ``{"laser": ds_laser, "plasma": ds_plasma, ...}``
    process_order, interprocess_edges, include_F, as_dataframe
        Same semantics as :func:`build_ground_truth_dag`.
    """
    order = process_order or PROCESS_ORDER

    process_observables: Dict[str, dict] = {}
    for proc in order:
        ds = datasets[proc]
        # Disambiguate inputs that share names across processes
        inputs = [
            f"{lab}_{proc}" if _is_shared_name(lab, proc, datasets, order) else lab
            for lab in ds.input_labels
        ]
        output = ds.target_labels[0]  # single output per process
        process_observables[proc] = {"inputs": inputs, "output": output}

    return build_ground_truth_dag(
        process_observables=process_observables,
        process_order=order,
        interprocess_edges=interprocess_edges,
        include_F=include_F,
        as_dataframe=as_dataframe,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_shared_name(
    label: str,
    current_proc: str,
    datasets: Dict[str, SCMDataset],
    order: List[str],
) -> bool:
    """Check if *label* appears as input in more than one process."""
    count = 0
    for proc in order:
        ds = datasets[proc]
        if label in ds.input_labels:
            count += 1
    return count > 1


def dag_to_edge_list(
    adj: np.ndarray | pd.DataFrame,
) -> List[Tuple[str, str]]:
    """Convert adjacency matrix to ``(parent, child)`` edge list.

    Parameters
    ----------
    adj : ndarray or DataFrame
        Adjacency matrix with ``A[i,j]=1 iff j->i``.

    Returns
    -------
    edges : list of (parent, child) tuples
    """
    if isinstance(adj, pd.DataFrame):
        labels = list(adj.index)
        A = adj.values
    else:
        labels = [str(i) for i in range(adj.shape[0])]
        A = adj

    edges = []
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j]:
                edges.append((labels[j], labels[i]))
    return edges
