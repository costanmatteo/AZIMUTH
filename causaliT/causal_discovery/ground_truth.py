"""
Ground Truth DAG Extraction from SCM Datasets.

Extracts the inter-process ground truth DAG at the level of observable
variables (input_labels + target_labels of each process, plus F),
excluding latent/intermediate nodes.

The inter-process DAG follows the manufacturing chain:
  laser inputs -> laser output -> plasma inputs -> plasma output -> ... -> F
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scm_ds.scm import SCM, SCMDataset


# ---------------------------------------------------------------------------
# Default process chain ordering
# ---------------------------------------------------------------------------

DEFAULT_PROCESS_ORDER = ["laser", "plasma", "galvanic", "microetch"]


def get_observable_variables(
    datasets: Dict[str, SCMDataset],
    process_order: Optional[List[str]] = None,
    include_F: bool = True,
) -> List[str]:
    """Return the ordered list of observable variable names across all processes.

    Observable variables are defined as the union of ``input_labels`` and
    ``target_labels`` for each process in the chain, plus optionally the
    reliability score *F*.

    Parameters
    ----------
    datasets : dict
        Mapping ``process_name -> SCMDataset``.
    process_order : list of str, optional
        Order of processes.  Defaults to :data:`DEFAULT_PROCESS_ORDER`.
    include_F : bool
        Whether to append ``"F"`` at the end.

    Returns
    -------
    list of str
        Ordered observable variable names.
    """
    process_order = process_order or DEFAULT_PROCESS_ORDER
    obs_vars: List[str] = []
    seen = set()
    for proc in process_order:
        if proc not in datasets:
            continue
        ds = datasets[proc]
        for v in ds.input_labels + ds.target_labels:
            if v not in seen:
                obs_vars.append(v)
                seen.add(v)
    if include_F and "F" not in seen:
        obs_vars.append("F")
    return obs_vars


def _build_intra_process_edges(ds: SCMDataset) -> List[Tuple[str, str]]:
    """Return directed edges among observable variables within a single process.

    Only edges between variables in ``input_labels`` and ``target_labels`` are
    kept (i.e. intermediate/latent nodes are marginalised out).  An edge
    ``(u, v)`` is included when there is a directed path from *u* to *v*
    through the full SCM that does not pass through another observable node.
    """
    observable = set(ds.input_labels + ds.target_labels)
    scm = ds.scm

    # Build reachability among observables via BFS on full DAG
    children: Dict[str, List[str]] = {}
    for spec in scm.specs.values():
        children.setdefault(spec.name, [])
        for p in spec.parents:
            children.setdefault(p, []).append(spec.name)

    edges: List[Tuple[str, str]] = []
    for src in observable:
        # BFS from src; stop at observable nodes (except src itself)
        visited = set()
        queue = [src]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            if node != src and node in observable:
                edges.append((src, node))
                continue  # don't traverse past an observable node
            for c in children.get(node, []):
                if c not in visited:
                    queue.append(c)
    return edges


def _build_inter_process_edges(
    datasets: Dict[str, SCMDataset],
    process_order: List[str],
) -> List[Tuple[str, str]]:
    """Return directed edges connecting the output of process *i* to the inputs
    of process *i+1* in the manufacturing chain.

    Convention: the target (output) of process *i* feeds into all inputs of
    process *i+1* that are not already inputs of an earlier process.
    """
    edges: List[Tuple[str, str]] = []
    upstream_outputs: List[str] = []

    for i, proc in enumerate(process_order):
        if proc not in datasets:
            continue
        ds = datasets[proc]
        if upstream_outputs:
            # Connect each upstream output to every input of the current process
            for up_out in upstream_outputs:
                for inp in ds.input_labels:
                    edges.append((up_out, inp))
        # Accumulate outputs
        upstream_outputs.extend(ds.target_labels)
    return edges


def _build_F_edges(
    datasets: Dict[str, SCMDataset],
    process_order: List[str],
) -> List[Tuple[str, str]]:
    """Return edges from every process output to ``F``."""
    edges: List[Tuple[str, str]] = []
    for proc in process_order:
        if proc not in datasets:
            continue
        ds = datasets[proc]
        for t in ds.target_labels:
            edges.append((t, "F"))
    return edges


def extract_ground_truth_dag(
    datasets: Dict[str, SCMDataset],
    process_order: Optional[List[str]] = None,
    include_F: bool = True,
    as_dataframe: bool = True,
) -> np.ndarray | pd.DataFrame:
    """Build the ground-truth adjacency matrix over observable variables.

    The DAG captures:

    * **Intra-process** edges (input -> output for each SCM, marginalising
      over intermediate nodes).
    * **Inter-process** edges (output of process *i* -> inputs of process
      *i+1*).
    * **F edges** (all process outputs -> F).

    Parameters
    ----------
    datasets : dict
        ``{process_name: SCMDataset}`` for the chain.
    process_order : list of str, optional
        Chain ordering.
    include_F : bool
        Append reliability ``F`` as a sink node.
    as_dataframe : bool
        If *True*, return a labelled ``pd.DataFrame``; otherwise ``np.ndarray``.

    Returns
    -------
    adj : ndarray or DataFrame
        ``adj[i, j] = 1`` means edge from variable *j* to variable *i*
        (i.e. *i* can see *j*, matching the convention in
        :meth:`SCM.adjacency`).
    """
    process_order = process_order or DEFAULT_PROCESS_ORDER
    obs_vars = get_observable_variables(datasets, process_order, include_F)
    idx = {v: i for i, v in enumerate(obs_vars)}
    n = len(obs_vars)
    adj = np.zeros((n, n), dtype=int)

    # 1. Intra-process edges
    for proc in process_order:
        if proc not in datasets:
            continue
        for u, v in _build_intra_process_edges(datasets[proc]):
            if u in idx and v in idx:
                adj[idx[v], idx[u]] = 1  # v <- u  (row=child, col=parent)

    # 2. Inter-process edges
    for u, v in _build_inter_process_edges(datasets, process_order):
        if u in idx and v in idx:
            adj[idx[v], idx[u]] = 1

    # 3. F edges
    if include_F:
        for u, v in _build_F_edges(datasets, process_order):
            if u in idx and v in idx:
                adj[idx[v], idx[u]] = 1

    if as_dataframe:
        return pd.DataFrame(adj, index=obs_vars, columns=obs_vars)
    return adj
