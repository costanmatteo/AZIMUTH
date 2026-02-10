"""
Ground Truth DAG Extraction from SCM Datasets.

Extracts the full-chain ground truth DAG at the level of observable
variables (input_labels + target_labels of each process, plus F),
excluding latent/intermediate nodes.

Each variable is prefixed with its process name (``process/var``) to
disambiguate names that appear in multiple processes (e.g. ``Duration``
exists in plasma, galvanic, and microetch as different physical
parameters).

The full-chain DAG captures:
  1. **Intra-process** edges derived from each SCM (marginalising
     intermediate/latent nodes).
  2. **Inter-process** edges: the output of process *i* feeds into the
     inputs of every downstream process *i+1, i+2, …*
  3. **F edges**: every process output → F.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

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


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------

def _prefixed(process: str, var: str) -> str:
    """Return ``'process/var'``."""
    return f"{process}/{var}"


def get_observable_variables(
    datasets: Dict[str, SCMDataset],
    process_order: Optional[List[str]] = None,
    include_F: bool = True,
) -> List[str]:
    """Return the ordered list of observable variable names across the chain.

    Every variable is prefixed with its process name to avoid collisions
    (e.g. ``plasma/Duration`` vs ``galvanic/Duration``).

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
        Ordered observable variable names (``process/var`` format).
    """
    process_order = process_order or DEFAULT_PROCESS_ORDER
    obs_vars: List[str] = []
    for proc in process_order:
        if proc not in datasets:
            continue
        ds = datasets[proc]
        for v in ds.input_labels + ds.target_labels:
            obs_vars.append(_prefixed(proc, v))
    if include_F:
        obs_vars.append("F")
    return obs_vars


def get_process_for_variable(
    var_name: str,
    datasets: Dict[str, SCMDataset],
    process_order: Optional[List[str]] = None,
) -> Optional[str]:
    """Return the process that owns a prefixed variable name."""
    if "/" in var_name:
        return var_name.split("/", 1)[0]
    return None


# ---------------------------------------------------------------------------
# Intra-process edges
# ---------------------------------------------------------------------------

def _build_intra_process_edges(
    process: str,
    ds: SCMDataset,
) -> List[Tuple[str, str]]:
    """Directed edges among observable variables within a single process.

    Uses BFS on the full SCM to find paths between observable variables
    (marginalising over intermediate/latent nodes).  Returns edges in
    prefixed format (``process/var``).
    """
    observable = set(ds.input_labels + ds.target_labels)
    scm = ds.scm

    # Build children map from the full SCM
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
                edges.append((_prefixed(process, src), _prefixed(process, node)))
                continue  # don't traverse past an observable node
            for c in children.get(node, []):
                if c not in visited:
                    queue.append(c)
    return edges


# ---------------------------------------------------------------------------
# Inter-process edges
# ---------------------------------------------------------------------------

def _build_inter_process_edges(
    datasets: Dict[str, SCMDataset],
    process_order: List[str],
) -> List[Tuple[str, str]]:
    """Directed edges connecting upstream outputs to downstream inputs.

    Every output of process *i* is connected to every input of every
    downstream process *j* (j > i).  This reflects the manufacturing
    chain where the material state propagates forward.
    """
    edges: List[Tuple[str, str]] = []
    upstream_outputs: List[Tuple[str, str]] = []  # (process, var)

    for proc in process_order:
        if proc not in datasets:
            continue
        ds = datasets[proc]
        # Connect all upstream outputs to every input of the current process
        for up_proc, up_var in upstream_outputs:
            for inp in ds.input_labels:
                edges.append((
                    _prefixed(up_proc, up_var),
                    _prefixed(proc, inp),
                ))
        # Accumulate outputs
        for t in ds.target_labels:
            upstream_outputs.append((proc, t))
    return edges


# ---------------------------------------------------------------------------
# F edges
# ---------------------------------------------------------------------------

def _build_F_edges(
    datasets: Dict[str, SCMDataset],
    process_order: List[str],
) -> List[Tuple[str, str]]:
    """Edges from every process output to ``F``."""
    edges: List[Tuple[str, str]] = []
    for proc in process_order:
        if proc not in datasets:
            continue
        ds = datasets[proc]
        for t in ds.target_labels:
            edges.append((_prefixed(proc, t), "F"))
    return edges


# ---------------------------------------------------------------------------
# Main DAG builder
# ---------------------------------------------------------------------------

def extract_ground_truth_dag(
    datasets: Dict[str, SCMDataset],
    process_order: Optional[List[str]] = None,
    include_F: bool = True,
    as_dataframe: bool = True,
) -> Union[np.ndarray, pd.DataFrame]:
    """Build the ground-truth adjacency matrix for the full process chain.

    The DAG captures:

    * **Intra-process** edges (input -> output for each SCM, marginalising
      over intermediate/latent nodes).
    * **Inter-process** edges (output of process *i* -> inputs of every
      downstream process).
    * **F edges** (all process outputs -> F).

    All variable names are prefixed with their process
    (``"process/var"``), so ``Duration`` in different processes becomes
    ``plasma/Duration``, ``galvanic/Duration``, ``microetch/Duration``.

    Parameters
    ----------
    datasets : dict
        ``{process_name: SCMDataset}`` for the chain.
    process_order : list of str, optional
        Chain ordering.
    include_F : bool
        Append reliability ``F`` as a sink node.
    as_dataframe : bool
        If *True*, return a labelled ``pd.DataFrame``; otherwise a
        ``np.ndarray``.

    Returns
    -------
    adj : ndarray or DataFrame
        ``adj[i, j] = 1`` means edge from variable *j* to variable *i*
        (i.e. *i* can see *j*; row = child, col = parent).
    """
    process_order = process_order or DEFAULT_PROCESS_ORDER
    obs_vars = get_observable_variables(datasets, process_order, include_F)
    idx = {v: i for i, v in enumerate(obs_vars)}
    n = len(obs_vars)
    adj = np.zeros((n, n), dtype=int)

    def _set_edge(parent: str, child: str):
        if parent in idx and child in idx:
            adj[idx[child], idx[parent]] = 1

    # 1. Intra-process edges
    for proc in process_order:
        if proc not in datasets:
            continue
        for u, v in _build_intra_process_edges(proc, datasets[proc]):
            _set_edge(u, v)

    # 2. Inter-process edges
    for u, v in _build_inter_process_edges(datasets, process_order):
        _set_edge(u, v)

    # 3. F edges
    if include_F:
        for u, v in _build_F_edges(datasets, process_order):
            _set_edge(u, v)

    if as_dataframe:
        return pd.DataFrame(adj, index=obs_vars, columns=obs_vars)
    return adj


# ---------------------------------------------------------------------------
# Per-process DAG extraction (matches CausaliT attention masks)
# ---------------------------------------------------------------------------

def _marginalised_observable_adj(
    ds: SCMDataset,
) -> pd.DataFrame:
    """Compute adjacency among observable variables, marginalising intermediates.

    Uses BFS on the full SCM graph to find causal paths between observable
    variables (``input_labels + target_labels``), skipping over any
    intermediate/latent nodes.

    Returns
    -------
    pd.DataFrame
        ``adj[child, parent] = 1`` for observable variables only.
    """
    observable = ds.input_labels + ds.target_labels
    obs_set = set(observable)
    scm = ds.scm

    # Build children map from the full SCM
    children: Dict[str, List[str]] = {}
    for spec in scm.specs.values():
        children.setdefault(spec.name, [])
        for p in spec.parents:
            children.setdefault(p, []).append(spec.name)

    n = len(observable)
    idx = {v: i for i, v in enumerate(observable)}
    adj = np.zeros((n, n), dtype=int)

    for src in observable:
        # BFS from src; stop at observable nodes (except src itself)
        visited = set()
        queue = [src]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            if node != src and node in obs_set:
                adj[idx[node], idx[src]] = 1
                continue  # don't traverse past an observable node
            for c in children.get(node, []):
                if c not in visited:
                    queue.append(c)

    return pd.DataFrame(adj, index=observable, columns=observable)


def extract_process_dag(
    ds: SCMDataset,
    as_dataframe: bool = True,
) -> Union[np.ndarray, pd.DataFrame]:
    """Extract the DAG for a single process (matching its attention masks).

    Uses BFS to marginalise over intermediate/latent nodes, so that
    the adjacency captures the causal paths between observable variables
    even when they are separated by many intermediate steps in the SCM.

    This mirrors the masks generated by ``SCMDataset.generate_ds()``:

    * ``enc_self_att_mask``: adjacency among ``input_labels``
    * ``dec_cross_att_mask``: adjacency from ``target_labels`` to
      ``input_labels``
    * ``dec_self_att_mask``: adjacency among ``target_labels``

    Here we return a single combined adjacency over
    ``input_labels + target_labels``.

    Parameters
    ----------
    ds : SCMDataset
        A single process dataset.
    as_dataframe : bool
        Label the matrix.

    Returns
    -------
    adj : ndarray or DataFrame
        ``adj[i, j] = 1`` means *i* <- *j*.
    """
    adj = _marginalised_observable_adj(ds)
    if as_dataframe:
        return adj
    return adj.values


def extract_process_masks(
    ds: SCMDataset,
) -> Dict[str, pd.DataFrame]:
    """Extract the three attention masks for a single process.

    Uses BFS marginalisation to find causal paths through intermediate
    nodes, so masks are correct even for processes with many latent
    variables (e.g. laser).

    Returns
    -------
    dict
        ``{"enc_self": DataFrame, "dec_cross": DataFrame,
          "dec_self": DataFrame}``.
    """
    adj = _marginalised_observable_adj(ds)
    return {
        "enc_self": adj.loc[ds.input_labels, ds.input_labels],
        "dec_cross": adj.loc[ds.target_labels, ds.input_labels],
        "dec_self": adj.loc[ds.target_labels, ds.target_labels],
    }
