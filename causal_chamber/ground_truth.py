"""
Ground Truth DAG Extraction from SCM Datasets.

Extracts the inter-process causal DAG at the level of observable variables
(process inputs/outputs + reliability F), filtering out latent/intermediate nodes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Sequence

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scm_ds.datasets import (
    ds_scm_laser, ds_scm_plasma, ds_scm_galvanic, ds_scm_microetch
)
from scm_ds.scm import SCMDataset


# ---------------------------------------------------------------------------
# Process chain definition
# ---------------------------------------------------------------------------

PROCESS_ORDER = ['laser', 'plasma', 'galvanic', 'microetch']

PROCESS_DATASETS: Dict[str, SCMDataset] = {
    'laser': ds_scm_laser,
    'plasma': ds_scm_plasma,
    'galvanic': ds_scm_galvanic,
    'microetch': ds_scm_microetch,
}

# Observable variable names per process (input_labels + target_labels)
PROCESS_OBSERVABLE_VARS: Dict[str, Dict[str, List[str]]] = {
    'laser': {
        'inputs': ['PowerTarget', 'AmbientTemp'],
        'outputs': ['ActualPower'],
    },
    'plasma': {
        'inputs': ['RF_Power', 'Duration'],
        'outputs': ['RemovalRate'],
    },
    'galvanic': {
        'inputs': ['CurrentDensity', 'Duration'],
        'outputs': ['Thickness'],
    },
    'microetch': {
        'inputs': ['Temperature', 'Concentration', 'Duration'],
        'outputs': ['RemovalDepth'],
    },
}


def get_all_observable_vars() -> List[str]:
    """Return ordered list of all observable variables across the pipeline + F."""
    obs = []
    for proc in PROCESS_ORDER:
        info = PROCESS_OBSERVABLE_VARS[proc]
        obs.extend(info['inputs'])
        obs.extend(info['outputs'])
    obs.append('F')
    return obs


def get_intra_process_edges(process_name: str) -> List[Tuple[str, str]]:
    """
    Get edges within a single process at the observable variable level.

    Each observable input of the process has an edge to each observable output.
    """
    info = PROCESS_OBSERVABLE_VARS[process_name]
    edges = []
    for inp in info['inputs']:
        for out in info['outputs']:
            edges.append((inp, out))
    return edges


def get_inter_process_edges() -> List[Tuple[str, str]]:
    """
    Get edges between processes: the output of process i feeds as context
    into process i+1's inputs (cross-process causal influence).

    In the Azimuth pipeline, the output of each upstream process causally
    influences the quality targets of downstream processes (via the adaptive
    target mechanism in ReliabilityFunction). We model this as: each process
    output has a directed edge to the next process's output (since the next
    process's output is a function of upstream quality).

    Additionally, every process output has an edge to F.
    """
    edges = []
    outputs_so_far = []

    for proc in PROCESS_ORDER:
        info = PROCESS_OBSERVABLE_VARS[proc]
        current_outputs = info['outputs']

        # Upstream outputs influence current process outputs
        for upstream_out in outputs_so_far:
            for cur_out in current_outputs:
                edges.append((upstream_out, cur_out))

        outputs_so_far.extend(current_outputs)

    # Every process output has an edge to F
    for proc in PROCESS_ORDER:
        for out in PROCESS_OBSERVABLE_VARS[proc]['outputs']:
            edges.append((out, 'F'))

    return edges


def get_ground_truth_edges() -> List[Tuple[str, str]]:
    """
    Return the complete ground truth edge list for the inter-process DAG
    at the observable variable level.
    """
    edges = []
    for proc in PROCESS_ORDER:
        edges.extend(get_intra_process_edges(proc))
    edges.extend(get_inter_process_edges())
    return edges


def get_ground_truth_adjacency(
    nodes: Optional[List[str]] = None,
    as_dataframe: bool = True,
) -> np.ndarray:
    """
    Build the ground truth adjacency matrix for the inter-process DAG.

    Convention: A[i, j] = 1 means node j -> node i (j is parent of i),
    consistent with scm_ds.scm.SCM.adjacency(positive_child=True).

    Parameters
    ----------
    nodes : list of str, optional
        Node ordering. Defaults to get_all_observable_vars().
    as_dataframe : bool
        If True, return a labeled pd.DataFrame.

    Returns
    -------
    np.ndarray or pd.DataFrame
        Adjacency matrix of shape (|V|, |V|).
    """
    if nodes is None:
        nodes = get_all_observable_vars()

    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n), dtype=int)

    for parent, child in get_ground_truth_edges():
        if parent in idx and child in idx:
            A[idx[child], idx[parent]] = 1  # positive_child convention

    if as_dataframe:
        return pd.DataFrame(A, index=nodes, columns=nodes)
    return A


def get_ground_truth_adjacency_parent_convention(
    nodes: Optional[List[str]] = None,
    as_dataframe: bool = True,
) -> np.ndarray:
    """
    Adjacency matrix with standard convention: A[i, j] = 1 means i -> j.
    This is the convention used by most causal discovery algorithms.
    """
    if nodes is None:
        nodes = get_all_observable_vars()

    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n), dtype=int)

    for parent, child in get_ground_truth_edges():
        if parent in idx and child in idx:
            A[idx[parent], idx[child]] = 1

    if as_dataframe:
        return pd.DataFrame(A, index=nodes, columns=nodes)
    return A


def get_process_datasets() -> Dict[str, SCMDataset]:
    """Return the process SCMDataset instances."""
    return PROCESS_DATASETS.copy()


def sample_pipeline_data(
    n: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample observational data from all processes and concatenate into a
    single DataFrame with columns for all observable variables.

    Note: Each process is sampled independently; cross-process dependencies
    are structural (captured by the DAG), not by passing actual values.
    """
    rng_seeds = np.random.SeedSequence(seed).spawn(len(PROCESS_ORDER))
    frames = {}
    for i, proc in enumerate(PROCESS_ORDER):
        ds = PROCESS_DATASETS[proc]
        df = ds.sample(n, seed=int(rng_seeds[i].generate_state(1)[0]))
        info = PROCESS_OBSERVABLE_VARS[proc]
        for col in info['inputs'] + info['outputs']:
            frames[col] = df[col].values

    return pd.DataFrame(frames)
