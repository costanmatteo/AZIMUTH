"""
Attention-based causal discovery from trained CausaliT / ProT models.

Extracts attention weights from a forward pass on trajectory data, aggregates
over batch and heads, and converts the resulting attention map into an
estimated adjacency matrix via thresholding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_attention_weights(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Run a forward pass and return raw attention weight tensors.

    Parameters
    ----------
    model : nn.Module
        A trained ProT (or CausaliT) model in eval mode.
    input_tensor : torch.Tensor
        Encoder input, shape ``(B, L_enc, D)``.
    target_tensor : torch.Tensor
        Decoder input, shape ``(B, L_dec, D)``.
    device : str
        Device for inference.

    Returns
    -------
    dict
        Keys ``"enc_self_att"``, ``"dec_self_att"``, ``"dec_cross_att"``,
        each mapping to a tensor of shape ``(B, H, L, S)`` or ``(B, L, S)``.
    """
    model = model.to(device).eval()
    input_tensor = input_tensor.to(device)
    target_tensor = target_tensor.to(device)

    _, (enc_self, dec_self, dec_cross), _, _ = model(input_tensor, target_tensor)

    return {
        "enc_self_att":  enc_self.detach().cpu(),
        "dec_self_att":  dec_self.detach().cpu(),
        "dec_cross_att": dec_cross.detach().cpu(),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_attention(
    att: torch.Tensor,
    method: str = "mean",
) -> np.ndarray:
    """Aggregate attention tensor over batch and heads.

    Parameters
    ----------
    att : torch.Tensor
        Shape ``(B, H, L, S)`` (multi-head) or ``(B, L, S)`` (single-head).
    method : str
        ``"mean"`` or ``"max"`` aggregation.

    Returns
    -------
    agg : ndarray
        Shape ``(L, S)`` aggregated attention map.
    """
    if att.dim() == 4:
        # (B, H, L, S) -> mean over B and H
        if method == "mean":
            agg = att.mean(dim=(0, 1))
        else:
            agg = att.amax(dim=(0, 1))
    elif att.dim() == 3:
        # (B, L, S) -> mean over B
        if method == "mean":
            agg = att.mean(dim=0)
        else:
            agg = att.amax(dim=0)
    else:
        raise ValueError(f"Unexpected attention tensor ndim={att.dim()}")

    return agg.numpy()


# ---------------------------------------------------------------------------
# Token -> variable mapping
# ---------------------------------------------------------------------------

def load_vars_map(path: Union[str, Path]) -> Dict[str, int]:
    """Load a variable mapping JSON (``{var_name: int_id}``)."""
    with open(path, "r") as f:
        return json.load(f)


def invert_vars_map(vars_map: Dict[str, int]) -> Dict[int, str]:
    """Invert ``{name: id}`` to ``{id: name}``."""
    return {v: k for k, v in vars_map.items()}


def map_attention_to_variables(
    att_map: np.ndarray,
    query_vars_map: Dict[str, int],
    key_vars_map: Dict[str, int],
) -> pd.DataFrame:
    """Label an ``(L, S)`` attention map with variable names.

    The attention maps produced by ProT have one token per variable, ordered
    by their integer ID in the vars_map.  This function labels rows (queries)
    and columns (keys) accordingly.

    Parameters
    ----------
    att_map : ndarray
        Shape ``(L, S)`` aggregated attention.
    query_vars_map : dict
        ``{var_name: id}`` for the query side (rows).
    key_vars_map : dict
        ``{var_name: id}`` for the key side (columns).

    Returns
    -------
    DataFrame
        Labelled attention map.
    """
    q_order = sorted(query_vars_map, key=query_vars_map.get)
    k_order = sorted(key_vars_map, key=key_vars_map.get)

    # Trim or pad if sizes don't match (safety)
    L, S = att_map.shape
    q_labels = q_order[:L] if len(q_order) >= L else q_order + [f"q_{i}" for i in range(len(q_order), L)]
    k_labels = k_order[:S] if len(k_order) >= S else k_order + [f"k_{i}" for i in range(len(k_order), S)]

    return pd.DataFrame(att_map, index=q_labels, columns=k_labels)


# ---------------------------------------------------------------------------
# Thresholding -> adjacency
# ---------------------------------------------------------------------------

def attention_to_adjacency(
    att_map: np.ndarray,
    threshold: float = 0.1,
    method: str = "absolute",
) -> np.ndarray:
    """Convert an aggregated attention map to a binary adjacency matrix.

    Convention:  ``A[i, j] = 1`` means key *j* attends to query *i*,
    i.e. *j -> i* (same convention as ``SCM.adjacency``).

    Parameters
    ----------
    att_map : ndarray
        Shape ``(L, S)`` with non-negative attention values.
    threshold : float
        Cut-off for edge inclusion.
    method : str
        ``"absolute"``  -- edge if ``att[i,j] >= threshold``.
        ``"quantile"``  -- edge if ``att[i,j] >= quantile(threshold)``
                          (threshold interpreted as quantile level 0-1).
        ``"top_k"``     -- keep *threshold* strongest edges per row.

    Returns
    -------
    adj : ndarray of int
        Binary adjacency matrix, same shape as *att_map*.
    """
    if method == "absolute":
        adj = (att_map >= threshold).astype(int)
    elif method == "quantile":
        q = np.quantile(att_map, threshold)
        adj = (att_map >= q).astype(int)
    elif method == "top_k":
        k = int(threshold)
        adj = np.zeros_like(att_map, dtype=int)
        for i in range(att_map.shape[0]):
            top_idx = np.argsort(att_map[i])[-k:]
            adj[i, top_idx] = 1
    else:
        raise ValueError(f"Unknown thresholding method: {method}")

    # Remove self-loops
    np.fill_diagonal(adj, 0)
    return adj


# ---------------------------------------------------------------------------
# High-level convenience
# ---------------------------------------------------------------------------

def discover_dag_from_attention(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    input_vars_map: Dict[str, int],
    target_vars_map: Dict[str, int],
    threshold: float = 0.1,
    threshold_method: str = "absolute",
    aggregation: str = "mean",
    attention_type: str = "dec_cross_att",
    device: str = "cpu",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end: model forward -> aggregate -> threshold -> labelled DAG.

    Parameters
    ----------
    model, input_tensor, target_tensor, device
        See :func:`extract_attention_weights`.
    input_vars_map, target_vars_map
        Variable name -> integer ID mappings.
    threshold, threshold_method
        See :func:`attention_to_adjacency`.
    aggregation
        ``"mean"`` or ``"max"`` (see :func:`aggregate_attention`).
    attention_type
        Which attention block to use: ``"enc_self_att"``,
        ``"dec_self_att"``, or ``"dec_cross_att"``.

    Returns
    -------
    att_df : DataFrame
        Labelled aggregated attention map.
    adj_df : DataFrame
        Labelled binary adjacency matrix.
    """
    atts = extract_attention_weights(model, input_tensor, target_tensor, device)
    raw = atts[attention_type]
    agg = aggregate_attention(raw, method=aggregation)

    # Determine query/key variable maps based on attention type
    if attention_type == "enc_self_att":
        q_map, k_map = input_vars_map, input_vars_map
    elif attention_type == "dec_self_att":
        q_map, k_map = target_vars_map, target_vars_map
    elif attention_type == "dec_cross_att":
        q_map, k_map = target_vars_map, input_vars_map
    else:
        raise ValueError(f"Unknown attention_type: {attention_type}")

    att_df = map_attention_to_variables(agg, q_map, k_map)

    adj = attention_to_adjacency(agg, threshold=threshold, method=threshold_method)
    adj_df = map_attention_to_variables(adj, q_map, k_map)

    return att_df, adj_df


def discover_full_dag(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    input_vars_map: Dict[str, int],
    target_vars_map: Dict[str, int],
    threshold: float = 0.1,
    threshold_method: str = "absolute",
    aggregation: str = "mean",
    device: str = "cpu",
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Extract and threshold all three attention blocks at once.

    Returns
    -------
    att_maps : dict of DataFrames
        Aggregated attention per block.
    adj_maps : dict of DataFrames
        Binary adjacency per block.
    """
    att_maps = {}
    adj_maps = {}

    for att_type in ["enc_self_att", "dec_self_att", "dec_cross_att"]:
        att_df, adj_df = discover_dag_from_attention(
            model=model,
            input_tensor=input_tensor,
            target_tensor=target_tensor,
            input_vars_map=input_vars_map,
            target_vars_map=target_vars_map,
            threshold=threshold,
            threshold_method=threshold_method,
            aggregation=aggregation,
            attention_type=att_type,
            device=device,
        )
        att_maps[att_type] = att_df
        adj_maps[att_type] = adj_df

    return att_maps, adj_maps
