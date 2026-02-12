"""
Attention-based Causal Discovery.

Extracts the causal graph from the attention weights of a trained CausaliT (ProT)
transformer model. The attention weights are aggregated across heads and batches,
then thresholded to produce a binary adjacency matrix.

For models with LieAttention, the learned DAG probabilities (phi) are also
available and can be used directly.
"""

import numpy as np
import torch
import warnings
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_forecaster(
    checkpoint_path: str,
    device: str = 'cpu',
):
    """
    Load a trained TransformerForecaster from a checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to .ckpt file.
    device : str
        Device for the model.

    Returns
    -------
    model : TransformerForecaster
        Loaded model in eval mode.
    """
    from causaliT.training.forecasters.transformer_forecaster import TransformerForecaster

    model = TransformerForecaster.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
    )
    model.eval()
    model.to(device)
    return model


def extract_attention_weights(
    model,
    data_input: torch.Tensor,
    data_target: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Run a forward pass and extract attention weight tensors.

    Parameters
    ----------
    model : TransformerForecaster
        The loaded model.
    data_input : torch.Tensor
        Encoder input tensor (B, L_enc, D).
    data_target : torch.Tensor
        Decoder target tensor (B, L_dec, D).

    Returns
    -------
    dict
        Keys: 'enc_self', 'dec_self', 'dec_cross'.
        Values: attention tensors. Each is a list (one per layer) of tensors
        with shape (B, H, L, S) or (B, L, S).
    """
    with torch.no_grad():
        forecast, (enc_self_att, dec_self_att, dec_cross_att), enc_mask, _ = model(
            data_input=data_input,
            data_trg=data_target,
        )

    return {
        'enc_self': enc_self_att,
        'dec_self': dec_self_att,
        'dec_cross': dec_cross_att,
        'forecast': forecast,
    }


def aggregate_attention(
    att_list: list,
    aggregation: str = 'mean',
) -> np.ndarray:
    """
    Aggregate attention weights across layers, heads, and batch.

    Parameters
    ----------
    att_list : list of torch.Tensor
        Attention tensors from multiple layers. Each has shape
        (B, H, L, S) or (B, L, S).
    aggregation : str
        'mean' (default) or 'max'.

    Returns
    -------
    np.ndarray
        Aggregated attention matrix of shape (L, S).
    """
    all_att = []
    for att in att_list:
        if att is None:
            continue
        a = att.detach().cpu().numpy()
        if a.ndim == 4:
            # (B, H, L, S) -> mean over batch and heads
            a = a.mean(axis=(0, 1))
        elif a.ndim == 3:
            # (B, L, S) -> mean over batch
            a = a.mean(axis=0)
        all_att.append(a)

    if not all_att:
        return None

    # Stack across layers and aggregate
    stacked = np.stack(all_att, axis=0)
    if aggregation == 'mean':
        return stacked.mean(axis=0)
    elif aggregation == 'max':
        return stacked.max(axis=0)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


def attention_to_adjacency(
    attention_matrix: np.ndarray,
    threshold: float = 0.1,
    normalize: bool = True,
) -> np.ndarray:
    """
    Convert an aggregated attention matrix to a binary adjacency matrix.

    Parameters
    ----------
    attention_matrix : np.ndarray
        Shape (L, S) attention weights.
    threshold : float
        Threshold above which an edge is considered present.
    normalize : bool
        If True, normalize rows to [0, 1] before thresholding.

    Returns
    -------
    np.ndarray
        Binary adjacency matrix.
    """
    A = attention_matrix.copy()
    if normalize:
        row_max = A.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        A = A / row_max
    adj = (A >= threshold).astype(int)
    # Remove self-loops
    np.fill_diagonal(adj, 0)
    return adj


def extract_lie_dag_probabilities(model) -> Dict[str, Optional[np.ndarray]]:
    """
    For models using LieAttention, extract the learned DAG probabilities.

    Returns
    -------
    dict
        Keys: 'enc_phi', 'dec_phi'. Values: DAG probability matrices or None.
    """
    result = {'enc_phi': None, 'dec_phi': None}

    try:
        enc_inner = model.model.encoder.layers[0].global_attention.inner_attention
        if hasattr(enc_inner, 'get_dag_probabilities'):
            with torch.no_grad():
                result['enc_phi'] = enc_inner.get_dag_probabilities().detach().cpu().numpy()
    except (AttributeError, IndexError):
        pass

    try:
        dec_inner = model.model.decoder.layers[0].global_self_attention.inner_attention
        if hasattr(dec_inner, 'get_dag_probabilities'):
            with torch.no_grad():
                result['dec_phi'] = dec_inner.get_dag_probabilities().detach().cpu().numpy()
    except (AttributeError, IndexError):
        pass

    return result


def build_token_to_var_map(
    input_labels: List[str],
    target_labels: List[str],
) -> Dict[str, List[str]]:
    """
    Build mapping from token positions to variable names.

    In flat mode (one token per variable), the encoder sequence has
    one token per input variable, and the decoder has one per target variable.

    Returns
    -------
    dict
        'enc_vars': list of variable names for encoder tokens.
        'dec_vars': list of variable names for decoder tokens.
        'all_vars': concatenated list.
    """
    return {
        'enc_vars': list(input_labels),
        'dec_vars': list(target_labels),
        'all_vars': list(input_labels) + list(target_labels),
    }


def run_attention_discovery(
    checkpoint_path: str,
    data_input: torch.Tensor,
    data_target: torch.Tensor,
    input_var_names: List[str],
    target_var_names: List[str],
    threshold: float = 0.1,
    device: str = 'cpu',
) -> Dict:
    """
    Complete attention-based causal discovery pipeline.

    Parameters
    ----------
    checkpoint_path : str
        Path to CausaliT checkpoint.
    data_input : torch.Tensor
        Encoder input (B, L_enc, D).
    data_target : torch.Tensor
        Decoder target (B, L_dec, D).
    input_var_names : list of str
        Variable names for encoder tokens.
    target_var_names : list of str
        Variable names for decoder tokens.
    threshold : float
        Threshold for attention-to-adjacency conversion.
    device : str
        Compute device.

    Returns
    -------
    dict with keys:
        'enc_self_attention': aggregated encoder self-attention matrix
        'dec_self_attention': aggregated decoder self-attention matrix
        'dec_cross_attention': aggregated decoder cross-attention matrix
        'enc_adjacency': binary adjacency from encoder self-attention
        'cross_adjacency': binary adjacency from cross-attention
        'lie_dag': DAG probabilities from LieAttention (if available)
        'var_map': token-to-variable mapping
        'forecast': model predictions
    """
    model = load_forecaster(checkpoint_path, device=device)

    data_input = data_input.to(device)
    data_target = data_target.to(device)

    # Extract attention weights
    att_dict = extract_attention_weights(model, data_input, data_target)

    # Aggregate attention matrices
    enc_self = aggregate_attention(att_dict['enc_self'])
    dec_self = aggregate_attention(att_dict['dec_self'])
    dec_cross = aggregate_attention(att_dict['dec_cross'])

    # Convert to adjacency
    enc_adj = None
    cross_adj = None
    if enc_self is not None:
        enc_adj = attention_to_adjacency(enc_self, threshold=threshold)
    if dec_cross is not None:
        cross_adj = attention_to_adjacency(dec_cross, threshold=threshold)

    # Extract LieAttention DAG if available
    lie_dag = extract_lie_dag_probabilities(model)

    var_map = build_token_to_var_map(input_var_names, target_var_names)

    return {
        'enc_self_attention': enc_self,
        'dec_self_attention': dec_self,
        'dec_cross_attention': dec_cross,
        'enc_adjacency': enc_adj,
        'cross_adjacency': cross_adj,
        'lie_dag': lie_dag,
        'var_map': var_map,
        'forecast': att_dict['forecast'].detach().cpu().numpy() if att_dict['forecast'] is not None else None,
    }
