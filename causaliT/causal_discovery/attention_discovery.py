"""
Attention-based Causal Discovery.

Given a trained CausaliT / ProT checkpoint, perform a forward pass on
trajectory batches and aggregate encoder self-attention and decoder
cross-attention weights into estimated causal graphs.

The mapping from token positions to observable variable names uses
the ``input_vars_map.json`` (encoder) and ``target_vars_map.json``
(decoder) produced during dataset generation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


class AttentionGraphExtractor:
    """Extract and aggregate attention weights from a trained CausaliT model.

    The extractor:

    1. Registers forward hooks on the relevant attention layers.
    2. Runs a forward pass on a batch of trajectories.
    3. Collects the raw attention matrices ``(B, H, L, L)`` or ``(B, H, L, S)``.
    4. Aggregates across batch and heads -> ``(L, L)`` or ``(L, S)``.
    5. Maps token positions to variable names and thresholds the matrix
       to produce a binary adjacency matrix.

    Parameters
    ----------
    model : nn.Module
        A loaded CausaliT / ProT model instance.
    input_vars_map : dict
        ``{variable_name: int_token_id}`` for encoder variables.
    target_vars_map : dict, optional
        ``{variable_name: int_token_id}`` for decoder variables.
    device : str
        Torch device.
    """

    def __init__(
        self,
        model: nn.Module,
        input_vars_map: Dict[str, int],
        target_vars_map: Optional[Dict[str, int]] = None,
        device: str = "cpu",
    ):
        self.model = model
        self.input_vars_map = input_vars_map
        self.target_vars_map = target_vars_map or {}
        self.device = torch.device(device)

        # Reverse maps: token_id -> variable_name
        self._inv_input = {v: k for k, v in input_vars_map.items()}
        self._inv_target = {v: k for k, v in self.target_vars_map.items()}

        # Storage for captured attention weights
        self._captured_attention: Dict[str, List[torch.Tensor]] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _attention_hook(self, layer_name: str):
        """Return a forward-hook closure that captures attention weights."""

        def hook_fn(module, input, output):
            # AttentionLayer.forward returns (V, A, entropy)
            # ScaledDotAttention.forward returns (V, A, entropy)
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]  # A: (B, H, L, S) or (B, L, S)
                if attn_weights is not None:
                    self._captured_attention.setdefault(layer_name, []).append(
                        attn_weights.detach().cpu()
                    )

        return hook_fn

    def register_hooks(
        self,
        enc_self_attn_names: Optional[List[str]] = None,
        dec_cross_attn_names: Optional[List[str]] = None,
    ) -> None:
        """Register forward hooks on named attention sub-modules.

        If names are not provided, the method tries to auto-discover
        attention layers by matching common naming patterns in ProT.
        """
        self.remove_hooks()
        self._captured_attention.clear()

        targets: Dict[str, str] = {}

        if enc_self_attn_names or dec_cross_attn_names:
            for name in enc_self_attn_names or []:
                targets[name] = name
            for name in dec_cross_attn_names or []:
                targets[name] = name
        else:
            # Auto-discover: look for attention layers in encoder / decoder
            for name, module in self.model.named_modules():
                # Match typical ProT naming patterns
                is_attn = (
                    "attention" in name.lower()
                    and hasattr(module, "forward")
                    and isinstance(module, nn.Module)
                )
                if is_attn:
                    # Classify as encoder self, decoder self, or decoder cross
                    if "enc" in name.lower() or "encoder" in name.lower():
                        targets[name] = name
                    elif "cross" in name.lower():
                        targets[name] = name
                    elif "dec" in name.lower() or "decoder" in name.lower():
                        targets[name] = name

        for mod_name, label in targets.items():
            parts = mod_name.split(".")
            mod = self.model
            for p in parts:
                mod = getattr(mod, p)
            hook = mod.register_forward_hook(self._attention_hook(label))
            self._hooks.append(hook)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Forward pass & aggregation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect_attention(
        self, dataloader: torch.utils.data.DataLoader, max_batches: int = 50
    ) -> Dict[str, torch.Tensor]:
        """Run forward passes and return mean attention per layer.

        Parameters
        ----------
        dataloader : DataLoader
            Yields batches compatible with ``model.forward()``.
        max_batches : int
            Limit the number of batches processed.

        Returns
        -------
        dict
            ``{layer_name: mean_attention}``.  Shape per entry:
            ``(L, S)`` after averaging over batch and heads.
        """
        self.model.eval()
        self._captured_attention.clear()

        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                batch = [
                    b.to(self.device) if isinstance(b, torch.Tensor) else b
                    for b in batch
                ]
                self.model(*batch)
            elif isinstance(batch, dict):
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                self.model(**batch)
            else:
                batch = batch.to(self.device)
                self.model(batch)

        # Aggregate
        results: Dict[str, torch.Tensor] = {}
        for layer_name, attn_list in self._captured_attention.items():
            stacked = torch.cat(attn_list, dim=0)  # (total_B, H, L, S) or (total_B, L, S)
            if stacked.dim() == 4:
                # Average over batch (dim 0) and heads (dim 1)
                mean_attn = stacked.mean(dim=(0, 1))  # (L, S)
            elif stacked.dim() == 3:
                mean_attn = stacked.mean(dim=0)  # (L, S)
            else:
                mean_attn = stacked
            results[layer_name] = mean_attn

        return results

    def collect_attention_from_tensor(
        self, x: torch.Tensor, **model_kwargs
    ) -> Dict[str, torch.Tensor]:
        """Run a single forward pass on a pre-built tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor (B, L, D).

        Returns
        -------
        dict
            ``{layer_name: mean_attention (L, S)}``.
        """
        self.model.eval()
        self._captured_attention.clear()

        with torch.no_grad():
            x = x.to(self.device)
            self.model(x, **model_kwargs)

        results: Dict[str, torch.Tensor] = {}
        for layer_name, attn_list in self._captured_attention.items():
            stacked = torch.cat(attn_list, dim=0)
            if stacked.dim() == 4:
                mean_attn = stacked.mean(dim=(0, 1))
            elif stacked.dim() == 3:
                mean_attn = stacked.mean(dim=0)
            else:
                mean_attn = stacked
            results[layer_name] = mean_attn
        return results

    # ------------------------------------------------------------------
    # Token -> variable mapping
    # ------------------------------------------------------------------

    def _map_attention_to_variables(
        self,
        attn_matrix: torch.Tensor,
        row_vars_map: Dict[str, int],
        col_vars_map: Dict[str, int],
    ) -> pd.DataFrame:
        """Map a token-level attention matrix to variable-level.

        When multiple tokens map to the same variable (e.g. multi-step
        sequences), values are averaged.

        Parameters
        ----------
        attn_matrix : Tensor
            Shape ``(L_row, L_col)``.
        row_vars_map : dict
            ``{var_name: token_id}`` for row (query) tokens.
        col_vars_map : dict
            ``{var_name: token_id}`` for column (key) tokens.

        Returns
        -------
        pd.DataFrame
            Variable-level attention, rows=query vars, cols=key vars.
        """
        A = attn_matrix.numpy() if isinstance(attn_matrix, torch.Tensor) else attn_matrix

        inv_row = {}
        for name, tid in row_vars_map.items():
            inv_row.setdefault(name, []).append(tid - 1)  # 0-indexed positions

        inv_col = {}
        for name, tid in col_vars_map.items():
            inv_col.setdefault(name, []).append(tid - 1)

        row_names = sorted(inv_row.keys(), key=lambda n: min(inv_row[n]))
        col_names = sorted(inv_col.keys(), key=lambda n: min(inv_col[n]))

        var_attn = np.zeros((len(row_names), len(col_names)))
        for ri, rname in enumerate(row_names):
            for ci, cname in enumerate(col_names):
                r_idxs = [i for i in inv_row[rname] if i < A.shape[0]]
                c_idxs = [i for i in inv_col[cname] if i < A.shape[1]]
                if r_idxs and c_idxs:
                    sub = A[np.ix_(r_idxs, c_idxs)]
                    var_attn[ri, ci] = sub.mean()

        return pd.DataFrame(var_attn, index=row_names, columns=col_names)

    def attention_to_variable_graph(
        self,
        attn_matrices: Dict[str, torch.Tensor],
        layer_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Convert aggregated attention to a variable-level attention DataFrame.

        Parameters
        ----------
        attn_matrices : dict
            Output of :meth:`collect_attention`.
        layer_name : str, optional
            Specific layer to use.  If *None*, use the first available
            encoder self-attention layer (or fall back to any layer).

        Returns
        -------
        pd.DataFrame
            Variable-level attention values (not yet thresholded).
        """
        if layer_name:
            attn = attn_matrices[layer_name]
            # Determine if encoder (square, self-attn) or cross-attn
            if attn.shape[0] == attn.shape[1]:
                return self._map_attention_to_variables(
                    attn, self.input_vars_map, self.input_vars_map
                )
            else:
                return self._map_attention_to_variables(
                    attn, self.target_vars_map, self.input_vars_map
                )

        # Auto-select: prefer encoder self-attention
        for name, attn in attn_matrices.items():
            if "enc" in name.lower() and attn.shape[0] == attn.shape[1]:
                return self._map_attention_to_variables(
                    attn, self.input_vars_map, self.input_vars_map
                )

        # Fall back to first available
        name, attn = next(iter(attn_matrices.items()))
        if attn.shape[0] == attn.shape[1]:
            return self._map_attention_to_variables(
                attn, self.input_vars_map, self.input_vars_map
            )
        return self._map_attention_to_variables(
            attn, self.target_vars_map, self.input_vars_map
        )

    # ------------------------------------------------------------------
    # Thresholding
    # ------------------------------------------------------------------

    @staticmethod
    def threshold_to_adjacency(
        var_attention: pd.DataFrame,
        threshold: float = 0.1,
        remove_self_loops: bool = True,
    ) -> pd.DataFrame:
        """Convert variable-level attention values into a binary adjacency matrix.

        Parameters
        ----------
        var_attention : pd.DataFrame
            Variable-level attention (e.g. output of
            :meth:`attention_to_variable_graph`).
        threshold : float
            Minimum attention weight to consider an edge present.
        remove_self_loops : bool
            Zero out the diagonal.

        Returns
        -------
        pd.DataFrame
            Binary adjacency matrix (1 = edge, 0 = no edge).
        """
        adj = (var_attention.values >= threshold).astype(int)
        if remove_self_loops:
            np.fill_diagonal(adj, 0)
        return pd.DataFrame(adj, index=var_attention.index, columns=var_attention.columns)

    @staticmethod
    def adaptive_threshold(
        var_attention: pd.DataFrame,
        percentile: float = 75.0,
        remove_self_loops: bool = True,
    ) -> pd.DataFrame:
        """Threshold using a percentile of the attention values.

        Parameters
        ----------
        var_attention : pd.DataFrame
            Variable-level attention.
        percentile : float
            Percentile of non-diagonal values to use as threshold.
        remove_self_loops : bool
            Zero out diagonal.

        Returns
        -------
        pd.DataFrame
            Binary adjacency.
        """
        vals = var_attention.values.copy()
        if remove_self_loops:
            np.fill_diagonal(vals, 0)
        flat = vals.flatten()
        flat = flat[flat > 0]  # only positive values
        if len(flat) == 0:
            thr = 0.0
        else:
            thr = np.percentile(flat, percentile)
        adj = (vals >= thr).astype(int)
        if remove_self_loops:
            np.fill_diagonal(adj, 0)
        return pd.DataFrame(adj, index=var_attention.index, columns=var_attention.columns)


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_vars_maps(data_dir: Union[str, Path]) -> Tuple[Dict, Dict]:
    """Load variable maps from a generated dataset directory.

    Returns
    -------
    input_vars_map, target_vars_map
    """
    data_dir = Path(data_dir)
    iv, tv = {}, {}
    iv_path = data_dir / "input_vars_map.json"
    tv_path = data_dir / "target_vars_map.json"
    if iv_path.exists():
        with open(iv_path) as f:
            iv = json.load(f)
    if tv_path.exists():
        with open(tv_path) as f:
            tv = json.load(f)
    return iv, tv
