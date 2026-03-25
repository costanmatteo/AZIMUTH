"""
Project-specific mask utilities for in-context masking.

Provides mask-building functions used by StageCausalForecaster when
use_in_context_masks is enabled (e.g., for dyconex datasets).
"""


def build_dyconex_in_context_masks(S, X, Y, config):
    """
    Build in-context attention masks from batch data features.

    Used by dyconex-style datasets where causal/category constraints
    are encoded in the data tensor columns.

    Args:
        S: Source tensor (batch, S_seq, features)
        X: Intermediate tensor (batch, X_seq, features)
        Y: Target tensor (batch, Y_seq, features)
        config: Dict with mask specs per decoder layer

    Returns:
        Dict of mask tensors keyed by layer name, or None.
    """
    raise NotImplementedError(
        "build_dyconex_in_context_masks requires a dataset-specific implementation. "
        "Disable in-context masks (use_in_context_masks: false) for non-dyconex datasets."
    )


def merge_masks(static_masks, in_context_masks):
    """
    Merge static (hard) masks with dynamic (in-context) masks.

    If both are present for the same key, combines via element-wise AND.
    If only one is present, returns it unchanged.
    If both are None, returns None.

    Args:
        static_masks: Dict of mask tensors or None
        in_context_masks: Dict of mask tensors or None

    Returns:
        Merged dict of mask tensors, or None if both inputs are None.
    """
    if static_masks is None and in_context_masks is None:
        return None
    if static_masks is None:
        return in_context_masks
    if in_context_masks is None:
        return static_masks

    merged = {}
    all_keys = set(list(static_masks.keys()) + list(in_context_masks.keys()))
    for key in all_keys:
        s_mask = static_masks.get(key)
        ic_mask = in_context_masks.get(key)
        if s_mask is not None and ic_mask is not None:
            merged[key] = s_mask * ic_mask  # element-wise AND for binary masks
        elif s_mask is not None:
            merged[key] = s_mask
        else:
            merged[key] = ic_mask
    return merged
