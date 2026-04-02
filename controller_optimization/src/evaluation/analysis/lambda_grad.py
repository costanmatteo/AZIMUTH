"""
Lambda_grad (Λ_grad) — Delta-Method complexity metric for AZIMUTH.

Computes the analytic complexity metric Λ_grad(D) (Method 1, Delta Method)
that quantifies how much process noise σ²_ψt is amplified by the surrogate
gradient ∂F̂/∂o_t, providing a first-order approximation to the irreducible
variance of F̂ under the current surrogate and predictors.

    Λ_grad(D) = (1/N) Σ_i Σ_t (∂F̂^(i)/∂o_t^(i))² · σ²_ψt(a_t^(i), c_day^(i))

INPUTS
------
- dataset D : N trajectories, each containing per-stage actions, outputs, context
- frozen predictors {g_ψt} : UncertaintyPredictor per stage → (μ_t, σ²_t)
- frozen surrogate f_Θ     : maps trajectory → scalar F̂ ∈ [0,1]

ALGORITHM (per trajectory τ^(i))
---------------------------------
1. Forward through each g_ψt(a_t, c_day) → σ²_ψt
2. Forward through f_Θ(τ) → F̂  (with o_t requiring grad)
3. Backward: ∂F̂/∂o_t for all t simultaneously
4. Accumulate: Σ_t (∂F̂/∂o_t)² · σ²_ψt
5. Average over N trajectories
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LambdaGradResult:
    """Results from Λ_grad computation."""

    lambda_grad: float                          # Scalar Λ_grad(D)
    per_stage: Dict[str, float]                 # Per-stage average contribution
    per_stage_grad_sq: Dict[str, float]         # Per-stage average (∂F̂/∂o_t)²
    per_stage_sigma_sq: Dict[str, float]        # Per-stage average σ²_ψt
    n_trajectories: int                         # Number of trajectories N
    process_names: List[str]                    # Ordered process names

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'lambda_grad': self.lambda_grad,
            'per_stage': self.per_stage,
            'per_stage_grad_sq': self.per_stage_grad_sq,
            'per_stage_sigma_sq': self.per_stage_sigma_sq,
            'n_trajectories': self.n_trajectories,
            'process_names': self.process_names,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Λ_grad(D) = {self.lambda_grad:.6e}  (N = {self.n_trajectories})",
            "",
            f"{'Stage':<14} {'Contribution':>14} {'(∂F̂/∂o)²':>14} {'σ²_ψt':>14}",
            "-" * 58,
        ]
        for name in self.process_names:
            lines.append(
                f"{name:<14} {self.per_stage[name]:>14.6e} "
                f"{self.per_stage_grad_sq[name]:>14.6e} "
                f"{self.per_stage_sigma_sq[name]:>14.6e}"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_lambda_grad(
    trajectories: List[Dict[str, Dict[str, torch.Tensor]]],
    surrogate,
    predictors: Optional[Dict[str, Any]] = None,
    process_chain=None,
    device: str = 'cpu',
    verbose: bool = False,
) -> LambdaGradResult:
    """
    Compute Λ_grad(D) over a dataset of observed trajectories.

    Args:
        trajectories: List of N trajectory dicts. Each trajectory is::

            {
                'laser': {
                    'inputs': tensor (batch, input_dim),      # a_t + c_day
                    'outputs_mean': tensor (batch, output_dim),
                    'outputs_var': tensor (batch, output_dim), # σ²_ψt
                    'outputs_sampled': tensor (batch, output_dim),  # o_t
                },
                'plasma': {...}, ...
            }

        surrogate: Frozen surrogate model with
            ``compute_reliability(trajectory) -> F̂`` (scalar, differentiable).
        predictors: Optional dict {process_name: predictor_module}.
            If provided together with *process_chain*, σ²_ψt is recomputed
            from the raw inputs via a fresh forward pass through each g_ψt.
            Otherwise σ²_ψt is read from ``outputs_var`` in the trajectory.
        process_chain: Optional ProcessChain (needed for input scaling when
            *predictors* are supplied).
        device: Torch device.

    Returns:
        LambdaGradResult with Λ_grad and per-stage diagnostics.
    """
    if not trajectories:
        raise ValueError("Empty trajectory list")

    # Infer process ordering from first trajectory
    process_names = list(trajectories[0].keys())

    N = len(trajectories)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Λ_grad DEBUG — compute_lambda_grad")
        print(f"{'='*70}")
        print(f"  N trajectories:   {N}")
        print(f"  Processes:        {process_names}")
        print(f"  σ² source:        {'predictors (recomputed)' if predictors is not None and process_chain is not None else 'outputs_var (pre-computed)'}")
        print(f"  Device:           {device}")
        # Inspect first trajectory shapes
        traj0 = trajectories[0]
        for name in process_names:
            data = traj0[name]
            o_src = data.get('outputs_sampled', data['outputs_mean'])
            print(f"  {name}:")
            print(f"    inputs shape:          {data['inputs'].shape}")
            print(f"    outputs_mean shape:    {data['outputs_mean'].shape}")
            print(f"    outputs_var shape:     {data['outputs_var'].shape}")
            print(f"    o_t source shape:      {o_src.shape}  "
                  f"(key={'outputs_sampled' if 'outputs_sampled' in data else 'outputs_mean'})")
            print(f"    outputs_var stats:     "
                  f"mean={data['outputs_var'].mean().item():.6e}, "
                  f"min={data['outputs_var'].min().item():.6e}, "
                  f"max={data['outputs_var'].max().item():.6e}")
        print(f"{'─'*70}")

    # Accumulators ── per-stage
    accum_contribution = {name: 0.0 for name in process_names}
    accum_grad_sq = {name: 0.0 for name in process_names}
    accum_sigma_sq = {name: 0.0 for name in process_names}
    accum_total = 0.0

    for i, traj in enumerate(trajectories):
        contrib, grad_sq, sigma_sq = _process_single_trajectory(
            traj, surrogate, predictors, process_chain,
            process_names, device, verbose=verbose, traj_idx=i,
        )
        for name in process_names:
            accum_contribution[name] += contrib[name]
            accum_grad_sq[name] += grad_sq[name]
            accum_sigma_sq[name] += sigma_sq[name]
            accum_total += contrib[name]

    # Average over N
    per_stage = {n: accum_contribution[n] / N for n in process_names}
    per_stage_grad_sq = {n: accum_grad_sq[n] / N for n in process_names}
    per_stage_sigma_sq = {n: accum_sigma_sq[n] / N for n in process_names}
    lambda_grad = accum_total / N

    if verbose:
        print(f"\n{'─'*70}")
        print(f"  Λ_grad DEBUG — FINAL RESULTS")
        print(f"{'─'*70}")
        print(f"  Λ_grad(D) = {lambda_grad:.6e}")
        print(f"")
        print(f"  {'Stage':<14} {'Contribution':>14} {'(∂F̂/∂o)²':>14} {'σ²_ψt':>14} {'% total':>10}")
        print(f"  {'─'*14} {'─'*14} {'─'*14} {'─'*14} {'─'*10}")
        for name in process_names:
            pct = 100.0 * per_stage[name] / lambda_grad if lambda_grad > 0 else 0.0
            print(f"  {name:<14} {per_stage[name]:>14.6e} "
                  f"{per_stage_grad_sq[name]:>14.6e} "
                  f"{per_stage_sigma_sq[name]:>14.6e} {pct:>9.1f}%")
        print(f"  {'─'*14} {'─'*14}")
        print(f"  {'TOTAL':<14} {lambda_grad:>14.6e}")
        print(f"{'='*70}\n")

    return LambdaGradResult(
        lambda_grad=lambda_grad,
        per_stage=per_stage,
        per_stage_grad_sq=per_stage_grad_sq,
        per_stage_sigma_sq=per_stage_sigma_sq,
        n_trajectories=N,
        process_names=process_names,
    )


def compute_lambda_grad_batched(
    trajectories: List[Dict[str, Dict[str, torch.Tensor]]],
    surrogate,
    predictors: Optional[Dict[str, Any]] = None,
    process_chain=None,
    device: str = 'cpu',
    batch_size: int = 64,
    verbose: bool = False,
) -> LambdaGradResult:
    """
    Batched version of :func:`compute_lambda_grad`.

    Stacks multiple trajectories into a single forward+backward pass for
    efficiency on GPU.  Falls back to the loop version when trajectories
    have heterogeneous shapes.

    Args:
        trajectories: Same as :func:`compute_lambda_grad`.
        surrogate: Frozen surrogate.
        predictors: Optional predictors dict.
        process_chain: Optional ProcessChain.
        device: Torch device.
        batch_size: Number of trajectories per mini-batch.

    Returns:
        LambdaGradResult
    """
    if not trajectories:
        raise ValueError("Empty trajectory list")

    process_names = list(trajectories[0].keys())
    N = len(trajectories)
    n_batches = (N + batch_size - 1) // batch_size

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Λ_grad DEBUG — compute_lambda_grad_batched")
        print(f"{'='*70}")
        print(f"  N trajectories: {N}")
        print(f"  Batch size:     {batch_size}")
        print(f"  N batches:      {n_batches}")
        print(f"  Processes:      {process_names}")
        print(f"  σ² source:      {'predictors (recomputed)' if predictors is not None and process_chain is not None else 'outputs_var (pre-computed)'}")
        print(f"  Device:         {device}")
        print(f"{'─'*70}")

    accum_contribution = {name: 0.0 for name in process_names}
    accum_grad_sq = {name: 0.0 for name in process_names}
    accum_sigma_sq = {name: 0.0 for name in process_names}
    accum_total = 0.0
    total_samples = 0   # count actual samples, not trajectory dicts

    for batch_idx, start in enumerate(range(0, N, batch_size)):
        end = min(start + batch_size, N)
        batch_trajs = trajectories[start:end]
        B = len(batch_trajs)

        if verbose:
            print(f"\n  ── Batch {batch_idx} (trajectories {start}..{end-1}, B={B}) {'─'*30}")

        # Stack trajectories into a single batched trajectory
        batched_traj, sigma_sq_tensors = _stack_trajectories(
            batch_trajs, process_names, device,
        )

        # Determine actual sample count from stacked tensor shape
        ref_name = process_names[0]
        batch_samples = batched_traj[ref_name]['outputs_mean'].shape[0]
        total_samples += batch_samples

        # Retrieve or recompute σ²
        if predictors is not None and process_chain is not None:
            sigma_sq_tensors = _recompute_sigma_sq(
                batched_traj, predictors, process_chain, process_names, device,
            )

        if verbose:
            print(f"  [Step 1] Stacked σ²_ψt (batch_samples={batch_samples}):")
            for name in process_names:
                s2 = sigma_sq_tensors[name]
                print(f"    {name}: shape={list(s2.shape)}, "
                      f"mean={s2.mean().item():.6e}, "
                      f"min={s2.min().item():.6e}, max={s2.max().item():.6e}")

        # Build trajectory with grad-enabled outputs for surrogate
        grad_traj, output_tensors = _prepare_grad_trajectory(
            batched_traj, process_names, device,
        )

        if verbose:
            print(f"  [Step 2] o_t (grad-enabled):")
            for name in process_names:
                o_t = output_tensors[name]
                print(f"    {name}: shape={list(o_t.shape)}, "
                      f"mean={o_t.mean().item():.6f}, std={o_t.std().item():.6f}")

        # Forward through surrogate → F̂ per sample
        F_hat = surrogate.compute_reliability(grad_traj)  # (B_samples,) or scalar
        if F_hat.dim() == 0:
            F_hat = F_hat.unsqueeze(0)

        if verbose:
            print(f"  [Step 3] F̂: shape={list(F_hat.shape)}, "
                  f"mean={F_hat.mean().item():.6f}, "
                  f"std={F_hat.std().item():.6f}, "
                  f"min={F_hat.min().item():.6f}, max={F_hat.max().item():.6f}")

        # Backward: ∂F̂/∂o_t for all stages simultaneously
        # .sum() so gradients are NOT scaled by 1/B — each sample gets its true ∂F̂^(k)/∂o_t^(k)
        F_hat.sum().backward()

        if verbose:
            print(f"  [Step 4] ∂F̂/∂o_t gradients (via .sum().backward()):")
            for name in process_names:
                grad = output_tensors[name].grad
                if grad is None:
                    print(f"    {name}: grad=None!")
                else:
                    print(f"    {name}: mean={grad.mean().item():.6e}, "
                          f"std={grad.std().item():.6e}, "
                          f"norm={grad.norm().item():.6e}")

        batch_total = 0.0
        for name in process_names:
            o_t = output_tensors[name]              # (B_samples, output_dim)
            grad = o_t.grad                         # (B_samples, output_dim)
            s2 = sigma_sq_tensors[name]             # (B_samples, output_dim)

            if grad is None:
                if verbose:
                    print(f"    {name}: grad=None → skipped")
                continue

            # Per-sample contributions: sum over output_dim, keep batch dim.
            # All current processes have output_dim=1, so this is trivial.
            # For multi-output stages this assumes independent noise dims
            # (no cross-covariance terms).
            contribs = ((grad ** 2) * s2).sum(dim=-1)  # (B_samples,)
            grad_sq_vals = (grad ** 2).sum(dim=-1)     # (B_samples,)
            s2_vals = s2.sum(dim=-1)                   # (B_samples,)

            # Sum over samples (will divide by total_samples at the end)
            accum_contribution[name] += contribs.sum().item()
            accum_grad_sq[name] += grad_sq_vals.sum().item()
            accum_sigma_sq[name] += s2_vals.sum().item()
            accum_total += contribs.sum().item()
            batch_total += contribs.sum().item()

        if verbose:
            print(f"  [Step 5] Batch {batch_idx} total (sum over {batch_samples} samples): "
                  f"{batch_total:.6e}")

    # Average over total number of samples (not trajectory dicts)
    per_stage = {n: accum_contribution[n] / total_samples for n in process_names}
    per_stage_grad_sq = {n: accum_grad_sq[n] / total_samples for n in process_names}
    per_stage_sigma_sq = {n: accum_sigma_sq[n] / total_samples for n in process_names}
    lambda_grad = accum_total / total_samples

    if verbose:
        print(f"\n{'─'*70}")
        print(f"  Λ_grad DEBUG — FINAL RESULTS (batched)")
        print(f"{'─'*70}")
        print(f"  Λ_grad(D) = {lambda_grad:.6e}  (total_samples={total_samples})")
        print(f"")
        print(f"  {'Stage':<14} {'Contribution':>14} {'(∂F̂/∂o)²':>14} {'σ²_ψt':>14} {'% total':>10}")
        print(f"  {'─'*14} {'─'*14} {'─'*14} {'─'*14} {'─'*10}")
        for name in process_names:
            pct = 100.0 * per_stage[name] / lambda_grad if lambda_grad > 0 else 0.0
            print(f"  {name:<14} {per_stage[name]:>14.6e} "
                  f"{per_stage_grad_sq[name]:>14.6e} "
                  f"{per_stage_sigma_sq[name]:>14.6e} {pct:>9.1f}%")
        print(f"  {'─'*14} {'─'*14}")
        print(f"  {'TOTAL':<14} {lambda_grad:>14.6e}")
        print(f"{'='*70}\n")

    return LambdaGradResult(
        lambda_grad=lambda_grad,
        per_stage=per_stage,
        per_stage_grad_sq=per_stage_grad_sq,
        per_stage_sigma_sq=per_stage_sigma_sq,
        n_trajectories=N,
        process_names=process_names,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _process_single_trajectory(
    traj: Dict[str, Dict[str, torch.Tensor]],
    surrogate,
    predictors: Optional[Dict[str, Any]],
    process_chain,
    process_names: List[str],
    device: str,
    verbose: bool = False,
    traj_idx: int = 0,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Compute per-stage contributions for a single trajectory.

    Returns:
        (contribution, grad_sq, sigma_sq) — each a dict {stage_name: float}.
    """
    contribution: Dict[str, float] = {}
    grad_sq_out: Dict[str, float] = {}
    sigma_sq_out: Dict[str, float] = {}

    if verbose:
        print(f"\n  ── Trajectory {traj_idx} {'─'*50}")

    # ── Step 1: Obtain σ²_ψt per stage ──────────────────────────────────
    sigma_sq_per_stage: Dict[str, torch.Tensor] = {}
    for name in process_names:
        if predictors is not None and process_chain is not None:
            # Re-run predictor on raw inputs → fresh σ²
            sigma_sq_per_stage[name] = _get_sigma_sq_from_predictor(
                traj[name], name, predictors, process_chain, device,
            )
        else:
            # Use pre-computed outputs_var
            sigma_sq_per_stage[name] = traj[name]['outputs_var'].detach().to(device)

    if verbose:
        print(f"  [Step 1] σ²_ψt per stage:")
        for name in process_names:
            s2 = sigma_sq_per_stage[name]
            print(f"    {name}: shape={list(s2.shape)}, "
                  f"mean={s2.mean().item():.6e}, "
                  f"min={s2.min().item():.6e}, max={s2.max().item():.6e}")

    # ── Step 2: Build trajectory with grad-enabled o_t ───────────────────
    output_tensors: Dict[str, torch.Tensor] = {}
    grad_traj: Dict[str, Dict[str, torch.Tensor]] = {}

    for name in process_names:
        data = traj[name]
        # o_t: use sampled outputs if available, else mean
        o_t_source = data.get('outputs_sampled', data['outputs_mean'])
        o_t = o_t_source.detach().clone().to(device).requires_grad_(True)
        output_tensors[name] = o_t

        grad_traj[name] = {
            'inputs': data['inputs'].detach().to(device),
            'outputs_mean': o_t,            # surrogate reads this
            'outputs_var': data['outputs_var'].detach().to(device),
            'outputs_sampled': o_t,          # or this
        }

    if verbose:
        print(f"  [Step 2] o_t (grad-enabled) per stage:")
        for name in process_names:
            o_t = output_tensors[name]
            print(f"    {name}: shape={list(o_t.shape)}, "
                  f"mean={o_t.mean().item():.6f}, "
                  f"std={o_t.std().item():.6f}, "
                  f"requires_grad={o_t.requires_grad}")

    # ── Step 3: Forward through surrogate → F̂ ───────────────────────────
    F_hat = surrogate.compute_reliability(grad_traj)
    F_hat_raw = F_hat.clone().detach()

    # Determine batch size for averaging
    # F_hat may be scalar (B=1) or vector (B samples)
    if F_hat.dim() > 0:
        batch_size = F_hat.shape[0]
    else:
        batch_size = 1

    if verbose:
        if F_hat_raw.dim() > 0:
            print(f"  [Step 3] F̂ = surrogate(τ): shape={list(F_hat_raw.shape)}, "
                  f"batch_size={batch_size}, "
                  f"mean={F_hat_raw.mean().item():.6f}, "
                  f"std={F_hat_raw.std().item():.6f}, "
                  f"min={F_hat_raw.min().item():.6f}, max={F_hat_raw.max().item():.6f}")
        else:
            print(f"  [Step 3] F̂ = surrogate(τ): {F_hat.item():.6f}  (scalar, batch_size=1)")

    # ── Step 4: Backward → ∂F̂/∂o_t for all t ────────────────────────────
    # Use .sum() NOT .mean() so that gradients are NOT scaled by 1/B.
    # With .sum(): ∂(Σ_k F̂^(k))/∂o_t^(k) = ∂F̂^(k)/∂o_t^(k) (correct per-sample grad)
    # With .mean(): gradients would be 1/B too small → (∂F̂/∂o)² is 1/B² too small.
    F_scalar = F_hat.sum() if F_hat.dim() > 0 else F_hat
    F_scalar.backward()

    if verbose:
        print(f"  [Step 4] ∂F̂/∂o_t (via .sum().backward() — unscaled per-sample grads):")
        for name in process_names:
            grad = output_tensors[name].grad
            if grad is None:
                print(f"    {name}: grad is None! (surrogate may not depend on this output)")
            else:
                print(f"    {name}: shape={list(grad.shape)}, "
                      f"mean={grad.mean().item():.6e}, "
                      f"std={grad.std().item():.6e}, "
                      f"min={grad.min().item():.6e}, max={grad.max().item():.6e}, "
                      f"norm={grad.norm().item():.6e}")

    # ── Step 5: Accumulate per-stage ─────────────────────────────────────
    # grad has shape (B, output_dim) with correct per-sample values.
    # Compute per-sample contributions, then AVERAGE over B.
    #
    # NOTE on .sum(dim=-1): currently all processes have output_dim=1,
    # so this sum is trivially correct (collapses [B,1] → [B]).
    # If a future process has output_dim>1 (multi-output), .sum(dim=-1)
    # assumes the noise dimensions are independent — i.e. the cross-terms
    # (∂F̂/∂o_t^d1)(∂F̂/∂o_t^d2)·Cov(d1,d2) are zero.  For correlated
    # multi-output stages, this would need a full covariance treatment.
    for name in process_names:
        grad = output_tensors[name].grad       # (B, output_dim) or (output_dim,)
        s2 = sigma_sq_per_stage[name]

        if grad is None:
            # Surrogate does not depend on this stage's output
            contribution[name] = 0.0
            grad_sq_out[name] = 0.0
            sigma_sq_out[name] = s2.mean().item() if s2.dim() > 0 else s2.item()
            if verbose:
                print(f"    {name}: grad=None → contribution=0")
            continue

        # Warn if multi-output (not currently expected)
        out_dim = grad.shape[-1] if grad.dim() > 1 else 1
        if out_dim > 1 and verbose:
            print(f"    ⚠ {name}: output_dim={out_dim} > 1 — "
                  f".sum(dim=-1) assumes independent noise dimensions")

        # Per-sample: sum over output_dim, then average over batch
        # (grad**2 * s2) shape: (B, output_dim) → sum dim=-1 → (B,) → mean → scalar
        if grad.dim() > 1:
            per_sample_contrib = ((grad ** 2) * s2).sum(dim=-1)   # (B,)
            per_sample_g2 = (grad ** 2).sum(dim=-1)               # (B,)
            per_sample_s2 = s2.sum(dim=-1)                        # (B,)
            contrib = per_sample_contrib.mean().item()
            g2 = per_sample_g2.mean().item()
            s2_val = per_sample_s2.mean().item()
        else:
            # Single sample, no batch dim
            contrib = ((grad ** 2) * s2).sum().item()
            g2 = (grad ** 2).sum().item()
            s2_val = s2.sum().item()

        contribution[name] = contrib
        grad_sq_out[name] = g2
        sigma_sq_out[name] = s2_val

    if verbose:
        traj_total = sum(contribution.values())
        print(f"  [Step 5] Per-stage contributions (batch-averaged, B={batch_size}):")
        print(f"    {'Stage':<14} {'(∂F̂/∂o)²·σ²':>14} {'(∂F̂/∂o)²':>14} {'σ²_ψt':>14}")
        print(f"    {'─'*14} {'─'*14} {'─'*14} {'─'*14}")
        for name in process_names:
            print(f"    {name:<14} {contribution[name]:>14.6e} "
                  f"{grad_sq_out[name]:>14.6e} {sigma_sq_out[name]:>14.6e}")
        print(f"    {'─'*14} {'─'*14}")
        print(f"    {'Σ_t':<14} {traj_total:>14.6e}")

    return contribution, grad_sq_out, sigma_sq_out


def _get_sigma_sq_from_predictor(
    stage_data: Dict[str, torch.Tensor],
    process_name: str,
    predictors: Dict[str, Any],
    process_chain,
    device: str,
) -> torch.Tensor:
    """Run a frozen predictor on raw inputs to obtain σ²_ψt.

    Uses process_chain's scaling utilities so the predictor receives
    correctly normalised inputs and the returned variance is unscaled.
    """
    predictor = predictors[process_name]
    predictor.eval()

    inputs = stage_data['inputs'].detach().to(device)

    # Find process index in chain
    proc_idx = process_chain.process_names.index(process_name)

    with torch.no_grad():
        scaled_inputs = process_chain.scale_inputs(inputs, proc_idx)
        _, var_scaled = predictor(scaled_inputs)
        var_unscaled = process_chain.unscale_variance(var_scaled, proc_idx)

    return var_unscaled


def _stack_trajectories(
    batch_trajs: List[Dict[str, Dict[str, torch.Tensor]]],
    process_names: List[str],
    device: str,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """Stack a list of single-sample trajectories into one batched trajectory.

    Returns:
        (batched_traj, sigma_sq_tensors)
    """
    batched: Dict[str, Dict[str, torch.Tensor]] = {}
    sigma_sq: Dict[str, torch.Tensor] = {}

    for name in process_names:
        keys = ['inputs', 'outputs_mean', 'outputs_var', 'outputs_sampled']
        stacked: Dict[str, torch.Tensor] = {}
        for k in keys:
            tensors = []
            for traj in batch_trajs:
                t = traj[name][k] if k in traj[name] else traj[name]['outputs_mean']
                if t.dim() == 1:
                    t = t.unsqueeze(0)
                tensors.append(t.to(device))
            stacked[k] = torch.cat(tensors, dim=0)
        batched[name] = stacked
        sigma_sq[name] = stacked['outputs_var'].detach()

    return batched, sigma_sq


def _recompute_sigma_sq(
    batched_traj: Dict[str, Dict[str, torch.Tensor]],
    predictors: Dict[str, Any],
    process_chain,
    process_names: List[str],
    device: str,
) -> Dict[str, torch.Tensor]:
    """Recompute σ² from predictors for a batched trajectory."""
    sigma_sq: Dict[str, torch.Tensor] = {}
    for name in process_names:
        sigma_sq[name] = _get_sigma_sq_from_predictor(
            batched_traj[name], name, predictors, process_chain, device,
        )
    return sigma_sq


def _prepare_grad_trajectory(
    batched_traj: Dict[str, Dict[str, torch.Tensor]],
    process_names: List[str],
    device: str,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """Create a trajectory copy where outputs require grad.

    Returns:
        (grad_traj, output_tensors) — output_tensors maps name → leaf tensor
        with requires_grad=True so that .grad is populated after backward.
    """
    grad_traj: Dict[str, Dict[str, torch.Tensor]] = {}
    output_tensors: Dict[str, torch.Tensor] = {}

    for name in process_names:
        data = batched_traj[name]
        o_t_source = data.get('outputs_sampled', data['outputs_mean'])
        o_t = o_t_source.detach().clone().to(device).requires_grad_(True)
        output_tensors[name] = o_t

        grad_traj[name] = {
            'inputs': data['inputs'].detach().to(device),
            'outputs_mean': o_t,
            'outputs_var': data['outputs_var'].detach().to(device),
            'outputs_sampled': o_t,
        }

    return grad_traj, output_tensors


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def run_lambda_grad_diagnostics(
    trajectories: List[Dict[str, Dict[str, torch.Tensor]]],
    surrogate,
    device: str = 'cpu',
    n_sample: int = 5,
    epsilon: float = 0.01,
) -> None:
    """
    Run 4 diagnostic checks on the data feeding Λ_grad and print results.

    Check 1 — σ² scale:
        Print raw outputs_var for *n_sample* trajectories.
        Compute std(outputs_mean) / sqrt(mean(outputs_var)) per stage.
        Should be ≈ 1 if the predictor is well calibrated.

    Check 2 — Predictor calibration ratio κ:
        κ = mean((o_t − μ_t)²) / mean(σ²_t).
        κ ≈ 1 ideal, κ >> 1 overconfident, κ << 1 underconfident.

    Check 3 — Is outputs_var variance or std?
        Compare sqrt(outputs_var) to outputs_mean scale
        and infer whether the stored value is σ² or σ.

    Check 4 — Gradient finite-difference check:
        Perturb o_t by ±ε for the last stage and compare
        (F̂(o+ε) − F̂(o−ε)) / (2ε)  vs  ∂F̂/∂o_t from backward.
    """
    if not trajectories:
        print("  No trajectories — skipping diagnostics.")
        return

    process_names = list(trajectories[0].keys())
    N = len(trajectories)
    n_show = min(n_sample, N)

    print(f"\n{'='*75}")
    print(f"  Λ_grad DIAGNOSTICS  (N={N} trajectories, {len(process_names)} stages)")
    print(f"{'='*75}")

    # ── Check 1 — σ² scale ──────────────────────────────────────────────
    print(f"\n  ┌─ CHECK 1: σ² scale ────────────────────────────────────────")

    # Print raw values for first n_show trajectories
    for i in range(n_show):
        traj = trajectories[i]
        print(f"  │ Trajectory {i}:")
        for name in process_names:
            var = traj[name]['outputs_var']
            mean = traj[name]['outputs_mean']
            # Show first few values per sample
            n_vals = min(3, var.shape[0]) if var.dim() > 0 else 1
            var_flat = var.reshape(-1)[:n_vals]
            mean_flat = mean.reshape(-1)[:n_vals]
            print(f"  │   {name}: outputs_var={[f'{v:.6e}' for v in var_flat.tolist()]}, "
                  f"outputs_mean={[f'{v:.6f}' for v in mean_flat.tolist()]}")

    # Aggregate ratio: std(outputs_mean) / sqrt(mean(outputs_var))
    print(f"  │")
    print(f"  │ Scale ratio  std(o_mean) / sqrt(mean(o_var))  per stage:")
    for name in process_names:
        all_means = []
        all_vars = []
        for traj in trajectories:
            all_means.append(traj[name]['outputs_mean'].detach().reshape(-1))
            all_vars.append(traj[name]['outputs_var'].detach().reshape(-1))
        all_means_t = torch.cat(all_means)
        all_vars_t = torch.cat(all_vars)
        std_of_mean = all_means_t.std().item()
        sqrt_mean_var = (all_vars_t.mean()).sqrt().item()
        ratio = std_of_mean / sqrt_mean_var if sqrt_mean_var > 0 else float('inf')
        print(f"  │   {name}: std(μ)={std_of_mean:.6e}, sqrt(mean(σ²))={sqrt_mean_var:.6e}, "
              f"ratio={ratio:.4f}  {'✓' if 0.2 < ratio < 5.0 else '⚠ OUT OF RANGE'}")
    print(f"  └──────────────────────────────────────────────────────────")

    # ── Check 2 — Calibration κ ─────────────────────────────────────────
    print(f"\n  ┌─ CHECK 2: Predictor calibration κ ──────────────────────────")
    print(f"  │ κ = mean((o_t − μ_t)²) / mean(σ²_t)")
    print(f"  │ Ideal: κ ≈ 1.0  |  κ >> 1: overconfident  |  κ << 1: underconfident")
    print(f"  │")
    for name in process_names:
        all_residuals_sq = []
        all_sigma_sq = []
        for traj in trajectories:
            data = traj[name]
            o_t = data.get('outputs_sampled', data['outputs_mean']).detach().reshape(-1)
            mu_t = data['outputs_mean'].detach().reshape(-1)
            s2_t = data['outputs_var'].detach().reshape(-1)
            all_residuals_sq.append((o_t - mu_t) ** 2)
            all_sigma_sq.append(s2_t)
        residuals_sq = torch.cat(all_residuals_sq)
        sigma_sq = torch.cat(all_sigma_sq)
        mean_res_sq = residuals_sq.mean().item()
        mean_sigma_sq = sigma_sq.mean().item()
        kappa = mean_res_sq / mean_sigma_sq if mean_sigma_sq > 0 else float('inf')
        if kappa < 0.1:
            status = "⚠ UNDERCONFIDENT (σ² too large)"
        elif kappa > 10.0:
            status = "⚠ OVERCONFIDENT (σ² too small)"
        elif 0.5 < kappa < 2.0:
            status = "✓ well calibrated"
        else:
            status = "~ marginal"
        print(f"  │   {name}: mean((o−μ)²)={mean_res_sq:.6e}, mean(σ²)={mean_sigma_sq:.6e}, "
              f"κ={kappa:.4f}  {status}")
    print(f"  └──────────────────────────────────────────────────────────")

    # ── Check 3 — σ² vs σ ───────────────────────────────────────────────
    print(f"\n  ┌─ CHECK 3: Is outputs_var storing σ² or σ? ────────────────")
    print(f"  │ Compare sqrt(outputs_var) and |outputs_mean| scale")
    print(f"  │")
    for name in process_names:
        all_vars = []
        all_means = []
        for traj in trajectories:
            all_vars.append(traj[name]['outputs_var'].detach().reshape(-1))
            all_means.append(traj[name]['outputs_mean'].detach().reshape(-1))
        vars_t = torch.cat(all_vars)
        means_t = torch.cat(all_means)

        raw_var_mean = vars_t.mean().item()
        sqrt_var_mean = vars_t.mean().sqrt().item()
        abs_mean = means_t.abs().mean().item()

        # If outputs_var is actually σ (not σ²), then raw value ~ scale of mean
        # If outputs_var is σ², then sqrt ~ scale of mean (or much smaller)
        coeff_var_if_variance = sqrt_var_mean / abs_mean if abs_mean > 0 else float('inf')
        coeff_var_if_std = raw_var_mean / abs_mean if abs_mean > 0 else float('inf')

        print(f"  │   {name}:")
        print(f"  │     mean(outputs_var) = {raw_var_mean:.6e}")
        print(f"  │     sqrt(mean(o_var)) = {sqrt_var_mean:.6e}")
        print(f"  │     mean(|o_mean|)    = {abs_mean:.6e}")
        print(f"  │     If σ²: CV = sqrt(σ²)/|μ| = {coeff_var_if_variance:.4f}")
        print(f"  │     If σ:  CV = σ/|μ|        = {coeff_var_if_std:.4f}")
        if coeff_var_if_variance < 2.0 and coeff_var_if_std > 2.0:
            print(f"  │     → Likely σ² (variance) ✓")
        elif coeff_var_if_std < 2.0 and coeff_var_if_variance < 0.01:
            print(f"  │     → Likely σ (std dev) ⚠  — Λ_grad may need squaring!")
        else:
            print(f"  │     → Ambiguous — inspect predictor output head")
    print(f"  └──────────────────────────────────────────────────────────")

    # ── Check 4 — Gradient finite-difference validation ─────────────────
    print(f"\n  ┌─ CHECK 4: Gradient vs finite-difference (single sample) ──")
    print(f"  │ Uses sample [0] only to avoid batch-averaging confusion.")
    print(f"  │ Perturb o_t by ±ε={epsilon}, compare ΔF̂/(2ε) vs ∂F̂/∂o_t")
    print(f"  │")

    traj = trajectories[0]  # use first trajectory

    # Extract single sample [0] from the batch for clean comparison
    def _single_sample(data_dict, idx=0):
        """Extract sample idx from a stage dict, keeping 2-D shape [1, dim]."""
        out = {}
        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor) and v.dim() > 1:
                out[k] = v[idx:idx+1].detach().to(device)
            elif isinstance(v, torch.Tensor):
                out[k] = v.unsqueeze(0).detach().to(device)
            else:
                out[k] = v
        return out

    single_traj = {name: _single_sample(traj[name]) for name in process_names}

    # --- Autograd gradient on single sample ---
    output_tensors_s: Dict[str, torch.Tensor] = {}
    grad_traj_s: Dict[str, Dict[str, torch.Tensor]] = {}
    for name in process_names:
        data = single_traj[name]
        o_t_source = data.get('outputs_sampled', data['outputs_mean'])
        o_t = o_t_source.detach().clone().requires_grad_(True)
        output_tensors_s[name] = o_t
        grad_traj_s[name] = {
            'inputs': data['inputs'],
            'outputs_mean': o_t,
            'outputs_var': data['outputs_var'],
            'outputs_sampled': o_t,
        }

    F_hat_s = surrogate.compute_reliability(grad_traj_s)
    # For a single sample F̂ may be scalar or [1] — both fine
    if F_hat_s.dim() > 0:
        F_hat_s = F_hat_s.squeeze()
    F_hat_val = F_hat_s.item()
    F_hat_s.backward()

    # --- Finite-difference and comparison for each stage ---
    for check_stage in process_names:
        ag = output_tensors_s[check_stage].grad
        if ag is None:
            print(f"  │   {check_stage}: grad=None — surrogate independent of this output")
            continue
        ag_val = ag.squeeze().item() if ag.numel() == 1 else ag.mean().item()

        # F̂(o + ε)
        traj_p: Dict[str, Dict[str, torch.Tensor]] = {}
        traj_m: Dict[str, Dict[str, torch.Tensor]] = {}
        for name in process_names:
            data = single_traj[name]
            o_src = data.get('outputs_sampled', data['outputs_mean']).detach().clone()
            o_p = o_src + epsilon if name == check_stage else o_src.clone()
            o_m = o_src - epsilon if name == check_stage else o_src.clone()
            traj_p[name] = {
                'inputs': data['inputs'], 'outputs_mean': o_p,
                'outputs_var': data['outputs_var'], 'outputs_sampled': o_p,
            }
            traj_m[name] = {
                'inputs': data['inputs'], 'outputs_mean': o_m,
                'outputs_var': data['outputs_var'], 'outputs_sampled': o_m,
            }

        with torch.no_grad():
            Fp = surrogate.compute_reliability(traj_p)
            Fm = surrogate.compute_reliability(traj_m)
            if Fp.dim() > 0: Fp = Fp.squeeze()
            if Fm.dim() > 0: Fm = Fm.squeeze()

        fd_val = (Fp.item() - Fm.item()) / (2 * epsilon)
        abs_err = abs(fd_val - ag_val)
        rel_err = abs_err / (abs(ag_val) + 1e-12) * 100

        if check_stage == process_names[-1]:
            # Detailed output for last stage
            print(f"  │   Stage: {check_stage}")
            print(f"  │   F̂(o)        = {F_hat_val:.8f}")
            print(f"  │   F̂(o+ε)      = {Fp.item():.8f}")
            print(f"  │   F̂(o−ε)      = {Fm.item():.8f}")
            print(f"  │   FD gradient  = (F̂⁺ − F̂⁻)/(2ε) = {fd_val:.8e}")
            print(f"  │   Autograd     = ∂F̂/∂o_t         = {ag_val:.8e}")
            print(f"  │   Abs error    = {abs_err:.8e}")
            print(f"  │   Rel error    = {rel_err:.2f}%")
            if rel_err < 5.0:
                print(f"  │   → ✓ Gradient is correct")
            elif rel_err < 20.0:
                print(f"  │   → ~ Acceptable (non-linearity at this ε)")
            else:
                print(f"  │   → ⚠ MISMATCH — check surrogate differentiability")
        else:
            # Compact output for other stages
            status = "✓" if rel_err < 5.0 else ("~" if rel_err < 20.0 else "⚠")
            print(f"  │   {check_stage}: FD={fd_val:.6e}, autograd={ag_val:.6e}, "
                  f"rel_err={rel_err:.1f}% {status}")

    print(f"  └──────────────────────────────────────────────────────────")
    print(f"{'='*75}\n")
