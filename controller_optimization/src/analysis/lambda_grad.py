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

        # Retrieve or recompute σ²
        if predictors is not None and process_chain is not None:
            sigma_sq_tensors = _recompute_sigma_sq(
                batched_traj, predictors, process_chain, process_names, device,
            )

        if verbose:
            print(f"  [Step 1] Stacked σ²_ψt:")
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
        F_hat = surrogate.compute_reliability(grad_traj)  # (B,) or scalar
        if F_hat.dim() == 0:
            F_hat = F_hat.unsqueeze(0)

        if verbose:
            print(f"  [Step 3] F̂: shape={list(F_hat.shape)}, "
                  f"mean={F_hat.mean().item():.6f}, "
                  f"std={F_hat.std().item():.6f}, "
                  f"min={F_hat.min().item():.6f}, max={F_hat.max().item():.6f}")

        # Backward: ∂F̂/∂o_t for all stages simultaneously
        # Sum over batch so we get per-element grads via the chain rule
        F_hat.sum().backward()

        if verbose:
            print(f"  [Step 4] ∂F̂/∂o_t gradients:")
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
            o_t = output_tensors[name]              # (B, output_dim)
            grad = o_t.grad                         # (B, output_dim)
            s2 = sigma_sq_tensors[name]             # (B, output_dim)

            if grad is None:
                if verbose:
                    print(f"    {name}: grad=None → skipped")
                continue

            grad_sq_vals = (grad ** 2).sum(dim=-1)  # (B,)
            s2_vals = s2.sum(dim=-1)                # (B,)
            contribs = ((grad ** 2) * s2).sum(dim=-1)  # (B,)

            accum_contribution[name] += contribs.sum().item()
            accum_grad_sq[name] += grad_sq_vals.sum().item()
            accum_sigma_sq[name] += s2_vals.sum().item()
            accum_total += contribs.sum().item()
            batch_total += contribs.sum().item()

        if verbose:
            print(f"  [Step 5] Batch {batch_idx} total contribution: {batch_total:.6e}")

    per_stage = {n: accum_contribution[n] / N for n in process_names}
    per_stage_grad_sq = {n: accum_grad_sq[n] / N for n in process_names}
    per_stage_sigma_sq = {n: accum_sigma_sq[n] / N for n in process_names}
    lambda_grad = accum_total / N

    if verbose:
        print(f"\n{'─'*70}")
        print(f"  Λ_grad DEBUG — FINAL RESULTS (batched)")
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
    if F_hat.dim() > 0:
        F_hat = F_hat.mean()  # collapse to scalar if batched

    if verbose:
        if F_hat_raw.dim() > 0:
            print(f"  [Step 3] F̂ = surrogate(τ): shape={list(F_hat_raw.shape)}, "
                  f"mean={F_hat_raw.mean().item():.6f}, "
                  f"std={F_hat_raw.std().item():.6f}, "
                  f"min={F_hat_raw.min().item():.6f}, max={F_hat_raw.max().item():.6f}")
        else:
            print(f"  [Step 3] F̂ = surrogate(τ): {F_hat.item():.6f}  (scalar)")
        print(f"           F̂ (scalar for backward): {F_hat.item():.6f}")

    # ── Step 4: Backward → ∂F̂/∂o_t for all t ────────────────────────────
    F_hat.backward()

    if verbose:
        print(f"  [Step 4] ∂F̂/∂o_t (gradients after backward):")
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
    for name in process_names:
        grad = output_tensors[name].grad       # (batch, output_dim) or (output_dim,)
        s2 = sigma_sq_per_stage[name]

        if grad is None:
            # Surrogate does not depend on this stage's output
            contribution[name] = 0.0
            grad_sq_out[name] = 0.0
            sigma_sq_out[name] = s2.sum().item()
            if verbose:
                print(f"    {name}: grad=None → contribution=0")
            continue

        g2 = (grad ** 2).sum().item()
        s2_val = s2.sum().item()
        contrib = ((grad ** 2) * s2).sum().item()

        contribution[name] = contrib
        grad_sq_out[name] = g2
        sigma_sq_out[name] = s2_val

    if verbose:
        traj_total = sum(contribution.values())
        print(f"  [Step 5] Per-stage contributions (this trajectory):")
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
