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
from dataclasses import dataclass
from typing import Dict, List, Any


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
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_sigma_sq_from_predictor(
    stage_data: Dict[str, torch.Tensor],
    process_name: str,
    predictors: Dict[str, Any],
    process_chain,
    device: str,
) -> torch.Tensor:
    """Run a frozen predictor on raw inputs to obtain σ²_ψt."""
    predictor = predictors[process_name]
    predictor.eval()

    inputs = stage_data['inputs'].detach().to(device)
    proc_idx = process_chain.process_names.index(process_name)

    with torch.no_grad():
        scaled_inputs = process_chain.scale_inputs(inputs, proc_idx)
        _, var_scaled = predictor(scaled_inputs)
        var_unscaled = process_chain.unscale_variance(var_scaled, proc_idx)

    return var_unscaled


def _require_outputs_sampled(data: Dict[str, torch.Tensor], name: str) -> torch.Tensor:
    """Return outputs_sampled or raise ValueError."""
    if 'outputs_sampled' not in data:
        raise ValueError(
            f"Stage '{name}': 'outputs_sampled' is missing. "
            f"This key is mandatory — no fallback to outputs_mean is allowed."
        )
    return data['outputs_sampled']


# ─────────────────────────────────────────────────────────────────────────────
# Core computation (loop version)
# ─────────────────────────────────────────────────────────────────────────────

def compute_lambda_grad(
    trajectories: List[Dict[str, Dict[str, torch.Tensor]]],
    surrogate,
    predictors: Dict[str, Any],
    process_chain,
    device: str = 'cpu',
    verbose: bool = False,
) -> LambdaGradResult:
    """
    Compute Λ_grad(D) over a dataset of observed trajectories.

    Args:
        trajectories: List of N trajectory dicts, each mapping
            process_name -> {'inputs', 'outputs_sampled', ...}.
        surrogate: Frozen surrogate with compute_reliability(traj) -> Tensor(B,).
        predictors: Dict {process_name: predictor_module} — mandatory.
        process_chain: ProcessChain exposing scale_inputs/unscale_variance — mandatory.
        device: Torch device.
        verbose: Print debug info.

    Returns:
        LambdaGradResult with Λ_grad and per-stage diagnostics.
    """
    if not trajectories:
        raise ValueError("Empty trajectory list")

    process_names = list(trajectories[0].keys())
    N = len(trajectories)

    if verbose:
        print(f"\n  Λ_grad — compute_lambda_grad  (N={N}, stages={process_names})")

    accum_contribution = {name: 0.0 for name in process_names}
    accum_grad_sq = {name: 0.0 for name in process_names}
    accum_sigma_sq = {name: 0.0 for name in process_names}

    for i, traj in enumerate(trajectories):
        # Step 1: σ²_ψt from predictors (fresh forward pass)
        sigma_sq_per_stage: Dict[str, torch.Tensor] = {}
        for name in process_names:
            sigma_sq_per_stage[name] = _get_sigma_sq_from_predictor(
                traj[name], name, predictors, process_chain, device,
            )

        # Step 2: Build grad-enabled trajectory using outputs_sampled
        output_tensors: Dict[str, torch.Tensor] = {}
        grad_traj: Dict[str, Dict[str, torch.Tensor]] = {}
        for name in process_names:
            data = traj[name]
            o_t = _require_outputs_sampled(data, name).detach().clone().to(device).requires_grad_(True)
            output_tensors[name] = o_t
            grad_traj[name] = {
                'inputs': data['inputs'].detach().to(device),
                'outputs_mean': o_t,
                'outputs_var': data['outputs_var'].detach().to(device),
                'outputs_sampled': o_t,
            }

        # Step 3: Forward through surrogate
        F_hat = surrogate.compute_reliability(grad_traj)  # (B,)
        if F_hat.dim() == 0:
            F_hat = F_hat.unsqueeze(0)
        B = F_hat.shape[0]

        # Step 4: Backward with .sum() — unscaled per-sample gradients
        F_hat.sum().backward()

        # Step 5: Accumulate per-stage (average over B within this trajectory)
        for name in process_names:
            grad = output_tensors[name].grad
            s2 = sigma_sq_per_stage[name]

            if grad is None:
                continue

            per_sample_contrib = ((grad ** 2) * s2).sum(dim=-1)  # (B,)
            per_sample_g2 = (grad ** 2).sum(dim=-1)
            per_sample_s2 = s2.sum(dim=-1)

            accum_contribution[name] += per_sample_contrib.mean().item()
            accum_grad_sq[name] += per_sample_g2.mean().item()
            accum_sigma_sq[name] += per_sample_s2.mean().item()

    # Average over N trajectories
    per_stage = {n: accum_contribution[n] / N for n in process_names}
    per_stage_grad_sq = {n: accum_grad_sq[n] / N for n in process_names}
    per_stage_sigma_sq = {n: accum_sigma_sq[n] / N for n in process_names}
    lambda_grad = sum(per_stage.values())

    if verbose:
        print(f"  Λ_grad(D) = {lambda_grad:.6e}")
        for name in process_names:
            print(f"    {name}: contrib={per_stage[name]:.6e}, "
                  f"(∂F̂/∂o)²={per_stage_grad_sq[name]:.6e}, "
                  f"σ²={per_stage_sigma_sq[name]:.6e}")

    return LambdaGradResult(
        lambda_grad=lambda_grad,
        per_stage=per_stage,
        per_stage_grad_sq=per_stage_grad_sq,
        per_stage_sigma_sq=per_stage_sigma_sq,
        n_trajectories=N,
        process_names=process_names,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Batched computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_lambda_grad_batched(
    trajectories: List[Dict[str, Dict[str, torch.Tensor]]],
    surrogate,
    predictors: Dict[str, Any],
    process_chain,
    device: str = 'cpu',
    batch_size: int = 64,
    verbose: bool = False,
) -> LambdaGradResult:
    """
    Batched version of compute_lambda_grad.

    Stacks multiple trajectories into a single forward+backward pass for
    efficiency on GPU. Implements the same formula independently.

    Args:
        trajectories: List of N trajectory dicts.
        surrogate: Frozen surrogate.
        predictors: Dict {process_name: predictor_module} — mandatory.
        process_chain: ProcessChain — mandatory.
        device: Torch device.
        batch_size: Number of trajectories per mini-batch.
        verbose: Print debug info.

    Returns:
        LambdaGradResult
    """
    if not trajectories:
        raise ValueError("Empty trajectory list")

    process_names = list(trajectories[0].keys())
    N = len(trajectories)

    if verbose:
        print(f"\n  Λ_grad — compute_lambda_grad_batched  "
              f"(N={N}, batch_size={batch_size}, stages={process_names})")

    accum_contribution = {name: 0.0 for name in process_names}
    accum_grad_sq = {name: 0.0 for name in process_names}
    accum_sigma_sq = {name: 0.0 for name in process_names}

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_trajs = trajectories[start:end]
        n_batch = len(batch_trajs)  # number of trajectories in this batch

        # Stack trajectories: cat along batch dim
        stacked: Dict[str, Dict[str, torch.Tensor]] = {}
        for name in process_names:
            tensors_inputs = []
            tensors_sampled = []
            tensors_var = []
            for traj in batch_trajs:
                data = traj[name]
                inp = data['inputs'].detach().to(device)
                o_s = _require_outputs_sampled(data, name).detach().to(device)
                o_v = data['outputs_var'].detach().to(device)
                if inp.dim() == 1:
                    inp = inp.unsqueeze(0)
                if o_s.dim() == 1:
                    o_s = o_s.unsqueeze(0)
                if o_v.dim() == 1:
                    o_v = o_v.unsqueeze(0)
                tensors_inputs.append(inp)
                tensors_sampled.append(o_s)
                tensors_var.append(o_v)
            stacked[name] = {
                'inputs': torch.cat(tensors_inputs, dim=0),
                'outputs_sampled': torch.cat(tensors_sampled, dim=0),
                'outputs_var': torch.cat(tensors_var, dim=0),
            }

        # Determine B (samples per trajectory) — assume uniform across batch
        ref_name = process_names[0]
        total_samples = stacked[ref_name]['outputs_sampled'].shape[0]
        B = total_samples // n_batch

        # σ²_ψt from predictors (fresh forward pass on stacked inputs)
        sigma_sq_tensors: Dict[str, torch.Tensor] = {}
        for name in process_names:
            sigma_sq_tensors[name] = _get_sigma_sq_from_predictor(
                stacked[name], name, predictors, process_chain, device,
            )

        # Build grad-enabled trajectory
        output_tensors: Dict[str, torch.Tensor] = {}
        grad_traj: Dict[str, Dict[str, torch.Tensor]] = {}
        for name in process_names:
            o_t = stacked[name]['outputs_sampled'].detach().clone().requires_grad_(True)
            output_tensors[name] = o_t
            grad_traj[name] = {
                'inputs': stacked[name]['inputs'],
                'outputs_mean': o_t,
                'outputs_var': stacked[name]['outputs_var'],
                'outputs_sampled': o_t,
            }

        # Forward + backward
        F_hat = surrogate.compute_reliability(grad_traj)  # (total_samples,)
        if F_hat.dim() == 0:
            F_hat = F_hat.unsqueeze(0)
        F_hat.sum().backward()

        # Accumulate: for each trajectory in the batch, compute mean over B samples
        for name in process_names:
            grad = output_tensors[name].grad
            s2 = sigma_sq_tensors[name]

            if grad is None:
                continue

            # Per-sample contributions: (total_samples, out_dim) -> (total_samples,)
            contrib_per_sample = ((grad ** 2) * s2).sum(dim=-1)
            g2_per_sample = (grad ** 2).sum(dim=-1)
            s2_per_sample = s2.sum(dim=-1)

            # Reshape to (n_batch, B), mean over B, then sum over trajectories
            contrib_per_traj = contrib_per_sample.reshape(n_batch, B).mean(dim=1)
            g2_per_traj = g2_per_sample.reshape(n_batch, B).mean(dim=1)
            s2_per_traj = s2_per_sample.reshape(n_batch, B).mean(dim=1)

            accum_contribution[name] += contrib_per_traj.sum().item()
            accum_grad_sq[name] += g2_per_traj.sum().item()
            accum_sigma_sq[name] += s2_per_traj.sum().item()

    # Average over N trajectories
    per_stage = {n: accum_contribution[n] / N for n in process_names}
    per_stage_grad_sq = {n: accum_grad_sq[n] / N for n in process_names}
    per_stage_sigma_sq = {n: accum_sigma_sq[n] / N for n in process_names}
    lambda_grad = sum(per_stage.values())

    if verbose:
        print(f"  Λ_grad(D) = {lambda_grad:.6e}")
        for name in process_names:
            print(f"    {name}: contrib={per_stage[name]:.6e}, "
                  f"(∂F̂/∂o)²={per_stage_grad_sq[name]:.6e}, "
                  f"σ²={per_stage_sigma_sq[name]:.6e}")

    return LambdaGradResult(
        lambda_grad=lambda_grad,
        per_stage=per_stage,
        per_stage_grad_sq=per_stage_grad_sq,
        per_stage_sigma_sq=per_stage_sigma_sq,
        n_trajectories=N,
        process_names=process_names,
    )
