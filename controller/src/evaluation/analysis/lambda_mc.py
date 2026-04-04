"""
Lambda_MC (Λ_MC) — Monte Carlo complexity metric for AZIMUTH (Method 2).

Estimates the complexity metric Λ_MC(D) by directly sampling perturbed
trajectories and measuring the empirical variance of the surrogate f_Θ
under those perturbations — NO linearity assumption required.

    Λ_MC(D) = (1/N) Σ_i  Var_{ô ~ N(μ,σ²)}[ f_Θ(τ̂^(i)) ]

Unlike Λ_grad (Method 1, Delta Method) which linearizes f_Θ around the
observed outputs via gradients, Λ_MC captures the full nonlinear response
of the surrogate to noise perturbations.

ALGORITHM (per trajectory τ^(i))
---------------------------------
1. Read μ_t, σ²_t from outputs_mean / outputs_var for each stage t
2. Draw S perturbations: ô_t^(s) = μ_t + ε·√(σ²_t + 1e-8), ε ~ N(0,I)
3. Build S perturbed trajectories (only outputs_sampled changes)
4. Compute F̂^(i,s) = surrogate.compute_reliability(τ̂^(i,s))
5. Sample variance: Λ_MC^(i) = (1/(S-1)) Σ_s (F̂^(i,s) - F̄^(i))²
6. Average over N: Λ_MC(D) = (1/N) Σ_i Λ_MC^(i)
"""

import torch
from dataclasses import dataclass
from typing import Dict, List, Any


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LambdaMCResult:
    """Results from Λ_MC computation."""

    lambda_mc: float                          # Scalar Λ_MC(D)
    per_stage: Dict[str, float]               # Per-stage contributions
    per_stage_sigma_sq: Dict[str, float]      # Per-stage mean σ²_ψt (for reference)
    n_trajectories: int                       # Number of trajectories N
    n_samples: int                            # Number of MC samples S
    process_names: List[str]                  # Ordered process names

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'lambda_mc': self.lambda_mc,
            'per_stage': self.per_stage,
            'per_stage_sigma_sq': self.per_stage_sigma_sq,
            'n_trajectories': self.n_trajectories,
            'n_samples': self.n_samples,
            'process_names': self.process_names,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Λ_MC(D) = {self.lambda_mc:.6e}  "
            f"(N = {self.n_trajectories}, S = {self.n_samples})",
            "",
            f"{'Stage':<14} {'Contribution':>14} {'σ²_ψt':>14}",
            "-" * 44,
        ]
        for name in self.process_names:
            lines.append(
                f"{name:<14} {self.per_stage[name]:>14.6e} "
                f"{self.per_stage_sigma_sq[name]:>14.6e}"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_lambda_mc(
    trajectories: List[Dict[str, Dict[str, torch.Tensor]]],
    surrogate,
    device: str = 'cpu',
    n_samples: int = 30,
    verbose: bool = False,
) -> LambdaMCResult:
    """
    Compute Λ_MC(D) over a dataset of observed trajectories (Method 2).

    Directly samples perturbed trajectories and measures the empirical
    variance of f_Θ — no linearity assumption.

    Args:
        trajectories: List of N trajectory dicts, each mapping
            process_name -> {'inputs', 'outputs_mean', 'outputs_var',
            'outputs_sampled'}.
        surrogate: Frozen surrogate with compute_reliability(traj) -> Tensor.
        device: Torch device.
        n_samples: Number of MC samples S per trajectory (default 30).
        verbose: Print debug info.

    Returns:
        LambdaMCResult with Λ_MC and per-stage diagnostics.
    """
    if not trajectories:
        raise ValueError("Empty trajectory list")
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2 for unbiased variance estimation")

    process_names = list(trajectories[0].keys())
    N = len(trajectories)
    S = n_samples

    if verbose:
        print(f"\n  Λ_MC — compute_lambda_mc  (N={N}, S={S}, stages={process_names})")

    accum_total = 0.0
    accum_per_stage = {name: 0.0 for name in process_names}
    accum_sigma_sq = {name: 0.0 for name in process_names}

    for i, traj in enumerate(trajectories):
        # ── Step 1: Read μ_t, σ²_t, inputs for each stage ──
        means: Dict[str, torch.Tensor] = {}
        variances: Dict[str, torch.Tensor] = {}
        inputs: Dict[str, torch.Tensor] = {}
        other_keys: Dict[str, Dict[str, torch.Tensor]] = {}

        for name in process_names:
            data = traj[name]
            means[name] = data['outputs_mean'].detach().to(device)
            variances[name] = data['outputs_var'].detach().to(device)
            inputs[name] = data['inputs'].detach().to(device)
            # Accumulate mean σ²_ψt for reference
            accum_sigma_sq[name] += variances[name].sum(dim=-1).mean().item()
            # Preserve any extra keys (besides the ones we handle)
            other_keys[name] = {}
            for k, v in data.items():
                if k not in ('inputs', 'outputs_mean', 'outputs_var', 'outputs_sampled'):
                    if isinstance(v, torch.Tensor):
                        other_keys[name][k] = v.detach().to(device)

        # ── Step 2 & 3: Draw S perturbations and build batched trajectory ──
        # For total Λ_MC: perturb ALL stages simultaneously
        # Stack S copies along batch dimension
        batched_traj: Dict[str, Dict[str, torch.Tensor]] = {}
        for name in process_names:
            mu = means[name]       # (B, output_dim)
            sigma_sq = variances[name]  # (B, output_dim)
            inp = inputs[name]     # (B, input_dim)

            # Repeat μ_t S times: (S*B, output_dim)
            mu_rep = mu.repeat(S, 1)
            sigma_sq_rep = sigma_sq.repeat(S, 1)
            inp_rep = inp.repeat(S, 1)

            # Draw ε ~ N(0, I), shape (S*B, output_dim)
            eps = torch.randn_like(mu_rep)
            o_sampled = mu_rep + eps * torch.sqrt(sigma_sq_rep + 1e-8)

            batched_traj[name] = {
                'inputs': inp_rep,
                'outputs_mean': mu_rep,
                'outputs_var': sigma_sq_rep,
                'outputs_sampled': o_sampled,
            }
            # Copy extra keys
            for k, v in other_keys[name].items():
                batched_traj[name][k] = v.repeat(S, *([1] * (v.dim() - 1)))

        # ── Step 4: Compute F̂^(i,s) for all S at once ──
        with torch.no_grad():
            F_hat_all = surrogate.compute_reliability(batched_traj)  # (S*B,)
            if F_hat_all.dim() == 0:
                F_hat_all = F_hat_all.unsqueeze(0)

        B = means[process_names[0]].shape[0]
        # Reshape to (S, B)
        F_hat_all = F_hat_all.reshape(S, B)

        # ── Step 5: Sample variance over S (unbiased, S-1 denominator) ──
        F_mean = F_hat_all.mean(dim=0)  # (B,)
        var_per_sample = ((F_hat_all - F_mean.unsqueeze(0)) ** 2).sum(dim=0) / (S - 1)  # (B,)
        # Average over batch dimension within this trajectory
        lambda_mc_i = var_per_sample.mean().item()
        accum_total += lambda_mc_i

        # ── Per-stage contributions: perturb ONLY stage t ──
        for name in process_names:
            mu = means[name]
            sigma_sq = variances[name]

            mu_rep = mu.repeat(S, 1)
            sigma_sq_rep = sigma_sq.repeat(S, 1)

            eps = torch.randn_like(mu_rep)
            o_perturbed = mu_rep + eps * torch.sqrt(sigma_sq_rep + 1e-8)

            # Build trajectory: stage `name` perturbed, others at mean
            stage_traj: Dict[str, Dict[str, torch.Tensor]] = {}
            for sname in process_names:
                inp_rep = inputs[sname].repeat(S, 1)
                mu_s_rep = means[sname].repeat(S, 1)
                var_s_rep = variances[sname].repeat(S, 1)

                if sname == name:
                    o_s = o_perturbed
                else:
                    o_s = mu_s_rep  # no noise — keep at mean

                stage_traj[sname] = {
                    'inputs': inp_rep,
                    'outputs_mean': mu_s_rep,
                    'outputs_var': var_s_rep,
                    'outputs_sampled': o_s,
                }
                for k, v in other_keys[sname].items():
                    stage_traj[sname][k] = v.repeat(S, *([1] * (v.dim() - 1)))

            with torch.no_grad():
                F_hat_stage = surrogate.compute_reliability(stage_traj)
                if F_hat_stage.dim() == 0:
                    F_hat_stage = F_hat_stage.unsqueeze(0)

            F_hat_stage = F_hat_stage.reshape(S, B)
            F_mean_stage = F_hat_stage.mean(dim=0)
            var_stage = ((F_hat_stage - F_mean_stage.unsqueeze(0)) ** 2).sum(dim=0) / (S - 1)
            accum_per_stage[name] += var_stage.mean().item()

    # ── Step 6: Average over N trajectories ──
    lambda_mc = accum_total / N
    per_stage = {n: accum_per_stage[n] / N for n in process_names}
    per_stage_sigma_sq = {n: accum_sigma_sq[n] / N for n in process_names}

    if verbose:
        print(f"  Λ_MC(D) = {lambda_mc:.6e}")
        for name in process_names:
            print(f"    {name}: contrib={per_stage[name]:.6e}, "
                  f"σ²={per_stage_sigma_sq[name]:.6e}")

    return LambdaMCResult(
        lambda_mc=lambda_mc,
        per_stage=per_stage,
        per_stage_sigma_sq=per_stage_sigma_sq,
        n_trajectories=N,
        n_samples=S,
        process_names=process_names,
    )
