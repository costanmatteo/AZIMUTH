"""
Monte Carlo estimation of L_min (Method 2, Section 3.4.1.2).

Estimates the irreducible loss floor using only the frozen components:
  - {g_ψt}: frozen uncertainty predictors (one per process stage)
  - f_Θ:   frozen surrogate (CasualiT / ProT transformer)

No knowledge of the SCM structure is required.

Formula (eq. 3.32):
    Λ_MC(D) = (1/N) Σ_i  (1/(S-1)) Σ_s (F̂^(i,s) - F̄^(i))²

where:
    N = number of trajectories (one per scenario, using trained controller)
    S = number of Monte Carlo perturbations per trajectory
    F̂^(i,s) = f_Θ(τ̂^(i,s)) with ô^(i,s)_t ~ N(μ_ψt, σ²_ψt)
    F̄^(i) = (1/S) Σ_s F̂^(i,s)

The (S-1) denominator gives an unbiased variance estimator.

Dataset D: uses the trained controller's trajectories — the actions a_t are
those chosen by the trained policy generators, and for each trajectory the
MC perturbation resamples only the outcomes ô_t ~ N(μ_t, σ²_t) while
keeping a_t fixed. This measures the irreducible noise at the controller's
operating point, which is what L_min represents.
"""

import numpy as np
import torch
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path
import json


@dataclass
class MonteCarloLminResult:
    """Results from Monte Carlo L_min estimation."""
    L_min_mc: float              # Unscaled Λ_MC(D)
    L_min_mc_scaled: float       # Λ_MC(D) × loss_scale
    std_error: float             # std(var_i) / sqrt(N)
    std_error_scaled: float      # std_error × loss_scale
    N_trajectories: int          # Number of trajectories used
    S_samples: int               # Monte Carlo samples per trajectory
    per_trajectory_var: np.ndarray  # var^(i) for each trajectory (unscaled)
    computation_time_s: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'L_min_mc': self.L_min_mc,
            'L_min_mc_scaled': self.L_min_mc_scaled,
            'std_error': self.std_error,
            'std_error_scaled': self.std_error_scaled,
            'N_trajectories': self.N_trajectories,
            'S_samples': self.S_samples,
            'per_trajectory_var_mean': float(np.mean(self.per_trajectory_var)),
            'per_trajectory_var_std': float(np.std(self.per_trajectory_var)),
            'computation_time_s': self.computation_time_s,
        }

    def save(self, path: Path):
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def compute_montecarlo_lmin(
    process_chain,
    surrogate,
    n_scenarios: int,
    S: int = 50,
    loss_scale: float = 100.0,
    verbose: bool = True,
) -> MonteCarloLminResult:
    """
    Compute Λ_MC(D) — Monte Carlo estimate of L_min (Method 2).

    For each scenario i = 1..N:
      1. Run one forward pass through the trained controller to get the
         actions {a^(i)_t} and the frozen UP predictions {μ_t, σ²_t}.
      2. Keep a^(i)_t FIXED, resample S times: ô^(i,s)_t ~ N(μ_t, σ²_t).
      3. Batch-evaluate the S perturbed trajectories through f_Θ.
      4. Compute unbiased sample variance of F̂^(i,s) across S samples.

    Λ_MC = mean of per-trajectory variances.

    This measures the irreducible variance at the trained controller's
    operating point — the part of the loss that cannot be reduced by
    further policy optimization.

    Args:
        process_chain: ProcessChain with frozen UPs and trained policies.
        surrogate: Frozen surrogate f_Θ (CasualiTSurrogate or ProTSurrogate).
        n_scenarios: Number of training scenarios (N trajectories).
        S: MC perturbation samples per trajectory (default 50). Must be >= 2.
        loss_scale: Scale factor matching training (default 100.0).
        verbose: Print progress.

    Returns:
        MonteCarloLminResult with Λ_MC(D) and diagnostics.

    Raises:
        ValueError: If S < 2 (need at least 2 samples for unbiased variance).
    """
    if S < 2:
        raise ValueError(
            f"S must be >= 2 for unbiased variance estimation (got S={S}). "
            f"With S=1 the (S-1) denominator is zero."
        )

    t0 = time.time()

    process_chain.eval()
    process_names = process_chain.process_names
    device = process_chain.device
    N = n_scenarios

    per_trajectory_var = np.zeros(N)

    with torch.no_grad():
        for i in range(N):
            # ── Step 1: forward pass with trained controller ────────────
            # Gets the controller's actions a^(i)_t AND the frozen UP
            # predictions μ_t(a^(i)_t), σ²_t(a^(i)_t) at those actions.
            ref_traj = process_chain.forward(batch_size=1, scenario_idx=i)

            # Collect per-stage data
            stage_inputs = []   # a^(i)_t — controller actions (fixed for MC)
            stage_mu = []       # μ_t(a^(i)_t) from frozen UP
            stage_var = []      # σ²_t(a^(i)_t) from frozen UP

            for proc_name in process_names:
                data = ref_traj[proc_name]
                stage_inputs.append(data['inputs'])        # (1, input_dim)
                stage_mu.append(data['outputs_mean'])      # (1, output_dim)
                stage_var.append(data['outputs_var'])       # (1, output_dim)

            # ── Step 2: sample S perturbed trajectories (batched) ────────
            # Actions a^(i)_t stay FIXED. Only outcomes are resampled:
            #   ô^(i,s)_t ~ N(μ_t, σ²_t) for s=1..S
            perturbed_trajectory = {}

            for t, proc_name in enumerate(process_names):
                mu_t = stage_mu[t]     # (1, output_dim)
                var_t = stage_var[t]   # (1, output_dim)
                inp_t = stage_inputs[t]  # (1, input_dim)

                # Expand to (S, dim)
                mu_expanded = mu_t.expand(S, -1)            # (S, output_dim)
                var_expanded = var_t.expand(S, -1)           # (S, output_dim)
                inp_expanded = inp_t.expand(S, -1)           # (S, input_dim)

                # Sample: ô^(i,s)_t ~ N(μ_t, σ²_t)
                std_t = torch.sqrt(var_expanded + 1e-8)
                epsilon = torch.randn_like(mu_expanded)
                outputs_sampled = mu_expanded + epsilon * std_t

                # Build trajectory dict with all fields the surrogate needs:
                # - TransformerForecaster path reads [inputs, outputs_mean, outputs_var]
                #   → outputs_mean = perturbed ô (token = [a_t, ô_t, σ²_t])
                # - SimpleSurrogateModel path reads [inputs, outputs_sampled]
                # - StageCausaliT path reads inputs (S) and outputs_sampled (X)
                #
                # All paths see the perturbation:
                perturbed_trajectory[proc_name] = {
                    'inputs': inp_expanded,
                    'outputs_mean': outputs_sampled,    # ← perturbed ô, not μ
                    'outputs_var': var_expanded,         # σ²_t from frozen UP
                    'outputs_sampled': outputs_sampled,  # ← perturbed ô
                }

            # ── Step 3: batch forward pass through f_Θ ──────────────────
            # All S perturbations in a single batch (S, seq_len, token_dim).
            F_hat = surrogate.compute_reliability(perturbed_trajectory)
            F_hat_np = F_hat.detach().cpu().numpy().ravel()  # (S,)

            # ── Step 4: unbiased sample variance with (S-1) denominator ──
            F_bar_i = np.mean(F_hat_np)
            var_i = np.sum((F_hat_np - F_bar_i) ** 2) / (S - 1)
            per_trajectory_var[i] = var_i

            if verbose and (i + 1) % max(1, N // 5) == 0:
                print(f"    MC L_min: trajectory {i+1}/{N}, "
                      f"var^(i)={var_i:.8f}, F̄={F_bar_i:.6f}")

    # ── Aggregate: Λ_MC = mean of per-trajectory variances ──────────────
    L_min_mc = float(np.mean(per_trajectory_var))
    L_min_mc_scaled = L_min_mc * loss_scale
    std_var = float(np.std(per_trajectory_var, ddof=1)) if N > 1 else 0.0
    std_error = std_var / np.sqrt(N) if N > 1 else 0.0
    std_error_scaled = std_error * loss_scale

    elapsed = time.time() - t0

    result = MonteCarloLminResult(
        L_min_mc=L_min_mc,
        L_min_mc_scaled=L_min_mc_scaled,
        std_error=std_error,
        std_error_scaled=std_error_scaled,
        N_trajectories=N,
        S_samples=S,
        per_trajectory_var=per_trajectory_var,
        computation_time_s=elapsed,
    )

    if verbose:
        print(f"\n  Monte Carlo L_min (Method 2) results:")
        print(f"    Λ_MC(D) (unscaled):  {L_min_mc:.10f}")
        print(f"    Λ_MC(D) (×{loss_scale:.0f}):     {L_min_mc_scaled:.6f}")
        print(f"    Std error (scaled):   ±{std_error_scaled:.6f}")
        print(f"    N trajectories:       {N}")
        print(f"    S samples/trajectory: {S}")
        print(f"    Computation time:     {elapsed:.1f}s")

    return result
