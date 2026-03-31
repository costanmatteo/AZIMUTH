"""
Monte Carlo estimation of L_min (Method 2, Section 3.4.1.2).

Estimates the irreducible loss floor using only the frozen components:
  - {g_ψt}: frozen uncertainty predictors (one per process stage)
  - f_Θ:   frozen surrogate (CasualiT / ProT transformer)

No knowledge of the SCM structure is required.

Formula (eq. 3.32):
    Λ_MC(D) = (1/N) Σ_i  (1/(S-1)) Σ_s (F̂^(i,s) - F̄^(i))²

where:
    N = number of trajectories in dataset D
    S = number of Monte Carlo perturbations per trajectory
    F̂^(i,s) = f_Θ(τ̂^(i,s)) with ô^(i,s)_t ~ N(μ_ψt, σ²_ψt)
    F̄^(i) = (1/S) Σ_s F̂^(i,s)

The (S-1) denominator gives an unbiased variance estimator.
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
    samples_per_scenario: int = 1,
    S: int = 50,
    loss_scale: float = 100.0,
    verbose: bool = True,
) -> MonteCarloLminResult:
    """
    Compute Λ_MC(D) — Monte Carlo estimate of L_min (Method 2).

    For each trajectory τ^(i) (one per scenario):
      1. Run the process chain to get the actions {a^(i)_t} and the
         frozen UP predictions {μ_t, σ²_t} for each stage t.
      2. Sample S perturbed outcomes ô^(i,s)_t ~ N(μ_t, σ²_t) per stage.
      3. Build S perturbed trajectories and batch-evaluate them through f_Θ.
      4. Compute unbiased sample variance of F̂^(i,s) across the S samples.
    Finally, Λ_MC = mean of per-trajectory variances.

    Args:
        process_chain: ProcessChain with frozen UPs and trained policies.
        surrogate: CasualiTSurrogate (transformer-based). Must NOT be
                   ProTSurrogate (analytical formula), since the MC method
                   is only meaningful when f_Θ is a learned model.
        n_scenarios: Number of training scenarios (trajectories N).
        samples_per_scenario: Batch size per scenario (typically 1 for MC).
        S: Number of MC perturbation samples per trajectory (default 50).
        loss_scale: Scale factor matching training (default 100.0).
        verbose: Print progress.

    Returns:
        MonteCarloLminResult with Λ_MC(D) and diagnostics.
    """
    t0 = time.time()

    process_chain.eval()
    process_names = process_chain.process_names
    N = n_scenarios  # one trajectory per scenario

    per_trajectory_var = np.zeros(N)

    with torch.no_grad():
        for i in range(N):
            # ── Step 1: get actions and UP predictions for trajectory i ──
            # Run a single forward pass to obtain the controller actions
            # and the frozen UP outputs (μ_t, σ²_t) at each stage.
            ref_trajectory = process_chain.forward(
                batch_size=1, scenario_idx=i
            )

            # Collect per-stage inputs (actions) and UP predictions
            stage_inputs = []   # a^(i)_t  — full input vectors per stage
            stage_mu = []       # μ_t(a^(i)_t)
            stage_var = []      # σ²_t(a^(i)_t)

            for proc_name in process_names:
                data = ref_trajectory[proc_name]
                stage_inputs.append(data['inputs'])        # (1, input_dim)
                stage_mu.append(data['outputs_mean'])      # (1, output_dim)
                stage_var.append(data['outputs_var'])       # (1, output_dim)

            # ── Step 2: sample S perturbed trajectories (batched) ────────
            # For each stage t, sample S outcomes from N(μ_t, σ²_t).
            # The actions a^(i)_t stay fixed; only outcomes are perturbed.
            perturbed_trajectory = {}

            for t, proc_name in enumerate(process_names):
                mu_t = stage_mu[t]    # (1, output_dim)
                var_t = stage_var[t]  # (1, output_dim)
                inp_t = stage_inputs[t]  # (1, input_dim)

                # Expand to (S, dim)
                mu_expanded = mu_t.expand(S, -1)           # (S, output_dim)
                var_expanded = var_t.expand(S, -1)          # (S, output_dim)
                inp_expanded = inp_t.expand(S, -1)          # (S, input_dim)

                # Sample: ô^(i,s)_t ~ N(μ_t, σ²_t)
                std_t = torch.sqrt(var_expanded + 1e-8)
                epsilon = torch.randn_like(mu_expanded)
                outputs_sampled = mu_expanded + epsilon * std_t

                perturbed_trajectory[proc_name] = {
                    'inputs': inp_expanded,
                    'outputs_mean': mu_expanded,
                    'outputs_var': var_expanded,
                    'outputs_sampled': outputs_sampled,
                }

            # ── Step 3: batch forward pass through f_Θ ──────────────────
            F_hat = surrogate.compute_reliability(perturbed_trajectory)
            # F_hat shape: (S,) or scalar — ensure 1D
            F_hat_np = F_hat.detach().cpu().numpy().ravel()

            # ── Step 4: unbiased sample variance ────────────────────────
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
