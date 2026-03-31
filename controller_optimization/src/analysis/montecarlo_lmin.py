"""
Monte Carlo estimation of L_min (Method 2, Section 3.4.1.2).

Estimates the irreducible loss floor using only the frozen components:
  - {g_ψt}: frozen uncertainty predictors (one per process stage)
  - f_Θ:   frozen surrogate (CasualiT / ProT transformer)

No knowledge of the SCM structure is required.

Formula (eq. 3.32):
    Λ_MC(D) = (1/N) Σ_i  (1/(S-1)) Σ_s (F̂^(i,s) - F̄^(i))²

where:
    N = number of trajectories in dataset D (baseline trajectories {τ'^(k)})
    S = number of Monte Carlo perturbations per trajectory
    F̂^(i,s) = f_Θ(τ̂^(i,s)) with ô^(i,s)_t ~ N(μ_ψt, σ²_ψt)
    F̄^(i) = (1/S) Σ_s F̂^(i,s)

The (S-1) denominator gives an unbiased variance estimator.

Dataset D: uses baseline trajectories {τ'^(k)} — the fixed-action trajectories
from the training scenarios — NOT the controller-generated trajectories.
This ensures Λ_MC measures irreducible noise independently of the controller.
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


def _build_baseline_trajectory_for_scenario(
    process_chain,
    baseline_trajectories: Dict,
    scenario_idx: int,
) -> Dict:
    """
    Build a single baseline trajectory τ'^(k) for scenario k.

    Uses the fixed baseline actions a'^(k)_t and runs them through
    the frozen UPs to get {μ_t, σ²_t}. This is the dataset D.

    Args:
        process_chain: ProcessChain with frozen UPs.
        baseline_trajectories: Dict {proc_name: {'inputs': (N, dim), 'outputs': (N, dim)}}.
        scenario_idx: Which scenario (row) to use.

    Returns:
        trajectory: Dict {proc_name: {'inputs': (1, in_dim),
                                       'outputs_mean': (1, out_dim),
                                       'outputs_var': (1, out_dim),
                                       'outputs_sampled': (1, out_dim)}}
    """
    device = process_chain.device
    trajectory = {}

    for t, proc_name in enumerate(process_chain.process_names):
        # Baseline inputs for this scenario — fixed actions a'^(k)_t
        baseline_inputs = baseline_trajectories[proc_name]['inputs']
        inp_np = baseline_inputs[scenario_idx]  # (input_dim,)
        inp_t = torch.tensor(inp_np, dtype=torch.float32, device=device).unsqueeze(0)  # (1, input_dim)

        # Run through frozen UP to get μ_t(a'^(k)_t) and σ²_t(a'^(k)_t)
        scaled = process_chain.scale_inputs(inp_t, t)
        mu_scaled, var_scaled = process_chain.uncertainty_predictors[t](scaled)
        mu_t = process_chain.unscale_outputs(mu_scaled, t)    # (1, output_dim)
        var_t = process_chain.unscale_variance(var_scaled, t)  # (1, output_dim)

        # Deterministic "sample" = mean (for the reference trajectory)
        trajectory[proc_name] = {
            'inputs': inp_t,
            'outputs_mean': mu_t,
            'outputs_var': var_t,
            'outputs_sampled': mu_t.clone(),  # deterministic baseline
        }

    return trajectory


def compute_montecarlo_lmin(
    process_chain,
    surrogate,
    baseline_trajectories: Dict,
    n_scenarios: int,
    S: int = 50,
    loss_scale: float = 100.0,
    verbose: bool = True,
) -> MonteCarloLminResult:
    """
    Compute Λ_MC(D) — Monte Carlo estimate of L_min (Method 2).

    Dataset D = baseline trajectories {τ'^(k)}, k=1..N.
    These use the FIXED baseline actions (no controller), so that
    Λ_MC measures the irreducible noise floor of f_Θ.

    Algorithm:
      for each baseline trajectory τ'^(i) in D:
          # Get fixed actions and UP predictions
          a^(i)_t = baseline inputs for scenario i
          μ_t, σ²_t = g_ψt(a^(i)_t)   (frozen UP)

          for s in 1..S:
              for each stage t:
                  ô^(i,s)_t ~ N(μ_t, σ²_t)
              τ̂^(i,s) = {a^(i)_t, ô^(i,s)_t}
              F̂^(i,s) = f_Θ(τ̂^(i,s))

          F̄^(i) = mean(F̂^(i,1..S))
          var^(i) = (1/(S-1)) Σ_s (F̂^(i,s) - F̄^(i))²

      Λ_MC = mean(var^(i))

    Args:
        process_chain: ProcessChain with frozen UPs.
        surrogate: Frozen surrogate f_Θ (CasualiTSurrogate or ProTSurrogate).
        baseline_trajectories: Dict from generate_baseline_trajectories().
            Structure: {proc_name: {'inputs': (N, dim), 'outputs': (N, dim)}}.
        n_scenarios: Number of baseline scenarios (N trajectories).
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
            # ── Step 1: build baseline trajectory for scenario i ────────
            # Uses fixed baseline actions a'^(i)_t, NOT controller actions.
            # Runs through frozen UPs to get μ_t(a'^(i)_t), σ²_t(a'^(i)_t).
            ref_traj = _build_baseline_trajectory_for_scenario(
                process_chain, baseline_trajectories, i
            )

            # Collect per-stage UP predictions
            stage_inputs = []   # a'^(i)_t — fixed baseline actions
            stage_mu = []       # μ_t(a'^(i)_t)
            stage_var = []      # σ²_t(a'^(i)_t)

            for proc_name in process_names:
                data = ref_traj[proc_name]
                stage_inputs.append(data['inputs'])        # (1, input_dim)
                stage_mu.append(data['outputs_mean'])      # (1, output_dim)
                stage_var.append(data['outputs_var'])       # (1, output_dim)

            # ── Step 2: sample S perturbed trajectories (batched) ────────
            # For each stage t, sample S outcomes from N(μ_t, σ²_t).
            # Actions a'^(i)_t stay fixed; only outcomes ô_t are perturbed.
            # Token for surrogate = [a_t, μ_t, σ²_t] per stage (Section 3.1.3),
            # but outputs_sampled carries the perturbed ô for paths that use it.
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
                # - TransformerForecaster path uses [inputs, outputs_mean, outputs_var]
                #   → outputs_mean IS the perturbed sample ô (the "observed" output)
                #     outputs_var stays as the UP prediction (token = [a_t, ô_t, σ²_t])
                # - SimpleSurrogateModel path uses [inputs, outputs_sampled]
                # - StageCausaliT path uses [inputs (S), outputs_sampled (X)]
                #
                # To make ALL paths see the perturbation:
                #   outputs_mean = ô^(i,s)_t  (perturbed sample)
                #   outputs_sampled = ô^(i,s)_t  (same)
                #   outputs_var = σ²_t  (original UP variance, unchanged)
                perturbed_trajectory[proc_name] = {
                    'inputs': inp_expanded,
                    'outputs_mean': outputs_sampled,    # ← perturbed ô, not μ
                    'outputs_var': var_expanded,         # σ²_t from frozen UP
                    'outputs_sampled': outputs_sampled,  # ← perturbed ô
                }

            # ── Step 3: batch forward pass through f_Θ ──────────────────
            # All S perturbations evaluated in a single batch (S, seq_len, token_dim).
            F_hat = surrogate.compute_reliability(perturbed_trajectory)
            # F_hat shape: (S,) or scalar — ensure 1D
            F_hat_np = F_hat.detach().cpu().numpy().ravel()

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
