"""
Explore F' (baseline reliability) across different seed_baseline values
while keeping seed_target fixed.

Computes F* from the target trajectory and, for each seed_baseline in [1, 100],
generates a single baseline trajectory and computes F' using ProTSurrogate
(formula-based, NOT CasualiT).
"""

import sys
import os
import io
import contextlib
from pathlib import Path
import numpy as np
import torch

# Add project root to path (same as train_controller.py)
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'causaliT'))

from configs.processes_config import (
    get_filtered_processes, DATASET_MODE, ST_DATASET_CONFIG, _build_st_processes
)
from configs.controller_config import CONTROLLER_CONFIG
from controller.src.core.target_generation import (
    generate_target_trajectory, generate_baseline_trajectories
)
from controller.src.models.surrogate.surrogate import ProTSurrogate


# =============================================================================
# PARAMETERS
# =============================================================================
SEED_TARGET = 42
SEED_BASELINE_RANGE = range(1, 101)  # 1..100 inclusive
DEVICE = 'cpu'


def _silent():
    """Context manager that redirects stdout to a throwaway buffer."""
    return contextlib.redirect_stdout(io.StringIO())


def main():
    # Filter processes according to the controller config (same as training).
    process_names = CONTROLLER_CONFIG.get('process_names', None)
    with _silent():
        selected_processes = get_filtered_processes(process_names)

    print("=" * 70)
    print("F' EXPLORATION: seed_target fixed, seed_baseline varied")
    print("=" * 70)
    print(f"DATASET_MODE:        {DATASET_MODE}")
    print(f"Processes:           {[p['name'] for p in selected_processes]}")
    print(f"seed_target:         {SEED_TARGET}")
    print(f"seed_baseline range: [{SEED_BASELINE_RANGE.start}, "
          f"{SEED_BASELINE_RANGE.stop - 1}]")
    print()

    # ------------------------------------------------------------------
    # 1) Generate target trajectory with fixed seed
    # ------------------------------------------------------------------
    with _silent():
        target_trajectory = generate_target_trajectory(
            process_configs=selected_processes,
            n_samples=1,
            seed=SEED_TARGET,
        )

    # ------------------------------------------------------------------
    # 2) Build ProTSurrogate (formula-based) and capture F*
    # ------------------------------------------------------------------
    with _silent():
        surrogate = ProTSurrogate(
            target_trajectory=target_trajectory,
            device=DEVICE,
            use_deterministic_sampling=True,
            process_configs=selected_processes,
            n_scenarios=1,
        )
    F_star = float(surrogate.F_star)
    print(f"F* = {F_star:.6f}")
    print()

    # ------------------------------------------------------------------
    # 3) Loop over seed_baseline: compute F' per seed
    # ------------------------------------------------------------------
    header = f"{'seed':>5} | {'F_prime':>10} | {'gap (F*-F_prime)':>18} | {'gap%':>8}"
    print(header)
    print("-" * len(header))

    f_prime_values = []
    for k in SEED_BASELINE_RANGE:
        with _silent():
            baseline = generate_baseline_trajectories(
                process_configs=selected_processes,
                target_trajectory=target_trajectory,
                n_baselines=1,
                seed_env=k,
                seed_noise=k,
            )

        # Build the dict for compute_reliability: 1-row tensors per process,
        # with zero variance (deterministic baseline evaluation).
        baseline_tensor = {}
        for process_name, data in baseline.items():
            inputs_row = torch.tensor(data['inputs'], dtype=torch.float32,
                                      device=DEVICE)
            outputs_row = torch.tensor(data['outputs'], dtype=torch.float32,
                                       device=DEVICE)
            baseline_tensor[process_name] = {
                'inputs': inputs_row,
                'outputs_mean': outputs_row,
                'outputs_var': torch.zeros_like(outputs_row),
            }

        with torch.no_grad():
            F_prime = surrogate.compute_reliability(baseline_tensor).item()

        f_prime_values.append(F_prime)
        gap = F_star - F_prime
        gap_pct = (gap / F_star * 100.0) if F_star != 0.0 else float('nan')
        print(f"{k:>5d} | {F_prime:>10.6f} | {gap:>18.6f} | {gap_pct:>7.2f}%")

    # ------------------------------------------------------------------
    # 4) Final statistics
    # ------------------------------------------------------------------
    f_prime_arr = np.array(f_prime_values, dtype=np.float64)
    print()
    print("=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)
    print(f"  F*           : {F_star:.6f}")
    print(f"  mean(F')     : {f_prime_arr.mean():.6f}")
    print(f"  std(F')      : {f_prime_arr.std(ddof=0):.6f}")
    print(f"  min(F')      : {f_prime_arr.min():.6f}")
    print(f"  max(F')      : {f_prime_arr.max():.6f}")
    print(f"  mean gap     : {(F_star - f_prime_arr.mean()):.6f}")
    print(f"  mean gap %   : {((F_star - f_prime_arr.mean()) / F_star * 100.0):.2f}%")


if __name__ == '__main__':
    main()
