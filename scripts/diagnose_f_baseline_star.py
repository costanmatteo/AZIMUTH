"""
Diagnostic: investigate why F_baseline ≈ F_star in compute_reliability.

Hypothesis: the adaptive target τ_i shifts so much toward baseline outputs
that Q_i ≈ 1 even without a controller, making F non-discriminative.

Root cause candidate: in ReliabilityFunction._compute_adaptive_target(),
the upstream outputs used to compute the shift are the BASELINE outputs
themselves. So for the baseline trajectory,
    delta = baseline_upstream - adaptive_baselines[upstream]
and the target tracks the baseline. The adaptive_baselines dict is a fixed
configured anchor, which typically matches the baseline's nominal upstream
value, so the shift follows the baseline rather than being anchored to the
target trajectory.

The fix: compute the shift as
    delta = upstream_baseline_output - upstream_target_output
where upstream_target_output comes from the target trajectory.

This script:
  1) Synthesizes a target trajectory and a batch of baseline trajectories
     using realistic distributions anchored on PROCESS_CONFIGS defaults.
  2) Runs the existing ReliabilityFunction to report τ_i vs base_target_i,
     and the shift fraction.
  3) Computes F under three target modes:
        F_adaptive  — current mode (adaptive)
        F_fixed     — adaptive_coefficients forced to {}
        F_star      — target trajectory outputs used as targets
  4) Prints a CSV of τ_i vs base_target_i vs y_i for 'galvanic' and
     'microetch' processes, sorted by y_i.
"""

from __future__ import annotations

import copy
import os
import sys

import numpy as np
import torch

# Make the repo importable regardless of cwd
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scm_ds.compute_reliability import ReliabilityFunction
from scm_ds.process_targets import PROCESS_CONFIGS, PROCESS_ORDER


# ---------------------------------------------------------------------------
# Synthetic trajectory generation
# ---------------------------------------------------------------------------

def _nominal_output_for(process_name: str) -> float:
    """Nominal output value to centre the synthetic trajectories on."""
    cfg = PROCESS_CONFIGS[process_name]
    base = cfg.get('base_target', 0.0)
    if isinstance(base, list):
        return float(np.mean(base))
    return float(base)


def make_target_trajectory(seed: int = 42) -> dict:
    """
    Build a single-sample target trajectory with one output per process.
    Outputs sit near base_target with a tiny deterministic offset so the
    target trajectory is a specific point in output space (not identically
    equal to base_target).
    """
    rng = np.random.RandomState(seed)
    traj = {}
    for name in PROCESS_ORDER:
        if name not in PROCESS_CONFIGS:
            continue
        nominal = _nominal_output_for(name)
        # Small offset so target ≠ base_target exactly
        offset = 0.05 * nominal * rng.standard_normal()
        out = np.array([[nominal + offset]], dtype=np.float32)  # (1, 1)
        traj[name] = {
            'outputs': out,
            'outputs_mean': out,
        }
    return traj


def make_baseline_trajectory(n: int = 200, seed: int = 0) -> dict:
    """
    Build n baseline trajectories. Outputs are sampled around each process's
    nominal output with moderate noise, emulating the spread of an
    uncontrolled baseline chain.
    """
    rng = np.random.RandomState(seed)
    traj = {}
    for name in PROCESS_ORDER:
        if name not in PROCESS_CONFIGS:
            continue
        nominal = _nominal_output_for(name)
        # Standard deviation ~= 10% of nominal, with a floor to keep it visible
        std = max(0.1 * abs(nominal), 0.05)
        out = (nominal + std * rng.standard_normal(size=(n, 1))).astype(np.float32)
        traj[name] = {
            'outputs_mean': out,
            'outputs_sampled': out,
        }
    return traj


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _make_fixed_configs(base_configs: dict) -> dict:
    """Return a copy of process_configs with adaptive_coefficients stripped."""
    fixed = copy.deepcopy(base_configs)
    for cfg in fixed.values():
        cfg['adaptive_coefficients'] = {}
    return fixed


def _make_targetlocked_configs(base_configs: dict, target_outputs: dict) -> dict:
    """
    Return process_configs where base_target for every process is set to its
    target-trajectory output. With no adaptive coefficients, Q_i is measured
    against the target trajectory directly — this is F*.
    """
    locked = copy.deepcopy(base_configs)
    for name, cfg in locked.items():
        if name not in target_outputs:
            continue
        y_star = float(np.asarray(target_outputs[name]).flatten()[0])
        cfg['base_target'] = y_star
        cfg['adaptive_coefficients'] = {}
    return locked


def compute_F(rf: ReliabilityFunction, trajectory: dict) -> np.ndarray:
    F = rf.compute_reliability(trajectory, return_quality_scores=False,
                               use_sampled_outputs=True)
    return _to_np(F).reshape(-1)


def compute_F_and_targets(rf: ReliabilityFunction, trajectory: dict):
    """
    Run the reliability computation and also extract the per-sample τ_i used
    internally. We re-create the adaptive-target step here so we can report τ_i
    without changing the library.
    """
    outputs = {}
    for name, data in trajectory.items():
        out = data.get('outputs_sampled', data.get('outputs_mean'))
        if isinstance(out, np.ndarray):
            out = torch.tensor(out, dtype=torch.float32)
        if out.dim() == 1:
            out = out.unsqueeze(-1)
        outputs[name] = out

    taus = {}
    for name in rf.process_order:
        if name not in outputs:
            continue
        cfg = rf.process_configs.get(name, {})
        tau = rf._compute_adaptive_target(name, outputs, cfg)
        # Normalise tau to per-sample 1-D numpy
        if isinstance(tau, torch.Tensor):
            tau_np = tau.detach().cpu().numpy()
        else:
            tau_np = np.asarray(tau)
        n = outputs[name].shape[0]
        tau_np = np.broadcast_to(tau_np.reshape(-1, tau_np.shape[-1])
                                 if tau_np.ndim else np.array([[float(tau_np)]]),
                                 (n, 1)).copy()
        taus[name] = tau_np.flatten()

    F = compute_F(rf, trajectory)
    return F, taus


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def main():
    np.set_printoptions(precision=4, suppress=True)

    N = 200
    target_traj = make_target_trajectory(seed=42)
    baseline_traj = make_baseline_trajectory(n=N, seed=0)

    # Extract target outputs in plain numpy form
    target_outputs = {
        name: np.asarray(data['outputs']).flatten()[0]
        for name, data in target_traj.items()
    }

    print("=" * 78)
    print("DIAGNOSTIC: F_baseline vs F_star")
    print("=" * 78)
    print(f"n_baselines={N}")
    print("target trajectory outputs:")
    for name, y in target_outputs.items():
        base = PROCESS_CONFIGS[name]['base_target']
        print(f"  {name:10s}: y*={y:.4f}  base_target={base}")
    print()

    # -----------------------------------------------------------------------
    # (1) τ_i vs y_i and base_target_i
    # -----------------------------------------------------------------------
    rf_adaptive = ReliabilityFunction(process_configs=PROCESS_CONFIGS)
    F_adaptive, taus = compute_F_and_targets(rf_adaptive, baseline_traj)

    print("-" * 78)
    print("(1) Does τ_i track y_i? — baseline trajectory")
    print("-" * 78)
    print(f"{'process':10s} {'|y-τ|':>10s} {'|y-base|':>10s} {'ratio':>8s} "
          f"{'mean_y':>10s} {'mean_τ':>10s} {'base':>10s}")
    for name in PROCESS_ORDER:
        if name not in baseline_traj:
            continue
        y = np.asarray(baseline_traj[name]['outputs_sampled']).flatten()
        tau = taus[name]
        base = PROCESS_CONFIGS[name]['base_target']
        base_scalar = float(np.mean(base)) if isinstance(base, list) else float(base)
        res_adaptive = np.abs(y - tau)
        res_fixed = np.abs(y - base_scalar)
        ratio = res_adaptive.mean() / (res_fixed.mean() + 1e-8)
        print(f"{name:10s} {res_adaptive.mean():10.4f} {res_fixed.mean():10.4f} "
              f"{ratio:8.3f} {y.mean():10.4f} {tau.mean():10.4f} {base_scalar:10.4f}")
    print()
    print("Interpretation: if |y-τ| << |y-base|, adaptive τ is tracking y "
          "(target follows the baseline).")
    print()

    # -----------------------------------------------------------------------
    # (2) Adaptive shift fraction
    # -----------------------------------------------------------------------
    print("-" * 78)
    print("(2) shift_fraction = |τ - base| / (|y - base| + 1e-8)")
    print("-" * 78)
    print(f"{'process':10s} {'mean':>10s} {'median':>10s} {'p90':>10s} "
          f"{'frac>0.5':>10s}")
    for name in PROCESS_ORDER:
        if name not in baseline_traj:
            continue
        y = np.asarray(baseline_traj[name]['outputs_sampled']).flatten()
        tau = taus[name]
        base = PROCESS_CONFIGS[name]['base_target']
        base_scalar = float(np.mean(base)) if isinstance(base, list) else float(base)
        num = np.abs(tau - base_scalar)
        den = np.abs(y - base_scalar) + 1e-8
        frac = num / den
        over = float(np.mean(frac > 0.5))
        print(f"{name:10s} {frac.mean():10.4f} {np.median(frac):10.4f} "
              f"{np.percentile(frac, 90):10.4f} {over:10.3f}")
    print()
    print("Interpretation: shift_fraction > 0.5 for most samples means the "
          "adaptive target is over-compensating toward y.")
    print()

    # -----------------------------------------------------------------------
    # (3) Three F modes
    # -----------------------------------------------------------------------
    fixed_configs = _make_fixed_configs(PROCESS_CONFIGS)
    rf_fixed = ReliabilityFunction(process_configs=fixed_configs)
    F_fixed = compute_F(rf_fixed, baseline_traj)

    star_configs = _make_targetlocked_configs(PROCESS_CONFIGS, target_outputs)
    rf_star = ReliabilityFunction(process_configs=star_configs)
    F_star_on_baseline = compute_F(rf_star, baseline_traj)

    # F* as a sanity check — apply the *adaptive* reliability to the target
    # trajectory (broadcast the single target sample to match N).
    target_batched = {
        name: {
            'outputs_mean': np.repeat(np.asarray(data['outputs']), N, axis=0),
            'outputs_sampled': np.repeat(np.asarray(data['outputs']), N, axis=0),
        }
        for name, data in target_traj.items()
    }
    F_star_target = compute_F(rf_adaptive, target_batched)

    print("-" * 78)
    print("(3) F computed three ways on the SAME baseline trajectory")
    print("-" * 78)
    print(f"F_adaptive (current)               : "
          f"{F_adaptive.mean():.4f} ± {F_adaptive.std():.4f}")
    print(f"F_fixed    (no adaptive shift)     : "
          f"{F_fixed.mean():.4f} ± {F_fixed.std():.4f}")
    print(f"F_star_on_baseline (target=y*)     : "
          f"{F_star_on_baseline.mean():.4f} ± {F_star_on_baseline.std():.4f}")
    print(f"F_star (adaptive on target traj)   : "
          f"{F_star_target.mean():.4f} ± {F_star_target.std():.4f}")
    print()
    gap = F_star_target.mean() - F_adaptive.mean()
    print(f"Gap F_star - F_adaptive = {gap:+.4f}")
    if abs(F_adaptive.mean() - F_fixed.mean()) < 1e-3:
        print("→ F_adaptive ≈ F_fixed: adaptive coefficients are NOT the culprit.")
    elif F_adaptive.mean() > F_fixed.mean() + 1e-3:
        print("→ F_adaptive > F_fixed: adaptive targets INFLATE baseline scores.")
    else:
        print("→ F_adaptive < F_fixed: adaptive targets deflate baseline scores.")
    print()

    # -----------------------------------------------------------------------
    # (4) CSV dump for galvanic and microetch
    # -----------------------------------------------------------------------
    print("-" * 78)
    print("(4) CSV: y_i (sorted) vs τ_i (adaptive), base_target_i, identity")
    print("-" * 78)
    for name in ['galvanic', 'microetch']:
        if name not in baseline_traj:
            continue
        y = np.asarray(baseline_traj[name]['outputs_sampled']).flatten()
        tau = taus[name]
        base = PROCESS_CONFIGS[name]['base_target']
        base_scalar = float(np.mean(base)) if isinstance(base, list) else float(base)
        order = np.argsort(y)
        y_s = y[order]
        tau_s = tau[order]
        # Print as CSV header + up to 15 quantile rows for compactness
        print(f"\n--- process: {name} ---")
        print("idx,y_i,tau_i,base_target_i,identity")
        # Show 15 evenly spaced indices so the reader can eyeball the trend
        idxs = np.linspace(0, len(y_s) - 1, 15).astype(int)
        for k in idxs:
            print(f"{k},{y_s[k]:.4f},{tau_s[k]:.4f},"
                  f"{base_scalar:.4f},{y_s[k]:.4f}")
        # Summary: correlations
        corr_tau_y = float(np.corrcoef(y_s, tau_s)[0, 1])
        print(f"corr(y_i, τ_i) = {corr_tau_y:+.4f}  "
              "(≈1 means τ tracks y → bug; ≈0 means τ ignores y)")

    print()
    print("=" * 78)
    print("Conclusion")
    print("=" * 78)
    print(
        "The adaptive target τ_i is computed from upstream outputs taken from\n"
        "the CURRENT trajectory (i.e. the baseline). The delta used for the\n"
        "shift is (upstream_baseline_output - adaptive_baselines[upstream]),\n"
        "so τ shifts every time the baseline upstream deviates from its\n"
        "configured anchor. The anchor is usually set to the baseline's\n"
        "nominal value, so τ effectively follows the baseline — driving\n"
        "Q_i ≈ 1 and F_baseline ≈ F_star.\n"
        "\n"
        "Fix: compute the delta against the TARGET trajectory's upstream\n"
        "output, i.e.\n"
        "    delta = upstream_baseline_output - upstream_target_output\n"
        "so τ is anchored to the target trajectory, not to the baseline."
    )


if __name__ == '__main__':
    main()
