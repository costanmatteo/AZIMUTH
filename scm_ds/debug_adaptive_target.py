"""Debug helper for per-dimension adaptive target computation.

Generates a small ST SCM, samples a few trajectories, and prints a
step-by-step breakdown of how ``_compute_adaptive_target`` builds the
adaptive target ζ*_t for each process. For each upstream contribution it
also shows what the LEGACY (pre-fix) logic would have produced — the
scalar mean collapse with a scalar baseline — so you can see numerically
whether the per-dim fix actually changes the signal.

Usage (from repo root)::

    python -m scm_ds.debug_adaptive_target \
        --n-processes 3 --p 3 --n-samples 3 --adaptive-mode linear

The output is purely informational (stdout). No files are written.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

# Make repo root importable when invoked as a script
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.processes_config import _build_st_processes  # noqa: E402
from scm_ds.compute_reliability import (  # noqa: E402
    ReliabilityFunction,
    _apply_adaptive_mode,
)
from scm_ds.datasets_st import STConfig, build_st_scm  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
# Formatting helpers
# ═══════════════════════════════════════════════════════════════════════

def _fmt(x, decimals: int = 4, max_rows: int = 3) -> str:
    """Pretty-print a scalar / 1D / 2D tensor-or-array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if np.isscalar(x) or (hasattr(x, "ndim") and x.ndim == 0):
        return f"{float(x):+.{decimals}f}"
    arr = np.asarray(x)
    if arr.ndim == 1:
        vals = ", ".join(f"{float(v):+.{decimals}f}" for v in arr)
        return f"[{vals}]"
    # 2D — one row per line
    lines = []
    for i in range(min(arr.shape[0], max_rows)):
        row = ", ".join(f"{float(v):+.{decimals}f}" for v in arr[i])
        lines.append(f"        [{row}]")
    if arr.shape[0] > max_rows:
        lines.append(f"        ... ({arr.shape[0] - max_rows} more rows)")
    return "\n" + "\n".join(lines)


def _banner(title: str, char: str = "═", width: int = 72) -> str:
    return f"\n{char * width}\n  {title}\n{char * width}"


# ═══════════════════════════════════════════════════════════════════════
# Build trajectory from a real ST SCM
# ═══════════════════════════════════════════════════════════════════════

def _build_trajectory(st_config: Dict, n_samples: int, seed: int) -> Tuple[Dict, object]:
    """Generate one SCM and sample ``n_samples`` trajectories for every process.

    In the real pipeline each process has its own SCM instance but they share
    the same topology. For debug purposes we reuse the same SCM across
    processes and change the seed so the environmental noise differs per
    process — this is enough to exercise the adaptive-target code path.

    Returns
    -------
    trajectory : dict
        {process_name: {"inputs": tensor, "outputs_mean": tensor}}
    scm : SCMDataset
        Reference SCM (for introspection only).
    """
    cfg = STConfig(**st_config["st_params"])
    scm = build_st_scm(cfg)
    n_procs = st_config["n_processes"]

    trajectory: Dict[str, Dict[str, torch.Tensor]] = {}
    for i in range(1, n_procs + 1):
        df = scm.sample(n=n_samples, seed=seed + i)
        inputs = df[scm.input_labels].values.astype(np.float32)
        outputs = df[scm.target_labels].values.astype(np.float32)
        trajectory[f"st_{i}"] = {
            "inputs": torch.tensor(inputs, dtype=torch.float32),
            "outputs_mean": torch.tensor(outputs, dtype=torch.float32),
        }
    return trajectory, scm


# ═══════════════════════════════════════════════════════════════════════
# Translate _build_st_processes output → ReliabilityFunction configs
# ═══════════════════════════════════════════════════════════════════════

def _translate_configs(process_list: List[Dict]) -> Tuple[Dict[str, Dict], List[str]]:
    """Convert ``surrogate_*`` keys to the plain ``*`` keys consumed by
    ``ReliabilityFunction`` (mirroring ``generate_dataset.py`` line 447)."""
    configs: Dict[str, Dict] = {}
    order: List[str] = []
    for pc in process_list:
        name = pc["name"]
        order.append(name)
        cfg = {
            "base_target": pc.get("surrogate_target", 0.0),
            "scale": pc.get("surrogate_scale", 1.0),
            "weight": pc.get("surrogate_weight", 1.0),
        }
        if "surrogate_adaptive_coefficients" in pc:
            cfg["adaptive_coefficients"] = pc["surrogate_adaptive_coefficients"]
            cfg["adaptive_baselines"] = pc["surrogate_adaptive_baselines"]
        for src, dst in [
            ("surrogate_adaptive_mode", "adaptive_mode"),
            ("surrogate_adaptive_coefficients2", "adaptive_coefficients2"),
            ("surrogate_adaptive_power", "adaptive_power"),
            ("surrogate_adaptive_band", "adaptive_band"),
            ("surrogate_adaptive_sharpness", "adaptive_sharpness"),
            ("surrogate_adaptive_max_shift", "adaptive_max_shift"),
        ]:
            if src in pc:
                cfg[dst] = pc[src]
        configs[name] = cfg
    return configs, order


# ═══════════════════════════════════════════════════════════════════════
# Per-process step-by-step breakdown
# ═══════════════════════════════════════════════════════════════════════

def debug_process(process_name: str,
                  outputs: Dict[str, torch.Tensor],
                  config: Dict,
                  max_rows: int = 3) -> Dict:
    """Replay ``_compute_adaptive_target`` for one process, printing every step.

    Also computes the LEGACY shift (scalar mean of upstream output vs.
    scalar baseline) so you can compare numerically.
    """
    base_target = config.get("base_target", 0.0)
    if isinstance(base_target, list):
        base_target_t = torch.tensor(base_target, dtype=torch.float32)
    else:
        base_target_t = torch.tensor([base_target], dtype=torch.float32)
    output_dim = base_target_t.numel()
    target_scalar_start = float(base_target_t.mean().item())

    adaptive_coeffs = config.get("adaptive_coefficients", {})
    adaptive_baselines = config.get("adaptive_baselines", {})
    mode = config.get("adaptive_mode", "linear")

    print(_banner(f"Process: {process_name}   (output_dim = {output_dim})"))
    print(f"  base_target (per-dim)  : {_fmt(base_target_t)}")

    if not adaptive_coeffs:
        print("  (no adaptive_coefficients — target = base_target, no shift)")
        return {
            "target_new": base_target_t.unsqueeze(0),
            "shift_new": torch.zeros(1),
            "shift_legacy": torch.zeros(1),
        }

    print(f"  adaptive_coefficients  : {adaptive_coeffs}")
    print("  adaptive_baselines     :")
    for k, v in adaptive_baselines.items():
        print(f"      {k} → {v}")
    print(f"  adaptive_mode          : {mode}")
    print(f"  scalar target start    : mean(base_target) = {target_scalar_start:+.4f}")

    # Per-upstream accumulators
    shift_new_total = None   # (batch,)
    shift_legacy_total = None  # (batch,)

    for upstream, coeff in adaptive_coeffs.items():
        if upstream not in outputs:
            print(f"\n  ⚠ upstream {upstream} not in outputs — skipped")
            continue

        upstream_out = outputs[upstream]  # (batch, m_up)
        batch = upstream_out.shape[0]
        m_up = upstream_out.shape[1] if upstream_out.dim() > 1 else 1

        # ── NEW (per-dim) ────────────────────────────────────────────
        baseline_raw = adaptive_baselines.get(upstream, 0.0)
        if isinstance(baseline_raw, (list, tuple)):
            baseline_t = torch.tensor(baseline_raw, dtype=torch.float32)
        else:
            baseline_t = baseline_raw
        delta_new = upstream_out - baseline_t  # (batch, m_up)
        adj_new = _apply_adaptive_mode(delta_new, coeff, mode, {})
        if isinstance(adj_new, torch.Tensor) and adj_new.dim() > 1:
            shift_new = adj_new.mean(dim=-1)  # (batch,)
        else:
            shift_new = adj_new

        # ── LEGACY (scalar mean) ─────────────────────────────────────
        # Emulate the old code exactly: upstream.mean(dim=-1), baseline =
        # mean of CURRENT process's calibrated_targets (the old buggy choice).
        legacy_baseline = target_scalar_start
        if upstream_out.dim() > 1:
            upstream_out_legacy = upstream_out.mean(dim=-1)
        else:
            upstream_out_legacy = upstream_out
        delta_legacy = upstream_out_legacy - legacy_baseline
        shift_legacy = _apply_adaptive_mode(delta_legacy, coeff, mode, {})

        # ── Print ───────────────────────────────────────────────────
        print(f"\n  ── upstream {upstream}  (coeff={coeff}, m_up={m_up}) ──")
        baseline_pretty = (
            _fmt(baseline_t) if isinstance(baseline_t, torch.Tensor)
            else f"{float(baseline_t):+.4f} (scalar)"
        )
        print(f"      NEW baseline (per-dim)           : {baseline_pretty}")
        print(f"      upstream_out  [{batch}×{m_up}] :{_fmt(upstream_out, max_rows=max_rows)}")
        print(f"      NEW   delta = out - baseline   :{_fmt(delta_new, max_rows=max_rows)}")
        if isinstance(adj_new, torch.Tensor) and adj_new.dim() > 1:
            print(f"      NEW   adjustment (f(delta)*c)  :{_fmt(adj_new, max_rows=max_rows)}")
        else:
            print(f"      NEW   adjustment (f(delta)*c)  : {_fmt(adj_new)}")
        print(f"      NEW   shift_j = mean(dim=-1)    : {_fmt(shift_new)}")
        print(f"      ···   (LEGACY) baseline scalar : {legacy_baseline:+.4f}")
        print(f"      ···   (LEGACY) upstream.mean   : {_fmt(upstream_out_legacy)}")
        print(f"      ···   (LEGACY) delta           : {_fmt(delta_legacy)}")
        print(f"      ···   (LEGACY) shift_j         : {_fmt(shift_legacy)}")

        diff_j = shift_new - shift_legacy
        print(f"      Δ (new - legacy) for this upstream : {_fmt(diff_j)}")

        shift_new_total = shift_new if shift_new_total is None else shift_new_total + shift_new
        shift_legacy_total = shift_legacy if shift_legacy_total is None else shift_legacy_total + shift_legacy

    # ── Totals ──────────────────────────────────────────────────────
    target_new_scalar = target_scalar_start + shift_new_total  # (batch,)
    target_legacy = target_scalar_start + shift_legacy_total  # (batch,)
    print("\n  ──── totals ────")
    print(f"  Σ NEW    shift per sample   : {_fmt(shift_new_total)}")
    print(f"  Σ LEGACY shift per sample   : {_fmt(shift_legacy_total)}")
    print(f"  Δ (new - legacy) per sample : {_fmt(shift_new_total - shift_legacy_total)}")
    print(f"  NEW adaptive target (batch,): {_fmt(target_new_scalar)}")
    print(f"     → broadcasts to (batch × {output_dim}) — same scalar along output dims")

    return {
        "target_new": target_new_scalar.unsqueeze(-1),  # (batch, 1)
        "shift_new": shift_new_total,
        "shift_legacy": shift_legacy_total,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-processes", type=int, default=3)
    parser.add_argument("--p", type=int, default=3, help="output_dim per process")
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--adaptive-coeff", type=float, default=0.3)
    parser.add_argument(
        "--adaptive-mode",
        type=str,
        default="linear",
        choices=["linear", "polynomial", "power", "softplus", "deadband", "tanh"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-inputs", type=int, default=4, help="ST param n")
    parser.add_argument("--m-stages", type=int, default=1, help="ST param m")
    parser.add_argument("--rho", type=float, default=0.1, help="ST noise intensity")
    args = parser.parse_args()

    st_dataset_config = {
        "n_processes": args.n_processes,
        "n_samples": args.n_samples,
        "adaptive_coeff": args.adaptive_coeff,
        "adaptive_mode": args.adaptive_mode,
        "st_params": {
            "n": args.n_inputs,
            "m": args.m_stages,
            "p": args.p,
            "me": 1,
            "env_mode": "A",
            "rho": args.rho,
            "cal_n": 500,  # keep calibration fast
        },
        "uncertainty_predictor": {},
    }

    print(_banner("Debug per-dimension adaptive target   (NEW vs LEGACY)", char="#"))
    print(f"  n_processes    = {args.n_processes}")
    print(f"  p (output_dim) = {args.p}")
    print(f"  n_samples      = {args.n_samples}")
    print(f"  adaptive_coeff = {args.adaptive_coeff}")
    print(f"  adaptive_mode  = {args.adaptive_mode}")
    print(f"  seed           = {args.seed}")
    print(f"  st_params      = n={args.n_inputs}, m={args.m_stages}, rho={args.rho}")

    # 1. Build process configs (uses the fixed _build_st_processes)
    print(_banner("Step 1 — build process configs (via _build_st_processes)", char="─"))
    process_list = _build_st_processes(st_dataset_config)
    configs, order = _translate_configs(process_list)
    for name in order:
        c = configs[name]
        base_fmt = c["base_target"] if isinstance(c["base_target"], list) else [c["base_target"]]
        print(f"  {name}: base_target={[round(v, 4) for v in base_fmt]}  "
              f"has_adaptive={bool(c.get('adaptive_coefficients'))}")
        if c.get("adaptive_baselines"):
            for k, v in c["adaptive_baselines"].items():
                kind = "list(per-dim)" if isinstance(v, (list, tuple)) else "scalar"
                print(f"      baseline[{k}] = {v}  ← {kind}")

    # 2. Sample trajectories from an ST SCM
    print(_banner("Step 2 — sample trajectories from ST SCM", char="─"))
    trajectory, scm = _build_trajectory(st_dataset_config, args.n_samples, args.seed)
    print(f"  scm.name            : {scm.meta.get('name', '?')}")
    print(f"  scm.input_labels    : {scm.input_labels}")
    print(f"  scm.target_labels   : {scm.target_labels}")
    for name in order:
        out = trajectory[name]["outputs_mean"]
        print(f"  {name}.outputs_mean  shape={tuple(out.shape)}")
        print(f"                         values ={_fmt(out, max_rows=args.n_samples)}")

    # 3. Per-process adaptive target breakdown
    print(_banner("Step 3 — per-process adaptive-target breakdown", char="─"))
    # Match the shape convention expected by _compute_adaptive_target:
    # (batch, output_dim), even when output_dim == 1.
    outputs: Dict[str, torch.Tensor] = {}
    for name in order:
        out = trajectory[name]["outputs_mean"]
        if out.dim() == 1:
            out = out.unsqueeze(-1)
        outputs[name] = out

    for name in order:
        debug_process(name, outputs, configs[name], max_rows=args.n_samples)

    # 4. Full F computation via ReliabilityFunction (sanity check)
    print(_banner("Step 4 — end-to-end F via ReliabilityFunction", char="─"))
    rfunc = ReliabilityFunction(
        process_configs=configs, process_order=order, device="cpu"
    )
    F, Q = rfunc.compute_reliability(
        trajectory, return_quality_scores=True, use_sampled_outputs=False
    )
    print(f"  F per sample : {_fmt(F)}")
    for name in order:
        print(f"    Q[{name}] : {_fmt(Q[name])}")
    print()


if __name__ == "__main__":
    main()
