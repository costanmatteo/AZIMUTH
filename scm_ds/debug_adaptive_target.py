"""Debug helper for the Shekel reliability function on an ST dataset.

Builds a small ST SCM, samples trajectories, calibrates Shekel widths, and
prints a step-by-step breakdown of the full dataset generation pipeline so
that every term of the Shekel formula

    F(o) = 1 / (1 + Σ_t Σ_k  d_t^k · (o_t^k - ζ*_t^k)²)

is visible and reproducible by hand.

Usage (from repo root)::

    python -m scm_ds.debug_adaptive_target \
        --n-processes 3 --p 2 --n-samples 4 \
        --shekel-sharpness 1.0 --adaptive-coeff 0.3

The output is purely informational (stdout). No files are written.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

# Make repo root importable when invoked as a script
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

def _fmt(x, decimals: int = 4, max_rows: int = 6) -> str:
    """Pretty-print a scalar / 1D / 2D tensor-or-array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if np.isscalar(x) or (hasattr(x, "ndim") and x.ndim == 0):
        return f"{float(x):+.{decimals}f}"
    arr = np.asarray(x)
    if arr.ndim == 1:
        vals = ", ".join(f"{float(v):+.{decimals}f}" for v in arr)
        return f"[{vals}]"
    lines = []
    for i in range(min(arr.shape[0], max_rows)):
        row = ", ".join(f"{float(v):+.{decimals}f}" for v in arr[i])
        lines.append(f"          [{row}]")
    if arr.shape[0] > max_rows:
        lines.append(f"          ... ({arr.shape[0] - max_rows} more rows)")
    return "\n" + "\n".join(lines)


def _banner(title: str, char: str = "═", width: int = 76) -> str:
    return f"\n{char * width}\n  {title}\n{char * width}"


# ═══════════════════════════════════════════════════════════════════════
# ST SCM construction and sampling
# ═══════════════════════════════════════════════════════════════════════

def _build_trajectory(st_config: Dict, n_samples: int, seed: int) -> Tuple[Dict, object]:
    """Generate one ST SCM and sample ``n_samples`` trajectories per process.

    In the real pipeline each process is an independent SCM. For debug
    purposes we reuse the same topology and change the seed across
    processes so the environmental/process noise differs per process —
    this is enough to exercise both the adaptive-target code path and the
    Shekel width calibration.

    Returns
    -------
    trajectory : dict
        {process_name: {"inputs": tensor, "outputs_mean": tensor,
                        "outputs_sampled": tensor}}
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
        out_t = torch.tensor(outputs, dtype=torch.float32)
        if out_t.dim() == 1:
            out_t = out_t.unsqueeze(-1)
        trajectory[f"st_{i}"] = {
            "inputs": torch.tensor(inputs, dtype=torch.float32),
            "outputs_mean": out_t,
            "outputs_sampled": out_t,
        }
    return trajectory, scm


# ═══════════════════════════════════════════════════════════════════════
# Translate _build_st_processes output → ReliabilityFunction configs
# ═══════════════════════════════════════════════════════════════════════

def _translate_configs(process_list: List[Dict]) -> Tuple[Dict[str, Dict], List[str]]:
    """Convert ``surrogate_*`` keys to the plain ``*`` keys consumed by
    ``ReliabilityFunction`` (mirrors ``generate_dataset.py``)."""
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
# Manual adaptive-target reconstruction (what ReliabilityFunction does)
# ═══════════════════════════════════════════════════════════════════════

def _manual_adaptive_target(process_name: str,
                            outputs: Dict[str, torch.Tensor],
                            config: Dict,
                            max_rows: int) -> torch.Tensor:
    """Replay ``_compute_adaptive_target`` for one process, printing every step.

    Returns the target as a ``(batch, 1)`` tensor (scalar broadcast along
    output dims) or ``(output_dim,)`` when no upstreams contribute.
    """
    base_target = config.get("base_target", 0.0)
    if isinstance(base_target, list):
        base_target_t = torch.tensor(base_target, dtype=torch.float32)
    else:
        base_target_t = torch.tensor([base_target], dtype=torch.float32)
    output_dim = base_target_t.numel()

    adaptive_coeffs = config.get("adaptive_coefficients", {})
    adaptive_baselines = config.get("adaptive_baselines", {})
    mode = config.get("adaptive_mode", "linear")

    print(f"\n  ▸ {process_name}   (output_dim = {output_dim})")
    print(f"      base_target (per-dim) τ₀ : {_fmt(base_target_t)}")

    if not adaptive_coeffs:
        print("      no adaptive_coefficients → ζ* = base_target (per-dim, no shift)")
        return base_target_t

    target_scalar_start = float(base_target_t.mean().item())
    print(f"      adaptive_mode            : {mode}")
    print(f"      adaptive_coefficients    : {adaptive_coeffs}")
    print( "      adaptive_baselines       :")
    for k, v in adaptive_baselines.items():
        print(f"          {k} → {v}")
    print(f"      scalar start  mean(τ₀)   : {target_scalar_start:+.4f}")

    target_scalar = target_scalar_start
    shift_total = None  # (batch,)

    for upstream, coeff in adaptive_coeffs.items():
        if upstream not in outputs:
            print(f"      ⚠ upstream {upstream} not in outputs — skipped")
            continue

        upstream_out = outputs[upstream]  # (batch, m_up)
        batch = upstream_out.shape[0]
        m_up = upstream_out.shape[1] if upstream_out.dim() > 1 else 1

        baseline_raw = adaptive_baselines.get(upstream, 0.0)
        if isinstance(baseline_raw, (list, tuple)):
            baseline_t = torch.tensor(baseline_raw, dtype=torch.float32)
        else:
            baseline_t = baseline_raw

        delta = upstream_out - baseline_t  # (batch, m_up)
        adj = _apply_adaptive_mode(delta, coeff, mode, {})
        if isinstance(adj, torch.Tensor) and adj.dim() > 1:
            shift_j = adj.mean(dim=-1)  # (batch,)
        else:
            shift_j = adj

        print(f"      ── upstream {upstream}  (coeff={coeff}, m_up={m_up})")
        baseline_pretty = (
            _fmt(baseline_t) if isinstance(baseline_t, torch.Tensor)
            else f"{float(baseline_t):+.4f} (scalar)"
        )
        print(f"          baseline ζ̄_j          : {baseline_pretty}")
        print(f"          upstream_out [{batch}×{m_up}] :{_fmt(upstream_out, max_rows=max_rows)}")
        print(f"          Δ = out - baseline     :{_fmt(delta, max_rows=max_rows)}")
        if isinstance(adj, torch.Tensor) and adj.dim() > 1:
            print(f"          f(Δ)·c (per-dim adj)  :{_fmt(adj, max_rows=max_rows)}")
        else:
            print(f"          f(Δ)·c (adj)          : {_fmt(adj)}")
        print(f"          shift_j = mean(adj,-1) : {_fmt(shift_j)}")

        shift_total = shift_j if shift_total is None else shift_total + shift_j

    target = torch.as_tensor(target_scalar, dtype=torch.float32) + shift_total  # (batch,)
    target = target.unsqueeze(-1)  # (batch, 1)

    print(f"      Σ shift per sample       : {_fmt(shift_total)}")
    print(f"      ζ* = mean(τ₀) + Σ shift  : {_fmt(target.squeeze(-1))}")
    print(f"          (broadcasts to (batch × {output_dim}))")
    return target


# ═══════════════════════════════════════════════════════════════════════
# Shekel computation — step-by-step
# ═══════════════════════════════════════════════════════════════════════

def _shekel_breakdown(order: List[str],
                      configs: Dict[str, Dict],
                      outputs: Dict[str, torch.Tensor],
                      widths: Dict[str, torch.Tensor],
                      max_rows: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute Shekel F by hand, printing every intermediate tensor.

        per-process partial sum :   S_t = Σ_k d_t^k · (o_t^k - ζ*_t^k)²
        global denominator      :   D   = Σ_t S_t
        reliability             :   F   = 1 / (1 + D)
    """
    batch_size = next(iter(outputs.values())).shape[0]
    total_sum = torch.zeros(batch_size)
    partial_sums: Dict[str, torch.Tensor] = {}

    for name in order:
        if name not in outputs:
            continue
        out = outputs[name]              # (batch, K)
        cfg = configs[name]
        d = widths[name]                 # (K,)

        print(f"\n  ▸ {name}")
        target = _manual_adaptive_target(name, outputs, cfg, max_rows)
        # Broadcast ζ* to (batch, K) explicitly for display
        tgt_full = target.expand_as(out) if target.dim() >= 1 else target
        diff = out - tgt_full
        diff_sq = diff ** 2
        weighted = d * diff_sq           # (batch, K) broadcast along k
        proc_sum = weighted.sum(dim=-1)  # (batch,)

        print(f"      o_t            [{out.shape[0]}×{out.shape[1]}]:{_fmt(out, max_rows=max_rows)}")
        print(f"      ζ*_t           [broadcast]   :{_fmt(tgt_full, max_rows=max_rows)}")
        print(f"      (o - ζ*)       :{_fmt(diff, max_rows=max_rows)}")
        print(f"      (o - ζ*)²      :{_fmt(diff_sq, max_rows=max_rows)}")
        print(f"      d_t            (per-dim)     : {_fmt(d)}")
        print(f"      d·(o-ζ*)²      :{_fmt(weighted, max_rows=max_rows)}")
        print(f"      S_t = Σ_k …   (batch,)     : {_fmt(proc_sum)}")

        total_sum = total_sum + proc_sum
        partial_sums[name] = proc_sum

    F = 1.0 / (1.0 + total_sum)

    print(_banner("Shekel — global aggregation", char="─"))
    print(f"  Σ_t S_t (denominator - 1) : {_fmt(total_sum)}")
    print(f"  1 + Σ_t S_t               : {_fmt(1.0 + total_sum)}")
    print(f"  F = 1 / (1 + Σ_t S_t)     : {_fmt(F)}")
    return F, partial_sums


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-processes", type=int, default=3)
    parser.add_argument("--p", type=int, default=2, help="output_dim per process")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--adaptive-coeff", type=float, default=0.3)
    parser.add_argument(
        "--adaptive-mode",
        type=str,
        default="linear",
        choices=["linear", "polynomial", "power", "softplus", "deadband", "tanh"],
    )
    parser.add_argument("--shekel-sharpness", type=float, default=1.0,
                        help="Global sharpness s in d_t^k = s / Var[o_t^k]")
    parser.add_argument("--n-calibration", type=int, default=256,
                        help="Samples used for Shekel width calibration")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-inputs", type=int, default=4, help="ST param n")
    parser.add_argument("--m-stages", type=int, default=2, help="ST param m")
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

    print(_banner("DEBUG — Shekel reliability on an ST dataset", char="#"))
    print(f"  n_processes       = {args.n_processes}")
    print(f"  p (output_dim)    = {args.p}")
    print(f"  n_samples (batch) = {args.n_samples}")
    print(f"  adaptive_coeff    = {args.adaptive_coeff}")
    print(f"  adaptive_mode     = {args.adaptive_mode}")
    print(f"  shekel_sharpness  = {args.shekel_sharpness}")
    print(f"  n_calibration     = {args.n_calibration}")
    print(f"  seed              = {args.seed}")
    print(f"  st_params         = n={args.n_inputs}, m={args.m_stages}, rho={args.rho}")

    # ── STEP 1 — Build a reference ST SCM ────────────────────────────
    print(_banner("Step 1 — build the reference ST SCM (build_st_scm)", char="─"))
    cfg = STConfig(**st_dataset_config["st_params"])
    ref_scm = build_st_scm(cfg)
    print(f"  scm.name            : {ref_scm.meta.get('name', '?')}")
    print(f"  scm.description     : {ref_scm.meta.get('description', '')}")
    print(f"  input_labels        : {ref_scm.input_labels}")
    print(f"  target_labels       : {ref_scm.target_labels}")
    print(f"  structural noise    : {ref_scm.structural_noise_vars}")
    print(f"  process noise       : {ref_scm.process_noise_vars}")
    print( "  per-node calibration (τ₀, scale):")
    for node, cc in ref_scm.process_configs.items():
        print(f"      {node:8s} base_target={cc['base_target']:+.4f}  "
              f"scale={cc['scale']:.4f}  weight={cc.get('weight', 1.0)}")

    # ── STEP 2 — Build the process-chain configuration ───────────────
    print(_banner("Step 2 — build process chain (_build_st_processes)", char="─"))
    process_list = _build_st_processes(st_dataset_config)
    configs, order = _translate_configs(process_list)
    for name in order:
        c = configs[name]
        base_fmt = c["base_target"] if isinstance(c["base_target"], list) else [c["base_target"]]
        print(f"  {name}: base_target={[round(v, 4) for v in base_fmt]}  "
              f"scale={c.get('scale')}  weight={c.get('weight')}  "
              f"has_adaptive={bool(c.get('adaptive_coefficients'))}")
        if c.get("adaptive_coefficients"):
            for up, coef in c["adaptive_coefficients"].items():
                bl = c.get("adaptive_baselines", {}).get(up)
                print(f"      upstream {up}: coeff={coef:+.4f}  baseline={bl}")

    # ── STEP 3 — Sample trajectories ─────────────────────────────────
    print(_banner("Step 3 — sample trajectories from the ST SCM", char="─"))
    trajectory, scm = _build_trajectory(st_dataset_config, args.n_samples, args.seed)
    for name in order:
        out = trajectory[name]["outputs_mean"]
        inp = trajectory[name]["inputs"]
        print(f"  {name}.inputs        shape={tuple(inp.shape)}")
        print(f"  {name}.outputs_mean  shape={tuple(out.shape)}  "
              f"values ={_fmt(out, max_rows=args.n_samples)}")

    # Normalise outputs to (batch, K) for uniform handling downstream.
    outputs: Dict[str, torch.Tensor] = {}
    for name in order:
        out = trajectory[name]["outputs_mean"]
        if out.dim() == 1:
            out = out.unsqueeze(-1)
        outputs[name] = out

    # ── STEP 4 — Shekel width calibration ────────────────────────────
    print(_banner("Step 4 — Shekel width calibration  d_t^k = s / Var[o_t^k]",
                  char="─"))
    cal_trajectory, _ = _build_trajectory(
        st_dataset_config, args.n_calibration, args.seed + 10_000
    )
    cal_list = [
        {name: {"outputs_sampled": cal_trajectory[name]["outputs_mean"][i:i + 1]}
         for name in order}
        for i in range(args.n_calibration)
    ]
    rfunc = ReliabilityFunction(
        process_configs=configs,
        process_order=order,
        device="cpu",
        reliability_formula="shekel",
        shekel_sharpness=args.shekel_sharpness,
    )
    rfunc.calibrate_shekel_widths(cal_list)
    print(f"  calibration samples  : {args.n_calibration}")
    print(f"  sharpness s          : {args.shekel_sharpness}")
    for name in order:
        stacked = cal_trajectory[name]["outputs_mean"]        # (N_cal, K)
        var = torch.clamp(stacked.var(dim=0), min=1e-8)       # (K,)
        expected_d = args.shekel_sharpness / var
        d_stored = rfunc._shekel_widths[name]
        print(f"  {name}:")
        print(f"      Var[o_t] (per-dim)        : {_fmt(var)}")
        print(f"      d_t = s / Var[o_t]        : {_fmt(expected_d)}")
        print(f"      stored in rfunc           : {_fmt(d_stored)}")

    # ── STEP 5 — Per-process Shekel breakdown ────────────────────────
    print(_banner("Step 5 — manual Shekel computation (per process, per sample)",
                  char="─"))
    F_manual, S_manual = _shekel_breakdown(
        order, configs, outputs, rfunc._shekel_widths, max_rows=args.n_samples
    )

    # ── STEP 6 — Reference computation via ReliabilityFunction ───────
    print(_banner("Step 6 — sanity check via ReliabilityFunction", char="─"))
    F_ref, S_ref = rfunc.compute_reliability(
        trajectory, return_quality_scores=True, use_sampled_outputs=False
    )
    print(f"  F (manual)    : {_fmt(F_manual)}")
    print(f"  F (rfunc)     : {_fmt(F_ref)}")
    max_abs = float((F_manual - F_ref).abs().max().item())
    print(f"  max |Δ|       : {max_abs:.2e}   "
          f"{'OK' if max_abs < 1e-5 else '⚠ MISMATCH'}")
    print("  per-process partial sums S_t:")
    for name in order:
        print(f"      {name}:  manual={_fmt(S_manual[name])}   rfunc={_fmt(S_ref[name])}")

    # ── STEP 7 — Comparison with Gaussian formula on same trajectory ─
    print(_banner("Step 7 — cross-check Gaussian F on the same trajectory",
                  char="─"))
    rfunc_g = ReliabilityFunction(
        process_configs=configs,
        process_order=order,
        device="cpu",
        reliability_formula="gaussian",
    )
    F_g, Q_g = rfunc_g.compute_reliability(
        trajectory, return_quality_scores=True, use_sampled_outputs=False
    )
    print(f"  F (gaussian)  : {_fmt(F_g)}")
    for name in order:
        print(f"      Q[{name}]   : {_fmt(Q_g[name])}")
    print()


if __name__ == "__main__":
    main()
