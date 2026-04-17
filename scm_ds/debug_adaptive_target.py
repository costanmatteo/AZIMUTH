"""Debug helper for the reliability function on an ST dataset.

Builds a small ST SCM, samples trajectories, and prints a step-by-step
breakdown of the full dataset generation pipeline so that every term of the
selected reliability formula is visible and reproducible by hand.

Two formulas are supported (selected via ``--reliability-formula`` or, by
default, via ``RELIABILITY_FORMULA`` in ``configs/processes_config.py``):

* **gaussian** — weighted average of per-process Gaussian quality scores::

      Q_i = mean_k exp(-(o_i^k - τ_i^k)² / s_i^k)
      F   = Σ_i w_i · Q_i / Σ_i w_i

* **shekel** — global Shekel function (requires width calibration)::

      F(o) = 1 / (1 + Σ_t Σ_k  d_t^k · (o_t^k - ζ*_t^k)²)

The steps printed and the tensors shown adapt to the chosen method.

Usage (from repo root)::

    # Gaussian (default from config)
    python -m scm_ds.debug_adaptive_target \
        --n-processes 3 --p 2 --n-samples 4 --adaptive-coeff 0.3

    # Shekel
    python -m scm_ds.debug_adaptive_target --reliability-formula shekel \
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

from configs.processes_config import (  # noqa: E402
    RELIABILITY_FORMULA as _CFG_RELIABILITY_FORMULA,
    SHEKEL_SHARPNESS as _CFG_SHEKEL_SHARPNESS,
    _build_st_processes,
)
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
        env_vars = scm.structural_noise_vars or []
        env = df[env_vars].values.astype(np.float32) if env_vars else np.zeros(
            (n_samples, 0), dtype=np.float32
        )
        # Stage intermediate variables (S_k or S_k_j) — everything that's in
        # the SCM context but isn't an input, env, output, or noise node.
        noise_set = set(env_vars) | set(scm.process_noise_vars or [])
        stage_names = [
            c for c in df.columns
            if c not in scm.input_labels
            and c not in scm.target_labels
            and c not in noise_set
        ]
        stages = df[stage_names].values.astype(np.float32) if stage_names else None
        out_t = torch.tensor(outputs, dtype=torch.float32)
        if out_t.dim() == 1:
            out_t = out_t.unsqueeze(-1)
        trajectory[f"st_{i}"] = {
            "inputs": torch.tensor(inputs, dtype=torch.float32),
            "env": torch.tensor(env, dtype=torch.float32),
            "env_labels": env_vars,
            "stages": None if stages is None else torch.tensor(stages, dtype=torch.float32),
            "stage_labels": stage_names,
            "input_labels": list(scm.input_labels),
            "outputs_mean": out_t,
            "outputs_sampled": out_t,
            "output_labels": list(scm.target_labels),
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
                            adaptive_targets: Dict[str, torch.Tensor],
                            config: Dict,
                            max_rows: int) -> torch.Tensor:
    """Replay ``_compute_adaptive_target`` (chain variant) for one process.

    Chain variant: the baseline for each upstream j is τ_j (its own adaptive
    target, from ``adaptive_targets``), not the static ζ⁽⁰⁾_j. The
    ``adaptive_baselines`` field in config is ignored.

    Returns a ``(batch, output_dim_i)`` tensor when adaptive coeffs are
    present, or ``(output_dim_i,)`` when the process has no upstream
    contributions (first process).
    """
    base_target = config.get("base_target", 0.0)
    if isinstance(base_target, list):
        base_target_t = torch.tensor(base_target, dtype=torch.float32)
    else:
        base_target_t = torch.tensor([base_target], dtype=torch.float32)
    output_dim = base_target_t.numel()

    adaptive_coeffs = config.get("adaptive_coefficients", {})
    mode = config.get("adaptive_mode", "linear")

    print(f"\n  ▸ {process_name}   (output_dim = {output_dim})")
    print(f"      base_target (per-dim) ζ⁽⁰⁾ : {_fmt(base_target_t)}")

    if not adaptive_coeffs:
        print("      no adaptive_coefficients → τ = base_target (per-dim, no shift)")
        return base_target_t

    print(f"      adaptive_mode              : {mode}")
    print(f"      adaptive_coefficients      : {adaptive_coeffs}")
    print( "      (chain variant: adaptive_baselines in config is IGNORED;")
    print( "       per-upstream baseline is τ_j from adaptive_targets)")

    target = base_target_t.unsqueeze(0)  # (1, output_dim_i) — broadcasts

    for upstream, coeff in adaptive_coeffs.items():
        if upstream not in outputs:
            print(f"      ⚠ upstream {upstream} not in outputs — skipped")
            continue
        if upstream not in adaptive_targets:
            print(f"      ⚠ upstream {upstream} has no τ_j yet (order bug?) — skipped")
            continue

        upstream_out = outputs[upstream]            # (batch, m_up)
        batch = upstream_out.shape[0]
        m_up = upstream_out.shape[1] if upstream_out.dim() > 1 else 1

        tau_j = adaptive_targets[upstream]          # (batch, m_up) pre-normalised

        delta = upstream_out - tau_j                # (batch, m_up) per-dim
        adj = _apply_adaptive_mode(delta, coeff, mode, {})
        shift_j = adj.sum(dim=-1, keepdim=True)  # (batch, 1)

        print(f"      ── upstream {upstream}  (coeff={coeff}, m_up={m_up})")
        print(f"          τ_j (chain baseline)    :{_fmt(tau_j, max_rows=max_rows)}")
        print(f"          upstream_out [{batch}×{m_up}] :{_fmt(upstream_out, max_rows=max_rows)}")
        print(f"          Δ = Y_j - τ_j (per-dim) :{_fmt(delta, max_rows=max_rows)}")
        print(f"          f(Δ)·c (per-dim adj)    :{_fmt(adj, max_rows=max_rows)}")
        print(f"          shift_j = sum(adj,-1)  :{_fmt(shift_j.squeeze(-1), max_rows=max_rows)}")

        target = target + shift_j                   # (batch, output_dim_i) via broadcast

    print(f"      τ_i = ζ⁽⁰⁾ + Σ shift (per-dim):{_fmt(target, max_rows=max_rows)}")
    print(f"          shape = (batch × {output_dim})")
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
    # chain variant: τ_j per-dim is propagated to downstream processes
    adaptive_targets: Dict[str, torch.Tensor] = {}

    for name in order:
        if name not in outputs:
            continue
        out = outputs[name]              # (batch, K)
        cfg = configs[name]
        d = widths[name]                 # (K,)

        print(f"\n  ▸ {name}")
        target = _manual_adaptive_target(name, outputs, adaptive_targets, cfg, max_rows)

        # Store τ_i normalised to (batch, output_dim_i) for downstream consumers
        if target.dim() == 1:
            target_stored = target.unsqueeze(0).expand(batch_size, -1)
        elif target.shape[0] == 1 and batch_size > 1:
            target_stored = target.expand(batch_size, -1)
        else:
            target_stored = target
        adaptive_targets[name] = target_stored

        # Broadcast τ* to (batch, K) explicitly for display
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
# Gaussian computation — step-by-step
# ═══════════════════════════════════════════════════════════════════════

def _gaussian_breakdown(order: List[str],
                        configs: Dict[str, Dict],
                        outputs: Dict[str, torch.Tensor],
                        max_rows: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute Gaussian F by hand, printing every intermediate tensor.

        per-dim quality     :  q_t^k  = exp(-(o_t^k - τ_t^k)² / s_t^k)
        per-process quality :  Q_t    = mean_k q_t^k
        weighted average    :  F      = Σ_t w_t · Q_t / Σ_t w_t
    """
    batch_size = next(iter(outputs.values())).shape[0]
    quality_scores: Dict[str, torch.Tensor] = {}
    weights: Dict[str, float] = {}
    # chain variant: τ_j per-dim is propagated to downstream processes
    adaptive_targets: Dict[str, torch.Tensor] = {}

    for name in order:
        if name not in outputs:
            continue
        out = outputs[name]              # (batch, K)
        cfg = configs[name]

        # Per-dim scales — broadcast to (K,)
        scales = cfg.get("scale", 1.0)
        if not isinstance(scales, list):
            scales = [scales]
        scale_t = torch.tensor(scales, dtype=torch.float32)  # (K,)

        # Per-process weight (reduce list → mean to match ReliabilityFunction)
        weight = cfg.get("weight", 1.0)
        if isinstance(weight, list):
            weight_val = sum(weight) / len(weight)
        else:
            weight_val = float(weight)
        weights[name] = weight_val

        print(f"\n  ▸ {name}")
        target = _manual_adaptive_target(name, outputs, adaptive_targets, cfg, max_rows)

        # Store τ_i normalised to (batch, output_dim_i) for downstream consumers
        if target.dim() == 1:
            target_stored = target.unsqueeze(0).expand(batch_size, -1)
        elif target.shape[0] == 1 and batch_size > 1:
            target_stored = target.expand(batch_size, -1)
        else:
            target_stored = target
        adaptive_targets[name] = target_stored

        # Broadcast τ to (batch, K) for display
        tgt_full = target.expand_as(out) if target.dim() >= 1 else target
        diff = out - tgt_full
        diff_sq = diff ** 2
        per_dim_q = torch.exp(-diff_sq / scale_t)   # (batch, K)
        quality = per_dim_q.mean(dim=-1)             # (batch,)

        print(f"      o_t            [{out.shape[0]}×{out.shape[1]}]:{_fmt(out, max_rows=max_rows)}")
        print(f"      τ_t            [broadcast]   :{_fmt(tgt_full, max_rows=max_rows)}")
        print(f"      (o - τ)        :{_fmt(diff, max_rows=max_rows)}")
        print(f"      (o - τ)²       :{_fmt(diff_sq, max_rows=max_rows)}")
        print(f"      scale s_t      (per-dim)     : {_fmt(scale_t)}")
        print(f"      q = exp(-Δ²/s) :{_fmt(per_dim_q, max_rows=max_rows)}")
        print(f"      Q_t = mean_k q (batch,)     : {_fmt(quality)}")
        print(f"      weight w_t    (scalar)      : {weight_val:+.4f}")
        print(f"      w_t · Q_t     (batch,)      : {_fmt(weight_val * quality)}")

        quality_scores[name] = quality

    total_weight = sum(weights.values())
    if total_weight > 0:
        weighted_sum = sum(weights[n] * quality_scores[n] for n in quality_scores)
        F = weighted_sum / total_weight
    else:
        F = torch.zeros(batch_size)
        weighted_sum = F.clone()

    print(_banner("Gaussian — weighted average aggregation", char="─"))
    print(f"  Σ_t w_t · Q_t             : {_fmt(weighted_sum)}")
    print(f"  Σ_t w_t                   : {total_weight:+.4f}")
    print(f"  F = Σ w·Q / Σ w           : {_fmt(F)}")
    return F, quality_scores


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
    parser.add_argument(
        "--reliability-formula",
        type=str,
        default=_CFG_RELIABILITY_FORMULA,
        choices=["gaussian", "shekel"],
        help=("Which reliability formula to debug. Defaults to "
              "RELIABILITY_FORMULA from configs/processes_config.py."),
    )
    parser.add_argument("--shekel-sharpness", type=float, default=_CFG_SHEKEL_SHARPNESS,
                        help="Global sharpness s in d_t^k = s / Var[o_t^k] (shekel only)")
    parser.add_argument("--n-calibration", type=int, default=256,
                        help="Samples used for Shekel width calibration (shekel only)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-inputs", type=int, default=4, help="ST param n")
    parser.add_argument("--m-stages", type=int, default=2, help="ST param m")
    parser.add_argument("--rho", type=float, default=0.1, help="ST noise intensity")
    args = parser.parse_args()

    formula = args.reliability_formula

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

    print(_banner(f"DEBUG — {formula.capitalize()} reliability on an ST dataset",
                  char="#"))
    print(f"  reliability_formula = {formula}")
    print(f"  n_processes       = {args.n_processes}")
    print(f"  p (output_dim)    = {args.p}")
    print(f"  n_samples (batch) = {args.n_samples}")
    print(f"  adaptive_coeff    = {args.adaptive_coeff}")
    print(f"  adaptive_mode     = {args.adaptive_mode}")
    if formula == "shekel":
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
                print(f"      upstream {up}: coeff={coef:+.4f}  "
                      f"adaptive_baselines={bl}  [IGNORED — chain variant]")

    # ── STEP 3 — Sample trajectories ─────────────────────────────────
    print(_banner("Step 3 — sample trajectories from the ST SCM", char="─"))
    trajectory, scm = _build_trajectory(st_dataset_config, args.n_samples, args.seed)
    x_lo, x_hi = cfg.x_domain
    e_lo, e_hi = cfg.e_domain
    print(f"  Inputs X_i     : {cfg.n} controllable, sampled Uniform({x_lo}, {x_hi})")
    print(f"  Environment E_j: {cfg.me} structural, sampled Uniform({e_lo}, {e_hi})"
          f"  (env_mode={cfg.env_mode}, α={cfg.alpha}, γ={cfg.gamma})")
    print(f"  Process noise  : Z_ln_* (lognormal mult) + Eps_add_* (gauss add) "
          f"+ Jump_* (Poisson-exp), scaled by ρ={cfg.rho}")
    print(f"  Stages S_k_j   : averaged sin((π/2)·x) across assigned inputs,"
          f" rescaled per-layer to [-2, 2]")
    for name in order:
        tr = trajectory[name]
        inp = tr["inputs"]
        env = tr["env"]
        stages = tr.get("stages")
        out = tr["outputs_mean"]
        print(f"\n  ── {name}")
        print(f"      controllable X  {tr['input_labels']}  "
              f"shape={tuple(inp.shape)}  values:{_fmt(inp, max_rows=args.n_samples)}")
        if env.numel() > 0:
            print(f"      environment E   {tr['env_labels']}  "
                  f"shape={tuple(env.shape)}  values:{_fmt(env, max_rows=args.n_samples)}")
        if stages is not None and stages.numel() > 0:
            print(f"      stages          {tr['stage_labels']}  "
                  f"shape={tuple(stages.shape)}  values:{_fmt(stages, max_rows=args.n_samples)}")
        print(f"      outputs Y       {tr['output_labels']}  "
              f"shape={tuple(out.shape)}  values:{_fmt(out, max_rows=args.n_samples)}")

    # Normalise outputs to (batch, K) for uniform handling downstream.
    outputs: Dict[str, torch.Tensor] = {}
    for name in order:
        out = trajectory[name]["outputs_mean"]
        if out.dim() == 1:
            out = out.unsqueeze(-1)
        outputs[name] = out

    # ── Branch on selected reliability formula ───────────────────────
    if formula == "shekel":
        # ── STEP 4 — Shekel width calibration ────────────────────────
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

        # ── STEP 5 — Per-process Shekel breakdown ────────────────────
        print(_banner("Step 5 — manual Shekel computation (per process, per sample)",
                      char="─"))
        F_manual, S_manual = _shekel_breakdown(
            order, configs, outputs, rfunc._shekel_widths, max_rows=args.n_samples
        )

        # ── STEP 6 — Reference computation via ReliabilityFunction ───
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

    else:  # formula == "gaussian"
        # ── STEP 4 — Per-process Gaussian breakdown ──────────────────
        print(_banner("Step 4 — manual Gaussian computation (per process, per sample)",
                      char="─"))
        F_manual, Q_manual = _gaussian_breakdown(
            order, configs, outputs, max_rows=args.n_samples
        )

        # ── STEP 5 — Reference computation via ReliabilityFunction ───
        print(_banner("Step 5 — sanity check via ReliabilityFunction", char="─"))
        rfunc = ReliabilityFunction(
            process_configs=configs,
            process_order=order,
            device="cpu",
            reliability_formula="gaussian",
        )
        F_ref, Q_ref = rfunc.compute_reliability(
            trajectory, return_quality_scores=True, use_sampled_outputs=False
        )
        print(f"  F (manual)    : {_fmt(F_manual)}")
        print(f"  F (rfunc)     : {_fmt(F_ref)}")
        max_abs = float((F_manual - F_ref).abs().max().item())
        print(f"  max |Δ|       : {max_abs:.2e}   "
              f"{'OK' if max_abs < 1e-5 else '⚠ MISMATCH'}")
        print("  per-process quality scores Q_t:")
        for name in order:
            print(f"      {name}:  manual={_fmt(Q_manual[name])}   rfunc={_fmt(Q_ref[name])}")
    print()


if __name__ == "__main__":
    main()
