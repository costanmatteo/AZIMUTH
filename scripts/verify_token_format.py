#!/usr/bin/env python3
"""
Verify that the surrogate token format produced at training time
(`convert_dataset.py`) matches exactly what the controller generates at
inference time (`ProcessChain.trajectory_to_prot_format`).

Runs end-to-end for BOTH supported models:
    - proT:           (x, y)
    - StageCausaliT:  (s, x, y)

Strategy
--------
Real training data (`causaliT/data/azimuth_surrogate/ds.npz`) is used when
present. When it isn't, a synthetic trajectories file is generated on the fly
from `configs.processes_config.PROCESSES`, `convert_dataset.py` is run over it
into a temp dataset dir, and the checks are performed against that.

A minimal ProcessChain stub is instantiated without touching the heavy
uncertainty-predictor machinery: we only need `process_names`, `device`, and
the pure methods involved in token construction. A runtime trajectory dict
matching the layout produced by `ProcessChain.forward` is built by hand from
the same synthetic source values so any mismatch between the two code paths
shows up immediately.

Exit code: 0 if all checks pass, 1 otherwise.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from configs.processes_config import PROCESSES, get_controllable_inputs  # noqa: E402
from addition_to_causaliT.surrogate_training.convert_dataset import (  # noqa: E402
    convert_trajectories_to_causalit_format,
    get_canonical_token_layout,
)
from controller.src.core.process_chain import ProcessChain  # noqa: E402


# ============================================================================
# Check infrastructure
# ============================================================================


class CheckReport:
    def __init__(self):
        self.results: List[Tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = ""):
        self.results.append((name, ok, detail))
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {name}" + (f"  — {detail}" if detail else ""))

    @property
    def all_ok(self) -> bool:
        return all(ok for _, ok, _ in self.results)


# ============================================================================
# Synthetic data builders
# ============================================================================


def _split_ctrl_env(process_config) -> Tuple[List[int], List[int]]:
    input_labels = process_config['input_labels']
    controllable = get_controllable_inputs(process_config)
    ctrl, env = [], []
    for i, lab in enumerate(input_labels):
        (ctrl if lab in controllable else env).append(i)
    return ctrl, env


def build_synthetic_trajectories(processes_config, n_samples: int,
                                 rng: np.random.RandomState) -> list:
    """
    Synthesize a full_trajectories.pt-like list of dicts using the per-process
    [inputs, env, outputs] separation that the offline pipeline uses.
    """
    trajs = []
    for i in range(n_samples):
        traj = {}
        for pc in processes_config:
            pname = pc['name']
            ctrl_idx, env_idx = _split_ctrl_env(pc)
            n_ctrl = len(ctrl_idx)
            n_env = len(env_idx)
            n_out = len(pc['output_labels'])

            # Give each variable a distinct scale so per-variable standardization
            # actually does something testable.
            ctrl_vals = rng.normal(
                loc=1.0 + 10 * np.arange(n_ctrl), scale=0.5, size=n_ctrl)
            env_vals = rng.normal(
                loc=50.0 + 3 * np.arange(n_env), scale=1.0, size=n_env) \
                if n_env > 0 else np.zeros(0)
            out_vals = rng.normal(
                loc=-5.0 + 2 * np.arange(n_out), scale=0.2, size=n_out)

            traj[pname] = {
                'inputs': torch.tensor(ctrl_vals, dtype=torch.float32),
                'env': torch.tensor(env_vals, dtype=torch.float32),
                'outputs': torch.tensor(out_vals, dtype=torch.float32),
            }
        trajs.append({'trajectory': traj, 'F': float(rng.uniform(0.3, 0.9))})
    return trajs


def build_runtime_trajectory_from_offline(
        offline_traj_list, processes_config,
        batch_indices: List[int]) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Construct a trajectory dict in the *runtime* layout emitted by
    `ProcessChain.forward`, picking a subset of samples from the offline
    list. Inputs are reordered into `input_labels` order (controllable at
    their declared positions, non-controllable at theirs) so the inference
    converter sees exactly what ProcessChain.forward would produce.
    """
    B = len(batch_indices)
    runtime: Dict[str, Dict[str, torch.Tensor]] = {}
    for pc in processes_config:
        pname = pc['name']
        input_dim = len(pc['input_labels'])
        ctrl_idx, env_idx = _split_ctrl_env(pc)
        out_dim = len(pc['output_labels'])

        inputs = torch.zeros(B, input_dim, dtype=torch.float32)
        outputs = torch.zeros(B, out_dim, dtype=torch.float32)
        for bi, sample_i in enumerate(batch_indices):
            tp = offline_traj_list[sample_i]['trajectory'][pname]
            ctrl_vals = tp['inputs'].flatten()
            env_vals = tp['env'].flatten() if 'env' in tp else torch.zeros(0)
            for out_i, pos in enumerate(ctrl_idx):
                inputs[bi, pos] = ctrl_vals[out_i]
            for out_i, pos in enumerate(env_idx):
                inputs[bi, pos] = env_vals[out_i]
            outputs[bi] = tp['outputs'].flatten()
        runtime[pname] = {
            'inputs': inputs,
            'outputs_mean': outputs,
            'outputs_var': torch.zeros_like(outputs),
            'outputs_sampled': outputs,
        }
    return runtime


# ============================================================================
# ProcessChain stub (no uncertainty predictors needed)
# ============================================================================


def _make_chain_stub(process_names: List[str], device: str) -> ProcessChain:
    """
    Build a ProcessChain instance without running its heavy __init__ so we
    can call `trajectory_to_prot_format` in isolation.
    """
    pc = ProcessChain.__new__(ProcessChain)
    pc.process_names = list(process_names)
    pc.device = device
    pc._surrogate_meta_cache = None
    return pc


# ============================================================================
# Per-model checks
# ============================================================================


def check_prot(dataset_dir: Path,
               offline_trajs: list,
               processes_config,
               report: CheckReport) -> None:
    print("\n[ProT] Running checks ...")

    ds_path = dataset_dir / 'ds.npz'
    meta_path = dataset_dir / 'dataset_metadata.json'
    if not ds_path.exists() or not meta_path.exists():
        report.add("prot/artifacts_exist", False,
                   f"Missing {ds_path.name} or {meta_path.name}")
        return
    report.add("prot/artifacts_exist", True, f"{ds_path.name} + metadata")

    ds = np.load(ds_path)
    meta = json.loads(meta_path.read_text())

    x_train = ds['x']  # (N, L, 2)
    y_train = ds['y']  # (N, 1, 2)
    report.add("prot/x_rank3", x_train.ndim == 3,
               f"x.shape={x_train.shape}")
    report.add("prot/y_rank3", y_train.ndim == 3,
               f"y.shape={y_train.shape}")
    report.add("prot/x_feature_dim_2", x_train.shape[-1] == 2,
               f"feature_dim={x_train.shape[-1]}")
    report.add("prot/y_feature_dim_2", y_train.shape[-1] == 2,
               f"feature_dim={y_train.shape[-1]}")

    # var_ids on training side
    x_var_ids_train = x_train[:, :, 1].astype(np.int64)
    report.add("prot/x_var_ids_consistent",
               np.all(x_var_ids_train == x_var_ids_train[0:1]),
               "var_ids identical across samples")
    report.add("prot/x_var_ids_ge1",
               int(x_var_ids_train.min()) >= 1,
               f"min var_id={int(x_var_ids_train.min())}")
    report.add("prot/x_var_ids_unique",
               len(np.unique(x_var_ids_train[0])) == x_var_ids_train.shape[1],
               f"{len(np.unique(x_var_ids_train[0]))} unique")

    # Standardization: column 0 should be near zero mean / unit std per column
    col0 = x_train[:, :, 0]
    col0_mean = np.abs(col0.mean(axis=0)).max()
    col0_std = col0.std(axis=0)
    report.add("prot/values_standardized_mean",
               col0_mean < 1e-4,
               f"max|mean|={col0_mean:.2e}")
    report.add("prot/values_standardized_std",
               np.allclose(col0_std, 1.0, atol=1e-4) or
               np.all((col0_std > 0.99) & (col0_std < 1.01)),
               f"std range=[{col0_std.min():.4f},{col0_std.max():.4f}]")

    # ---------------- Inference-side conversion -------------------------
    B = 4
    runtime = build_runtime_trajectory_from_offline(
        offline_trajs, processes_config, batch_indices=list(range(B)))
    chain = _make_chain_stub([p['name'] for p in processes_config], 'cpu')

    # Point the surrogate metadata loader at our temp dataset dir by monkey-
    # patching the cache.
    chain._surrogate_meta_cache = {
        'model_name': 'proT',
        'meta': meta,
        'layout': meta['token_layout'],
        'norm': meta['normalization'],
    }

    X_inf, Y_inf = chain.trajectory_to_prot_format(runtime)

    report.add("prot/inf_feature_dim_2", X_inf.shape[-1] == 2,
               f"X_inf.shape={tuple(X_inf.shape)}")
    report.add("prot/inf_seq_len_matches",
               X_inf.shape[1] == x_train.shape[1],
               f"{X_inf.shape[1]} vs {x_train.shape[1]}")
    report.add("prot/inf_y_shape",
               tuple(Y_inf.shape) == (B, 1, 2),
               f"Y_inf.shape={tuple(Y_inf.shape)}")

    # var_ids must match position-by-position
    x_var_ids_inf = X_inf[0, :, 1].detach().cpu().numpy().astype(np.int64)
    report.add("prot/inf_var_ids_match_training",
               np.array_equal(x_var_ids_inf, x_var_ids_train[0]),
               f"train={x_var_ids_train[0][:6]}...  inf={x_var_ids_inf[:6]}...")

    # Values: inference should be standardized too. Compare inference values on
    # batch sample 0 to the *standardized* training value for the same raw
    # sample 0 (they should be element-wise equal since the same values flow
    # through the same stats).
    train_vals_0 = x_train[0, :, 0]
    inf_vals_0 = X_inf[0, :, 0].detach().cpu().numpy()
    max_abs_diff = float(np.max(np.abs(train_vals_0 - inf_vals_0)))
    report.add("prot/inf_values_identical_to_training",
               max_abs_diff < 1e-5,
               f"max|train-inf|={max_abs_diff:.2e}")

    # Distributions compatible: per-column mean/std of inference across the
    # small batch should be finite and comparable magnitude to training.
    inf_col0 = X_inf[:, :, 0].detach().cpu().numpy()
    report.add("prot/inf_values_finite",
               np.all(np.isfinite(inf_col0)),
               "inference values are finite")
    report.add("prot/inf_value_scale_compatible",
               float(np.abs(inf_col0).max()) < 20.0,
               f"max|inf_value|={float(np.abs(inf_col0).max()):.2f}")


def check_stage_causal(dataset_dir: Path,
                       offline_trajs: list,
                       processes_config,
                       report: CheckReport) -> None:
    print("\n[StageCausaliT] Running checks ...")

    ds_path = dataset_dir / 'ds.npz'
    meta_path = dataset_dir / 'dataset_metadata.json'
    if not ds_path.exists() or not meta_path.exists():
        report.add("sc/artifacts_exist", False,
                   f"Missing {ds_path.name} or {meta_path.name}")
        return
    report.add("sc/artifacts_exist", True, f"{ds_path.name} + metadata")

    ds = np.load(ds_path)
    meta = json.loads(meta_path.read_text())

    for key in ('s', 'x', 'y'):
        report.add(f"sc/{key}_rank3", ds[key].ndim == 3,
                   f"{key}.shape={ds[key].shape}")
        report.add(f"sc/{key}_feature_dim_2", ds[key].shape[-1] == 2,
                   f"feature_dim={ds[key].shape[-1]}")

    # var_ids: S should be 1..n_s, X should be 1..n_x, Y should be 1
    for key in ('s', 'x', 'y'):
        vids = ds[key][:, :, 1].astype(np.int64)
        report.add(f"sc/{key}_var_ids_consistent",
                   np.all(vids == vids[0:1]),
                   "var_ids identical across samples")
        report.add(f"sc/{key}_var_ids_ge1",
                   int(vids.min()) >= 1,
                   f"min={int(vids.min())}")

    n_s = ds['s'].shape[1]
    n_x = ds['x'].shape[1]
    report.add("sc/s_var_ids_range",
               list(ds['s'][0, :, 1].astype(int)) == list(range(1, n_s + 1)),
               f"expected 1..{n_s}")
    report.add("sc/x_var_ids_range",
               list(ds['x'][0, :, 1].astype(int)) == list(range(1, n_x + 1)),
               f"expected 1..{n_x}")

    for key in ('s', 'x'):
        col0 = ds[key][:, :, 0]
        col0_mean = np.abs(col0.mean(axis=0)).max()
        col0_std = col0.std(axis=0)
        report.add(f"sc/{key}_values_standardized_mean",
                   col0_mean < 1e-4,
                   f"max|mean|={col0_mean:.2e}")
        report.add(f"sc/{key}_values_standardized_std",
                   np.all((col0_std > 0.99) & (col0_std < 1.01)),
                   f"std range=[{col0_std.min():.4f},{col0_std.max():.4f}]")

    # ---------------- Inference-side conversion -------------------------
    B = 4
    runtime = build_runtime_trajectory_from_offline(
        offline_trajs, processes_config, batch_indices=list(range(B)))
    chain = _make_chain_stub([p['name'] for p in processes_config], 'cpu')
    chain._surrogate_meta_cache = {
        'model_name': 'StageCausaliT',
        'meta': meta,
        'layout': meta['token_layout'],
        'norm': meta['normalization'],
    }

    S_inf, X_inf, Y_inf = chain.trajectory_to_prot_format(runtime)

    report.add("sc/inf_s_shape",
               tuple(S_inf.shape) == (B, n_s, 2),
               f"S_inf={tuple(S_inf.shape)}")
    report.add("sc/inf_x_shape",
               tuple(X_inf.shape) == (B, n_x, 2),
               f"X_inf={tuple(X_inf.shape)}")
    report.add("sc/inf_y_shape",
               tuple(Y_inf.shape) == (B, 1, 2),
               f"Y_inf={tuple(Y_inf.shape)}")

    # Values on sample 0 must match training element-wise
    train_s0 = ds['s'][0, :, 0]
    inf_s0 = S_inf[0, :, 0].detach().cpu().numpy()
    train_x0 = ds['x'][0, :, 0]
    inf_x0 = X_inf[0, :, 0].detach().cpu().numpy()
    report.add("sc/inf_s_values_identical_to_training",
               float(np.max(np.abs(train_s0 - inf_s0))) < 1e-5,
               f"max|Δ|={float(np.max(np.abs(train_s0 - inf_s0))):.2e}")
    report.add("sc/inf_x_values_identical_to_training",
               float(np.max(np.abs(train_x0 - inf_x0))) < 1e-5,
               f"max|Δ|={float(np.max(np.abs(train_x0 - inf_x0))):.2e}")


# ============================================================================
# Main
# ============================================================================


def _select_processes() -> List[Dict[str, Any]]:
    """Pick the processes that are actually usable offline (physical mode)."""
    # PROCESSES may be ST or physical depending on DATASET_MODE; we don't need
    # a trained predictor for this script, so use whatever is configured.
    return list(PROCESSES)


def main() -> int:
    print("=" * 70)
    print("Surrogate token-format verification")
    print("=" * 70)

    processes_config = _select_processes()
    process_names = [p['name'] for p in processes_config]
    print(f"Using {len(processes_config)} processes: {process_names}")

    # Sanity-check the canonical layout
    layout = get_canonical_token_layout(processes_config)
    prot_n = len(layout['proT']['x_tokens'])
    sc_n_s = len(layout['stage_causal']['s_tokens'])
    sc_n_x = len(layout['stage_causal']['x_tokens'])
    print(f"Canonical layout: proT n_x={prot_n}  |  "
          f"stage_causal n_s={sc_n_s}, n_x={sc_n_x}")

    # Synthesize offline trajectories
    n_samples = 128
    rng = np.random.RandomState(0)
    offline_trajs = build_synthetic_trajectories(processes_config, n_samples, rng)

    # Write to a temp file and convert for BOTH model types
    report = CheckReport()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        traj_path = tmpdir / 'synthetic_trajectories.pt'
        torch.save(offline_trajs, traj_path)

        # ---- proT ----
        prot_dir = tmpdir / 'azimuth_surrogate_prot'
        convert_trajectories_to_causalit_format(
            trajectories_path=str(traj_path),
            output_dir=str(prot_dir),
            model_type='proT',
        )
        check_prot(prot_dir, offline_trajs, processes_config, report)

        # ---- StageCausaliT ----
        sc_dir = tmpdir / 'azimuth_surrogate_sc'
        convert_trajectories_to_causalit_format(
            trajectories_path=str(traj_path),
            output_dir=str(sc_dir),
            model_type='StageCausaliT',
        )
        check_stage_causal(sc_dir, offline_trajs, processes_config, report)

    # ---- Optional: if real training data exists, also verify it -------
    real_dir = REPO_ROOT / 'causaliT' / 'data' / 'azimuth_surrogate'
    if (real_dir / 'ds.npz').exists():
        print(f"\nReal training data found at {real_dir}, running sanity checks...")
        real_ds = np.load(real_dir / 'ds.npz')
        print(f"  Keys: {list(real_ds.files)}")
        for k in real_ds.files:
            arr = real_ds[k]
            report.add(f"real/{k}_rank3", arr.ndim == 3, f"shape={arr.shape}")
            report.add(f"real/{k}_feature_dim_2", arr.shape[-1] == 2,
                       f"feature_dim={arr.shape[-1]}")
    else:
        print(f"\n(No real training data at {real_dir} — verified on synthetic only.)")

    print("\n" + "=" * 70)
    if report.all_ok:
        print("FORMAT ALIGNED — safe to retrain surrogate")
        print("=" * 70)
        return 0
    else:
        n_fail = sum(1 for _, ok, _ in report.results if not ok)
        print(f"FORMAT MISMATCH — {n_fail} checks failed")
        print("=" * 70)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
