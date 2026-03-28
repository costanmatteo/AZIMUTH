"""
Step 0: Genera dataset completo per l'intera catena di processi.

Per ogni campione, percorre la traiettoria completa attraverso tutti i processi
usando gli SCM da scm_ds/, e calcola F con ReliabilityFunction.

Usa: python generate_dataset.py [--n_samples 2000 --seed 42 ...]

Output:
- data/per_process/{process_name}_dataset.pt  → {inputs, outputs} per processo
- data/trajectories/full_trajectories.pt      → traiettorie complete + F
"""

import sys
from pathlib import Path
import argparse
import torch
import numpy as np

# Add project root to path
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from configs.processes_config import (
    PROCESSES, DATASET_MODE, ST_DATASET_CONFIG, _build_st_processes,
)


def main():
    parser = argparse.ArgumentParser(description='Generate dataset for all processes')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Override n_samples (default: from processes_config)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='data/',
                        help='Base output directory')

    # ST dataset complexity overrides
    parser.add_argument('--st_n', type=int, default=None,
                        help='ST input variables per process (overrides st_params.n)')
    parser.add_argument('--st_m', type=int, default=None,
                        help='ST cascaded stages per process (overrides st_params.m)')
    parser.add_argument('--st_rho', type=float, default=None,
                        help='ST noise intensity [0,1] (overrides st_params.rho)')
    parser.add_argument('--st_n_processes', type=int, default=None,
                        help='Number of ST processes in sequence (overrides n_processes)')

    args = parser.parse_args()

    # If ST dataset params are overridden via CLI, rebuild processes dynamically
    _st_overrides = {
        k: v for k, v in [('n', args.st_n), ('m', args.st_m), ('rho', args.st_rho)]
        if v is not None
    }
    _has_n_processes_override = args.st_n_processes is not None
    if (_st_overrides or _has_n_processes_override) and DATASET_MODE == 'st':
        import copy as _copy
        _st_cfg = _copy.deepcopy(ST_DATASET_CONFIG)
        _st_cfg['st_params'].update(_st_overrides)
        if _has_n_processes_override:
            _st_cfg['n_processes'] = args.st_n_processes
        _custom_processes = _build_st_processes(_st_cfg)
        # Monkey-patch so the rest of the script uses the new processes
        import configs.processes_config as _proc_mod
        _proc_mod.PROCESSES = _custom_processes
        print(f"\n[ST Override] Rebuilt processes with: {_st_overrides}"
              f"{f', n_processes={args.st_n_processes}' if _has_n_processes_override else ''}")

    # Re-read PROCESSES after potential monkey-patching
    from configs.processes_config import PROCESSES as current_processes

    output_dir = Path(args.output_dir)
    per_process_dir = output_dir / 'per_process'
    trajectories_dir = output_dir / 'trajectories'
    per_process_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    # Resolve n_samples: CLI override > process config
    n_samples = args.n_samples if args.n_samples is not None else current_processes[0].get('n_samples', 2000)

    print("=" * 70)
    print("AZIMUTH - STEP 0: GENERATE DATASET")
    print("=" * 70)
    print(f"\nDataset mode: {DATASET_MODE}")
    print(f"Processes: {[p['name'] for p in current_processes]}")
    print(f"Samples: {n_samples}")
    print(f"Seed: {args.seed}")
    print(f"Output dir: {output_dir}")

    # Import SCM data generation
    from uncertainty_predictor.src.data.preprocessing import generate_scm_data

    # Import ReliabilityFunction
    from reliability_function import ReliabilityFunction

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Step 1: Generate per-process datasets via SCM ───────────────────────
    print(f"\n[1/3] Generating per-process SCM datasets...")

    per_process_data = {}
    for proc in current_processes:
        proc_name = proc['name']
        scm_type = proc['scm_dataset_type']

        extra_kwargs = {}
        if scm_type == 'st' and 'st_params' in proc:
            extra_kwargs['st_params'] = proc['st_params']

        X, y, input_cols, output_cols, E, env_cols = generate_scm_data(
            n_samples=n_samples,
            seed=args.seed,
            dataset_type=scm_type,
            **extra_kwargs
        )

        # Separate control inputs from environmental variables
        n_env = len(env_cols)
        n_control = X.shape[1] - n_env
        X_control = X[:, :n_control]
        # E is already returned separately by generate_scm_data

        inputs_tensor = torch.tensor(X_control, dtype=torch.float32)
        env_tensor = torch.tensor(E, dtype=torch.float32)
        outputs_tensor = torch.tensor(y, dtype=torch.float32)

        per_process_data[proc_name] = {
            'inputs': inputs_tensor,       # controllable only
            'env': env_tensor,             # environmental (not controllable)
            'outputs': outputs_tensor,
            'input_columns': input_cols[:n_control],
            'env_columns': env_cols,
            'output_columns': output_cols,
        }

        # Save per-process dataset (inputs includes env for UP compatibility)
        save_path = per_process_dir / f'{proc_name}_dataset.pt'
        torch.save({
            'inputs': torch.tensor(X, dtype=torch.float32),  # control + env (UP needs both)
            'outputs': outputs_tensor,
        }, save_path)
        print(f"  {proc_name}: control {inputs_tensor.shape}, env {env_tensor.shape}, "
              f"outputs {outputs_tensor.shape} → {save_path}")

    # ── Step 2: Save DAG image ──────────────────────────────────────────────
    print(f"\n[2/4] Saving DAG image...")
    if DATASET_MODE == 'st':
        try:
            from scm_ds.datasets_st import STConfig, build_st_scm
            proc0 = current_processes[0]
            st_p = proc0.get('st_params', {})
            scm = build_st_scm(STConfig(**st_p), dag_image_dir=str(output_dir))
            print(f"  DAG saved to: {output_dir}/")
        except Exception as e:
            print(f"  Warning: Could not save DAG image: {e}")
    else:
        print("  Skipped (non-ST dataset mode)")

    # ── Step 3: Build full trajectories and compute F ───────────────────────
    print(f"\n[3/4] Building full trajectories and computing F...")

    # Build process configs for ReliabilityFunction
    # For ST mode, use surrogate_* fields; for physical mode, use default PROCESS_CONFIGS
    if DATASET_MODE == 'st':
        rf_process_configs = {}
        rf_process_order = []
        for proc in current_processes:
            pname = proc['name']
            rf_process_order.append(pname)
            rf_cfg = {
                'base_target': proc.get('surrogate_target', 0.0),
                'scale': proc.get('surrogate_scale', 1.0),
                'weight': proc.get('surrogate_weight', 1.0),
            }
            if 'surrogate_adaptive_coefficients' in proc:
                rf_cfg['adaptive_coefficients'] = proc['surrogate_adaptive_coefficients']
                rf_cfg['adaptive_baselines'] = proc['surrogate_adaptive_baselines']
            rf_process_configs[pname] = rf_cfg

        rf = ReliabilityFunction(
            process_configs=rf_process_configs,
            process_order=rf_process_order
        )
    else:
        rf = ReliabilityFunction()

    full_trajectories = []
    n = n_samples

    for i in range(n):
        # Build trajectory dict for ReliabilityFunction (needs inputs with env)
        trajectory_for_rf = {}
        for proc in current_processes:
            pname = proc['name']
            # RF expects inputs = control + env concatenated
            full_inputs = torch.cat([
                per_process_data[pname]['inputs'][i:i+1],
                per_process_data[pname]['env'][i:i+1],
            ], dim=1)
            trajectory_for_rf[pname] = {
                'inputs': full_inputs,
                'outputs_mean': per_process_data[pname]['outputs'][i:i+1],
                'outputs_sampled': per_process_data[pname]['outputs'][i:i+1],
            }

        F = rf.compute_reliability(trajectory_for_rf)
        F_val = F.item() if isinstance(F, torch.Tensor) else float(F)

        full_trajectories.append({
            'trajectory': {
                pname: {
                    'inputs': per_process_data[pname]['inputs'][i],
                    'env': per_process_data[pname]['env'][i],
                    'outputs': per_process_data[pname]['outputs'][i],
                }
                for pname in [p['name'] for p in current_processes]
            },
            'F': F_val,
        })

    # Save full trajectories
    traj_path = trajectories_dir / 'full_trajectories.pt'
    torch.save(full_trajectories, traj_path)

    F_values = [t['F'] for t in full_trajectories]
    print(f"\n  Trajectories: {len(full_trajectories)}")
    # Show per-process structure
    sample_traj = full_trajectories[0]['trajectory']
    for pname, pdata in sample_traj.items():
        print(f"  {pname}: inputs={pdata['inputs'].shape}, "
              f"env={pdata['env'].shape}, outputs={pdata['outputs'].shape}")
    print(f"  F statistics: mean={np.mean(F_values):.4f}, "
          f"std={np.std(F_values):.4f}, "
          f"min={np.min(F_values):.4f}, max={np.max(F_values):.4f}")
    print(f"  Saved to: {traj_path}")

    # ── Step 4: Summary ─────────────────────────────────────────────────────
    print(f"\n[4/4] Dataset generation complete!")
    print("\n" + "=" * 70)
    print("GENERATED FILES")
    print("=" * 70)
    for proc in current_processes:
        pname = proc['name']
        print(f"  {per_process_dir / f'{pname}_dataset.pt'}")
    print(f"  {traj_path}")

    print("\n" + "=" * 70)
    print("STEP 0 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nNext step: Run train_predictor.py to train uncertainty predictors")


if __name__ == '__main__':
    main()
