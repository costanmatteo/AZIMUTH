"""
Step 0: Genera dataset completo per l'intera catena di processi.

Per ogni campione, percorre la traiettoria completa attraverso tutti i processi
usando gli SCM da scm_ds/, e calcola F con ReliabilityFunction.

Usa: python generate_dataset.py [--n_samples 2000 --seed 42 ...]

Output:
- scm_ds/predictor_dataset/per_process/{process_name}_dataset.pt  → {inputs, outputs} per processo
- scm_ds/predictor_dataset/trajectories/full_trajectories.pt      → traiettorie complete + F
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



def _save_st_dag(n: int, m: int, rho: float, save_path: str,
                 me: int = 0, p: int = 1, env_mode: str = 'A',
                 dpi: int = 200):
    """Save a B&W academic DAG matching the thesis figure.

    Notation (thesis convention):
        S_1..S_n  = input variables   (code: X_1..X_n)
        X_1..X_m  = stage variables   (code: S_1..S_m)
        Y         = output

    Structure:
        S inputs are partitioned across stages.  Each X_k = f(subset of S, X_{k-1}).
        The chain X_1 -> X_2 -> ... -> X_m -> Y runs horizontally.
        S inputs sit above the X they feed into, with vertical f arrows.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Compute actual input partition (uniform, matching _compute_width)
    m_eff = min(m, n)
    base = n // m_eff
    rem = n % m_eff
    widths = [base + (1 if k < rem else 0) for k in range(m_eff)]

    # Build input groups per stage (1-indexed)
    groups = []
    idx = 0
    for k in range(m_eff):
        groups.append((idx + 1, idx + widths[k]))  # (first, last) 1-indexed
        idx += widths[k]

    # Decide which stages to show (ellipsis for large m)
    max_show = 5
    if m_eff <= max_show + 1:
        show = list(range(m_eff))
    else:
        show = [0, 1, None, m_eff - 2, m_eff - 1]  # None = ellipsis

    # Layout
    x_sp = 1.8
    y_s = 1.2   # S row (top)
    y_x = 0.0   # X row (bottom)
    fig_w = max(5.5, (len(show) + 1) * x_sp + 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, 2.4), facecolor='white')

    col_x = {}
    cx = 0.8

    for si, slot in enumerate(show):
        if slot is None:
            # Ellipsis column
            ax.text(cx, y_s, r'$\cdots$', ha='center', va='center',
                    fontsize=12, color='black')
            ax.text(cx, y_x, r'$\cdots$', ha='center', va='center',
                    fontsize=12, color='black')
            # Horizontal arrows into/out of ellipsis
            prev_cx = col_x[si - 1]
            ax.annotate('', xy=(cx - 0.32, y_x),
                        xytext=(prev_cx + 0.22, y_x),
                        arrowprops=dict(arrowstyle='->', lw=0.9, color='black'))
            col_x[si] = cx
            cx += x_sp
            continue

        col_x[si] = cx
        k = slot + 1  # 1-indexed stage
        first_s, last_s = groups[slot]

        # S label (input group) — above X
        if first_s == last_s:
            s_label = f'$S_{{{first_s}}}$'
        elif last_s - first_s == 1:
            s_label = f'$S_{{{first_s}}},\\, S_{{{last_s}}}$'
        else:
            s_label = f'$S_{{{first_s}}} \\cdot\\cdot S_{{{last_s}}}$'
        ax.text(cx, y_s, s_label, ha='center', va='center', fontsize=10,
                fontfamily='serif')

        # X label (stage) — on the chain
        ax.text(cx, y_x, f'$X_{{{k}}}$', ha='center', va='center',
                fontsize=10, fontfamily='serif')

        # Vertical arrow S -> X
        ax.annotate('', xy=(cx, y_x + 0.25),
                    xytext=(cx, y_s - 0.25),
                    arrowprops=dict(arrowstyle='->', lw=0.9, color='black'))

        # Horizontal arrow from previous stage
        if si > 0:
            prev_cx = col_x[si - 1]
            if show[si - 1] is None:
                # Arrow from ellipsis
                ax.annotate('', xy=(cx - 0.22, y_x),
                            xytext=(prev_cx + 0.32, y_x),
                            arrowprops=dict(arrowstyle='->', lw=0.9,
                                            color='black'))
            else:
                ax.annotate('', xy=(cx - 0.22, y_x),
                            xytext=(prev_cx + 0.22, y_x),
                            arrowprops=dict(arrowstyle='->', lw=0.9,
                                            color='black'))

        cx += x_sp

    # Y node
    y_cx = cx
    ax.text(y_cx, y_x, '$Y$', ha='center', va='center', fontsize=10,
            fontfamily='serif')

    # Arrow X_m -> Y
    last_si = [i for i, s in enumerate(show) if s is not None][-1]
    ax.annotate('', xy=(y_cx - 0.18, y_x),
                xytext=(col_x[last_si] + 0.22, y_x),
                arrowprops=dict(arrowstyle='->', lw=0.9, color='black'))

    # Caption
    ax.text((0.8 + y_cx) / 2, y_x - 0.55,
            f'DAG of a single stage  ($n={n},\\ m={m_eff},\\ \\rho={rho},'
            f'\\ m_e={me},\\ p={p},\\ \\mathrm{{env}}={env_mode}$)',
            ha='center', va='top', fontsize=9, fontfamily='serif')

    ax.set_xlim(0, y_cx + 0.6)
    ax.set_ylim(y_x - 0.75, y_s + 0.45)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout(pad=0.2)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Generate dataset for all processes')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Override n_samples (default: from processes_config)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='scm_ds/predictor_dataset/',
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
    from scm_ds.reliability_function import ReliabilityFunction

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
            proc0 = current_processes[0]
            st_p = proc0.get('st_params', {})
            dag_path = output_dir / 'dag.png'
            _save_st_dag(
                n=st_p.get('n', 5),
                m=st_p.get('m', 3),
                rho=st_p.get('rho', 0.0),
                save_path=str(dag_path),
                me=st_p.get('me', 0),
                p=st_p.get('p', 1),
                env_mode=st_p.get('env_mode', 'A'),
            )
            print(f"  DAG saved to: {dag_path}")
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

    # ── Step 4: Convert to causaliT format ───────────────────────────────
    print(f"\n[4/4] Converting to causaliT format...")
    from scm_ds.convert_azimuth_trajectories import (
        extract_process_info, build_arrays, build_masks, build_metadata,
    )

    causalit_dir = REPO_ROOT / 'scm_ds' / 'causalit_dataset'
    causalit_dir.mkdir(parents=True, exist_ok=True)

    process_names, process_dims = extract_process_info(full_trajectories)
    s_arr, x_arr, y_arr, s_labels, x_labels, t_labels = build_arrays(
        full_trajectories, process_names, process_dims
    )
    np.savez_compressed(str(causalit_dir / 'ds.npz'), s=s_arr, x=x_arr, y=y_arr)

    import json as _json
    metadata = build_metadata(s_labels, x_labels, t_labels)
    with open(causalit_dir / 'dataset_metadata.json', 'w', encoding='utf-8') as f:
        _json.dump(metadata, f, indent=2, sort_keys=True, ensure_ascii=False)

    import pandas as _pd
    masks = build_masks(process_names, process_dims, s_labels, x_labels, t_labels)
    for fname, df in masks.items():
        df.to_csv(causalit_dir / fname)

    print(f"  S={s_arr.shape}, X={x_arr.shape}, Y={y_arr.shape}")
    print(f"  Saved to: {causalit_dir}/")

    # ── Step 5: Summary ─────────────────────────────────────────────────────
    print(f"\n[5/5] Dataset generation complete!")
    print("\n" + "=" * 70)
    print("GENERATED FILES")
    print("=" * 70)
    for proc in current_processes:
        pname = proc['name']
        print(f"  {per_process_dir / f'{pname}_dataset.pt'}")
    print(f"  {traj_path}")
    print(f"  {causalit_dir}/ (causaliT format)")

    print("\n" + "=" * 70)
    print("STEP 0 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"  {causalit_dir}/")
    print("\nNext step: Run train_predictor.py to train uncertainty predictors")


if __name__ == '__main__':
    main()
