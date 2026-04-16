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
import random
import torch
import numpy as np

# Add project root to path
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from configs.processes_config import (
    PROCESSES, DATASET_MODE, ST_DATASET_CONFIG, _build_st_processes,
)



def _save_st_dag(n: int, m: int, rho: float, save_path: str,
                 me: int = 0, p: int = 1,
                 output_overlap: float = 0.0, env_mode: str = 'A',
                 dpi: int = 200):
    """Save a B&W academic DAG for the ST dataset.

    For p > 1, draws p independent parallel chains (one per output
    partition) stacked vertically.  Shared inputs (from output_overlap)
    are highlighted with a dashed border.

    Notation (thesis convention):
        S_1..S_n    = input variables
        X_k^{(r)}   = stage k of chain r  (X_k when p=1)
        Y_r         = output of chain r   (Y when p=1)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ── Build output partitions (mirrors datasets_st.py) ─────────────
    if p == 1:
        partitions = [list(range(n))]
    else:
        base_size = n // p
        remainder = n % p
        partitions = []
        idx = 0
        for r in range(p):
            size = base_size + (1 if r < remainder else 0)
            partitions.append(list(range(idx, idx + size)))
            idx += size

    # Apply overlap (mirrors datasets_st.py)
    shared_indices = set()
    if output_overlap > 0.0 and p > 1:
        for r in range(p - 1):
            p_curr = partitions[r]
            p_next = partitions[r + 1]
            n_share = max(1, int(output_overlap * min(len(p_curr), len(p_next))))
            for si in p_curr[-n_share:]:
                if si not in p_next:
                    p_next.append(si)
                shared_indices.add(si)
            for si in p_next[:n_share]:
                if si not in p_curr:
                    p_curr.append(si)
                shared_indices.add(si)

    # ── Per-chain layout data ─────────────────────────────────────────
    chain_data = []
    for r in range(p):
        part = partitions[r]
        n_sub = len(part)
        m_eff = min(m, n_sub)

        # Uniform width distribution within chain
        base = n_sub // m_eff
        rem = n_sub % m_eff
        widths = [base + (1 if k < rem else 0) for k in range(m_eff)]

        # Build input groups per stage (0-indexed input indices)
        groups = []
        idx = 0
        for k in range(m_eff):
            groups.append(part[idx:idx + widths[k]])
            idx += widths[k]

        # Decide which stages to show (ellipsis for large m)
        max_show = 5
        if m_eff <= max_show + 1:
            show = list(range(m_eff))
        else:
            show = [0, 1, None, m_eff - 2, m_eff - 1]

        chain_data.append({
            'm_eff': m_eff,
            'groups': groups,
            'show': show,
        })

    # ── Layout parameters ─────────────────────────────────────────────
    x_sp = 1.8          # horizontal spacing between stages
    row_gap = 1.2        # vertical gap between S row and X row
    chain_h = 2.8        # vertical height of one chain cell
    left_margin = 1.6 if p > 1 else 0.8

    max_stage_cols = max(len(cd['show']) for cd in chain_data) + 1  # +1 for Y
    chain_w = left_margin + max_stage_cols * x_sp + 0.4

    # Grid: up to 3 columns
    n_grid_cols = 1
    n_grid_rows = (p + n_grid_cols - 1) // n_grid_cols

    total_data_w = n_grid_cols * chain_w

    if p == 1:
        fig_w = max(5.5, chain_w + 0.5)
        fig_h = 2.4
    else:
        fig_w = min(16.0, total_data_w + 0.5)
        fig_h = max(3.0, n_grid_rows * 2.6 + 0.8)

    # Auto-scale label font size for crowded multi-chain layouts
    if p > 1:
        inches_per_stage = fig_w * x_sp / total_data_w
        fs = min(10, max(7, round(inches_per_stage * 8.5)))
    else:
        fs = 10
    fs_cap = max(7, fs - 1)  # caption slightly smaller

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor='white')

    dashed_bbox = dict(boxstyle='round,pad=0.12', edgecolor='black',
                       facecolor='none', linestyle='--', linewidth=0.8)

    def _s_label_text(sorted_0idx):
        """Build S label string from sorted 0-indexed input indices."""
        s_ids = [i + 1 for i in sorted_0idx]
        if len(s_ids) == 1:
            return f'$S_{{{s_ids[0]}}}$'
        if len(s_ids) == 2:
            return f'$S_{{{s_ids[0]}}},\\, S_{{{s_ids[-1]}}}$'
        return f'$S_{{{s_ids[0]}}} \\cdot\\cdot S_{{{s_ids[-1]}}}$'

    for r, cd in enumerate(chain_data):
        gc = r % n_grid_cols              # grid column  (0, 1, 2)
        gr = r // n_grid_cols             # grid row
        x_origin = gc * chain_w
        y_base = -gr * chain_h
        y_s = y_base + row_gap / 2     # S row (top)
        y_x = y_base - row_gap / 2     # X row (bottom)

        show = cd['show']
        groups = cd['groups']
        m_eff = cd['m_eff']

        col_x = {}
        cx = x_origin + left_margin

        # Chain label (p > 1 only)
        if p > 1:
            ax.text(x_origin + left_margin - 1.0, (y_s + y_x) / 2,
                    f'$r\\!=\\!{r + 1}$',
                    ha='center', va='center', fontsize=fs,
                    fontfamily='serif', fontstyle='italic')

        for si, slot in enumerate(show):
            if slot is None:
                # Ellipsis column
                ax.text(cx, y_s, r'$\cdots$', ha='center', va='center',
                        fontsize=fs + 2, color='black')
                ax.text(cx, y_x, r'$\cdots$', ha='center', va='center',
                        fontsize=fs + 2, color='black')
                prev_cx = col_x[si - 1]
                ax.annotate('', xy=(cx - 0.32, y_x),
                            xytext=(prev_cx + 0.22, y_x),
                            arrowprops=dict(arrowstyle='->', lw=0.9,
                                            color='black'))
                col_x[si] = cx
                cx += x_sp
                continue

            col_x[si] = cx
            k = slot + 1  # 1-indexed stage
            stage_inputs = groups[slot]

            # S label (input group) — above X
            # Split shared / non-shared so dashed box wraps only shared S
            sorted_inp = sorted(stage_inputs)
            shared_in = [i for i in sorted_inp if i in shared_indices]
            plain_in = [i for i in sorted_inp if i not in shared_indices]

            if not shared_in or not plain_in:
                # Homogeneous group → single combined label
                s_label = _s_label_text(sorted_inp)
                ax.text(cx, y_s, s_label, ha='center', va='center',
                        fontsize=fs, fontfamily='serif',
                        bbox=dashed_bbox if shared_in else None)
            else:
                # Mixed group → stacked vertically to avoid horizontal overlap
                sub_parts = [(plain_in, False), (shared_in, True)]
                sub_parts.sort(key=lambda t: t[0][0])
                v_off = 0.20
                for j, (idxs, is_sh) in enumerate(sub_parts):
                    y_pos = y_s + (0.5 - j) * v_off
                    ax.text(cx, y_pos, _s_label_text(idxs),
                            ha='center', va='center',
                            fontsize=fs, fontfamily='serif',
                            bbox=dashed_bbox if is_sh else None)

            # X label (stage node) — on the chain
            if p == 1:
                x_label = f'$X_{{{k}}}$'
            else:
                x_label = f'$X_{{{k}}}^{{({r + 1})}}$'

            ax.text(cx, y_x, x_label, ha='center', va='center',
                    fontsize=fs, fontfamily='serif')

            # Vertical arrow S -> X
            ax.annotate('', xy=(cx, y_x + 0.25),
                        xytext=(cx, y_s - 0.25),
                        arrowprops=dict(arrowstyle='->', lw=0.9,
                                        color='black'))

            # Horizontal arrow from previous stage
            if si > 0:
                prev_cx = col_x[si - 1]
                if show[si - 1] is None:
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
        y_label = '$Y$' if p == 1 else f'$Y_{{{r + 1}}}$'
        ax.text(y_cx, y_x, y_label, ha='center', va='center',
                fontsize=fs, fontfamily='serif')

        # Arrow X_m -> Y
        last_si = [i for i, s in enumerate(show) if s is not None][-1]
        ax.annotate('', xy=(y_cx - 0.18, y_x),
                    xytext=(col_x[last_si] + 0.22, y_x),
                    arrowprops=dict(arrowstyle='->', lw=0.9, color='black'))

    # ── Caption ───────────────────────────────────────────────────────
    caption_y = -(n_grid_rows - 1) * chain_h - row_gap / 2 - 0.55

    caption = (
        f'$n={n},\\ m={m},\\ p={p},\\ \\rho={rho},'
        f'\\ \\mathrm{{overlap}}={output_overlap},'
        f'\\ m_e={me},\\ \\mathrm{{env}}={env_mode}$'
    )
    ax.text(total_data_w / 2, caption_y, caption,
            ha='center', va='top', fontsize=fs_cap, fontfamily='serif')

    # ── Finalize ──────────────────────────────────────────────────────
    ax.set_xlim(0, total_data_w)
    ax.set_ylim(caption_y - 0.3, row_gap / 2 + 0.45)
    ax.set_aspect('equal' if p == 1 else 'auto')
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

    from scm_ds import ReliabilityFunction, ShekelReliabilityFunction, calibrate_shekel_configs

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
                output_overlap=st_p.get('output_overlap', 0.0),
                env_mode=st_p.get('env_mode', 'A'),
            )
            print(f"  DAG saved to: {dag_path}")
        except Exception as e:
            print(f"  Warning: Could not save DAG image: {e}")
    else:
        print("  Skipped (non-ST dataset mode)")

    # ── Step 3: Build full trajectories and compute F ───────────────────────
    print(f"\n[3/4] Building full trajectories and computing F...")

    # Build process configs for ReliabilityFunction / ShekelReliabilityFunction
    # For ST mode, use surrogate_* fields; for physical mode, use default PROCESS_CONFIGS
    rf_type = ST_DATASET_CONFIG.get('reliability_function_type', 'gaussian') if DATASET_MODE == 'st' else 'gaussian'
    shekel_s = ST_DATASET_CONFIG.get('shekel_s', 1.0) if DATASET_MODE == 'st' else 1.0

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
                for src_key, dst_key in [
                    ('surrogate_adaptive_mode',           'adaptive_mode'),
                    ('surrogate_adaptive_coefficients2',  'adaptive_coefficients2'),
                    ('surrogate_adaptive_power',          'adaptive_power'),
                    ('surrogate_adaptive_band',           'adaptive_band'),
                    ('surrogate_adaptive_sharpness',      'adaptive_sharpness'),
                    ('surrogate_adaptive_max_shift',      'adaptive_max_shift'),
                ]:
                    if src_key in proc:
                        rf_cfg[dst_key] = proc[src_key]
            rf_process_configs[pname] = rf_cfg

        if rf_type == 'shekel':
            # Build calibration trajectories from the generated per-process data
            # (single trajectory containing all n_samples as batch)
            cal_trajectory = {
                pname: {'outputs_mean': per_process_data[pname]['outputs']}
                for pname in rf_process_order
            }
            shekel_configs = calibrate_shekel_configs(
                rf_process_configs, rf_process_order,
                calibration_trajectories=[cal_trajectory],
                s=shekel_s,
            )
            rf = ShekelReliabilityFunction(
                process_configs=shekel_configs,
                process_order=rf_process_order,
                s=shekel_s,
            )
            # Store calibrated Shekel params back into process configs so the
            # controller surrogate can use them without re-calibrating.
            for proc in current_processes:
                pname = proc['name']
                if pname in shekel_configs:
                    proc['surrogate_shekel_center'] = shekel_configs[pname].get('shekel_center')
                    proc['surrogate_shekel_sigma'] = shekel_configs[pname].get('shekel_sigma')
            print(f"  Reliability function: Shekel (s={shekel_s})")
        else:
            rf = ReliabilityFunction(
                process_configs=rf_process_configs,
                process_order=rf_process_order,
            )
            print(f"  Reliability function: Gaussian (classic)")
    else:
        rf = ReliabilityFunction()
        print(f"  Reliability function: Gaussian (legacy)")

    full_trajectories = []
    n = n_samples

    for i in range(n):
        # Build trajectory dict for reliability function (needs inputs with env)
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

    # Copy converted data to causaliT/data/ for training
    import shutil
    causalit_data_dir = REPO_ROOT / 'causaliT' / 'data' / 'azimuth_surrogate'
    if causalit_data_dir.exists():
        shutil.rmtree(causalit_data_dir)
    shutil.copytree(causalit_dir, causalit_data_dir)
    print(f"  Copied to: {causalit_data_dir}/ (ready for causaliT training)")

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
    print(f"  {causalit_data_dir}/ (causaliT training data)")

    print("\n" + "=" * 70)
    print("STEP 0 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"  causaliT data ready at: {causalit_data_dir}/")
    print("\nNext step: Run train_predictor.py to train uncertainty predictors")


if __name__ == '__main__':
    main()
