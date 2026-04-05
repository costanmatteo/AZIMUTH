#!/usr/bin/env python3
"""
Scatter plot: Improvement vs baseline (%) vs Λ_MC for seed-sweep runs.

Each point = one run (seed_target × seed_baseline combination).
X = Λ_MC  (from theoretical_analysis_data.json → lambda_mc.lambda_mc)
Y = improvement_pct  (from final_results.json → train.improvement_pct)

Usage:
    python plot_improvement_vs_lambda_mc.py [--sweep_dir PATH] [--output PATH] [--no-trend]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── marker shapes cycled per seed_baseline ──────────────────────────────────
_MARKERS = ['o', 's', '^', 'D', 'x']


def _load_run(run_dir: Path) -> dict | None:
    """Load Λ_MC and improvement_pct from a single run directory."""
    results_file = run_dir / "final_results.json"
    if not results_file.exists():
        return None

    try:
        with open(results_file) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Warning: could not load {results_file}: {e}")
        return None

    # ── improvement_pct ─────────────────────────────────────────────────────
    train = data.get('train', {})
    improvement_pct = train.get('improvement_pct')
    if improvement_pct is None:
        # fallback: compute from F values
        f_actual = train.get('F_actual_mean')
        f_baseline = train.get('F_baseline_mean')
        if f_actual is not None and f_baseline is not None and f_baseline != 0:
            improvement_pct = (f_actual - f_baseline) / abs(f_baseline) * 100
        else:
            print(f"  Warning: {run_dir.name} — cannot compute improvement_pct, skipping")
            return None

    # ── Λ_MC ────────────────────────────────────────────────────────────────
    lambda_mc = None

    # primary: theoretical_analysis_data.json
    ta_file = run_dir / "theoretical_analysis_data.json"
    if ta_file.exists():
        try:
            with open(ta_file) as f:
                ta_data = json.load(f)
            lambda_mc = ta_data.get('lambda_mc', {}).get('lambda_mc')
        except Exception:
            pass

    # fallback: final_results.json → theoretical_analysis.lambda_mc
    if lambda_mc is None:
        lambda_mc = data.get('theoretical_analysis', {}).get('lambda_mc')

    if lambda_mc is None:
        print(f"  Warning: {run_dir.name} — Λ_MC not found, skipping")
        return None

    # ── seeds ───────────────────────────────────────────────────────────────
    sc = data.get('config', {}).get('scenarios', {})

    return {
        'run_name':      run_dir.name,
        'lambda_mc':     float(lambda_mc),
        'improvement_pct': float(improvement_pct),
        'seed_target':   sc.get('seed_target'),
        'seed_baseline': sc.get('seed_baseline'),
    }


def plot_improvement_vs_lambda_mc(
    sweep_dir: str | Path,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (6.0, 4.0),
    show_trend: bool = True,
) -> str | None:
    """
    Create scatter plot of improvement_pct vs Λ_MC.

    Parameters
    ----------
    sweep_dir : path to the sweep results directory
    save_path : where to save the PNG (default: sweep_dir/improvement_vs_lambda_mc.png)
    figsize   : matplotlib figure size
    show_trend: overlay a linear trend line

    Returns
    -------
    Path to the saved PNG, or None if no valid data.
    """
    sweep_dir = Path(sweep_dir)
    if save_path is None:
        save_path = sweep_dir / "improvement_vs_lambda_mc.png"
    save_path = Path(save_path)

    # ── load all runs ───────────────────────────────────────────────────────
    runs = []
    for d in sorted(sweep_dir.iterdir()):
        if not d.is_dir():
            continue
        r = _load_run(d)
        if r is not None:
            runs.append(r)

    if not runs:
        print("No valid runs with both Λ_MC and improvement_pct.")
        return None

    print(f"  Improvement vs Λ_MC plot: {len(runs)} runs loaded")

    # ── arrays ──────────────────────────────────────────────────────────────
    x = np.array([r['lambda_mc'] for r in runs])
    y = np.array([r['improvement_pct'] for r in runs])
    seed_targets = np.array([r['seed_target'] for r in runs])
    seed_baselines = np.array([r['seed_baseline'] for r in runs])

    unique_sb = sorted(set(sb for sb in seed_baselines if sb is not None))
    unique_st = sorted(set(st for st in seed_targets if st is not None))

    sb_to_marker = {sb: _MARKERS[i % len(_MARKERS)] for i, sb in enumerate(unique_sb)}
    # fallback marker for None
    sb_to_marker[None] = 'o'

    # ── colormap for seed_target ────────────────────────────────────────────
    if len(unique_st) <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.viridis
    st_norm = matplotlib.colors.Normalize(
        vmin=min(unique_st) if unique_st else 0,
        vmax=max(unique_st) if unique_st else 1,
    )
    st_mapper = matplotlib.cm.ScalarMappable(norm=st_norm, cmap=cmap)

    # ── plot ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    for r, xi, yi in zip(runs, x, y):
        st = r['seed_target']
        sb = r['seed_baseline']
        color = st_mapper.to_rgba(st) if st is not None else 'grey'
        marker = sb_to_marker.get(sb, 'o')
        ax.scatter(xi, yi, c=[color], marker=marker, s=38, alpha=0.75,
                   edgecolors='white', linewidths=0.3)

    # parity line
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    # trend line
    if show_trend and len(x) >= 2:
        coeffs = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = np.polyval(coeffs, x_line)
        ax.plot(x_line, y_line, '--', color='#555555', linewidth=1.0, alpha=0.7,
                label=f'trend (slope={coeffs[0]:.2f})')
        ax.legend(fontsize=7, loc='best', framealpha=0.6)

    # annotation: mean ± std
    mean_imp = np.mean(y)
    std_imp = np.std(y)
    ax.annotate(
        f'mean = {mean_imp:.2f}%\nstd  = {std_imp:.2f}%',
        xy=(0.02, 0.97), xycoords='axes fraction',
        fontsize=7, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#ccc'),
    )

    ax.set_xlabel('Λ_MC', fontsize=9)
    ax.set_ylabel('Improvement vs baseline (%)', fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2, linewidth=0.4)

    # ── colorbar for seed_target ────────────────────────────────────────────
    cb = fig.colorbar(st_mapper, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label('seed_target', fontsize=7)
    cb.ax.tick_params(labelsize=6)

    # ── legend for seed_baseline markers ────────────────────────────────────
    from matplotlib.lines import Line2D
    handles = []
    for sb in unique_sb:
        handles.append(Line2D(
            [0], [0], marker=sb_to_marker[sb], color='grey', linestyle='None',
            markersize=5, label=f'seed_b={sb}',
        ))
    if handles:
        leg = ax.legend(
            handles=handles, fontsize=6, loc='lower right',
            framealpha=0.7, title='seed_baseline', title_fontsize=6,
            handletextpad=0.3, borderpad=0.3,
        )
        ax.add_artist(leg)
        # re-add trend legend if present
        if show_trend and len(x) >= 2:
            ax.legend(fontsize=7, loc='upper right', framealpha=0.6)

    fig.tight_layout(pad=0.8)
    fig.savefig(str(save_path), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)

    print(f"  Saved: {save_path}")
    return str(save_path)


def main():
    parser = argparse.ArgumentParser(
        description='Scatter plot: Improvement vs baseline (%) vs Λ_MC')
    parser.add_argument('--sweep_dir', default='checkpoints/sweep',
                        help='Directory containing sweep run results')
    parser.add_argument('--output', default=None,
                        help='Output PNG path')
    parser.add_argument('--no-trend', action='store_true',
                        help='Disable linear trend line')
    parser.add_argument('--figsize', nargs=2, type=float, default=[6.0, 4.0],
                        metavar=('W', 'H'), help='Figure size in inches')
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        raise SystemExit(f"Error: sweep directory not found: {sweep_dir}")

    plot_improvement_vs_lambda_mc(
        sweep_dir=sweep_dir,
        save_path=args.output,
        figsize=tuple(args.figsize),
        show_trend=not args.no_trend,
    )


if __name__ == '__main__':
    main()
