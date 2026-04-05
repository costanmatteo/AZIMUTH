#!/usr/bin/env python3
"""
Win rate vs Λ_MC scatter plot for AZIMUTH complexity sweep.

Standalone module — requires only numpy + matplotlib.

Usage:
    from plot_win_rate_vs_lambda_mc import plot_win_rate_vs_lambda_mc
    plot_win_rate_vs_lambda_mc('checkpoints/complexity_sweep', 'win_vs_lambda_mc.png')
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_lambda_mc(run_dir: Path) -> float | None:
    """Return scalar Λ_MC for a run, trying two sources."""
    # Primary: theoretical_analysis_data.json
    ta_path = run_dir / 'theoretical_analysis_data.json'
    if ta_path.exists():
        try:
            with open(ta_path) as f:
                ta = json.load(f)
            val = ta.get('lambda_mc', {}).get('lambda_mc')
            if val is not None:
                return float(val)
        except Exception:
            pass

    # Fallback: final_results.json → theoretical_analysis.lambda_mc
    fr_path = run_dir / 'final_results.json'
    if fr_path.exists():
        try:
            with open(fr_path) as f:
                fr = json.load(f)
            val = fr.get('theoretical_analysis', {}).get('lambda_mc')
            if val is not None:
                return float(val)
        except Exception:
            pass

    return None


def _load_run_info(run_dir: Path) -> dict | None:
    """Load st_params, train metrics and Λ_MC for a single run."""
    fr_path = run_dir / 'final_results.json'
    if not fr_path.exists():
        return None
    try:
        with open(fr_path) as f:
            data = json.load(f)
    except Exception:
        return None

    st = data.get('st_params')
    if not st:
        return None

    train = data.get('train', {})
    f_actual = train.get('F_actual_mean')
    f_baseline = train.get('F_baseline_mean')
    if f_actual is None or f_baseline is None:
        return None

    n_proc = data.get('n_processes')
    lmc = _load_lambda_mc(run_dir)

    return {
        'config_key': (st.get('n'), st.get('m'), st.get('rho'), n_proc),
        'lambda_mc': lmc,
        'win': float(f_actual) > float(f_baseline),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main plot function
# ─────────────────────────────────────────────────────────────────────────────

def plot_win_rate_vs_lambda_mc(
    sweep_dir,
    save_path=None,
    figsize=(8, 5),
    show_std=True,
    show_trend=True,
):
    """
    Scatter plot: # wins vs baseline (Y) against mean Λ_MC (X) per configuration.

    Parameters
    ----------
    sweep_dir : str or Path
        Root directory of the complexity sweep checkpoints.
    save_path : str or Path or None
        Where to save the PNG.  If None the figure is returned (not saved).
    figsize : tuple
        Figure size in inches.
    show_std : bool
        Show horizontal error bars (±σ of Λ_MC across seeds).
    show_trend : bool
        Overlay a dashed linear trend line.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure (always returned; also saved if *save_path* given).
    """
    sweep_dir = Path(sweep_dir)

    # ── Scan all run directories (nested: config_dir / run_dir) ──
    groups = defaultdict(lambda: {'lambda_mc': [], 'wins': 0, 'total': 0})

    for config_dir in sorted(sweep_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        for run_dir in sorted(config_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            info = _load_run_info(run_dir)
            if info is None:
                continue
            key = info['config_key']
            if info['lambda_mc'] is not None:
                groups[key]['lambda_mc'].append(info['lambda_mc'])
            if info['win']:
                groups[key]['wins'] += 1
            groups[key]['total'] += 1

    # ── Filter configs with at least one Λ_MC value ──
    xs, ys, xerr, win_rates = [], [], [], []
    n_skipped = 0

    for key, g in groups.items():
        if not g['lambda_mc']:
            n_skipped += 1
            continue
        arr = np.array(g['lambda_mc'])
        xs.append(arr.mean())
        xerr.append(arr.std() if len(arr) > 1 else 0.0)
        ys.append(g['wins'])
        wr = g['wins'] / g['total'] if g['total'] > 0 else 0.0
        win_rates.append(wr)

    xs = np.array(xs)
    ys = np.array(ys)
    xerr = np.array(xerr)
    win_rates = np.array(win_rates)

    # ── Build figure ──
    fig, ax = plt.subplots(figsize=figsize)

    if len(xs) == 0:
        ax.text(0.5, 0.5, 'No configurations with Λ_MC data available',
                ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_xlabel('Λ_MC  (mean over seeds)')
        ax.set_ylabel('# wins vs baseline')
        if save_path is not None:
            fig.savefig(str(save_path), dpi=150, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
        return fig

    norm = plt.Normalize(0, 1)
    cmap = plt.cm.RdYlGn

    # Error bars
    if show_std:
        ax.errorbar(xs, ys, xerr=xerr, fmt='none', ecolor='#999999',
                    elinewidth=0.8, capsize=2, capthick=0.6, zorder=1)

    # Scatter
    sc = ax.scatter(xs, ys, c=win_rates, cmap=cmap, norm=norm,
                    s=70, edgecolors='#333333', linewidths=0.5,
                    alpha=0.88, zorder=2)

    # Trend line
    if show_trend and len(xs) >= 3:
        coeffs = np.polyfit(xs, ys, 1)
        x_fit = np.linspace(xs.min(), xs.max(), 200)
        y_fit = np.polyval(coeffs, x_fit)
        ax.plot(x_fit, y_fit, '--', color='#555555', linewidth=1.0,
                alpha=0.7, label=f'trend (slope={coeffs[0]:.2f})', zorder=0)
        ax.legend(fontsize=8, loc='best', framealpha=0.6)

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('Win rate (%)', fontsize=9)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])

    ax.set_xlabel('Λ_MC  (mean over seeds)', fontsize=10)
    ax.set_ylabel('# wins vs baseline', fontsize=10)
    ax.set_title('Win count vs Λ_MC per configuration', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.25, linewidth=0.4)

    # Y-axis integer ticks
    y_max = int(ys.max()) + 1
    ax.set_yticks(range(0, y_max + 1))

    # Note about skipped configs
    if n_skipped > 0:
        ax.annotate(
            f'{n_skipped} config(s) excluded (no Λ_MC data)',
            xy=(0.02, 0.02), xycoords='axes fraction',
            fontsize=7, color='#888888', style='italic',
        )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot # wins vs Λ_MC')
    parser.add_argument('--sweep_dir', default='checkpoints/complexity_sweep',
                        help='Complexity sweep checkpoint directory')
    parser.add_argument('--output', default=None,
                        help='Output PNG path (default: <sweep_dir>/win_rate_vs_lambda_mc.png)')
    parser.add_argument('--no-std', action='store_true', help='Hide Λ_MC std error bars')
    parser.add_argument('--no-trend', action='store_true', help='Hide linear trend line')
    args = parser.parse_args()

    sd = Path(args.sweep_dir)
    out = Path(args.output) if args.output else sd / 'win_rate_vs_lambda_mc.png'

    fig = plot_win_rate_vs_lambda_mc(sd, out, show_std=not args.no_std,
                                     show_trend=not args.no_trend)
    plt.close(fig)
    print(f'Saved: {out}')
