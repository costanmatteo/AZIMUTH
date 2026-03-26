#!/usr/bin/env python3
"""
Generate aggregated PDF report from complexity sweep results.

Usage:
    python generate_complexity_sweep_report.py [--sweep_dir PATH] [--output complexity_sweep_report.pdf]

This script:
1. Loads results from all complexity sweep runs
2. Computes per-configuration win rates and aggregate statistics
3. Generates matplotlib plots embedded as base64
4. Renders a pixel-faithful HTML layout to PDF via WeasyPrint
   (landscape A4, Courier monospace, matches generate_sweep_report.py exactly)
"""

import argparse
import base64
import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── WeasyPrint for HTML → PDF ────────────────────────────────────────────────
try:
    from weasyprint import HTML as WPHtml, CSS
except ImportError:
    raise SystemExit(
        "WeasyPrint is required.  Install with:\n"
        "    pip install weasyprint --break-system-packages"
    )


# ════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_run_results(run_dir: Path) -> dict | None:
    results_file = run_dir / "final_results.json"
    if not results_file.exists():
        return None
    try:
        with open(results_file) as f:
            data = json.load(f)
        config = data.get('config', {})
        sc = config.get('scenarios', {})
        st = data.get('st_params', {})
        return {
            'run_name':          run_dir.name,
            'seed_target':       sc.get('seed_target'),
            'seed_baseline':     sc.get('seed_baseline'),
            # complexity parameters
            'st_n':              st.get('n')   if st else None,
            'st_m':              st.get('m')   if st else None,
            'st_rho':            st.get('rho') if st else None,
            'n_processes':       data.get('n_processes'),
            # train
            'F_star_train':      data.get('train', {}).get('F_star'),
            'F_baseline_train':  data.get('train', {}).get('F_baseline_mean'),
            'F_actual_train':    data.get('train', {}).get('F_actual_mean'),
            # test
            'F_star_test':       data.get('test', {}).get('F_star'),
            'F_baseline_test':   data.get('test', {}).get('F_baseline_mean'),
            'F_actual_test':     data.get('test', {}).get('F_actual_mean'),
        }
    except Exception as e:
        print(f"  Warning: could not load {results_file}: {e}")
        return None


def aggregate_results(sweep_dir: Path) -> pd.DataFrame:
    results = []
    # Results are nested: sweep_dir/config_dir/run_dir/final_results.json
    for config_dir in sorted(sweep_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        for run_dir in sorted(config_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            r = load_run_results(run_dir)
            if r is not None:
                results.append(r)
    if not results:
        return pd.DataFrame()
    print(f"Loaded {len(results)} runs")
    return pd.DataFrame(results)


def compute_win_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (st_n, st_m, st_rho[, n_processes]) and compute win rate per config."""
    df = df.copy()
    # win = controller closer to F* than baseline (gap-based, same as sweep.sh)
    gap_baseline = (df['F_star_test'] - df['F_baseline_test']).abs()
    gap_ctrl     = (df['F_star_test'] - df['F_actual_test']).abs()
    df['controller_wins'] = gap_ctrl < gap_baseline

    group_cols = ['st_n', 'st_m', 'st_rho']
    if 'n_processes' in df.columns and df['n_processes'].notna().any():
        group_cols.append('n_processes')

    grouped = df.groupby(group_cols).agg(
        n_runs=('controller_wins', 'count'),
        n_wins=('controller_wins', 'sum'),
        F_star_mean=('F_star_test', 'mean'),
        F_baseline_mean=('F_baseline_test', 'mean'),
        F_actual_mean=('F_actual_test', 'mean'),
    ).reset_index()

    grouped['win_rate_pct'] = 100.0 * grouped['n_wins'] / grouped['n_runs']
    grouped['mean_improvement_pct'] = (
        (grouped['F_actual_mean'] - grouped['F_baseline_mean'])
        / grouped['F_baseline_mean'].abs().clip(lower=1e-10) * 100
    )
    return grouped


def compute_stats(df: pd.DataFrame, win_df: pd.DataFrame) -> dict:
    """Compute aggregate KPI statistics for the report."""
    n_runs   = len(df)
    n_cfgs   = len(win_df)
    # gap-based win rate (same as sweep.sh): |gap_ctrl| < |gap_baseline|
    gb = (df['F_star_test'] - df['F_baseline_test']).abs()
    gc = (df['F_star_test'] - df['F_actual_test']).abs()
    overall_win_rate = float(100.0 * (gc < gb).mean())
    best_row = win_df.loc[win_df['win_rate_pct'].idxmax()]
    worst_row = win_df.loc[win_df['win_rate_pct'].idxmin()]

    has_nproc = 'n_processes' in win_df.columns and win_df['n_processes'].notna().any()

    def _cfg_label(row):
        s = f"n={int(row['st_n'])} m={int(row['st_m'])} ρ={row['st_rho']:.2f}"
        if has_nproc and pd.notna(row.get('n_processes')):
            s += f" p={int(row['n_processes'])}"
        return s

    # ── gap-based statistics (same as sweep report) ──
    gb_tr = (df['F_star_train']  - df['F_baseline_train']).abs()
    gc_tr = (df['F_star_train']  - df['F_actual_train']).abs()
    gd_tr = gb_tr - gc_tr                              # positive = ctrl better

    gb_te = (df['F_star_test']   - df['F_baseline_test']).abs()
    gc_te = (df['F_star_test']   - df['F_actual_test']).abs()
    gd_te = gb_te - gc_te

    wins = (gc_te.abs() < gb_te.abs()).sum()

    best_idx  = gc_te.abs().idxmin()
    worst_idx = gc_te.abs().idxmax()

    return {
        'n_runs':           n_runs,
        'n_cfgs':           n_cfgs,
        'overall_win_rate': overall_win_rate,
        'wins':             int(wins),
        'win_rate':         100.0 * wins / n_runs,
        'best_cfg':         _cfg_label(best_row),
        'best_wr':          float(best_row['win_rate_pct']),
        'worst_cfg':        _cfg_label(worst_row),
        'worst_wr':         float(worst_row['win_rate_pct']),
        'median_wr':        float(win_df['win_rate_pct'].median()),
        'has_nproc':        has_nproc,
        'df':               df,
        'win_df':           win_df,
        # train gaps
        'gb_tr_min':  gb_tr.min(),  'gb_tr_med':  gb_tr.median(),  'gb_tr_max':  gb_tr.max(),
        'gc_tr_min':  gc_tr.min(),  'gc_tr_med':  gc_tr.median(),  'gc_tr_max':  gc_tr.max(),
        'gd_tr_min':  gd_tr.min(),  'gd_tr_med':  gd_tr.median(),  'gd_tr_max':  gd_tr.max(),
        # test gaps
        'gb_te_min':  gb_te.min(),  'gb_te_med':  gb_te.median(),  'gb_te_max':  gb_te.max(),
        'gc_te_min':  gc_te.min(),  'gc_te_med':  gc_te.median(),  'gc_te_max':  gc_te.max(),
        'gd_te_min':  gd_te.min(),  'gd_te_med':  gd_te.median(),  'gd_te_max':  gd_te.max(),
        # best / worst run
        'best_run':   df.loc[best_idx,  'run_name'],
        'best_gap':   gc_te.loc[best_idx],
        'worst_run':  df.loc[worst_idx, 'run_name'],
        'worst_gap':  gc_te.loc[worst_idx],
        # generalisation
        'degrad':     gc_te.median() - gc_tr.median(),
        # F*
        'fstar_min':  df['F_star_test'].min(),
        'fstar_med':  df['F_star_test'].median(),
        'fstar_max':  df['F_star_test'].max(),
    }


# ════════════════════════════════════════════════════════════════════════════
# 1b.  CONFIG LOADING  (controller params + process parameter ranges)
# ════════════════════════════════════════════════════════════════════════════

def load_sweep_config(sweep_dir: Path) -> dict | None:
    """Load controller config from the first run and collect all ST param values."""
    first_cfg = None
    all_st = []

    for config_dir in sorted(sweep_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        for run_dir in sorted(config_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            results_file = run_dir / "final_results.json"
            if not results_file.exists():
                continue
            try:
                with open(results_file) as f:
                    data = json.load(f)
                if first_cfg is None:
                    first_cfg = {
                        'config':       data.get('config', {}),
                        'dataset_mode': data.get('dataset_mode', 'unknown'),
                        'n_processes':  data.get('n_processes'),
                    }
                st = data.get('st_params')
                if st:
                    all_st.append(st)
            except Exception:
                continue

    if first_cfg is None:
        return None

    first_cfg['all_st_params'] = all_st

    # ── Load uncertainty predictor config ──
    try:
        import sys
        # Project root = two levels up from this script (euler/complexity_sweep/ → root)
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent
        if not (project_root / 'configs').is_dir():
            # Fallback: walk up from sweep_dir
            project_root = sweep_dir.resolve()
            for _ in range(8):
                if (project_root / 'configs').is_dir():
                    break
                project_root = project_root.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from configs.uncertainty_config import (
            GLOBAL_UNCERTAINTY_CONFIG, DEFAULT_ST_UNCERTAINTY_PREDICTOR
        )
        first_cfg['uncertainty_config'] = GLOBAL_UNCERTAINTY_CONFIG
        first_cfg['uncertainty_model'] = DEFAULT_ST_UNCERTAINTY_PREDICTOR.get('model', {})
        first_cfg['uncertainty_training'] = DEFAULT_ST_UNCERTAINTY_PREDICTOR.get('training', {})
    except Exception as e:
        print(f"  Warning: could not load uncertainty config: {e}")
        first_cfg['uncertainty_config'] = {}
        first_cfg['uncertainty_model'] = {}
        first_cfg['uncertainty_training'] = {}

    # ── Load surrogate config ──
    try:
        from configs.surrogate_config import SURROGATE_CONFIG
        first_cfg['surrogate_config'] = SURROGATE_CONFIG
    except Exception as e:
        print(f"  Warning: could not load surrogate config: {e}")
        first_cfg['surrogate_config'] = {}

    return first_cfg


def _range_str(values) -> str:
    """Format a list of numeric values as 'min … max' or single value."""
    vals = [v for v in values if v is not None]
    if not vals:
        return '?'
    lo, hi = min(vals), max(vals)
    if lo == hi:
        return f"{lo}"
    if isinstance(lo, int) and isinstance(hi, int):
        return f"{lo} … {hi}"
    return f"{lo:.3g} … {hi:.3g}"


# ════════════════════════════════════════════════════════════════════════════
# 2.  PLOTS  (returns base64 PNG string)
# ════════════════════════════════════════════════════════════════════════════

_GREEN  = '#1D9E75'
_RED    = '#D85A30'
_AMBER  = '#BA7517'
_MONO   = 'DejaVu Sans Mono'


def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def plot_marginals(win_df: pd.DataFrame) -> str:
    """Bar/scatter of win rate vs each complexity parameter (marginal effects)."""
    has_nproc = 'n_processes' in win_df.columns and win_df['n_processes'].notna().any()
    n_panels  = 4 if has_nproc else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(4.8 * n_panels, 3.6))
    if n_panels == 1:
        axes = [axes]

    norm = plt.Normalize(0, 100)
    cmap = plt.cm.RdYlGn

    params = [
        ('st_n',   'n  (input variables)', axes[0]),
        ('st_m',   'm  (cascaded stages)',  axes[1]),
        ('st_rho', 'ρ  (noise intensity)',  axes[2]),
    ]
    if has_nproc:
        params.append(('n_processes', 'n_processes  (chain)', axes[3]))

    for param, label, ax in params:
        df_s = win_df.dropna(subset=[param]).sort_values(param)
        if param in ('st_n', 'st_m', 'n_processes'):
            grp = df_s.groupby(param)['win_rate_pct'].agg(['mean', 'std', 'count'])
            x   = grp.index.values
            y   = grp['mean'].values
            yerr = grp['std'].values / np.sqrt(np.maximum(grp['count'].values, 1))
            bars = ax.bar(x, y, yerr=yerr, color='steelblue', edgecolor='#444',
                          alpha=0.75, capsize=3, width=0.6 * (x[1] - x[0]) if len(x) > 1 else 0.5)
            for bar, val in zip(bars, y):
                bar.set_facecolor(cmap(norm(val)))
        else:
            ax.scatter(df_s[param], df_s['win_rate_pct'],
                       c=df_s['win_rate_pct'], cmap='RdYlGn', vmin=0, vmax=100,
                       s=55, edgecolors='#444', linewidths=0.4, alpha=0.85)
            if len(df_s) >= 5:
                z = np.polyfit(df_s[param], df_s['win_rate_pct'], deg=2)
                p = np.poly1d(z)
                xs = np.linspace(df_s[param].min(), df_s[param].max(), 120)
                ax.plot(xs, np.clip(p(xs), 0, 100), 'b--', lw=1.4, alpha=0.55)
        ax.axhline(50, color='#999', lw=0.7, ls=':')
        ax.set_ylim(-5, 105)
        ax.set_xlabel(label, fontsize=7, fontfamily=_MONO)
        ax.set_ylabel('Win Rate (%)', fontsize=7, fontfamily=_MONO)
        ax.set_title(f'Win rate vs {label.split("(")[0].strip()}',
                     fontsize=7.5, fontfamily=_MONO, fontweight='bold')
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.18, lw=0.4)

    fig.tight_layout(pad=0.6)
    return _b64(fig)


def _make_heatmap(win_df: pd.DataFrame, px: str, py: str,
                  lx: str, ly: str) -> str:
    """2D win-rate heatmap for two parameters, averaged over the third."""
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    wdf = win_df.dropna(subset=[px, py]).copy()

    n_bins = min(8, len(wdf[px].unique()))
    wdf['xb'] = pd.cut(wdf[px], bins=n_bins, include_lowest=True)
    wdf['yb'] = pd.cut(wdf[py], bins=n_bins, include_lowest=True)
    pivot = wdf.pivot_table(values='win_rate_pct', index='yb', columns='xb', aggfunc='mean')

    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center',
                fontsize=8, transform=ax.transAxes)
        fig.tight_layout(pad=0.6)
        return _b64(fig)

    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto',
                   vmin=0, vmax=100, origin='lower')
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cb.ax.tick_params(labelsize=5)
    cb.set_label('Win %', fontsize=6, fontfamily=_MONO)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=35,
                       ha='right', fontsize=5, fontfamily=_MONO)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index], fontsize=5, fontfamily=_MONO)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                col = 'white' if v < 30 or v > 70 else '#222'
                ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                        color=col, fontsize=5.5, fontweight='bold', fontfamily=_MONO)

    ax.set_xlabel(lx, fontsize=7, fontfamily=_MONO)
    ax.set_ylabel(ly, fontsize=7, fontfamily=_MONO)
    ax.set_title(f'Win rate: {lx} × {ly}', fontsize=7.5,
                 fontfamily=_MONO, fontweight='bold')
    fig.tight_layout(pad=0.6)
    return _b64(fig)


def plot_heatmap_n_rho(win_df: pd.DataFrame) -> str:
    return _make_heatmap(win_df, 'st_n', 'st_rho', 'n (inputs)', 'ρ (noise)')

def plot_heatmap_n_m(win_df: pd.DataFrame) -> str:
    return _make_heatmap(win_df, 'st_n', 'st_m', 'n (inputs)', 'm (stages)')

def plot_heatmap_m_rho(win_df: pd.DataFrame) -> str:
    return _make_heatmap(win_df, 'st_m', 'st_rho', 'm (stages)', 'ρ (noise)')


def plot_3d_scatter(win_df: pd.DataFrame) -> str:
    """3D scatter: (n, m, rho) → colour = win rate."""
    fig = plt.figure(figsize=(5.0, 4.0))
    ax  = fig.add_subplot(111, projection='3d')
    sc  = ax.scatter(
        win_df['st_n'], win_df['st_m'], win_df['st_rho'],
        c=win_df['win_rate_pct'], cmap='RdYlGn', vmin=0, vmax=100,
        s=40 + win_df['n_runs'] * 3,
        alpha=0.82, edgecolors='#333', linewidths=0.3,
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.08)
    cb.ax.tick_params(labelsize=5)
    cb.set_label('Win %', fontsize=6, fontfamily=_MONO)
    ax.set_xlabel('n', fontsize=6, fontfamily=_MONO, labelpad=4)
    ax.set_ylabel('m', fontsize=6, fontfamily=_MONO, labelpad=4)
    ax.set_zlabel('ρ', fontsize=6, fontfamily=_MONO, labelpad=4)
    ax.set_title('Win rate in (n, m, ρ) space', fontsize=7.5,
                 fontfamily=_MONO, fontweight='bold')
    ax.tick_params(labelsize=5)
    fig.tight_layout(pad=0.6)
    return _b64(fig)


def plot_winrate_distribution(win_df: pd.DataFrame) -> str:
    """Histogram of win-rate distribution across configurations."""
    fig, axes = plt.subplots(1, 2, figsize=(4.2, 3.4))

    # left: histogram
    ax = axes[0]
    wr = win_df['win_rate_pct'].dropna()
    bins = np.linspace(0, 100, 21)
    ax.hist(wr, bins=bins, color='steelblue', edgecolor='white',
            linewidth=0.4, alpha=0.80)
    ax.axvline(50,  color=_AMBER,  lw=0.9, ls='--', label='50 %')
    ax.axvline(wr.mean(), color=_GREEN, lw=0.9, ls='-',  label=f'mean {wr.mean():.1f}%')
    ax.set_xlabel('Win rate (%)', fontsize=7, fontfamily=_MONO)
    ax.set_ylabel('# configs', fontsize=7, fontfamily=_MONO)
    ax.set_title('Win-rate distribution', fontsize=7.5,
                 fontfamily=_MONO, fontweight='bold')
    ax.legend(fontsize=5.5, framealpha=0.6)
    ax.tick_params(labelsize=6)
    ax.grid(True, axis='y', alpha=0.18, lw=0.4)

    # right: sorted win rates (waterfall)
    ax = axes[1]
    wr_sorted = wr.sort_values().reset_index(drop=True)
    colors_bar = [_GREEN if v >= 50 else _RED for v in wr_sorted]
    ax.bar(range(len(wr_sorted)), wr_sorted, color=colors_bar,
           width=1.0, linewidth=0)
    ax.axhline(50, color='black', lw=0.6)
    ax.set_xlabel('Config (sorted)', fontsize=7, fontfamily=_MONO)
    ax.set_ylabel('Win rate (%)', fontsize=7, fontfamily=_MONO)
    ax.set_title('Win rate per config (sorted)', fontsize=7.5,
                 fontfamily=_MONO, fontweight='bold')
    ax.tick_params(labelsize=6)
    ax.grid(True, axis='y', alpha=0.18, lw=0.4)

    fig.tight_layout(pad=0.6)
    return _b64(fig)


# ════════════════════════════════════════════════════════════════════════════
# 3.  CSS  (identical to generate_sweep_report.py)
# ════════════════════════════════════════════════════════════════════════════

PAGE_CSS = """
@page {
  size: 297mm 210mm;
  margin: 0;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Courier New', Courier, monospace;
  font-size: 9px;
  line-height: 1.45;
  color: #1a1a1a;
  background: white;
}
.page {
  width: 297mm;
  height: 210mm;
  padding: 13px 18px 9px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  page-break-after: always;
}
.page:last-child { page-break-after: auto; }
.page-flow {
  width: 297mm;
  padding: 13px 18px 9px;
  overflow: visible;
  page-break-after: auto;
}
.page-flow .tbl thead { display: table-header-group; }
.page-flow .tbl tr    { page-break-inside: avoid; }

.hdr-row { display: flex; justify-content: space-between; align-items: baseline;
           margin-bottom: 2px; }
.title   { font-size: 11px; font-weight: 500; }
.meta    { font-size: 8px; color: #666; }
.rule-heavy { border: none; border-top: 1px solid #1a1a1a; margin: 3px 0 7px; }
.rule-thin  { border: none; border-top: 0.5px solid #aaa;  margin: 2px 0 5px; }

.g  { color: #1D9E75; }
.r  { color: #D85A30; }
.a  { color: #BA7517; }
.dg { color: #1D9E75; font-weight: 500; }
.dr { color: #D85A30; font-weight: 500; }
.da { color: #BA7517; font-weight: 500; }

/* ── KPI row ── */
.kpi-row { display: table; width: 100%; border: 0.5px solid #ccc;
           margin-bottom: 7px; table-layout: fixed; }
.kpi     { display: table-cell; padding: 5px 8px;
           border-right: 0.5px solid #ccc; vertical-align: top; }
.kpi:last-child { border-right: none; }
.kpi-l   { font-size: 7px; color: #888; text-transform: uppercase;
           letter-spacing: 0.4px; }
.kpi-v   { font-size: 13px; font-weight: 500; margin: 1px 0; }
.kpi-s   { font-size: 7px; color: #888; }

/* ── section headings ── */
.sec-head { font-size: 7px; font-weight: 500; text-transform: uppercase;
            letter-spacing: 0.9px; color: #888; margin-bottom: 3px;
            margin-top: 5px; }
.blk-title { font-size: 7.5px; font-weight: 500; text-transform: uppercase;
             letter-spacing: 0.5px; color: #888; margin-bottom: 3px;
             border-left: 2px solid #1a1a1a; padding-left: 4px; }

/* ── stat rows ── */
.stat-lbl    { font-size: 7.5px; font-weight: 500; color: #1a1a1a;
               margin-bottom: 2px; margin-top: 4px; }
.stat-sublbl { font-size: 7px; color: #888; }
.row { display: flex; justify-content: space-between; padding: 1.5px 0;
       border-bottom: 0.5px solid #eee; font-size: 8px; gap: 6px; }
.row:last-child { border-bottom: none; }
.rk  { color: #888; }
.rv  { font-weight: 500; }

/* ── four-column stats grid ── */
.stats-grid  { display: table; width: 100%; margin-bottom: 7px;
               table-layout: fixed; }
.sg-col      { display: table-cell; padding-right: 12px;
               vertical-align: top; }
.sg-col:last-child { padding-right: 0; }
.sg-inner    { display: table; width: 100%; table-layout: fixed; }
.sg-sub      { display: table-cell; padding-right: 5px; vertical-align: top; }
.sg-sub:last-child { padding-right: 0; }

/* ── plot grid ── */
.plot-grid { display: flex; width: 100%; flex: 1; gap: 6px; }
.plot-cell { flex: 1 1 0; min-width: 0; }
.plot-img  { width: 100%; height: auto; display: block; }
.plot-cap  { font-size: 6.5px; color: #888; margin-top: 2px;
             font-style: italic; line-height: 1.3; }

/* ── runs table ── */
.tbl { width: 100%; border-collapse: collapse; font-size: 8px; }
.tbl th {
  text-align: left; padding: 3px 4px;
  background: #f5f5f5; border-bottom: 0.5px solid #1a1a1a;
  font-size: 7.5px; font-weight: 500; color: #888; white-space: nowrap;
}
.tbl th .def { font-size: 6.5px; font-weight: 400; color: #aaa;
               display: block; letter-spacing: 0; text-transform: none; }
.tbl td { padding: 2px 4px; border-bottom: 0.5px solid #eee;
          white-space: nowrap; }
.tbl tr:nth-child(even) td { background: #fafafa; }
.tbl tr:last-child td { border-bottom: none; }

/* ── footer ── */
.footer { margin-top: auto; border-top: 1px solid #1a1a1a; padding-top: 3px;
          display: flex; justify-content: space-between;
          font-size: 7px; color: #888; }

/* ── legend ── */
.legend { font-size: 7px; color: #888; margin-top: 4px; }

/* ── config section ── */
.cfg-grid  { display: table; width: 100%; table-layout: fixed;
             margin-bottom: 5px; }
.cfg-col   { display: table-cell; vertical-align: top; padding-right: 12px; }
.cfg-col:last-child { padding-right: 0; }
.cfg-row   { display: flex; justify-content: space-between; padding: 1px 0;
             border-bottom: 0.5px solid #eee; font-size: 7.5px; }
.cfg-row:last-child { border-bottom: none; }
.cfg-k     { color: #888; }
.cfg-v     { font-weight: 500; }
.cfg-sub   { font-size: 7px; color: #aaa; margin-left: 6px; }
"""


# ════════════════════════════════════════════════════════════════════════════
# 4.  HTML ASSEMBLY
# ════════════════════════════════════════════════════════════════════════════

def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return f'{float(v):.4f}'

def _pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return f'{float(v):.1f}%'

def _sign(v) -> str:
    return f"+{v:.3f}" if v >= 0 else f"{v:.3f}"

def _wr_cls(v: float) -> str:
    if v >= 70:   return 'dg'
    if v >= 40:   return 'da'
    return 'dr'


def _cfg_row(key: str, val, sub: str = '') -> str:
    sub_html = f'<span class="cfg-sub">{sub}</span>' if sub else ''
    return f'<div class="cfg-row"><span class="cfg-k">{key}{sub_html}</span><span class="cfg-v">{val}</span></div>'


def build_config_html(sweep_cfg: dict | None) -> str:
    """Build a compact HTML section showing controller config and process parameter ranges."""
    if sweep_cfg is None:
        return ''

    cfg = sweep_cfg.get('config', {})
    pg = cfg.get('policy_generator', {})
    tr = cfg.get('training', {})
    sc = cfg.get('scenarios', {})
    cl = tr.get('curriculum_learning', {})
    sr = cfg.get('surrogate', {})
    val = cfg.get('validation', {})
    dataset_mode = sweep_cfg.get('dataset_mode', '?')
    n_procs = sweep_cfg.get('n_processes', '?')
    all_st = sweep_cfg.get('all_st_params', [])

    # ── Controller column ──
    arch = pg.get('architecture', '?')
    hidden = pg.get('hidden_sizes', '?')
    if arch == 'custom' and hidden:
        arch_str = f"custom {hidden}"
    else:
        arch_str = arch

    sched = tr.get('lr_scheduler')
    if isinstance(sched, dict):
        sched_str = f"{sched.get('type', '?')}"
        if sched.get('T_max'):
            sched_str += f" T_max={sched['T_max']}"
    else:
        sched_str = 'none'

    cl_str = 'off'
    if cl.get('enabled'):
        cl_str = (f"on &middot; warmup {cl.get('warmup_fraction', '?')} &middot; "
                  f"&#955;bc {cl.get('lambda_bc_start', '?')}&#8594;{cl.get('lambda_bc_end', '?')} "
                  f"&middot; {cl.get('reliability_weight_curve', '?')}")

    ctrl_html = (
        _cfg_row('Architecture', arch_str)
        + _cfg_row('Dropout', pg.get('dropout', '?'))
        + _cfg_row('BatchNorm', pg.get('use_batchnorm', '?'))
        + _cfg_row('Scenario encoder', pg.get('use_scenario_encoder', '?'))
        + _cfg_row('Epochs', tr.get('epochs', '?'))
        + _cfg_row('Batch size', tr.get('batch_size', '?'))
        + _cfg_row('Learning rate', tr.get('learning_rate', '?'))
        + _cfg_row('Optimizer', tr.get('optimizer', '?'))
        + _cfg_row('Weight decay', tr.get('weight_decay', '?'))
        + _cfg_row('Grad clip norm', tr.get('gradient_clip_norm', 'none'))
        + _cfg_row('LR scheduler', sched_str)
    )

    # ── Training details column ──
    train_html = (
        _cfg_row('&#955;bc', tr.get('lambda_bc', '?'))
        + _cfg_row('Reliability scale', tr.get('reliability_loss_scale', '?'))
        + _cfg_row('Patience', tr.get('patience', '?'))
        + _cfg_row('Early stop metric', tr.get('early_stopping_metric', '?'))
        + _cfg_row('Curriculum', cl_str)
        + _cfg_row('Surrogate', sr.get('type', '?'))
        + _cfg_row('Deterministic', sr.get('use_deterministic_sampling', '?'))
        + _cfg_row('Validation', f"within={val.get('within_scenario_enabled', '?')} "
                   f"split={val.get('within_scenario_split', '?')}")
        + _cfg_row('n_train scenarios', sc.get('n_train', '?'))
        + _cfg_row('n_test scenarios', sc.get('n_test', '?'))
        + _cfg_row('seed_target', f"{sc.get('seed_target', '?')} (varies per run)")
        + _cfg_row('seed_baseline', f"{sc.get('seed_baseline', '?')} (varies per run)")
    )

    # ── Uncertainty predictor column ──
    up_cfg = sweep_cfg.get('uncertainty_config', {})
    up_model = sweep_cfg.get('uncertainty_model', {})
    up_train = sweep_cfg.get('uncertainty_training', {})

    up_html = (
        _cfg_row('Method', up_cfg.get('uncertainty_method', '?'))
        + _cfg_row('SWAG start epoch', up_cfg.get('swag_start_epoch', '?'))
        + _cfg_row('SWAG LR', up_cfg.get('swag_learning_rate', '?'))
        + _cfg_row('SWAG max rank', up_cfg.get('swag_max_rank', '?'))
        + _cfg_row('SWAG n samples', up_cfg.get('swag_n_samples', '?'))
    )
    if up_model:
        up_html += (
            _cfg_row('Architecture', up_model.get('hidden_sizes', '?'))
            + _cfg_row('Dropout', up_model.get('dropout_rate', '?'))
            + _cfg_row('BatchNorm', up_model.get('use_batchnorm', '?'))
        )
    if up_train:
        up_html += (
            _cfg_row('Epochs', up_train.get('epochs', '?'))
            + _cfg_row('Batch size', up_train.get('batch_size', '?'))
            + _cfg_row('Learning rate', up_train.get('learning_rate', '?'))
            + _cfg_row('Patience', up_train.get('patience', '?'))
            + _cfg_row('Loss', up_train.get('loss_type', '?'))
        )

    # ── Surrogate column ──
    surr_cfg = sweep_cfg.get('surrogate_config', {})
    surr_model = surr_cfg.get('model', {})
    surr_train = surr_cfg.get('training', {})

    surr_html = _cfg_row('Type', sr.get('type', '?'))
    if sr.get('type') == 'casualit' and surr_model:
        surr_html += (
            _cfg_row('CasualiT model', surr_model.get('casualit_model', '?'))
            + _cfg_row('d_model enc', surr_model.get('d_model_enc', '?'))
            + _cfg_row('d_model dec', surr_model.get('d_model_dec', '?'))
            + _cfg_row('d_ff', surr_model.get('d_ff', '?'))
            + _cfg_row('e_layers', surr_model.get('e_layers', '?'))
            + _cfg_row('d_layers', surr_model.get('d_layers', '?'))
            + _cfg_row('n_heads', surr_model.get('n_heads', '?'))
            + _cfg_row('Dropout', f"{surr_model.get('dropout_emb', '?')} / {surr_model.get('dropout_attn_out', '?')} / {surr_model.get('dropout_ff', '?')}", 'emb / attn / ff')
            + _cfg_row('Epochs', surr_train.get('max_epochs', '?'))
            + _cfg_row('Batch size', surr_train.get('batch_size', '?'))
            + _cfg_row('Learning rate', surr_train.get('learning_rate', '?'))
            + _cfg_row('Patience', surr_train.get('patience', '?'))
            + _cfg_row('Loss weights', f"x={surr_train.get('loss_weight_x', '?')} y={surr_train.get('loss_weight_y', '?')}")
        )
    elif sr.get('type') == 'reliability_function':
        surr_html += _cfg_row('Description', 'Mathematical formula')
        surr_html += _cfg_row('Deterministic', sr.get('use_deterministic_sampling', '?'))

    # ── Process column (with ranges from all runs) ──
    proc_html = _cfg_row('Dataset mode', dataset_mode)
    proc_html += _cfg_row('N processes', n_procs)

    if all_st and dataset_mode == 'st':
        proc_html += (
            _cfg_row('ST n', _range_str([s.get('n') for s in all_st]), 'input vars')
            + _cfg_row('ST m', _range_str([s.get('m') for s in all_st]), 'stages')
            + _cfg_row('ST p', _range_str([s.get('p') for s in all_st]), 'outputs')
            + _cfg_row('ST me', _range_str([s.get('me') for s in all_st]), 'env vars')
            + _cfg_row('ST &#945;', _range_str([s.get('alpha') for s in all_st]), 'shift')
            + _cfg_row('ST &#947;', _range_str([s.get('gamma') for s in all_st]), 'mult')
            + _cfg_row('ST &#961;', _range_str([s.get('rho') for s in all_st]), 'noise')
            + _cfg_row('env_mode', _range_str([s.get('env_mode') for s in all_st]))
            + _cfg_row('x_domain', all_st[0].get('x_domain', '?'))
        )

    return f"""
  <div class="sec-head">00 &#8212; configuration</div>
  <hr class="rule-thin">
  <div class="cfg-grid">
    <div class="cfg-col">
      <div class="blk-title">Controller &middot; policy generator</div>
      {ctrl_html}
    </div>
    <div class="cfg-col">
      <div class="blk-title">Controller &middot; training</div>
      {train_html}
    </div>
    <div class="cfg-col">
      <div class="blk-title">Uncertainty predictor</div>
      {up_html}
    </div>
    <div class="cfg-col">
      <div class="blk-title">Surrogate</div>
      {surr_html}
    </div>
    <div class="cfg-col">
      <div class="blk-title">Processes &middot; parameter ranges</div>
      {proc_html}
    </div>
  </div>
"""


def build_page1_html(s: dict, now: datetime,
                     b64_marginals: str, b64_wr_dist: str,
                     sweep_dir: str, config_html: str = '') -> str:
    n     = s['n_runs']
    nc    = s['n_cfgs']
    ts    = now.strftime('%Y-%m-%d &nbsp;%H:%M:%S')
    wr    = s['overall_win_rate']
    wr_cls = _wr_cls(wr)

    return f"""
<div class="page">
  <div class="hdr-row">
    <span class="title">Complexity Sweep Report</span>
    <span class="meta">{ts} &nbsp;·&nbsp; {sweep_dir} &nbsp;·&nbsp; {n} runs &nbsp;·&nbsp; {nc} configs &nbsp;·&nbsp; page 1 / 3</span>
  </div>
  <hr class="rule-heavy">

  <div class="kpi-row">
    <div class="kpi">
      <div class="kpi-l">Total runs</div>
      <div class="kpi-v">{n}</div>
      <div class="kpi-s">{nc} unique configs</div>
    </div>
    <div class="kpi">
      <div class="kpi-l">Controller wins</div>
      <div class="kpi-v g">{s['wins']}/{n}</div>
      <div class="kpi-s">win rate {s['win_rate']:.1f}%</div>
    </div>
    <div class="kpi">
      <div class="kpi-l">Median config win rate</div>
      <div class="kpi-v">{s['median_wr']:.1f}%</div>
      <div class="kpi-s">across {nc} configurations</div>
    </div>
    <div class="kpi">
      <div class="kpi-l">Best config</div>
      <div class="kpi-v g">{s['best_wr']:.1f}%</div>
      <div class="kpi-s">{s['best_cfg']}</div>
    </div>
    <div class="kpi">
      <div class="kpi-l">Worst config</div>
      <div class="kpi-v r">{s['worst_wr']:.1f}%</div>
      <div class="kpi-s">{s['worst_cfg']}</div>
    </div>
  </div>

  {config_html}

  <div class="sec-head">01 &#8212; aggregate statistics</div>
  <hr class="rule-thin">

  <div class="stats-grid">
    <!-- train -->
    <div class="sg-col">
      <div class="blk-title">Train split</div>
      <div class="sg-inner">
        <div class="sg-sub">
          <div class="stat-lbl">Gap baseline</div>
          <div class="stat-sublbl">F* &#8722; F'</div>
          <div class="row"><span class="rk">Min</span><span class="rv">{_fmt(s['gb_tr_min'])}</span></div>
          <div class="row"><span class="rk">Median</span><span class="rv">{_fmt(s['gb_tr_med'])}</span></div>
          <div class="row"><span class="rk">Max</span><span class="rv">{_fmt(s['gb_tr_max'])}</span></div>
        </div>
        <div class="sg-sub">
          <div class="stat-lbl">Gap ctrl</div>
          <div class="stat-sublbl">F* &#8722; F</div>
          <div class="row"><span class="rk">Min</span><span class="rv g">{_fmt(s['gc_tr_min'])}</span></div>
          <div class="row"><span class="rk">Median</span><span class="rv">{_fmt(s['gc_tr_med'])}</span></div>
          <div class="row"><span class="rk">Max</span><span class="rv r">{_fmt(s['gc_tr_max'])}</span></div>
        </div>
        <div class="sg-sub">
          <div class="stat-lbl">Gap &#916;</div>
          <div class="stat-sublbl">(F*&#8722;F') &#8722; (F*&#8722;F)</div>
          <div class="row"><span class="rk">Min</span><span class="rv">{_sign(s['gd_tr_min'])}</span></div>
          <div class="row"><span class="rk">Median</span><span class="rv">{_sign(s['gd_tr_med'])}</span></div>
          <div class="row"><span class="rk">Max</span><span class="rv">{_sign(s['gd_tr_max'])}</span></div>
        </div>
      </div>
    </div>
    <!-- test -->
    <div class="sg-col">
      <div class="blk-title">Test split</div>
      <div class="sg-inner">
        <div class="sg-sub">
          <div class="stat-lbl">Gap baseline</div>
          <div class="stat-sublbl">F* &#8722; F'</div>
          <div class="row"><span class="rk">Min</span><span class="rv">{_fmt(s['gb_te_min'])}</span></div>
          <div class="row"><span class="rk">Median</span><span class="rv">{_fmt(s['gb_te_med'])}</span></div>
          <div class="row"><span class="rk">Max</span><span class="rv">{_fmt(s['gb_te_max'])}</span></div>
        </div>
        <div class="sg-sub">
          <div class="stat-lbl">Gap ctrl</div>
          <div class="stat-sublbl">F* &#8722; F</div>
          <div class="row"><span class="rk">Min</span><span class="rv g">{_fmt(s['gc_te_min'])}</span></div>
          <div class="row"><span class="rk">Median</span><span class="rv">{_fmt(s['gc_te_med'])}</span></div>
          <div class="row"><span class="rk">Max</span><span class="rv r">{_fmt(s['gc_te_max'])}</span></div>
        </div>
        <div class="sg-sub">
          <div class="stat-lbl">Gap &#916;</div>
          <div class="stat-sublbl">(F*&#8722;F') &#8722; (F*&#8722;F)</div>
          <div class="row"><span class="rk">Min</span><span class="rv">{_sign(s['gd_te_min'])}</span></div>
          <div class="row"><span class="rk">Median</span><span class="rv">{_sign(s['gd_te_med'])}</span></div>
          <div class="row"><span class="rk">Max</span><span class="rv">{_sign(s['gd_te_max'])}</span></div>
        </div>
      </div>
    </div>
    <!-- best/worst + generalisation -->
    <div class="sg-col">
      <div class="blk-title">Best &amp; worst &middot; generalisation</div>
      <div class="row"><span class="rk">Best run (min gap tr.)</span><span class="rv g">{s['best_run']} &#8212; {_fmt(s['best_gap'])}</span></div>
      <div class="row"><span class="rk">Worst run (max gap tr.)</span><span class="rv r">{s['worst_run']} &#8212; {_fmt(s['worst_gap'])}</span></div>
      <div style="margin-top:4px;border-top:0.5px solid #eee;padding-top:3px">
        <div class="row"><span class="rk">Median gap ctrl train</span><span class="rv">{_fmt(s['gc_tr_med'])}</span></div>
        <div class="row"><span class="rk">Median gap ctrl test</span><span class="rv">{_fmt(s['gc_te_med'])}</span></div>
        <div class="row"><span class="rk">Degradation (tst&#8722;tr)</span><span class="rv a">{_sign(s['degrad'])}</span></div>
      </div>
    </div>
    <!-- F* -->
    <div class="sg-col" style="width:14%">
      <div class="blk-title">F* (target)</div>
      <div class="stat-sublbl" style="margin-top:3px;margin-bottom:4px">varies per run if seed_target differs</div>
      <div class="row"><span class="rk">Min</span><span class="rv">{_fmt(s['fstar_min'])}</span></div>
      <div class="row"><span class="rk">Median</span><span class="rv">{_fmt(s['fstar_med'])}</span></div>
      <div class="row"><span class="rk">Max</span><span class="rv">{_fmt(s['fstar_max'])}</span></div>
    </div>
  </div>

  <div class="sec-head">02 &#8212; marginal effects &nbsp;&#183;&nbsp; win rate vs each complexity parameter</div>
  <hr class="rule-thin">

  <div class="plot-grid" style="flex: 1;">
    <div class="plot-cell" style="flex: 3.5;">
      <img class="plot-img" src="data:image/png;base64,{b64_marginals}">
      <div class="plot-cap">Win rate vs n, m, &#961; (marginal) &#8212; bars/points coloured green&#8594;red by win rate</div>
    </div>
    <div class="plot-cell" style="flex: 1.5;">
      <img class="plot-img" src="data:image/png;base64,{b64_wr_dist}">
      <div class="plot-cap">Win-rate distribution across configs (left) &amp; sorted waterfall (right)</div>
    </div>
  </div>

  <div class="footer">
    <span>auto-generated &nbsp;&#183;&nbsp; {sweep_dir} &nbsp;&#183;&nbsp; complexity_sweep_report.pdf &nbsp;&#183;&nbsp; heatmaps on next page</span>
    <span>controller_optimization &middot; generate_complexity_sweep_report.py</span>
  </div>
</div>
"""


def build_page2_html(s: dict, now: datetime,
                     b64_hn_rho: str, b64_hn_m: str,
                     b64_hm_rho: str, b64_3d: str,
                     sweep_dir: str) -> str:
    ts = now.strftime('%Y-%m-%d &nbsp;%H:%M:%S')
    n  = s['n_runs']

    return f"""
<div class="page">
  <div class="hdr-row">
    <span class="title">Complexity Sweep Report &#8212; 2D Interactions &amp; 3D Space</span>
    <span class="meta">{ts} &nbsp;·&nbsp; {n} runs &nbsp;·&nbsp; page 2 / 3</span>
  </div>
  <hr class="rule-heavy">

  <div class="sec-head">03 &#8212; 2D parameter interactions &nbsp;&#183;&nbsp; win-rate heatmaps (averaged over the third parameter)</div>
  <hr class="rule-thin">

  <div class="plot-grid" style="flex: 1.4;">
    <div class="plot-cell">
      <img class="plot-img" src="data:image/png;base64,{b64_hn_rho}">
      <div class="plot-cap">n &#215; &#961; heatmap &nbsp;(averaged over m)</div>
    </div>
    <div class="plot-cell">
      <img class="plot-img" src="data:image/png;base64,{b64_hn_m}">
      <div class="plot-cap">n &#215; m heatmap &nbsp;(averaged over &#961;)</div>
    </div>
    <div class="plot-cell">
      <img class="plot-img" src="data:image/png;base64,{b64_hm_rho}">
      <div class="plot-cap">m &#215; &#961; heatmap &nbsp;(averaged over n)</div>
    </div>
  </div>

  <div class="sec-head" style="margin-top:6px;">04 &#8212; 3D complexity space &nbsp;&#183;&nbsp; colour = win rate &nbsp;&#183;&nbsp; size &#8733; n_runs per config</div>
  <hr class="rule-thin">

  <div class="plot-grid" style="flex: 1.4;">
    <div class="plot-cell" style="flex: 1.6;">
      <img class="plot-img" src="data:image/png;base64,{b64_3d}">
      <div class="plot-cap">Win rate in (n, m, &#961;) space</div>
    </div>
    <div class="plot-cell" style="flex: 1.4; padding-top: 6px;">
      <div class="blk-title">Reading guide</div>
      <div class="row"><span class="rk">n</span><span class="rv">input variables &nbsp;&#8212;&nbsp; larger = more inputs</span></div>
      <div class="row"><span class="rk">m</span><span class="rv">cascaded stages &nbsp;&#8212;&nbsp; larger = deeper chain</span></div>
      <div class="row"><span class="rk">&#961;</span><span class="rv">noise intensity &nbsp;&#8212;&nbsp; larger = noisier SCM</span></div>
      <div class="row"><span class="rk">Win rate</span><span class="rv">% runs where F &gt; F&prime; (ctrl beats baseline)</span></div>
      <div class="row"><span class="rk">Color</span><span class="rv"><span class="dg">green</span> &ge; 70 % &nbsp;&#183;&nbsp; <span class="da">amber</span> 40&#8211;70 % &nbsp;&#183;&nbsp; <span class="dr">red</span> &lt; 40 %</span></div>
    </div>
  </div>

  <div class="footer">
    <span>auto-generated &nbsp;&#183;&nbsp; {sweep_dir} &nbsp;&#183;&nbsp; complexity_sweep_report.pdf &nbsp;&#183;&nbsp; configurations table on next page</span>
    <span>controller_optimization &middot; generate_complexity_sweep_report.py</span>
  </div>
</div>
"""


def build_page3_html(win_df: pd.DataFrame, now: datetime, sweep_dir: str,
                     df_raw: pd.DataFrame | None = None) -> str:
    """All-configurations table with per-seed-pair detail rows, sorted by win rate descending."""
    win_df = win_df.sort_values('win_rate_pct', ascending=False).reset_index(drop=True)
    ts     = now.strftime('%Y-%m-%d &nbsp;%H:%M:%S')

    has_nproc = 'n_processes' in win_df.columns and win_df['n_processes'].notna().any()

    nproc_th = '<th>n_proc<span class="def">chain len</span></th>' if has_nproc else ''

    # Build group key columns
    group_cols = ['st_n', 'st_m', 'st_rho']
    if has_nproc:
        group_cols.append('n_processes')

    # Pre-compute per-run gaps and win flag on df_raw
    if df_raw is not None:
        df_detail = df_raw.copy()
        df_detail['gap_baseline'] = (df_detail['F_star_test'] - df_detail['F_baseline_test']).abs()
        df_detail['gap_ctrl_train'] = (df_detail['F_star_train'] - df_detail['F_actual_train']).abs()
        df_detail['gap_ctrl_test'] = (df_detail['F_star_test'] - df_detail['F_actual_test']).abs()
        df_detail['gap_delta_train'] = df_detail['gap_baseline'] - df_detail['gap_ctrl_train']
        df_detail['gap_delta_test'] = df_detail['gap_baseline'] - df_detail['gap_ctrl_test']
        df_detail['win'] = df_detail['gap_ctrl_test'] < df_detail['gap_baseline']
        imp_denom = df_detail['F_baseline_test'].abs().clip(lower=1e-10)
        df_detail['improvement_pct'] = (
            (df_detail['F_actual_test'] - df_detail['F_baseline_test']) / imp_denom * 100
        )
    else:
        df_detail = None

    rows_html = ''
    for _, r in win_df.iterrows():
        wr_cls = _wr_cls(r['win_rate_pct'])
        nproc_td = (f'<td>{int(r["n_processes"]) if pd.notna(r.get("n_processes")) else "N/A"}</td>'
                    if has_nproc else '')

        # ── Compute per-config gap averages from detail rows ──
        gap_bl_mean = gap_ct_tr_mean = gap_ct_te_mean = gap_d_tr_mean = gap_d_te_mean = float('nan')
        if df_detail is not None:
            mask = (df_detail['st_n'] == r['st_n']) & \
                   (df_detail['st_m'] == r['st_m']) & \
                   (df_detail['st_rho'] == r['st_rho'])
            if has_nproc and pd.notna(r.get('n_processes')):
                mask &= (df_detail['n_processes'] == r['n_processes'])
            cfg_runs = df_detail[mask].sort_values('improvement_pct', ascending=False)
            gap_bl_mean = cfg_runs['gap_baseline'].mean()
            gap_ct_tr_mean = cfg_runs['gap_ctrl_train'].mean()
            gap_ct_te_mean = cfg_runs['gap_ctrl_test'].mean()
            gap_d_tr_mean = cfg_runs['gap_delta_train'].mean()
            gap_d_te_mean = cfg_runs['gap_delta_test'].mean()
        else:
            cfg_runs = pd.DataFrame()

        # ── Summary row for this configuration ──
        rows_html += f"""
      <tr style="font-weight:bold; background:#f0f0f0;">
        <td>{int(r['st_n'])}</td>
        <td>{int(r['st_m'])}</td>
        <td>{r['st_rho']:.3f}</td>
        {nproc_td}
        <td>{int(r['n_runs'])}</td>
        <td>{int(r['n_wins'])}</td>
        <td class="{wr_cls}">{r['win_rate_pct']:.1f}%</td>
        <td>{_fmt(gap_bl_mean)}</td>
        <td>{_fmt(gap_ct_tr_mean)}</td>
        <td>{_fmt(gap_ct_te_mean)}</td>
        <td>{_fmt(gap_d_tr_mean)}</td>
        <td>{_fmt(gap_d_te_mean)}</td>
      </tr>"""

        # ── Individual seed-pair rows ──
        if not cfg_runs.empty:
            for _, run in cfg_runs.iterrows():
                win_str = '&#10003;' if run['win'] else '&#10007;'
                win_cls = 'dg' if run['win'] else 'dr'
                nproc_td_run = (f'<td></td>' if has_nproc else '')
                seed_label = f"s<sub>t</sub>={int(run['seed_target'])}&nbsp;s<sub>b</sub>={int(run['seed_baseline'])}"
                rows_html += f"""
      <tr style="color:#666; font-size:6.5px;">
        <td colspan="3" style="text-align:left; padding-left:12px;">{seed_label}</td>
        {nproc_td_run}
        <td></td>
        <td></td>
        <td class="{win_cls}">{win_str}</td>
        <td>{_fmt(run['gap_baseline'])}</td>
        <td>{_fmt(run['gap_ctrl_train'])}</td>
        <td>{_fmt(run['gap_ctrl_test'])}</td>
        <td>{_fmt(run['gap_delta_train'])}</td>
        <td>{_fmt(run['gap_delta_test'])}</td>
      </tr>"""

    n_total = len(df_detail) if df_detail is not None else 0

    return f"""
<div class="page-flow">
  <div class="hdr-row">
    <span class="title">Complexity Sweep Report &#8212; All Configurations (detailed)</span>
    <span class="meta">{ts} &nbsp;&#183;&nbsp; {len(win_df)} configurations &nbsp;&#183;&nbsp; {n_total} runs &nbsp;&#183;&nbsp; page 3+</span>
  </div>
  <hr class="rule-heavy">

  <div class="sec-head">
    05 &#8212; all configurations &amp; seed pairs
    <span style="font-size:7px;font-weight:400;text-transform:none;letter-spacing:0">
      sorted by win rate &middot; descending &nbsp;&#183;&nbsp; bold rows = config summary (mean over seed pairs) &nbsp;&#183;&nbsp; detail rows = individual seed pairs &nbsp;&#183;&nbsp; gap &#916; positive = controller better
    </span>
  </div>
  <hr class="rule-thin">

  <table class="tbl">
    <thead>
      <tr>
        <th>n<span class="def">inputs</span></th>
        <th>m<span class="def">stages</span></th>
        <th>&#961;<span class="def">noise</span></th>
        {nproc_th}
        <th>Runs</th>
        <th>Wins</th>
        <th>Win rate</th>
        <th>Gap baseline<span class="def">|F*&#8722;F&prime;|</span></th>
        <th>Gap ctrl train<span class="def">|F*&#8722;F|</span></th>
        <th>Gap ctrl test<span class="def">|F*&#8722;F|</span></th>
        <th>Gap &#916; train<span class="def">g<sub>bl</sub>&#8722;g<sub>ctrl</sub></span></th>
        <th>Gap &#916; test<span class="def">g<sub>bl</sub>&#8722;g<sub>ctrl</sub></span></th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>

  <div class="legend">
    Win rate coloured:
    <span class="dg">green &ge; 70%</span> &nbsp;&#183;&nbsp;
    <span class="da">amber 40&#8211;70%</span> &nbsp;&#183;&nbsp;
    <span class="dr">red &lt; 40%</span> &nbsp;&#183;&nbsp;
    &#10003; = win &nbsp;&#183;&nbsp; &#10007; = loss &nbsp;&#183;&nbsp;
    gap &#916; positive = controller closer to F* than baseline
  </div>

  <div class="footer">
    <span>auto-generated &nbsp;&#183;&nbsp; {sweep_dir} &nbsp;&#183;&nbsp; complexity_sweep_report.pdf</span>
    <span>controller_optimization &middot; generate_complexity_sweep_report.py</span>
  </div>
</div>
"""


# ════════════════════════════════════════════════════════════════════════════
# 5.  MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def generate_complexity_sweep_report(sweep_dir: Path,
                                     output_path: Path | None = None):
    print(f"Scanning complexity sweep directory: {sweep_dir}")
    df_raw = aggregate_results(sweep_dir)

    if df_raw.empty:
        print("No results found.  Cannot generate report.")
        return None

    core = ['st_n', 'st_m', 'st_rho', 'F_star_test', 'F_baseline_test', 'F_actual_test']
    before = len(df_raw)
    df_raw = df_raw.dropna(subset=core).reset_index(drop=True)
    if len(df_raw) < before:
        print(f"  Dropped {before - len(df_raw)} incomplete runs "
              f"({len(df_raw)} valid remaining)")

    if df_raw.empty:
        print("No valid runs remaining.")
        return None

    print(f"\nValid runs: {len(df_raw)}")

    # ── win rates per config ──────────────────────────────────────────────
    win_df = compute_win_rates(df_raw)
    print(f"Unique configurations: {len(win_df)}")

    # ── aggregate stats ───────────────────────────────────────────────────
    s = compute_stats(df_raw, win_df)

    # ── configuration ─────────────────────────────────────────────────────
    sweep_cfg = load_sweep_config(sweep_dir)
    config_html = build_config_html(sweep_cfg)

    # ── plots ─────────────────────────────────────────────────────────────
    print("\nGenerating plots…")
    b64_marginals = plot_marginals(win_df)
    b64_wr_dist   = plot_winrate_distribution(win_df)
    b64_hn_rho    = plot_heatmap_n_rho(win_df)
    b64_hn_m      = plot_heatmap_n_m(win_df)
    b64_hm_rho    = plot_heatmap_m_rho(win_df)
    b64_3d        = plot_3d_scatter(win_df)
    print("  Done.")

    # ── HTML assembly ──────────────────────────────────────────────────────
    now = datetime.now()
    sd  = str(sweep_dir)

    html_body = (
        build_page1_html(s, now, b64_marginals, b64_wr_dist, sd, config_html)
        + build_page2_html(s, now, b64_hn_rho, b64_hn_m, b64_hm_rho, b64_3d, sd)
        + build_page3_html(win_df, now, sd, df_raw=df_raw)
    )

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>{PAGE_CSS}</style>
</head>
<body>
{html_body}
</body>
</html>"""

    # ── render to PDF ──────────────────────────────────────────────────────
    if output_path is None:
        output_path = sweep_dir / 'complexity_sweep_report.pdf'

    print(f"\nRendering PDF → {output_path}")
    WPHtml(string=full_html).write_pdf(str(output_path))
    print(f"Report saved: {output_path}")

    # ── CSV summaries ──────────────────────────────────────────────────────
    csv_all = sweep_dir / 'complexity_all_runs.csv'
    df_raw.to_csv(csv_all, index=False)
    print(f"CSV all runs : {csv_all}")

    csv_cfg = sweep_dir / 'complexity_configs_winrate.csv'
    win_df.to_csv(csv_cfg, index=False)
    print(f"CSV win rates: {csv_cfg}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate complexity sweep report PDF')
    parser.add_argument(
        '--sweep_dir',
        default='controller_optimization/checkpoints/complexity_sweep',
        help='Directory containing complexity sweep run results')
    parser.add_argument(
        '--output', default=None,
        help='Output PDF path (default: <sweep_dir>/complexity_sweep_report.pdf)')
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        raise SystemExit(f"Error: sweep directory not found: {sweep_dir}")

    generate_complexity_sweep_report(
        sweep_dir, Path(args.output) if args.output else None)


if __name__ == '__main__':
    main()