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
    df['controller_wins'] = df['F_actual_train'] > df['F_baseline_train']

    group_cols = ['st_n', 'st_m', 'st_rho']
    if 'n_processes' in df.columns and df['n_processes'].notna().any():
        group_cols.append('n_processes')

    grouped = df.groupby(group_cols).agg(
        n_runs=('controller_wins', 'count'),
        n_wins=('controller_wins', 'sum'),
        F_star_mean=('F_star_train', 'mean'),
        F_baseline_mean=('F_baseline_train', 'mean'),
        F_actual_mean=('F_actual_train', 'mean'),
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
    overall_win_rate = float(100.0 * (df['F_actual_train'] > df['F_baseline_train']).mean())
    best_row = win_df.loc[win_df['win_rate_pct'].idxmax()]
    worst_row = win_df.loc[win_df['win_rate_pct'].idxmin()]

    has_nproc = 'n_processes' in win_df.columns and win_df['n_processes'].notna().any()

    def _cfg_label(row):
        s = f"n={int(row['st_n'])} m={int(row['st_m'])} ρ={row['st_rho']:.2f}"
        if has_nproc and pd.notna(row.get('n_processes')):
            s += f" p={int(row['n_processes'])}"
        return s

    return {
        'n_runs':           n_runs,
        'n_cfgs':           n_cfgs,
        'overall_win_rate': overall_win_rate,
        'best_cfg':         _cfg_label(best_row),
        'best_wr':          float(best_row['win_rate_pct']),
        'worst_cfg':        _cfg_label(worst_row),
        'worst_wr':         float(worst_row['win_rate_pct']),
        'median_wr':        float(win_df['win_rate_pct'].median()),
        'has_nproc':        has_nproc,
        'df':               df,
        'win_df':           win_df,
    }


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
.row { display: flex; justify-content: space-between; padding: 1.5px 0;
       border-bottom: 0.5px solid #eee; font-size: 8px; gap: 6px; }
.row:last-child { border-bottom: none; }
.rk  { color: #888; }
.rv  { font-weight: 500; }

/* ── plot grid ── */
.plot-grid { display: flex; width: 100%; flex: 1; gap: 6px; }
.plot-cell { flex: 1 1 0; min-width: 0; }
.plot-img  { width: 100%; height: auto; display: block; }
.plot-cap  { font-size: 6.5px; color: #888; margin-top: 2px;
             text-align: center; font-style: italic; }

/* ── table ── */
.tbl { width: 100%; border-collapse: collapse; font-size: 7.5px; }
.tbl th { background: #f4f4f4; font-weight: 500; text-align: left;
          padding: 2.5px 5px; border: 0.5px solid #ccc;
          text-transform: uppercase; font-size: 7px; letter-spacing: 0.3px; }
.tbl td { padding: 2px 5px; border: 0.5px solid #e4e4e4; }
.tbl tr:nth-child(even) td { background: #fafafa; }
.tbl .def { font-size: 6px; color: #aaa; display: block; }

/* ── footer ── */
.footer { display: flex; justify-content: space-between; font-size: 6.5px;
          color: #aaa; margin-top: auto; padding-top: 4px; }

/* ── legend ── */
.legend { font-size: 6.5px; color: #888; margin-top: 4px; }
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

def _wr_cls(v: float) -> str:
    if v >= 70:   return 'dg'
    if v >= 40:   return 'da'
    return 'dr'


def build_page1_html(s: dict, now: datetime,
                     b64_marginals: str, b64_wr_dist: str,
                     sweep_dir: str) -> str:
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
      <div class="kpi-l">Overall win rate</div>
      <div class="kpi-v {wr_cls}">{wr:.1f}%</div>
      <div class="kpi-s">F&nbsp;&gt;&nbsp;F&prime; across all runs</div>
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

  <div class="sec-head">01 &#8212; marginal effects &nbsp;&#183;&nbsp; win rate vs each complexity parameter</div>
  <hr class="rule-thin">

  <div class="plot-grid" style="flex: 2.2;">
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

  <div class="sec-head">02 &#8212; 2D parameter interactions &nbsp;&#183;&nbsp; win-rate heatmaps (averaged over the third parameter)</div>
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

  <div class="sec-head" style="margin-top:6px;">03 &#8212; 3D complexity space &nbsp;&#183;&nbsp; colour = win rate &nbsp;&#183;&nbsp; size &#8733; n_runs per config</div>
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


def build_page3_html(win_df: pd.DataFrame, now: datetime, sweep_dir: str) -> str:
    """All-configurations table, sorted by win rate descending."""
    win_df = win_df.sort_values('win_rate_pct', ascending=False).reset_index(drop=True)
    ts     = now.strftime('%Y-%m-%d &nbsp;%H:%M:%S')

    has_nproc = 'n_processes' in win_df.columns and win_df['n_processes'].notna().any()

    nproc_th = '<th>n_proc<span class="def">chain len</span></th>' if has_nproc else ''

    rows_html = ''
    for _, r in win_df.iterrows():
        wr_cls = _wr_cls(r['win_rate_pct'])
        nproc_td = (f'<td>{int(r["n_processes"]) if pd.notna(r.get("n_processes")) else "N/A"}</td>'
                    if has_nproc else '')
        rows_html += f"""
      <tr>
        <td>{int(r['st_n'])}</td>
        <td>{int(r['st_m'])}</td>
        <td>{r['st_rho']:.3f}</td>
        {nproc_td}
        <td>{int(r['n_runs'])}</td>
        <td>{int(r['n_wins'])}</td>
        <td class="{wr_cls}">{r['win_rate_pct']:.1f}%</td>
        <td>{_fmt(r['F_star_mean'])}</td>
        <td>{_fmt(r['F_baseline_mean'])}</td>
        <td>{_fmt(r['F_actual_mean'])}</td>
        <td>{_pct(r['mean_improvement_pct'])}</td>
      </tr>"""

    return f"""
<div class="page-flow">
  <div class="hdr-row">
    <span class="title">Complexity Sweep Report &#8212; All Configurations</span>
    <span class="meta">{ts} &nbsp;&#183;&nbsp; {len(win_df)} configurations &nbsp;&#183;&nbsp; page 3 / 3</span>
  </div>
  <hr class="rule-heavy">

  <div class="sec-head">
    04 &#8212; all configurations
    <span style="font-size:7px;font-weight:400;text-transform:none;letter-spacing:0">
      sorted by win rate &middot; descending &nbsp;&#183;&nbsp; win = F &gt; F&prime; &nbsp;&#183;&nbsp; improvement = (F&#8722;F&prime;)/|F&prime;|
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
        <th>F* mean</th>
        <th>F&prime; baseline</th>
        <th>F controller</th>
        <th>Improvement<span class="def">(F&#8722;F&prime;)/|F&prime;|</span></th>
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
    improvement positive = controller beats baseline
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

    core = ['st_n', 'st_m', 'st_rho', 'F_star_train', 'F_baseline_train', 'F_actual_train']
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
        build_page1_html(s, now, b64_marginals, b64_wr_dist, sd)
        + build_page2_html(s, now, b64_hn_rho, b64_hn_m, b64_hm_rho, b64_3d, sd)
        + build_page3_html(win_df, now, sd)
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