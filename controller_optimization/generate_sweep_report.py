#!/usr/bin/env python3
"""
Generate aggregated PDF report from parameter sweep results.

Usage:
    python generate_sweep_report.py [--sweep_dir PATH] [--output sweep_report.pdf]

This script:
1. Loads results from all sweep runs
2. Computes aggregate statistics (train + test splits)
3. Generates matplotlib plots embedded as base64
4. Renders a pixel-faithful HTML layout to PDF via WeasyPrint
   (landscape A4, Courier monospace, matches the design spec exactly)
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
import matplotlib.gridspec as gridspec

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
        return {
            'run_name':          run_dir.name,
            'seed_target':       sc.get('seed_target'),
            'seed_baseline':     sc.get('seed_baseline'),
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
    rows = []
    for d in sorted(sweep_dir.iterdir()):
        if not d.is_dir():
            continue
        r = load_run_results(d)
        if r is not None:
            rows.append(r)
            print(f"  Loaded: {d.name}")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# 2.  STATISTICS
# ════════════════════════════════════════════════════════════════════════════

def compute_stats(df: pd.DataFrame) -> dict:
    gb_tr = df['F_star_train']  - df['F_baseline_train']
    gc_tr = df['F_star_train']  - df['F_actual_train']
    gd_tr = gb_tr - gc_tr                              # positive = ctrl better

    gb_te = df['F_star_test']   - df['F_baseline_test']
    gc_te = df['F_star_test']   - df['F_actual_test']
    gd_te = gb_te - gc_te

    wins = (gc_tr < gb_tr).sum()

    # derived columns stored back into df for plotting / table
    df = df.copy()
    df['gap_baseline_train'] = gb_tr
    df['gap_ctrl_train']     = gc_tr
    df['gap_delta_train']    = gd_tr
    df['gap_baseline_test']  = gb_te
    df['gap_ctrl_test']      = gc_te
    df['gap_delta_test']     = gd_te

    best_idx  = gc_tr.idxmin()
    worst_idx = gc_tr.idxmax()

    return {
        'df': df,
        'n_runs':         len(df),
        'wins':           wins,
        'win_rate':       100.0 * wins / len(df),
        # train
        'gb_tr_min':  gb_tr.min(),  'gb_tr_med':  gb_tr.median(),  'gb_tr_max':  gb_tr.max(),
        'gc_tr_min':  gc_tr.min(),  'gc_tr_med':  gc_tr.median(),  'gc_tr_max':  gc_tr.max(),
        'gd_tr_min':  gd_tr.min(),  'gd_tr_med':  gd_tr.median(),  'gd_tr_max':  gd_tr.max(),
        # test
        'gb_te_min':  gb_te.min(),  'gb_te_med':  gb_te.median(),  'gb_te_max':  gb_te.max(),
        'gc_te_min':  gc_te.min(),  'gc_te_med':  gc_te.median(),  'gc_te_max':  gc_te.max(),
        'gd_te_min':  gd_te.min(),  'gd_te_med':  gd_te.median(),  'gd_te_max':  gd_te.max(),
        # best / worst
        'best_run':   df.loc[best_idx,  'run_name'],
        'best_gap':   gc_tr.min(),
        'worst_run':  df.loc[worst_idx, 'run_name'],
        'worst_gap':  gc_tr.max(),
        # generalisation
        'degrad':     gc_te.median() - gc_tr.median(),
        # F*
        'fstar_min':  df['F_star_train'].min(),
        'fstar_med':  df['F_star_train'].median(),
        'fstar_max':  df['F_star_train'].max(),
    }


# ════════════════════════════════════════════════════════════════════════════
# 3.  PLOTS  (returns base64 PNG string)
# ════════════════════════════════════════════════════════════════════════════

_GREEN  = '#1D9E75'
_RED    = '#D85A30'
_AMBER  = '#BA7517'
_MONO   = 'DejaVu Sans Mono'   # closest fallback to Courier on Linux


def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def plot_scatter(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    fstar = df['F_star_train']
    ax.scatter(fstar, df['F_baseline_train'], color=_RED,   s=14, alpha=0.55,
               linewidths=0, label="Baseline F'")
    ax.scatter(fstar, df['F_actual_train'],   color='#2563EB', s=14, alpha=0.55,
               linewidths=0, label='Controller F')
    lo = min(fstar.min(), df['F_baseline_train'].min(), df['F_actual_train'].min()) - 0.02
    hi = fstar.max() + 0.02
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.4)
    ax.set_xlabel('F*  (target)', fontsize=7, fontfamily=_MONO)
    ax.set_ylabel('F  (achieved)', fontsize=7, fontfamily=_MONO)
    ax.set_title('F* vs F\' (red) and F* vs F (blue)', fontsize=7.5,
                 fontfamily=_MONO, fontweight='bold')
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, framealpha=0.6)
    ax.grid(True, alpha=0.18, lw=0.4)
    fig.tight_layout(pad=0.6)
    return _b64(fig)


def plot_boxplot(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    data   = [df['gap_baseline_train'].dropna(), df['gap_ctrl_train'].dropna()]
    labels = ["Gap baseline\n(F* − F')", "Gap ctrl\n(F* − F)"]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.45,
                    medianprops=dict(color='black', lw=1.2))
    bp['boxes'][0].set(facecolor=_RED,   alpha=0.55)
    bp['boxes'][1].set(facecolor=_GREEN, alpha=0.55)
    ax.set_ylabel('Gap', fontsize=7, fontfamily=_MONO)
    ax.set_title('Gap distribution: baseline vs controller', fontsize=7.5,
                 fontfamily=_MONO, fontweight='bold')
    ax.tick_params(labelsize=6)
    ax.grid(True, axis='y', alpha=0.18, lw=0.4)
    fig.tight_layout(pad=0.6)
    return _b64(fig)


def plot_improvement(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 3.4))

    # left: overlapping distributions
    ax = axes[0]
    bins = np.linspace(
        min(df['gap_baseline_train'].min(), df['gap_ctrl_train'].min()),
        max(df['gap_baseline_train'].max(), df['gap_ctrl_train'].max()),
        30
    )
    ax.hist(df['gap_baseline_train'], bins=bins, color=_RED,   alpha=0.55,
            label="Baseline", density=True)
    ax.hist(df['gap_ctrl_train'],     bins=bins, color=_GREEN, alpha=0.55,
            label="Ctrl",     density=True)
    ax.set_xlabel('Gap', fontsize=7, fontfamily=_MONO)
    ax.set_ylabel('Density', fontsize=7, fontfamily=_MONO)
    ax.set_title('Gap overlap', fontsize=7.5, fontfamily=_MONO, fontweight='bold')
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.18, lw=0.4)

    # right: delta distribution
    ax = axes[1]
    deltas = df['gap_delta_train'].sort_values()
    colors_bar = [_GREEN if v >= 0 else _RED for v in deltas]
    ax.bar(range(len(deltas)), deltas, color=colors_bar, width=1.0, linewidth=0)
    ax.axhline(0, color='black', lw=0.6)
    ax.set_xlabel('Run (sorted)', fontsize=7, fontfamily=_MONO)
    ax.set_ylabel('Gap Δ  (F − F\')', fontsize=7, fontfamily=_MONO)
    ax.set_title('Gap reduction per run', fontsize=7.5,
                 fontfamily=_MONO, fontweight='bold')
    ax.tick_params(labelsize=6)
    ax.grid(True, axis='y', alpha=0.18, lw=0.4)

    fig.tight_layout(pad=0.6)
    return _b64(fig)


def plot_heatmap(df: pd.DataFrame) -> str:
    df = df.copy()
    df['seed_t'] = pd.to_numeric(df['seed_target'],  errors='coerce')
    df['seed_b'] = pd.to_numeric(df['seed_baseline'], errors='coerce')
    df = df.dropna(subset=['seed_t', 'seed_b', 'gap_delta_train'])

    pivot = df.pivot_table(index='seed_t', columns='seed_b',
                           values='gap_delta_train', aggfunc='mean')

    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 0.01)
    im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn',
                   vmin=-vmax, vmax=vmax, origin='upper')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(c)) for c in pivot.columns],
                       fontsize=5, fontfamily=_MONO)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(int(i)) for i in pivot.index],
                       fontsize=5, fontfamily=_MONO)
    ax.set_xlabel('seed_b', fontsize=7, fontfamily=_MONO)
    ax.set_ylabel('seed_t', fontsize=7, fontfamily=_MONO)
    ax.set_title('Gap reduction by seed combination', fontsize=7.5,
                 fontfamily=_MONO, fontweight='bold')
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cb.ax.tick_params(labelsize=5)
    fig.tight_layout(pad=0.6)
    return _b64(fig)


# ════════════════════════════════════════════════════════════════════════════
# 4.  HTML TEMPLATE
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

/* ── four‑column stats grid ── */
.stats-grid  { display: table; width: 100%; margin-bottom: 7px;
               table-layout: fixed; }
.sg-col      { display: table-cell; padding-right: 12px;
               vertical-align: top; }
.sg-col:last-child { padding-right: 0; }
.sg-inner    { display: table; width: 100%; table-layout: fixed; }
.sg-sub      { display: table-cell; padding-right: 5px; vertical-align: top; }
.sg-sub:last-child { padding-right: 0; }

/* ── plot grid ── */
.plot-grid { display: table; width: 100%; flex: 1; table-layout: fixed; }
.plot-cell { display: table-cell; padding: 0 3px; vertical-align: top; }
.plot-cell:first-child { padding-left: 0; }
.plot-cell:last-child  { padding-right: 0; }
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
.legend { font-size: 7px; color: #888; margin-top: 4px; }
"""


def _fmt(v, digits=4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return f"{v:.{digits}f}"


def _sign(v) -> str:
    return f"+{v:.3f}" if v >= 0 else f"{v:.3f}"


def _delta_cls(v) -> str:
    if v >= 0.05:  return 'dg'
    if v >= 0:     return 'da'
    return 'dr'


def _gap_ctrl_cls(v, q25, q75) -> str:
    if v <= q25: return 'g'
    if v >= q75: return 'r'
    return ''


def build_page1_html(s: dict, now: datetime,
                     b64_scatter, b64_box, b64_imp, b64_heat,
                     sweep_dir: str) -> str:
    n    = s['n_runs']
    ts   = now.strftime('%Y-%m-%d &nbsp;%H:%M:%S')
    return f"""
<div class="page">
  <div class="hdr-row">
    <span class="title">Controller Sweep Report</span>
    <span class="meta">{ts} &nbsp;·&nbsp; {sweep_dir} &nbsp;·&nbsp; {n} runs &nbsp;·&nbsp; page 1 / 2</span>
  </div>
  <hr class="rule-heavy">

  <div class="kpi-row">
    <div class="kpi">
      <div class="kpi-l">Total runs</div>
      <div class="kpi-v">{n}</div>
      <div class="kpi-s">valid runs</div>
    </div>
    <div class="kpi">
      <div class="kpi-l">Controller wins</div>
      <div class="kpi-v g">{s['wins']}/{n}</div>
      <div class="kpi-s">win rate {s['win_rate']:.1f}%</div>
    </div>
    <div class="kpi">
      <div class="kpi-l">Median gap &#916; (train)</div>
      <div class="kpi-v g">{_sign(s['gd_tr_med'])}</div>
      <div class="kpi-s">F&#8722;F' improvement</div>
    </div>
    <div class="kpi">
      <div class="kpi-l">Best ctrl gap (train)</div>
      <div class="kpi-v">{_fmt(s['gc_tr_min'])}</div>
      <div class="kpi-s">{s['best_run']}</div>
    </div>
  </div>

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
          <div class="stat-sublbl">F &#8722; F'</div>
          <div class="row"><span class="rk">Min</span><span class="rv a">{_sign(s['gd_tr_min'])}</span></div>
          <div class="row"><span class="rk">Median</span><span class="rv g">{_sign(s['gd_tr_med'])}</span></div>
          <div class="row"><span class="rk">Max</span><span class="rv g">{_sign(s['gd_tr_max'])}</span></div>
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
          <div class="stat-sublbl">F &#8722; F'</div>
          <div class="row"><span class="rk">Min</span><span class="rv a">{_sign(s['gd_te_min'])}</span></div>
          <div class="row"><span class="rk">Median</span><span class="rv g">{_sign(s['gd_te_med'])}</span></div>
          <div class="row"><span class="rk">Max</span><span class="rv g">{_sign(s['gd_te_max'])}</span></div>
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

  <div class="sec-head">02 &#8212; visualizations</div>
  <hr class="rule-thin">
  <div class="plot-grid">
    <div class="plot-cell">
      <img class="plot-img" src="data:image/png;base64,{b64_scatter}">
      <div class="plot-cap">F* vs F' (red) and F* vs F (blue) &#8212; diagonal = perfect</div>
    </div>
    <div class="plot-cell">
      <img class="plot-img" src="data:image/png;base64,{b64_box}">
      <div class="plot-cap">Gap distribution: baseline vs controller</div>
    </div>
    <div class="plot-cell">
      <img class="plot-img" src="data:image/png;base64,{b64_imp}">
      <div class="plot-cap">Gap overlap + gap reduction per run</div>
    </div>
    <div class="plot-cell">
      <img class="plot-img" src="data:image/png;base64,{b64_heat}">
      <div class="plot-cap">Gap reduction by seed combination &#8212; green = ctrl better</div>
    </div>
  </div>

  <div class="footer">
    <span>auto-generated &nbsp;&#183;&nbsp; {sweep_dir} &nbsp;&#183;&nbsp; sweep_report.pdf &nbsp;&#183;&nbsp; all runs table on next page</span>
    <span>controller_optimization &middot; generate_sweep_report.py</span>
  </div>
</div>
"""


def build_page2_html(df: pd.DataFrame, now: datetime, sweep_dir: str) -> str:
    df = df.sort_values('gap_ctrl_train', ascending=True).reset_index(drop=True)

    q25 = df['gap_ctrl_train'].quantile(0.25)
    q75 = df['gap_ctrl_train'].quantile(0.75)
    ts  = now.strftime('%Y-%m-%d &nbsp;%H:%M:%S')

    rows_html = ''
    for _, r in df.iterrows():
        gc_cls = _gap_ctrl_cls(r['gap_ctrl_train'], q25, q75)
        gct_cls = _gap_ctrl_cls(r['gap_ctrl_test'],  q25, q75)
        dtr_cls = _delta_cls(r['gap_delta_train'])
        dte_cls = _delta_cls(r['gap_delta_test'])

        rows_html += f"""
      <tr>
        <td>{r['run_name']}</td>
        <td>{int(r['seed_target']) if pd.notna(r['seed_target']) else 'N/A'}</td>
        <td>{int(r['seed_baseline']) if pd.notna(r['seed_baseline']) else 'N/A'}</td>
        <td>{_fmt(r['F_star_train'])}</td>
        <td>{_fmt(r['F_baseline_train'])}</td>
        <td>{_fmt(r['F_actual_train'])}</td>
        <td>{_fmt(r['gap_baseline_train'])}</td>
        <td class="{gc_cls}">{_fmt(r['gap_ctrl_train'])}</td>
        <td class="{gct_cls}">{_fmt(r['gap_ctrl_test'])}</td>
        <td class="{dtr_cls}">{_sign(r['gap_delta_train'])}</td>
        <td class="{dte_cls}">{_sign(r['gap_delta_test'])}</td>
      </tr>"""

    return f"""
<div class="page">
  <div class="hdr-row">
    <span class="title">Controller Sweep Report &#8212; All Runs</span>
    <span class="meta">{ts} &nbsp;&#183;&nbsp; {len(df)} runs &nbsp;&#183;&nbsp; page 2 / 2</span>
  </div>
  <hr class="rule-heavy">

  <div class="sec-head">
    03 &#8212; all runs
    <span style="font-size:7px;font-weight:400;text-transform:none;letter-spacing:0">
      sorted by gap ctrl train &middot; ascending &nbsp;&#183;&nbsp;
      &#916; = F&#8722;F' &nbsp;&#183;&nbsp; positive = ctrl better
    </span>
  </div>
  <hr class="rule-thin">

  <table class="tbl">
    <thead>
      <tr>
        <th>Run</th>
        <th>Seed T</th>
        <th>Seed B</th>
        <th>F*</th>
        <th>F' baseline</th>
        <th>F controller</th>
        <th>Gap baseline<span class="def">F* &#8722; F'</span></th>
        <th>Gap ctrl &#8212; train<span class="def">F* &#8722; F</span></th>
        <th>Gap ctrl &#8212; test<span class="def">F* &#8722; F</span></th>
        <th>Gap &#916; &#8212; train<span class="def">(F*&#8722;F') &#8722; (F*&#8722;F)</span></th>
        <th>Gap &#916; &#8212; test<span class="def">(F*&#8722;F') &#8722; (F*&#8722;F)</span></th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>

  <div class="legend">
    Gap ctrl colored:
    <span class="g">green = best quartile</span> &nbsp;&#183;&nbsp;
    <span class="r">red = worst quartile</span> &nbsp;&#183;&nbsp;
    Gap &#916; colored:
    <span class="dg">green = ctrl better</span> &nbsp;&#183;&nbsp;
    <span class="da">amber = marginal (&lt;0.05)</span> &nbsp;&#183;&nbsp;
    <span class="dr">red = baseline better</span> &nbsp;&#183;&nbsp;
    F* shown per-run as it may vary across seed combinations
  </div>

  <div class="footer">
    <span>auto-generated &nbsp;&#183;&nbsp; {sweep_dir} &nbsp;&#183;&nbsp; sweep_report.pdf</span>
    <span>controller_optimization &middot; generate_sweep_report.py</span>
  </div>
</div>
"""


# ════════════════════════════════════════════════════════════════════════════
# 5.  MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def generate_sweep_report(sweep_dir: Path, output_path: Path | None = None):
    print(f"Scanning sweep directory: {sweep_dir}")
    df_raw = aggregate_results(sweep_dir)

    if df_raw.empty:
        print("No results found.  Cannot generate report.")
        return None

    print(f"\nLoaded {len(df_raw)} runs")

    core = ['F_star_train', 'F_baseline_train', 'F_actual_train']
    before = len(df_raw)
    df_raw = df_raw.dropna(subset=core).reset_index(drop=True)
    if len(df_raw) < before:
        print(f"  Dropped {before - len(df_raw)} runs with missing F values "
              f"({len(df_raw)} valid remaining)")

    if df_raw.empty:
        print("No valid runs remaining.")
        return None

    # ── stats ────────────────────────────────────────────────────────────────
    s = compute_stats(df_raw)
    df = s['df']

    # ── plots ────────────────────────────────────────────────────────────────
    print("\nGenerating plots…")
    b64_scatter = plot_scatter(df)
    b64_box     = plot_boxplot(df)
    b64_imp     = plot_improvement(df)
    b64_heat    = plot_heatmap(df)
    print("  Done.")

    # ── HTML assembly ─────────────────────────────────────────────────────────
    now = datetime.now()
    sd  = str(sweep_dir)

    html_body = (
        build_page1_html(s, now, b64_scatter, b64_box, b64_imp, b64_heat, sd)
        + build_page2_html(df, now, sd)
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

    # ── render to PDF ─────────────────────────────────────────────────────────
    if output_path is None:
        output_path = sweep_dir / 'sweep_report.pdf'

    print(f"\nRendering PDF → {output_path}")
    WPHtml(string=full_html).write_pdf(str(output_path))
    print(f"Report saved: {output_path}")

    # ── CSV summary ───────────────────────────────────────────────────────────
    csv_path = sweep_dir / 'sweep_results_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"CSV summary : {csv_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate sweep report PDF')
    parser.add_argument('--sweep_dir', default='controller_optimization/checkpoints/sweep',
                        help='Directory containing sweep run results')
    parser.add_argument('--output', default=None,
                        help='Output PDF path (default: <sweep_dir>/sweep_report.pdf)')
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        raise SystemExit(f"Error: sweep directory not found: {sweep_dir}")

    generate_sweep_report(sweep_dir, Path(args.output) if args.output else None)


if __name__ == '__main__':
    main()