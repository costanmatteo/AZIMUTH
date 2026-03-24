"""
PDF Report Generator — CasualiT Surrogate Training
====================================================
Generates a single A4-landscape page via WeasyPrint (HTML → PDF).

Visual style is identical to the Uncertainty Predictor report:
  - Courier New monospace throughout
  - Same colour tokens: #1D9E75 green / #D85A30 red / #BA7517 amber
  - Same KPI row (4 tiles, top of left column)
  - Same section-header style (uppercase, letter-spaced, grey)
  - Same row layout (key right-aligned grey / value bold)
  - Same table style (thin rules, uppercase headers)
  - Same footer bar
  - Right column: 4 plots (A loss · B scatter · C residuals · D F-distribution)
    + conditional DAG section at bottom (only with LieAttention phi tensors)

Drop-in replacement for generate_pdf_report() in train_surrogate.py.
Signature unchanged — main() needs no edits.

Requirements: weasyprint, matplotlib, numpy
"""

from __future__ import annotations

import base64
import math
import tempfile
import traceback
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np

# matplotlib — imported lazily inside plot helpers to avoid backend issues

# ─────────────────────────────────────────────────────────────────────────────
#  Colour helpers
# ─────────────────────────────────────────────────────────────────────────────
_GREEN = '#1D9E75'
_RED   = '#D85A30'
_AMBER = '#BA7517'
_GRAY  = '#888888'
_MGRAY = '#cccccc'
_BGRAY = '#f7f7f7'


def _r2_col(v: float) -> str:
    return _GREEN if v >= 0.90 else (_AMBER if v >= 0.70 else _RED)


def _loss_col(v: float) -> str:
    return _GREEN if v < 0.005 else (_AMBER if v < 0.02 else _RED)


def _fmt_lr(lr: float) -> str:
    if lr == 0:
        return '0'
    exp = int(math.floor(math.log10(abs(lr))))
    m   = lr / (10 ** exp)
    return f'{m:.3f} x 10<sup>{exp}</sup>'


def _pct(v: float) -> str:
    return f'{v * 100:.1f}%'


# ─────────────────────────────────────────────────────────────────────────────
#  matplotlib plot helpers — return base-64 PNG data-URIs
# ─────────────────────────────────────────────────────────────────────────────

def _fig_to_b64(fig) -> str:
    import matplotlib.pyplot as plt
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def _mpl_ax_style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, fontsize=8, fontweight='bold', pad=4)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.grid(True, alpha=0.25, linestyle='--', color=_MGRAY)
    ax.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_edgecolor(_MGRAY)
        sp.set_linewidth(0.6)


def _plot_loss(history: dict, best_epoch: int) -> str:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 2.1))
    fig.patch.set_facecolor('white')
    ep = range(len(history['train_loss']))
    ax.plot(ep, history['train_loss'], color=_GREEN, lw=1.5, label='Train MSE')
    ax.plot(ep, history['val_loss'],   color=_RED,   lw=1.5, ls='--', label='Val MSE')
    if best_epoch is not None and best_epoch < len(history['train_loss']):
        ax.axvline(best_epoch, color=_AMBER, lw=1.0, ls=':',
                   label=f'best (ep. {best_epoch})')
    ax.legend(fontsize=6.5, framealpha=0.8, edgecolor=_MGRAY, loc='upper right')
    _mpl_ax_style(ax, 'Training History — MSE Loss', 'Epoch', 'MSE Loss')
    fig.tight_layout(pad=0.4)
    return _fig_to_b64(fig)


def _plot_scatter(preds: np.ndarray, targets: np.ndarray, r2: float) -> str:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(2.5, 2.1))
    fig.patch.set_facecolor('white')
    ax.scatter(targets, preds, c=preds, cmap='viridis', alpha=0.5, s=8, lw=0)
    lo = min(float(targets.min()), float(preds.min())) - 0.02
    hi = max(float(targets.max()), float(preds.max())) + 0.02
    ax.plot([lo, hi], [lo, hi], color=_RED, lw=1.4, ls='--', label='Perfect')
    props = dict(boxstyle='round,pad=0.25', facecolor=_BGRAY, edgecolor=_MGRAY, alpha=0.9)
    ax.text(0.05, 0.94, f'R² = {r2:.4f}',
            transform=ax.transAxes, fontsize=7, va='top', bbox=props)
    ax.legend(fontsize=6, framealpha=0.8, edgecolor=_MGRAY)
    _mpl_ax_style(ax, 'Predicted vs True F', 'True F', 'Predicted F')
    fig.tight_layout(pad=0.4)
    return _fig_to_b64(fig)


def _plot_residuals(preds: np.ndarray, targets: np.ndarray) -> str:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    res = preds - targets
    fig, ax = plt.subplots(figsize=(2.5, 2.1))
    fig.patch.set_facecolor('white')
    ax.hist(res, bins=28, color=_GREEN, edgecolor='white', alpha=0.85, lw=0.3)
    ax.axvline(0, color=_RED, lw=1.4, ls='--')
    props = dict(boxstyle='round,pad=0.25', facecolor=_BGRAY, edgecolor=_MGRAY, alpha=0.9)
    ax.text(0.97, 0.95, f'μ={res.mean():.4f}\nσ={res.std():.4f}',
            transform=ax.transAxes, fontsize=6, va='top', ha='right', bbox=props)
    _mpl_ax_style(ax, 'Residual Distribution', 'F_pred − F_true', 'Count')
    fig.tight_layout(pad=0.4)
    return _fig_to_b64(fig)


def _plot_fdist(preds: np.ndarray, targets: np.ndarray) -> str:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(2.5, 2.1))
    fig.patch.set_facecolor('white')
    ax.hist(targets, bins=28, color=_GREEN, alpha=0.55, density=True,
            label='True F',      edgecolor='white', lw=0.3)
    ax.hist(preds,   bins=28, color=_AMBER, alpha=0.55, density=True,
            label='Predicted F', edgecolor='white', lw=0.3)
    ax.legend(fontsize=6, framealpha=0.8, edgecolor=_MGRAY)
    _mpl_ax_style(ax, 'F Distribution', 'F value', 'Density')
    fig.tight_layout(pad=0.4)
    return _fig_to_b64(fig)


def _plot_dag(phi: np.ndarray, proc_names: list) -> str:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n      = len(proc_names)
    binary = (phi >= 0.5).astype(float)
    fig, ax = plt.subplots(figsize=(3.8, 1.9))
    fig.patch.set_facecolor('white')
    im = ax.imshow(phi, cmap='Greens', vmin=0, vmax=1, aspect='auto')
    cb = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
    cb.set_label('Edge prob.', fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(proc_names, fontsize=6.5, rotation=45, ha='right')
    ax.set_yticklabels(proc_names, fontsize=6.5)
    for i in range(n):
        for j in range(n):
            if binary[i, j] == 1:
                ax.plot(j, i, 'o', color=_GREEN, ms=4.5, mew=0.5, mec='white')
            else:
                ax.plot(j, i, 'x', color='white', ms=3.5, mew=0.7)
    ax.set_xlabel('Source', fontsize=7)
    ax.set_ylabel('Target', fontsize=7)
    ax.set_title('Estimated Causal DAG (LieAttention φ)', fontsize=8,
                 fontweight='bold', pad=3)
    ax.tick_params(labelsize=6.5)
    fig.tight_layout(pad=0.35)
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Extract phi tensor from trainer/model if available
# ─────────────────────────────────────────────────────────────────────────────

def _extract_phi(trainer) -> Optional[np.ndarray]:
    model = trainer.model
    if hasattr(model, 'get_dag_adjacency'):
        try:
            return model.get_dag_adjacency().detach().cpu().numpy()
        except Exception:
            pass
    for attr in ('phi', 'phi_tensors'):
        val = getattr(trainer, attr, None)
        if val is None:
            continue
        try:
            if isinstance(val, dict):
                val = list(val.values())[0]
            if hasattr(val, 'detach'):
                val = val.detach().cpu().numpy()
            val = np.array(val)
            if val.ndim == 3:
                val = val.mean(axis=0)
            return val
        except Exception:
            pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  CSS (identical tokens to uncertainty predictor HTML)
# ─────────────────────────────────────────────────────────────────────────────

PAGE_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #f0f0f0;
  font-family: 'Courier New', Courier, monospace;
  padding: 0;
}
.page {
  background: white;
  width: 297mm;
  height: 210mm;
  display: flex;
  flex-direction: column;
  padding: 18px 20px 10px;
  color: #1a1a1a;
  font-size: 9px;
  line-height: 1.45;
  overflow: hidden;
}
.hdr-row {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  margin-bottom: 2px;
}
.hdr-title  { font-size: 11px; font-weight: 500; }
.hdr-meta   { font-size: 8px; color: #666; margin-top: 2px; }
.badge      { font-size: 8px; font-weight: 500; padding: 1px 5px; border: 0.5px solid; }
.badge-g    { border-color: #1D9E75; color: #1D9E75; }
.badge-r    { border-color: #D85A30; color: #D85A30; }
.badge-a    { border-color: #BA7517; color: #BA7517; }
.rule-heavy { border: none; border-top: 1px solid #1a1a1a; margin: 3px 0 4px; }
.rule-thin  { border: none; border-top: 0.5px solid #aaa; margin: 2px 0 3px; }
.g { color: #1D9E75; }
.r { color: #D85A30; }
.a { color: #BA7517; }
.body {
  display: flex;
  gap: 10px;
  flex: 1;
  min-height: 0;
}
.left  {
  width: 290px;
  flex-shrink: 0;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.divider { width: 0.5px; background: #ddd; flex-shrink: 0; }
.right {
  flex: 1;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  gap: 0;
}

/* ── KPI row ── */
.kpi-row {
  display: grid;
  grid-template-columns: repeat(4, minmax(0,1fr));
  border: 0.5px solid #ccc;
  margin-bottom: 5px;
}
.kpi            { padding: 4px 5px; border-right: 0.5px solid #ccc; }
.kpi:last-child { border-right: none; }
.kpi-l { font-size: 7px; color: #888; text-transform: uppercase; letter-spacing: 0.4px; }
.kpi-v { font-size: 11px; font-weight: 500; margin: 1px 0; }
.kpi-s { font-size: 7px; color: #888; }

/* ── Section headers ── */
.sec-head {
  font-size: 7px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.9px;
  color: #888;
  margin-top: 4px;
  margin-bottom: 2px;
}

/* ── Two-column key/value grid ── */
.two { display: grid; grid-template-columns: minmax(0,1fr) minmax(0,1fr); gap: 6px; }
.row {
  display: flex;
  justify-content: space-between;
  padding: 1.5px 0;
  border-bottom: 0.5px solid #eee;
  font-size: 8px;
  gap: 4px;
}
.row:last-child { border-bottom: none; }
.rk { color: #888; flex-shrink: 0; }
.rv { font-weight: 500; text-align: right; }

/* ── Metrics table ── */
.tbl { width: 100%; border-collapse: collapse; font-size: 7.5px; }
.tbl th {
  text-align: left;
  padding: 2px 3px;
  border-bottom: 0.5px solid #1a1a1a;
  font-weight: 500;
  font-size: 7px;
  color: #888;
  text-transform: uppercase;
  letter-spacing: 0.3px;
}
.tbl td { padding: 2px 3px; border-bottom: 0.5px solid #eee; }
.tbl tr:last-child td { border-bottom: none; }
.note { font-size: 7px; color: #888; margin-top: 2px; }

/* ── Convergence badge row ── */
.conv-row { margin-top: 3px; }
.conv-badge {
  display: inline-block;
  font-size: 7px;
  font-weight: 500;
  padding: 1px 6px;
  border: 0.5px solid;
}

/* ── Plot area ── */
.plot-label {
  font-size: 7px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.7px;
  color: #888;
  margin-top: 4px;
  margin-bottom: 1px;
}
.plot-box {
  background: #f7f7f7;
  border: 0.5px solid #ccc;
  flex: 1;
  min-height: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}
.plot-box img { width: 100%; height: 100%; object-fit: contain; }
.plots-row { display: flex; gap: 4px; flex: 1; min-height: 0; }
.plot-cap  { font-size: 7px; color: #999; margin-top: 1px; font-style: italic; }

/* ── DAG section ── */
.dag-placeholder {
  background: #f7f7f7;
  border: 0.5px solid #ccc;
  padding: 8px;
  font-size: 8px;
  color: #888;
  font-style: italic;
  text-align: center;
  margin-top: 2px;
}

/* ── Footer ── */
.footer {
  margin-top: 4px;
  border-top: 1px solid #1a1a1a;
  padding-top: 3px;
  display: flex;
  justify-content: space-between;
  font-size: 7px;
  color: #888;
}
"""


# ─────────────────────────────────────────────────────────────────────────────
#  HTML builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_html(trainer, eval_results: dict, config: dict,
                b64_loss: str, b64_scatter: str,
                b64_residuals: str, b64_fdist: str,
                b64_dag: Optional[str]) -> str:

    now       = datetime.now()
    date_str  = now.strftime('%Y-%m-%d')
    time_str  = now.strftime('%H:%M:%S')

    model   = trainer.model
    history = trainer.history
    mc      = config['model']
    tc      = config['training']
    dc      = config['data']

    # counts
    n_train = len(trainer.train_loader.dataset)  if trainer.train_loader  else 0
    n_val   = len(trainer.val_loader.dataset)    if trainer.val_loader    else 0
    n_test  = len(trainer.test_loader.dataset)   if trainer.test_loader   else 0
    total_p = sum(p.numel() for p in model.parameters())
    n_proc  = getattr(model, 'n_processes', '?') or '?'
    n_feat  = getattr(model, 'n_features',  '?') or '?'

    # metrics
    bvl   = trainer.best_val_loss
    r2    = eval_results['test_r2']
    rmse  = eval_results['test_rmse']
    mae   = eval_results['test_mae']
    mse   = eval_results['test_mse']

    final_train = history['train_loss'][-1] if history.get('train_loss') else float('nan')
    final_val   = history['val_loss'][-1]   if history.get('val_loss')   else float('nan')

    # colour tags
    r2_col   = _r2_col(r2)
    bvl_col  = _loss_col(bvl)
    converged = bvl < 0.01

    def _col_class(col):
        return {_GREEN: 'g', _RED: 'r', _AMBER: 'a'}.get(col, '')

    # header badge
    if converged:
        badge_html = '<span class="badge badge-g">converged</span>'
    else:
        badge_html = '<span class="badge badge-r">not converged</span>'

    # meta line
    sched_str = 'scheduler on' if tc.get('use_scheduler') else 'no scheduler'
    meta_str  = (f'{date_str}&nbsp;&nbsp;{time_str}'
                 f'&nbsp;·&nbsp;seed {dc.get("random_seed", "-")}'
                 f'&nbsp;·&nbsp;epochs {trainer.best_epoch} / {tc["max_epochs"]}'
                 f'&nbsp;·&nbsp;patience {tc.get("patience", "-")}'
                 f'&nbsp;·&nbsp;{sched_str}')

    # KPI tiles
    kpi_bvl_cls  = _col_class(bvl_col)
    kpi_r2_cls   = _col_class(r2_col)
    kpi_rmse_cls = _col_class(_r2_col(r2))
    kpi_mae_cls  = _col_class(_r2_col(r2))

    # learning rate formatted
    lr_html = _fmt_lr(tc['learning_rate'])

    # convergence badge inline
    if converged:
        conv_badge = ('<span class="conv-badge" '
                      'style="border-color:#1D9E75;color:#1D9E75;">CONVERGED</span>')
    else:
        conv_badge = ('<span class="conv-badge" '
                      'style="border-color:#D85A30;color:#D85A30;">NOT CONVERGED</span>')

    # R² cell colour
    r2_cell = f'<td class="{_col_class(r2_col)}">{r2:.4f}</td>'

    # processes list
    proc_str = ', '.join(dc.get('process_names', []))

    # DAG right-bottom section
    if b64_dag is not None:
        dag_section = f'''
      <div class="plot-label">D — causal structure (estimated DAG)</div>
      <div class="plot-box" style="flex:1.6">
        <img src="{b64_dag}" alt="DAG">
      </div>
      <div class="plot-cap">
        Directed edges from attention &phi;-weights (threshold 0.50) &nbsp;&middot;&nbsp;
        rows = target nodes, cols = source nodes
      </div>'''
    else:
        dag_section = '''
      <div class="plot-label">D — causal structure (estimated DAG)</div>
      <div class="dag-placeholder">
        DAG estimation not available — model uses standard self-attention (no LieAttention).<br>
        Enable LieAttention to recover the causal graph structure.
      </div>'''

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
{PAGE_CSS}
</style>
</head>
<body>
<div class="page">

  <!-- HEADER -->
  <div class="hdr-row">
    <div>
      <div class="hdr-title">CasualiT Surrogate — Training Report</div>
      <div class="hdr-meta">{meta_str}</div>
    </div>
    {badge_html}
  </div>
  <hr class="rule-heavy">

  <!-- BODY -->
  <div class="body">

    <!-- LEFT COLUMN -->
    <div class="left">

      <!-- KPI row -->
      <div class="kpi-row">
        <div class="kpi">
          <div class="kpi-l">Best val loss</div>
          <div class="kpi-v {kpi_bvl_cls}">{bvl:.6f}</div>
          <div class="kpi-s">epoch {trainer.best_epoch}</div>
        </div>
        <div class="kpi">
          <div class="kpi-l">Test R²</div>
          <div class="kpi-v {kpi_r2_cls}">{r2:.4f}</div>
          <div class="kpi-s">MSE {mse:.6f}</div>
        </div>
        <div class="kpi">
          <div class="kpi-l">Test RMSE</div>
          <div class="kpi-v {kpi_rmse_cls}">{rmse:.4f}</div>
          <div class="kpi-s">lower is better</div>
        </div>
        <div class="kpi">
          <div class="kpi-l">Test MAE</div>
          <div class="kpi-v {kpi_mae_cls}">{mae:.4f}</div>
          <div class="kpi-s">n_test {n_test}</div>
        </div>
      </div>

      <!-- 01 Model configuration -->
      <div class="sec-head">01 — model configuration</div>
      <hr class="rule-thin">
      <div class="two">
        <div>
          <div class="row"><span class="rk">Architecture</span><span class="rv">SimpleSurrogateModel</span></div>
          <div class="row"><span class="rk">Type</span><span class="rv">Transformer Encoder</span></div>
          <div class="row"><span class="rk">d_model</span><span class="rv">{mc['d_model_enc']}</span></div>
          <div class="row"><span class="rk">d_ff</span><span class="rv">{mc['d_ff']}</span></div>
          <div class="row"><span class="rk">Attn. heads</span><span class="rv">{mc['n_heads']}</span></div>
          <div class="row"><span class="rk">Enc. layers</span><span class="rv">{mc['e_layers']}</span></div>
          <div class="row"><span class="rk">Dropout</span><span class="rv">{mc['dropout_emb']}</span></div>
          <div class="row"><span class="rk">Activation</span><span class="rv">GeLU + Sigmoid</span></div>
          <div class="row"><span class="rk">Parameters</span><span class="rv">{total_p:,}</span></div>
          <div class="row"><span class="rk">Input dim</span><span class="rv">{n_feat} feat × {n_proc} proc.</span></div>
          <div class="row"><span class="rk">Output</span><span class="rv">scalar F ∈ [0, 1]</span></div>
        </div>
        <div>
          <div class="row"><span class="rk">Train samples</span><span class="rv">{n_train:,} ({round(100*n_train/(n_train+n_val+n_test))}%)</span></div>
          <div class="row"><span class="rk">Val samples</span><span class="rv">{n_val:,} ({round(100*n_val/(n_train+n_val+n_test))}%)</span></div>
          <div class="row"><span class="rk">Test samples</span><span class="rv">{n_test:,} ({round(100*n_test/(n_train+n_val+n_test))}%)</span></div>
          <div class="row"><span class="rk">Trajectories</span><span class="rv">{dc.get('n_trajectories', '-')}</span></div>
          <div class="row"><span class="rk">Scenarios</span><span class="rv">{dc.get('n_scenarios', '-')}</span></div>
          <div class="row"><span class="rk">Processes</span><span class="rv">{proc_str}</span></div>
          <div class="row"><span class="rk">Random seed</span><span class="rv">{dc.get('random_seed', '-')}</span></div>
        </div>
      </div>

      <!-- 02 Training & results -->
      <div class="sec-head">02 — training &amp; results</div>
      <hr class="rule-thin">
      <div class="two">
        <div>
          <div class="row"><span class="rk">Epochs</span><span class="rv">{trainer.best_epoch} / {tc['max_epochs']}</span></div>
          <div class="row"><span class="rk">Batch size</span><span class="rv">{tc['batch_size']}</span></div>
          <div class="row"><span class="rk">Learning rate</span><span class="rv">{lr_html}</span></div>
          <div class="row"><span class="rk">Weight decay</span><span class="rv">{tc['weight_decay']}</span></div>
          <div class="row"><span class="rk">Loss fn</span><span class="rv">MSE</span></div>
          <div class="row"><span class="rk">Optimizer</span><span class="rv">AdamW</span></div>
          <div class="row"><span class="rk">Scheduler</span><span class="rv">{'ReduceLROnPlateau' if tc.get('use_scheduler') else 'None'}</span></div>
          <div class="row"><span class="rk">Patience</span><span class="rv">{tc.get('patience', '-')}</span></div>
          <div class="row"><span class="rk">Device</span><span class="rv">{getattr(trainer, 'device', 'auto')}</span></div>
        </div>
        <div>
          <div class="row"><span class="rk">Final train MSE</span><span class="rv">{final_train:.6f}</span></div>
          <div class="row"><span class="rk">Final val MSE</span><span class="rv">{final_val:.6f}</span></div>
          <div class="row"><span class="rk">Best val loss</span><span class="rv {kpi_bvl_cls}">{bvl:.6f}</span></div>
          <div class="row"><span class="rk">Test MSE</span><span class="rv">{mse:.6f}</span></div>
          <div class="row"><span class="rk">Test RMSE</span><span class="rv">{rmse:.4f}</span></div>
          <div class="row"><span class="rk">Test MAE</span><span class="rv">{mae:.4f}</span></div>
          <div class="row"><span class="rk">Test R²</span><span class="rv {kpi_r2_cls}">{r2:.4f}</span></div>
          <div class="row"><span class="rk">Converged</span>
            <span class="rv {'g' if converged else 'r'}">{'Yes' if converged else 'No'}</span>
          </div>
        </div>
      </div>
      <div class="conv-row">{conv_badge}</div>

      <!-- 03 Test metrics table -->
      <div class="sec-head" style="margin-top:4px">03 — test metrics</div>
      <hr class="rule-thin">
      <table class="tbl">
        <thead>
          <tr>
            <th>Output</th><th>MSE</th><th>RMSE</th><th>MAE</th><th>R²</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>F (Reliability)</td>
            <td>{mse:.6f}</td>
            <td>{rmse:.4f}</td>
            <td>{mae:.4f}</td>
            {r2_cell}
          </tr>
        </tbody>
      </table>
      <div class="note">R² ≥ 0.90 → good &nbsp;·&nbsp; 0.70–0.90 → acceptable &nbsp;·&nbsp; &lt;0.70 → poor</div>

    </div><!-- /left -->

    <div class="divider"></div>

    <!-- RIGHT COLUMN -->
    <div class="right">

      <!-- A: Loss curve (tall) -->
      <div class="plot-label">A — training history</div>
      <div class="plot-box" style="flex:2.0">
        <img src="{b64_loss}" alt="loss curve">
      </div>
      <div class="plot-cap">Training and Validation Loss (MSE)</div>

      <!-- B: scatter + residuals (side by side) -->
      <div class="plot-label">B — predictions &amp; residuals</div>
      <div class="plots-row" style="flex:1.9">
        <div style="flex:1;display:flex;flex-direction:column;min-height:0">
          <div class="plot-box" style="flex:1">
            <img src="{b64_scatter}" alt="scatter">
          </div>
          <div class="plot-cap">Predicted vs True F</div>
        </div>
        <div style="flex:1;display:flex;flex-direction:column;min-height:0">
          <div class="plot-box" style="flex:1">
            <img src="{b64_residuals}" alt="residuals">
          </div>
          <div class="plot-cap">Residual Distribution</div>
        </div>
        <div style="flex:1;display:flex;flex-direction:column;min-height:0">
          <div class="plot-box" style="flex:1">
            <img src="{b64_fdist}" alt="f distribution">
          </div>
          <div class="plot-cap">F Distribution: True vs Predicted</div>
        </div>
      </div>

      <!-- C / D: DAG (bottom) -->
      {dag_section}

    </div><!-- /right -->

  </div><!-- /body -->

  <!-- FOOTER -->
  <div class="footer">
    <span>auto-generated &nbsp;·&nbsp; {config['report'].get('output_dir', 'causaliT/reports/surrogate')}</span>
    <span>causaliT surrogate</span>
  </div>

</div><!-- /page -->
</body>
</html>"""
    return html


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf_report(trainer, eval_results: dict, config: dict,
                        output_dir: str) -> Optional[Path]:
    """
    Generate a single-page A4-landscape PDF report for CasualiT surrogate training.

    Style is identical to the Uncertainty Predictor report (HTML → WeasyPrint).

    Args:
        trainer      : SurrogateTrainer (needs .model, .history, .best_epoch,
                       .best_val_loss, .{train,val,test}_loader)
        eval_results : dict with test_mse, test_mae, test_rmse, test_r2,
                       predictions (np.ndarray), targets (np.ndarray)
        config       : SURROGATE_CONFIG dict
        output_dir   : directory where the PDF is written

    Returns:
        Path to the generated PDF, or None on failure.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print('Warning: matplotlib not available — skipping PDF report')
        return None

    try:
        from weasyprint import HTML as WPHtml
    except ImportError:
        print('Warning: weasyprint not available — skipping PDF report')
        print('Install with: pip install weasyprint')
        return None

    try:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ts_str   = datetime.now().strftime('%Y%m%d_%H%M%S')
        pdf_path = out_dir / f'surrogate_training_report_{ts_str}.pdf'

        print(f'\nGenerating PDF report: {pdf_path}')

        preds   = eval_results['predictions']
        targets = eval_results['targets']

        # Generate plots (inline base64 — no temp files needed)
        print('  Rendering plots...')
        b64_loss     = _plot_loss(trainer.history, trainer.best_epoch)
        b64_scatter  = _plot_scatter(preds, targets, eval_results['test_r2'])
        b64_residuals = _plot_residuals(preds, targets)
        b64_fdist    = _plot_fdist(preds, targets)

        # DAG (conditional)
        phi = _extract_phi(trainer)
        if phi is not None:
            proc_names = config['data'].get('process_names', [])
            n = min(phi.shape[0], len(proc_names)) if proc_names else phi.shape[0]
            if not proc_names:
                proc_names = [f'P{i}' for i in range(n)]
            b64_dag = _plot_dag(phi[:n, :n], proc_names[:n])
            print('  DAG rendered from LieAttention phi tensors.')
        else:
            b64_dag = None
            print('  DAG not available (standard attention — no phi tensors).')

        # Build HTML
        print('  Building HTML...')
        html_str = _build_html(
            trainer, eval_results, config,
            b64_loss, b64_scatter, b64_residuals, b64_fdist, b64_dag,
        )

        # Render PDF
        print('  Rendering PDF via WeasyPrint...')
        WPHtml(string=html_str).write_pdf(str(pdf_path))

        print(f'PDF report saved: {pdf_path}')
        return pdf_path

    except Exception:
        print('Warning: PDF report generation failed.')
        traceback.print_exc()
        return None