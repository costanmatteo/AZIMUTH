"""
PDF Report Generator — CasualiT Surrogate Training
===================================================
A4 landscape, single-page layout.

Style spec (mirrors predictor / controller reports):
  - Font family : Courier (monospace) — regular, bold, oblique
  - Page        : A4 landscape, 12 mm margins
  - Layout      : left column (108 mm) | 6 mm gap | right column (remaining)
                  right column split: top 66 % (plots 2x2) / bottom 34 % (DAG)
  - Header      : title left, date right, 1.5 pt green accent top, 1 pt gray rule bottom
  - Footer      : page number right, 0.5 pt gray rule top
  - Colors      : green #1D9E75 / red #D85A30 / amber #BA7517 / grays
  - KPI boxes   : 2x2 grid, C_BGRAY bg, color-coded values
  - Plots       : matplotlib DejaVu Sans, DPI 150, unified color palette
  - DAG section : conditional on LieAttention phi tensors; placeholder otherwise

Drop-in replacement for generate_pdf_report() in train_surrogate.py.
Signature is identical — main() needs no changes.
"""

from __future__ import annotations

import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# ── ReportLab ────────────────────────────────────────────────────────────────
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, FrameBreak,
    Paragraph, Spacer, Image, Table, TableStyle,
)
from reportlab.platypus.flowables import HRFlowable

# ─────────────────────────────────────────────────────────────────────────────
#  Page geometry
# ─────────────────────────────────────────────────────────────────────────────
PW, PH  = landscape(A4)          # 841.89 x 595.28 pts
M       = 1.2  * cm              # uniform margin
HDR_H   = 1.6  * cm              # header band height
FTR_H   = 0.65 * cm              # footer band height
LW      = 108  * mm              # left-column width
GAP     = 6    * mm              # gap between columns
FULL_W  = PW - 2 * M             # total text width
RW      = FULL_W - LW - GAP      # right-column width
BODY_H  = PH - 2 * M - HDR_H - FTR_H - 2 * mm   # usable body height

# ─────────────────────────────────────────────────────────────────────────────
#  Colour palette
# ─────────────────────────────────────────────────────────────────────────────
C_GREEN  = colors.HexColor('#1D9E75')
C_RED    = colors.HexColor('#D85A30')
C_AMBER  = colors.HexColor('#BA7517')
C_BLACK  = colors.black
C_MUTED  = colors.HexColor('#666666')
C_LGRAY  = colors.HexColor('#EEEEEE')
C_MGRAY  = colors.HexColor('#CCCCCC')
C_BGRAY  = colors.HexColor('#F7F7F7')
C_TGRAY  = colors.HexColor('#888888')
C_DKGRN  = colors.HexColor('#145C46')    # top-accent line

# matplotlib hex equivalents (for plot code)
MPL_GREEN = '#1D9E75'
MPL_RED   = '#D85A30'
MPL_AMBER = '#BA7517'
MPL_MGRAY = '#CCCCCC'
MPL_BGRAY = '#F7F7F7'

# ─────────────────────────────────────────────────────────────────────────────
#  Font-size tokens
# ─────────────────────────────────────────────────────────────────────────────
FS_TITLE   = 12
FS_SUB     = 8
FS_SECTION = 8
FS_BODY    = 7.5
FS_KPI_LBL = 6.5
FS_KPI_VAL = 13
FS_KPI_SUB = 6.5
FS_CAPTION = 6.5
FS_FOOTER  = 7

# ─────────────────────────────────────────────────────────────────────────────
#  Style factory
# ─────────────────────────────────────────────────────────────────────────────

def _ps(name, size, bold=False, italic=False,
        color=C_BLACK, align=TA_LEFT, lm=1.35):
    font = 'Courier-Bold' if bold else ('Courier-Oblique' if italic else 'Courier')
    return ParagraphStyle(
        name, fontName=font, fontSize=size,
        leading=size * lm, textColor=color, alignment=align,
    )

ST_SECTION = _ps('sr_sec',     FS_SECTION, bold=True,   color=C_TGRAY)
ST_BODY    = _ps('sr_body',    FS_BODY)
ST_KPI_LBL = _ps('sr_kpilbl', FS_KPI_LBL, bold=True,   color=C_TGRAY,  align=TA_CENTER)
ST_KPI_SUB = _ps('sr_kpisub', FS_KPI_SUB, italic=True, color=C_MUTED,  align=TA_CENTER)
ST_CAPTION = _ps('sr_cap',    FS_CAPTION, italic=True, color=C_MUTED,  align=TA_CENTER)
ST_FOOTER  = _ps('sr_ftr',    FS_FOOTER,                color=C_TGRAY,  align=TA_RIGHT)

# ─────────────────────────────────────────────────────────────────────────────
#  Helper primitives
# ─────────────────────────────────────────────────────────────────────────────

def _hex(c) -> str:
    """Return '#rrggbb' string from a ReportLab color."""
    try:
        r, g, b = int(c.red * 255), int(c.green * 255), int(c.blue * 255)
        return f'#{r:02x}{g:02x}{b:02x}'
    except Exception:
        return '#000000'


def _r2_color(v: float):
    return C_GREEN if v >= 0.9 else (C_AMBER if v >= 0.7 else C_RED)


def _loss_color(v: float):
    return C_GREEN if v < 0.005 else (C_AMBER if v < 0.02 else C_RED)


def _fmt_lr(lr: float) -> str:
    import math
    if lr == 0:
        return '0'
    exp = int(math.floor(math.log10(abs(lr))))
    m   = lr / (10 ** exp)
    return f'{m:.3f} x 10^{exp}'


def _hr():
    return HRFlowable(width='100%', thickness=0.5, color=C_MGRAY,
                      spaceAfter=1.5, spaceBefore=1.5)


def _section_hdr(text: str) -> list:
    return [
        Spacer(1, 2.5 * mm),
        Paragraph(text, ST_SECTION),
        _hr(),
    ]


def _kv(key: str, val: str, val_color=C_BLACK) -> Paragraph:
    vc = _hex(val_color)
    return Paragraph(
        f'<font name="Courier-Bold">{key}:</font>'
        f'  <font color="{vc}">{val}</font>',
        ST_BODY,
    )


# ── KPI tile ──────────────────────────────────────────────────────────────────

def _kpi_tile(label: str, value: str, sub: str, val_color) -> Table:
    val_style = ParagraphStyle(
        f'kpiv_{label[:4]}',
        fontName='Courier-Bold', fontSize=FS_KPI_VAL,
        leading=FS_KPI_VAL * 1.2, textColor=val_color, alignment=TA_CENTER,
    )
    cell_w = LW / 2 - 3.5 * mm
    inner = Table(
        [[Paragraph(label, ST_KPI_LBL)],
         [Paragraph(value, val_style)],
         [Paragraph(sub,   ST_KPI_SUB)]],
        colWidths=[cell_w],
    )
    inner.setStyle(TableStyle([
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING',    (0, 0), (-1, -1), 1),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
    ]))
    outer = Table([[inner]], colWidths=[cell_w])
    outer.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), C_BGRAY),
        ('BOX',           (0, 0), (-1, -1), 0.75, C_MGRAY),
        ('TOPPADDING',    (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING',   (0, 0), (-1, -1), 3),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 3),
    ]))
    return outer


def _kpi_grid(kpis: list) -> Table:
    """4 KPI tiles in a 2x2 grid."""
    w = LW / 2 - 2 * mm
    t = Table(
        [[_kpi_tile(*kpis[0]), _kpi_tile(*kpis[1])],
         [_kpi_tile(*kpis[2]), _kpi_tile(*kpis[3])]],
        colWidths=[w, w],
    )
    t.setStyle(TableStyle([
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 1),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 1),
        ('TOPPADDING',    (0, 0), (-1, -1), 1),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
    ]))
    return t


def _badge(converged: bool) -> Paragraph:
    if converged:
        bg, fg, txt = '#1D9E75', '#FFFFFF', 'CONVERGED'
    else:
        bg, fg, txt = '#D85A30', '#FFFFFF', 'NOT CONVERGED'
    s = ParagraphStyle(
        'badge', fontName='Courier-Bold', fontSize=7, leading=9,
        textColor=colors.HexColor(fg), backColor=colors.HexColor(bg),
        alignment=TA_CENTER, borderPadding=(2, 8, 2, 8),
    )
    return Paragraph(txt, s)


# ─────────────────────────────────────────────────────────────────────────────
#  matplotlib plots
# ─────────────────────────────────────────────────────────────────────────────

def _mpl_style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=8, fontweight='bold', pad=4)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.grid(True, alpha=0.25, linestyle='--', color=MPL_MGRAY)
    ax.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_edgecolor(MPL_MGRAY)
        sp.set_linewidth(0.6)


def _save(fig, path, dpi=150):
    import matplotlib.pyplot as plt
    fig.savefig(str(path), dpi=dpi, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def _plot_loss(history: dict, best_epoch: int, out: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    fig.patch.set_facecolor('white')
    ep = range(len(history['train_loss']))
    ax.plot(ep, history['train_loss'], color=MPL_GREEN, lw=1.5, label='Train MSE')
    ax.plot(ep, history['val_loss'],   color=MPL_RED,   lw=1.5, ls='--', label='Val MSE')
    if best_epoch is not None and best_epoch < len(history['train_loss']):
        ax.axvline(best_epoch, color=MPL_AMBER, lw=1.0, ls=':',
                   label=f'best (ep. {best_epoch})')
    ax.legend(fontsize=6.5, framealpha=0.8, edgecolor=MPL_MGRAY, loc='upper right')
    _mpl_style(ax, 'Training History \u2014 MSE Loss', 'Epoch', 'MSE Loss')
    fig.tight_layout(pad=0.4)
    _save(fig, out)


def _plot_scatter(preds, targets, r2, out: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    fig.patch.set_facecolor('white')
    ax.scatter(targets, preds, c=preds, cmap='viridis', alpha=0.5, s=10, lw=0)
    lo = min(float(targets.min()), float(preds.min())) - 0.02
    hi = max(float(targets.max()), float(preds.max())) + 0.02
    ax.plot([lo, hi], [lo, hi], color=MPL_RED, lw=1.5, ls='--', label='Perfect prediction')
    props = dict(boxstyle='round,pad=0.3', facecolor=MPL_BGRAY, edgecolor=MPL_MGRAY, alpha=0.9)
    ax.text(0.05, 0.93, f'R\u00b2 = {r2:.4f}',
            transform=ax.transAxes, fontsize=7, va='top', bbox=props)
    ax.legend(fontsize=6.5, framealpha=0.8, edgecolor=MPL_MGRAY)
    _mpl_style(ax, 'Predicted vs True Reliability F', 'True F', 'Predicted F')
    fig.tight_layout(pad=0.4)
    _save(fig, out)


def _plot_residuals(preds, targets, out: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    res = preds - targets
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    fig.patch.set_facecolor('white')
    ax.hist(res, bins=30, color=MPL_GREEN, edgecolor='white', alpha=0.85, lw=0.4)
    ax.axvline(0, color=MPL_RED, lw=1.5, ls='--')
    props = dict(boxstyle='round,pad=0.3', facecolor=MPL_BGRAY, edgecolor=MPL_MGRAY, alpha=0.9)
    ax.text(0.97, 0.95, f'Mean: {res.mean():.4f}\nStd:  {res.std():.4f}',
            transform=ax.transAxes, fontsize=6.5, va='top', ha='right', bbox=props)
    _mpl_style(ax, 'Residual Distribution', 'F_pred \u2212 F_true', 'Count')
    fig.tight_layout(pad=0.4)
    _save(fig, out)


def _plot_fdist(preds, targets, out: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    fig.patch.set_facecolor('white')
    ax.hist(targets, bins=30, color=MPL_GREEN, alpha=0.55, density=True,
            label='True F', edgecolor='white', lw=0.4)
    ax.hist(preds,   bins=30, color=MPL_AMBER, alpha=0.55, density=True,
            label='Predicted F', edgecolor='white', lw=0.4)
    ax.legend(fontsize=6.5, framealpha=0.8, edgecolor=MPL_MGRAY)
    _mpl_style(ax, 'F Distribution: True vs Predicted', 'F value', 'Density')
    fig.tight_layout(pad=0.4)
    _save(fig, out)


def _plot_dag(phi: np.ndarray, proc_names: list, out: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(proc_names)
    binary = (phi >= 0.5).astype(float)

    fig_w = float(RW) / 72 * 0.92
    fig_h = float(BODY_H) * 0.30 / 72 * 0.88
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('white')

    im = ax.imshow(phi, cmap='Greens', vmin=0, vmax=1, aspect='auto')
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label('Edge Probability', fontsize=6)
    cb.ax.tick_params(labelsize=5.5)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(proc_names, fontsize=6.5, rotation=45, ha='right')
    ax.set_yticklabels(proc_names, fontsize=6.5)

    for i in range(n):
        for j in range(n):
            if binary[i, j] == 1:
                ax.plot(j, i, 'o', color=MPL_GREEN, ms=5, mew=0.5, mec='white')
            else:
                ax.plot(j, i, 'x', color='white',   ms=4, mew=0.8)

    ax.set_xlabel('Source node', fontsize=7)
    ax.set_ylabel('Target node', fontsize=7)
    ax.set_title('Estimated Causal DAG (LieAttention \u03c6)',
                 fontsize=8, fontweight='bold', pad=4)
    ax.tick_params(labelsize=6.5)
    fig.tight_layout(pad=0.4)
    _save(fig, out)


# ─────────────────────────────────────────────────────────────────────────────
#  Header / footer page callbacks
# ─────────────────────────────────────────────────────────────────────────────

def _make_callbacks(title: str, date: str):
    def _draw(canvas, doc):
        canvas.saveState()
        # Top accent line (dark green)
        canvas.setStrokeColor(C_DKGRN)
        canvas.setLineWidth(1.5)
        canvas.line(M, PH - 1.5, PW - M, PH - 1.5)
        # Title
        canvas.setFont('Courier-Bold', FS_TITLE)
        canvas.setFillColor(C_BLACK)
        canvas.drawString(M, PH - M - 0.38 * cm, title)
        # Date
        canvas.setFont('Courier', FS_SUB)
        canvas.setFillColor(C_MUTED)
        canvas.drawRightString(PW - M, PH - M - 0.38 * cm, date)
        # Header rule
        canvas.setStrokeColor(C_MGRAY)
        canvas.setLineWidth(1.0)
        canvas.line(M, PH - M - HDR_H + 2, PW - M, PH - M - HDR_H + 2)
        # Footer rule
        canvas.setLineWidth(0.5)
        canvas.line(M, M + FTR_H - 1, PW - M, M + FTR_H - 1)
        # Page number
        canvas.setFont('Courier', FS_FOOTER)
        canvas.setFillColor(C_TGRAY)
        canvas.drawRightString(PW - M, M + 1.5, f'Page {doc.page}')
        canvas.restoreState()
    return _draw, _draw


# ─────────────────────────────────────────────────────────────────────────────
#  Left-column flowables
# ─────────────────────────────────────────────────────────────────────────────

def _left_column(trainer, eval_results: dict, config: dict) -> list:
    model   = trainer.model
    history = trainer.history

    n_train = len(trainer.train_loader.dataset)  if trainer.train_loader  else 0
    n_val   = len(trainer.val_loader.dataset)    if trainer.val_loader    else 0
    n_test  = len(trainer.test_loader.dataset)   if trainer.test_loader   else 0
    total_p = sum(p.numel() for p in model.parameters())
    n_proc  = getattr(model, 'n_processes', '?') or '?'
    n_feat  = getattr(model, 'n_features',  '?') or '?'

    bvl  = trainer.best_val_loss
    r2   = eval_results['test_r2']
    rmse = eval_results['test_rmse']
    mae  = eval_results['test_mae']

    # ── KPI grid ──────────────────────────────────────────────────────────────
    kpis = [
        ('BEST VAL LOSS',  f'{bvl:.6f}',  f'epoch {trainer.best_epoch}',   _loss_color(bvl)),
        ('TEST R\u00b2',   f'{r2:.4f}',   f'MSE {eval_results["test_mse"]:.6f}', _r2_color(r2)),
        ('TEST RMSE',      f'{rmse:.4f}', 'lower is better',                _r2_color(r2)),
        ('TEST MAE',       f'{mae:.4f}',  f'n_test {n_test}',               _r2_color(r2)),
    ]
    story = [_kpi_grid(kpis), Spacer(1, 3 * mm)]

    # ── 01 Model architecture ──────────────────────────────────────────────────
    story += _section_hdr('01 \u2014 MODEL ARCHITECTURE')
    mc = config['model']
    for k, v in [
        ('Architecture',  'SimpleSurrogateModel (Transformer Enc.)'),
        ('d_model',       str(mc['d_model_enc'])),
        ('d_ff',          str(mc['d_ff'])),
        ('Attn. Heads',   str(mc['n_heads'])),
        ('Enc. Layers',   str(mc['e_layers'])),
        ('Dropout',       str(mc['dropout_emb'])),
        ('Total Params',  f'{total_p:,}'),
        ('Input Dim',     f'{n_feat} feat x {n_proc} proc.'),
        ('Output',        'scalar F in [0, 1]'),
        ('Activation',    'GeLU + Sigmoid head'),
    ]:
        story.append(_kv(k, v))
    story.append(Spacer(1, 1.5 * mm))

    # ── 02 Training configuration ──────────────────────────────────────────────
    story += _section_hdr('02 \u2014 TRAINING CONFIGURATION')
    tc   = config['training']
    sched = 'ReduceLROnPlateau' if tc.get('use_scheduler') else 'None'
    for k, v in [
        ('Max Epochs',    str(tc['max_epochs'])),
        ('Batch Size',    str(tc['batch_size'])),
        ('Learning Rate', _fmt_lr(tc['learning_rate'])),
        ('Weight Decay',  str(tc['weight_decay'])),
        ('Loss Fn',       'MSE'),
        ('Optimizer',     'AdamW'),
        ('Scheduler',     sched),
        ('Patience',      str(tc.get('patience', '-'))),
        ('Device',        str(getattr(trainer, 'device', 'auto'))),
    ]:
        story.append(_kv(k, v))
    story.append(Spacer(1, 1.5 * mm))

    # ── 03 Dataset ────────────────────────────────────────────────────────────
    story += _section_hdr('03 \u2014 DATASET')
    dc = config['data']
    for k, v in [
        ('Train Samples', str(n_train)),
        ('Val Samples',   str(n_val)),
        ('Test Samples',  str(n_test)),
        ('Trajectories',  str(dc.get('n_trajectories', '-'))),
        ('Scenarios',     str(dc.get('n_scenarios', '-'))),
        ('Processes',     ', '.join(dc.get('process_names', []))),
        ('Random Seed',   str(dc.get('random_seed', '-'))),
    ]:
        story.append(_kv(k, v))
    story.append(Spacer(1, 1.5 * mm))

    # ── 04 Training results ───────────────────────────────────────────────────
    story += _section_hdr('04 \u2014 TRAINING RESULTS')
    final_train = history['train_loss'][-1] if history.get('train_loss') else float('nan')
    final_val   = history['val_loss'][-1]   if history.get('val_loss')   else float('nan')
    for k, v in [
        ('Best Epoch',      str(trainer.best_epoch)),
        ('Best Val Loss',   f'{bvl:.6f}'),
        ('Final Train MSE', f'{final_train:.6f}'),
        ('Final Val MSE',   f'{final_val:.6f}'),
    ]:
        story.append(_kv(k, v))
    story.append(Spacer(1, 1.5 * mm))
    story.append(_badge(bvl < 0.01))
    story.append(Spacer(1, 1.5 * mm))

    # ── 05 Test metrics table ─────────────────────────────────────────────────
    story += _section_hdr('05 \u2014 TEST METRICS')
    r2c = _hex(_r2_color(r2))
    hdr = [Paragraph(f'<b>{h}</b>', ST_BODY)
           for h in ['Metric', 'MSE', 'RMSE', 'MAE', 'R\u00b2']]
    row = [
        Paragraph('<font name="Courier">F (Reliability)</font>', ST_BODY),
        Paragraph(f'<font name="Courier">{eval_results["test_mse"]:.6f}</font>', ST_BODY),
        Paragraph(f'<font name="Courier">{rmse:.4f}</font>', ST_BODY),
        Paragraph(f'<font name="Courier">{mae:.4f}</font>', ST_BODY),
        Paragraph(f'<font name="Courier" color="{r2c}"><b>{r2:.4f}</b></font>', ST_BODY),
    ]
    cw = [LW * 0.33, LW * 0.17, LW * 0.17, LW * 0.17, LW * 0.16]
    tbl = Table([hdr, row], colWidths=cw)
    tbl.setStyle(TableStyle([
        ('FONTNAME',      (0, 0), (-1, 0), 'Courier-Bold'),
        ('FONTSIZE',      (0, 0), (-1, -1), FS_BODY),
        ('ALIGN',         (1, 0), (-1, -1), 'CENTER'),
        ('ALIGN',         (0, 0), (0, -1), 'LEFT'),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING',    (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LINEABOVE',     (0, 0), (-1, 0), 1.5, colors.black),
        ('LINEBELOW',     (0, 0), (-1, 0), 0.5, colors.black),
        ('LINEBELOW',     (0, -1), (-1, -1), 1.5, colors.black),
        ('BACKGROUND',    (0, 1), (-1, 1), C_LGRAY),
    ]))
    story.append(tbl)
    return story


# ─────────────────────────────────────────────────────────────────────────────
#  Right-column top: 2x2 plot grid
# ─────────────────────────────────────────────────────────────────────────────

def _right_top(trainer, eval_results: dict, tmp: Path) -> list:
    preds   = eval_results['predictions']
    targets = eval_results['targets']
    r2      = eval_results['test_r2']

    p_loss  = tmp / 'sr_loss.png'
    p_scat  = tmp / 'sr_scatter.png'
    p_res   = tmp / 'sr_residuals.png'
    p_fdist = tmp / 'sr_fdist.png'

    _plot_loss(trainer.history, trainer.best_epoch, p_loss)
    _plot_scatter(preds, targets, r2,   p_scat)
    _plot_residuals(preds, targets,     p_res)
    _plot_fdist(preds, targets,         p_fdist)

    iw = (RW - 5 * mm) / 2
    ih = BODY_H * 0.63 / 2 - 6 * mm

    def _img_cell(path, caption):
        t = Table(
            [[Image(str(path), width=iw, height=ih)],
             [Paragraph(caption, ST_CAPTION)]],
            colWidths=[iw],
        )
        t.setStyle(TableStyle([
            ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING',    (0, 0), (-1, -1), 1),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
        ]))
        return t

    grid = Table(
        [[_img_cell(p_loss,  'Training History \u2014 MSE Loss'),
          _img_cell(p_scat,  'Predicted vs True F')],
         [_img_cell(p_res,   'Residual Distribution'),
          _img_cell(p_fdist, 'F Distribution: True vs Predicted')]],
        colWidths=[RW / 2, RW / 2],
    )
    grid.setStyle(TableStyle([
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 2),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 2),
        ('TOPPADDING',    (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    return [grid]


# ─────────────────────────────────────────────────────────────────────────────
#  Right-column bottom: DAG section
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
        if val is not None:
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


def _right_bottom(trainer, config: dict, tmp: Path) -> list:
    story = list(_section_hdr('06 \u2014 CAUSAL STRUCTURE (ESTIMATED DAG)'))
    phi = _extract_phi(trainer)

    if phi is None:
        ph_s = ParagraphStyle(
            'dag_ph', fontName='Courier-Oblique', fontSize=FS_BODY,
            leading=FS_BODY * 1.5, textColor=C_MUTED, alignment=TA_CENTER,
        )
        box = Table(
            [[Paragraph(
                'DAG ESTIMATION NOT AVAILABLE\n'
                'Model uses standard self-attention (no LieAttention).\n'
                'Enable LieAttention to recover the causal graph.',
                ph_s,
            )]],
            colWidths=[RW - 6 * mm],
        )
        box.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), C_BGRAY),
            ('BOX',           (0, 0), (-1, -1), 0.75, C_MGRAY),
            ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING',    (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        story.append(box)
    else:
        proc_names = config['data'].get('process_names', [])
        n = min(phi.shape[0], len(proc_names)) if proc_names else phi.shape[0]
        if not proc_names:
            proc_names = [f'P{i}' for i in range(n)]
        phi = phi[:n, :n]
        proc_names = proc_names[:n]

        p_dag = tmp / 'sr_dag.png'
        _plot_dag(phi, proc_names, p_dag)

        dag_h = BODY_H * 0.27
        dag_w = RW - 4 * mm
        story.append(Image(str(p_dag), width=dag_w, height=dag_h))
        story.append(Spacer(1, 1 * mm))
        story.append(Paragraph(
            'Directed edges recovered from attention \u03c6-weights (threshold = 0.50). '
            'Rows = target nodes, Cols = source nodes.',
            ST_CAPTION,
        ))
    return story


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf_report(trainer, eval_results: dict, config: dict,
                        output_dir: str) -> Optional[Path]:
    """
    Generate a professional PDF report for CasualiT surrogate training.

    Layout  : A4 landscape, Courier monospace, green/amber/red KPI badges,
              three-frame layout with header/footer, unified matplotlib plots,
              conditional DAG section (shown only when LieAttention phi tensors
              are available on the trainer/model).

    Args:
        trainer     : SurrogateTrainer instance (must have .model, .history,
                      .best_epoch, .best_val_loss, .{train,val,test}_loader)
        eval_results: dict with keys test_mse, test_mae, test_rmse, test_r2,
                      predictions, targets
        config      : SURROGATE_CONFIG dict
        output_dir  : directory where the PDF will be written

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
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ts      = datetime.now()
        ts_file = ts.strftime('%Y%m%d_%H%M%S')
        ts_disp = ts.strftime('%Y-%m-%d  %H:%M:%S')
        pdf_path = out_dir / f'surrogate_training_report_{ts_file}.pdf'

        print(f'\nGenerating PDF report: {pdf_path}')

        with tempfile.TemporaryDirectory() as _tmp:
            tmp = Path(_tmp)

            # ── Build flowable lists ──────────────────────────────────────────
            left  = _left_column(trainer, eval_results, config)
            r_top = _right_top(trainer, eval_results, tmp)
            r_bot = _right_bottom(trainer, config, tmp)

            # ── Frames ───────────────────────────────────────────────────────
            body_y  = M + FTR_H + 1 * mm
            right_x = M + LW + GAP

            f_left = Frame(
                M, body_y, LW, BODY_H,
                leftPadding=0, rightPadding=0,
                topPadding=0, bottomPadding=0, id='left',
            )
            f_r_top = Frame(
                right_x, body_y + BODY_H * 0.34, RW, BODY_H * 0.66,
                leftPadding=0, rightPadding=0,
                topPadding=0, bottomPadding=0, id='r_top',
            )
            f_r_bot = Frame(
                right_x, body_y, RW, BODY_H * 0.34,
                leftPadding=0, rightPadding=0,
                topPadding=0, bottomPadding=0, id='r_bot',
            )

            # ── Header/footer callbacks ───────────────────────────────────────
            on_page, _ = _make_callbacks(
                'CasualiT \u2014 Surrogate Training Report',
                ts_disp,
            )

            # ── Assemble document ─────────────────────────────────────────────
            doc = BaseDocTemplate(
                str(pdf_path),
                pagesize=landscape(A4),
                leftMargin=M, rightMargin=M,
                topMargin=M + HDR_H, bottomMargin=M + FTR_H,
            )
            doc.addPageTemplates([
                PageTemplate(
                    id='main',
                    frames=[f_left, f_r_top, f_r_bot],
                    onPage=on_page,
                )
            ])
            doc.build(left + [FrameBreak()] + r_top + [FrameBreak()] + r_bot)

        print(f'PDF report saved: {pdf_path}')
        return pdf_path

    except Exception:
        print('Warning: PDF report generation failed.')
        traceback.print_exc()
        return None