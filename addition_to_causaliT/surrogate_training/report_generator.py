"""
PDF Report Generator for Surrogate Transformer Training
A4 Landscape, single page. Layout and style mirror uncertainty_predictor/report_generator.py exactly:
  - Courier monospace font throughout
  - Same color palette (C_GREEN / C_RED / C_AMBER)
  - Same page geometry (HDR / left-column / right-column / FTR frames)
  - Same KPI bar (4 boxes, label / big-value / sub-line)
  - Same section-header style (small-caps + thin HR)
  - Same kv_table helper (key LEFT, value RIGHT)

Public API
----------
    generate_surrogate_training_report(
        config, history, eval_results,
        input_dim, output_dim, total_params,
        n_train, n_val, n_test,
        checkpoint_dir,
        timestamp=None,
        floor_metrics=None,
    ) -> str          # path to the generated PDF
"""

import math
from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, FrameBreak,
    Paragraph, Spacer, Image, Table, TableStyle,
)
from reportlab.platypus.flowables import HRFlowable, KeepInFrame

# ── page geometry (identical to uncertainty_predictor) ────────────────────────
PW, PH  = landscape(A4)
M       = 0.8 * cm
HDR_H   = 1.6 * cm
FTR_H   = 0.65 * cm
LW      = 108 * mm
GAP     = 6 * mm
FULL_W  = PW - 2 * M
RW      = FULL_W - LW - GAP
BODY_H  = PH - 2 * M - HDR_H - FTR_H - 2 * mm

# ── colors (same palette) ─────────────────────────────────────────────────────
C_GREEN = colors.HexColor('#1D9E75')
C_RED   = colors.HexColor('#D85A30')
C_AMBER = colors.HexColor('#BA7517')
C_BLACK = colors.black
C_GRAY  = colors.HexColor('#AAAAAA')
C_LGRAY = colors.HexColor('#DDDDDD')
C_VGRAY = colors.HexColor('#F8F8F8')
C_MUTED = colors.HexColor('#666666')

# ── font sizes ────────────────────────────────────────────────────────────────
FS_TITLE   = 15
FS_META    = 7.5
FS_SECTION = 7
FS_BODY    = 7.5
FS_KPI_LBL = 6.5
FS_KPI_VAL = 11
FS_KPI_SUB = 6.5
FS_CAPTION = 6.5
FS_BADGE   = 6.5


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════

def r2_color(v):
    return C_GREEN if v >= 0.9 else (C_AMBER if v >= 0.7 else C_RED)

def rmse_color(v, threshold_good=0.05, threshold_ok=0.15):
    """Lower is better."""
    return C_GREEN if v <= threshold_good else (C_AMBER if v <= threshold_ok else C_RED)

def floor_color(v, threshold_good=0.01, threshold_ok=0.05):
    """Floor fidelity MSE — lower is better."""
    return C_GREEN if v <= threshold_good else (C_AMBER if v <= threshold_ok else C_RED)


def fmt_lr(lr):
    if lr == 0:
        return "0"
    exp  = int(math.floor(math.log10(abs(lr))))
    mant = lr / (10 ** exp)
    return f"{mant:.3f} x 10^{exp}"


def short_dir(d, max_len=35):
    parts = Path(d).parts
    short = str(Path(*parts[-2:])) if len(parts) >= 2 else str(d)
    return short if len(short) <= max_len else '...' + short[-(max_len - 3):]


def scale_img(path, max_w, max_h):
    if not Path(path).exists():
        return _placeholder(Path(path).name, max_w, max_h)
    img   = Image(str(path))
    scale = min(max_w / img.imageWidth, max_h / img.imageHeight)
    img.drawWidth  = img.imageWidth  * scale
    img.drawHeight = img.imageHeight * scale
    return img


def _placeholder(name, w, h):
    st = ParagraphStyle('_ph', fontName='Courier', fontSize=FS_CAPTION,
                        alignment=TA_CENTER, textColor=C_GRAY)
    t  = Table([[Paragraph(name, st)]], colWidths=[w], rowHeights=[h])
    t.setStyle(TableStyle([
        ('BOX',        (0, 0), (-1, -1), 0.5, C_GRAY),
        ('BACKGROUND', (0, 0), (-1, -1), C_VGRAY),
        ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN',      (0, 0), (-1, -1), 'CENTER'),
    ]))
    return t


# ── style factory (identical signature to uncertainty_predictor) ──────────────
def _s(name, size, bold=False, italic=False, color=C_BLACK, align=TA_LEFT, leading=None):
    font = 'Courier-Bold' if bold else ('Courier-Oblique' if italic else 'Courier')
    return ParagraphStyle(name, fontName=font, fontSize=size,
                          leading=leading or size * 1.3,
                          textColor=color, alignment=align)

# header / meta
ST_TITLE    = _s('st_title',   FS_TITLE,   bold=False)
ST_META     = _s('st_meta',    FS_META,    color=C_MUTED)
# KPI
ST_KPI_LBL  = _s('st_kpi_l',  FS_KPI_LBL, color=C_MUTED)
ST_KPI_VAL  = _s('st_kpi_v',  FS_KPI_VAL, bold=True)
ST_KPI_SUB  = _s('st_kpi_s',  FS_KPI_SUB, color=C_MUTED)
# section
ST_SECTION  = _s('st_sec',    FS_SECTION, bold=False, color=C_MUTED)
# body kv
ST_KEY      = _s('st_key',    FS_BODY,    color=C_BLACK, align=TA_LEFT)
ST_VAL      = _s('st_val',    FS_BODY,    color=C_BLACK, align=TA_RIGHT)
ST_VAL_G    = _s('st_val_g',  FS_BODY,    color=C_GREEN, align=TA_RIGHT)
ST_VAL_R    = _s('st_val_r',  FS_BODY,    color=C_RED,   align=TA_RIGHT)
ST_VAL_A    = _s('st_val_a',  FS_BODY,    color=C_AMBER, align=TA_RIGHT)
# badge
ST_BADGE_G  = _s('st_bg_g',  FS_BADGE,   color=C_GREEN, align=TA_CENTER)
ST_BADGE_R  = _s('st_bg_r',  FS_BADGE,   color=C_RED,   align=TA_CENTER)
ST_BADGE_A  = _s('st_bg_a',  FS_BADGE,   color=C_AMBER, align=TA_CENTER)
# caption / footer
ST_CAPTION  = _s('st_cap',   FS_CAPTION,  italic=True)
ST_NOTE     = _s('st_note',  FS_CAPTION,  italic=True,  color=C_MUTED)
ST_FOOTER_L = _s('st_ftrl',  FS_CAPTION,  color=C_MUTED, align=TA_LEFT)
ST_FOOTER_R = _s('st_ftrr',  FS_CAPTION,  color=C_MUTED, align=TA_RIGHT)


def _cv(c, align=TA_RIGHT):
    """Dynamic color + align body style."""
    return ParagraphStyle(f'_dyn{id(c)}{align}', fontName='Courier',
                          fontSize=FS_BODY, leading=FS_BODY * 1.3,
                          textColor=c, alignment=align)


def _ckpi(c):
    """Dynamic color KPI val style."""
    return ParagraphStyle(f'_kpi{id(c)}', fontName='Courier-Bold',
                          fontSize=FS_KPI_VAL, leading=FS_KPI_VAL * 1.2,
                          textColor=c, alignment=TA_LEFT)


def section_header(title, width):
    """Section divider: small-caps label + thin HR below (same as uncertainty_predictor)."""
    return [
        Spacer(1, 5),
        Paragraph(title.upper(), ST_SECTION),
        HRFlowable(width=width, thickness=0.4, color=C_LGRAY, spaceAfter=1),
    ]


def kv_table(rows, col_w, key_frac=0.55):
    """
    Two-column key/value table.  rows = [(key_str, val_str, val_style?), ...]
    val_style defaults to ST_VAL.
    """
    kw = col_w * key_frac
    vw = col_w * (1 - key_frac)
    data = []
    for row in rows:
        key  = row[0]
        val  = row[1]
        vstyle = row[2] if len(row) > 2 else ST_VAL
        data.append([Paragraph(key, ST_KEY), Paragraph(str(val), vstyle)])
    t = Table(data, colWidths=[kw, vw])
    t.setStyle(TableStyle([
        ('LINEBELOW',     (0, 0), (-1, -2), 0.2, C_LGRAY),
        ('TOPPADDING',    (0, 0), (-1, -1), 1.2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.2),
        ('LEFTPADDING',   (0, 0), (-1, -1), 0),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
    ]))
    return t


# ════════════════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════════════════

def _header_flowables(d):
    cfg        = d['config']
    ts         = d.get('timestamp', datetime.now())
    ts_str     = ts.strftime('%Y-%m-%d %H:%M:%S') if isinstance(ts, datetime) else str(ts)
    hist       = d.get('history', {})
    eval_res   = d.get('eval_results', {})
    seed       = cfg.get('data', {}).get('random_seed',
                 cfg.get('training', {}).get('seed', '—'))
    best_epoch = hist.get('best_epoch', '—')
    max_epochs = cfg.get('training', {}).get('max_epochs',
                 cfg.get('training', {}).get('epochs', '—'))

    r2v      = float(eval_res.get('test_r2', 0.0))
    if r2v >= 0.9:
        badge_txt, badge_st, badge_col = "R²≥0.90", ST_BADGE_G, C_GREEN
    elif r2v >= 0.7:
        badge_txt, badge_st, badge_col = "R²≥0.70", ST_BADGE_A, C_AMBER
    else:
        badge_txt, badge_st, badge_col = "R²<0.70", ST_BADGE_R, C_RED

    title_p  = Paragraph("Surrogate Transformer \u2014 Training Report", ST_TITLE)
    badge_w  = 2.5 * cm
    badge_p  = Paragraph(badge_txt, badge_st)

    row1 = Table([[title_p, badge_p]], colWidths=[FULL_W - badge_w, badge_w])
    row1.setStyle(TableStyle([
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 0),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
        ('TOPPADDING',    (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('BOX',           (1, 0), (1, 0),   0.5, badge_col),
        ('ALIGN',         (1, 0), (1, 0),   'CENTER'),
    ]))

    meta_p = Paragraph(
        f"{ts_str}  \u00b7  seed {seed}  \u00b7  "
        f"epochs {best_epoch} / {max_epochs}",
        ST_META)
    rule = HRFlowable(width=FULL_W, thickness=1, color=C_BLACK, spaceAfter=0)
    return [row1, Spacer(1, 2), meta_p, Spacer(1, 2), rule]


# ════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════════════════════════

def _footer_flowables(d):
    cfg  = d['config']
    chk  = short_dir(cfg.get('checkpoints', {}).get('save_dir',
           cfg.get('training', {}).get('checkpoint_dir', '')))
    rule = HRFlowable(width=FULL_W, thickness=1, color=C_BLACK,
                      spaceBefore=2, spaceAfter=2)
    tbl  = Table(
        [[Paragraph(f"auto-generated  \u00b7  {chk}", ST_FOOTER_L),
          Paragraph("surrogate_transformer", ST_FOOTER_R)]],
        colWidths=[FULL_W * 0.75, FULL_W * 0.25],
    )
    tbl.setStyle(TableStyle([
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 0),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
        ('TOPPADDING',    (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('ALIGN',         (1, 0), (1, 0),   'RIGHT'),
    ]))
    return [rule, tbl]


# ════════════════════════════════════════════════════════════════════════════
#  LEFT COLUMN
# ════════════════════════════════════════════════════════════════════════════

def _build_left(d):
    cfg      = d['config']
    history  = d.get('history', {})
    eval_res = d.get('eval_results', {})
    floor    = d.get('floor_metrics') or {}
    n_train  = d.get('n_train', 0)
    n_val    = d.get('n_val',   0)
    n_test   = d.get('n_test',  0)

    model_cfg = cfg.get('model', {})
    tr_cfg    = cfg.get('training', {})
    data_cfg  = cfg.get('data', {})
    chk_cfg   = cfg.get('checkpoints', {})

    F = []

    # ── KPI bar ───────────────────────────────────────────────────────────────
    best_val  = float(history.get('best_val_loss',  history.get('final_val_loss', 0.0)))
    final_val = float(history.get('final_val_loss', best_val))
    r2v       = float(eval_res.get('test_r2',   0.0))
    rmsev     = float(eval_res.get('test_rmse', 0.0))
    floor_mse_raw = floor.get('floor_mse', floor.get('mse_floor', None))
    has_floor = floor_mse_raw is not None and float(floor_mse_raw) > 0
    floor_mse = float(floor_mse_raw) if has_floor else 0.0

    cw = LW / 4

    kpi_lbl = [Paragraph(t, ST_KPI_LBL) for t in
               ["BEST VAL MSE", "TEST R\u00b2", "TEST RMSE", "FLOOR MSE"]]
    kpi_val = [
        Paragraph(f"{best_val:.5f}",  ST_KPI_VAL),
        Paragraph(f"{r2v:.4f}",       _ckpi(r2_color(r2v))),
        Paragraph(f"{rmsev:.5f}",     _ckpi(rmse_color(rmsev))),
        Paragraph(f"{floor_mse:.5f}" if has_floor else "\u2014",
                  _ckpi(floor_color(floor_mse)) if has_floor else _ckpi(C_GRAY)),
    ]
    kpi_sub = [
        Paragraph(f"final {final_val:.5f}",              ST_KPI_SUB),
        Paragraph(f"MAE {eval_res.get('test_mae', 0):.5f}", ST_KPI_SUB),
        Paragraph(f"MSE {eval_res.get('test_mse', 0):.5f}", ST_KPI_SUB),
        Paragraph(f"top-decile F\u2265q\u2089\u2080" if has_floor else "not computed",
                  ST_KPI_SUB),
    ]

    kpi_tbl = Table([kpi_lbl, kpi_val, kpi_sub], colWidths=[cw] * 4)
    kpi_tbl.setStyle(TableStyle([
        ('BOX',           (0, 0), (-1, -1), 0.5, C_BLACK),
        ('INNERGRID',     (0, 0), (-1, -1), 0.5, C_BLACK),
        ('TOPPADDING',    (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING',   (0, 0), (-1, -1), 4),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 4),
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
    ]))
    F.append(kpi_tbl)

    # ── Section 01 — model configuration ──────────────────────────────────────
    F += section_header("01 \u2014 model configuration", LW)

    half = LW / 2
    arch_rows = [
        ("Type",          "TransformerForecaster"),
        ("d_model_enc",   str(model_cfg.get('d_model_enc', '—'))),
        ("d_model_dec",   str(model_cfg.get('d_model_dec', '—'))),
        ("d_ff",          str(model_cfg.get('d_ff', '—'))),
        ("d_qk",          str(model_cfg.get('d_qk', '—'))),
        ("e_layers",      str(model_cfg.get('e_layers', '—'))),
        ("d_layers",      str(model_cfg.get('d_layers', '—'))),
        ("n_heads",       str(model_cfg.get('n_heads', '—'))),
    ]
    reg_rows = [
        ("Activation",    str(model_cfg.get('activation', '—'))),
        ("Norm",          str(model_cfg.get('norm', '—'))),
        ("Final norm",    str(model_cfg.get('use_final_norm', '—'))),
        ("Dropout emb",   str(model_cfg.get('dropout_emb', '—'))),
        ("Dropout attn",  str(model_cfg.get('dropout_attn_out', '—'))),
        ("Dropout ff",    str(model_cfg.get('dropout_ff', '—'))),
        ("Input dim",     str(d.get('input_dim', '—'))),
        ("Total params",  f"{d.get('total_params', 0):,}"),
    ]
    sec01 = Table([[kv_table(arch_rows, half), kv_table(reg_rows, half)]],
                  colWidths=[half, half])
    sec01.setStyle(TableStyle([
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING',    (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('LEFTPADDING',   (0, 0), (-1, -1), 0),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
    ]))
    F.append(sec01)

    # ── Section 02 — data & training parameters ────────────────────────────────
    F += section_header("02 \u2014 data & training parameters", LW)

    total_samples = (n_train + n_val + n_test) or 1
    processes = data_cfg.get('process_names', []) or []
    proc_str  = ', '.join(str(p) for p in processes if p is not None) or '—'

    data_rows = [
        ("Processes",     proc_str),
        ("Trajectories",  str(data_cfg.get('n_trajectories', '—'))),
        ("Scenarios",     str(data_cfg.get('n_scenarios', '—'))),
        ("Train",         f"{n_train:,} ({n_train/total_samples*100:.0f}%)"),
        ("Val",           f"{n_val:,}   ({n_val/total_samples*100:.0f}%)"),
        ("Test",          f"{n_test:,}  ({n_test/total_samples*100:.0f}%)"),
        ("Random seed",   str(data_cfg.get('random_seed', '—'))),
    ]
    train_rows = [
        ("Max epochs",    str(tr_cfg.get('max_epochs', tr_cfg.get('epochs', '—')))),
        ("Batch size",    str(tr_cfg.get('batch_size', '—'))),
        ("Learning rate", fmt_lr(tr_cfg.get('learning_rate', 0))),
        ("Weight decay",  str(tr_cfg.get('weight_decay', '—'))),
        ("Loss fn",       str(tr_cfg.get('loss_fn', 'mse'))),
        ("Patience",      str(tr_cfg.get('patience', '—'))),
        ("Scheduler",     str(tr_cfg.get('use_scheduler', False))),
    ]

    col_w = LW / 2
    sec02 = Table([[kv_table(data_rows, col_w), kv_table(train_rows, col_w)]],
                  colWidths=[col_w, col_w])
    sec02.setStyle(TableStyle([
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING',    (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('LEFTPADDING',   (0, 0), (-1, -1), 0),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
        ('RIGHTPADDING',  (0, 0), (0, -1),  3),
        ('LEFTPADDING',   (1, 0), (1, -1),  3),
    ]))
    F.append(sec02)

    # ── Section 03 — test metrics ─────────────────────────────────────────────
    F += section_header("03 \u2014 test metrics", LW)

    cws3 = [r * LW for r in [0.20, 0.16, 0.16, 0.16, 0.16, 0.16]]
    hdr3 = [Paragraph(h, _s('_h3', FS_BODY, bold=False, color=C_MUTED)) for h in
            ["Metric", "MSE", "RMSE", "MAE", "R\u00b2", "Floor MSE"]]

    mse_  = float(eval_res.get('test_mse',  0.0))
    rmse_ = float(eval_res.get('test_rmse', 0.0))
    mae_  = float(eval_res.get('test_mae',  0.0))
    r2_   = float(eval_res.get('test_r2',   0.0))
    fl_   = float(floor.get('floor_mse', floor.get('mse_floor', 0.0)))
    fl_b  = floor.get('floor_bias', None)

    row3 = [
        Paragraph("Reliability F",          _s('_m0', FS_BODY)),
        Paragraph(f"{mse_:.5f}",            _s('_m1', FS_BODY, align=TA_RIGHT)),
        Paragraph(f"{rmse_:.5f}",           _cv(rmse_color(rmse_))),
        Paragraph(f"{mae_:.5f}",            _s('_m3', FS_BODY, align=TA_RIGHT)),
        Paragraph(f"{r2_:.4f}",             _cv(r2_color(r2_))),
        Paragraph(f"{fl_:.5f}" if fl_ else "\u2014",
                                            _cv(floor_color(fl_) if fl_ else C_GRAY)),
    ]

    mt = Table([hdr3, row3], colWidths=cws3)
    mt.setStyle(TableStyle([
        ('BOX',           (0, 0), (-1, -1), 0.5, C_BLACK),
        ('LINEBELOW',     (0, 0), (-1,  0), 0.5, C_BLACK),
        ('INNERGRID',     (0, 0), (-1, -1), 0.3, C_LGRAY),
        ('TOPPADDING',    (0, 0), (-1, -1), 1.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5),
        ('LEFTPADDING',   (0, 0), (-1, -1), 2),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 2),
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
    ]))
    F.append(mt)

    note_parts = ["Floor MSE = MSE restricted to top-decile reliability scores (F \u2265 q\u2089\u2080)"]
    if fl_b is not None:
        note_parts.append(f"  \u00b7  floor bias {fl_b:+.5f}")
    F.append(Paragraph('  '.join(note_parts), ST_NOTE))

    # ── Section 04 — training results ─────────────────────────────────────────
    F += section_header("04 \u2014 training results", LW)

    final_ep  = history.get('final_epoch',  history.get('best_epoch', '—'))
    best_ep   = history.get('best_epoch',   '—')
    max_ep    = tr_cfg.get('max_epochs', tr_cfg.get('epochs', '—'))
    patience  = tr_cfg.get('patience', None)
    early_stopped = history.get('early_stopped', False)
    best_tr   = history.get('best_train_loss',  None)
    final_tr  = history.get('final_train_loss', history.get('train_loss', [None])[-1]
                            if isinstance(history.get('train_loss'), list) else None)
    final_vl  = history.get('final_val_loss',   best_val)

    # Best epoch label with early-stopping indicator
    if early_stopped:
        best_ep_label = f"{best_ep}  (early stopping, patience {patience})"
        best_ep_style = _cv(C_GREEN, align=TA_RIGHT)
    elif best_ep != '—' and max_ep != '—' and int(best_ep) == int(max_ep):
        best_ep_label = f"{best_ep}  (final epoch)"
        best_ep_style = ST_VAL
    else:
        best_ep_label = str(best_ep)
        best_ep_style = ST_VAL

    # Final epoch label
    if early_stopped:
        final_ep_label = f"{final_ep} / {max_ep}"
    else:
        final_ep_label = str(final_ep)

    res_rows = [
        ("Final epoch",       final_ep_label),
        ("Best epoch",        best_ep_label, best_ep_style),
        ("Final train MSE",   f"{float(final_tr):.6f}" if final_tr is not None else "—"),
        ("Final val MSE",     f"{float(final_vl):.6f}"),
        ("Best val MSE",      f"{best_val:.6f}"),
    ]

    # add MAE history if available
    final_tr_mae = history.get('final_train_mae', None)
    final_vl_mae = history.get('final_val_mae',   None)
    if final_tr_mae is not None:
        res_rows.append(("Final train MAE",  f"{float(final_tr_mae):.6f}"))
    if final_vl_mae is not None:
        res_rows.append(("Final val MAE",    f"{float(final_vl_mae):.6f}"))

    F.append(kv_table(res_rows, LW))

    # ── Section 05 — misc ─────────────────────────────────────────────────────
    F += section_header("05 \u2014 misc", LW)
    seed2    = data_cfg.get('random_seed', tr_cfg.get('seed', '—'))
    verbose  = cfg.get('logging', {}).get('verbose', '—')
    k_fold   = tr_cfg.get('k_fold', 1)
    F.append(Paragraph(
        f"Random seed: {seed2}  \u00b7  k-fold: {k_fold}  \u00b7  verbose: {verbose}",
        _s('_misc', FS_BODY)))

    return F


# ════════════════════════════════════════════════════════════════════════════
#  RIGHT COLUMN  — images
# ════════════════════════════════════════════════════════════════════════════

def _build_right(d):
    chk_dir = Path(d.get('checkpoint_dir',
              d['config'].get('checkpoints', {}).get('save_dir',
              d['config'].get('training', {}).get('checkpoint_dir', '.'))))

    # Expected plot filenames (produced by the surrogate training script)
    PLOTS = [
        ('loss_curves.png',      'Loss Curves (MSE)'),
        ('mae_curves.png',       'MAE Curves'),
        ('pred_vs_target.png',   'Predictions vs True Reliability F'),
        ('lr_schedule.png',      'Learning Rate Schedule'),
    ]

    F = []
    n_slots  = len(PLOTS)
    img_gap  = 3
    cap_h    = 10
    img_h    = (BODY_H - (n_slots - 1) * img_gap - n_slots * cap_h) / n_slots

    for fname, caption in PLOTS:
        img = scale_img(chk_dir / fname, RW, img_h)
        cap = Paragraph(f"<i>{caption}</i>", ST_CAPTION)

        img_tbl = Table([[img]], colWidths=[RW])
        img_tbl.setStyle(TableStyle([
            ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
            ('LEFTPADDING',   (0, 0), (-1, -1), 0),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
            ('TOPPADDING',    (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))
        F.append(img_tbl)
        F.append(cap)
        F.append(Spacer(1, img_gap))

    return F


# ════════════════════════════════════════════════════════════════════════════
#  BUILD PDF
# ════════════════════════════════════════════════════════════════════════════

def _build_pdf(d, out_path):
    hdr_f = Frame(M, PH - M - HDR_H, FULL_W, HDR_H, id='hdr',
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    lft_f = Frame(M, M + FTR_H, LW, BODY_H, id='lft',
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    rgt_f = Frame(M + LW + GAP, M + FTR_H, RW, BODY_H, id='rgt',
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    ftr_f = Frame(M, M, FULL_W, FTR_H, id='ftr',
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)

    pt  = PageTemplate(id='main',
                       frames=[hdr_f, lft_f, rgt_f, ftr_f],
                       pagesize=landscape(A4))
    doc = BaseDocTemplate(str(out_path), pagesize=landscape(A4),
                          leftMargin=M, rightMargin=M, topMargin=M, bottomMargin=M,
                          pageTemplates=[pt])

    lk = KeepInFrame(LW, BODY_H, _build_left(d),  mode='shrink')
    rk = KeepInFrame(RW, BODY_H, _build_right(d), mode='shrink')

    doc.build(
        _header_flowables(d) + [FrameBreak()] +
        [lk]                 + [FrameBreak()] +
        [rk]                 + [FrameBreak()] +
        _footer_flowables(d)
    )


# ════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ════════════════════════════════════════════════════════════════════════════

def generate_surrogate_training_report(
    config,
    history,
    eval_results,
    input_dim,
    output_dim,
    total_params,
    n_train,
    n_val,
    n_test,
    checkpoint_dir,
    timestamp=None,
    floor_metrics=None,
):
    """
    Generate a PDF training report for the surrogate transformer.

    Parameters
    ----------
    config : dict
        Full SURROGATE_CONFIG dictionary (or equivalent structure with
        'model', 'training', 'data', 'checkpoints', 'logging' keys).
    history : dict
        Training history with keys such as:
          best_epoch, best_val_loss, final_epoch, final_val_loss,
          final_train_loss, final_train_mae, final_val_mae,
          train_loss (list), val_loss (list), train_mae (list), val_mae (list),
          learning_rate (list).
    eval_results : dict
        Test-set evaluation results:
          test_mse, test_mae, test_rmse, test_r2,
          predictions (ndarray), targets (ndarray).
    input_dim : int
        Number of input features per sequence step.
    output_dim : int
        Output dimension (typically 1 for scalar reliability F).
    total_params : int
        Total number of trainable model parameters.
    n_train, n_val, n_test : int
        Dataset split sizes.
    checkpoint_dir : str | Path
        Directory where the report (and plots) are saved.
    timestamp : datetime, optional
        Training timestamp; uses now() if not provided.
    floor_metrics : dict, optional
        Surrogate-specific reliability floor fidelity metrics:
          floor_mse  — MSE restricted to top-decile F scores,
          floor_bias — mean bias in that region.

    Returns
    -------
    str
        Absolute path to the generated PDF file.
    """
    if timestamp is None:
        timestamp = datetime.now()

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    out_path = checkpoint_dir / 'surrogate_training_report.pdf'

    d = dict(
        config=config,
        history=history,
        eval_results=eval_results,
        floor_metrics=floor_metrics or {},
        input_dim=input_dim,
        output_dim=output_dim,
        total_params=total_params,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        timestamp=timestamp,
        checkpoint_dir=checkpoint_dir,
    )

    _build_pdf(d, out_path)
    print(f"Surrogate report generated: {out_path}")
    return str(out_path)


# ════════════════════════════════════════════════════════════════════════════
#  CLASS-BASED ALIAS  (mirrors UncertaintyReportGenerator)
# ════════════════════════════════════════════════════════════════════════════

class SurrogateReportGenerator:
    """
    Thin class wrapper — kept for symmetry with UncertaintyReportGenerator.
    Prefer the functional API (generate_surrogate_training_report) in new code.
    """

    def generate(
        self,
        config,
        history,
        eval_results,
        input_dim,
        output_dim,
        total_params,
        n_train,
        n_val,
        n_test,
        checkpoint_dir,
        timestamp=None,
        floor_metrics=None,
    ):
        return generate_surrogate_training_report(
            config=config,
            history=history,
            eval_results=eval_results,
            input_dim=input_dim,
            output_dim=output_dim,
            total_params=total_params,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            checkpoint_dir=checkpoint_dir,
            timestamp=timestamp,
            floor_metrics=floor_metrics,
        )


# ════════════════════════════════════════════════════════════════════════════
#  QUICK SMOKE-TEST  (python surrogate_report_generator.py)
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import numpy as np
    from datetime import datetime

    # ── minimal fake config (matches SURROGATE_CONFIG structure) ──────────────
    cfg = {
        'data': {
            'n_trajectories': 1000,
            'n_val_trajectories': 200,
            'n_test_trajectories': 200,
            'n_scenarios': 4,
            'process_names': ['laser', 'plasma', 'galvanic', 'microetch'],
            'random_seed': 42,
        },
        'model': {
            'd_model_enc': 64,
            'd_model_dec': 32,
            'd_ff': 128,
            'd_qk': 32,
            'e_layers': 2,
            'd_layers': 2,
            'n_heads': 4,
            'dropout_emb': 0.1,
            'dropout_attn_out': 0.0,
            'dropout_ff': 0.1,
            'activation': 'gelu',
            'norm': 'batch',
            'use_final_norm': True,
        },
        'training': {
            'max_epochs': 200,
            'batch_size': 64,
            'learning_rate': 1e-3,
            'weight_decay': 0.01,
            'loss_fn': 'mse',
            'patience': 20,
            'use_scheduler': True,
            'scheduler_factor': 0.5,
            'scheduler_patience': 10,
            'seed': 42,
        },
        'checkpoints': {
            'save_dir': '/tmp/surrogate_demo',
        },
        'logging': {
            'verbose': True,
        },
    }

    hist = {
        'best_epoch':        87,
        'best_val_loss':     0.003142,
        'final_epoch':       107,
        'final_val_loss':    0.003850,
        'final_train_loss':  0.002710,
        'final_train_mae':   0.039800,
        'final_val_mae':     0.044200,
        'train_loss':        list(np.linspace(0.08, 0.0027, 108)),
        'val_loss':          list(np.linspace(0.09, 0.0039, 108)),
        'train_mae':         list(np.linspace(0.20, 0.0398, 108)),
        'val_mae':           list(np.linspace(0.22, 0.0442, 108)),
        'learning_rate':     [1e-3 * (0.5 ** (i // 30)) for i in range(108)],
        'early_stopped':     True,
    }

    eval_res = {
        'test_mse':  0.003219,
        'test_mae':  0.041500,
        'test_rmse': 0.056740,
        'test_r2':   0.9413,
        'predictions': np.random.uniform(0, 1, 200),
        'targets':     np.random.uniform(0, 1, 200),
    }

    floor = {
        'floor_mse':  0.004831,
        'floor_bias': -0.008210,
    }

    out = generate_surrogate_training_report(
        config=cfg,
        history=hist,
        eval_results=eval_res,
        input_dim=20,
        output_dim=1,
        total_params=48_320,
        n_train=1000,
        n_val=200,
        n_test=200,
        checkpoint_dir='/tmp/surrogate_demo',
        timestamp=datetime(2025, 6, 15, 14, 32, 7),
        floor_metrics=floor,
    )
    print(f"\nDone → {out}")