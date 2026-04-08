"""
PDF Report Generator for Uncertainty Quantification Training
A4 Landscape, single page per process. Layout mirrors the HTML reference design.
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
    NextPageTemplate, PageBreak,
    Paragraph, Spacer, Image, Table, TableStyle,
)
from reportlab.platypus.flowables import HRFlowable, KeepInFrame

try:
    from pypdf import PdfReader, PdfWriter, Transformation
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

# ── page geometry ─────────────────────────────────────────────────────────────
PW, PH  = landscape(A4)
M       = 0.8 * cm
HDR_H   = 1.6 * cm
FTR_H   = 0.65 * cm
LW      = 108 * mm
GAP     = 6 * mm
FULL_W  = PW - 2 * M
RW      = FULL_W - LW - GAP
BODY_H  = PH - 2 * M - HDR_H - FTR_H - 2 * mm

# ── colors ────────────────────────────────────────────────────────────────────
C_GREEN = colors.HexColor('#1D9E75')
C_RED   = colors.HexColor('#D85A30')
C_AMBER = colors.HexColor('#BA7517')
C_BLACK = colors.black
C_GRAY  = colors.HexColor('#AAAAAA')
C_LGRAY = colors.HexColor('#DDDDDD')
C_VGRAY = colors.HexColor('#F8F8F8')
C_MUTED = colors.HexColor('#666666')

# ── font sizes — match HTML reference exactly ─────────────────────────────────
FS_TITLE   = 15    # "Uncertainty Quantification — Training Report"
FS_META    = 7.5
FS_SECTION = 7     # "01 — MODEL CONFIGURATION" uppercase
FS_BODY    = 7.5   # all kv rows, table cells
FS_KPI_LBL = 6.5   # "BEST VAL NLL" small caps label
FS_KPI_VAL = 11    # "-0.5476" medium-bold value
FS_KPI_SUB = 6.5   # "final -0.5024" sub line
FS_CAPTION = 6.5
FS_BADGE   = 6.5


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════

def r2_color(v):
    return C_GREEN if v >= 0.9 else (C_AMBER if v >= 0.7 else C_RED)

def cal_color(v):
    return C_GREEN if 0.8 <= v <= 1.2 else (C_AMBER if 0.5 <= v <= 1.8 else C_RED)

def cov_color(err):
    return C_GREEN if abs(err) <= 5 else (C_AMBER if abs(err) <= 10 else C_RED)


def fmt_lr(lr):
    """'1.000 x 10^-3'  — plain ASCII, no Unicode superscripts (black-box glyphs)."""
    if lr == 0:
        return "0"
    exp  = int(math.floor(math.log10(abs(lr))))
    mant = lr / (10 ** exp)
    return f"{mant:.3f} x 10^{exp}"


def short_dir(d, max_len=35):
    parts = Path(d).parts
    short = str(Path(*parts[-2:])) if len(parts) >= 2 else str(d)
    return short if len(short) <= max_len else '...' + short[-(max_len - 3):]


def fmt_cols(cols, max_show=5):
    if len(cols) <= max_show:
        return ', '.join(cols)
    first = cols[0]
    last5 = cols[max_show - 1]
    extra = len(cols) - max_show
    return f"{first}..{last5}, {cols[max_show]} (+{extra} more)" if extra > 0 else ', '.join(cols)


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
        ('BOX',        (0,0),(-1,-1), 0.5, C_GRAY),
        ('BACKGROUND', (0,0),(-1,-1), C_VGRAY),
        ('VALIGN',     (0,0),(-1,-1), 'MIDDLE'),
        ('ALIGN',      (0,0),(-1,-1), 'CENTER'),
    ]))
    return t


# ── style factory ────────────────────────────────────────────────────────────
def _s(name, size, bold=False, italic=False, color=C_BLACK, align=TA_LEFT, leading=None):
    font = 'Courier-Bold' if bold else ('Courier-Oblique' if italic else 'Courier')
    return ParagraphStyle(name, fontName=font, fontSize=size,
                          leading=leading or size * 1.3,
                          textColor=color, alignment=align)

# header
ST_TITLE    = _s('st_title',   FS_TITLE,   bold=False)
ST_META     = _s('st_meta',    FS_META,    color=C_MUTED)
# KPI
ST_KPI_LBL  = _s('st_kpi_l',  FS_KPI_LBL, color=C_MUTED)          # small label row
ST_KPI_VAL  = _s('st_kpi_v',  FS_KPI_VAL, bold=True)               # big value row
ST_KPI_SUB  = _s('st_kpi_s',  FS_KPI_SUB, color=C_MUTED)           # sub line
# section header
ST_SECTION  = _s('st_sec',    FS_SECTION, bold=False, color=C_MUTED)
# body — key LEFT, value RIGHT (matches HTML table columns)
ST_KEY      = _s('st_key',    FS_BODY,    color=C_BLACK, align=TA_LEFT)
ST_VAL      = _s('st_val',    FS_BODY,    color=C_BLACK, align=TA_RIGHT)
ST_VAL_G    = _s('st_val_g',  FS_BODY,    color=C_GREEN, align=TA_RIGHT)
ST_VAL_R    = _s('st_val_r',  FS_BODY,    color=C_RED,   align=TA_RIGHT)
ST_VAL_A    = _s('st_val_a',  FS_BODY,    color=C_AMBER, align=TA_RIGHT)
# badge
ST_BADGE_G  = _s('st_bg_g',  FS_BADGE,   color=C_GREEN, align=TA_CENTER)
ST_BADGE_R  = _s('st_bg_r',  FS_BADGE,   color=C_RED,   align=TA_CENTER)
# caption / footer / note
ST_CAPTION  = _s('st_cap',   FS_CAPTION,  italic=True)
ST_NOTE     = _s('st_note',  FS_CAPTION,  italic=True, color=C_MUTED)
ST_FOOTER_L = _s('st_ftrl',  FS_CAPTION,  color=C_MUTED, align=TA_LEFT)
ST_FOOTER_R = _s('st_ftrr',  FS_CAPTION,  color=C_MUTED, align=TA_RIGHT)
# section-03 labels (left-aligned, no right-align)
ST_SEC3_LBL = _s('st_s3l',   FS_KPI_LBL,  color=C_MUTED, align=TA_LEFT)
ST_SEC3_VAL = _s('st_s3v',   FS_BODY,     color=C_BLACK, align=TA_LEFT)


def _cv(c, align=TA_RIGHT):
    """Dynamic color+align body style."""
    return ParagraphStyle(f'_dyn{id(c)}{align}', fontName='Courier',
                          fontSize=FS_BODY, leading=FS_BODY*1.3,
                          textColor=c, alignment=align)

def _ckpi(c):
    """Dynamic color KPI val style."""
    return ParagraphStyle(f'_kpi{id(c)}', fontName='Courier-Bold',
                          fontSize=FS_KPI_VAL, leading=FS_KPI_VAL*1.2,
                          textColor=c, alignment=TA_LEFT)


def section_header(title, width):
    """Section divider: small-caps label + thin HR below."""
    return [
        Spacer(1, 5),
        Paragraph(title.upper(), ST_SECTION),
        HRFlowable(width=width, thickness=0.4, color=C_LGRAY, spaceAfter=1),
    ]


def kv_table(rows, col_w, key_frac=0.45):
    """
    key LEFT-aligned, value RIGHT-aligned — exact match to HTML reference.
    rows = list of (key_str, val_para_or_str [, val_style])
    """
    KEY_W = col_w * key_frac
    VAL_W = col_w * (1 - key_frac)
    data  = []
    for row in rows:
        key, val = row[0], row[1]
        vs       = row[2] if len(row) > 2 else ST_VAL
        k = Paragraph(key, ST_KEY) if isinstance(key, str) else key
        v = Paragraph(val, vs)     if isinstance(val, str) else val
        data.append([k, v])
    t = Table(data, colWidths=[KEY_W, VAL_W])
    t.setStyle(TableStyle([
        ('VALIGN',        (0,0),(-1,-1), 'TOP'),
        ('TOPPADDING',    (0,0),(-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0),(-1,-1), 1.5),
        ('LEFTPADDING',   (0,0),(-1,-1), 0),
        ('RIGHTPADDING',  (0,0),(-1,-1), 0),
        ('LINEBELOW',     (0,0),(-1,-2), 0.3, C_LGRAY),
    ]))
    return t


# ════════════════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════════════════

def _header_flowables(d):
    cfg        = d['config']
    cov        = d.get('coverage_results') or {}
    well_cal   = cov.get('well_calibrated', False)
    ts         = d.get('timestamp', datetime.now())
    ts_str     = ts.strftime('%Y-%m-%d %H:%M:%S') if isinstance(ts, datetime) else str(ts)
    tr_cfg     = cfg.get('training', {})
    epochs     = tr_cfg.get('epochs', '\u2014')
    max_epochs = tr_cfg.get('max_epochs', tr_cfg.get('epochs', '\u2014'))
    patience   = tr_cfg.get('patience', '\u2014')
    rand_state = cfg.get('data',{}).get('random_state',
                 cfg.get('misc',{}).get('random_seed', '\u2014'))

    title_p   = Paragraph("Uncertainty Quantification \u2014 Training Report", ST_TITLE)
    badge_txt = "calibrated" if well_cal else "not calibrated"
    badge_st  = ST_BADGE_G   if well_cal else ST_BADGE_R
    badge_col = C_GREEN      if well_cal else C_RED
    badge_p   = Paragraph(badge_txt, badge_st)
    badge_w   = 2.5 * cm

    row1 = Table([[title_p, badge_p]], colWidths=[FULL_W - badge_w, badge_w])
    row1.setStyle(TableStyle([
        ('VALIGN',        (0,0),(-1,-1), 'MIDDLE'),
        ('LEFTPADDING',   (0,0),(-1,-1), 0),
        ('RIGHTPADDING',  (0,0),(-1,-1), 0),
        ('TOPPADDING',    (0,0),(-1,-1), 0),
        ('BOTTOMPADDING', (0,0),(-1,-1), 0),
        ('BOX',           (1,0),(1,0),   0.5, badge_col),
        ('ALIGN',         (1,0),(1,0),   'CENTER'),
    ]))

    meta_p = Paragraph(
        f"{ts_str}  \u00b7  seed {rand_state}  \u00b7  "
        f"epochs {epochs} / {max_epochs}  \u00b7  patience {patience}",
        ST_META)
    rule   = HRFlowable(width=FULL_W, thickness=1, color=C_BLACK, spaceAfter=0)
    return [row1, Spacer(1, 2), meta_p, Spacer(1, 2), rule]


# ════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════════════════════════

def _footer_flowables(d):
    cfg   = d['config']
    chk   = short_dir(cfg.get('training',{}).get('checkpoint_dir',''))
    rule  = HRFlowable(width=FULL_W, thickness=1, color=C_BLACK,
                       spaceBefore=2, spaceAfter=2)
    tbl   = Table(
        [[Paragraph(f"auto-generated  \u00b7  {chk}", ST_FOOTER_L),
          Paragraph("uncertainty_predictor", ST_FOOTER_R)]],
        colWidths=[FULL_W * 0.75, FULL_W * 0.25],
    )
    tbl.setStyle(TableStyle([
        ('VALIGN',        (0,0),(-1,-1), 'MIDDLE'),
        ('LEFTPADDING',   (0,0),(-1,-1), 0),
        ('RIGHTPADDING',  (0,0),(-1,-1), 0),
        ('TOPPADDING',    (0,0),(-1,-1), 0),
        ('BOTTOMPADDING', (0,0),(-1,-1), 0),
    ]))
    return [rule, tbl]


# ════════════════════════════════════════════════════════════════════════════
#  LEFT COLUMN
# ════════════════════════════════════════════════════════════════════════════

def _build_left(d):
    cfg      = d['config']
    history  = d.get('history', {})
    metrics  = d.get('metrics', {})
    cov      = d.get('coverage_results') or {}
    n_train  = d.get('n_train', 0)
    n_val    = d.get('n_val',   0)
    n_test   = d.get('n_test',  0)
    total    = n_train + n_val + n_test or 1

    model_cfg = cfg.get('model', {})
    tr_cfg    = cfg.get('training', {})
    data_cfg  = cfg.get('data', {})
    unc_cfg   = cfg.get('uncertainty', {})
    misc_cfg  = cfg.get('misc', {})

    F = []  # flowables

    # ── KPI bar ───────────────────────────────────────────────────────────────
    best_nll = history.get('best_val_loss',  history.get('best_val_nll',  0.0))
    val_nll  = history.get('final_val_loss', history.get('final_val_nll', 0.0))
    first    = next((m for k,m in metrics.items() if k != 'Overall'), {})
    r2v      = first.get('R2',  0.0)
    msev     = first.get('MSE', 0.0)
    act_cov  = cov.get('actual_coverage',   0.0)
    exp_cov  = cov.get('expected_coverage', 0.0)
    cov_err  = cov.get('coverage_error',    0.0)
    well_cal = cov.get('well_calibrated',   False)
    calr     = first.get('Calibration_Ratio',
               first.get('Cal_Ratio',
               first.get('calibration_ratio', 1.0)))

    cw = LW / 4

    # KPI: 3 rows — label / big value / sub
    kpi_lbl = [Paragraph(t, ST_KPI_LBL) for t in
               ["BEST VAL NLL", "TEST R\u00b2", "COVERAGE", "CAL. RATIO"]]
    kpi_val = [
        Paragraph(f"{best_nll:.4f}", ST_KPI_VAL),
        Paragraph(f"{r2v:.4f}",      _ckpi(r2_color(r2v))),
        Paragraph(f"{act_cov:.1f}%", _ckpi(cov_color(cov_err))),
        Paragraph(f"{calr:.4f}",     _ckpi(cal_color(calr))),
    ]
    cal_lbl = "calibrated" if well_cal else "NOT calibrated"
    kpi_sub = [
        Paragraph(f"final {val_nll:.4f}",   ST_KPI_SUB),
        Paragraph(f"MSE {msev:.5f}",        ST_KPI_SUB),
        Paragraph(f"expected {exp_cov:.1f}%", ST_KPI_SUB),
        Paragraph(cal_lbl, _s('_calsub', FS_KPI_SUB,
                              color=C_GREEN if well_cal else C_RED)),
    ]

    kpi_tbl = Table([kpi_lbl, kpi_val, kpi_sub], colWidths=[cw]*4)
    kpi_tbl.setStyle(TableStyle([
        ('BOX',           (0,0),(-1,-1), 0.5, C_BLACK),
        ('INNERGRID',     (0,0),(-1,-1), 0.5, C_BLACK),
        ('TOPPADDING',    (0,0),(-1,-1), 3),
        ('BOTTOMPADDING', (0,0),(-1,-1), 3),
        ('LEFTPADDING',   (0,0),(-1,-1), 4),
        ('RIGHTPADDING',  (0,0),(-1,-1), 4),
        ('VALIGN',        (0,0),(-1,-1), 'TOP'),
    ]))
    F.append(kpi_tbl)

    # ── Section 01 — model configuration ──────────────────────────────────────
    F += section_header("01 \u2014 model configuration", LW)

    half       = LW / 2
    model_type = model_cfg.get('model_type', 'UncertaintyPredictor')
    use_ens    = model_cfg.get('use_ensemble', False)
    use_swag   = model_cfg.get('use_swag', False)

    arch = [("Type",         model_type)]
    if use_ens:
        n_ens = model_cfg.get('n_ensemble_models', '?')
        arch.append(("Models in ensemble", str(n_ens)))
        arch.append(("Params per model",   f"{d.get('total_params',0)//(n_ens or 1):,}"))
    if use_swag:
        rank = model_cfg.get('swag_max_rank', None)
        if rank is not None:
            arch.append(("SWAG rank", str(rank)))
    arch += [
        ("Hidden layers", str(model_cfg.get('hidden_sizes', '\u2014'))),
        ("Dropout rate",  str(model_cfg.get('dropout_rate', '\u2014'))),
        ("Batch norm",    str(model_cfg.get('use_batchnorm', '\u2014'))),
        ("Min variance",  str(model_cfg.get('min_variance', '\u2014'))),
        ("Input dim",     str(d.get('input_dim', '\u2014'))),
        ("Output dim",    str(d.get('output_dim', '\u2014'))),
        ("Parameters",    f"{d.get('total_params', 0):,}"),
    ]

    inp_cols  = data_cfg.get('input_columns', [])
    out_cols  = data_cfg.get('output_columns', [])
    tr_pct    = data_cfg.get('train_size', n_train/total) * 100
    val_pct   = data_cfg.get('val_size',   n_val  /total) * 100
    tst_pct   = data_cfg.get('test_size',  n_test /total) * 100

    dset = [
        ("File",         str(data_cfg.get('csv_path', '\u2014'))),
        ("Input cols",   fmt_cols(inp_cols)),
        ("Output cols",  ', '.join(out_cols) if out_cols else '\u2014'),
        ("Train",        f"{n_train:,} ({tr_pct:.0f}%)"),
        ("Validation",   f"{n_val:,} ({val_pct:.0f}%)"),
        ("Test",         f"{n_test:,} ({tst_pct:.0f}%)"),
        ("Scaling",      str(data_cfg.get('scaling_method', '\u2014'))),
        ("Random state", str(data_cfg.get('random_state',
                             misc_cfg.get('random_seed', '\u2014')))),
    ]

    COL_GAP = 5 * mm
    col_w   = (LW - COL_GAP) / 2
    sec01 = Table([[kv_table(arch, col_w), kv_table(dset, col_w)]],
                  colWidths=[col_w, col_w])
    sec01.setStyle(TableStyle([
        ('VALIGN',        (0,0),(-1,-1), 'TOP'),
        ('TOPPADDING',    (0,0),(-1,-1), 0),
        ('BOTTOMPADDING', (0,0),(-1,-1), 0),
        ('LEFTPADDING',   (0,0),(-1,-1), 0),
        ('RIGHTPADDING',  (0,0),(-1,-1), 0),
        ('RIGHTPADDING',  (0,0),(0,-1),  COL_GAP / 2),
        ('LEFTPADDING',   (1,0),(1,-1),  COL_GAP / 2),
    ]))
    F.append(sec01)

    # ── Section 02 — training & results ───────────────────────────────────────
    F += section_header("02 \u2014 training & results", LW)

    lr_str  = fmt_lr(tr_cfg.get('learning_rate', 0))
    chk_str = short_dir(tr_cfg.get('checkpoint_dir', ''))

    _dash = "\u2014"
    _ep = tr_cfg.get('epochs', _dash)
    _max_ep = tr_cfg.get('max_epochs', _ep)
    train_rows = [
        ("Epochs",           f"{_ep} / {_max_ep}"),
        ("Batch size",       str(tr_cfg.get('batch_size', '\u2014'))),
        ("Learning rate",    lr_str),
        ("Weight decay",     str(tr_cfg.get('weight_decay', '\u2014'))),
        ("Loss",             str(tr_cfg.get('loss_type', '\u2014'))),
        ("Var. penalty \u03b1", str(tr_cfg.get('variance_penalty_alpha',
                                               tr_cfg.get('alpha', '\u2014')))),
        ("Patience",         str(tr_cfg.get('patience', '\u2014'))),
    ]
    swa = history.get('swa_start_epoch', None)
    if swa is not None:
        train_rows.append(("SWA start epoch", str(swa)))
    train_rows.append(("Device", str(tr_cfg.get('device', '\u2014'))))

    ftr_nll  = history.get('final_train_loss', history.get('final_train_nll', 0.0))
    fval_nll = history.get('final_val_loss',   history.get('final_val_nll',   0.0))
    bval_nll = history.get('best_val_loss',    history.get('best_val_nll',    0.0))
    ftr_mse  = history.get('final_train_mse',  0.0)
    fval_mse = history.get('final_val_mse',    0.0)
    act2     = cov.get('actual_coverage',   0.0)
    exp2     = cov.get('expected_coverage', 0.0)
    err2     = cov.get('coverage_error',    0.0)
    wc2      = cov.get('well_calibrated',   False)

    results_rows = [
        ("Train NLL",     f"{ftr_nll:.6f}"),
        ("Val NLL",       f"{fval_nll:.6f}"),
        ("Best val NLL",  f"{bval_nll:.6f}", ST_VAL_G),
        ("Train MSE",     f"{ftr_mse:.6f}"),
        ("Val MSE",       f"{fval_mse:.6f}"),
        ("Expected cov.", f"{exp2:.1f}%"),
        ("Actual cov.",   f"{act2:.1f}%",   _cv(cov_color(err2))),
        ("Cov. error",    f"{err2:+.1f}%",  _cv(cov_color(err2))),
        ("Calibrated",    "Yes" if wc2 else "No",
                          ST_VAL_G if wc2 else ST_VAL_R),
    ]

    sec02 = Table([[kv_table(train_rows, col_w), kv_table(results_rows, col_w)]],
                  colWidths=[col_w, col_w])
    sec02.setStyle(TableStyle([
        ('VALIGN',        (0,0),(-1,-1), 'TOP'),
        ('TOPPADDING',    (0,0),(-1,-1), 0),
        ('BOTTOMPADDING', (0,0),(-1,-1), 0),
        ('LEFTPADDING',   (0,0),(-1,-1), 0),
        ('RIGHTPADDING',  (0,0),(-1,-1), 0),
        ('RIGHTPADDING',  (0,0),(0,-1),  COL_GAP / 2),
        ('LEFTPADDING',   (1,0),(1,-1),  COL_GAP / 2),
    ]))
    F.append(sec02)

    # ── Section 03 — uncertainty decomposition ────────────────────────────────
    if 'mean_aleatoric' in cov:
        F += section_header("03 \u2014 uncertainty decomposition", LW)
        alea  = cov.get('mean_aleatoric',  0.0)
        epist = cov.get('mean_epistemic',  0.0)
        ratio = cov.get('epistemic_ratio', 0.0)
        rst   = ST_SEC3_VAL if ratio <= 50 else _s('_rat_a', FS_BODY, color=C_AMBER)
        cw3   = LW / 3
        unc   = Table(
            [[Paragraph("Mean aleatoric", ST_SEC3_LBL),
              Paragraph("Mean epistemic",  ST_SEC3_LBL),
              Paragraph("Epistemic ratio", ST_SEC3_LBL)],
             [Paragraph(f"{alea:.6f}",   ST_SEC3_VAL),
              Paragraph(f"{epist:.6f}",  ST_SEC3_VAL),
              Paragraph(f"{ratio:.1f}%", rst)]],
            colWidths=[cw3]*3,
        )
        unc.setStyle(TableStyle([
            ('LINEBELOW',     (0,0),(-1,0),  0.3, C_LGRAY),
            ('TOPPADDING',    (0,0),(-1,-1), 2),
            ('BOTTOMPADDING', (0,0),(-1,-1), 2),
            ('LEFTPADDING',   (0,0),(-1,-1), 0),
            ('RIGHTPADDING',  (0,0),(-1,-1), 0),
            ('VALIGN',        (0,0),(-1,-1), 'TOP'),
        ]))
        F.append(unc)

    # ── Section 04 — test metrics ─────────────────────────────────────────────
    F += section_header("04 \u2014 test metrics (per output)", LW)

    cws4   = [r * LW for r in [0.16, 0.12, 0.12, 0.11, 0.12, 0.12, 0.13, 0.12]]
    hdr4   = [Paragraph(h, _s('_h4', FS_BODY, bold=False, color=C_MUTED)) for h in
              ["Output","MSE","RMSE","MAE","R\u00b2","Mean var","Cal. ratio","NLL"]]
    rows4  = [hdr4]

    for oname, om in metrics.items():
        if oname == 'Overall':
            continue
        r2_  = om.get('R2',  0.0)
        cal_ = om.get('Calibration_Ratio', om.get('Cal_Ratio',
               om.get('calibration_ratio', 1.0)))
        nll_ = om.get('NLL', None)
        rows4.append([
            Paragraph(str(oname),                                    _s('_tn',FS_BODY)),
            Paragraph(f"{om.get('MSE',  0):.4f}",                   _s('_tm',FS_BODY,align=TA_RIGHT)),
            Paragraph(f"{om.get('RMSE', 0):.4f}",                   _s('_tr',FS_BODY,align=TA_RIGHT)),
            Paragraph(f"{om.get('MAE',  0):.4f}",                   _s('_ta',FS_BODY,align=TA_RIGHT)),
            Paragraph(f"{r2_:.4f}",   _cv(r2_color(r2_))),
            Paragraph(f"{om.get('Mean_Variance', om.get('Mean_Var',0)):.4f}", _s('_tv',FS_BODY,align=TA_RIGHT)),
            Paragraph(f"{cal_:.4f}",  _cv(cal_color(cal_))),
            Paragraph(f"{nll_:.4f}" if nll_ is not None else "\u2014", _s('_tnll',FS_BODY,align=TA_RIGHT)),
        ])

    mt = Table(rows4, colWidths=cws4)
    mt.setStyle(TableStyle([
        ('BOX',           (0,0),(-1,-1), 0.5, C_BLACK),
        ('LINEBELOW',     (0,0),(-1, 0), 0.5, C_BLACK),
        ('INNERGRID',     (0,0),(-1,-1), 0.3, C_LGRAY),
        ('TOPPADDING',    (0,0),(-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0),(-1,-1), 1.5),
        ('LEFTPADDING',   (0,0),(-1,-1), 2),
        ('RIGHTPADDING',  (0,0),(-1,-1), 2),
        ('VALIGN',        (0,0),(-1,-1), 'TOP'),
    ]))
    F.append(mt)
    F.append(Paragraph(
        "Cal. ratio ~1.0 = well calibrated  \u00b7  <1.0 under-confident  \u00b7  >1.0 over-confident",
        ST_NOTE))

    # ── Section 05 — uncertainty & misc ───────────────────────────────────────
    F += section_header("05 \u2014 uncertainty & misc", LW)
    conf  = unc_cfg.get('confidence_level', 0.95)
    rseed = misc_cfg.get('random_seed', data_cfg.get('random_state', '\u2014'))
    F.append(Paragraph(
        f"Confidence level: {conf*100:.0f}%  \u00b7  Random seed: {rseed}",
        _s('_misc', FS_BODY)))

    return F


# ════════════════════════════════════════════════════════════════════════════
#  RIGHT COLUMN
# ════════════════════════════════════════════════════════════════════════════

def _build_right(d):
    chk_dir = Path(d.get('checkpoint_dir') or
                   d['config'].get('training', {}).get('checkpoint_dir', '.'))
    F = []

    avail = BODY_H - 80
    h_ab = avail * 0.45          # section A può usare più altezza ora che B/C non ci sono

    # ── Section A ──────────────────────────────────────────────────────────
    F += section_header("A \u2014 training history", RW)
    a_img = scale_img(chk_dir / 'training_history.png', RW, h_ab)
    a_img.hAlign = 'LEFT'
    F.append(a_img)
    F.append(Paragraph("Training and Validation Loss (NLL and MSE)", ST_CAPTION))

    return F


def _build_section_b(d, frame_w, frame_h):
    """Contenuto minipage B — predictions with uncertainty."""
    chk_dir = Path(d.get('checkpoint_dir') or
                   d['config'].get('training', {}).get('checkpoint_dir', '.'))
    HDR_OVERHEAD = 18   # section_header (spacer + label + rule)
    CAP_H        = 10   # caption height
    img_h = frame_h - HDR_OVERHEAD - CAP_H - 4

    F = section_header("B \u2014 predictions with uncertainty", frame_w)
    p_combined = chk_dir / 'predictions_combined.png'
    if p_combined.exists():
        img = scale_img(p_combined, frame_w, img_h)
        img.hAlign = 'LEFT'
        F.append(img)
    else:
        F.append(_placeholder('predictions_combined.png', frame_w, img_h))
    F.append(Paragraph("Training and Validation Predictions with Uncertainty Bounds",
                        ST_CAPTION))
    return F


def _build_section_c(d, frame_w, frame_h):
    """Contenuto minipage C — scatter with uncertainty coloring."""
    chk_dir = Path(d.get('checkpoint_dir') or
                   d['config'].get('training', {}).get('checkpoint_dir', '.'))
    HDR_OVERHEAD = 18
    CAP_H        = 10
    img_h = frame_h - HDR_OVERHEAD - CAP_H - 4

    F = section_header("C \u2014 scatter with uncertainty coloring", frame_w)
    img = scale_img(chk_dir / 'scatter_with_uncertainty.png', frame_w, img_h)
    img.hAlign = 'LEFT'
    F.append(img)
    F.append(Paragraph("Scatter Plot with Uncertainty Coloring", ST_CAPTION))
    return F


# ════════════════════════════════════════════════════════════════════════════
#  PAGE BUILDER
# ════════════════════════════════════════════════════════════════════════════

def _build_page(d, out_path):
    # ── Pagina 1: frames originali ────────────────────────────────────────
    hdr_f = Frame(M, PH-M-HDR_H, FULL_W, HDR_H, id='hdr',
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    lft_f = Frame(M, M+FTR_H, LW, BODY_H, id='lft',
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    rgt_f = Frame(M+LW+GAP, M+FTR_H, RW, BODY_H, id='rgt',
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    ftr_f = Frame(M, M, FULL_W, FTR_H, id='ftr',
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)

    pt_main = PageTemplate(id='main',
                           frames=[hdr_f, lft_f, rgt_f, ftr_f],
                           pagesize=landscape(A4))

    # ── Pagina 2: due minipage affiancate (left = B, right = C) ─────────
    MINI_GAP = GAP                              # spazio tra le due minipage
    MINI_W   = (FULL_W - MINI_GAP) / 2          # larghezza di ciascuna minipage
    MINI_H   = PH - 2 * M                       # altezza piena

    lft_f2 = Frame(M, M, MINI_W, MINI_H, id='chart_left',
                   leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    rgt_f2 = Frame(M + MINI_W + MINI_GAP, M, MINI_W, MINI_H, id='chart_right',
                   leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)

    pt_charts = PageTemplate(id='charts',
                             frames=[lft_f2, rgt_f2],
                             pagesize=landscape(A4))

    doc = BaseDocTemplate(str(out_path), pagesize=landscape(A4),
                          leftMargin=M, rightMargin=M, topMargin=M, bottomMargin=M,
                          pageTemplates=[pt_main, pt_charts])

    lk = KeepInFrame(LW, BODY_H, _build_left(d),  mode='shrink')
    rk = KeepInFrame(RW, BODY_H, _build_right(d), mode='shrink')

    b_content = _build_section_b(d, MINI_W, MINI_H)
    c_content = _build_section_c(d, MINI_W, MINI_H)

    doc.build(
        # ── Pagina 1 ──
        _header_flowables(d) + [FrameBreak()] +
        [lk]                 + [FrameBreak()] +
        [rk]                 + [FrameBreak()] +
        _footer_flowables(d) +
        # ── Pagina 2 ──
        [NextPageTemplate('charts'), PageBreak()] +
        b_content + [FrameBreak()] +
        c_content
    )


# ════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ════════════════════════════════════════════════════════════════════════════

def generate_uncertainty_training_report(
    config, history, metrics, input_dim, output_dim, total_params,
    n_train, n_val, n_test, checkpoint_dir, timestamp=None, coverage_results=None,
):
    if timestamp is None:
        timestamp = datetime.now()
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    out = checkpoint_dir / 'uncertainty_training_report.pdf'
    _build_page(dict(config=config, history=history, metrics=metrics,
                     input_dim=input_dim, output_dim=output_dim, total_params=total_params,
                     n_train=n_train, n_val=n_val, n_test=n_test,
                     timestamp=timestamp, coverage_results=coverage_results), out)
    return str(out)


class UncertaintyReportGenerator:
    def __init__(self, output_path):
        self.output_path = Path(output_path)

    def generate(self, config, history, metrics, input_dim, output_dim,
                 total_params, n_train, n_val, n_test,
                 timestamp=None, coverage_results=None):
        if timestamp is None:
            timestamp = datetime.now()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        _build_page(dict(config=config, history=history, metrics=metrics,
                         input_dim=input_dim, output_dim=output_dim, total_params=total_params,
                         n_train=n_train, n_val=n_val, n_test=n_test,
                         timestamp=timestamp, coverage_results=coverage_results),
                    self.output_path)
        return str(self.output_path)


def combine_process_reports(report_paths, output_path, process_names=None):
    if not PYPDF_AVAILABLE:
        import warnings; warnings.warn("pypdf not available."); return None
    writer = PdfWriter()
    for i, path in enumerate(report_paths):
        p = Path(path)
        name = process_names[i] if process_names else f"Process {i+1}"
        if p.exists():
            r = PdfReader(str(p))
            print(f"Adding {len(r.pages)} page(s) from {name}...")
            for page in r.pages: writer.add_page(page)
        else:
            print(f"Warning: not found for {name}: {p}")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as f: writer.write(f)
    return str(out)


def create_2up_pdf(input_pdf, output_pdf, page_size=None):
    import shutil; shutil.copy(str(input_pdf), str(output_pdf)); return str(output_pdf)