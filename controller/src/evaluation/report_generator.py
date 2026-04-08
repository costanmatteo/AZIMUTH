"""
PDF Report Generator for Controller Optimization — A4 portrait.
Style: Helvetica, compact layout with tables and charts rearranged
for vertical (portrait) orientation.
"""

import math
import os
from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate,
    Paragraph, Spacer, Image, Table, TableStyle, PageBreak,
)
from reportlab.platypus.flowables import HRFlowable

try:
    from pypdf import PdfReader, PdfWriter
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

# ── page geometry (A4 portrait) ──────────────────────────────────────────────
PW, PH = A4               # 595.28 x 841.89 pts
M  = 1.2 * cm            # margins
TW = PW - 2 * M          # text width  ≈ 793 pts

def get_report_chart_sizes():
    """Return figsize (inches) for page-3 chart slots.

    Returns:
        (left_figsize, right_figsize) — each a (width, height) tuple in inches.
        *left_figsize*  is for each chart in the baseline/gap paired rows (2 per row).
        *right_figsize* is for each of the 3 stacked theoretical charts (full width).
    """
    pair_w   = TW / 2
    hdr_h    = 1.8 * cm
    ftr_h    = 1.5 * cm
    avail    = PH - 2 * M - hdr_h - ftr_h
    # Top portion: 2 rows of paired images (~40% of available height)
    pair_h   = int(avail * 0.4 / 2)
    # Bottom portion: 3 theoretical charts (~60% of available height)
    n_right  = 3
    inter_gap = 4
    theo_avail = avail * 0.6
    rh       = int((theo_avail - (n_right - 1) * inter_gap) / n_right)
    left_figsize  = (pair_w / 72.0, pair_h / 72.0)
    right_figsize = (TW     / 72.0, rh     / 72.0)
    return left_figsize, right_figsize

# ── colors ────────────────────────────────────────────────────────────────────
C_GREEN = colors.HexColor('#1D9E75')
C_RED   = colors.HexColor('#D85A30')
C_AMBER = colors.HexColor('#BA7517')
C_BLACK = colors.black
C_MUTED = colors.HexColor('#666666')
C_LGRAY = colors.HexColor('#EEEEEE')
C_MGRAY = colors.HexColor('#CCCCCC')
C_BGRAY = colors.HexColor('#F7F7F7')
C_TGRAY = colors.HexColor('#888888')

# ── font sizes ────────────────────────────────────────────────────────────────
FS_TITLE   = 11
FS_SUB     = 7
FS_SECTION = 7
FS_BODY    = 7
FS_KPI_LBL = 6.5
FS_KPI_VAL = 11
FS_KPI_SUB = 6.5
FS_BLK     = 6.5
FS_STATUS  = 6.5
FS_NOTE    = 6.5

# ── style factory ────────────────────────────────────────────────────────────
def _s(name, size, bold=False, italic=False, color=C_BLACK, align=TA_LEFT):
    font = 'Helvetica-Bold' if bold else ('Helvetica-Oblique' if italic else 'Helvetica')
    return ParagraphStyle(name, fontName=font, fontSize=size,
                          leading=size * 1.35, textColor=color, alignment=align)

ST_TITLE    = _s('ct_title',  FS_TITLE)
ST_SUB      = _s('ct_sub',    FS_SUB,     color=C_MUTED)
ST_SECTION  = _s('ct_sec',    FS_SECTION, bold=True, color=C_TGRAY)
ST_BLK      = _s('ct_blk',    FS_BLK,     bold=True, color=C_TGRAY)
ST_KEY      = _s('ct_key',    FS_BODY,    color=C_MUTED)
ST_VAL      = _s('ct_val',    FS_BODY,    align=TA_RIGHT)
ST_VAL_G    = _s('ct_val_g',  FS_BODY,    color=C_GREEN,  align=TA_RIGHT)
ST_VAL_R    = _s('ct_val_r',  FS_BODY,    color=C_RED,    align=TA_RIGHT)
ST_VAL_A    = _s('ct_val_a',  FS_BODY,    color=C_AMBER,  align=TA_RIGHT)
ST_KPI_LBL  = _s('ct_kl',    FS_KPI_LBL, color=C_TGRAY)
ST_KPI_VAL  = _s('ct_kv',    FS_KPI_VAL)
ST_KPI_SUB  = _s('ct_ks',    FS_KPI_SUB, color=C_TGRAY)
ST_NOTE     = _s('ct_note',   FS_NOTE,    color=C_TGRAY)
ST_NOTE_G   = _s('ct_note_g', FS_NOTE,    color=C_GREEN)
ST_NOTE_R   = _s('ct_note_r', FS_NOTE,    color=C_RED)
ST_STATUS_A = _s('ct_sta',    FS_STATUS,  color=C_AMBER, align=TA_CENTER)
ST_STATUS_G = _s('ct_stg',    FS_STATUS,  color=C_GREEN, align=TA_CENTER)
ST_STATUS_R = _s('ct_str',    FS_STATUS,  color=C_RED,   align=TA_CENTER)
ST_CAPTION  = _s('ct_cap',    FS_NOTE,    italic=True, color=C_TGRAY)
ST_FTR      = _s('ct_ftr',    FS_NOTE,    color=C_TGRAY)
ST_FTR_R    = _s('ct_ftrr',   FS_NOTE,    color=C_TGRAY, align=TA_RIGHT)
ST_TRAJ_H   = _s('ct_th',     5.5,        bold=False, color=C_TGRAY)
ST_TRAJ_C   = _s('ct_tc',     5.5)
ST_TRAJ_G   = _s('ct_tg',     FS_BODY,    color=C_GREEN)
ST_TRAJ_R   = _s('ct_tr',     FS_BODY,    color=C_RED)


# Formula comparison styles (italic, muted — used when CasualiT surrogate is active)
C_FORM     = colors.HexColor('#999999')  # Muted gray for formula sub-rows
ST_KEY_FORM = _s('ct_key_f', FS_BODY, italic=True, color=C_FORM)
ST_VAL_FORM = _s('ct_val_f', FS_BODY, italic=True, color=C_FORM, align=TA_RIGHT)
ST_VAL_FORM_G = _s('ct_val_fg', FS_BODY, italic=True, color=C_GREEN, align=TA_RIGHT)
ST_VAL_FORM_R = _s('ct_val_fr', FS_BODY, italic=True, color=C_RED, align=TA_RIGHT)

def _dyn(c, align=TA_RIGHT):
    return ParagraphStyle(f'_d{id(c)}{align}', fontName='Helvetica',
                          fontSize=FS_BODY, leading=FS_BODY * 1.35,
                          textColor=c, alignment=align)

# ── helpers ───────────────────────────────────────────────────────────────────
def fmt_lr(v):
    if v == 0: return "0"
    exp  = int(math.floor(math.log10(abs(v))))
    mant = v / (10 ** exp)
    return f"{mant:.4f} x 10^{exp}"

def short_dir(d, max_len=50):
    parts = Path(d).parts
    short = str(Path(*parts[-3:])) if len(parts) >= 3 else str(d)
    return short if len(short) <= max_len else '...' + short[-(max_len - 3):]

def _fval(v):
    if v is None:
        return 0.0, None
    if isinstance(v, dict):
        return float(v.get('mean', 0.0)), v.get('std', None)
    if isinstance(v, (list, tuple)):
        import numpy as _np
        arr = [float(x) for x in v if x is not None]
        if not arr:
            return 0.0, None
        return float(_np.mean(arr)), (float(_np.std(arr)) if len(arr) > 1 else None)
    try:
        return float(v), None
    except (TypeError, ValueError):
        return 0.0, None

def _last(v, default=0.0):
    if v is None:
        return default
    if isinstance(v, (list, tuple)):
        return float(v[-1]) if len(v) > 0 else default
    try:
        import numpy as _np
        if isinstance(v, _np.ndarray):
            return float(v.flat[-1]) if v.size > 0 else default
    except ImportError:
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return default

def _fstr(v, fmt='.6f'):
    mean, std = _fval(v)
    if std is not None:
        return f"{mean:{fmt}} \u00b1 {std:{fmt}}"
    return f"{mean:{fmt}}"

def scale_img(path, max_w, max_h):
    if not Path(path).exists():
        return _placeholder(Path(path).name, max_w, max_h)
    img   = Image(str(path))
    scale = min(max_w / img.imageWidth, max_h / img.imageHeight)
    img.drawWidth  = img.imageWidth  * scale
    img.drawHeight = img.imageHeight * scale
    return img

def scale_img_fw(path, target_w, max_h):
    """Scale image to fill target_w, keeping aspect ratio. If height exceeds
    max_h, shrink both dimensions proportionally."""
    if not Path(path).exists():
        return _placeholder(Path(path).name, target_w, max_h)
    img   = Image(str(path))
    scale = target_w / img.imageWidth
    h     = img.imageHeight * scale
    if h > max_h:
        scale = max_h / img.imageHeight
    img.drawWidth  = img.imageWidth  * scale
    img.drawHeight = img.imageHeight * scale
    return img

def _placeholder(name, w, h):
    st = ParagraphStyle('_ph', fontName='Helvetica', fontSize=FS_NOTE,
                        alignment=TA_CENTER, textColor=colors.HexColor('#AAAAAA'))
    t  = Table([[Paragraph(name, st)]], colWidths=[w], rowHeights=[h])
    t.setStyle(TableStyle([
        ('BOX',        (0, 0), (-1, -1), 0.5, C_MGRAY),
        ('BACKGROUND', (0, 0), (-1, -1), C_BGRAY),
        ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    return t

def _rule_heavy():
    return HRFlowable(width=TW, thickness=1.5, color=C_BLACK,
                      spaceBefore=3, spaceAfter=7)

def _rule_thin():
    return HRFlowable(width=TW, thickness=0.5, color=C_MGRAY,
                      spaceBefore=3, spaceAfter=4)

def section_header(title):
    return [
        Paragraph(title.upper(), ST_SECTION),
        HRFlowable(width=TW, thickness=0.5, color=C_MGRAY, spaceAfter=3),
    ]

def blk_title(title):
    return Paragraph(title, ST_BLK)

def kv_table(rows, col_w, key_frac=0.52):
    kw = col_w * key_frac
    vw = col_w * (1 - key_frac)
    data = []
    for row in rows:
        key, val = row[0], row[1]
        vs  = row[2] if len(row) > 2 else ST_VAL
        ks  = row[3] if len(row) > 3 else ST_KEY
        k   = Paragraph(key, ks)
        v   = Paragraph(val, vs) if isinstance(val, str) else val
        data.append([k, v])
    t = Table(data, colWidths=[kw, vw])
    t.setStyle(TableStyle([
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING',    (0, 0), (-1, -1), 1.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5),
        ('LEFTPADDING',   (0, 0), (-1, -1), 0),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
        ('LINEBELOW',     (0, 0), (-1, -2), 0.4, C_LGRAY),
    ]))
    return t

def _wrap_col(items, w):
    rows = [[item] for item in items]
    t = Table(rows, colWidths=[w])
    t.setStyle(TableStyle([
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 0),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
        ('TOPPADDING',    (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
    ]))
    return t

def two_col(left_items, right_items, gap=12):
    cw = (TW - gap) / 2
    outer = Table(
        [[_wrap_col(left_items, cw), Spacer(gap, 1), _wrap_col(right_items, cw)]],
        colWidths=[cw, gap, cw])
    outer.setStyle(TableStyle([
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 0),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
        ('TOPPADDING',    (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    return outer

def three_col(col1, col2, col3, gap=10):
    cw = (TW - 2 * gap) / 3
    outer = Table(
        [[_wrap_col(col1, cw), Spacer(gap, 1),
          _wrap_col(col2, cw), Spacer(gap, 1),
          _wrap_col(col3, cw)]],
        colWidths=[cw, gap, cw, gap, cw])
    outer.setStyle(TableStyle([
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 0),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
        ('TOPPADDING',    (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    return outer

def four_col(c1, c2, c3, c4, gap=8):
    cw = (TW - 3 * gap) / 4
    outer = Table(
        [[_wrap_col(c1, cw), Spacer(gap, 1),
          _wrap_col(c2, cw), Spacer(gap, 1),
          _wrap_col(c3, cw), Spacer(gap, 1),
          _wrap_col(c4, cw)]],
        colWidths=[cw, gap, cw, gap, cw, gap, cw])
    outer.setStyle(TableStyle([
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 0),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
        ('TOPPADDING',    (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    return outer

def five_col(c1, c2, c3, c4, c5, gap=8):
    cw = (TW - 4 * gap) / 5
    outer = Table(
        [[_wrap_col(c1, cw), Spacer(gap, 1),
          _wrap_col(c2, cw), Spacer(gap, 1),
          _wrap_col(c3, cw), Spacer(gap, 1),
          _wrap_col(c4, cw), Spacer(gap, 1),
          _wrap_col(c5, cw)]],
        colWidths=[cw, gap, cw, gap, cw, gap, cw, gap, cw])
    outer.setStyle(TableStyle([
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 0),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
        ('TOPPADDING',    (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    return outer

def img_pair(left_path, right_path, left_cap, right_cap, h, gap=0):
    cw  = (TW - gap) / 2
    lt = Table(
        [[scale_img(left_path,  cw, h)],
         [Paragraph(left_cap,  ST_CAPTION)]],
        colWidths=[cw])
    rt = Table(
        [[scale_img(right_path, cw, h)],
         [Paragraph(right_cap, ST_CAPTION)]],
        colWidths=[cw])
    for t in (lt, rt):
        t.setStyle(TableStyle([
            ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
            ('ALIGN',         (0, 0), (0,  0),  'CENTER'),
            ('LEFTPADDING',   (0, 0), (-1, -1), 0),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
            ('TOPPADDING',    (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
        ]))
    if gap > 0:
        outer = Table([[lt, Spacer(gap, 1), rt]],
                      colWidths=[cw, gap, cw])
    else:
        outer = Table([[lt, rt]], colWidths=[cw, cw])
    outer.setStyle(TableStyle([
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 0),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
        ('TOPPADDING',    (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    return outer


def img_quad(paths, captions, h):
    """Four images side by side, no gap, forced to same height."""
    cw = TW / 4
    # Load all images and compute uniform height
    imgs = []
    for p in paths:
        if not Path(p).exists():
            imgs.append(None)
        else:
            imgs.append(Image(str(p)))
    # Find the uniform height: scale each to fit cw width, take the minimum
    heights = []
    for img in imgs:
        if img is not None:
            w_scale = cw / img.imageWidth
            heights.append(img.imageHeight * w_scale)
    target_h = min(min(heights), h) if heights else h
    # Build cells with uniform height
    cells = []
    for img, p, cap in zip(imgs, paths, captions):
        if img is None:
            el = _placeholder(Path(p).name, cw, target_h)
        else:
            scale = target_h / img.imageHeight
            img.drawWidth  = img.imageWidth * scale
            img.drawHeight = target_h
        cell = Table(
            [[img if img is not None else el],
             [Paragraph(cap, ST_CAPTION)]],
            colWidths=[cw], rowHeights=[target_h, None])
        cell.setStyle(TableStyle([
            ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
            ('ALIGN',         (0, 0), (0,  0),  'CENTER'),
            ('LEFTPADDING',   (0, 0), (-1, -1), 0),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
            ('TOPPADDING',    (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
        ]))
        cells.append(cell)
    outer = Table([cells], colWidths=[cw] * 4)
    outer.setStyle(TableStyle([
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 0),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
        ('TOPPADDING',    (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    return outer


def img_stack(paths, captions, w, h_each, gap=4):
    """Stack images vertically within a given width, each capped at h_each."""
    rows = []
    for p, cap in zip(paths, captions):
        img = scale_img(p, w, h_each)
        cell = Table(
            [[img], [Paragraph(cap, ST_CAPTION)]],
            colWidths=[w])
        cell.setStyle(TableStyle([
            ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
            ('ALIGN',         (0, 0), (0,  0),  'CENTER'),
            ('LEFTPADDING',   (0, 0), (-1, -1), 0),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
            ('TOPPADDING',    (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
        ]))
        rows.append([cell])
        if gap > 0:
            rows.append([Spacer(1, gap)])
    t = Table(rows, colWidths=[w])
    t.setStyle(TableStyle([
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 0),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
        ('TOPPADDING',    (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    return t


def img_grid_2x2(paths, captions, total_w, total_h, gap=4):
    """Four images in a 2x2 grid within total_w x total_h."""
    cw = (total_w - gap) / 2
    rh = (total_h - gap) / 2
    cells = []
    for p, cap in zip(paths, captions):
        img = scale_img(p, cw, rh)
        cell = Table(
            [[img], [Paragraph(cap, ST_CAPTION)]],
            colWidths=[cw])
        cell.setStyle(TableStyle([
            ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
            ('ALIGN',         (0, 0), (0,  0),  'CENTER'),
            ('LEFTPADDING',   (0, 0), (-1, -1), 0),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
            ('TOPPADDING',    (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
        ]))
        cells.append(cell)
    # Arrange as 2 rows x 2 cols
    grid = Table(
        [[cells[0], cells[1]],
         [cells[2], cells[3]]],
        colWidths=[cw, cw],
        rowHeights=[rh + 12, rh + 12])  # +12 for caption
    grid.setStyle(TableStyle([
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 0),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
        ('TOPPADDING',    (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    return grid


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — text metrics (portrait: 2-column layout)
# ════════════════════════════════════════════════════════════════════════════

def _page1(d, total_pages):
    cfg    = d['config']
    hist   = d.get('training_history', {})
    fm     = d.get('final_metrics', {})
    adv    = d.get('advanced_metrics') or {}
    ts     = d.get('timestamp', datetime.now())
    ts_str = ts.strftime('%Y-%m-%d %H:%M:%S') if isinstance(ts, datetime) else str(ts)

    tr_cfg = cfg.get('training', {})
    pg_cfg = cfg.get('policy_generator', {})

    F_star = d.get('F_star', 0.0)
    F_bl   = d.get('F_baseline', 0.0)
    F_act  = d.get('F_actual', 0.0)
    F_form = d.get('F_formula')  # Formula-based F (only when using CasualiT)

    fstar_v, _       = _fval(F_star)
    fbl_v,   fbl_s   = _fval(F_bl)
    fact_v,  fact_s  = _fval(F_act)
    fform_v, fform_s = _fval(F_form) if F_form is not None else (None, None)

    epochs     = tr_cfg.get('epochs', hist.get('epochs_trained', '\u2014'))
    max_epochs = tr_cfg.get('max_epochs', tr_cfg.get('epochs', '\u2014'))
    patience   = tr_cfg.get('patience', '\u2014')
    seed       = cfg.get('misc', {}).get('random_seed', '\u2014')

    completed  = hist.get('completed', True)
    status_txt = "complete"  if completed else "incomplete"
    status_st  = ST_STATUS_G if completed else ST_STATUS_A
    status_col = C_GREEN     if completed else C_AMBER

    # When formula is available (CasualiT mode): primary metrics use formula F,
    # sub-rows "■ surrogate" show the CasualiT values.
    # When formula is NOT available: primary metrics use surrogate F directly.
    # Improvement = gap reduction vs baseline:
    #   gap_bl  = |F_baseline - F_star|
    #   gap_ctrl = |F_controller - F_star|
    #   improvement = (gap_bl - gap_ctrl) / gap_bl × 100
    gap_bl = abs(fbl_v - fstar_v)
    if fform_v is not None:
        # Primary = formula, secondary = surrogate (CasualiT)
        gap_form = abs(fform_v - fstar_v)
        gap_surr = abs(fact_v  - fstar_v)
        improv_pct      = (gap_bl - gap_form) / gap_bl * 100 if gap_bl else 0.0
        improv_surr_pct = (gap_bl - gap_surr) / gap_bl * 100 if gap_bl else 0.0
        gap_pct         = abs(fstar_v - fform_v) / fstar_v * 100 if fstar_v else 0.0
        gap_surr_pct    = abs(fstar_v - fact_v)  / fstar_v * 100 if fstar_v else 0.0
    else:
        gap_ctrl = abs(fact_v - fstar_v)
        improv_pct      = (gap_bl - gap_ctrl) / gap_bl * 100 if gap_bl else 0.0
        gap_pct         = abs(fstar_v - fact_v) / fstar_v * 100 if fstar_v else 0.0
        improv_surr_pct = 0.0
        gap_surr_pct    = 0.0
    best_loss  = _last(hist.get('best_total_loss',  hist.get('best_loss',  0.0)))
    final_loss = _last(hist.get('final_total_loss', hist.get('total_loss', 0.0)))
    rob_std    = fact_s if fact_s is not None else _last(fm.get('robustness_std', 0.0))

    # Train gap-reduction (same formula as improv_pct but on train F values)
    fbl_train  = adv.get('F_baseline_train_mean', fbl_v)
    fact_train = adv.get('F_actual_train_mean', fact_v)
    rob_std_train = adv.get('F_actual_train_std', rob_std)
    gap_bl_train   = abs(fbl_train - fstar_v)
    gap_ctrl_train = abs(fact_train - fstar_v)
    improv_train = (gap_bl_train - gap_ctrl_train) / gap_bl_train * 100 if gap_bl_train else 0.0
    # Test uses the locally computed improv_pct (already gap-reduction)
    improv_test = improv_pct

    F = []

    # ── header ────────────────────────────────────────────────────────────────
    title_p  = Paragraph("Controller Optimization \u2014 Training Report", ST_TITLE)
    meta_str = (f"{ts_str}  \u00b7  seed {seed}  \u00b7  "
                f"epochs {epochs} / {max_epochs}  \u00b7  patience {patience}")
    badge_w  = 2.2 * cm
    badge_p  = Paragraph(status_txt, status_st)
    hdr_tbl  = Table([[title_p, badge_p]], colWidths=[TW - badge_w, badge_w])
    hdr_tbl.setStyle(TableStyle([
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 0),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
        ('TOPPADDING',    (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('BOX',           (1, 0), (1,  0),  0.5, status_col),
        ('ALIGN',         (1, 0), (1,  0),  'CENTER'),
    ]))
    F.append(hdr_tbl)
    F.append(Spacer(1, 1))
    F.append(Paragraph(meta_str, ST_SUB))
    F.append(_rule_heavy())

    # ── KPI bar ───────────────────────────────────────────────────────────────
    if fform_v is not None:
        n_kpi = 7
        cw = TW / n_kpi
        kpi_data = [
            [Paragraph("Controller F (test)",  ST_KPI_LBL),
             Paragraph("Controller F (train)", ST_KPI_LBL),
             Paragraph("Formula F",            ST_KPI_LBL),
             Paragraph("Target F*",            ST_KPI_LBL),
             Paragraph("vs baseline (test)",   ST_KPI_LBL),
             Paragraph("vs baseline (train)",  ST_KPI_LBL),
             Paragraph("Best loss",            ST_KPI_LBL)],
            [Paragraph(f"{fact_v:.4f}",     ST_KPI_VAL),
             Paragraph(f"{fact_train:.4f}", ST_KPI_VAL),
             Paragraph(f"{fform_v:.4f}",    ST_KPI_VAL),
             Paragraph(f"{fstar_v:.4f}",    ST_KPI_VAL),
             Paragraph(f"{improv_test:+.2f}%",
                       _s('_kv_g', FS_KPI_VAL,
                          color=C_GREEN if improv_test >= 0 else C_RED)),
             Paragraph(f"{improv_train:+.2f}%",
                       _s('_kv_t', FS_KPI_VAL,
                          color=C_GREEN if improv_train >= 0 else C_RED)),
             Paragraph(f"{best_loss:.4f}", ST_KPI_VAL)],
            [Paragraph(f"\u00b1{rob_std:.4f} robustness",     ST_KPI_SUB),
             Paragraph(f"\u00b1{rob_std_train:.4f} robustness", ST_KPI_SUB),
             Paragraph(f"\u00b1{fform_s:.4f}" if fform_s else "", ST_KPI_SUB),
             Paragraph(f"gap {gap_pct:.1f}%",               ST_KPI_SUB),
             Paragraph(f"F\u2019_test = {fbl_v:.4f}",       ST_KPI_SUB),
             Paragraph(f"F\u2019_train = {fbl_train:.4f}",   ST_KPI_SUB),
             Paragraph(f"final {final_loss:.2f}",            ST_KPI_SUB)],
        ]
    else:
        n_kpi = 6
        cw = TW / n_kpi
        kpi_data = [
            [Paragraph("Controller F (test)",  ST_KPI_LBL),
             Paragraph("Controller F (train)", ST_KPI_LBL),
             Paragraph("Target F*",            ST_KPI_LBL),
             Paragraph("vs baseline (test)",   ST_KPI_LBL),
             Paragraph("vs baseline (train)",  ST_KPI_LBL),
             Paragraph("Best loss",            ST_KPI_LBL)],
            [Paragraph(f"{fact_v:.4f}",     ST_KPI_VAL),
             Paragraph(f"{fact_train:.4f}", ST_KPI_VAL),
             Paragraph(f"{fstar_v:.4f}",    ST_KPI_VAL),
             Paragraph(f"{improv_test:+.2f}%",
                       _s('_kv_g', FS_KPI_VAL,
                          color=C_GREEN if improv_test >= 0 else C_RED)),
             Paragraph(f"{improv_train:+.2f}%",
                       _s('_kv_t', FS_KPI_VAL,
                          color=C_GREEN if improv_train >= 0 else C_RED)),
             Paragraph(f"{best_loss:.4f}", ST_KPI_VAL)],
            [Paragraph(f"\u00b1{rob_std:.4f} robustness",     ST_KPI_SUB),
             Paragraph(f"\u00b1{rob_std_train:.4f} robustness", ST_KPI_SUB),
             Paragraph(f"gap {gap_pct:.1f}%",               ST_KPI_SUB),
             Paragraph(f"F\u2019_test = {fbl_v:.4f}",       ST_KPI_SUB),
             Paragraph(f"F\u2019_train = {fbl_train:.4f}",   ST_KPI_SUB),
             Paragraph(f"final {final_loss:.2f}",            ST_KPI_SUB)],
        ]
    kpi_tbl = Table(kpi_data, colWidths=[cw] * n_kpi)
    kpi_tbl.setStyle(TableStyle([
        ('BOX',           (0, 0), (-1, -1), 0.5, C_MGRAY),
        ('INNERGRID',     (0, 0), (-1, -1), 0.5, C_MGRAY),
        ('TOPPADDING',    (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING',   (0, 0), (-1, -1), 7),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 7),
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
    ]))
    F.append(kpi_tbl)
    F.append(Spacer(1, 7))

    # ── 01 — configuration & training (4 columns: arch | training | scores | performance) ──
    F += section_header("01 \u2014 configuration & training parameters")

    n_scenarios = d.get('n_scenarios', tr_cfg.get('n_scenarios', '\u2014'))
    scen_cfg = cfg.get('scenarios', {})
    n_train    = scen_cfg.get('n_train', '\u2014')
    seeds_t    = scen_cfg.get('seed_target',
                    cfg.get('seeds_target',
                        cfg.get('misc', {}).get('seeds_target', '\u2014')))
    seeds_b    = scen_cfg.get('seed_baseline',
                    cfg.get('seeds_baseline',
                        cfg.get('misc', {}).get('seeds_baseline', '\u2014')))
    obs_mode   = pg_cfg.get('observation_mode', '\u2014')
    bellman_cfg = cfg.get('bellman', {})
    bellman_lvl = bellman_cfg.get('level', '\u2014')
    _lvl_desc  = {1: 'fixed var, indep.',
                  2: 'action-dep var, indep.',
                  3: 'action-dep var, correlated'}
    lvl_str    = (f"{bellman_lvl} ({_lvl_desc[bellman_lvl]})"
                  if bellman_lvl in _lvl_desc else str(bellman_lvl))
    lam_bc  = tr_cfg.get('lambda_bc',   tr_cfg.get('lam_bc', '\u2014'))
    chk     = short_dir(tr_cfg.get('checkpoint_dir', ''))

    arch_rows = [
        ("Policy",          str(pg_cfg.get('architecture', '\u2014')) + ' ' +
                            str(pg_cfg.get('hidden_sizes', ''))),
        ("Activation",      str(pg_cfg.get('activation',   'N/A'))),
        ("Dropout rate",    str(pg_cfg.get('dropout_rate', 'N/A'))),
        ("Obs. mode",       str(obs_mode)),
        ("Bellman level",   lvl_str),
        ("Processes",       ', '.join(cfg.get('process_names') or ['all'])),
        ("n_train",         str(n_train)),
        ("Train scenarios", str(n_scenarios)),
        ("Seeds (T / B)",   f"{seeds_t} / {seeds_b}"),
    ]
    bs = tr_cfg.get('batch_size', '\u2014')
    train_rows = [
        ("Epochs",          f"{epochs} / {max_epochs}"),
        ("Batch size",      f"{bs:,}" if isinstance(bs, int) else str(bs)),
        ("Learning rate",   fmt_lr(tr_cfg.get('learning_rate', 0))),
        ("Weight decay",    fmt_lr(tr_cfg.get('weight_decay', 0))),
        ("\u03bb_BC",       fmt_lr(float(lam_bc)) if isinstance(lam_bc, (int, float)) else str(lam_bc)),
        ("Patience",        str(patience)),
        ("Device",          str(tr_cfg.get('device', '\u2014'))),
        ("Checkpoint dir",  chk),
    ]
    cw2 = (TW - 12) / 2

    # Row 1: Architecture | Training
    F.append(two_col(
        [blk_title("Architecture")] + [kv_table(arch_rows,  cw2)],
        [blk_title("Training")]     + [kv_table(train_rows, cw2)],
        gap=12,
    ))
    F.append(Spacer(1, 5))

    # Row 2: Reliability scores | Performance
    gap_pct_train = abs(fstar_v - fact_train) / fstar_v * 100 if fstar_v else 0.0
    reliability_rows = [
        ("F* (target)",                  f"{fstar_v:.6f}"),
        ("F\u2019 (baseline, test)",      _fstr(F_bl)),
        ("F\u2019 (baseline, train)",     f"{fbl_train:.6f}"),
    ] + ([
        ("F (formula)",                   _fstr(F_form)),
        ("\u2514 surrogate (test)",       _fstr(F_act),
         ST_VAL_FORM, ST_KEY_FORM),
        ("\u2514 surrogate (train)",      f"{fact_train:.6f} \u00b1 {rob_std_train:.6f}",
         ST_VAL_FORM, ST_KEY_FORM),
    ] if fform_v is not None else [
        ("F (controller, test)",          _fstr(F_act)),
        ("F (controller, train)",         f"{fact_train:.6f} \u00b1 {rob_std_train:.6f}"),
    ]) + [
        ("Improvement vs baseline (test)",  f"{improv_test:+.2f}%",
         ST_VAL_G if improv_test >= 0 else ST_VAL_R),
        ("Improvement vs baseline (train)", f"{improv_train:+.2f}%",
         ST_VAL_G if improv_train >= 0 else ST_VAL_R),
    ] + ([
        ("\u2514 surrogate",
         f"{improv_surr_pct:+.2f}%",
         ST_VAL_FORM_G if improv_surr_pct >= 0 else ST_VAL_FORM_R,
         ST_KEY_FORM),
    ] if fform_v is not None else []) + [
        ("Gap from target (test)",        f"{gap_pct:.2f}%", ST_VAL_R),
        ("Gap from target (train)",       f"{gap_pct_train:.2f}%", ST_VAL_R),
    ] + ([
        ("\u2514 surrogate",
         f"{gap_surr_pct:.2f}%",
         ST_VAL_FORM_R,
         ST_KEY_FORM),
    ] if fform_v is not None else []) + [
        ("Robustness (std, test)",        f"{rob_std:.6f}"),
        ("Robustness (std, train)",       f"{rob_std_train:.6f}"),
    ]
    F.append(two_col(
        [blk_title("Reliability scores")] + [kv_table(reliability_rows, cw2)],
        [blk_title("Performance")] + [kv_table(_perf_rows(adv, n_scenarios), cw2)],
        gap=12,
    ))
    F.append(Spacer(1, 7))

    # ── 02 — loss decomposition & generalization (3 + 2 cols) ─────────────────
    F += section_header(
        "02 \u2014 loss decomposition & L_min Bellman  \u00b7  "
        "03 \u2014 overfitting & generalization")

    theo         = d.get('theoretical_data') or {}
    theo_summary = theo.get('summary', {})
    final_total  = _last(hist.get('final_total_loss', hist.get('total_loss', final_loss)))
    _rel_raw     = hist.get('final_reliability_loss', hist.get('reliability_loss', None))
    _bc_raw      = hist.get('final_bc_loss',          hist.get('bc_loss',          None))
    final_rel    = _last(_rel_raw) if _rel_raw is not None else '\u2014'
    final_bc     = _last(_bc_raw)  if _bc_raw  is not None else '\u2014'

    def _tv(v, fmt='.6f'):
        if v is None or v == '\u2014': return '\u2014'
        try:    return f"{float(v):{fmt}}"
        except: return str(v)

    def _theo(key, sk=None, fallback='\u2014'):
        v = theo.get(key)
        if v is not None and not isinstance(v, (list, tuple)):
            if hasattr(v, '__len__') and hasattr(v, 'flat'):
                return _last(v) if len(v) > 0 else fallback
            return v
        if sk and sk in theo_summary:
            return theo_summary[sk]
        if isinstance(v, (list, tuple)):
            return _last(v) if v else fallback
        return fallback

    lmin_emp = _theo('lmin_empirical', 'final_L_min')
    gap_red  = _theo('gap_reducible',  'final_gap')
    eff      = _theo('efficiency',     'final_efficiency')
    viol     = theo.get('n_violations', theo_summary.get('n_violations', 0))
    var_f    = _theo('var_f',  'theoretical_Var_F_final')
    bias2    = _theo('bias2',  'empirical_Var_F_final')
    gap_r    = _theo('gap_reducible', 'final_gap')
    pct_str  = theo.get('decomp_pct', '\u2014')
    total_ep = epochs if isinstance(epochs, int) else '?'

    # Bellman data — forward MC is the primary benchmark, backward is informational
    bellman   = theo.get('bellman_lmin', {})
    lmin_bel  = bellman.get('L_min_bellman')  # backward (informational)
    lmin_fwd  = bellman.get('L_min_forward')  # forward MC (primary)
    lmin_fse  = bellman.get('L_min_forward_se')
    viol_bel  = bellman.get('n_violations', viol)
    level_res = bellman.get('level_results')  # multi-level results (parallel_levels)
    # Lambda_grad (Delta Method approximation of L_min)
    lg_data   = theo.get('lambda_grad', {})
    lg_val    = lg_data.get('lambda_grad')  # scalar Λ_grad(D)
    # Lambda_MC (Monte Carlo Method 2)
    lmc_data  = theo.get('lambda_mc', {})
    lmc_val   = lmc_data.get('lambda_mc')   # scalar Λ_MC(D)
    # Compute gap and efficiency using forward MC L_min (primary)
    if lmin_fwd is not None and final_total != '\u2014':
        try:
            _ft = float(final_total)
            gap_bel = _ft - float(lmin_fwd)
            eff_bel = float(lmin_fwd) / _ft * 100 if _ft > 0 else 0.0
        except (TypeError, ValueError):
            gap_bel, eff_bel = None, None
    else:
        gap_bel, eff_bel = None, None
    # Compute efficiency using backward induction L_min
    if lmin_bel is not None and final_total != '\u2014':
        try:
            _ft = float(final_total)
            eff_bel_bwd = float(lmin_bel) / _ft * 100 if _ft > 0 else 0.0
        except (TypeError, ValueError):
            eff_bel_bwd = None
    else:
        eff_bel_bwd = None

    # overfitting data
    ttg  = adv.get('train_test_gap')       or adv.get('overfitting')    or {}
    wig  = adv.get('within_scenario_gap')  or adv.get('intra_scenario') or {}
    # Formula-based train-test gap (only when CasualiT surrogate is used)
    fttg = adv.get('formula_train_test_gap') or {}

    def _scalar(v):
        if v is None: return None
        if isinstance(v, (list, tuple)): return v[0] if len(v) else None
        if hasattr(v, 'item'): return v.item()
        return v

    def _ov_note(v):
        if v is None: return ''
        return ' (possible overfit)' if float(v) < -0.005 else ' \u2014 consistent'

    mg_tr  = _scalar(ttg.get('mean_gap_train',  ttg.get('mean_F_train', None)))
    mg_te  = _scalar(ttg.get('mean_gap_test',   ttg.get('mean_F_test',  None)))
    diff   = _scalar(ttg.get('diff',            ttg.get('gap_train_minus_test', None)))
    # Formula-based equivalents
    fmg_tr = _scalar(fttg.get('mean_gap_train'))
    fmg_te = _scalar(fttg.get('mean_gap_test'))
    fdiff  = _scalar(fttg.get('train_test_gap'))
    cv_tr  = _scalar(ttg.get('dataset_cv_train', ttg.get('cv_train', None)))
    cv_te  = _scalar(ttg.get('dataset_cv_test',  ttg.get('cv_test',  None)))
    mf_tr  = wig.get('mean_f_train',  wig.get('mean_F_train_split', None))
    mf_val = wig.get('mean_f_val',    wig.get('mean_F_val_split',   None))
    gap_iv = wig.get('gap',           wig.get('gap_train_minus_val', None))
    div_ep = wig.get('divergent_epochs', wig.get('n_divergent_epochs', 0)) or 0

    losses_rows = [
        ("Total",       _tv(final_total)),
        ("Reliability", _tv(final_rel)   if final_rel != '\u2014' else '\u2014'),
        ("BC",          _tv(final_bc)    if final_bc  != '\u2014' else '\u2014'),
        ("Best total",  _tv(best_loss),   ST_VAL_G),
    ]
    if lmin_fwd is not None:
        lmin_rows = [
            ("L_min Bellman (forward)",
             f"{float(lmin_fwd):.6f} \u00b1 {float(lmin_fse):.6f}"
             if lmin_fse is not None else _tv(lmin_fwd)),
            ("L_min Bellman (backward)",  _tv(lmin_bel) if lmin_bel is not None else '\u2014'),
            ("\u039b_grad (Delta Method)", _tv(lg_val) if lg_val is not None else '\u2014'),
            ("\u039b_MC (Monte Carlo)", _tv(lmc_val) if lmc_val is not None else '\u2014'),
            ("\u039b_MC / \u039b_grad",
             f"{float(lmc_val) / float(lg_val):.3f}"
             if lmc_val is not None and lg_val is not None and float(lg_val) > 0
             else '\u2014'),
            ("Gap (obs \u2212 L_min fwd)",  _tv(gap_bel) if gap_bel is not None else '\u2014'),
            ("Efficiency (forward)",
             f"{eff_bel:.1f}%" if eff_bel is not None else '\u2014',
             ST_VAL_G),
            ("Efficiency (backward)",
             f"{eff_bel_bwd:.1f}%" if eff_bel_bwd is not None else '\u2014',
             ST_VAL_G),
            (f"Violations (loss&lt;L_min)", f"{viol_bel} / {total_ep}",
             ST_VAL_G if viol_bel == 0 else ST_VAL_R),
        ]
        # Multi-level comparison rows (when parallel_levels was enabled)
        if level_res is not None:
            _lvl_names = {1: 'L1 (fixed σ², Σ=I)',
                          2: 'L2 (free σ², Σ=I)',
                          3: 'L3 (free σ², full Σ)'}
            lmin_rows.append(("", ""))  # spacer
            for _lvl in sorted(level_res.keys(), key=lambda x: int(x)):
                _lr = level_res[_lvl]
                _lf = _lr.get('L_min_forward')
                _lfe = _lr.get('L_min_forward_se')
                if _lf is not None:
                    _val = (f"{float(_lf):.6f} ± {float(_lfe):.6f}"
                            if _lfe is not None else _tv(_lf))
                    lmin_rows.append((_lvl_names.get(int(_lvl), f'Level {_lvl}'), _val))
    else:
        lmin_rows = [
            ("Var[F]",                   _tv(lmin_emp)),
            ("\u039b_grad (Delta Method)", _tv(lg_val) if lg_val is not None else '\u2014'),
            ("\u039b_MC (Monte Carlo)", _tv(lmc_val) if lmc_val is not None else '\u2014'),
            ("\u039b_MC / \u039b_grad",
             f"{float(lmc_val) / float(lg_val):.3f}"
             if lmc_val is not None and lg_val is not None and float(lg_val) > 0
             else '\u2014'),
            ("Gap (reducible)",          _tv(gap_red)),
            ("Efficiency",               f"{float(eff)*100:.1f}%" if eff != '\u2014' else '\u2014',
             ST_VAL_G),
            (f"Violations (loss&lt;L_min)", f"{viol} / {total_ep}",
             ST_VAL_G if viol == 0 else ST_VAL_R),
        ]
    decomp_rows = [
        ("Var[F]",                            _tv(lmin_emp)),
        ("Var(F) \u2014 irreducible",        _tv(var_f)),
        ("Bias\u00b2 \u2014 irreducible",    _tv(bias2)),
        ("Gap \u2014 reducible",             _tv(gap_r)),
        ("% of loss",                        str(pct_str)),
    ]
    cross_rows = [
        ("Mean gap \u2014 train",         f"{float(mg_tr):.6f}"  if mg_tr  is not None else '\u2014'),
    ]
    if fmg_tr is not None:
        cross_rows.append(("\u2514 formula", f"{float(fmg_tr):.6f}", ST_VAL_FORM, ST_KEY_FORM))
    cross_rows.append(
        ("Mean gap \u2014 test",          f"{float(mg_te):.6f}"  if mg_te  is not None else '\u2014'))
    if fmg_te is not None:
        cross_rows.append(("\u2514 formula", f"{float(fmg_te):.6f}", ST_VAL_FORM, ST_KEY_FORM))
    cross_rows.append(
        ("Diff (train \u2212 test)",
         f"{float(diff):.6f}{_ov_note(diff)}" if diff is not None else '\u2014',
         ST_VAL_A if diff is not None and float(diff) < -0.005 else ST_VAL))
    if fdiff is not None:
        cross_rows.append(("\u2514 formula", f"{float(fdiff):.6f}", ST_VAL_FORM, ST_KEY_FORM))
    cross_rows += [
        ("Dataset CV \u2014 train",       f"{float(cv_tr):.4f}"  if cv_tr  is not None else '\u2014'),
        ("Dataset CV \u2014 test",        f"{float(cv_te):.4f}"  if cv_te  is not None else '\u2014'),
    ]
    intra_rows = [
        ("Mean F \u2014 train split",     f"{float(mf_tr):.6f}"  if mf_tr  is not None and mf_tr  != '\u2014' else '\u2014'),
        ("Mean F \u2014 val split",       f"{float(mf_val):.6f}" if mf_val is not None and mf_val != '\u2014' else '\u2014'),
        ("Gap (train \u2212 val)",
         f"{float(gap_iv):.6f}{_ov_note(gap_iv)}" if gap_iv is not None and gap_iv != '\u2014' else '\u2014',
         ST_VAL_G if gap_iv is not None and abs(float(gap_iv)) < 0.005 else ST_VAL),
        (f"Divergent epochs (>0.01)",
         f"{div_ep} / {total_ep}",
         ST_VAL_G if div_ep == 0 else ST_VAL_R),
    ]

    # Row 1: Final losses | L_min Bellman
    F.append(two_col(
        [blk_title("Final losses")]                        + [kv_table(losses_rows,  cw2)],
        [blk_title("L_min Bellman (backward induction)")] + [kv_table(lmin_rows,    cw2)],
        gap=12,
    ))
    F.append(Spacer(1, 5))

    # Row 2: Decomposition | Cross-scenario overfitting
    F.append(two_col(
        [blk_title("Decomposition")]                       + [kv_table(decomp_rows,  cw2)],
        [blk_title("Cross-scenario overfitting")]          + [kv_table(cross_rows,   cw2)],
        gap=12,
    ))
    F.append(Spacer(1, 5))

    # intra-scenario — compact half-width table
    F.append(blk_title("Intra-scenario (last 50 epochs)"))
    F.append(kv_table(intra_rows, TW * 0.5, key_frac=0.42))
    F.append(Spacer(1, 7))

    F += _footer(d, 1, total_pages)

    return F


def _page4_trajectory(d, total_pages):
    """Trajectory comparison tables (section 04) — one page per scenario."""
    cfg    = d['config']
    fstar_v, _       = _fval(d.get('F_star', 0.0))
    fbl_v,   fbl_s   = _fval(d.get('F_baseline', 0.0))
    fact_v,  fact_s  = _fval(d.get('F_actual', 0.0))

    F = []

    # ── 04 — trajectory comparison (all scenarios, best run, controllable inputs) ──
    traj = d.get('trajectory_values') or {}
    traj_list = d.get('trajectory_values_list', [])
    if not traj_list and traj:
        traj_list = [traj]

    def _to_np(arr):
        """Convert tensor or array to numpy, return first sample."""
        if hasattr(arr, 'detach'):
            arr = arr.detach().cpu().numpy()
        if hasattr(arr, '__len__') and len(arr) > 0:
            return arr[0]
        return arr

    evo_paths = d.get('evolution_plot_paths', {})
    evo_colors = d.get('evolution_color_maps', {})  # {proc_name: {var_label: hex_color}}

    def _color_dot(hex_color):
        """Inline HTML for a small colored square."""
        return (f'<font color="{hex_color}">\u25a0</font> ')

    if traj_list:
        F += section_header("04 \u2014 trajectory comparison (best run per scenario)")

        # Width split: data table ~58%, plot ~42%
        data_w = TW * 0.58
        plot_w = TW * 0.42

        for traj_i_idx, traj_i in enumerate(traj_list):
            # Each scenario starts on a new page
            if traj_i_idx > 0:
                F.append(PageBreak())

            sc_idx = traj_i.get('scenario_idx', traj_i_idx)
            ctrl_info = traj_i.get('controllable_info', {})

            # Scenario sub-header with F values
            tf_s  = traj_i.get('F_star',     fstar_v)
            tf_bl = traj_i.get('F_baseline', fbl_v)
            tf_ac = traj_i.get('F_actual',   fact_v)
            tf_form_ac = traj_i.get('F_formula_actual')
            sc_info = f"F* {tf_s:.4f}  \u00b7  F\u2019 {tf_bl:.4f}  \u00b7  F {tf_ac:.4f}"
            if tf_form_ac is not None:
                sc_info += f"  \u00b7  F(formula) {tf_form_ac:.4f}"
            F.append(Paragraph(
                f"<b>Scenario {sc_idx}</b> &mdash; {sc_info}", ST_NOTE))
            F.append(Spacer(1, 2))

            p_names = traj_i.get('process_names', [])
            t_traj  = traj_i.get('target_trajectory', {})
            b_traj  = traj_i.get('baseline_trajectory', {})
            a_traj  = traj_i.get('actual_trajectory', {})

            # Pre-compute max data rows across processes for uniform plot height
            max_data_rows = 0
            for pi, pn in enumerate(p_names):
                pi_info = ctrl_info.get(pn, {})
                ci = pi_info.get('controllable_indices', [])
                ol = pi_info.get('output_labels', [])
                nr = (len(ci) if pi > 0 else 0) + len(ol)
                max_data_rows = max(max_data_rows, nr)
            uniform_tbl_h = 13 + max_data_rows * 14

            # Build one row per process: [data_subtable | evolution_plot]
            for proc_idx, proc in enumerate(p_names):
                p_info = ctrl_info.get(proc, {})
                input_labels = p_info.get('input_labels', [])
                output_labels = p_info.get('output_labels', [])
                ctrl_indices = p_info.get('controllable_indices', [])

                t_inputs = _to_np(t_traj.get(proc, {}).get('inputs', []))
                a_inputs = _to_np(a_traj.get(proc, {}).get('inputs', []))
                t_outputs = _to_np(t_traj.get(proc, {}).get('outputs', []))
                b_outputs = _to_np(b_traj.get(proc, {}).get('outputs', []))
                a_outputs_raw = a_traj.get(proc, {})
                a_outputs = _to_np(a_outputs_raw.get('outputs_mean',
                                   a_outputs_raw.get('outputs', [])))

                # Build data rows for this process
                proc_hdr = [Paragraph(h, ST_TRAJ_H) for h in
                            ["Variable", "Target", "Controller", "\u0394", ""]]
                data_cws = [r * data_w for r in [0.22, 0.18, 0.18, 0.18, 0.24]]
                proc_rows = [proc_hdr]

                # Color map for this process (from evolution plots)
                proc_cm = evo_colors.get(proc, {})

                # Controllable input rows
                if ctrl_indices:
                    for ci in ctrl_indices:
                        lbl = input_labels[ci] if ci < len(input_labels) else f"input_{ci}"
                        dot = _color_dot(proc_cm[lbl]) if lbl in proc_cm else ''
                        t_v = float(t_inputs[ci]) if hasattr(t_inputs, '__len__') and ci < len(t_inputs) else 0.0
                        a_v = float(a_inputs[ci]) if hasattr(a_inputs, '__len__') and ci < len(a_inputs) else 0.0
                        delta = a_v - t_v
                        proc_rows.append([
                            Paragraph(f"{dot}{lbl}",    ST_TRAJ_C),
                            Paragraph(f"{t_v:.4f}",     ST_TRAJ_C),
                            Paragraph(f"{a_v:.4f}",     ST_TRAJ_C),
                            Paragraph(f"{delta:+.4f}",  ST_TRAJ_C),
                            Paragraph("input",          ST_NOTE),
                        ])

                # Output row
                for oi, olbl in enumerate(output_labels):
                    dot = _color_dot(proc_cm[olbl]) if olbl in proc_cm else ''
                    t_v = float(t_outputs[oi]) if hasattr(t_outputs, '__len__') and oi < len(t_outputs) else 0.0
                    b_v = float(b_outputs[oi]) if hasattr(b_outputs, '__len__') and oi < len(b_outputs) else 0.0
                    a_v = float(a_outputs[oi]) if hasattr(a_outputs, '__len__') and oi < len(a_outputs) else 0.0
                    d_bl = b_v - t_v
                    d_ctrl = a_v - t_v
                    proc_rows.append([
                        Paragraph(f"{dot}{olbl}",    ST_TRAJ_C),
                        Paragraph(f"{t_v:.4f}",      ST_TRAJ_C),
                        Paragraph(f"{a_v:.4f}",      ST_TRAJ_C),
                        Paragraph(f"{d_ctrl:+.4f}",  ST_TRAJ_C),
                        Paragraph(f"output (\u0394bl {d_bl:+.4f})", ST_NOTE),
                    ])

                n_data_rows = len(proc_rows) - 1  # minus header

                data_tbl = Table(proc_rows, colWidths=data_cws)
                data_tbl.setStyle(TableStyle([
                    ('LINEBELOW',     (0, 0), (-1,  0), 0.5, C_BLACK),
                    ('LINEBELOW',     (0, 1), (-1, n_data_rows), 0.3, C_LGRAY),
                    ('TOPPADDING',    (0, 0), (-1, -1), 1.5),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5),
                    ('LEFTPADDING',   (0, 0), (-1, -1), 2),
                    ('RIGHTPADDING',  (0, 0), (-1, -1), 2),
                    ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
                ]))

                # Evolution plot for this process (if available)
                evo_key = (sc_idx, proc)
                evo_path = evo_paths.get(evo_key)

                # Image sized to match this process's actual data rows
                img_w = plot_w - 4
                actual_tbl_h = 13 + n_data_rows * 14

                if evo_path and os.path.exists(evo_path):
                    from reportlab.platypus import Image as RLImage
                    evo_img = RLImage(evo_path, width=img_w, height=actual_tbl_h)
                    right_cell = evo_img
                else:
                    right_cell = Paragraph("", ST_NOTE)

                # Process label + side-by-side layout
                F.append(Paragraph(f"<b>{proc}</b>", ST_TRAJ_C))
                side_tbl = Table([[data_tbl, right_cell]],
                                 colWidths=[data_w, plot_w])
                plot_top = 15 if proc_idx == 0 else 0
                side_tbl.setStyle(TableStyle([
                    ('VALIGN',       (0, 0), (-1, -1), 'TOP'),
                    ('LEFTPADDING',  (0, 0), (-1, -1), 0),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                    ('TOPPADDING',   (0, 0), (0,  0),  0),
                    ('TOPPADDING',   (1, 0), (1,  0),  plot_top),
                    ('BOTTOMPADDING',(0, 0), (-1, -1), 0),
                ]))
                F.append(side_tbl)

            F.append(Spacer(1, 2))

        # Footer legend
        foot_p = Paragraph(
            "\u0394 = controller \u2212 target  \u00b7  "
            "\u0394bl = baseline output \u2212 target output  \u00b7  "
            "best of 10 runs per scenario  \u00b7  "
            "plots show X/Y evolution across training epochs", ST_NOTE)
        F.append(foot_p)
        F += _footer(d, total_pages, total_pages)

    return F


def _perf_rows(adv, n_scenarios):
    """Build performance rows from advanced_metrics dict."""
    sr_tr = adv.get('success_rate_train') or {}
    sr_te = adv.get('success_rate_test')  or {}
    wct_d = adv.get('worst_case_gap_train') or adv.get('train_results') or {}
    wce_d = adv.get('worst_case_gap_test')  or adv.get('test_results')  or {}

    _n_sc  = int(n_scenarios) if isinstance(n_scenarios, (int, float)) else 5
    n_sc_tr = sr_tr.get('n_scenarios', sr_tr.get('n_total', _n_sc))
    n_sc_te = sr_te.get('n_scenarios', sr_te.get('n_total', _n_sc))
    ok_tr   = sr_tr.get('n_successful', sr_tr.get('n_success', sr_tr.get('n_above_threshold', 0))) or 0
    ok_te   = sr_te.get('n_successful', sr_te.get('n_success', sr_te.get('n_above_threshold', 0))) or 0
    pct_tr_r = sr_tr.get('success_rate', sr_tr.get('rate',
               ok_tr / n_sc_tr if n_sc_tr else 0))
    pct_te_r = sr_te.get('success_rate', sr_te.get('rate',
               ok_te / n_sc_te if n_sc_te else 0))
    pct_tr  = (float(pct_tr_r) if pct_tr_r is not None else 0.0) * 100
    pct_te  = (float(pct_te_r) if pct_te_r is not None else 0.0) * 100
    wc_tr   = wct_d.get('worst_case_gap', wct_d.get('gap', None))
    wc_te   = wce_d.get('worst_case_gap', wce_d.get('gap', None))
    wc_tr_s = wct_d.get('worst_case_scenario_idx', wct_d.get('worst_case_scenario', ''))
    wc_te_s = wce_d.get('worst_case_scenario_idx', wce_d.get('worst_case_scenario', ''))
    thresh  = adv.get('success_threshold_pct', 95) or 95

    # Formula-based metrics (only when CasualiT surrogate is used)
    fsr_tr = adv.get('formula_success_rate_train') or {}
    fsr_te = adv.get('formula_success_rate_test')  or {}
    fwc_tr = adv.get('formula_worst_case_gap_train') or {}
    fwc_te = adv.get('formula_worst_case_gap_test')  or {}
    has_formula = bool(fsr_tr)

    if has_formula:
        f_ok_tr = fsr_tr.get('n_successful', fsr_tr.get('n_success', 0)) or 0
        f_n_tr  = fsr_tr.get('n_scenarios', fsr_tr.get('n_total', _n_sc))
        f_pct_tr = (float(fsr_tr.get('success_rate', f_ok_tr / f_n_tr if f_n_tr else 0)) or 0.0) * 100
        rows = [
            (f"Win rate vs baseline \u2014 train",
             f"{f_ok_tr}/{f_n_tr} ({f_pct_tr:.1f}%)",
             ST_VAL_G if f_pct_tr > 50 else ST_VAL_R),
            ("\u2514 surrogate",
             f"{ok_tr}/{n_sc_tr} ({pct_tr:.1f}%)",
             ST_VAL_FORM_G if pct_tr > 50 else ST_VAL_FORM_R,
             ST_KEY_FORM),
        ]
    else:
        rows = [
            (f"Win rate vs baseline \u2014 train",
             f"{ok_tr}/{n_sc_tr} ({pct_tr:.1f}%)",
             ST_VAL_G if pct_tr > 50 else ST_VAL_R),
        ]

    if has_formula:
        f_ok_te = fsr_te.get('n_successful', fsr_te.get('n_success', 0)) or 0
        f_n_te  = fsr_te.get('n_scenarios', fsr_te.get('n_total', _n_sc))
        f_pct_te = (float(fsr_te.get('success_rate', f_ok_te / f_n_te if f_n_te else 0)) or 0.0) * 100
        rows.append(
            (f"Win rate vs baseline \u2014 test",
             f"{f_ok_te}/{f_n_te} ({f_pct_te:.1f}%)",
             ST_VAL_G if f_pct_te > 50 else ST_VAL_R))
        rows.append(("\u2514 surrogate",
                     f"{ok_te}/{n_sc_te} ({pct_te:.1f}%)",
                     ST_VAL_FORM_G if pct_te > 50 else ST_VAL_FORM_R,
                     ST_KEY_FORM))
    else:
        rows.append(
            (f"Win rate vs baseline \u2014 test",
             f"{ok_te}/{n_sc_te} ({pct_te:.1f}%)",
             ST_VAL_G if pct_te > 50 else ST_VAL_R))

    if has_formula:
        fwc_tr_v = fwc_tr.get('worst_case_gap', None)
        fwc_tr_s = fwc_tr.get('worst_case_scenario_idx', '')
        rows.append(
            (f"Worst-case gap \u2014 train",
             f"{float(fwc_tr_v):.6f} at sc. {fwc_tr_s}"
             if isinstance(fwc_tr_v, (int, float)) else '\u2014'))
        rows.append(("\u2514 surrogate",
                     f"{float(wc_tr):.6f} at sc. {wc_tr_s}"
                     if isinstance(wc_tr, (int, float)) else '\u2014',
                     ST_VAL_FORM, ST_KEY_FORM))
    else:
        rows.append(
            (f"Worst-case gap \u2014 train",
             f"{float(wc_tr):.6f} at sc. {wc_tr_s}"
             if isinstance(wc_tr, (int, float)) else '\u2014'))

    if has_formula:
        fwc_te_v = fwc_te.get('worst_case_gap', None)
        fwc_te_s = fwc_te.get('worst_case_scenario_idx', '')
        rows.append(
            (f"Worst-case gap \u2014 test",
             f"{float(fwc_te_v):.6f} at sc. {fwc_te_s}"
             if isinstance(fwc_te_v, (int, float)) else '\u2014'))
        rows.append(("\u2514 surrogate",
                     f"{float(wc_te):.6f} at sc. {wc_te_s}"
                     if isinstance(wc_te, (int, float)) else '\u2014',
                     ST_VAL_FORM, ST_KEY_FORM))
    else:
        rows.append(
            (f"Worst-case gap \u2014 test",
             f"{float(wc_te):.6f} at sc. {wc_te_s}"
         if isinstance(wc_te, (int, float)) else '\u2014'))

    return rows


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — training charts (full width)
# ════════════════════════════════════════════════════════════════════════════


def _mini_header(d):
    """Reusable mini-header block for chart pages."""
    cfg    = d['config']
    ts     = d.get('timestamp', datetime.now())
    ts_str = ts.strftime('%Y-%m-%d %H:%M:%S') if isinstance(ts, datetime) else str(ts)
    hist   = d.get('training_history', {})
    seed   = cfg.get('misc', {}).get('random_seed', '\u2014')
    epochs = cfg.get('training', {}).get('epochs',
             hist.get('epochs_trained', '\u2014'))
    max_ep = cfg.get('training', {}).get('max_epochs',
             cfg.get('training', {}).get('epochs', '\u2014'))
    completed  = hist.get('completed', True)
    status_txt = "complete"  if completed else "incomplete"
    status_st  = ST_STATUS_G if completed else ST_STATUS_A
    status_col = C_GREEN     if completed else C_AMBER

    F = []
    title_p = Paragraph("Controller Optimization \u2014 Training Report", ST_TITLE)
    meta_p  = Paragraph(
        f"{ts_str}  \u00b7  seed {seed}  \u00b7  epochs {epochs} / {max_ep}", ST_SUB)
    badge_w = 2.2 * cm
    badge_p = Paragraph(status_txt, status_st)
    hdr_tbl = Table([[title_p, badge_p]], colWidths=[TW - badge_w, badge_w])
    hdr_tbl.setStyle(TableStyle([
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 0),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
        ('TOPPADDING',    (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('BOX',           (1, 0), (1,  0),  0.5, status_col),
        ('ALIGN',         (1, 0), (1,  0),  'CENTER'),
    ]))
    F.append(hdr_tbl)
    F.append(Spacer(1, 1))
    F.append(meta_p)
    F.append(_rule_heavy())
    return F


def _page2(d):
    """Training charts page: training_history + loss_chart stacked, full width."""
    cfg    = d['config']
    chk    = Path(d.get('checkpoint_dir') or cfg.get('training', {}).get('checkpoint_dir', '.'))

    F = _mini_header(d)

    # Available height for plots
    hdr_h  = 1.8 * cm   # mini-header + rule (generous)
    ftr_h  = 1.5 * cm   # footer (rule + table + spacing)
    avail  = PH - 2 * M - hdr_h - ftr_h
    cap_h  = 12          # caption row height allowance

    # ── Full-width: training_history + loss_chart stacked ────────────────
    # 2 images + 2 captions + 1 gap spacer
    h_each = int((avail - 2 * cap_h - 4) / 2)
    charts = img_stack(
        [chk / 'training_history.png',  chk / 'loss_chart.png'],
        ["Training losses & weights \u00b7 reliability evolution",
         "Train vs validation \u2014 overfitting detection"],
        TW, h_each, gap=4)

    F.append(charts)
    return F


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — comparison charts (portrait: stacked vertically)
# ════════════════════════════════════════════════════════════════════════════

def _page3(d):
    """Comparison charts page — portrait layout: paired rows on top,
    theoretical charts stacked below at full width."""
    cfg    = d['config']
    chk    = Path(d.get('checkpoint_dir') or cfg.get('training', {}).get('checkpoint_dir', '.'))

    F = _mini_header(d)

    # Available height for plots
    hdr_h  = 1.8 * cm   # mini-header + rule (generous)
    ftr_h  = 1.5 * cm   # footer (rule + table + spacing)
    avail  = PH - 2 * M - hdr_h - ftr_h

    no_pad = TableStyle([
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 0),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
        ('TOPPADDING',    (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ])

    # ── TOP: baseline/gap paired (2 rows of 2 images, full page width) ──
    pair_w = TW / 2
    pair_h = int(avail * 0.4 / 2)  # ~40% of available height for 2 rows

    def _img_pair(p1, p2, row_h):
        i1 = scale_img(p1, pair_w, row_h)
        i2 = scale_img(p2, pair_w, row_h)
        row = Table([[i1, i2]], colWidths=[pair_w, pair_w])
        row.setStyle(no_pad)
        return row

    row_train = _img_pair(
        chk / 'baseline_vs_controller_train.png',
        chk / 'gap_distribution_train.png', pair_h)
    row_test = _img_pair(
        chk / 'baseline_vs_controller_test.png',
        chk / 'gap_distribution_test.png', pair_h)

    top_block = Table([[row_train], [row_test]], colWidths=[TW])
    top_block.setStyle(no_pad)
    F.append(top_block)
    F.append(Spacer(1, 6))

    # ── BOTTOM: 3 theoretical charts stacked, full width ─────────────────
    n_right   = 3
    inter_gap = 4
    theo_avail = avail * 0.6 - 6  # remaining height minus spacer
    rh = int((theo_avail - (n_right - 1) * inter_gap) / n_right)

    # Compute exact figsize (inches) for theoretical PNGs.
    chart_w_in     = TW / 72.0
    chart_h_in     = rh  / 72.0
    report_figsize = (chart_w_in, chart_h_in)

    # Regenerate the 3 theoretical PNGs at the exact report dimensions.
    theo = d.get('theoretical_data') or {}
    if not (theo.get('epochs') and len(theo.get('epochs', [])) > 0):
        _json_path = chk / 'theoretical_analysis_data.json'
        if _json_path.exists():
            import json as _json
            with open(_json_path) as _f:
                theo = _json.load(_f)

    if theo.get('epochs') and len(theo['epochs']) > 0:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from controller.src.evaluation.analysis.theoretical_visualization import (
            plot_loss_vs_L_min, plot_efficiency_over_time, plot_loss_decomposition,
        )
        bellman_data = theo.get('bellman_lmin', None)
        lambda_grad_data = theo.get('lambda_grad', None)
        lambda_mc_data = theo.get('lambda_mc', None)

        p = chk / 'loss_vs_L_min.png'
        plot_loss_vs_L_min(
            epochs=theo['epochs'],
            observed_loss=theo['observed_loss'],
            theoretical_L_min=theo['theoretical_L_min'],
            save_path=str(p),
            figsize=report_figsize,
            bellman_lmin=bellman_data,
            lambda_grad=lambda_grad_data,
            lambda_mc=lambda_mc_data,
        )
        plt.close()

        p = chk / 'training_efficiency.png'
        plot_efficiency_over_time(
            epochs=theo['epochs'],
            efficiency=theo['efficiency'],
            save_path=str(p),
            figsize=report_figsize,
            bellman_lmin=bellman_data,
            observed_loss=theo['observed_loss'],
            lambda_grad=lambda_grad_data,
            lambda_mc=lambda_mc_data,
        )
        plt.close()

        p = chk / 'loss_decomposition.png'
        plot_loss_decomposition(
            Var_F=theo['theoretical_Var_F'][-1],
            Bias2=theo['theoretical_Bias2'][-1],
            gap=theo['gap'][-1],
            save_path=str(p),
            figsize=report_figsize,
            bellman_lmin=bellman_data,
            lambda_grad=lambda_grad_data,
            lambda_mc=lambda_mc_data,
        )
        plt.close()

    theo_charts = [
        (chk / 'loss_vs_L_min.png',       "Loss vs L_min Bellman"),
        (chk / 'training_efficiency.png',  "Training efficiency"),
        (chk / 'loss_decomposition.png',   "Loss decomposition"),
    ]
    r_rows = []
    for p, _cap in theo_charts:
        img = scale_img_fw(p, TW, rh)
        r_rows.append([img])
    bottom_block = Table(r_rows, colWidths=[TW])
    bottom_block.setStyle(no_pad)

    F.append(bottom_block)
    return F


# ════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════════════════════════

def _footer(d, page_num, total_pages):
    cfg  = d['config']
    chk  = short_dir(str(d.get('checkpoint_dir') or cfg.get('training', {}).get('checkpoint_dir', '')))
    left = Paragraph(f"auto-generated \u2014 {chk}", ST_FTR)
    right = Paragraph(f"page {page_num} / {total_pages}", ST_FTR_R)
    rule = HRFlowable(width=TW, thickness=1, color=C_BLACK,
                      spaceBefore=8, spaceAfter=3)
    tbl = Table([[left, right]], colWidths=[TW * 0.7, TW * 0.3])
    tbl.setStyle(TableStyle([
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 0),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
        ('TOPPADDING',    (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    return [rule, tbl]


# ════════════════════════════════════════════════════════════════════════════
#  BUILD PDF
# ════════════════════════════════════════════════════════════════════════════

def _build_pdf(d, out_path):
    body_frame = Frame(M, M, TW, PH - 2 * M, id='body',
                       leftPadding=0, rightPadding=0,
                       topPadding=0, bottomPadding=0)
    pt  = PageTemplate(id='main', frames=[body_frame], pagesize=A4)
    doc = BaseDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=M, rightMargin=M, topMargin=M, bottomMargin=M,
        pageTemplates=[pt])

    traj = d.get('trajectory_values') or {}
    traj_list = d.get('trajectory_values_list', [])
    has_traj = bool(traj_list or traj)
    total_pages = 4 if has_traj else 3

    story = (
        _page1(d, total_pages) +
        [PageBreak()] +
        _page3(d) + _footer(d, 2, total_pages) +
        [PageBreak()] +
        _page2(d) + _footer(d, 3, total_pages)
    )
    if has_traj:
        story += [PageBreak()] + _page4_trajectory(d, total_pages)
    doc.build(story)


# ════════════════════════════════════════════════════════════════════════════
#  PUBLIC API  — keep existing signatures intact
# ════════════════════════════════════════════════════════════════════════════

def generate_controller_report(
    config,
    training_history,
    final_metrics,
    process_metrics,
    F_star,
    F_baseline,
    F_actual,
    checkpoint_dir,
    timestamp=None,
    n_scenarios=None,
    advanced_metrics=None,
    trajectory_values=None,
    trajectory_values_list=None,
    evolution_plot_paths=None,
    evolution_color_maps=None,
    theoretical_data=None,
    F_formula=None,
):
    if timestamp is None:
        timestamp = datetime.now()
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    out = checkpoint_dir / 'controller_report.pdf'
    d = dict(
        config=config,
        training_history=training_history,
        final_metrics=final_metrics,
        process_metrics=process_metrics,
        F_star=F_star,
        F_baseline=F_baseline,
        F_actual=F_actual,
        timestamp=timestamp,
        n_scenarios=n_scenarios,
        advanced_metrics=advanced_metrics or {},
        trajectory_values=trajectory_values,
        trajectory_values_list=trajectory_values_list or [],
        evolution_plot_paths=evolution_plot_paths or {},
        evolution_color_maps=evolution_color_maps or {},
        theoretical_data=theoretical_data or {},
        checkpoint_dir=checkpoint_dir,
        F_formula=F_formula,
    )
    _build_pdf(d, out)
    return str(out)


class ControllerReportGenerator:
    """Class-based interface — kept for backward compatibility."""

    def __init__(self, output_path):
        self.output_path = Path(output_path)

    def generate(self, config, training_history, final_metrics, process_metrics,
                 F_star, F_baseline, F_actual, timestamp, n_scenarios=None,
                 advanced_metrics=None, trajectory_values=None,
                 trajectory_values_list=None,
                 theoretical_data=None, F_formula=None):
        if timestamp is None:
            timestamp = datetime.now()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        d = dict(
            config=config,
            training_history=training_history,
            final_metrics=final_metrics,
            process_metrics=process_metrics,
            F_star=F_star,
            F_baseline=F_baseline,
            F_actual=F_actual,
            timestamp=timestamp,
            n_scenarios=n_scenarios,
            advanced_metrics=advanced_metrics or {},
            trajectory_values=trajectory_values,
            theoretical_data=theoretical_data or {},
            checkpoint_dir=self.output_path.parent,
            F_formula=F_formula,
        )
        _build_pdf(d, self.output_path)
        return str(self.output_path)


def create_2up_pdf(input_pdf, output_pdf, page_size=None):
    """Stub — kept for backward compatibility."""
    import shutil
    shutil.copy(str(input_pdf), str(output_pdf))
    return str(output_pdf)