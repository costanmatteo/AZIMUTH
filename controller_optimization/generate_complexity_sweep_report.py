"""
generate_complexity_sweep_report.py
------------------------------------
Generates a PDF report for the complexity sweep.

Same design system as generate_sweep_report.py:
  - Courier font throughout
  - Colors: #1D9E75 green | #D85A30 red | #BA7517 amber
  - No colored backgrounds
  - A4 LANDSCAPE, multi-page
  - 0.5 pt borders, 1.1 cm margins

Directory structure expected (auto-detected):
  complexity_sweep/
    config_001__n5_m2_rho0.30_P3/
      seed_t11_b21/
        results.json
      seed_t11_b31/
        results.json
      ...
    config_002__n3_m1_rho0.10_P4/
      ...

Each results.json must contain at least:
  {
    "F_star":     float,
    "F_baseline": float,
    "F_actual":   float,
    "seed_target":   int,
    "seed_baseline": int
  }

Public API (mirrors generate_sweep_report.py):
  generate_complexity_sweep_report(sweep_dir, output_path)
  aggregate_complexity_results(sweep_dir)  -> pd.DataFrame  (one row per run)
  generate_config_stats(df)               -> pd.DataFrame  (one row per config)
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────

PW, PH = landscape(A4)        # 841.89 × 595.28 pt
M      = 1.1 * cm
FULL_W = PW - 2 * M

C_GREEN = colors.HexColor("#1D9E75")
C_RED   = colors.HexColor("#D85A30")
C_AMBER = colors.HexColor("#BA7517")
C_BLACK = colors.HexColor("#111111")
C_MUTED = colors.HexColor("#777777")
C_RULE  = colors.HexColor("#CCCCCC")
C_ALT   = colors.HexColor("#F9F9F9")

FONT    = "Courier"
FONTB   = "Courier-Bold"

# ─────────────────────────────────────────────
#  PARAGRAPH STYLES
# ─────────────────────────────────────────────

def _S(name, font=FONT, size=8, leading=None, color=C_BLACK, bold=False):
    return ParagraphStyle(
        name,
        fontName=FONTB if bold else font,
        fontSize=size,
        leading=leading or size + 2,
        textColor=color,
        spaceAfter=0,
        spaceBefore=0,
    )

S_TITLE    = _S("title",   size=11, bold=True)
S_SUB      = _S("sub",     size=8,  color=C_MUTED)
S_SEC      = _S("sec",     size=7,  bold=True, color=C_BLACK)
S_BODY     = _S("body",    size=8)
S_MUTED    = _S("muted",   size=6.5, color=C_MUTED)
S_CAP      = _S("cap",     size=7,  color=C_MUTED)
S_KV_KEY   = _S("kv_key",  size=8)
S_KV_VAL   = _S("kv_val",  size=8.5, bold=True)
S_TH       = _S("th",      size=7.5, bold=True)
S_TH_DEF   = _S("th_def",  size=6.5, color=C_MUTED)
S_TD       = _S("td",      size=8.5)
S_FOOT     = _S("foot",    size=7,  color=C_MUTED)

def _P(text, style=S_BODY):
    return Paragraph(text, style)

def _HR(thickness=0.5, c=C_RULE):
    return HRFlowable(width=FULL_W, thickness=thickness, color=c, spaceAfter=2, spaceBefore=2)

# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────

def _parse_config_dir(dirname: str) -> dict | None:
    """
    Extract (n_vars, n_stages, rho, n_processes) from a directory name such as:
        config_001__n5_m2_rho0.30_P3
        n5_m2_rho0.30_P3
        config_042
    Returns None if parsing fails.
    """
    m_n   = re.search(r'n(\d+)',      dirname)
    m_m   = re.search(r'm(\d+)',      dirname)
    m_rho = re.search(r'rho([0-9.]+)', dirname)
    m_P   = re.search(r'[Pp](\d+)',   dirname)
    config_id_m = re.search(r'config[_]?(\d+)', dirname)

    n_vars     = int(m_n.group(1))   if m_n   else None
    n_stages   = int(m_m.group(1))   if m_m   else None
    rho        = float(m_rho.group(1)) if m_rho else None
    n_proc     = int(m_P.group(1))   if m_P   else None
    config_id  = int(config_id_m.group(1)) if config_id_m else None

    if any(v is None for v in [n_vars, n_stages, rho, n_proc]):
        return None

    return dict(
        config_id=config_id,
        config_name=dirname,
        n_vars=n_vars,
        n_stages=n_stages,
        rho=rho,
        n_processes=n_proc,
    )


def _load_run_result(run_dir: Path) -> dict | None:
    """Try to load results.json (or results.pkl) from a run directory."""
    # try JSON first
    for fname in ["results.json", "result.json", "eval_results.json"]:
        p = run_dir / fname
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception:
                pass
    # try pickle
    for fname in ["results.pkl", "result.pkl"]:
        p = run_dir / fname
        if p.exists():
            try:
                import pickle
                with open(p, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
    return None


def aggregate_complexity_results(sweep_dir: str | Path) -> pd.DataFrame:
    """
    Walk sweep_dir and build a DataFrame with one row per run.

    Expected structure:
        sweep_dir/
          <config_dir>/
            <seed_dir>/
              results.json
    """
    sweep_dir = Path(sweep_dir)
    rows = []

    for config_dir in sorted(sweep_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        cfg = _parse_config_dir(config_dir.name)
        if cfg is None:
            continue

        for seed_dir in sorted(config_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            result = _load_run_result(seed_dir)
            if result is None:
                continue

            # parse seed from directory name
            m_st = re.search(r't(\d+)', seed_dir.name)
            m_sb = re.search(r'b(\d+)', seed_dir.name)
            seed_t = int(m_st.group(1)) if m_st else result.get("seed_target", -1)
            seed_b = int(m_sb.group(1)) if m_sb else result.get("seed_baseline", -1)

            F_star    = float(result.get("F_star",     result.get("f_star",     np.nan)))
            F_base    = float(result.get("F_baseline", result.get("f_baseline", np.nan)))
            F_actual  = float(result.get("F_actual",   result.get("f_actual",   np.nan)))

            row = dict(
                config_name   = cfg["config_name"],
                config_id     = cfg["config_id"],
                n_vars        = cfg["n_vars"],
                n_stages      = cfg["n_stages"],
                rho           = cfg["rho"],
                n_processes   = cfg["n_processes"],
                seed_target   = seed_t,
                seed_baseline = seed_b,
                run_name      = f"{config_dir.name}__{seed_dir.name}",
                F_star        = F_star,
                F_baseline    = F_base,
                F_actual      = F_actual,
                win           = int(F_actual > F_base) if not (np.isnan(F_actual) or np.isnan(F_base)) else 0,
            )
            row["gap_baseline"] = F_star - F_base   if not np.isnan(F_star - F_base)  else np.nan
            row["gap_ctrl"]     = F_star - F_actual if not np.isnan(F_star - F_actual) else np.nan
            row["gap_delta"]    = F_actual - F_base if not np.isnan(F_actual - F_base) else np.nan
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No run results found in '{sweep_dir}'")

    return pd.DataFrame(rows)


def generate_config_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-config statistics. Returns a DataFrame with one row per config,
    sorted by win_rate descending.
    """
    records = []
    for config_name, g in df.groupby("config_name", sort=False):
        n_runs      = len(g)
        n_wins      = int(g["win"].sum())
        win_rate    = n_wins / n_runs if n_runs > 0 else 0.0

        records.append(dict(
            config_name    = config_name,
            config_id      = g["config_id"].iloc[0],
            n_vars         = g["n_vars"].iloc[0],
            n_stages       = g["n_stages"].iloc[0],
            rho            = g["rho"].iloc[0],
            n_processes    = g["n_processes"].iloc[0],
            n_runs         = n_runs,
            n_wins         = n_wins,
            win_rate       = win_rate,
            F_actual_mean  = g["F_actual"].mean(),
            F_actual_std   = g["F_actual"].std(),
            F_baseline_mean= g["F_baseline"].mean(),
            gap_ctrl_mean  = g["gap_ctrl"].mean(),
            gap_ctrl_min   = g["gap_ctrl"].min(),
            gap_ctrl_max   = g["gap_ctrl"].max(),
            gap_delta_mean = g["gap_delta"].mean(),
            gap_delta_min  = g["gap_delta"].min(),
            gap_delta_max  = g["gap_delta"].max(),
        ))

    cfg_df = pd.DataFrame(records).sort_values("win_rate", ascending=False).reset_index(drop=True)
    return cfg_df


# ─────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────

_PLOT_W = 3.5 * cm
_PLOT_H = 3.5 * cm


def _plot_win_rate_distribution(cfg_df: pd.DataFrame, path: Path):
    """Histogram of per-config win rates."""
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    bins = np.linspace(0, 1, 11)
    ax.hist(cfg_df["win_rate"], bins=bins, color="#111111", edgecolor="white", linewidth=0.4)
    ax.axvline(cfg_df["win_rate"].median(), color="#D85A30", lw=1.2, ls="--", label=f"median={cfg_df['win_rate'].median():.2f}")
    ax.set_xlabel("Win rate W(c)", fontsize=7, fontfamily="monospace")
    ax.set_ylabel("# configs",    fontsize=7, fontfamily="monospace")
    ax.set_title("Win Rate Distribution", fontsize=8, fontfamily="monospace", fontweight="bold")
    ax.legend(fontsize=6, frameon=False)
    ax.tick_params(labelsize=6)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_win_rate_vs_rho(cfg_df: pd.DataFrame, path: Path):
    """Scatter: win rate vs noise rho, coloured by n_vars."""
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    sc = ax.scatter(cfg_df["rho"], cfg_df["win_rate"],
                    c=cfg_df["n_vars"], cmap="viridis",
                    s=22, edgecolors="none", alpha=0.85)
    # trend line
    z = np.polyfit(cfg_df["rho"], cfg_df["win_rate"], 1)
    x_fit = np.linspace(cfg_df["rho"].min(), cfg_df["rho"].max(), 100)
    ax.plot(x_fit, np.poly1d(z)(x_fit), color="#D85A30", lw=1.0, ls="--")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("n_vars", fontsize=6, fontfamily="monospace")
    cbar.ax.tick_params(labelsize=5)
    ax.set_xlabel("Noise ρ",    fontsize=7, fontfamily="monospace")
    ax.set_ylabel("Win rate",   fontsize=7, fontfamily="monospace")
    ax.set_title("Win Rate vs ρ", fontsize=8, fontfamily="monospace", fontweight="bold")
    ax.tick_params(labelsize=6)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_win_rate_vs_n_vars(cfg_df: pd.DataFrame, path: Path):
    """Box plot of win rate grouped by n_vars."""
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    groups = sorted(cfg_df["n_vars"].unique())
    data   = [cfg_df.loc[cfg_df["n_vars"] == g, "win_rate"].values for g in groups]
    bp = ax.boxplot(data, positions=range(len(groups)), widths=0.5,
                    patch_artist=True,
                    medianprops=dict(color="#D85A30", lw=1.5),
                    boxprops=dict(facecolor="#F0F0F0", linewidth=0.7),
                    whiskerprops=dict(linewidth=0.7),
                    capprops=dict(linewidth=0.7),
                    flierprops=dict(marker=".", markersize=3, linestyle="none"))
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([str(g) for g in groups], fontsize=6)
    ax.set_xlabel("n_vars (n)", fontsize=7, fontfamily="monospace")
    ax.set_ylabel("Win rate",   fontsize=7, fontfamily="monospace")
    ax.set_title("Win Rate vs n_vars", fontsize=8, fontfamily="monospace", fontweight="bold")
    ax.tick_params(labelsize=6)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_sensitivity_scatter(cfg_df: pd.DataFrame, path: Path):
    """
    2-panel scatter: win rate vs n_stages and win rate vs n_processes (P).
    Both coloured by rho.
    """
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.8), sharey=True)
    norm = plt.Normalize(cfg_df["rho"].min(), cfg_df["rho"].max())
    cmap = plt.cm.plasma

    for ax, xkey, xlabel in zip(
        axes,
        ["n_stages", "n_processes"],
        ["n_stages (m)", "n_processes (P)"]
    ):
        sc = ax.scatter(cfg_df[xkey], cfg_df["win_rate"],
                        c=cfg_df["rho"], cmap=cmap, norm=norm,
                        s=20, edgecolors="none", alpha=0.9)
        ax.set_xlabel(xlabel, fontsize=7, fontfamily="monospace")
        ax.tick_params(labelsize=6)
        ax.spines[["top","right"]].set_visible(False)

    axes[0].set_ylabel("Win rate", fontsize=7, fontfamily="monospace")
    fig.suptitle("Win Rate vs Complexity Axes", fontsize=8,
                 fontfamily="monospace", fontweight="bold", y=1.01)
    cbar = fig.colorbar(sc, ax=axes, pad=0.02, fraction=0.03)
    cbar.set_label("ρ", fontsize=6, fontfamily="monospace")
    cbar.ax.tick_params(labelsize=5)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_plots(cfg_df: pd.DataFrame, tmp_dir: Path) -> list[Path]:
    paths = [
        tmp_dir / "p1_win_rate_dist.png",
        tmp_dir / "p2_win_rate_rho.png",
        tmp_dir / "p3_win_rate_nvars.png",
        tmp_dir / "p4_sensitivity.png",
    ]
    _plot_win_rate_distribution(cfg_df, paths[0])
    _plot_win_rate_vs_rho(cfg_df, paths[1])
    _plot_win_rate_vs_n_vars(cfg_df, paths[2])
    _plot_sensitivity_scatter(cfg_df, paths[3])
    return paths


# ─────────────────────────────────────────────
#  CELL HELPERS
# ─────────────────────────────────────────────

def _win_color(win_rate: float):
    if win_rate >= 0.9:  return C_GREEN
    if win_rate >= 0.6:  return C_AMBER
    return C_RED

def _gap_delta_color(v: float):
    if v > 0.05: return C_GREEN
    if v > 0:    return C_AMBER
    return C_RED

def _quartile_color(v, q25, q75):
    """Green = best (low gap = close to target), Red = worst."""
    if v <= q25: return C_GREEN
    if v >= q75: return C_RED
    return C_BLACK

def _fmt(v, ndigits=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{ndigits}f}"

def _pct(v):
    if v is None or np.isnan(v):
        return "—"
    return f"{v*100:.1f}%"


# ─────────────────────────────────────────────
#  PAGE 1 BUILDING BLOCKS
# ─────────────────────────────────────────────

def _build_header(sweep_dir: str, df: pd.DataFrame, cfg_df: pd.DataFrame) -> list:
    """Title + subtitle row."""
    n_runs    = len(df)
    n_configs = len(cfg_df)
    ts        = datetime.now().strftime("%Y-%m-%d  %H:%M")
    short_dir = str(Path(sweep_dir).name)

    hdr_data = [[
        _P("Controller Complexity Sweep — Report", S_TITLE),
        _P(f"{ts}  ·  {short_dir}  ·  {n_configs} configs  ·  {n_runs} runs", S_SUB),
    ]]
    tbl = Table(hdr_data, colWidths=[FULL_W * 0.5, FULL_W * 0.5])
    tbl.setStyle(TableStyle([
        ("VALIGN",  (0,0), (-1,-1), "BOTTOM"),
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ("TOPPADDING",   (0,0), (-1,-1), 0),
        ("BOTTOMPADDING",(0,0), (-1,-1), 2),
    ]))
    return [tbl, HRFlowable(width=FULL_W, thickness=1, color=C_BLACK, spaceAfter=4, spaceBefore=2)]


def _build_kpi_bar(df: pd.DataFrame, cfg_df: pd.DataFrame) -> list:
    """4-cell KPI bar."""
    n_runs    = len(df)
    n_configs = len(cfg_df)
    n_wins    = int(df["win"].sum())
    win_rate  = n_wins / n_runs if n_runs > 0 else 0.0

    median_wr      = cfg_df["win_rate"].median()
    best_cfg       = cfg_df.iloc[0]
    best_wr        = best_cfg["win_rate"]
    best_name_raw  = best_cfg["config_name"]
    # truncate long names
    best_name = best_name_raw if len(best_name_raw) <= 30 else "…" + best_name_raw[-28:]

    # KPI cell builders
    def kpi(top_label, top_val, top_color, bot_label, bot_val=None):
        val_str = _P(f'<font color="#{top_color.hexval()[2:]}">{top_val}</font>', S_KV_VAL)
        rows = [
            [_P(top_label, S_KV_KEY)],
            [val_str],
            [_P(bot_label, S_MUTED)],
        ]
        if bot_val is not None:
            rows.append([_P(bot_val, S_MUTED)])
        return rows

    wc = _win_color(win_rate)
    mc = _win_color(median_wr)
    bc = _win_color(best_wr)

    cells = [
        [
            _P("Total runs", S_KV_KEY),
            _P(str(n_runs), S_KV_VAL),
            _P(f"{n_configs} LHS configurations", S_MUTED),
        ],
        [
            _P("Overall win rate", S_KV_KEY),
            Paragraph(f'<font color="#{wc.hexval()[2:]}">{n_wins}/{n_runs}  ({_pct(win_rate)})</font>', S_KV_VAL),
            _P("runs where F > F' (baseline)", S_MUTED),
        ],
        [
            _P("Median config win rate", S_KV_KEY),
            Paragraph(f'<font color="#{mc.hexval()[2:]}">{_pct(median_wr)}</font>', S_KV_VAL),
            _P("W(c) across all configs", S_MUTED),
        ],
        [
            _P("Best config", S_KV_KEY),
            Paragraph(f'<font color="#{bc.hexval()[2:]}">{_pct(best_wr)}</font>', S_KV_VAL),
            _P(best_name, S_MUTED),
        ],
    ]

    cw = FULL_W / 4
    tbl = Table([cells], colWidths=[cw, cw, cw, cw], rowHeights=None)
    tbl.setStyle(TableStyle([
        ("BOX",        (0,0), (-1,-1), 0.5, C_BLACK),
        ("INNERGRID",  (0,0), (-1,-1), 0.5, C_BLACK),
        ("VALIGN",     (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0),(-1,-1), 5),
        ("RIGHTPADDING",(0,0),(-1,-1), 5),
        ("TOPPADDING",  (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ]))
    return [tbl, Spacer(1, 6)]


def _build_aggregate_stats(cfg_df: pd.DataFrame) -> list:
    """Section 01 — aggregate statistics (3-column layout)."""
    story = [
        _P("01 — aggregate statistics", S_SEC),
        _HR(),
    ]

    wr = cfg_df["win_rate"]
    gc = cfg_df["gap_ctrl_mean"]
    gd = cfg_df["gap_delta_mean"]

    # ── Col 1: Win Rate stats ──
    def stat_block(title, definition, series, color_fn=None):
        q25, q75 = series.quantile(0.25), series.quantile(0.75)
        rows = []
        for label, val in [("Min", series.min()), ("Median", series.median()),
                            ("Max", series.max()), ("Std dev", series.std())]:
            if color_fn:
                c = color_fn(val, q25, q75)
                vs = Paragraph(f'<font color="#{c.hexval()[2:]}">{_fmt(val)}</font>', S_BODY)
            else:
                vs = _P(_fmt(val), S_BODY)
            rows.append([_P(label, S_MUTED), vs])
        inner = Table(rows, colWidths=[40, 50])
        inner.setStyle(TableStyle([
            ("LEFTPADDING",  (0,0),(-1,-1), 0),
            ("RIGHTPADDING", (0,0),(-1,-1), 4),
            ("TOPPADDING",   (0,0),(-1,-1), 1),
            ("BOTTOMPADDING",(0,0),(-1,-1), 1),
        ]))
        return [
            _P(f"<b>{title}</b>", _S("blk", size=7.5, bold=True)),
            _P(definition, S_MUTED),
            inner,
        ]

    col1 = stat_block("Win Rate  W(c)", "fraction of wins per config",
                      wr, color_fn=lambda v, *_: _win_color(v))
    col2 = stat_block("Gap ctrl  (mean)",  "F* − F  averaged over seeds",
                      gc, color_fn=_quartile_color)
    col3 = stat_block("Gap Δ  (mean)", "F − F'  averaged over seeds",
                      gd, color_fn=lambda v, *_: _gap_delta_color(v))

    # ── Col 4: LHS parameter ranges ──
    col4_rows = [
        [_P("n_vars",    S_MUTED), _P(f"{int(cfg_df['n_vars'].min())}–{int(cfg_df['n_vars'].max())}",    S_BODY)],
        [_P("n_stages",  S_MUTED), _P(f"{int(cfg_df['n_stages'].min())}–{int(cfg_df['n_stages'].max())}",  S_BODY)],
        [_P("rho",       S_MUTED), _P(f"{cfg_df['rho'].min():.2f}–{cfg_df['rho'].max():.2f}",             S_BODY)],
        [_P("n_proc.",   S_MUTED), _P(f"{int(cfg_df['n_processes'].min())}–{int(cfg_df['n_processes'].max())}",S_BODY)],
        [_P("configs",   S_MUTED), _P(str(len(cfg_df)), S_BODY)],
        [_P("runs/cfg",  S_MUTED), _P(str(int(cfg_df["n_runs"].median())), S_BODY)],
    ]
    col4_inner = Table(col4_rows, colWidths=[45, 50])
    col4_inner.setStyle(TableStyle([
        ("LEFTPADDING",  (0,0),(-1,-1), 0),
        ("RIGHTPADDING", (0,0),(-1,-1), 4),
        ("TOPPADDING",   (0,0),(-1,-1), 1),
        ("BOTTOMPADDING",(0,0),(-1,-1), 1),
    ]))
    col4 = [_P("<b>LHS parameter ranges</b>", _S("blk2", size=7.5, bold=True)),
            _P("complexity space θ = (n, m, ρ, P)", S_MUTED),
            col4_inner]

    def wrap_col(items):
        return [[item] for item in items]

    cw = FULL_W / 4
    grid_data = [[col1, col2, col3, col4]]

    # flatten each col to a single cell using a sub-Table
    def col_cell(items):
        t = Table([[i] for i in items], colWidths=[cw - 8])
        t.setStyle(TableStyle([
            ("LEFTPADDING",  (0,0),(-1,-1), 4),
            ("RIGHTPADDING", (0,0),(-1,-1), 0),
            ("TOPPADDING",   (0,0),(-1,-1), 2),
            ("BOTTOMPADDING",(0,0),(-1,-1), 1),
        ]))
        return t

    outer = Table(
        [[col_cell(col1), col_cell(col2), col_cell(col3), col_cell(col4)]],
        colWidths=[cw, cw, cw, cw],
    )
    outer.setStyle(TableStyle([
        ("BOX",      (0,0), (-1,-1), 0.5, C_BLACK),
        ("INNERGRID",(0,0), (-1,-1), 0.5, C_BLACK),
        ("VALIGN",   (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",   (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING",  (0,0),(-1,-1), 0),
        ("RIGHTPADDING", (0,0),(-1,-1), 0),
    ]))
    story.append(outer)
    story.append(Spacer(1, 6))
    return story


def _build_plots_row(plot_paths: list[Path]) -> list:
    """Section 02 — 4 plots in a grid row."""
    story = [
        _P("02 — visualizations", S_SEC),
        _HR(),
    ]

    captions = [
        "Win rate histogram across all LHS configs",
        "Win rate vs noise ρ  (colour = n_vars)",
        "Win rate distribution grouped by n_vars",
        "Win rate vs n_stages and P  (colour = ρ)",
    ]

    plot_w = FULL_W / 4 - 4
    plot_h = plot_w * 0.75

    cells = []
    for p, cap in zip(plot_paths, captions):
        img = Image(str(p), width=plot_w, height=plot_h)
        cells.append([img, _P(cap, S_CAP)])

    tbl = Table([cells], colWidths=[FULL_W / 4] * 4)
    tbl.setStyle(TableStyle([
        ("VALIGN",       (0,0),(-1,-1), "TOP"),
        ("LEFTPADDING",  (0,0),(-1,-1), 2),
        ("RIGHTPADDING", (0,0),(-1,-1), 2),
        ("TOPPADDING",   (0,0),(-1,-1), 0),
        ("BOTTOMPADDING",(0,0),(-1,-1), 0),
    ]))
    story.append(tbl)
    return story


def _build_footer(sweep_dir: str, output_path: str) -> list:
    out_name = Path(output_path).name
    short    = Path(sweep_dir).name
    data = [[
        _P(f"auto-generated  ·  {short}  ·  {out_name}", S_FOOT),
        _P("controller_optimization · generate_complexity_sweep_report.py", S_FOOT),
    ]]
    tbl = Table(data, colWidths=[FULL_W * 0.6, FULL_W * 0.4])
    tbl.setStyle(TableStyle([
        ("ALIGN",        (1,0), (1,0), "RIGHT"),
        ("LEFTPADDING",  (0,0),(-1,-1), 0),
        ("RIGHTPADDING", (0,0),(-1,-1), 0),
        ("TOPPADDING",   (0,0),(-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 0),
    ]))
    return [HRFlowable(width=FULL_W, thickness=1, color=C_BLACK, spaceBefore=4, spaceAfter=2), tbl]


# ─────────────────────────────────────────────
#  PAGE 2…N — CONFIGS TABLE
# ─────────────────────────────────────────────

def _th(text, definition=""):
    """Two-line table header: bold label + muted definition."""
    return [_P(f"<b>{text}</b>", S_TH), _P(definition, S_TH_DEF)]


def _build_configs_table_header_compact(sweep_dir: str, page: int) -> list:
    data = [[
        _P("Controller Complexity Sweep — Configs Table", _S("ct", size=9, bold=True)),
        _P(f"page {page}  ·  {Path(sweep_dir).name}", S_MUTED),
    ]]
    tbl = Table(data, colWidths=[FULL_W * 0.5, FULL_W * 0.5])
    tbl.setStyle(TableStyle([
        ("VALIGN",       (0,0),(-1,-1), "BOTTOM"),
        ("LEFTPADDING",  (0,0),(-1,-1), 0),
        ("RIGHTPADDING", (0,0),(-1,-1), 0),
        ("TOPPADDING",   (0,0),(-1,-1), 0),
        ("BOTTOMPADDING",(0,0),(-1,-1), 2),
    ]))
    return [tbl, HRFlowable(width=FULL_W, thickness=1, color=C_BLACK, spaceAfter=4, spaceBefore=2)]


def _build_configs_table(cfg_df: pd.DataFrame) -> list:
    """
    Multi-page table of per-config results, sorted by win_rate desc.
    Columns: rank, config, n_vars, n_stages, ρ, P, runs, wins,
             win_rate, gap_ctrl_mean, gap_delta_mean, F_actual_mean ± std
    """
    # compute quartiles for gap_ctrl coloring
    q25 = cfg_df["gap_ctrl_mean"].quantile(0.25)
    q75 = cfg_df["gap_ctrl_mean"].quantile(0.75)

    # Column widths (must sum to FULL_W)
    col_w = [
        20,  # rank
        135, # config name
        28,  # n
        28,  # m
        30,  # rho
        28,  # P
        28,  # runs
        28,  # wins
        48,  # win_rate
        55,  # gap_ctrl_mean
        55,  # gap_delta_mean
        72,  # F_actual mean ± std
    ]
    assert abs(sum(col_w) - FULL_W) < 2, f"col widths sum={sum(col_w)} vs FULL_W={FULL_W:.1f}"

    header = [
        _th("#"),
        _th("Config"),
        _th("n"),
        _th("m"),
        _th("ρ"),
        _th("P"),
        _th("runs"),
        _th("wins"),
        _th("Win rate", "W(c)"),
        _th("Gap ctrl", "F*−F  mean"),
        _th("Gap Δ",    "F−F'  mean"),
        _th("F actual", "mean ± std"),
    ]

    rows = [header]
    ts   = []

    for i, (_, row) in enumerate(cfg_df.iterrows()):
        wr   = row["win_rate"]
        gc   = row["gap_ctrl_mean"]
        gd   = row["gap_delta_mean"]
        fa   = row["F_actual_mean"]
        fstd = row["F_actual_std"]

        wc  = _win_color(wr)
        gcc = _quartile_color(gc, q25, q75)
        gdc = _gap_delta_color(gd)

        name_raw  = row["config_name"]
        name_disp = name_raw if len(name_raw) <= 32 else "…" + name_raw[-30:]

        bg = C_ALT if i % 2 == 1 else colors.white

        rows.append([
            _P(str(i+1),                                                      S_TD),
            _P(name_disp,                                                     S_TD),
            _P(str(int(row["n_vars"])),                                       S_TD),
            _P(str(int(row["n_stages"])),                                     S_TD),
            _P(f"{row['rho']:.2f}",                                           S_TD),
            _P(str(int(row["n_processes"])),                                  S_TD),
            _P(str(int(row["n_runs"])),                                       S_TD),
            _P(str(int(row["n_wins"])),                                       S_TD),
            Paragraph(f'<font color="#{wc.hexval()[2:]}">{_pct(wr)}</font>',  S_TD),
            Paragraph(f'<font color="#{gcc.hexval()[2:]}">{_fmt(gc)}</font>', S_TD),
            Paragraph(f'<font color="#{gdc.hexval()[2:]}">{_fmt(gd)}</font>', S_TD),
            _P(f"{_fmt(fa)} ± {_fmt(fstd, 4)}",                              S_TD),
        ])
        ts.append(("BACKGROUND", (0, i+1), (-1, i+1), bg))

    base_style = [
        ("BOX",            (0,0), (-1,-1), 0.5, C_BLACK),
        ("INNERGRID",      (0,0), (-1,-1), 0.3, C_RULE),
        ("LINEBELOW",      (0,0), (-1,0),  0.5, C_BLACK),
        ("VALIGN",         (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING",    (0,0), (-1,-1), 3),
        ("RIGHTPADDING",   (0,0), (-1,-1), 3),
        ("TOPPADDING",     (0,0), (-1,-1), 2),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 2),
    ] + ts

    tbl = Table(rows, colWidths=col_w, repeatRows=1)
    tbl.setStyle(TableStyle(base_style))

    footer_note = _P(
        "Win rate: W(c) = fraction of seed pairs where F > F'  ·  "
        "Gap ctrl: F*−F (smaller = better)  ·  Gap Δ: F−F' (positive = controller beats baseline)  ·  "
        "green = best quartile  ·  red = worst quartile",
        S_MUTED,
    )

    return [tbl, Spacer(1, 4), footer_note]


# ─────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────

def generate_complexity_sweep_report(sweep_dir: str | Path, output_path: str | Path) -> None:
    """
    Generate the complexity sweep PDF report.

    Parameters
    ----------
    sweep_dir   : path to the complexity sweep root directory
    output_path : destination path for the output PDF
    """
    sweep_dir   = str(sweep_dir)
    output_path = str(output_path)

    print(f"[complexity_sweep_report] Loading results from '{sweep_dir}' …")
    df     = aggregate_complexity_results(sweep_dir)
    cfg_df = generate_config_stats(df)
    print(f"  → {len(df)} runs across {len(cfg_df)} configs")

    # generate plots in a temp dir
    import tempfile, shutil
    tmp_dir = Path(tempfile.mkdtemp(prefix="complexity_report_"))
    try:
        plot_paths = _make_plots(cfg_df, tmp_dir)

        doc = SimpleDocTemplate(
            output_path,
            pagesize=landscape(A4),
            leftMargin=M, rightMargin=M,
            topMargin=M, bottomMargin=M,
        )

        story: list = []

        # ── PAGE 1 ──
        story += _build_header(sweep_dir, df, cfg_df)
        story += _build_kpi_bar(df, cfg_df)
        story += _build_aggregate_stats(cfg_df)
        story += _build_plots_row(plot_paths)
        story += _build_footer(sweep_dir, output_path)

        # ── PAGE 2…N ──
        story.append(PageBreak())
        story += _build_configs_table_header_compact(sweep_dir, page=2)
        story += _build_configs_table(cfg_df)
        story += _build_footer(sweep_dir, output_path)

        doc.build(story)
        print(f"[complexity_sweep_report] Report saved → '{output_path}'")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate complexity sweep report PDF")
    parser.add_argument("sweep_dir",   help="Path to the complexity sweep directory")
    parser.add_argument("output_path", nargs="?",
                        help="Output PDF path (default: <sweep_dir>/complexity_sweep_report.pdf)")
    args = parser.parse_args()

    out = args.output_path or str(Path(args.sweep_dir) / "complexity_sweep_report.pdf")
    generate_complexity_sweep_report(args.sweep_dir, out)