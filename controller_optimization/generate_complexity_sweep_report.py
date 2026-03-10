#!/usr/bin/env python3
"""
Generate report for dataset complexity sensitivity analysis.

Aggregates results from complexity sweep and shows how controller win rate
varies with dataset complexity parameters (n, m, rho).

Usage:
    python generate_complexity_sweep_report.py [--sweep_dir PATH] [--output FILE]
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus.flowables import HRFlowable


def load_run_results(run_dir: Path) -> dict:
    """Load results from a single complexity sweep run."""
    results_file = run_dir / "final_results.json"
    if not results_file.exists():
        return None

    try:
        with open(results_file, 'r') as f:
            data = json.load(f)

        config = data.get('config', {})
        scenarios_config = config.get('scenarios', {})
        st_params = data.get('st_params', {})

        return {
            'run_name': run_dir.name,
            'seed_target': scenarios_config.get('seed_target'),
            'seed_baseline': scenarios_config.get('seed_baseline'),
            # ST complexity parameters
            'st_n': st_params.get('n') if st_params else None,
            'st_m': st_params.get('m') if st_params else None,
            'st_rho': st_params.get('rho') if st_params else None,
            'n_processes': data.get('n_processes'),
            # Train metrics
            'F_star_train': data.get('train', {}).get('F_star'),
            'F_baseline_train': data.get('train', {}).get('F_baseline_mean'),
            'F_actual_train': data.get('train', {}).get('F_actual_mean'),
            # Test metrics
            'F_star_test': data.get('test', {}).get('F_star'),
            'F_baseline_test': data.get('test', {}).get('F_baseline_mean'),
            'F_actual_test': data.get('test', {}).get('F_actual_mean'),
        }
    except Exception as e:
        print(f"  Warning: error loading {results_file}: {e}")
        return None


def aggregate_results(sweep_dir: Path) -> pd.DataFrame:
    """Load all results from complexity sweep directory."""
    results = []
    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_results = load_run_results(run_dir)
        if run_results is not None:
            results.append(run_results)

    if not results:
        print("No results found!")
        return pd.DataFrame()

    print(f"Loaded {len(results)} runs")
    return pd.DataFrame(results)


def compute_win_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group runs by (st_n, st_m, st_rho) configuration and compute win rate.

    A "win" is when the controller gap (F* - F) is smaller than the baseline gap (F* - F').

    Returns:
        DataFrame with one row per configuration, including win_rate_pct and n_runs.
    """
    # Controller wins when F_actual > F_baseline (closer to F*)
    df = df.copy()
    df['controller_wins'] = df['F_actual_train'] > df['F_baseline_train']

    # Group by complexity configuration
    group_cols = ['st_n', 'st_m', 'st_rho']
    if 'n_processes' in df.columns and df['n_processes'].notna().any():
        group_cols.append('n_processes')
    grouped = df.groupby(group_cols).agg(
        n_runs=('controller_wins', 'count'),
        n_wins=('controller_wins', 'sum'),
        F_star_mean=('F_star_train', 'mean'),
        F_baseline_mean=('F_baseline_train', 'mean'),
        F_actual_mean=('F_actual_train', 'mean'),
        gap_baseline_mean=('F_baseline_train', lambda x: (df.loc[x.index, 'F_star_train'] - x).mean()),
        gap_controller_mean=('F_actual_train', lambda x: (df.loc[x.index, 'F_star_train'] - x).mean()),
    ).reset_index()

    grouped['win_rate_pct'] = 100.0 * grouped['n_wins'] / grouped['n_runs']
    grouped['mean_improvement_pct'] = (
        (grouped['F_actual_mean'] - grouped['F_baseline_mean'])
        / grouped['F_baseline_mean'].abs().clip(lower=1e-10) * 100
    )

    return grouped


def plot_win_rate_vs_param(win_df: pd.DataFrame, param: str, label: str,
                           save_path: Path):
    """Scatter plot of win rate vs a single complexity parameter."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Color by win rate
    scatter = ax.scatter(
        win_df[param], win_df['win_rate_pct'],
        c=win_df['win_rate_pct'], cmap='RdYlGn', vmin=0, vmax=100,
        s=80 + win_df['n_runs'] * 5,  # size ~ number of runs
        edgecolors='black', linewidths=0.5, alpha=0.8,
    )
    plt.colorbar(scatter, ax=ax, label='Win Rate (%)')

    # Trend line (LOWESS or polynomial)
    if len(win_df) >= 5:
        z = np.polyfit(win_df[param], win_df['win_rate_pct'], deg=2)
        p = np.poly1d(z)
        x_smooth = np.linspace(win_df[param].min(), win_df[param].max(), 100)
        ax.plot(x_smooth, np.clip(p(x_smooth), 0, 100), 'b--', alpha=0.5,
                linewidth=2, label='Quadratic trend')
        ax.legend()

    ax.set_xlabel(label, fontsize=12, fontweight='bold')
    ax.set_ylabel('Controller Win Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Win Rate vs {label}', fontsize=14, fontweight='bold')
    ax.set_ylim(-5, 105)
    ax.axhline(50, color='gray', linestyle=':', alpha=0.5, label='50%')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def plot_win_rate_heatmap_2d(win_df: pd.DataFrame, param_x: str, param_y: str,
                              label_x: str, label_y: str, save_path: Path):
    """2D heatmap of win rate for two parameters (averaging over the third)."""
    fig, ax = plt.subplots(figsize=(9, 7))

    # Bin the continuous parameters for heatmap
    n_bins = min(8, len(win_df[param_x].unique()))
    win_df = win_df.copy()
    win_df['x_bin'] = pd.cut(win_df[param_x], bins=n_bins, include_lowest=True)
    win_df['y_bin'] = pd.cut(win_df[param_y], bins=n_bins, include_lowest=True)

    pivot = win_df.pivot_table(
        values='win_rate_pct', index='y_bin', columns='x_bin', aggfunc='mean'
    )

    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        ax.text(0.5, 0.5, 'Not enough data for heatmap',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path

    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100,
                   origin='lower')
    plt.colorbar(im, ax=ax, label='Win Rate (%)')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index], fontsize=8)

    # Annotate values
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = 'white' if val < 30 or val > 70 else 'black'
                ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                        color=color, fontsize=9, fontweight='bold')

    ax.set_xlabel(label_x, fontsize=12, fontweight='bold')
    ax.set_ylabel(label_y, fontsize=12, fontweight='bold')
    ax.set_title(f'Win Rate: {label_x} vs {label_y}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def plot_3d_scatter(win_df: pd.DataFrame, save_path: Path):
    """3D scatter plot of win rate colored by (n, m, rho)."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        win_df['st_n'], win_df['st_m'], win_df['st_rho'],
        c=win_df['win_rate_pct'], cmap='RdYlGn', vmin=0, vmax=100,
        s=60 + win_df['n_runs'] * 3, alpha=0.8, edgecolors='black', linewidths=0.3,
    )

    fig.colorbar(scatter, ax=ax, label='Win Rate (%)', shrink=0.6)
    ax.set_xlabel('n (inputs)', fontsize=10)
    ax.set_ylabel('m (stages)', fontsize=10)
    ax.set_zlabel('rho (noise)', fontsize=10)
    ax.set_title('Win Rate in Complexity Space (n, m, rho)', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def plot_summary_panel(win_df: pd.DataFrame, save_path: Path):
    """Summary panel with marginal win rate distributions for each parameter."""
    has_nproc = 'n_processes' in win_df.columns and win_df['n_processes'].notna().any()
    n_panels = 4 if has_nproc else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5))

    params = [
        ('st_n', 'n (input variables)', axes[0]),
        ('st_m', 'm (cascaded stages)', axes[1]),
        ('st_rho', 'rho (noise intensity)', axes[2]),
    ]
    if has_nproc:
        params.append(('n_processes', 'n_processes (chain length)', axes[3]))

    for param, label, ax in params:
        # Sort by parameter value
        df_sorted = win_df.sort_values(param)

        # Bar / scatter depending on parameter type
        if param in ('st_n', 'st_m', 'n_processes'):
            # Discrete: group and average
            grouped = df_sorted.groupby(param)['win_rate_pct'].agg(['mean', 'std', 'count'])
            x = grouped.index.values
            y = grouped['mean'].values
            yerr = grouped['std'].values / np.sqrt(grouped['count'].values)

            bars = ax.bar(x, y, yerr=yerr, color='steelblue', edgecolor='black',
                          alpha=0.7, capsize=3, width=0.6)

            # Color bars by value
            norm = plt.Normalize(0, 100)
            cmap = plt.cm.RdYlGn
            for bar, val in zip(bars, y):
                bar.set_facecolor(cmap(norm(val)))
        else:
            # Continuous: scatter with trend
            ax.scatter(df_sorted[param], df_sorted['win_rate_pct'],
                       c=df_sorted['win_rate_pct'], cmap='RdYlGn', vmin=0, vmax=100,
                       s=60, edgecolors='black', linewidths=0.5, alpha=0.8)
            if len(df_sorted) >= 5:
                z = np.polyfit(df_sorted[param], df_sorted['win_rate_pct'], deg=2)
                p = np.poly1d(z)
                x_smooth = np.linspace(df_sorted[param].min(), df_sorted[param].max(), 100)
                ax.plot(x_smooth, np.clip(p(x_smooth), 0, 100), 'b--', alpha=0.6, linewidth=2)

        ax.set_xlabel(label, fontsize=11, fontweight='bold')
        ax.set_ylabel('Win Rate (%)', fontsize=11, fontweight='bold')
        ax.set_ylim(-5, 105)
        ax.axhline(50, color='gray', linestyle=':', alpha=0.5)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Controller Win Rate vs Dataset Complexity Parameters',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


class ComplexitySweepReportGenerator:
    """PDF report generator for complexity sweep results."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.styles = getSampleStyleSheet()
        self.story = []

        self.styles.add(ParagraphStyle(
            name='ReportTitle', parent=self.styles['Heading1'],
            fontSize=18, leading=22, alignment=TA_CENTER, spaceAfter=6))
        self.styles.add(ParagraphStyle(
            name='ReportSubtitle', parent=self.styles['Normal'],
            fontSize=11, leading=14, alignment=TA_CENTER, spaceAfter=12))
        self.styles.add(ParagraphStyle(
            name='SectionTitle', parent=self.styles['Heading2'],
            fontSize=12, leading=14, fontName='Helvetica-Bold',
            spaceAfter=6, spaceBefore=12))
        self.styles['BodyText'].fontSize = 9
        self.styles['BodyText'].leading = 11

    def add_title(self, timestamp: datetime):
        self.story.append(Paragraph(
            "<b>Dataset Complexity Sensitivity Analysis</b>", self.styles['ReportTitle']))
        self.story.append(Paragraph(
            f"Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}", self.styles['ReportSubtitle']))
        self.story.append(HRFlowable(
            width="100%", thickness=2, color=colors.black, spaceAfter=12))

    def add_section(self, title: str):
        self.story.append(Paragraph(f"<b>{title}</b>", self.styles['SectionTitle']))
        self.story.append(HRFlowable(
            width="100%", thickness=1, color=colors.darkgray, spaceAfter=6))

    def add_image(self, image_path: Path, width: float = 6*inch, caption: str = None):
        if image_path.exists():
            img = Image(str(image_path), width=width, height=width*0.6)
            self.story.append(img)
            if caption:
                self.story.append(Paragraph(f"<i>{caption}</i>", self.styles['BodyText']))
            self.story.append(Spacer(1, 0.3*cm))

    def add_summary_table(self, df_all: pd.DataFrame, win_df: pd.DataFrame):
        self.add_section("Summary Statistics")

        overall_win_rate = 100 * (df_all['F_actual_train'] > df_all['F_baseline_train']).mean()

        has_nproc = 'n_processes' in win_df.columns and win_df['n_processes'].notna().any()

        data = [
            ['Metric', 'Value'],
            ['Total Runs', f"{len(df_all)}"],
            ['Unique Configurations', f"{len(win_df)}"],
            ['Overall Win Rate', f"{overall_win_rate:.1f}%"],
            ['', ''],
            ['Parameter Ranges', ''],
            ['  n (inputs)', f"[{win_df['st_n'].min()}, {win_df['st_n'].max()}]"],
            ['  m (stages)', f"[{win_df['st_m'].min()}, {win_df['st_m'].max()}]"],
            ['  rho (noise)', f"[{win_df['st_rho'].min():.3f}, {win_df['st_rho'].max():.3f}]"],
        ]
        if has_nproc:
            data.append(['  n_processes', f"[{int(win_df['n_processes'].min())}, {int(win_df['n_processes'].max())}]"])
        data += [
            ['', ''],
            ['Win Rate Range', f"[{win_df['win_rate_pct'].min():.1f}%, {win_df['win_rate_pct'].max():.1f}%]"],
            ['Win Rate Median', f"{win_df['win_rate_pct'].median():.1f}%"],
            ['', ''],
            ['Best Config', ''],
        ]

        best = win_df.loc[win_df['win_rate_pct'].idxmax()]
        worst = win_df.loc[win_df['win_rate_pct'].idxmin()]
        nproc_best = f", n_proc={int(best['n_processes'])}" if has_nproc else ""
        nproc_worst = f", n_proc={int(worst['n_processes'])}" if has_nproc else ""
        data.append(['  Best', f"n={int(best['st_n'])}, m={int(best['st_m'])}, rho={best['st_rho']:.3f}{nproc_best} -> {best['win_rate_pct']:.1f}%"])
        data.append(['  Worst', f"n={int(worst['st_n'])}, m={int(worst['st_m'])}, rho={worst['st_rho']:.3f}{nproc_worst} -> {worst['win_rate_pct']:.1f}%"])

        table = Table(data, colWidths=[2.5*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        self.story.append(table)
        self.story.append(Spacer(1, 0.5*cm))

    def add_configs_table(self, win_df: pd.DataFrame):
        self.add_section("All Configurations (sorted by win rate)")
        df_sorted = win_df.sort_values('win_rate_pct', ascending=False)

        has_nproc = 'n_processes' in win_df.columns and win_df['n_processes'].notna().any()

        header = ['n', 'm', 'rho']
        if has_nproc:
            header.append('nProc')
        header += ['Runs', 'Win Rate', 'F* (mean)', 'F\' (mean)', 'F (mean)']
        data = [header]

        for _, row in df_sorted.iterrows():
            row_data = [
                int(row['st_n']),
                int(row['st_m']),
                f"{row['st_rho']:.3f}",
            ]
            if has_nproc:
                row_data.append(int(row['n_processes']) if not np.isnan(row.get('n_processes', float('nan'))) else 'N/A')
            row_data += [
                int(row['n_runs']),
                f"{row['win_rate_pct']:.1f}%",
                f"{row['F_star_mean']:.4f}" if not np.isnan(row['F_star_mean']) else 'N/A',
                f"{row['F_baseline_mean']:.4f}" if not np.isnan(row['F_baseline_mean']) else 'N/A',
                f"{row['F_actual_mean']:.4f}" if not np.isnan(row['F_actual_mean']) else 'N/A',
            ]
            data.append(row_data)

        col_widths = [0.5*inch, 0.5*inch, 0.7*inch]
        if has_nproc:
            col_widths.append(0.5*inch)
        col_widths += [0.5*inch, 0.8*inch, 0.9*inch, 0.9*inch, 0.9*inch]
        table = Table(data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 0.95, 0.95)]),
        ]))
        self.story.append(table)

    def build(self):
        doc = SimpleDocTemplate(
            str(self.output_path), pagesize=A4,
            rightMargin=1*cm, leftMargin=1*cm,
            topMargin=1*cm, bottomMargin=1*cm)
        doc.build(self.story)


def generate_complexity_sweep_report(sweep_dir: Path, output_path: Path = None):
    """Main function to generate the complexity sweep report."""
    print(f"Scanning complexity sweep directory: {sweep_dir}")

    df = aggregate_results(sweep_dir)
    if df.empty:
        print("No results found. Cannot generate report.")
        return None

    # Filter out runs with missing ST params
    df = df.dropna(subset=['st_n', 'st_m', 'st_rho'])
    if df.empty:
        print("No runs with ST params found. Are these complexity sweep results?")
        return None

    print(f"Valid runs with ST params: {len(df)}")

    # Compute win rates per configuration
    win_df = compute_win_rates(df)
    print(f"Unique configurations: {len(win_df)}")
    print(f"Overall win rate: {100 * (df['F_actual_train'] > df['F_baseline_train']).mean():.1f}%")

    # Create plots directory
    plots_dir = sweep_dir / 'complexity_plots'
    plots_dir.mkdir(exist_ok=True)

    print("\nGenerating plots...")

    # 1. Summary panel (marginals)
    print("  - Summary panel (marginal effects)...")
    summary_path = plot_summary_panel(win_df, plots_dir / 'summary_panel.png')

    has_nproc = 'n_processes' in win_df.columns and win_df['n_processes'].notna().any()

    # 2. Individual parameter plots
    print("  - Win rate vs n...")
    n_path = plot_win_rate_vs_param(win_df, 'st_n', 'n (input variables)',
                                     plots_dir / 'winrate_vs_n.png')
    print("  - Win rate vs m...")
    m_path = plot_win_rate_vs_param(win_df, 'st_m', 'm (cascaded stages)',
                                     plots_dir / 'winrate_vs_m.png')
    print("  - Win rate vs rho...")
    rho_path = plot_win_rate_vs_param(win_df, 'st_rho', 'rho (noise intensity)',
                                       plots_dir / 'winrate_vs_rho.png')
    nproc_path = None
    if has_nproc:
        print("  - Win rate vs n_processes...")
        nproc_path = plot_win_rate_vs_param(win_df, 'n_processes', 'n_processes (chain length)',
                                             plots_dir / 'winrate_vs_nproc.png')

    # 3. 2D heatmaps
    print("  - Heatmap: n vs rho...")
    heatmap_n_rho = plot_win_rate_heatmap_2d(
        win_df, 'st_n', 'st_rho', 'n (inputs)', 'rho (noise)',
        plots_dir / 'heatmap_n_rho.png')
    print("  - Heatmap: n vs m...")
    heatmap_n_m = plot_win_rate_heatmap_2d(
        win_df, 'st_n', 'st_m', 'n (inputs)', 'm (stages)',
        plots_dir / 'heatmap_n_m.png')
    print("  - Heatmap: m vs rho...")
    heatmap_m_rho = plot_win_rate_heatmap_2d(
        win_df, 'st_m', 'st_rho', 'm (stages)', 'rho (noise)',
        plots_dir / 'heatmap_m_rho.png')

    # 4. 3D scatter
    print("  - 3D scatter...")
    scatter_3d_path = plot_3d_scatter(win_df, plots_dir / 'scatter_3d.png')

    # Generate PDF report
    if output_path is None:
        output_path = sweep_dir / 'complexity_sweep_report.pdf'

    print(f"\nGenerating PDF report: {output_path}")

    report = ComplexitySweepReportGenerator(output_path)
    report.add_title(datetime.now())
    report.add_summary_table(df, win_df)

    # Summary panel
    report.add_section("Marginal Effects")
    report.add_image(summary_path, width=6.5*inch,
                     caption="Win rate vs each complexity parameter (marginal). "
                             "Bars/points colored by win rate (green=high, red=low).")

    report.story.append(PageBreak())

    # Individual plots
    report.add_section("Win Rate vs Individual Parameters")
    report.add_image(n_path, width=5*inch,
                     caption="Win rate vs n (input variables). Larger n = more complex.")
    report.add_image(rho_path, width=5*inch,
                     caption="Win rate vs rho (noise intensity). Higher rho = noisier.")

    report.story.append(PageBreak())
    report.add_image(m_path, width=5*inch,
                     caption="Win rate vs m (cascaded stages). More stages = deeper.")
    if nproc_path:
        report.add_image(nproc_path, width=5*inch,
                         caption="Win rate vs n_processes (chain length). More processes = longer chain.")

    # Heatmaps
    report.add_section("2D Parameter Interactions")
    report.add_image(heatmap_n_rho, width=5*inch,
                     caption="Win rate heatmap: n vs rho (averaged over m).")
    report.story.append(PageBreak())
    report.add_image(heatmap_n_m, width=5*inch,
                     caption="Win rate heatmap: n vs m (averaged over rho).")
    report.add_image(heatmap_m_rho, width=5*inch,
                     caption="Win rate heatmap: m vs rho (averaged over n).")

    report.story.append(PageBreak())

    # 3D scatter
    report.add_section("3D Complexity Space")
    report.add_image(scatter_3d_path, width=5.5*inch,
                     caption="Win rate in (n, m, rho) space. Color = win rate, size = number of runs.")

    report.story.append(PageBreak())

    # Configurations table
    report.add_configs_table(win_df)

    report.build()

    # Save CSV summaries
    csv_all = sweep_dir / 'complexity_all_runs.csv'
    df.to_csv(csv_all, index=False)

    csv_configs = sweep_dir / 'complexity_configs_winrate.csv'
    win_df.to_csv(csv_configs, index=False)

    print(f"\nResults saved:")
    print(f"  Report:     {output_path}")
    print(f"  All runs:   {csv_all}")
    print(f"  Win rates:  {csv_configs}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate complexity sweep report PDF',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sweep_dir', type=str,
                        default='controller_optimization/checkpoints/complexity_sweep',
                        help='Directory containing complexity sweep results')
    parser.add_argument('--output', type=str, default=None,
                        help='Output PDF file path')
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"Error: Sweep directory not found: {sweep_dir}")
        return

    output_path = Path(args.output) if args.output else None
    generate_complexity_sweep_report(sweep_dir, output_path)


if __name__ == '__main__':
    main()
