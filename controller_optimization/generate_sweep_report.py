#!/usr/bin/env python3
"""
Generate aggregated PDF report from parameter sweep results.

Usage:
    python generate_sweep_report.py [--sweep_dir PATH] [--output sweep_report.pdf]

This script:
1. Loads results from all sweep runs
2. Computes aggregate statistics
3. Generates visualizations including target vs baseline vs actual scatter
4. Creates a comprehensive PDF report
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

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus.flowables import HRFlowable


def load_run_results(run_dir: Path) -> dict:
    """Load results from a single run directory."""
    results_file = run_dir / "final_results.json"
    if not results_file.exists():
        return None

    try:
        with open(results_file, 'r') as f:
            data = json.load(f)

        # Extract key metrics
        config = data.get('config', {})
        scenarios_config = config.get('scenarios', {})

        return {
            'run_name': run_dir.name,
            'seed_target': scenarios_config.get('seed_target'),
            'seed_baseline': scenarios_config.get('seed_baseline'),
            # Train metrics
            'F_star_train': data.get('train', {}).get('F_star_mean'),
            'F_baseline_train': data.get('train', {}).get('F_baseline_mean'),
            'F_actual_train': data.get('train', {}).get('F_actual_mean'),
            'F_actual_std_train': data.get('train', {}).get('F_actual_std'),
            'improvement_train': data.get('train', {}).get('improvement_pct'),
            'target_gap_train': data.get('train', {}).get('target_gap_pct'),
            # Test metrics
            'F_star_test': data.get('test', {}).get('F_star_mean'),
            'F_baseline_test': data.get('test', {}).get('F_baseline_mean'),
            'F_actual_test': data.get('test', {}).get('F_actual_mean'),
            'improvement_test': data.get('test', {}).get('improvement_pct'),
            # Advanced metrics
            'success_rate_train': data.get('advanced_metrics', {}).get('success_rate_train', {}).get('success_rate_pct'),
            'success_rate_test': data.get('advanced_metrics', {}).get('success_rate_test', {}).get('success_rate_pct'),
            'worst_case_gap_train': data.get('advanced_metrics', {}).get('worst_case_gap_train', {}).get('worst_case_gap'),
            'worst_case_gap_test': data.get('advanced_metrics', {}).get('worst_case_gap_test', {}).get('worst_case_gap'),
        }
    except Exception as e:
        print(f"Error loading {results_file}: {e}")
        return None


def aggregate_results(sweep_dir: Path) -> pd.DataFrame:
    """Aggregate results from all runs in sweep directory."""
    results = []

    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        run_results = load_run_results(run_dir)
        if run_results is not None:
            results.append(run_results)
            print(f"  Loaded: {run_dir.name}")

    if not results:
        print("No results found!")
        return pd.DataFrame()

    return pd.DataFrame(results)


def plot_target_baseline_actual_scatter(df: pd.DataFrame, save_path: Path):
    """
    Two side-by-side scatter plots:
    - Left: F* vs F_baseline (red)
    - Right: F* vs F_actual (blue)
    Both have diagonal line for perfect match.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    F_star = df['F_star_train'].values
    F_baseline = df['F_baseline_train'].values
    F_actual = df['F_actual_train'].values

    # Common limits for both plots (filter out None/NaN)
    all_values = np.array([v for v in np.concatenate([F_star, F_baseline, F_actual]) if v is not None and not np.isnan(float(v))])
    if len(all_values) == 0:
        print("  WARNING: No valid data for scatter plot, skipping.")
        plt.close()
        return save_path
    min_val = all_values.min()
    max_val = all_values.max()
    margin = (max_val - min_val) * 0.1

    # Calculate gaps
    gap_baseline = F_star - F_baseline
    gap_actual = F_star - F_actual
    controller_wins = np.sum(gap_actual < gap_baseline)
    n_runs = len(df)

    # LEFT PLOT: Baseline
    ax1 = axes[0]
    ax1.scatter(F_baseline, F_star,
                c='red', s=100, alpha=0.6,
                edgecolors='darkred', linewidths=1.5,
                marker='s')
    ax1.plot([min_val - margin, max_val + margin],
             [min_val - margin, max_val + margin],
             'k--', linewidth=2, alpha=0.5, label='Perfect (F = F*)')
    ax1.set_xlabel('F\' (Baseline Reliability)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('F* (Target Reliability)', fontsize=12, fontweight='bold')
    ax1.set_title('Baseline (No Controller)', fontsize=14, fontweight='bold', color='darkred')
    ax1.set_xlim(min_val - margin, max_val + margin)
    ax1.set_ylim(min_val - margin, max_val + margin)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')
    ax1.set_aspect('equal')

    # Stats for baseline
    stats1 = f'Gap range: [{gap_baseline.min():.4f}, {gap_baseline.max():.4f}]\nMedian gap: {np.median(gap_baseline):.4f}'
    ax1.text(0.02, 0.98, stats1, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))

    # RIGHT PLOT: Controller
    ax2 = axes[1]
    ax2.scatter(F_actual, F_star,
                c='blue', s=100, alpha=0.6,
                edgecolors='darkblue', linewidths=1.5,
                marker='o')
    ax2.plot([min_val - margin, max_val + margin],
             [min_val - margin, max_val + margin],
             'k--', linewidth=2, alpha=0.5, label='Perfect (F = F*)')
    ax2.set_xlabel('F (Controller Reliability)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F* (Target Reliability)', fontsize=12, fontweight='bold')
    ax2.set_title('Controller', fontsize=14, fontweight='bold', color='darkblue')
    ax2.set_xlim(min_val - margin, max_val + margin)
    ax2.set_ylim(min_val - margin, max_val + margin)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right')
    ax2.set_aspect('equal')

    # Stats for controller
    stats2 = f'Gap range: [{gap_actual.min():.4f}, {gap_actual.max():.4f}]\nMedian gap: {np.median(gap_actual):.4f}'
    ax2.text(0.02, 0.98, stats2, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Overall title
    fig.suptitle(f'Target vs Reliability Comparison (n={n_runs}, Controller wins: {controller_wins}/{n_runs} = {100*controller_wins/n_runs:.1f}%)',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def plot_improvement_distribution(df: pd.DataFrame, save_path: Path):
    """
    Plot distribution of gaps: F* - F (smaller is better).

    Shows how close the controller gets to the target compared to baseline.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Calculate gaps (F* - F, smaller = better)
    gap_baseline_train = df['F_star_train'] - df['F_baseline_train']
    gap_actual_train = df['F_star_train'] - df['F_actual_train']

    # Left plot: Gap distribution comparison (Train)
    ax1 = axes[0]
    bins = np.linspace(
        min(gap_baseline_train.min(), gap_actual_train.min()),
        max(gap_baseline_train.max(), gap_actual_train.max()),
        25
    )
    ax1.hist(gap_baseline_train, bins=bins, color='red', edgecolor='darkred',
             alpha=0.5, label='Baseline Gap (F* - F\')')
    ax1.hist(gap_actual_train, bins=bins, color='blue', edgecolor='darkblue',
             alpha=0.5, label='Controller Gap (F* - F)')
    ax1.axvline(0, color='green', linestyle='-', linewidth=2, label='Perfect (gap = 0)')
    ax1.set_xlabel('Gap (F* - F)', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Gap Distribution: Baseline vs Controller (Train)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right plot: Gap reduction (how much controller improved over baseline)
    ax2 = axes[1]
    gap_reduction = gap_baseline_train - gap_actual_train  # Positive = controller better
    ax2.hist(gap_reduction, bins=25, color='forestgreen', edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='No improvement')

    # Count wins/losses
    wins = (gap_reduction > 0).sum()
    losses = (gap_reduction < 0).sum()
    ties = (gap_reduction == 0).sum()

    ax2.set_xlabel('Gap Reduction (positive = controller better)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title(f'Controller Improvement: {wins} wins, {losses} losses', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def plot_seed_heatmap(df: pd.DataFrame, save_path: Path):
    """Plot heatmap of gap reduction by seed combination.

    Gap reduction = baseline_gap - controller_gap
    Positive = controller better (green)
    Negative = baseline better (red)
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate gap reduction
    df = df.copy()
    df['gap_reduction'] = (df['F_star_train'] - df['F_baseline_train']) - (df['F_star_train'] - df['F_actual_train'])

    # Create pivot table
    pivot = df.pivot_table(values='gap_reduction', index='seed_baseline', columns='seed_target', aggfunc='first')

    # Handle case with single value
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        ax.text(0.5, 0.5, 'Not enough data for heatmap\n(need multiple seed values)',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path

    # Plot heatmap - use diverging colormap centered at 0
    vmax = max(abs(pivot.values.min()), abs(pivot.values.max()))
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)

    # Set ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns.astype(int))
    ax.set_yticklabels(pivot.index.astype(int))

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Gap Reduction (positive = controller better)', fontsize=11)

    # Add value annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = 'white' if abs(val) > vmax * 0.5 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        color=text_color, fontsize=8)

    ax.set_xlabel('seed_target', fontsize=12)
    ax.set_ylabel('seed_baseline', fontsize=12)
    ax.set_title('Gap Reduction by Seed Combination\n(Green = Controller Better, Red = Baseline Better)', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def plot_f_values_boxplot(df: pd.DataFrame, save_path: Path):
    """Create boxplot comparing gap distributions: F* - F' (baseline) vs F* - F (controller)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate gaps (smaller = better, closer to target)
    gap_baseline = df['F_star_train'] - df['F_baseline_train']
    gap_controller = df['F_star_train'] - df['F_actual_train']

    data = [
        gap_baseline.dropna(),
        gap_controller.dropna()
    ]
    labels = ['Baseline Gap\n(F* - F\')', 'Controller Gap\n(F* - F)']
    colors_list = ['red', 'blue']

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Add horizontal line at 0 (perfect match)
    ax.axhline(0, color='green', linestyle='--', linewidth=2, label='Perfect (gap = 0)')

    ax.set_ylabel('Gap (F* - F)', fontsize=12)
    ax.set_title('Gap Distribution: Baseline vs Controller\n(smaller gap = better)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def generate_summary_stats(df: pd.DataFrame) -> dict:
    """Compute summary statistics from sweep results using gaps (F* - F)."""

    # Calculate gaps
    gap_baseline = df['F_star_train'] - df['F_baseline_train']
    gap_controller = df['F_star_train'] - df['F_actual_train']
    gap_reduction = gap_baseline - gap_controller  # Positive = controller better

    # Count wins (controller gap < baseline gap)
    controller_wins = (gap_controller < gap_baseline).sum()

    stats = {
        'n_runs': len(df),
        'controller_wins': controller_wins,
        'controller_win_rate': 100 * controller_wins / len(df) if len(df) > 0 else 0,

        # Gap statistics (F* - F, smaller is better)
        'gap_baseline_min': gap_baseline.min(),
        'gap_baseline_max': gap_baseline.max(),
        'gap_baseline_median': gap_baseline.median(),

        'gap_controller_min': gap_controller.min(),
        'gap_controller_max': gap_controller.max(),
        'gap_controller_median': gap_controller.median(),

        # Gap reduction (baseline_gap - controller_gap, positive = improvement)
        'gap_reduction_min': gap_reduction.min(),
        'gap_reduction_max': gap_reduction.max(),
        'gap_reduction_median': gap_reduction.median(),

        # Best and worst runs based on controller gap
        'best_run': df.loc[gap_controller.idxmin(), 'run_name'] if len(df) > 0 and gap_controller.notna().any() else None,
        'best_gap': gap_controller.min(),
        'worst_run': df.loc[gap_controller.idxmax(), 'run_name'] if len(df) > 0 and gap_controller.notna().any() else None,
        'worst_gap': gap_controller.max(),
    }

    return stats


class SweepReportGenerator:
    """PDF report generator for sweep results."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.styles = getSampleStyleSheet()
        self.story = []

        # Custom styles
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=6
        ))

        self.styles.add(ParagraphStyle(
            name='ReportSubtitle',
            parent=self.styles['Normal'],
            fontSize=11,
            leading=14,
            alignment=TA_CENTER,
            spaceAfter=12
        ))

        self.styles.add(ParagraphStyle(
            name='SectionTitle',
            parent=self.styles['Heading2'],
            fontSize=12,
            leading=14,
            fontName='Helvetica-Bold',
            spaceAfter=6,
            spaceBefore=12
        ))

        # Modify existing BodyText style instead of adding new one
        self.styles['BodyText'].fontSize = 9
        self.styles['BodyText'].leading = 11

    def add_title(self, timestamp: datetime):
        """Add report title."""
        title = Paragraph("<b>Controller Seed Sweep Report</b>", self.styles['ReportTitle'])
        date_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        subtitle = Paragraph(f"Generated: {date_str}", self.styles['ReportSubtitle'])

        self.story.append(title)
        self.story.append(subtitle)
        self.story.append(HRFlowable(width="100%", thickness=2, color=colors.black, spaceAfter=12))

    def add_section(self, title: str):
        """Add section title."""
        para = Paragraph(f"<b>{title}</b>", self.styles['SectionTitle'])
        line = HRFlowable(width="100%", thickness=1, color=colors.darkgray, spaceAfter=6)
        self.story.append(para)
        self.story.append(line)

    def add_summary_table(self, stats: dict):
        """Add summary statistics table using gap-based metrics."""
        self.add_section("Summary Statistics")

        data = [
            ['Metric', 'Value'],
            ['Total Runs', f"{stats['n_runs']}"],
            ['Controller Wins', f"{stats['controller_wins']}/{stats['n_runs']} ({stats['controller_win_rate']:.1f}%)"],
            ['', ''],
            ['Baseline Gap (F* - F\')', '(smaller = better)'],
            ['  Range', f"[{stats['gap_baseline_min']:.4f}, {stats['gap_baseline_max']:.4f}]"],
            ['  Median', f"{stats['gap_baseline_median']:.4f}"],
            ['', ''],
            ['Controller Gap (F* - F)', '(smaller = better)'],
            ['  Range', f"[{stats['gap_controller_min']:.4f}, {stats['gap_controller_max']:.4f}]"],
            ['  Median', f"{stats['gap_controller_median']:.4f}"],
            ['', ''],
            ['Gap Reduction', '(positive = controller better)'],
            ['  Range', f"[{stats['gap_reduction_min']:.4f}, {stats['gap_reduction_max']:.4f}]"],
            ['  Median', f"{stats['gap_reduction_median']:.4f}"],
            ['', ''],
            ['Best Run (smallest gap)', f"{stats['best_run']} (gap: {stats['best_gap']:.4f})"],
            ['Worst Run (largest gap)', f"{stats['worst_run']} (gap: {stats['worst_gap']:.4f})"],
        ]

        table = Table(data, colWidths=[3*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))

        self.story.append(table)
        self.story.append(Spacer(1, 0.5*cm))

    def add_image(self, image_path: Path, width: float = 6*inch, caption: str = None):
        """Add image to report."""
        if image_path.exists():
            img = Image(str(image_path), width=width, height=width*0.6)
            self.story.append(img)
            if caption:
                cap = Paragraph(f"<i>{caption}</i>", self.styles['BodyText'])
                self.story.append(cap)
            self.story.append(Spacer(1, 0.3*cm))

    def add_runs_table(self, df: pd.DataFrame):
        """Add table with all runs sorted by controller gap (smallest = best)."""
        self.add_section("All Runs (sorted by controller gap, smallest first)")

        # Calculate gaps and sort
        df = df.copy()
        df['gap_baseline'] = df['F_star_train'] - df['F_baseline_train']
        df['gap_controller'] = df['F_star_train'] - df['F_actual_train']
        df_sorted = df.sort_values('gap_controller', ascending=True)

        # Create table data
        data = [['Run', 'seed_t', 'seed_b', 'F*', 'Gap Baseline', 'Gap Controller', 'Winner']]

        for _, row in df_sorted.iterrows():
            gap_b = row['gap_baseline']
            gap_c = row['gap_controller']
            winner = '✓ Ctrl' if gap_c < gap_b else '✗ Base'

            data.append([
                row['run_name'][:15],
                int(row['seed_target']) if pd.notna(row['seed_target']) else 'N/A',
                int(row['seed_baseline']) if pd.notna(row['seed_baseline']) else 'N/A',
                f"{row['F_star_train']:.4f}" if pd.notna(row['F_star_train']) else 'N/A',
                f"{gap_b:.4f}" if pd.notna(gap_b) else 'N/A',
                f"{gap_c:.4f}" if pd.notna(gap_c) else 'N/A',
                winner,
            ])

        # Create table with smaller font
        table = Table(data, colWidths=[1.2*inch, 0.6*inch, 0.6*inch, 0.8*inch, 0.9*inch, 0.9*inch, 0.7*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 0.95, 0.95)]),
        ]))

        self.story.append(table)

    def build(self):
        """Build the PDF document."""
        doc = SimpleDocTemplate(
            str(self.output_path),
            pagesize=A4,
            rightMargin=1*cm,
            leftMargin=1*cm,
            topMargin=1*cm,
            bottomMargin=1*cm
        )
        doc.build(self.story)


def generate_sweep_report(sweep_dir: Path, output_path: Path = None):
    """Main function to generate the sweep report."""
    print(f"Scanning sweep directory: {sweep_dir}")

    # Load all results
    df = aggregate_results(sweep_dir)

    if df.empty:
        print("No results found. Cannot generate report.")
        return None

    print(f"\nLoaded {len(df)} runs")

    # Drop runs with missing core metrics
    core_cols = ['F_star_train', 'F_baseline_train', 'F_actual_train']
    valid_before = len(df)
    df = df.dropna(subset=core_cols)
    if len(df) < valid_before:
        print(f"  Dropped {valid_before - len(df)} runs with missing F values ({len(df)} valid runs remaining)")
    df = df.reset_index(drop=True)

    if df.empty:
        print("No valid runs with complete metrics. Cannot generate report.")
        return None

    # Compute summary statistics
    stats = generate_summary_stats(df)

    # Create plots directory
    plots_dir = sweep_dir / 'sweep_plots'
    plots_dir.mkdir(exist_ok=True)

    # Generate plots
    print("\nGenerating plots...")

    print("  - Target vs Baseline vs Actual scatter plot...")
    scatter_path = plot_target_baseline_actual_scatter(df, plots_dir / 'target_baseline_actual_scatter.png')

    print("  - Improvement distribution...")
    dist_path = plot_improvement_distribution(df, plots_dir / 'improvement_distribution.png')

    print("  - F values boxplot...")
    boxplot_path = plot_f_values_boxplot(df, plots_dir / 'f_values_boxplot.png')

    print("  - Gap reduction heatmap...")
    heatmap_path = plot_seed_heatmap(df, plots_dir / 'gap_reduction_heatmap.png')

    # Generate PDF report
    if output_path is None:
        output_path = sweep_dir / 'sweep_report.pdf'

    print(f"\nGenerating PDF report: {output_path}")

    report = SweepReportGenerator(output_path)
    report.add_title(datetime.now())
    report.add_summary_table(stats)

    # Add scatter plot
    report.add_section("Target vs Baseline vs Actual (All Runs)")
    report.add_image(scatter_path, width=6.5*inch,
                     caption="Figure 1: Scatter plot showing controller (blue) vs baseline (red). Points on diagonal = perfect match with target F*.")

    report.story.append(PageBreak())

    # Add boxplot
    report.add_section("Gap Distribution")
    report.add_image(boxplot_path, width=5*inch,
                     caption="Figure 2: Gap distribution (F* - F). Smaller gap = better. Green line = perfect (gap=0).")

    # Add gap distribution
    report.add_section("Gap Comparison: Baseline vs Controller")
    report.add_image(dist_path, width=6*inch,
                     caption="Figure 3: Left: overlapping gap distributions. Right: gap reduction (positive = controller wins).")

    report.story.append(PageBreak())

    # Add heatmap
    report.add_section("Gap Reduction by Seed Combination")
    report.add_image(heatmap_path, width=5.5*inch,
                     caption="Figure 4: Gap reduction by seed combination. Green = controller better, Red = baseline better.")

    report.story.append(PageBreak())

    # Add runs table
    report.add_runs_table(df)

    # Build PDF
    report.build()

    # Save CSV summary
    csv_path = sweep_dir / 'sweep_results_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"Results CSV saved: {csv_path}")

    print(f"\nReport generated: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate sweep report PDF')
    parser.add_argument('--sweep_dir', type=str,
                        default='controller_optimization/checkpoints/sweep',
                        help='Directory containing sweep run results')
    parser.add_argument('--output', type=str, default=None,
                        help='Output PDF file path')
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"Error: Sweep directory not found: {sweep_dir}")
        return

    output_path = Path(args.output) if args.output else None
    generate_sweep_report(sweep_dir, output_path)


if __name__ == '__main__':
    main()
