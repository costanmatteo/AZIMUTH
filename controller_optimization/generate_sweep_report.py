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
    Create scatter plot of F_star (target), F_baseline, and F_actual for all runs.
    Each run is a point on x-axis, y-axis shows F values.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    n_runs = len(df)
    x = np.arange(n_runs)

    # Plot F_star (target) - green diamonds
    ax.scatter(x, df['F_star_train'].values, c='green', marker='D', s=50,
               label='F* (Target)', alpha=0.8, zorder=3)

    # Plot F_baseline - red squares
    ax.scatter(x, df['F_baseline_train'].values, c='red', marker='s', s=50,
               label="F' (Baseline)", alpha=0.8, zorder=2)

    # Plot F_actual - blue circles
    ax.scatter(x, df['F_actual_train'].values, c='blue', marker='o', s=50,
               label='F (Actual)', alpha=0.8, zorder=4)

    # Connect points with thin lines for each run
    for i in range(n_runs):
        ax.plot([i, i, i],
                [df['F_baseline_train'].iloc[i], df['F_actual_train'].iloc[i], df['F_star_train'].iloc[i]],
                'k-', alpha=0.2, linewidth=0.5)

    # Add mean lines
    ax.axhline(df['F_star_train'].mean(), color='green', linestyle='--', alpha=0.5,
               label=f'Mean F* = {df["F_star_train"].mean():.4f}')
    ax.axhline(df['F_baseline_train'].mean(), color='red', linestyle='--', alpha=0.5,
               label=f"Mean F' = {df['F_baseline_train'].mean():.4f}")
    ax.axhline(df['F_actual_train'].mean(), color='blue', linestyle='--', alpha=0.5,
               label=f'Mean F = {df["F_actual_train"].mean():.4f}')

    ax.set_xlabel('Run Index', fontsize=12)
    ax.set_ylabel('Reliability (F)', fontsize=12)
    ax.set_title('Target vs Baseline vs Actual Reliability Across All Sweep Runs', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Set x-ticks to show every 10th run
    tick_positions = np.arange(0, n_runs, max(1, n_runs // 10))
    ax.set_xticks(tick_positions)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def plot_improvement_distribution(df: pd.DataFrame, save_path: Path):
    """Plot distribution of improvement percentages."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Train improvement distribution
    ax1 = axes[0]
    ax1.hist(df['improvement_train'].dropna(), bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(df['improvement_train'].mean(), color='red', linestyle='--',
                label=f'Mean: {df["improvement_train"].mean():.2f}%')
    ax1.axvline(df['improvement_train'].median(), color='orange', linestyle=':',
                label=f'Median: {df["improvement_train"].median():.2f}%')
    ax1.set_xlabel('Improvement (%)', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Train Improvement Distribution', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Test improvement distribution
    ax2 = axes[1]
    test_imp = df['improvement_test'].dropna()
    if len(test_imp) > 0:
        ax2.hist(test_imp, bins=20, color='forestgreen', edgecolor='black', alpha=0.7)
        ax2.axvline(test_imp.mean(), color='red', linestyle='--',
                    label=f'Mean: {test_imp.mean():.2f}%')
        ax2.axvline(test_imp.median(), color='orange', linestyle=':',
                    label=f'Median: {test_imp.median():.2f}%')
    ax2.set_xlabel('Improvement (%)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Test Improvement Distribution', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def plot_seed_heatmap(df: pd.DataFrame, metric: str, save_path: Path, title: str = None):
    """Plot heatmap of metric as function of seed_target and seed_baseline."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create pivot table
    pivot = df.pivot_table(values=metric, index='seed_baseline', columns='seed_target', aggfunc='mean')

    # Plot heatmap
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns.astype(int))
    ax.set_yticklabels(pivot.index.astype(int))

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric, fontsize=11)

    # Add value annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = 'white' if val < pivot.values.mean() else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color=text_color, fontsize=8)

    ax.set_xlabel('seed_target', fontsize=12)
    ax.set_ylabel('seed_baseline', fontsize=12)
    ax.set_title(title or f'{metric} by Seed Combination', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def plot_f_values_boxplot(df: pd.DataFrame, save_path: Path):
    """Create boxplot comparing F_star, F_baseline, F_actual distributions."""
    fig, ax = plt.subplots(figsize=(8, 6))

    data = [
        df['F_star_train'].dropna(),
        df['F_baseline_train'].dropna(),
        df['F_actual_train'].dropna()
    ]
    labels = ['F* (Target)', "F' (Baseline)", 'F (Actual)']
    colors_list = ['green', 'red', 'blue']

    bp = ax.boxplot(data, labels=labels, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Reliability (F)', fontsize=12)
    ax.set_title('Distribution of Reliability Values Across Sweep', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add mean markers
    means = [d.mean() for d in data]
    ax.scatter([1, 2, 3], means, marker='D', color='black', s=50, zorder=5, label='Mean')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def generate_summary_stats(df: pd.DataFrame) -> dict:
    """Compute summary statistics from sweep results."""
    stats = {
        'n_runs': len(df),
        'n_successful': len(df[df['improvement_train'] > 0]),

        # F_star statistics
        'F_star_mean': df['F_star_train'].mean(),
        'F_star_std': df['F_star_train'].std(),
        'F_star_min': df['F_star_train'].min(),
        'F_star_max': df['F_star_train'].max(),

        # F_baseline statistics
        'F_baseline_mean': df['F_baseline_train'].mean(),
        'F_baseline_std': df['F_baseline_train'].std(),
        'F_baseline_min': df['F_baseline_train'].min(),
        'F_baseline_max': df['F_baseline_train'].max(),

        # F_actual statistics
        'F_actual_mean': df['F_actual_train'].mean(),
        'F_actual_std': df['F_actual_train'].std(),
        'F_actual_min': df['F_actual_train'].min(),
        'F_actual_max': df['F_actual_train'].max(),

        # Improvement statistics
        'improvement_mean': df['improvement_train'].mean(),
        'improvement_std': df['improvement_train'].std(),
        'improvement_min': df['improvement_train'].min(),
        'improvement_max': df['improvement_train'].max(),
        'improvement_median': df['improvement_train'].median(),

        # Success rate
        'success_rate_mean': df['success_rate_train'].mean() if 'success_rate_train' in df else None,

        # Best and worst runs
        'best_run': df.loc[df['improvement_train'].idxmax(), 'run_name'] if len(df) > 0 else None,
        'best_improvement': df['improvement_train'].max(),
        'worst_run': df.loc[df['improvement_train'].idxmin(), 'run_name'] if len(df) > 0 else None,
        'worst_improvement': df['improvement_train'].min(),
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
        """Add summary statistics table."""
        self.add_section("Summary Statistics")

        data = [
            ['Metric', 'Value'],
            ['Total Runs', f"{stats['n_runs']}"],
            ['Successful Runs (improvement > 0)', f"{stats['n_successful']} ({100*stats['n_successful']/stats['n_runs']:.1f}%)"],
            ['', ''],
            ['F* (Target) Mean ± Std', f"{stats['F_star_mean']:.4f} ± {stats['F_star_std']:.4f}"],
            ['F* Range', f"[{stats['F_star_min']:.4f}, {stats['F_star_max']:.4f}]"],
            ['', ''],
            ["F' (Baseline) Mean ± Std", f"{stats['F_baseline_mean']:.4f} ± {stats['F_baseline_std']:.4f}"],
            ["F' Range", f"[{stats['F_baseline_min']:.4f}, {stats['F_baseline_max']:.4f}]"],
            ['', ''],
            ['F (Actual) Mean ± Std', f"{stats['F_actual_mean']:.4f} ± {stats['F_actual_std']:.4f}"],
            ['F Range', f"[{stats['F_actual_min']:.4f}, {stats['F_actual_max']:.4f}]"],
            ['', ''],
            ['Improvement Mean ± Std', f"{stats['improvement_mean']:.2f}% ± {stats['improvement_std']:.2f}%"],
            ['Improvement Median', f"{stats['improvement_median']:.2f}%"],
            ['Improvement Range', f"[{stats['improvement_min']:.2f}%, {stats['improvement_max']:.2f}%]"],
            ['', ''],
            ['Best Run', f"{stats['best_run']} ({stats['best_improvement']:.2f}%)"],
            ['Worst Run', f"{stats['worst_run']} ({stats['worst_improvement']:.2f}%)"],
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
        """Add table with all runs sorted by improvement."""
        self.add_section("All Runs (sorted by improvement)")

        # Sort by improvement
        df_sorted = df.sort_values('improvement_train', ascending=False)

        # Create table data
        data = [['Run', 'seed_t', 'seed_b', 'F*', "F'", 'F', 'Improvement']]

        for _, row in df_sorted.iterrows():
            data.append([
                row['run_name'][:15],
                int(row['seed_target']) if pd.notna(row['seed_target']) else 'N/A',
                int(row['seed_baseline']) if pd.notna(row['seed_baseline']) else 'N/A',
                f"{row['F_star_train']:.4f}" if pd.notna(row['F_star_train']) else 'N/A',
                f"{row['F_baseline_train']:.4f}" if pd.notna(row['F_baseline_train']) else 'N/A',
                f"{row['F_actual_train']:.4f}" if pd.notna(row['F_actual_train']) else 'N/A',
                f"{row['improvement_train']:.2f}%" if pd.notna(row['improvement_train']) else 'N/A',
            ])

        # Create table with smaller font
        table = Table(data, colWidths=[1.2*inch, 0.6*inch, 0.6*inch, 0.8*inch, 0.8*inch, 0.8*inch, 1*inch])
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

    print("  - Improvement heatmap...")
    heatmap_path = plot_seed_heatmap(
        df, 'improvement_train', plots_dir / 'improvement_heatmap.png',
        title='Improvement (%) by Seed Combination'
    )

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
                     caption="Figure 1: F* (target), F' (baseline), and F (actual) for each sweep run")

    report.story.append(PageBreak())

    # Add boxplot
    report.add_section("Reliability Distribution")
    report.add_image(boxplot_path, width=5*inch,
                     caption="Figure 2: Distribution of reliability values across all runs")

    # Add improvement distribution
    report.add_section("Improvement Distribution")
    report.add_image(dist_path, width=6*inch,
                     caption="Figure 3: Distribution of improvement percentages")

    report.story.append(PageBreak())

    # Add heatmap
    report.add_section("Seed Combination Heatmap")
    report.add_image(heatmap_path, width=5.5*inch,
                     caption="Figure 4: Improvement as a function of seed_target and seed_baseline")

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
