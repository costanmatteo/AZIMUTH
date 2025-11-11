"""
Report PDF generator per controller optimization.

Simile a uncertainty_predictor report ma per sistema completo.
"""

from datetime import datetime
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus.flowables import HRFlowable
import json


def generate_controller_report(
    config,
    training_history,
    final_metrics,
    process_metrics,
    F_star,
    F_baseline,
    F_actual,
    checkpoint_dir,
    timestamp=None
):
    """
    Genera report PDF completo per controller optimization.

    Sezioni:
    1. Configuration (processi, policy generators, training params)
    2. Training History (loss plots)
    3. Final Metrics Comparison:
       - Tabella: | Metric | Target (F*) | Baseline (F') | Actual (F) | Improvement |
    4. Process-wise Analysis:
       - Tabella metriche per ogni processo
    5. Trajectory Comparison Plots
    6. Reliability Comparison Bar Chart

    Returns:
        str: Path al PDF generato
    """
    if timestamp is None:
        timestamp = datetime.now()

    checkpoint_dir = Path(checkpoint_dir)
    report_path = checkpoint_dir / 'controller_report.pdf'

    # Create PDF
    doc = SimpleDocTemplate(str(report_path), pagesize=A4)
    story = []
    styles = getSampleStyleSheet()

    # Title style
    title_style = ParagraphStyle(
        name='ReportTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        spaceAfter=10
    )

    # Section style
    section_style = ParagraphStyle(
        name='SectionTitle',
        parent=styles['Heading2'],
        fontSize=14,
        leading=16,
        fontName='Helvetica-Bold',
        spaceAfter=6,
        spaceBefore=10
    )

    # Body style
    body_style = ParagraphStyle(
        name='BodyText',
        parent=styles['Normal'],
        fontSize=10,
        leading=12,
    )

    # Title
    title = Paragraph("<b>Controller Optimization Report</b>", title_style)
    date_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
    subtitle = Paragraph(f"Generated: {date_str}", styles['Normal'])

    story.append(title)
    story.append(subtitle)
    story.append(Spacer(1, 0.3*cm))

    # Section 1: Configuration
    story.append(Paragraph("<b>1. Configuration</b>", section_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=6))

    config_text = f"""<b>Processes:</b> {', '.join(config['process_names'])}<br/>
<b>Policy Architecture:</b> {config['policy_generator']['architecture']}<br/>
<b>Training Epochs:</b> {config['training']['epochs']}<br/>
<b>Batch Size:</b> {config['training']['batch_size']}<br/>
<b>Learning Rate:</b> {config['training']['learning_rate']}<br/>
<b>Behavior Cloning Weight (λ_BC):</b> {config['training']['lambda_bc']}<br/>
<b>Device:</b> {config['training']['device']}"""

    story.append(Paragraph(config_text, body_style))
    story.append(Spacer(1, 0.3*cm))

    # Section 2: Final Metrics
    story.append(Paragraph("<b>2. Final Metrics</b>", section_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=6))

    # Metrics table
    improvement = final_metrics.get('improvement', 0) * 100
    target_gap = final_metrics.get('target_gap', 0) * 100

    metrics_data = [
        ['Metric', 'Value'],
        ['Target Reliability (F*)', f"{F_star:.6f}"],
        ['Baseline Reliability (F\')', f"{F_baseline:.6f}"],
        ['Controller Reliability (F)', f"{F_actual:.6f}"],
        ['Improvement over Baseline', f"{improvement:+.2f}%"],
        ['Gap from Target', f"{target_gap:.2f}%"],
    ]

    metrics_table = Table(metrics_data, colWidths=[10*cm, 6*cm])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    story.append(metrics_table)
    story.append(Spacer(1, 0.3*cm))

    # Section 3: Process-wise Metrics
    story.append(Paragraph("<b>3. Process-wise Metrics</b>", section_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=6))

    if 'actual' in process_metrics:
        process_data = [['Process', 'Input MSE', 'Output MSE', 'Combined MSE']]

        for process_name, metrics in process_metrics['actual'].items():
            process_data.append([
                process_name.capitalize(),
                f"{metrics['input_mse']:.6f}",
                f"{metrics['output_mse']:.6f}",
                f"{metrics['combined_mse']:.6f}"
            ])

        process_table = Table(process_data, colWidths=[4*cm, 4*cm, 4*cm, 4*cm])
        process_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(process_table)
        story.append(Spacer(1, 0.3*cm))

    # Section 4: Visualizations
    story.append(Paragraph("<b>4. Visualizations</b>", section_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=6))

    # Add plots
    plot_files = [
        ('training_history.png', 'Training History'),
        ('trajectory_comparison.png', 'Trajectory Comparison'),
        ('reliability_comparison.png', 'Reliability Comparison'),
        ('process_improvements.png', 'Process Improvements')
    ]

    for plot_file, plot_title in plot_files:
        plot_path = checkpoint_dir / plot_file
        if plot_path.exists():
            story.append(Paragraph(f"<b>{plot_title}</b>", body_style))
            story.append(Spacer(1, 0.2*cm))
            img = Image(str(plot_path), width=16*cm, height=10*cm)
            story.append(img)
            story.append(Spacer(1, 0.3*cm))

    # Build PDF
    doc.build(story)

    return report_path


if __name__ == '__main__':
    # Test report generation
    print("Testing controller report generation...")

    # Dummy data
    config = {
        'process_names': ['laser', 'plasma', 'galvanic'],
        'policy_generator': {'architecture': 'medium'},
        'training': {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'lambda_bc': 0.1,
            'device': 'cpu'
        }
    }

    training_history = {
        'total_loss': [1.0, 0.8, 0.6],
        'reliability_loss': [0.5, 0.4, 0.3],
        'bc_loss': [0.5, 0.4, 0.3],
        'F_values': [0.7, 0.8, 0.9]
    }

    final_metrics = {
        'improvement': 0.15,
        'target_gap': 0.05
    }

    process_metrics = {
        'actual': {
            'laser': {'input_mse': 0.001, 'output_mse': 0.002, 'combined_mse': 0.0015},
            'plasma': {'input_mse': 0.003, 'output_mse': 0.004, 'combined_mse': 0.0035}
        }
    }

    from pathlib import Path
    checkpoint_dir = Path('test_report')
    checkpoint_dir.mkdir(exist_ok=True)

    report_path = generate_controller_report(
        config=config,
        training_history=training_history,
        final_metrics=final_metrics,
        process_metrics=process_metrics,
        F_star=0.95,
        F_baseline=0.82,
        F_actual=0.93,
        checkpoint_dir=checkpoint_dir,
        timestamp=datetime.now()
    )

    print(f"✓ Report generated: {report_path}")
