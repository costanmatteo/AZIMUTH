"""
Report PDF generator per controller optimization.

LaTeX-style report with two-column layout for controller optimization
"""

from datetime import datetime
from pathlib import Path
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, Frame, PageTemplate, KeepTogether
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus.flowables import HRFlowable

# Optional PDF manipulation for 2-up layout
try:
    from pypdf import PdfReader, PdfWriter, Transformation
    PYPDF_AVAILABLE = True
except ImportError as e:
    PYPDF_AVAILABLE = False
    import sys
    print(f"Warning: pypdf not available (reason: {e})", file=sys.stderr)
    print(f"Python executable: {sys.executable}", file=sys.stderr)
    print(f"Install with: {sys.executable} -m pip install pypdf", file=sys.stderr)


class ControllerReportGenerator:
    """LaTeX-style PDF report generator for controller optimization results"""

    def __init__(self, output_path):
        self.output_path = Path(output_path)
        self.styles = getSampleStyleSheet()
        self.story = []

        # Title style
        if 'ReportTitle' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='ReportTitle',
                parent=self.styles['Heading1'],
                fontSize=16,
                leading=19,
                alignment=TA_CENTER,
                spaceAfter=3
            ))

        # Subtitle style
        if 'ReportSubtitle' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='ReportSubtitle',
                parent=self.styles['Normal'],
                fontSize=10,
                leading=12,
                alignment=TA_CENTER,
                spaceAfter=6
            ))

        # Section title style
        if 'SectionTitle' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='SectionTitle',
                parent=self.styles['Heading2'],
                fontSize=10,
                leading=12,
                fontName='Helvetica-Bold',
                spaceAfter=1,
                spaceBefore=4
            ))

        # Body text style
        if 'BodyText' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='BodyText',
                parent=self.styles['Normal'],
                fontSize=7,
                leading=9,
                leftIndent=10
            ))

    def add_title(self, timestamp):
        """Add centered title and date"""
        title = Paragraph("<b>Controller Optimization Training Report</b>", self.styles['ReportTitle'])
        date_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        subtitle = Paragraph(f"Date: {date_str}", self.styles['ReportSubtitle'])

        self.story.append(title)
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.1*cm))

    def add_section_title(self, title):
        """Add section title with horizontal line"""
        para = Paragraph(f"<b>{title}</b>", self.styles['SectionTitle'])
        line = HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=4)
        self.story.append(para)
        self.story.append(line)

    def create_two_column_section(self, config, training_history, F_star, F_baseline, F_actual, final_metrics):
        """Create two-column layout for compact info"""

        # Left column data
        left_col = []

        # Configuration
        left_col.append(Paragraph("<b>Configuration</b>", self.styles['SectionTitle']))
        left_col.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=4))

        process_names = ', '.join(config['process_names'])
        config_text = f"""• <b>Processes:</b> {process_names}<br/>
• <b>Policy Architecture:</b> {config['policy_generator']['architecture']}<br/>
• <b>Hidden Sizes:</b> {config['policy_generator'].get('hidden_sizes', 'N/A')}<br/>
• <b>Activation:</b> {config['policy_generator'].get('activation', 'N/A')}<br/>
• <b>Dropout Rate:</b> {config['policy_generator'].get('dropout_rate', 'N/A')}"""
        left_col.append(Paragraph(config_text, self.styles['BodyText']))
        left_col.append(Spacer(1, 0.15*cm))

        # Training Parameters
        left_col.append(Paragraph("<b>Training Parameters</b>", self.styles['SectionTitle']))
        left_col.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=4))

        epochs_run = len(training_history.get('total_loss', []))
        epochs_total = config['training']['epochs']

        training_text = f"""• <b>Epochs:</b> {epochs_run}/{epochs_total}<br/>
• <b>Batch Size:</b> {config['training']['batch_size']}<br/>
• <b>Learning Rate:</b> {config['training']['learning_rate']}<br/>
• <b>Weight Decay:</b> {config['training'].get('weight_decay', 0.0)}<br/>
• <b>Behavior Cloning Weight (λ_BC):</b> {config['training']['lambda_bc']}<br/>
• <b>Patience:</b> {config['training'].get('patience', 'N/A')}<br/>
• <b>Device:</b> {config['training']['device']}<br/>
• <b>Checkpoint Dir:</b> {config['training']['checkpoint_dir']}"""
        left_col.append(Paragraph(training_text, self.styles['BodyText']))

        # Right column data
        right_col = []

        # Training Results
        right_col.append(Paragraph("<b>Training Results</b>", self.styles['SectionTitle']))
        right_col.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=4))

        final_total_loss = training_history['total_loss'][-1] if training_history.get('total_loss') else 0.0
        final_reliability_loss = training_history['reliability_loss'][-1] if training_history.get('reliability_loss') else 0.0
        final_bc_loss = training_history['bc_loss'][-1] if training_history.get('bc_loss') else 0.0
        best_total_loss = min(training_history['total_loss']) if training_history.get('total_loss') else 0.0

        results_text = f"""• <b>Final Total Loss:</b> {final_total_loss:.6f}<br/>
• <b>Final Reliability Loss:</b> {final_reliability_loss:.6f}<br/>
• <b>Final BC Loss:</b> {final_bc_loss:.6f}<br/>
• <b>Best Total Loss:</b> {best_total_loss:.6f}"""

        right_col.append(Paragraph(results_text, self.styles['BodyText']))
        right_col.append(Spacer(1, 0.15*cm))

        # Reliability Metrics
        right_col.append(Paragraph("<b>Reliability Metrics</b>", self.styles['SectionTitle']))
        right_col.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=4))

        improvement = final_metrics.get('improvement', 0) * 100
        target_gap = final_metrics.get('target_gap', 0) * 100

        reliability_text = f"""• <b>Target Reliability (F*):</b> {F_star:.6f}<br/>
• <b>Baseline Reliability (F'):</b> {F_baseline:.6f}<br/>
• <b>Controller Reliability (F):</b> {F_actual:.6f}<br/>
• <b>Improvement over Baseline:</b> {improvement:+.2f}%<br/>
• <b>Gap from Target:</b> {target_gap:.2f}%"""

        right_col.append(Paragraph(reliability_text, self.styles['BodyText']))
        right_col.append(Spacer(1, 0.15*cm))

        # Miscellaneous Parameters
        right_col.append(Paragraph("<b>Miscellaneous</b>", self.styles['SectionTitle']))
        right_col.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=4))

        misc_text = f"""• <b>Random Seed:</b> {config.get('misc', {}).get('random_seed', 'N/A')}"""
        if 'verbose' in config.get('misc', {}):
            misc_text += f"<br/>• <b>Verbose:</b> {config['misc']['verbose']}"
        right_col.append(Paragraph(misc_text, self.styles['BodyText']))

        # Create two-column table
        data = [[left_col, right_col]]
        col_table = Table(data, colWidths=[9*cm, 9*cm])
        col_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ]))

        self.story.append(col_table)
        self.story.append(Spacer(1, 0.2*cm))

    def add_process_metrics_table(self, process_metrics):
        """Add process-wise metrics table in LaTeX style"""
        self.add_section_title("Process-wise Metrics")

        if 'actual' not in process_metrics:
            return

        # Build table data
        headers = ['Process', 'Input MSE', 'Output MSE', 'Combined MSE']
        data = [headers]

        for process_name, metrics in process_metrics['actual'].items():
            row = [
                process_name.capitalize(),
                f"{metrics['input_mse']:.6f}",
                f"{metrics['output_mse']:.6f}",
                f"{metrics['combined_mse']:.6f}"
            ]
            data.append(row)

        # Create table with adjusted column widths
        col_widths = [4.5*cm, 4.5*cm, 4.5*cm, 4.5*cm]
        table = Table(data, colWidths=col_widths)
        table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),

            # Data rows
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),

            # Top and bottom lines
            ('LINEABOVE', (0, 0), (-1, 0), 1.5, colors.black),
            ('LINEABOVE', (0, 1), (-1, 1), 0.5, colors.black),
            ('LINEBELOW', (0, -1), (-1, -1), 1.5, colors.black),
        ]))

        self.story.append(table)
        self.story.append(Spacer(1, 0.1*cm))

    def add_plots_stacked(self, checkpoint_dir):
        """Add controller optimization plots stacked vertically"""
        self.add_section_title("Training Visualization")

        checkpoint_dir = Path(checkpoint_dir)

        # Training history plot
        history_plot = checkpoint_dir / 'training_history.png'
        if history_plot.exists():
            img = Image(str(history_plot))
            img_width, img_height = img.imageWidth, img.imageHeight
            aspect_ratio = img_height / img_width

            # Larger width for stacked layout
            new_width = 16*cm
            new_height = new_width * aspect_ratio

            # Max height constraint
            if new_height > 10*cm:
                new_height = 10*cm
                new_width = new_height / aspect_ratio

            img.drawWidth = new_width
            img.drawHeight = new_height

            # Center the image
            img_table = Table([[img]], colWidths=[18*cm])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            self.story.append(img_table)

            caption = Paragraph("<i>Training History - Total Loss and Components</i>", self.styles['Normal'])
            caption_table = Table([[caption]], colWidths=[18*cm])
            caption_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ]))
            self.story.append(caption_table)
            self.story.append(Spacer(1, 0.15*cm))

        # Trajectory comparison plot
        trajectory_plot = checkpoint_dir / 'trajectory_comparison.png'
        if trajectory_plot.exists():
            img = Image(str(trajectory_plot))
            img_width, img_height = img.imageWidth, img.imageHeight
            aspect_ratio = img_height / img_width

            new_width = 16*cm
            new_height = new_width * aspect_ratio

            if new_height > 10*cm:
                new_height = 10*cm
                new_width = new_height / aspect_ratio

            img.drawWidth = new_width
            img.drawHeight = new_height

            img_table = Table([[img]], colWidths=[18*cm])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            self.story.append(img_table)

            caption = Paragraph("<i>Trajectory Comparison - Target vs Baseline vs Controller</i>", self.styles['Normal'])
            caption_table = Table([[caption]], colWidths=[18*cm])
            caption_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ]))
            self.story.append(caption_table)
            self.story.append(Spacer(1, 0.15*cm))

        # Reliability comparison plot
        reliability_plot = checkpoint_dir / 'reliability_comparison.png'
        if reliability_plot.exists():
            img = Image(str(reliability_plot))
            img_width, img_height = img.imageWidth, img.imageHeight
            aspect_ratio = img_height / img_width

            new_width = 16*cm
            new_height = new_width * aspect_ratio

            if new_height > 10*cm:
                new_height = 10*cm
                new_width = new_height / aspect_ratio

            img.drawWidth = new_width
            img.drawHeight = new_height

            img_table = Table([[img]], colWidths=[18*cm])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            self.story.append(img_table)

            caption = Paragraph("<i>Reliability Comparison - System Performance</i>", self.styles['Normal'])
            caption_table = Table([[caption]], colWidths=[18*cm])
            caption_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ]))
            self.story.append(caption_table)
            self.story.append(Spacer(1, 0.15*cm))

        # Process improvements plot
        improvements_plot = checkpoint_dir / 'process_improvements.png'
        if improvements_plot.exists():
            img = Image(str(improvements_plot))
            img_width, img_height = img.imageWidth, img.imageHeight
            aspect_ratio = img_height / img_width

            new_width = 16*cm
            new_height = new_width * aspect_ratio

            if new_height > 10*cm:
                new_height = 10*cm
                new_width = new_height / aspect_ratio

            img.drawWidth = new_width
            img.drawHeight = new_height

            img_table = Table([[img]], colWidths=[18*cm])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            self.story.append(img_table)

            caption = Paragraph("<i>Process-wise Improvements</i>", self.styles['Normal'])
            caption_table = Table([[caption]], colWidths=[18*cm])
            caption_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ]))
            self.story.append(caption_table)
            self.story.append(Spacer(1, 0.15*cm))

    def generate(self, config, training_history, final_metrics, process_metrics,
                 F_star, F_baseline, F_actual, timestamp):
        """Generate the complete PDF"""

        # Add all sections
        self.add_title(timestamp)
        self.create_two_column_section(config, training_history, F_star, F_baseline, F_actual, final_metrics)
        self.add_process_metrics_table(process_metrics)
        self.add_plots_stacked(Path(config['training']['checkpoint_dir']))

        # Build PDF
        doc = SimpleDocTemplate(
            str(self.output_path),
            pagesize=A4,
            rightMargin=1.5*cm,
            leftMargin=1.5*cm,
            topMargin=1.5*cm,
            bottomMargin=1.5*cm,
        )

        doc.build(self.story)
        print(f"PDF report generated: {self.output_path}")


def create_2up_pdf(input_pdf_path, output_pdf_path):
    """
    Convert a PDF to 2-up format: 2 pages side-by-side on A4 landscape

    Args:
        input_pdf_path: Path to the input PDF
        output_pdf_path: Path to save the 2-up PDF

    Raises:
        ImportError: If pypdf is not available
    """
    if not PYPDF_AVAILABLE:
        raise ImportError("pypdf library is required for 2-up layout. Install with: pip install pypdf")

    reader = PdfReader(input_pdf_path)
    writer = PdfWriter()

    # A4 landscape dimensions in points (1 point = 1/72 inch)
    a4_width, a4_height = landscape(A4)  # 842 x 595 points

    # Process pages in pairs
    num_pages = len(reader.pages)
    for i in range(0, num_pages, 2):
        # Create new blank page (A4 landscape)
        blank_page = writer.add_blank_page(width=a4_width, height=a4_height)

        # Calculate scaling to fit A5 size (half of A4 landscape width)
        target_width = a4_width / 2
        target_height = a4_height

        # Get first page (left side)
        page1 = reader.pages[i]
        orig_width = float(page1.mediabox.width)
        orig_height = float(page1.mediabox.height)
        scale = min(target_width / orig_width, target_height / orig_height)

        # Calculate centering offsets
        scaled_width = orig_width * scale
        scaled_height = orig_height * scale
        offset_x_left = (target_width - scaled_width) / 2
        offset_y = (target_height - scaled_height) / 2

        # Create transformation for first page (left side)
        transformation_left = Transformation().scale(sx=scale, sy=scale).translate(tx=offset_x_left, ty=offset_y)
        blank_page.merge_transformed_page(page1, transformation_left, expand=False)

        # Get second page (right side) if it exists
        if i + 1 < num_pages:
            page2 = reader.pages[i + 1]

            # Calculate offset for right page
            offset_x_right = target_width + (target_width - scaled_width) / 2

            # Create transformation for second page (right side)
            transformation_right = Transformation().scale(sx=scale, sy=scale).translate(tx=offset_x_right, ty=offset_y)
            blank_page.merge_transformed_page(page2, transformation_right, expand=False)

    # Write output
    with open(output_pdf_path, 'wb') as output_file:
        writer.write(output_file)


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
    Generate a LaTeX-style controller optimization training report

    Args:
        config: Configuration dictionary
        training_history: Training history dictionary
        final_metrics: Final metrics dictionary
        process_metrics: Process-wise metrics dictionary
        F_star: Target reliability
        F_baseline: Baseline reliability
        F_actual: Actual controller reliability
        checkpoint_dir: Directory to save the report
        timestamp: Training timestamp (optional)

    Returns:
        Path to the generated PDF report
    """
    if timestamp is None:
        timestamp = datetime.now()

    checkpoint_dir = Path(checkpoint_dir)
    final_report_path = checkpoint_dir / 'controller_report.pdf'

    # Try to generate 2-up layout if pypdf is available
    if PYPDF_AVAILABLE:
        temp_report_path = checkpoint_dir / 'controller_report_temp.pdf'

        # Generate the original PDF
        generator = ControllerReportGenerator(temp_report_path)
        generator.generate(config, training_history, final_metrics, process_metrics,
                          F_star, F_baseline, F_actual, timestamp)

        # Convert to 2-up format
        try:
            create_2up_pdf(temp_report_path, final_report_path)
            temp_report_path.unlink()
            print(f"2-up PDF report generated: {final_report_path}")
        except Exception as e:
            print(f"Warning: Failed to create 2-up layout: {e}")
            print(f"Falling back to standard layout")
            # If 2-up fails, rename temp to final
            temp_report_path.rename(final_report_path)
            print(f"PDF report generated: {final_report_path}")
    else:
        # Generate standard PDF without 2-up layout
        print("Note: pypdf not available, generating standard layout (install pypdf for 2-up layout)")
        generator = ControllerReportGenerator(final_report_path)
        generator.generate(config, training_history, final_metrics, process_metrics,
                          F_star, F_baseline, F_actual, timestamp)

    return final_report_path


if __name__ == '__main__':
    # Test report generation
    print("Testing controller report generation...")

    # Dummy data
    config = {
        'process_names': ['laser', 'plasma', 'galvanic'],
        'policy_generator': {
            'architecture': 'medium',
            'hidden_sizes': [128, 64],
            'activation': 'relu',
            'dropout_rate': 0.1
        },
        'training': {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'lambda_bc': 0.1,
            'patience': 10,
            'device': 'cpu',
            'checkpoint_dir': 'test_report'
        },
        'misc': {
            'random_seed': 42,
            'verbose': True
        }
    }

    training_history = {
        'total_loss': [1.0, 0.8, 0.6, 0.5, 0.4],
        'reliability_loss': [0.5, 0.4, 0.3, 0.25, 0.2],
        'bc_loss': [0.5, 0.4, 0.3, 0.25, 0.2],
        'F_values': [0.7, 0.75, 0.8, 0.85, 0.9]
    }

    final_metrics = {
        'improvement': 0.15,
        'target_gap': 0.05
    }

    process_metrics = {
        'actual': {
            'laser': {'input_mse': 0.001, 'output_mse': 0.002, 'combined_mse': 0.0015},
            'plasma': {'input_mse': 0.003, 'output_mse': 0.004, 'combined_mse': 0.0035},
            'galvanic': {'input_mse': 0.002, 'output_mse': 0.003, 'combined_mse': 0.0025}
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
