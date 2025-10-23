"""
PDF Report Generator for Neural Network Training

LaTeX-style report with two-column layout
"""

from datetime import datetime
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, Frame, PageTemplate, KeepTogether
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus.flowables import HRFlowable


class TrainingReportGenerator:
    """LaTeX-style PDF report generator for training results"""

    def __init__(self, output_path):
        self.output_path = Path(output_path)
        self.styles = getSampleStyleSheet()
        self.story = []

        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=4
        ))

        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='ReportSubtitle',
            parent=self.styles['Normal'],
            fontSize=13,
            leading=16,
            alignment=TA_CENTER,
            spaceAfter=8
        ))

        # Section title style
        self.styles.add(ParagraphStyle(
            name='SectionTitle',
            parent=self.styles['Heading2'],
            fontSize=12,
            leading=14,
            fontName='Helvetica-Bold',
            spaceAfter=2,
            spaceBefore=6
        ))

        # Body text style
        self.styles.add(ParagraphStyle(
            name='BodyText',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=13,
            leftIndent=10
        ))

    def add_title(self, timestamp):
        """Add centered title and date"""
        title = Paragraph("<b>Neural Network Training Report</b>", self.styles['ReportTitle'])
        date_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        subtitle = Paragraph(f"Date: {date_str}", self.styles['ReportSubtitle'])

        self.story.append(title)
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.2*cm))

    def add_section_title(self, title):
        """Add section title with horizontal line"""
        para = Paragraph(f"<b>{title}</b>", self.styles['SectionTitle'])
        line = HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=4)
        self.story.append(para)
        self.story.append(line)

    def create_two_column_section(self, config, history, metrics, input_dim, output_dim,
                                  total_params, n_train, n_val, n_test):
        """Create two-column layout for compact info"""

        # Left column data
        left_col = []

        # Model Configuration
        left_col.append(Paragraph("<b>Model Configuration</b>", self.styles['SectionTitle']))
        left_col.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=4))

        model_text = f"""• <b>Type:</b> {config['model']['model_type']}<br/>
• <b>Hidden Layers:</b> {config['model']['hidden_sizes']}<br/>
• <b>Dropout Rate:</b> {config['model']['dropout_rate']}<br/>
• <b>Input Dimension:</b> {input_dim}<br/>
• <b>Output Dimension:</b> {output_dim}<br/>
• <b>Total Parameters:</b> {total_params:,}"""
        left_col.append(Paragraph(model_text, self.styles['BodyText']))
        left_col.append(Spacer(1, 0.3*cm))

        # Dataset
        left_col.append(Paragraph("<b>Dataset</b>", self.styles['SectionTitle']))
        left_col.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=4))

        dataset_text = f"""• <b>File:</b> {config['data']['csv_path']}<br/>
• <b>Train Samples:</b> {n_train:,}<br/>
• <b>Validation Samples:</b> {n_val:,}<br/>
• <b>Test Samples:</b> {n_test:,}<br/>
• <b>Scaling Method:</b> {config['data']['scaling_method']}"""
        left_col.append(Paragraph(dataset_text, self.styles['BodyText']))

        # Right column data
        right_col = []

        # Training Parameters
        right_col.append(Paragraph("<b>Training Parameters</b>", self.styles['SectionTitle']))
        right_col.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=4))

        epochs_run = len(history['train_losses'])
        epochs_total = config['training']['epochs']
        training_text = f"""• <b>Epochs:</b> {epochs_run}/{epochs_total}<br/>
• <b>Batch Size:</b> {config['training']['batch_size']}<br/>
• <b>Learning Rate:</b> {config['training']['learning_rate']}<br/>
• <b>Loss Function:</b> {config['training']['loss_function'].upper()}<br/>
• <b>Device:</b> {config['training']['device']}"""
        right_col.append(Paragraph(training_text, self.styles['BodyText']))
        right_col.append(Spacer(1, 0.3*cm))

        # Training Results
        right_col.append(Paragraph("<b>Training Results</b>", self.styles['SectionTitle']))
        right_col.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=4))

        final_train_loss = history['train_losses'][-1]
        final_val_loss = history['val_losses'][-1]
        best_val_loss = min(history['val_losses'])

        results_text = f"""• <b>Final Train Loss:</b> {final_train_loss:.6f}<br/>
• <b>Final Val Loss:</b> {final_val_loss:.6f}<br/>
• <b>Best Val Loss:</b> {best_val_loss:.6f}"""
        right_col.append(Paragraph(results_text, self.styles['BodyText']))

        # Create two-column table
        data = [[left_col, right_col]]
        col_table = Table(data, colWidths=[9*cm, 9*cm])
        col_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ]))

        self.story.append(col_table)
        self.story.append(Spacer(1, 0.4*cm))

    def add_metrics_table(self, metrics):
        """Add metrics table"""
        self.add_section_title("Test Metrics")

        # Build table data
        headers = ['Output', 'MAE', 'RMSE', 'R²']
        data = [headers]

        for output_name, output_metrics in metrics.items():
            row = [
                output_name,
                f"{output_metrics['MAE']:.4f}",
                f"{output_metrics['RMSE']:.4f}",
                f"{output_metrics['R2']:.4f}"
            ]
            data.append(row)

        # Create table
        table = Table(data, colWidths=[4*cm, 3*cm, 3*cm, 3*cm])
        table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),

            # Data rows
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),

            # Top and bottom lines
            ('LINEABOVE', (0, 0), (-1, 0), 1.5, colors.black),
            ('LINEABOVE', (0, 1), (-1, 1), 0.5, colors.black),
            ('LINEBELOW', (0, -1), (-1, -1), 1.5, colors.black),
        ]))

        self.story.append(table)
        self.story.append(Spacer(1, 0.4*cm))

    def add_plots_side_by_side(self, checkpoint_dir):
        """Add two plots side by side"""
        self.add_section_title("Training Visualization")

        checkpoint_dir = Path(checkpoint_dir)

        # Prepare images
        history_plot = checkpoint_dir / 'training_history.png'
        predictions_plot = checkpoint_dir / 'predictions.png'

        images = []

        if history_plot.exists():
            img1 = Image(str(history_plot))
            img_width, img_height = img1.imageWidth, img1.imageHeight
            aspect_ratio = img_height / img_width

            new_width = 8.5*cm
            new_height = new_width * aspect_ratio

            if new_height > 6*cm:
                new_height = 6*cm
                new_width = new_height / aspect_ratio

            img1.drawWidth = new_width
            img1.drawHeight = new_height

            caption1 = Paragraph("<i>Training and Validation Loss</i>", self.styles['Normal'])
            images.append([img1, caption1])

        if predictions_plot.exists():
            img2 = Image(str(predictions_plot))
            img_width, img_height = img2.imageWidth, img2.imageHeight
            aspect_ratio = img_height / img_width

            new_width = 8.5*cm
            new_height = new_width * aspect_ratio

            if new_height > 6*cm:
                new_height = 6*cm
                new_width = new_height / aspect_ratio

            img2.drawWidth = new_width
            img2.drawHeight = new_height

            caption2 = Paragraph("<i>Predictions vs True Values</i>", self.styles['Normal'])
            images.append([img2, caption2])

        # Create side-by-side layout
        if len(images) == 2:
            img_table_data = [[images[0][0], images[1][0]]]
            caption_table_data = [[images[0][1], images[1][1]]]

            img_table = Table(img_table_data, colWidths=[9*cm, 9*cm])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))

            caption_table = Table(caption_table_data, colWidths=[9*cm, 9*cm])
            caption_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))

            self.story.append(img_table)
            self.story.append(caption_table)

    def generate(self, config, history, metrics, input_dim, output_dim,
                total_params, n_train, n_val, n_test, timestamp):
        """Generate the complete PDF"""

        # Add all sections
        self.add_title(timestamp)
        self.create_two_column_section(config, history, metrics, input_dim, output_dim,
                                       total_params, n_train, n_val, n_test)
        self.add_metrics_table(metrics)
        self.add_plots_side_by_side(Path(config['training']['checkpoint_dir']))

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


def generate_training_report(
    config,
    history,
    metrics,
    input_dim,
    output_dim,
    total_params,
    n_train,
    n_val,
    n_test,
    checkpoint_dir,
    timestamp=None
):
    """
    Generate a LaTeX-style training report

    Args:
        config: Configuration dictionary
        history: Training history dictionary
        metrics: Calculated metrics dictionary
        input_dim: Input dimension
        output_dim: Output dimension
        total_params: Total number of parameters
        n_train: Number of training samples
        n_val: Number of validation samples
        n_test: Number of test samples
        checkpoint_dir: Directory to save the report
        timestamp: Training timestamp (optional)
    """
    if timestamp is None:
        timestamp = datetime.now()

    checkpoint_dir = Path(checkpoint_dir)
    report_path = checkpoint_dir / 'training_report.pdf'

    generator = TrainingReportGenerator(report_path)
    generator.generate(config, history, metrics, input_dim, output_dim,
                      total_params, n_train, n_val, n_test, timestamp)

    return report_path
