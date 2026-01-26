"""
PDF Report Generator for Uncertainty Quantification Training

LaTeX-style report with two-column layout for UQ models
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


class UncertaintyReportGenerator:
    """LaTeX-style PDF report generator for uncertainty quantification training results"""

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
        title = Paragraph("<b>Uncertainty Quantification Training Report</b>", self.styles['ReportTitle'])
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

    def create_two_column_section(self, config, history, metrics, input_dim, output_dim,
                                  total_params, n_train, n_val, n_test, coverage_results=None):
        """Create two-column layout for compact info"""

        # Left column data
        left_col = []

        # Model Configuration
        left_col.append(Paragraph("<b>Model Configuration</b>", self.styles['SectionTitle']))
        left_col.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=4))

        # Check if ensemble mode
        use_ensemble = config['model'].get('use_ensemble', False)
        if use_ensemble:
            n_ensemble = config['model'].get('n_ensemble_models', 5)
            params_per_model = total_params // n_ensemble if n_ensemble > 0 else total_params
            model_text = f"""• <b>Type:</b> Deep Ensemble<br/>
• <b>Base Architecture:</b> {config['model']['model_type']}<br/>
• <b>Models in Ensemble:</b> {n_ensemble}<br/>
• <b>Hidden Layers:</b> {config['model']['hidden_sizes']}<br/>
• <b>Dropout Rate:</b> {config['model']['dropout_rate']}<br/>
• <b>Batch Normalization:</b> {config['model']['use_batchnorm']}<br/>
• <b>Min Variance:</b> {config['model']['min_variance']}<br/>
• <b>Input Dimension:</b> {input_dim}<br/>
• <b>Output Dimension:</b> {output_dim}<br/>
• <b>Params per Model:</b> {params_per_model:,}<br/>
• <b>Total Parameters:</b> {total_params:,}"""
        else:
            model_text = f"""• <b>Type:</b> {config['model']['model_type']}<br/>
• <b>Hidden Layers:</b> {config['model']['hidden_sizes']}<br/>
• <b>Dropout Rate:</b> {config['model']['dropout_rate']}<br/>
• <b>Batch Normalization:</b> {config['model']['use_batchnorm']}<br/>
• <b>Min Variance:</b> {config['model']['min_variance']}<br/>
• <b>Input Dimension:</b> {input_dim}<br/>
• <b>Output Dimension:</b> {output_dim}<br/>
• <b>Total Parameters:</b> {total_params:,}"""
        left_col.append(Paragraph(model_text, self.styles['BodyText']))
        left_col.append(Spacer(1, 0.15*cm))

        # Dataset
        left_col.append(Paragraph("<b>Dataset</b>", self.styles['SectionTitle']))
        left_col.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=4))

        input_cols = ', '.join(config['data']['input_columns'])
        output_cols = ', '.join(config['data']['output_columns'])
        dataset_text = f"""• <b>File:</b> {config['data']['csv_path']}<br/>
• <b>Input Columns:</b> {input_cols}<br/>
• <b>Output Columns:</b> {output_cols}<br/>
• <b>Train Samples:</b> {n_train:,} ({config['data']['train_size']*100:.0f}%)<br/>
• <b>Validation Samples:</b> {n_val:,} ({config['data']['val_size']*100:.0f}%)<br/>
• <b>Test Samples:</b> {n_test:,} ({config['data']['test_size']*100:.0f}%)<br/>
• <b>Scaling Method:</b> {config['data']['scaling_method']}<br/>
• <b>Random State:</b> {config['data']['random_state']}"""
        left_col.append(Paragraph(dataset_text, self.styles['BodyText']))

        # Right column data
        right_col = []

        # Training Parameters
        right_col.append(Paragraph("<b>Training Parameters</b>", self.styles['SectionTitle']))
        right_col.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=4))

        epochs_run = len(history['train_losses'])
        epochs_total = config['training']['epochs']

        # Loss function parameters
        loss_type = config['training'].get('loss_type', 'gaussian_nll')
        if loss_type == 'gaussian_nll':
            loss_params = f"""• <b>Loss Function:</b> Gaussian NLL<br/>
• <b>Variance Penalty (α):</b> {config['training'].get('variance_penalty_alpha', 1.0)}"""
        elif loss_type == 'energy_score':
            loss_params = f"""• <b>Loss Function:</b> Energy Score<br/>
• <b>MC Samples:</b> {config['training'].get('energy_score_samples', 50)}<br/>
• <b>Diversity Penalty (β):</b> {config['training'].get('energy_score_beta', 1.0)}"""
        else:
            loss_params = f"• <b>Loss Function:</b> {loss_type}"

        training_text = f"""• <b>Epochs:</b> {epochs_run}/{epochs_total}<br/>
• <b>Batch Size:</b> {config['training']['batch_size']}<br/>
• <b>Learning Rate:</b> {config['training']['learning_rate']}<br/>
• <b>Weight Decay:</b> {config['training']['weight_decay']}<br/>
{loss_params}<br/>
• <b>Patience:</b> {config['training']['patience']}<br/>
• <b>Device:</b> {config['training']['device']}<br/>
• <b>Checkpoint Dir:</b> {config['training']['checkpoint_dir']}"""
        right_col.append(Paragraph(training_text, self.styles['BodyText']))
        right_col.append(Spacer(1, 0.15*cm))

        # Training Results
        right_col.append(Paragraph("<b>Training Results</b>", self.styles['SectionTitle']))
        right_col.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=4))

        final_train_loss = history['train_losses'][-1]
        final_val_loss = history['val_losses'][-1]
        best_val_loss = min(history['val_losses'])

        final_train_mse = history['train_mse'][-1] if 'train_mse' in history else 0.0
        final_val_mse = history['val_mse'][-1] if 'val_mse' in history else 0.0

        # Use generic "Loss" label instead of NLL for compatibility
        loss_label = "Loss" if loss_type == 'energy_score' else "NLL"
        results_text = f"""• <b>Final Train {loss_label}:</b> {final_train_loss:.6f}<br/>
• <b>Final Val {loss_label}:</b> {final_val_loss:.6f}<br/>
• <b>Best Val {loss_label}:</b> {best_val_loss:.6f}<br/>
• <b>Final Train MSE:</b> {final_train_mse:.6f}<br/>
• <b>Final Val MSE:</b> {final_val_mse:.6f}"""

        # Add coverage information if available
        if coverage_results is not None:
            results_text += f"""<br/>• <b>Expected Coverage:</b> {coverage_results['expected_coverage']:.1f}%<br/>
• <b>Actual Coverage:</b> {coverage_results['actual_coverage']:.1f}%<br/>
• <b>Coverage Error:</b> {coverage_results['coverage_error']:.1f}%<br/>
• <b>Well Calibrated:</b> {'Yes' if coverage_results['well_calibrated'] else 'No'}"""

            # Add ensemble-specific uncertainty decomposition
            if 'mean_aleatoric' in coverage_results:
                results_text += f"""<br/><br/><b>Uncertainty Decomposition:</b><br/>
• <b>Mean Aleatoric:</b> {coverage_results['mean_aleatoric']:.6f}<br/>
• <b>Mean Epistemic:</b> {coverage_results['mean_epistemic']:.6f}<br/>
• <b>Epistemic Ratio:</b> {coverage_results['epistemic_ratio']:.1f}%"""

        right_col.append(Paragraph(results_text, self.styles['BodyText']))
        right_col.append(Spacer(1, 0.15*cm))

        # Uncertainty Parameters
        right_col.append(Paragraph("<b>Uncertainty Parameters</b>", self.styles['SectionTitle']))
        right_col.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=4))

        uncertainty_text = f"""• <b>Confidence Level:</b> {config['uncertainty']['confidence_level']*100:.0f}%"""
        right_col.append(Paragraph(uncertainty_text, self.styles['BodyText']))
        right_col.append(Spacer(1, 0.15*cm))

        # Miscellaneous Parameters
        right_col.append(Paragraph("<b>Miscellaneous</b>", self.styles['SectionTitle']))
        right_col.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=4))

        misc_text = f"""• <b>Random Seed:</b> {config['misc']['random_seed']}"""
        if 'verbose' in config['misc']:
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

    def add_metrics_table(self, metrics):
        """Add metrics table with uncertainty-specific metrics"""
        self.add_section_title("Test Metrics")

        # Build table data with uncertainty metrics
        headers = ['Output', 'MSE', 'RMSE', 'MAE', 'R²', 'Mean Var', 'Cal. Ratio', 'NLL']
        data = [headers]

        for output_name, output_metrics in metrics.items():
            if output_name == 'Overall':
                continue

            # Standard metrics
            row = [
                output_name,
                f"{output_metrics['MSE']:.4f}",
                f"{output_metrics['RMSE']:.4f}",
                f"{output_metrics['MAE']:.4f}",
                f"{output_metrics['R2']:.4f}",
            ]

            # Uncertainty metrics
            if 'Mean_Variance' in output_metrics:
                row.append(f"{output_metrics['Mean_Variance']:.4f}")
                row.append(f"{output_metrics['Calibration_Ratio']:.4f}")
                row.append(f"{output_metrics['NLL']:.4f}")
            else:
                row.extend(['N/A', 'N/A', 'N/A'])

            data.append(row)

        # Create table with adjusted column widths
        col_widths = [2.2*cm, 1.9*cm, 1.9*cm, 1.9*cm, 1.9*cm, 1.9*cm, 1.9*cm, 1.9*cm]
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




    def add_scm_graph(self, checkpoint_dir):
        """Add SCM graph visualization if available"""
        checkpoint_dir = Path(checkpoint_dir)
        scm_graph = checkpoint_dir / 'scm_graph.png'

        if scm_graph.exists():
            self.add_section_title("Structural Causal Model (SCM)")

            img = Image(str(scm_graph))
            img_width, img_height = img.imageWidth, img.imageHeight
            aspect_ratio = img_height / img_width

            # Set appropriate size for SCM graph
            new_width = 10*cm
            new_height = new_width * aspect_ratio

            # Max height constraint
            if new_height > 7*cm:
                new_height = 7*cm
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

            caption = Paragraph("<i>Structural Causal Model - Causal Graph Structure</i>", self.styles['Normal'])
            caption_table = Table([[caption]], colWidths=[18*cm])
            caption_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ]))
            self.story.append(caption_table)
            self.story.append(Spacer(1, 0.2*cm))

    def add_plots_stacked(self, checkpoint_dir):
        """Add uncertainty-specific plots stacked vertically"""
        self.add_section_title("Training Visualization")

        checkpoint_dir = Path(checkpoint_dir)

        # Training history plot
        history_plot = checkpoint_dir / 'training_history.png'
        if history_plot.exists():
            img1 = Image(str(history_plot))
            img_width, img_height = img1.imageWidth, img1.imageHeight
            aspect_ratio = img_height / img_width

            # Larger width for stacked layout
            new_width = 16*cm
            new_height = new_width * aspect_ratio

            # Max height constraint
            if new_height > 10*cm:
                new_height = 10*cm
                new_width = new_height / aspect_ratio

            img1.drawWidth = new_width
            img1.drawHeight = new_height

            # Center the image
            img_table = Table([[img1]], colWidths=[18*cm])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            self.story.append(img_table)

            caption1 = Paragraph("<i>Training and Validation Loss (NLL and MSE)</i>", self.styles['Normal'])
            caption_table = Table([[caption1]], colWidths=[18*cm])
            caption_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ]))
            self.story.append(caption_table)
            self.story.append(Spacer(1, 0.15*cm))

        # Predictions with uncertainty plots - side by side layout
        predictions_plot = checkpoint_dir / 'predictions_with_uncertainty.png'
        training_predictions_plot = checkpoint_dir / 'training_predictions_with_uncertainty.png'

        if predictions_plot.exists() and training_predictions_plot.exists():
            # Load both images
            img2 = Image(str(predictions_plot))
            img3 = Image(str(training_predictions_plot))

            # Calculate dimensions for side-by-side layout
            img_width, img_height = img2.imageWidth, img2.imageHeight
            aspect_ratio = img_height / img_width

            # Each image gets half the page width (with some spacing)
            new_width = 8.5*cm
            new_height = new_width * aspect_ratio

            # Max height constraint
            if new_height > 8*cm:
                new_height = 8*cm
                new_width = new_height / aspect_ratio

            img2.drawWidth = new_width
            img2.drawHeight = new_height
            img3.drawWidth = new_width
            img3.drawHeight = new_height

            # Create two-column table for images
            img_table = Table([[img2, img3]], colWidths=[9*cm, 9*cm])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            self.story.append(img_table)

            # Create two-column table for captions
            caption_left = Paragraph("<i>Validation Predictions with Uncertainty Bounds</i>", self.styles['Normal'])
            caption_right = Paragraph("<i>Training Data with Uncertainty Bounds</i>", self.styles['Normal'])
            caption_table = Table([[caption_left, caption_right]], colWidths=[9*cm, 9*cm])
            caption_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ]))
            self.story.append(caption_table)
            self.story.append(Spacer(1, 0.15*cm))
        elif predictions_plot.exists():
            # Fallback to single plot if training plot doesn't exist
            img2 = Image(str(predictions_plot))
            img_width, img_height = img2.imageWidth, img2.imageHeight
            aspect_ratio = img_height / img_width

            new_width = 16*cm
            new_height = new_width * aspect_ratio

            if new_height > 10*cm:
                new_height = 10*cm
                new_width = new_height / aspect_ratio

            img2.drawWidth = new_width
            img2.drawHeight = new_height

            img_table = Table([[img2]], colWidths=[18*cm])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            self.story.append(img_table)

            caption2 = Paragraph("<i>Predictions with Uncertainty Bounds</i>", self.styles['Normal'])
            caption_table = Table([[caption2]], colWidths=[18*cm])
            caption_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ]))
            self.story.append(caption_table)
            self.story.append(Spacer(1, 0.15*cm))

        # Scatter plot with uncertainty coloring
        scatter_plot = checkpoint_dir / 'scatter_with_uncertainty.png'
        if scatter_plot.exists():
            img3 = Image(str(scatter_plot))
            img_width, img_height = img3.imageWidth, img3.imageHeight
            aspect_ratio = img_height / img_width

            # Larger width for stacked layout
            new_width = 18*cm
            new_height = new_width * aspect_ratio

            # Max height constraint
            if new_height > 14*cm:
                new_height = 14*cm
                new_width = new_height / aspect_ratio

            img3.drawWidth = new_width
            img3.drawHeight = new_height

            # Center the image
            img_table = Table([[img3]], colWidths=[18*cm])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            self.story.append(img_table)

            caption3 = Paragraph("<i>Scatter Plot with Uncertainty Coloring</i>", self.styles['Normal'])
            caption_table = Table([[caption3]], colWidths=[18*cm])
            caption_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ]))
            self.story.append(caption_table)
            self.story.append(Spacer(1, 0.15*cm))

        # Ensemble-specific plots (stacked uncertainty and decomposition side by side)
        stacked_plot = checkpoint_dir / 'predictions_stacked_uncertainty.png'
        decomp_plot = checkpoint_dir / 'uncertainty_decomposition.png'

        if stacked_plot.exists() and decomp_plot.exists():
            # Load both images
            img_stacked = Image(str(stacked_plot))
            img_decomp = Image(str(decomp_plot))

            # Calculate dimensions for side-by-side layout
            img_width, img_height = img_stacked.imageWidth, img_stacked.imageHeight
            aspect_ratio = img_height / img_width

            # Each image gets half the page width
            new_width = 8.5*cm
            new_height = new_width * aspect_ratio

            # Max height constraint
            if new_height > 7*cm:
                new_height = 7*cm
                new_width = new_height / aspect_ratio

            img_stacked.drawWidth = new_width
            img_stacked.drawHeight = new_height

            # Decomposition plot may have different aspect ratio
            img_width2, img_height2 = img_decomp.imageWidth, img_decomp.imageHeight
            aspect_ratio2 = img_height2 / img_width2
            new_width2 = 8.5*cm
            new_height2 = new_width2 * aspect_ratio2
            if new_height2 > 7*cm:
                new_height2 = 7*cm
                new_width2 = new_height2 / aspect_ratio2

            img_decomp.drawWidth = new_width2
            img_decomp.drawHeight = new_height2

            # Create two-column table for images
            img_table = Table([[img_stacked, img_decomp]], colWidths=[9*cm, 9*cm])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            self.story.append(img_table)

            # Captions
            caption_left = Paragraph("<i>Stacked Uncertainty (Aleatoric + Epistemic)</i>", self.styles['Normal'])
            caption_right = Paragraph("<i>Uncertainty Decomposition</i>", self.styles['Normal'])
            caption_table = Table([[caption_left, caption_right]], colWidths=[9*cm, 9*cm])
            caption_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ]))
            self.story.append(caption_table)
            self.story.append(Spacer(1, 0.15*cm))

        elif stacked_plot.exists():
            # Only stacked plot exists
            img_stacked = Image(str(stacked_plot))
            img_width, img_height = img_stacked.imageWidth, img_stacked.imageHeight
            aspect_ratio = img_height / img_width

            new_width = 16*cm
            new_height = new_width * aspect_ratio
            if new_height > 10*cm:
                new_height = 10*cm
                new_width = new_height / aspect_ratio

            img_stacked.drawWidth = new_width
            img_stacked.drawHeight = new_height

            img_table = Table([[img_stacked]], colWidths=[18*cm])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            self.story.append(img_table)

            caption = Paragraph("<i>Stacked Uncertainty Bands (Aleatoric + Epistemic)</i>", self.styles['Normal'])
            caption_table = Table([[caption]], colWidths=[18*cm])
            caption_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ]))
            self.story.append(caption_table)
            self.story.append(Spacer(1, 0.15*cm))

    def generate(self, config, history, metrics, input_dim, output_dim,
                total_params, n_train, n_val, n_test, timestamp, coverage_results=None):
        """Generate the complete PDF"""

        # Add all sections
        self.add_title(timestamp)
        self.create_two_column_section(config, history, metrics, input_dim, output_dim,
                                       total_params, n_train, n_val, n_test, coverage_results)
        self.add_metrics_table(metrics)
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
        # Apply scale first, then translate (operations are read right-to-left in matrix multiplication)
        transformation_left = Transformation().scale(sx=scale, sy=scale).translate(tx=offset_x_left, ty=offset_y)
        blank_page.merge_transformed_page(page1, transformation_left, expand=False)

        # Get second page (right side) if it exists
        if i + 1 < num_pages:
            page2 = reader.pages[i + 1]

            # Calculate offset for right page
            offset_x_right = target_width + (target_width - scaled_width) / 2

            # Create transformation for second page (right side)
            # Apply scale first, then translate
            transformation_right = Transformation().scale(sx=scale, sy=scale).translate(tx=offset_x_right, ty=offset_y)
            blank_page.merge_transformed_page(page2, transformation_right, expand=False)

    # Write output
    with open(output_pdf_path, 'wb') as output_file:
        writer.write(output_file)


def generate_uncertainty_training_report(
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
    timestamp=None,
    coverage_results=None
):
    """
    Generate a LaTeX-style uncertainty quantification training report

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
        coverage_results: Prediction interval coverage results (optional)
    """
    if timestamp is None:
        timestamp = datetime.now()

    checkpoint_dir = Path(checkpoint_dir)
    final_report_path = checkpoint_dir / 'training_report.pdf'

    # Try to generate 2-up layout if pypdf is available
    if PYPDF_AVAILABLE:
        temp_report_path = checkpoint_dir / 'training_report_temp.pdf'

        # Generate the original PDF
        generator = UncertaintyReportGenerator(temp_report_path)
        generator.generate(config, history, metrics, input_dim, output_dim,
                          total_params, n_train, n_val, n_test, timestamp, coverage_results)

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
        generator = UncertaintyReportGenerator(final_report_path)
        generator.generate(config, history, metrics, input_dim, output_dim,
                          total_params, n_train, n_val, n_test, timestamp, coverage_results)

    return final_report_path


def combine_process_reports(report_paths, output_path, process_names=None):
    """
    Combine multiple process reports into a single PDF

    Args:
        report_paths: List of paths to individual process reports
        output_path: Path to save the combined report
        process_names: Optional list of process names (for display)

    Returns:
        Path to the combined report
    """
    if not PYPDF_AVAILABLE:
        print("Warning: pypdf not available, cannot combine reports")
        return None

    writer = PdfWriter()

    # Add all pages from all reports
    for i, report_path in enumerate(report_paths):
        report_path = Path(report_path)
        if not report_path.exists():
            process_name = process_names[i] if process_names else f"Process {i+1}"
            print(f"Warning: Report not found for {process_name}: {report_path}")
            continue

        try:
            reader = PdfReader(report_path)
            process_name = process_names[i] if process_names else f"Process {i+1}"
            print(f"Adding {len(reader.pages)} page(s) from {process_name} report...")

            for page in reader.pages:
                writer.add_page(page)

        except Exception as e:
            process_name = process_names[i] if process_names else f"Process {i+1}"
            print(f"Error reading report for {process_name}: {e}")
            continue

    # Write combined PDF
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as output_file:
        writer.write(output_file)

    print(f"\n✓ Combined report generated: {output_path}")
    print(f"  Total pages: {len(writer.pages)}")

    return output_path
