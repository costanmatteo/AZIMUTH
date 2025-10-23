"""
PDF Report Generator for Neural Network Training

Simple, text-based report with plots
"""

from datetime import datetime
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.enums import TA_LEFT


class TrainingReportGenerator:
    """Simple PDF report generator for training results"""

    def __init__(self, output_path):
        self.output_path = Path(output_path)
        self.styles = getSampleStyleSheet()
        self.story = []

        # Create custom text style with font size 15
        self.styles.add(ParagraphStyle(
            name='CustomText',
            parent=self.styles['Normal'],
            fontSize=15,
            leading=18,
            alignment=TA_LEFT
        ))

    def add_text_content(self, config, history, metrics, input_dim, output_dim,
                        total_params, n_train, n_val, n_test, timestamp):
        """Add all text information in a compact format"""

        # Build text content
        text = f"<b>Neural Network Training Report</b><br/>"
        text += f"Date: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}<br/><br/>"

        # Model configuration
        text += f"<b>Model Configuration:</b><br/>"
        text += f"Type: {config['model']['model_type']}, "
        text += f"Hidden layers: {config['model']['hidden_sizes']}, "
        text += f"Dropout: {config['model']['dropout_rate']}<br/>"
        text += f"Input dim: {input_dim}, Output dim: {output_dim}, "
        text += f"Total parameters: {total_params:,}<br/><br/>"

        # Dataset info
        text += f"<b>Dataset:</b><br/>"
        text += f"File: {config['data']['csv_path']}<br/>"
        text += f"Train samples: {n_train}, Validation: {n_val}, Test: {n_test}<br/>"
        text += f"Scaling: {config['data']['scaling_method']}<br/><br/>"

        # Training parameters
        text += f"<b>Training:</b><br/>"
        text += f"Epochs: {len(history['train_losses'])}/{config['training']['epochs']}, "
        text += f"Batch size: {config['training']['batch_size']}, "
        text += f"Learning rate: {config['training']['learning_rate']}<br/>"
        text += f"Loss function: {config['training']['loss_function'].upper()}, "
        text += f"Device: {config['training']['device']}<br/>"

        # Final losses
        final_train_loss = history['train_losses'][-1]
        final_val_loss = history['val_losses'][-1]
        best_val_loss = min(history['val_losses'])
        text += f"Final train loss: {final_train_loss:.6f}, "
        text += f"Final val loss: {final_val_loss:.6f}, "
        text += f"Best val loss: {best_val_loss:.6f}<br/><br/>"

        # Metrics
        text += f"<b>Test Metrics:</b><br/>"
        for output_name, output_metrics in metrics.items():
            text += f"{output_name}: "
            text += f"MAE={output_metrics['MAE']:.4f}, "
            text += f"RMSE={output_metrics['RMSE']:.4f}, "
            text += f"R²={output_metrics['R2']:.4f}<br/>"

        # Add as paragraph
        para = Paragraph(text, self.styles['CustomText'])
        self.story.append(para)
        self.story.append(Spacer(1, 0.2*inch))

    def add_plots(self, checkpoint_dir):
        """Add the two plots side by side or stacked"""
        checkpoint_dir = Path(checkpoint_dir)

        # Training history plot
        history_plot = checkpoint_dir / 'training_history.png'
        if history_plot.exists():
            img = Image(str(history_plot))
            # Calculate size to fit
            img_width, img_height = img.imageWidth, img.imageHeight
            aspect_ratio = img_height / img_width

            new_width = 3.5*inch
            new_height = new_width * aspect_ratio

            if new_height > 2.5*inch:
                new_height = 2.5*inch
                new_width = new_height / aspect_ratio

            img.drawWidth = new_width
            img.drawHeight = new_height
            self.story.append(img)
            self.story.append(Spacer(1, 0.1*inch))

        # Predictions plot
        predictions_plot = checkpoint_dir / 'predictions.png'
        if predictions_plot.exists():
            img = Image(str(predictions_plot))
            img_width, img_height = img.imageWidth, img.imageHeight
            aspect_ratio = img_height / img_width

            new_width = 3.5*inch
            new_height = new_width * aspect_ratio

            if new_height > 2.5*inch:
                new_height = 2.5*inch
                new_width = new_height / aspect_ratio

            img.drawWidth = new_width
            img.drawHeight = new_height
            self.story.append(img)

    def generate(self):
        """Generate the PDF"""
        doc = SimpleDocTemplate(
            str(self.output_path),
            pagesize=A4,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
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
    Generate a simple training report

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

    # Add content
    generator.add_text_content(
        config, history, metrics, input_dim, output_dim,
        total_params, n_train, n_val, n_test, timestamp
    )
    generator.add_plots(checkpoint_dir)

    # Generate PDF
    generator.generate()

    return report_path
