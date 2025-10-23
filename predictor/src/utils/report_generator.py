"""
Generatore di report PDF per il training della rete neurale

Questo modulo genera un report completo in formato PDF con tutte le informazioni
del training: configurazione, metriche, grafici, timestamp, ecc.
"""

from datetime import datetime
from pathlib import Path
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT


class TrainingReportGenerator:
    """
    Generatore di report PDF per il training

    Args:
        output_path (str or Path): Percorso dove salvare il PDF
        title (str): Titolo del report
    """

    def __init__(self, output_path, title="Training Report - Neural Network"):
        self.output_path = Path(output_path)
        self.title = title
        self.styles = getSampleStyleSheet()
        self.story = []

        # Crea stili personalizzati
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))

        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=12
        ))

    def add_title_page(self, timestamp=None):
        """Aggiunge la pagina del titolo"""
        if timestamp is None:
            timestamp = datetime.now()

        # Titolo principale
        self.story.append(Spacer(1, 2*inch))
        title = Paragraph(self.title, self.styles['CustomTitle'])
        self.story.append(title)
        self.story.append(Spacer(1, 0.5*inch))

        # Data e ora
        date_text = f"<b>Data Training:</b> {timestamp.strftime('%d/%m/%Y')}<br/>"
        date_text += f"<b>Ora:</b> {timestamp.strftime('%H:%M:%S')}"
        date_para = Paragraph(date_text, self.styles['Normal'])
        self.story.append(date_para)
        self.story.append(Spacer(1, 0.3*inch))

        # Separator line
        self.story.append(Spacer(1, 1*inch))

    def add_section_title(self, title):
        """Aggiunge un titolo di sezione"""
        self.story.append(Spacer(1, 0.2*inch))
        heading = Paragraph(title, self.styles['CustomHeading'])
        self.story.append(heading)
        self.story.append(Spacer(1, 0.1*inch))

    def add_configuration_section(self, config, input_dim, output_dim, total_params):
        """Aggiunge la sezione di configurazione"""
        self.add_section_title("1. Configurazione del Modello")

        # Crea tabella con configurazione
        data = [
            ['Parametro', 'Valore'],
            ['Tipo Modello', config['model']['model_type']],
            ['Hidden Layers', str(config['model']['hidden_sizes'])],
            ['Dropout Rate', f"{config['model']['dropout_rate']}"],
            ['Input Dimension', f"{input_dim}"],
            ['Output Dimension', f"{output_dim}"],
            ['Parametri Totali', f"{total_params:,}"],
        ]

        table = Table(data, colWidths=[3*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ]))

        self.story.append(table)
        self.story.append(Spacer(1, 0.2*inch))

    def add_data_section(self, config, n_train, n_val, n_test):
        """Aggiunge la sezione dati"""
        self.add_section_title("2. Dataset")

        # File path
        file_path = config['data']['csv_path']
        text = f"<b>File dati:</b> {file_path}<br/>"
        text += f"<b>Metodo di scaling:</b> {config['data']['scaling_method']}<br/><br/>"
        self.story.append(Paragraph(text, self.styles['Normal']))

        # Split info
        data = [
            ['Set', 'Campioni', 'Percentuale'],
            ['Training', f"{n_train}", f"{config['data']['train_size']*100:.0f}%"],
            ['Validation', f"{n_val}", f"{config['data']['val_size']*100:.0f}%"],
            ['Test', f"{n_test}", f"{config['data']['test_size']*100:.0f}%"],
            ['Totale', f"{n_train + n_val + n_test}", "100%"],
        ]

        table = Table(data, colWidths=[2*inch, 2*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ]))

        self.story.append(table)
        self.story.append(Spacer(1, 0.1*inch))

        # Input/Output columns
        text = f"<b>Colonne Input ({len(config['data']['input_columns'])}):</b><br/>"
        text += ", ".join(config['data']['input_columns']) + "<br/><br/>"
        text += f"<b>Colonne Output ({len(config['data']['output_columns'])}):</b><br/>"
        text += ", ".join(config['data']['output_columns'])
        self.story.append(Paragraph(text, self.styles['Normal']))
        self.story.append(Spacer(1, 0.2*inch))

    def add_training_section(self, config, history):
        """Aggiunge la sezione training"""
        self.add_section_title("3. Parametri di Training")

        # Parametri
        data = [
            ['Parametro', 'Valore'],
            ['Epochs', f"{config['training']['epochs']}"],
            ['Batch Size', f"{config['training']['batch_size']}"],
            ['Learning Rate', f"{config['training']['learning_rate']}"],
            ['Loss Function', config['training']['loss_function'].upper()],
            ['Early Stopping Patience', f"{config['training']['patience']}"],
            ['Device', config['training']['device']],
            ['Epochs Effettive', f"{len(history['train_losses'])}"],
        ]

        table = Table(data, colWidths=[3*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ]))

        self.story.append(table)
        self.story.append(Spacer(1, 0.1*inch))

        # Loss finale
        final_train_loss = history['train_losses'][-1]
        final_val_loss = history['val_losses'][-1]
        best_val_loss = min(history['val_losses'])

        text = f"<b>Loss Finale (Training):</b> {final_train_loss:.6f}<br/>"
        text += f"<b>Loss Finale (Validation):</b> {final_val_loss:.6f}<br/>"
        text += f"<b>Miglior Loss (Validation):</b> {best_val_loss:.6f}"
        self.story.append(Paragraph(text, self.styles['Normal']))
        self.story.append(Spacer(1, 0.2*inch))

    def add_metrics_section(self, metrics):
        """Aggiunge la sezione metriche"""
        self.add_section_title("4. Metriche di Valutazione (Test Set)")

        # Metriche per ogni output
        for output_name, output_metrics in metrics.items():
            text = f"<b>{output_name}:</b><br/>"
            self.story.append(Paragraph(text, self.styles['Normal']))

            data = [
                ['Metrica', 'Valore'],
                ['MAE', f"{output_metrics['MAE']:.6f}"],
                ['MSE', f"{output_metrics['MSE']:.6f}"],
                ['RMSE', f"{output_metrics['RMSE']:.6f}"],
                ['R² Score', f"{output_metrics['R2']:.6f}"],
            ]

            table = Table(data, colWidths=[2*inch, 2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ]))

            self.story.append(table)
            self.story.append(Spacer(1, 0.1*inch))

    def add_image(self, image_path, max_width=6*inch, max_height=4*inch, title=None):
        """Aggiunge un'immagine al report con dimensioni controllate"""
        if title:
            self.story.append(Paragraph(title, self.styles['Normal']))
            self.story.append(Spacer(1, 0.1*inch))

        if Path(image_path).exists():
            # Crea l'immagine con larghezza massima
            img = Image(str(image_path))

            # Calcola il rapporto per mantenere le proporzioni
            img_width, img_height = img.imageWidth, img.imageHeight
            aspect_ratio = img_height / img_width

            # Imposta larghezza e calcola altezza
            new_width = max_width
            new_height = new_width * aspect_ratio

            # Se l'altezza è troppo grande, ridimensiona in base all'altezza
            if new_height > max_height:
                new_height = max_height
                new_width = new_height / aspect_ratio

            img.drawWidth = new_width
            img.drawHeight = new_height

            self.story.append(img)
            self.story.append(Spacer(1, 0.2*inch))
        else:
            text = f"<i>Immagine non trovata: {image_path}</i>"
            self.story.append(Paragraph(text, self.styles['Normal']))
            self.story.append(Spacer(1, 0.2*inch))

    def add_plots_section(self, checkpoint_dir):
        """Aggiunge la sezione con i grafici"""
        # Aggiungi page break per i grafici
        self.story.append(PageBreak())

        self.add_section_title("5. Grafici")

        checkpoint_dir = Path(checkpoint_dir)

        # Training history plot
        history_plot = checkpoint_dir / 'training_history.png'
        self.add_image(history_plot, max_width=5.5*inch, max_height=3.5*inch, title="<b>Training History (Loss)</b>")

        # Predictions plot
        predictions_plot = checkpoint_dir / 'predictions.png'
        self.add_image(predictions_plot, max_width=5.5*inch, max_height=3.5*inch, title="<b>Predizioni vs Valori Reali (Test Set)</b>")

    def add_footer_section(self, checkpoint_dir):
        """Aggiunge sezione finale con info file salvati"""
        self.add_section_title("6. File Salvati")

        text = f"<b>Directory:</b> {checkpoint_dir}/<br/><br/>"
        text += "File generati:<br/>"
        text += "• best_model.pth - Miglior modello salvato<br/>"
        text += "• scalers.pkl - Scaler per preprocessing<br/>"
        text += "• training_history.json - Storia completa del training<br/>"
        text += "• training_history.png - Grafico loss<br/>"
        text += "• predictions.png - Grafico predizioni<br/>"
        text += "• training_report.pdf - Questo report<br/>"

        self.story.append(Paragraph(text, self.styles['Normal']))

    def generate(self):
        """Genera il PDF"""
        doc = SimpleDocTemplate(
            str(self.output_path),
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )

        doc.build(self.story)
        print(f"Report PDF generato: {self.output_path}")


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
    Funzione helper per generare un report completo

    Args:
        config: Dizionario configurazione
        history: Dizionario con history del training
        metrics: Dizionario con metriche calcolate
        input_dim: Dimensione input
        output_dim: Dimensione output
        total_params: Numero totale parametri
        n_train: Numero campioni training
        n_val: Numero campioni validation
        n_test: Numero campioni test
        checkpoint_dir: Directory dove salvare il report
        timestamp: Timestamp del training (opzionale)
    """
    checkpoint_dir = Path(checkpoint_dir)
    report_path = checkpoint_dir / 'training_report.pdf'

    generator = TrainingReportGenerator(report_path)

    # Aggiungi sezioni
    generator.add_title_page(timestamp)
    generator.add_configuration_section(config, input_dim, output_dim, total_params)
    generator.add_data_section(config, n_train, n_val, n_test)
    generator.add_training_section(config, history)
    generator.add_metrics_section(metrics)
    generator.add_plots_section(checkpoint_dir)
    generator.add_footer_section(checkpoint_dir)

    # Genera PDF
    generator.generate()

    return report_path
