"""
PDF Report Generator for Causal Analysis.

Generates an A4 "Causal Analysis Report" using ReportLab, following the
same style as existing Azimuth reports (uncertainty_predictor and
controller_optimization report generators).
"""

from datetime import datetime
from pathlib import Path
import warnings

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus.flowables import HRFlowable


class CausalAnalysisReportGenerator:
    """PDF report generator for causal analysis results."""

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.styles = getSampleStyleSheet()
        self.story = []

        self._setup_styles()

    def _setup_styles(self):
        """Define custom paragraph styles."""
        style_defs = {
            'ReportTitle': {
                'parent': 'Heading1',
                'fontSize': 18,
                'leading': 22,
                'alignment': TA_CENTER,
                'spaceAfter': 6,
            },
            'ReportSubtitle': {
                'parent': 'Normal',
                'fontSize': 11,
                'leading': 14,
                'alignment': TA_CENTER,
                'spaceAfter': 12,
            },
            'SectionTitle': {
                'parent': 'Heading2',
                'fontSize': 13,
                'leading': 16,
                'fontName': 'Helvetica-Bold',
                'spaceAfter': 4,
                'spaceBefore': 10,
            },
            'SubsectionTitle': {
                'parent': 'Heading3',
                'fontSize': 11,
                'leading': 13,
                'fontName': 'Helvetica-Bold',
                'spaceAfter': 3,
                'spaceBefore': 6,
            },
            'BodyText': {
                'parent': 'Normal',
                'fontSize': 9,
                'leading': 12,
                'alignment': TA_JUSTIFY,
            },
            'Caption': {
                'parent': 'Normal',
                'fontSize': 8,
                'leading': 10,
                'alignment': TA_CENTER,
                'spaceAfter': 8,
                'fontName': 'Helvetica-Oblique',
            },
            'BulletItem': {
                'parent': 'Normal',
                'fontSize': 9,
                'leading': 12,
                'leftIndent': 20,
                'bulletIndent': 10,
            },
        }

        for name, opts in style_defs.items():
            if name not in self.styles:
                parent_name = opts.pop('parent', 'Normal')
                parent = self.styles[parent_name]
                self.styles.add(ParagraphStyle(name=name, parent=parent, **opts))

    # ----- Building blocks -----

    def add_title_page(self, timestamp: datetime = None):
        """Add title page."""
        if timestamp is None:
            timestamp = datetime.now()

        self.story.append(Spacer(1, 3 * cm))
        self.story.append(Paragraph(
            '<b>Causal Analysis Report</b>', self.styles['ReportTitle']
        ))
        self.story.append(Paragraph(
            'Azimuth Pipeline', self.styles['ReportSubtitle']
        ))
        self.story.append(Spacer(1, 1 * cm))
        self.story.append(Paragraph(
            f'Date: {timestamp.strftime("%Y-%m-%d %H:%M")}',
            self.styles['ReportSubtitle']
        ))
        self.story.append(Spacer(1, 0.5 * cm))
        self.story.append(Paragraph(
            'Inspired by: Gamella, Peters, B&uuml;hlmann — '
            '<i>Causal chambers as a real-world physical testbed for AI methodology</i>, '
            'Nature Machine Intelligence, 2025',
            self.styles['Caption']
        ))
        self.story.append(PageBreak())

    def add_section(self, title: str):
        """Add a section heading with horizontal rule."""
        self.story.append(Paragraph(f'<b>{title}</b>', self.styles['SectionTitle']))
        self.story.append(HRFlowable(
            width='100%', thickness=1, color=colors.black, spaceAfter=6
        ))

    def add_subsection(self, title: str):
        """Add a subsection heading."""
        self.story.append(Paragraph(f'<b>{title}</b>', self.styles['SubsectionTitle']))

    def add_text(self, text: str):
        """Add body text paragraph."""
        self.story.append(Paragraph(text, self.styles['BodyText']))
        self.story.append(Spacer(1, 0.2 * cm))

    def add_bullet(self, text: str):
        """Add a bullet point."""
        self.story.append(Paragraph(
            f'&bull; {text}', self.styles['BulletItem']
        ))

    def add_image(self, image_path: str, width: float = 16 * cm, caption: str = ''):
        """Add an image with optional caption, auto-scaled to fit page."""
        img_path = Path(image_path)
        if not img_path.exists():
            self.story.append(Paragraph(
                f'<i>[Image not found: {image_path}]</i>', self.styles['Caption']
            ))
            return

        # Max usable height on page (A4 minus margins minus some padding)
        max_height = A4[1] - 8 * cm  # ~21.7 cm

        # Get image dimensions using PIL for reliable sizing
        try:
            from PIL import Image as PILImage
            with PILImage.open(str(img_path)) as pil_img:
                iw, ih = pil_img.size
            aspect = ih / iw
            computed_height = width * aspect
            if computed_height > max_height:
                computed_height = max_height
                width = max_height / aspect
            img = Image(str(img_path), width=width, height=computed_height)
        except ImportError:
            # Fallback: use a conservative fixed height
            img = Image(str(img_path), width=width, height=min(width, max_height))

        img.hAlign = 'CENTER'
        self.story.append(img)
        if caption:
            self.story.append(Paragraph(caption, self.styles['Caption']))
        self.story.append(Spacer(1, 0.3 * cm))

    def add_table(self, data: list, col_widths: list = None):
        """
        Add a styled table.

        Parameters
        ----------
        data : list of lists
            Table data. First row is header.
        col_widths : list, optional
            Column widths.
        """
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.9, 0.9, 0.9)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 3),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.Color(0.97, 0.97, 0.97)]),
        ])

        t = Table(data, colWidths=col_widths, repeatRows=1)
        t.setStyle(style)
        t.hAlign = 'CENTER'
        self.story.append(t)
        self.story.append(Spacer(1, 0.3 * cm))

    def add_spacer(self, height_cm: float = 0.5):
        self.story.append(Spacer(1, height_cm * cm))

    def add_page_break(self):
        self.story.append(PageBreak())

    # ----- Report sections -----

    def add_executive_summary(self, summary_data: dict):
        """Add executive summary page."""
        self.add_section('Executive Summary')

        n_processes = summary_data.get('n_processes', 4)
        n_vars = summary_data.get('n_variables', 0)
        analyses_run = summary_data.get('analyses_run', [])
        key_findings = summary_data.get('key_findings', [])

        self.add_text(
            f'This report presents a causal analysis of the Azimuth manufacturing '
            f'pipeline, comprising <b>{n_processes}</b> sequential processes and '
            f'<b>{n_vars}</b> observable variables.'
        )
        self.add_text(f'Analyses performed: {", ".join(analyses_run)}.')

        if key_findings:
            self.add_subsection('Key Findings')
            for finding in key_findings:
                self.add_bullet(finding)

        self.add_page_break()

    def add_discovery_section(self, results: dict, figure_dir: Path):
        """Add Causal Graph Analysis section."""
        self.add_section('Section 1 — Causal Graph Analysis')

        self.add_text(
            'This section compares the ground truth DAG (derived from the SCM '
            'structural equations) with the causal graph estimated from the '
            'attention weights of the CausaliT transformer.'
        )

        # Ground truth DAG figure
        self.add_image(
            str(figure_dir / 'ground_truth_dag.png'),
            width=12 * cm,
            caption='Figure 1: Ground truth DAG of the inter-process causal structure.'
        )

        # Attention heatmap
        att_path = figure_dir / 'attention_heatmap.png'
        if att_path.exists():
            self.add_image(
                str(att_path), width=14 * cm,
                caption='Figure 2: Aggregated attention weight matrix (encoder self-attention).'
            )

        # DAG comparison
        dag_path = figure_dir / 'dag_comparison.png'
        if dag_path.exists():
            self.add_image(
                str(dag_path), width=16 * cm,
                caption='Figure 3: Ground truth G* (left) vs estimated graph from attention (right). '
                        'Black = TP, red = FP, grey dashed = FN.'
            )

        # Metrics table
        if 'metrics' in results:
            metrics = results['metrics']
            table_data = [['Method', 'Precision', 'Recall', 'F1', 'SHD']]
            for method, m in metrics.items():
                table_data.append([
                    method,
                    f"{m.get('precision', 0):.3f}",
                    f"{m.get('recall', 0):.3f}",
                    f"{m.get('f1', 0):.3f}",
                    f"{m.get('shd', '-')}",
                ])
            self.add_table(table_data)

        # Metrics bar chart
        bar_path = figure_dir / 'metrics_bar_chart.png'
        if bar_path.exists():
            self.add_image(
                str(bar_path), width=12 * cm,
                caption='Figure 4: Causal discovery metrics comparison.'
            )

        self.add_page_break()

    def add_interventional_section(self, results: dict, figure_dir: Path):
        """Add Interventional Validation section."""
        self.add_section('Section 2 — Interventional Validation')

        self.add_text(
            'This section validates causal effects by comparing observational '
            'and interventional distributions generated via the do-operator. '
            'We verify that interventions have the expected effects on downstream '
            'variables and that the reliability function F responds correctly.'
        )

        # Summary violins
        violin_path = figure_dir / 'intervention_summary.png'
        if violin_path.exists():
            self.add_image(
                str(violin_path), width=16 * cm,
                caption='Figure 5: Observational vs interventional distributions for each intervention.'
            )

        # Summary table
        if 'summary_table' in results and not results['summary_table'].empty:
            df = results['summary_table']
            cols = ['process', 'intervention', 'variable', 'obs_mean', 'int_mean',
                    'effect_size', 'p_value', 'significant']
            cols = [c for c in cols if c in df.columns]
            table_data = [cols]
            for _, row in df.iterrows():
                table_data.append([
                    str(row.get(c, '')) if c not in ('obs_mean', 'int_mean', 'effect_size', 'p_value')
                    else f"{row.get(c, 0):.3f}" if c != 'p_value' else f"{row.get(c, 0):.2e}"
                    for c in cols
                ])
            self.add_subsection('Causal Effect Summary')
            self.add_table(table_data)

        # F scatter
        f_scatter_path = figure_dir / 'f_scatter.png'
        if f_scatter_path.exists():
            self.add_image(
                str(f_scatter_path), width=10 * cm,
                caption='Figure 6: F (formula) vs F (CausaliT) under interventions.'
            )

        self.add_page_break()

    def add_ood_section(self, results: dict, figure_dir: Path):
        """Add OOD Robustness section."""
        self.add_section('Section 3 — OOD Robustness')

        self.add_text(
            'This section evaluates the robustness of the pipeline models under '
            'distributional shift. For each process, one key input variable is '
            'shifted beyond its training range and we measure output degradation.'
        )

        # Summary table
        if 'summary_table' in results and not results['summary_table'].empty:
            df = results['summary_table']
            table_data = [list(df.columns)]
            for _, row in df.iterrows():
                table_data.append([
                    f"{v:.3f}" if isinstance(v, float) else str(v) for v in row.values
                ])
            self.add_table(table_data)

        # Bar chart
        bar_path = figure_dir / 'ood_bar_chart.png'
        if bar_path.exists():
            self.add_image(
                str(bar_path), width=12 * cm,
                caption='Figure 7: Output statistics ID vs OOD per process.'
            )

        # Attention comparison
        att_path = figure_dir / 'ood_attention_comparison.png'
        if att_path.exists():
            self.add_image(
                str(att_path), width=14 * cm,
                caption='Figure 8: Attention stability under distribution shift.'
            )

        self.add_page_break()

    def add_symbolic_section(self, results: dict, figure_dir: Path):
        """Add Symbolic Regression section."""
        self.add_section('Section 4 — Symbolic Regression')

        self.add_text(
            'This section tests whether symbolic regression methods can rediscover '
            'the structural equations of each process from input-output data. '
            'We compare the discovered equations with the known ground truth.'
        )

        # Equations comparison figure
        eq_path = figure_dir / 'symbolic_equations.png'
        if eq_path.exists():
            self.add_image(
                str(eq_path), width=16 * cm,
                caption='Figure 9: True vs predicted output for each process (symbolic regression).'
            )

        # Summary table
        if 'summary_table' in results and not results['summary_table'].empty:
            df = results['summary_table']
            cols = [c for c in ['process', 'output', 'best_method', 'best_r2', 'poly_r2']
                    if c in df.columns]
            if 'pysr_r2' in df.columns:
                cols.append('pysr_r2')
            table_data = [cols]
            for _, row in df.iterrows():
                table_data.append([
                    f"{row[c]:.4f}" if isinstance(row.get(c), float) else str(row.get(c, ''))
                    for c in cols
                ])
            self.add_subsection('Symbolic Regression Results')
            self.add_table(table_data)

        # Known equations table
        from causal_chamber.symbolic_analysis import KNOWN_EQUATIONS
        eq_data = [['Process', 'True Equation', 'Key Features']]
        for proc, info in KNOWN_EQUATIONS.items():
            eq_data.append([
                proc,
                info['equation_description'][:60],
                ', '.join(info['key_features']),
            ])
        self.add_subsection('Ground Truth Equations')
        self.add_table(eq_data)

        self.add_page_break()

    def add_validation_section(self, results: dict, figure_dir: Path):
        """Add Causal Validation section (pipeline p-value matrices + F)."""
        self.add_section('Section 5 — Causal Validation')

        self.add_text(
            'This section validates the ground truth causal edges using two-sample '
            'KS tests (Kolmogorov-Smirnov), following the methodology of the Causal '
            'Chamber paper (Gamella et al., Tables 5-8). For each edge, we intervene '
            'on the parent and test whether the child distribution changes. '
            'Joint pipeline trajectories are assembled through all 4 processes and '
            'reliability F is computed via the ReliabilityFunction, validating '
            'inter-process coupling.'
        )

        # Summary stats
        s = results.get('summary', {})
        self.add_subsection('Summary')
        self.add_bullet(f"Ground truth edges: {s.get('n_ground_truth_edges', 0)}")
        self.add_bullet(f"Intra-process edges checked: {s.get('n_intra', 0)}")
        self.add_bullet(f"Pipeline edges checked (inter-process + F): {s.get('n_pipeline', 0)}")
        self.add_bullet(f"Validated: {s.get('n_validated', 0)}")
        self.add_bullet(f"Invalidated: {s.get('n_invalidated', 0)}")
        self.add_bullet(f"Validation rate: {s.get('validation_rate', 0):.1%}")

        # Validation heatmap figure
        val_path = figure_dir / 'causal_validation_summary.png'
        if val_path.exists():
            self.add_image(
                str(val_path), width=16 * cm,
                caption='Figure 10: Per-process p-value heatmaps + pipeline F row. '
                        'Red boxes = ground truth edges. Dark = significant.'
            )

        # Validated edges table
        if results.get('validated_edges'):
            from causal_chamber.causal_validation import format_pvalue
            table_data = [['Parent', 'Child', 'p-value', 'Type']]
            for parent, child, p, edge_type in sorted(results['validated_edges'], key=lambda x: x[2]):
                table_data.append([parent, child, format_pvalue(p), edge_type])
            self.add_subsection('Validated Edges')
            self.add_table(table_data)

        if results.get('invalidated_edges'):
            from causal_chamber.causal_validation import format_pvalue
            table_data = [['Parent', 'Child', 'p-value', 'Type']]
            for parent, child, p, edge_type in results['invalidated_edges']:
                table_data.append([parent, child, format_pvalue(p), edge_type])
            self.add_subsection('Invalidated Edges')
            self.add_table(table_data)
            self.add_text(
                '<i>Note: Inter-process output-to-output edges (e.g. ActualPower -> '
                'RemovalRate) are coupled only through the reliability function F\'s '
                'adaptive targets, not through direct data flow between SCMs. The KS '
                'test correctly shows no distributional effect on the downstream output '
                'itself. The coupling is validated through the output -> F edges.</i>'
            )

        self.add_page_break()

    def add_conclusions(self, conclusions: dict):
        """Add conclusions section."""
        self.add_section('Conclusions')

        if 'summary' in conclusions:
            self.add_text(conclusions['summary'])

        if 'strengths' in conclusions:
            self.add_subsection('Strengths')
            for item in conclusions['strengths']:
                self.add_bullet(item)

        if 'weaknesses' in conclusions:
            self.add_subsection('Weaknesses')
            for item in conclusions['weaknesses']:
                self.add_bullet(item)

        if 'recommendations' in conclusions:
            self.add_subsection('Recommendations')
            for item in conclusions['recommendations']:
                self.add_bullet(item)

        if 'skipped' in conclusions and conclusions['skipped']:
            self.add_subsection('Skipped Analyses')
            for item in conclusions['skipped']:
                self.add_bullet(f'<i>{item}</i>')

    # ----- Build -----

    def build(self):
        """Generate the PDF document."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(
            str(self.output_path),
            pagesize=A4,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )
        doc.build(self.story)
        return self.output_path
