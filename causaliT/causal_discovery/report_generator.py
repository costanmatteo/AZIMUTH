"""
Causal Analysis Report Generator.

Produces a comprehensive PDF report covering all 5 analysis categories:
1. Ground Truth & Attention-based Discovery (DAG heatmaps + side-by-side graphs)
2. Causal Discovery Validation (metrics tables)
3. Interventional Analysis (pre/post comparison tables)
4. Out-of-Distribution Analysis (degradation report)
5. Symbolic Regression (equation recovery results)

Uses ReportLab following the same style as
``uncertainty_predictor/src/utils/report_generator.py``.
"""

from __future__ import annotations

import io
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.platypus.flowables import HRFlowable


class CausalAnalysisReportGenerator:
    """Generate a PDF causal analysis report."""

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.styles = getSampleStyleSheet()
        self.story = []
        self._setup_styles()

    def _setup_styles(self):
        """Register custom paragraph styles."""
        custom = [
            ("ReportTitle", "Heading1", 16, 19, TA_CENTER, 3),
            ("ReportSubtitle", "Normal", 10, 12, TA_CENTER, 6),
            ("SectionTitle", "Heading2", 11, 13, TA_LEFT, 2),
            ("SubSectionTitle", "Heading3", 9, 11, TA_LEFT, 1),
            ("BodySmall", "Normal", 7, 9, TA_LEFT, 0),
        ]
        for name, parent, fs, ld, align, after in custom:
            if name not in self.styles:
                self.styles.add(ParagraphStyle(
                    name=name,
                    parent=self.styles[parent],
                    fontSize=fs,
                    leading=ld,
                    alignment=align,
                    spaceAfter=after,
                ))

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------

    def _add_title(self, text: str, timestamp: Optional[datetime] = None):
        self.story.append(Paragraph(f"<b>{text}</b>", self.styles["ReportTitle"]))
        ts = timestamp or datetime.now()
        self.story.append(Paragraph(
            f"Date: {ts.strftime('%Y-%m-%d %H:%M:%S')}",
            self.styles["ReportSubtitle"],
        ))
        self.story.append(Spacer(1, 0.3 * cm))

    def _add_section(self, title: str):
        self.story.append(Spacer(1, 0.2 * cm))
        self.story.append(Paragraph(f"<b>{title}</b>", self.styles["SectionTitle"]))
        self.story.append(HRFlowable(
            width="100%", thickness=1, color=colors.black, spaceAfter=4,
        ))

    def _add_subsection(self, title: str):
        self.story.append(Paragraph(f"<b>{title}</b>", self.styles["SubSectionTitle"]))

    def _add_text(self, text: str):
        self.story.append(Paragraph(text, self.styles["BodySmall"]))
        self.story.append(Spacer(1, 0.1 * cm))

    def _add_dataframe_table(
        self,
        df: pd.DataFrame,
        max_rows: int = 30,
        font_size: int = 6,
    ):
        """Render a DataFrame as a ReportLab table."""
        if df is None or df.empty:
            self._add_text("<i>No data available.</i>")
            return

        df_display = df.head(max_rows).copy()

        # Format floats
        for col in df_display.columns:
            if df_display[col].dtype in [np.float64, np.float32, float]:
                df_display[col] = df_display[col].map(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                )

        header = [str(c) for c in df_display.columns]
        data = [header] + [
            [str(v) for v in row] for row in df_display.values
        ]

        col_widths = [max(2.0 * cm, 18.0 * cm / len(header))] * len(header)
        table = Table(data, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("LEADING", (0, 0), (-1, -1), font_size + 2),
            ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.85, 0.85, 0.85)),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        self.story.append(table)
        self.story.append(Spacer(1, 0.2 * cm))

    def _add_image(self, img_path: str, width: float = 16, caption: str = ""):
        """Add a PNG image to the report."""
        p = Path(img_path)
        if not p.exists():
            self._add_text(f"<i>Image not found: {img_path}</i>")
            return

        img = Image(str(p))
        aspect = img.imageHeight / img.imageWidth
        img.drawWidth = width * cm
        img.drawHeight = width * cm * aspect
        if img.drawHeight > 12 * cm:
            img.drawHeight = 12 * cm
            img.drawWidth = img.drawHeight / aspect

        img_table = Table([[img]], colWidths=[18 * cm])
        img_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        self.story.append(img_table)

        if caption:
            cap = Paragraph(f"<i>{caption}</i>", self.styles["Normal"])
            cap_table = Table([[cap]], colWidths=[18 * cm])
            cap_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]))
            self.story.append(cap_table)
        self.story.append(Spacer(1, 0.15 * cm))

    def _add_heatmap_image(
        self,
        matrix: np.ndarray | pd.DataFrame,
        title: str = "",
        width: float = 10,
    ):
        """Render a matrix as a heatmap and embed it."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if isinstance(matrix, pd.DataFrame):
                labels = list(matrix.columns)
                data = matrix.values.astype(float)
            else:
                data = np.asarray(matrix, dtype=float)
                labels = [str(i) for i in range(data.shape[0])]

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(data, cmap="Blues", aspect="auto")
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(min(data.shape[0], len(labels))))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
            ax.set_yticklabels(labels[:data.shape[0]], fontsize=6)
            if title:
                ax.set_title(title, fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.8)
            plt.tight_layout()

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                plt.savefig(f.name, dpi=150, bbox_inches="tight")
                plt.close(fig)
                self._add_image(f.name, width=width, caption=title)
        except ImportError:
            self._add_text(f"<i>matplotlib not available for heatmap: {title}</i>")

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def add_dag_section(
        self,
        adj_true: np.ndarray | pd.DataFrame,
        att_maps: Optional[Dict[str, pd.DataFrame]] = None,
        adj_estimated: Optional[Dict[str, pd.DataFrame]] = None,
    ):
        """Section 1: Ground Truth & Attention-based Discovery."""
        self._add_section("1. Ground Truth & Attention-based Discovery")

        self._add_subsection("Ground Truth DAG (Adjacency Matrix)")
        self._add_heatmap_image(adj_true, title="Ground Truth DAG", width=10)

        if att_maps:
            for att_type, att_df in att_maps.items():
                self._add_subsection(f"Attention Map: {att_type}")
                self._add_heatmap_image(att_df, title=f"Mean Attention - {att_type}", width=10)

        if adj_estimated:
            for att_type, adj_df in adj_estimated.items():
                self._add_subsection(f"Estimated DAG: {att_type}")
                self._add_heatmap_image(adj_df, title=f"Estimated DAG - {att_type}", width=10)

    def add_discovery_metrics_section(
        self,
        attention_results: Optional[pd.DataFrame] = None,
        classical_results: Optional[pd.DataFrame] = None,
    ):
        """Section 2: Causal Discovery Validation."""
        self._add_section("2. Causal Discovery Validation")

        if attention_results is not None:
            self._add_subsection("Attention-based Discovery Metrics")
            self._add_dataframe_table(attention_results)

        if classical_results is not None:
            self._add_subsection("Classical Baselines (GES / PC)")
            self._add_dataframe_table(classical_results)

    def add_intervention_section(
        self,
        f_comparison: Optional[pd.DataFrame] = None,
        distributional_comparisons: Optional[Dict[str, pd.DataFrame]] = None,
    ):
        """Section 3: Interventional Analysis."""
        self._add_section("3. Interventional Analysis")

        if f_comparison is not None:
            self._add_subsection("Reliability F under Interventions")
            self._add_dataframe_table(f_comparison)

        if distributional_comparisons:
            for label, df in distributional_comparisons.items():
                self._add_subsection(f"Distribution Comparison: {label}")
                self._add_dataframe_table(df)

    def add_ood_section(
        self,
        ood_summary: Optional[pd.DataFrame] = None,
        stability_results: Optional[Dict[str, dict]] = None,
    ):
        """Section 4: Out-of-Distribution Analysis."""
        self._add_section("4. Out-of-Distribution Analysis")

        if ood_summary is not None:
            self._add_subsection("OOD Distributional Shifts")
            self._add_dataframe_table(ood_summary)

        if stability_results:
            self._add_subsection("Attention Structure Stability")
            rows = []
            for att_type, metrics in stability_results.items():
                row = {"attention_type": att_type, **metrics}
                rows.append(row)
            df_stab = pd.DataFrame(rows)
            self._add_dataframe_table(df_stab)

    def add_symbolic_section(
        self,
        symbolic_results: Optional[pd.DataFrame] = None,
    ):
        """Section 5: Symbolic Regression."""
        self._add_section("5. Symbolic Regression")

        if symbolic_results is not None:
            self._add_dataframe_table(symbolic_results)
        else:
            self._add_text("<i>Symbolic regression results not available.</i>")

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def generate(
        self,
        adj_true=None,
        att_maps=None,
        adj_estimated=None,
        attention_results=None,
        classical_results=None,
        f_comparison=None,
        distributional_comparisons=None,
        ood_summary=None,
        stability_results=None,
        symbolic_results=None,
        timestamp=None,
    ):
        """Generate the full Causal Analysis Report PDF.

        All parameters are optional -- sections are included only when
        their data is provided.
        """
        self._add_title("Causal Analysis Report", timestamp)

        # Section 1
        if adj_true is not None:
            self.add_dag_section(adj_true, att_maps, adj_estimated)
            self.story.append(PageBreak())

        # Section 2
        if attention_results is not None or classical_results is not None:
            self.add_discovery_metrics_section(attention_results, classical_results)
            self.story.append(PageBreak())

        # Section 3
        if f_comparison is not None or distributional_comparisons:
            self.add_intervention_section(f_comparison, distributional_comparisons)
            self.story.append(PageBreak())

        # Section 4
        if ood_summary is not None or stability_results:
            self.add_ood_section(ood_summary, stability_results)
            self.story.append(PageBreak())

        # Section 5
        if symbolic_results is not None:
            self.add_symbolic_section(symbolic_results)

        # Build PDF
        doc = SimpleDocTemplate(
            str(self.output_path),
            pagesize=A4,
            rightMargin=1.5 * cm,
            leftMargin=1.5 * cm,
            topMargin=1.5 * cm,
            bottomMargin=1.5 * cm,
        )
        doc.build(self.story)
        return self.output_path


def generate_causal_report(
    output_path: str,
    **kwargs,
) -> Path:
    """Convenience function to generate a causal analysis report.

    Parameters
    ----------
    output_path : str
        Path for the output PDF.
    **kwargs
        Passed to ``CausalAnalysisReportGenerator.generate()``.

    Returns
    -------
    Path
        Path to the generated PDF.
    """
    gen = CausalAnalysisReportGenerator(output_path)
    return gen.generate(**kwargs)
