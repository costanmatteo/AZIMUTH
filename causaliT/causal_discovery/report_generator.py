"""
Causal Analysis Report Generator.

Generates a comprehensive PDF report including all five causal analyses:

1. Ground-truth DAG vs attention-estimated DAG (heatmaps + graph)
2. Discovery metrics tables (precision, recall, F1, SHD)
3. Interventional validation results
4. OOD robustness analysis
5. Symbolic regression results

Follows the same ReportLab A4 style as
``uncertainty_predictor/src/utils/report_generator.py``.
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


class CausalAnalysisReportGenerator:
    """ReportLab-based PDF generator for the Causal Analysis Report.

    Parameters
    ----------
    output_path : str or Path
        Path for the output PDF file.
    """

    def __init__(self, output_path: Union[str, Path]):
        self.output_path = Path(output_path)
        self._check_reportlab()
        self._init_styles()
        self.story = []

    @staticmethod
    def _check_reportlab():
        try:
            import reportlab  # noqa: F401
        except ImportError:
            raise ImportError(
                "reportlab is required for PDF generation. "
                "Install with: pip install reportlab"
            )

    def _init_styles(self):
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        self.styles = getSampleStyleSheet()

        if "ReportTitle" not in self.styles:
            self.styles.add(ParagraphStyle(
                name="ReportTitle",
                parent=self.styles["Heading1"],
                fontSize=16,
                leading=19,
                alignment=TA_CENTER,
                spaceAfter=3,
            ))

        if "ReportSubtitle" not in self.styles:
            self.styles.add(ParagraphStyle(
                name="ReportSubtitle",
                parent=self.styles["Normal"],
                fontSize=10,
                leading=12,
                alignment=TA_CENTER,
                spaceAfter=6,
            ))

        if "SectionTitle" not in self.styles:
            self.styles.add(ParagraphStyle(
                name="SectionTitle",
                parent=self.styles["Heading2"],
                fontSize=10,
                leading=12,
                fontName="Helvetica-Bold",
                spaceAfter=1,
                spaceBefore=4,
            ))

        if "BodyText" not in self.styles:
            self.styles.add(ParagraphStyle(
                name="BodyText",
                parent=self.styles["Normal"],
                fontSize=7,
                leading=9,
                leftIndent=10,
            ))

        if "CaptionStyle" not in self.styles:
            self.styles.add(ParagraphStyle(
                name="CaptionStyle",
                parent=self.styles["Normal"],
                fontSize=8,
                leading=10,
                alignment=TA_CENTER,
            ))

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------

    def add_title(self, title: str = "Causal Analysis Report", timestamp: Optional[datetime] = None):
        from reportlab.platypus import Paragraph, Spacer
        from reportlab.lib.units import cm

        ts = timestamp or datetime.now()
        self.story.append(Paragraph(f"<b>{title}</b>", self.styles["ReportTitle"]))
        self.story.append(Paragraph(
            f"Date: {ts.strftime('%Y-%m-%d %H:%M:%S')}",
            self.styles["ReportSubtitle"],
        ))
        self.story.append(Spacer(1, 0.1 * cm))

    def add_section(self, title: str):
        from reportlab.platypus import Paragraph
        from reportlab.platypus.flowables import HRFlowable
        from reportlab.lib import colors

        self.story.append(Paragraph(f"<b>{title}</b>", self.styles["SectionTitle"]))
        self.story.append(HRFlowable(
            width="100%", thickness=1, color=colors.black, spaceAfter=4
        ))

    def add_text(self, text: str):
        from reportlab.platypus import Paragraph
        self.story.append(Paragraph(text, self.styles["BodyText"]))

    def add_spacer(self, height_cm: float = 0.2):
        from reportlab.platypus import Spacer
        from reportlab.lib.units import cm
        self.story.append(Spacer(1, height_cm * cm))

    def add_dataframe_table(self, df: pd.DataFrame, title: str = ""):
        """Add a pandas DataFrame as a formatted table."""
        from reportlab.platypus import Table, TableStyle, Paragraph
        from reportlab.lib import colors
        from reportlab.lib.units import cm

        if title:
            self.add_text(f"<b>{title}</b>")
            self.add_spacer(0.1)

        # Build table data
        header = [""] + list(df.columns)
        data = [header]
        for idx, row in df.iterrows():
            formatted = [str(idx)]
            for val in row:
                if isinstance(val, float):
                    formatted.append(f"{val:.4f}")
                else:
                    formatted.append(str(val))
            data.append(formatted)

        n_cols = len(header)
        col_width = min(2.5 * cm, 17.0 * cm / n_cols)
        table = Table(data, colWidths=[col_width] * n_cols)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.85, 0.85, 0.85)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 0.95, 0.95)]),
        ]))
        self.story.append(table)
        self.add_spacer(0.15)

    def add_image_from_figure(self, fig, width_cm: float = 16.0, caption: str = ""):
        """Render a matplotlib figure into the PDF."""
        from reportlab.platypus import Image, Table, TableStyle, Paragraph
        from reportlab.lib.units import cm

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)

        img = Image(buf)
        aspect = img.imageHeight / img.imageWidth
        img.drawWidth = width_cm * cm
        img.drawHeight = width_cm * cm * aspect

        max_h = 12 * cm
        if img.drawHeight > max_h:
            img.drawHeight = max_h
            img.drawWidth = max_h / aspect

        img_table = Table([[img]], colWidths=[18 * cm])
        img_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        self.story.append(img_table)

        if caption:
            cap = Paragraph(f"<i>{caption}</i>", self.styles["CaptionStyle"])
            cap_table = Table([[cap]], colWidths=[18 * cm])
            cap_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]))
            self.story.append(cap_table)
        self.add_spacer(0.15)

    # ------------------------------------------------------------------
    # Section generators
    # ------------------------------------------------------------------

    def add_ground_truth_section(
        self,
        gt_adj: pd.DataFrame,
        est_adj: Optional[pd.DataFrame] = None,
        var_attention: Optional[pd.DataFrame] = None,
    ):
        """Section 1: Ground Truth & Estimated DAG.

        Renders heatmaps of the ground-truth adjacency, attention weights,
        and estimated adjacency side-by-side.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.add_section("1. Ground Truth & Attention-based Discovery")

        n_plots = 1 + (est_adj is not None) + (var_attention is not None)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.5))
        if n_plots == 1:
            axes = [axes]

        idx = 0
        # Ground truth
        im = axes[idx].imshow(gt_adj.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        axes[idx].set_xticks(range(len(gt_adj.columns)))
        axes[idx].set_xticklabels(gt_adj.columns, rotation=90, fontsize=6)
        axes[idx].set_yticks(range(len(gt_adj.index)))
        axes[idx].set_yticklabels(gt_adj.index, fontsize=6)
        axes[idx].set_title("Ground Truth DAG", fontsize=9)
        fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        idx += 1

        # Attention weights (continuous)
        if var_attention is not None:
            im2 = axes[idx].imshow(var_attention.values, cmap="Reds", aspect="auto")
            axes[idx].set_xticks(range(len(var_attention.columns)))
            axes[idx].set_xticklabels(var_attention.columns, rotation=90, fontsize=6)
            axes[idx].set_yticks(range(len(var_attention.index)))
            axes[idx].set_yticklabels(var_attention.index, fontsize=6)
            axes[idx].set_title("Attention Weights", fontsize=9)
            fig.colorbar(im2, ax=axes[idx], fraction=0.046, pad=0.04)
            idx += 1

        # Estimated adjacency (binary)
        if est_adj is not None:
            im3 = axes[idx].imshow(est_adj.values, cmap="Greens", vmin=0, vmax=1, aspect="auto")
            axes[idx].set_xticks(range(len(est_adj.columns)))
            axes[idx].set_xticklabels(est_adj.columns, rotation=90, fontsize=6)
            axes[idx].set_yticks(range(len(est_adj.index)))
            axes[idx].set_yticklabels(est_adj.index, fontsize=6)
            axes[idx].set_title("Estimated DAG (Attention)", fontsize=9)
            fig.colorbar(im3, ax=axes[idx], fraction=0.046, pad=0.04)

        plt.tight_layout()
        self.add_image_from_figure(fig, caption="Ground truth vs estimated causal structure")
        plt.close(fig)

    def add_discovery_metrics_section(self, metrics_df: pd.DataFrame):
        """Section 2: Discovery Validation Metrics Table."""
        self.add_section("2. Causal Discovery Validation")
        self.add_dataframe_table(metrics_df, title="Discovery Metrics by Method")

    def add_threshold_sweep_section(self, sweep_df: pd.DataFrame):
        """Optional: threshold sweep results as a plot."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.add_section("2b. Threshold Sweep Analysis")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(sweep_df["threshold"], sweep_df["precision"], "b-o", label="Precision", markersize=3)
        ax.plot(sweep_df["threshold"], sweep_df["recall"], "r-s", label="Recall", markersize=3)
        ax.plot(sweep_df["threshold"], sweep_df["f1"], "g-^", label="F1", markersize=3)
        ax2 = ax.twinx()
        ax2.plot(sweep_df["threshold"], sweep_df["shd"], "k--", label="SHD", alpha=0.5)
        ax2.set_ylabel("SHD", fontsize=8)
        ax.set_xlabel("Threshold", fontsize=8)
        ax.set_ylabel("Score", fontsize=8)
        ax.set_title("Discovery Metrics vs Attention Threshold", fontsize=9)
        ax.legend(loc="upper left", fontsize=7)
        ax2.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self.add_image_from_figure(fig, caption="Precision/Recall/F1/SHD vs threshold")
        plt.close(fig)

    def add_interventional_section(
        self,
        intervention_results: Dict[str, pd.DataFrame],
        pvalue_matrix: Optional[pd.DataFrame] = None,
    ):
        """Section 3: Interventional Validation."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.add_section("3. Interventional Validation")

        # Summary of each intervention
        for label, df in intervention_results.items():
            self.add_text(f"<b>{label}</b>")
            # Only show significant shifts
            sig = df[df["ks_pvalue"] < 0.05]
            if not sig.empty:
                display_cols = ["mean_shift", "ks_statistic", "ks_pvalue"]
                avail = [c for c in display_cols if c in sig.columns]
                self.add_dataframe_table(sig[avail])
            else:
                self.add_text("No statistically significant distributional shifts detected.")
            self.add_spacer(0.1)

        # P-value heatmap
        if pvalue_matrix is not None:
            fig, ax = plt.subplots(figsize=(10, 5))
            im = ax.imshow(
                -np.log10(pvalue_matrix.values + 1e-20),
                cmap="YlOrRd", aspect="auto",
            )
            ax.set_xticks(range(len(pvalue_matrix.columns)))
            ax.set_xticklabels(pvalue_matrix.columns, rotation=90, fontsize=6)
            ax.set_yticks(range(len(pvalue_matrix.index)))
            ax.set_yticklabels(pvalue_matrix.index, fontsize=6)
            ax.set_title("Interventional p-value matrix (-log10)", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            self.add_image_from_figure(
                fig, caption="P-value matrix: darker = more significant effect"
            )
            plt.close(fig)

    def add_ood_section(self, ood_results: Dict[str, pd.DataFrame]):
        """Section 4: OOD Robustness Analysis."""
        self.add_section("4. Out-of-Distribution Analysis")

        for shift_desc, df in ood_results.items():
            self.add_text(f"<b>{shift_desc}</b>")
            display_cols = [c for c in ["id_mean", "ood_mean", "mean_shift", "relative_shift", "ks_pvalue"]
                           if c in df.columns]
            self.add_dataframe_table(df[display_cols])
            self.add_spacer(0.1)

    def add_symbolic_section(self, symbolic_df: pd.DataFrame):
        """Section 5: Symbolic Regression Results."""
        self.add_section("5. Symbolic Regression")
        display_cols = [c for c in symbolic_df.columns
                        if c not in ("ground_truth_expr",)]
        self.add_dataframe_table(symbolic_df[display_cols])

        # Also show ground truth expressions
        self.add_spacer(0.1)
        self.add_text("<b>Ground Truth Expressions:</b>")
        for _, row in symbolic_df.iterrows():
            proc = row.get("process", "")
            out = row.get("output_var", "")
            gt = row.get("ground_truth_expr", "N/A")
            self.add_text(f"  {proc}/{out}: <font face='Courier'>{gt}</font>")

    # ------------------------------------------------------------------
    # Generate PDF
    # ------------------------------------------------------------------

    def generate(self):
        """Build and save the PDF document."""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate

        doc = SimpleDocTemplate(
            str(self.output_path),
            pagesize=A4,
            rightMargin=1.5 * cm,
            leftMargin=1.5 * cm,
            topMargin=1.5 * cm,
            bottomMargin=1.5 * cm,
        )
        doc.build(self.story)
        print(f"Causal Analysis Report generated: {self.output_path}")
        return self.output_path
