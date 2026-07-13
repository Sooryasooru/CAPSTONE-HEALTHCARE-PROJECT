"""
HAIP — PDF Report Generator
src/reports/pdf_report.py

Decoupled, pure-function report builder. Knows nothing about HAIP's cards,
FastAPI, Dash, or the agent. You hand it plain Python data (dicts, DataFrames,
Plotly figures, Q&A lists) and it returns a PDF path.

Sections (all optional — pass None or omit to skip):
    - Hospital upload flow : KPIs, charts, EDA, model metrics, forecast, recommendation
    - Doctor analytics     : KPIs + chart(s)
    - Agent Q&A            : question / answer / tools_used trail

Usage
-----
    from src.reports.pdf_report import build_report, ReportData, HospitalSection

    data = ReportData(
        hospital_name="Aster Medcity",
        hospital=HospitalSection(
            kpis={"Total Admissions": 6832, "Avg LOS (days)": 4.2, "ICU Sepsis Rate": "3.1%"},
            charts=[fig1, fig2],                    # Plotly figures
            eda={"Rows": 6832, "Columns": 14, "Missing %": "0.4%"},
            model_metrics={"R2": 0.87, "MAE": 2.3, "RMSE": 3.1},
            feature_importance={"age": 0.31, "prior_admissions": 0.22, "los": 0.18},
            forecast=forecast_series,               # pandas Series (index=date)
            recommendation="Admissions trending up ~8% next quarter; plan ICU capacity.",
        ),
        # doctor=..., agent=...
    )
    path = build_report(data, out_path="data/processed/haip_report.pdf")
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, HRFlowable, KeepTogether,
)

# ---------------------------------------------------------------------------
# Brand palette
# ---------------------------------------------------------------------------
BRAND_PRIMARY   = colors.HexColor("#1a5276")   # deep clinical blue
BRAND_ACCENT    = colors.HexColor("#2e86c1")
BRAND_LIGHT     = colors.HexColor("#eaf2f8")
BRAND_TEXT      = colors.HexColor("#2c3e50")
BRAND_MUTED     = colors.HexColor("#7f8c8d")

PLATFORM_TITLE  = "HAIP"
PLATFORM_SUB    = "Healthcare Analytics & Intelligence Platform"


# ---------------------------------------------------------------------------
# Data containers  (pure data — you fill these from your cards)
# ---------------------------------------------------------------------------
@dataclass
class HospitalSection:
    kpis: Optional[dict[str, Any]] = None
    charts: list = field(default_factory=list)          # Plotly figures
    eda: Optional[dict[str, Any]] = None
    model_metrics: Optional[dict[str, Any]] = None
    feature_importance: Optional[dict[str, float]] = None
    forecast: Optional[pd.Series] = None                # index = date/period
    recommendation: Optional[str] = None


@dataclass
class DoctorSection:
    kpis: Optional[dict[str, Any]] = None
    charts: list = field(default_factory=list)          # Plotly figures
    table: Optional[pd.DataFrame] = None                # e.g. per-doctor stats


@dataclass
class AgentQA:
    question: str
    answer: str
    tools_used: list[str] = field(default_factory=list)


@dataclass
class AgentSection:
    qa: list[AgentQA] = field(default_factory=list)


@dataclass
class ReportData:
    hospital_name: str = "Hospital"
    hospital: Optional[HospitalSection] = None
    doctor: Optional[DoctorSection] = None
    agent: Optional[AgentSection] = None


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    s = {}
    s["h1"] = ParagraphStyle(
        "h1", parent=base["Heading1"], fontName="Helvetica-Bold",
        fontSize=16, textColor=BRAND_PRIMARY, spaceBefore=6, spaceAfter=10,
    )
    s["h2"] = ParagraphStyle(
        "h2", parent=base["Heading2"], fontName="Helvetica-Bold",
        fontSize=12, textColor=BRAND_ACCENT, spaceBefore=12, spaceAfter=6,
    )
    s["body"] = ParagraphStyle(
        "body", parent=base["Normal"], fontName="Helvetica",
        fontSize=9.5, textColor=BRAND_TEXT, leading=14,
    )
    s["muted"] = ParagraphStyle(
        "muted", parent=base["Normal"], fontName="Helvetica",
        fontSize=8, textColor=BRAND_MUTED, leading=11,
    )
    s["cover_title"] = ParagraphStyle(
        "cover_title", parent=base["Title"], fontName="Helvetica-Bold",
        fontSize=34, textColor=BRAND_PRIMARY, spaceAfter=4, alignment=1,
    )
    s["cover_sub"] = ParagraphStyle(
        "cover_sub", parent=base["Normal"], fontName="Helvetica",
        fontSize=12, textColor=BRAND_MUTED, alignment=1, spaceAfter=2,
    )
    s["agent_a"] = ParagraphStyle(
        "agent_a", parent=s["body"], leftIndent=8, borderPadding=6,
        backColor=BRAND_LIGHT, leading=14,
    )
    return s


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fig_to_image(fig, width_mm: float = 165, dpi_scale: int = 2) -> Optional[RLImage]:
    """Render a Plotly figure to a PNG (via kaleido) and wrap as a reportlab Image.
    Returns None on failure so a broken chart never kills the whole report."""
    try:
        png = fig.to_image(format="png", scale=dpi_scale)
    except Exception:
        return None
    bio = io.BytesIO(png)
    img = RLImage(bio)
    ratio = img.imageHeight / float(img.imageWidth)
    img.drawWidth = width_mm * mm
    img.drawHeight = width_mm * mm * ratio
    return img


def _kv_table(d: dict[str, Any], styles, col1="Metric", col2="Value") -> Table:
    """Two-column key/value table with brand styling."""
    rows = [[Paragraph(f"<b>{col1}</b>", styles["body"]),
             Paragraph(f"<b>{col2}</b>", styles["body"])]]
    for k, v in d.items():
        rows.append([Paragraph(str(k), styles["body"]),
                     Paragraph(_fmt(v), styles["body"])])
    t = Table(rows, colWidths=[95 * mm, 70 * mm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BRAND_LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d5dbdb")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def _df_table(df: pd.DataFrame, styles, max_rows: int = 15) -> Table:
    df = df.head(max_rows)
    header = [Paragraph(f"<b>{c}</b>", styles["body"]) for c in df.columns]
    rows = [header]
    for _, r in df.iterrows():
        rows.append([Paragraph(_fmt(v), styles["body"]) for v in r])
    t = Table(rows, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BRAND_LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d5dbdb")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:,.3f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:,.2f}"
    return str(v)


# ---------------------------------------------------------------------------
# Header / footer (drawn on every page)
# ---------------------------------------------------------------------------
def _make_header_footer(hospital_name: str):
    def _hf(canvas, doc):
        canvas.saveState()
        w, h = A4
        # header band
        canvas.setFillColor(BRAND_PRIMARY)
        canvas.rect(0, h - 16 * mm, w, 16 * mm, fill=1, stroke=0)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 11)
        canvas.drawString(15 * mm, h - 10.5 * mm, f"{PLATFORM_TITLE} — {PLATFORM_SUB}")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(w - 15 * mm, h - 10.5 * mm, hospital_name)
        # footer
        canvas.setStrokeColor(BRAND_MUTED)
        canvas.setLineWidth(0.3)
        canvas.line(15 * mm, 12 * mm, w - 15 * mm, 12 * mm)
        canvas.setFillColor(BRAND_MUTED)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(15 * mm, 8 * mm,
                          "Decision-support output — not a diagnosis. Requires human clinical review.")
        canvas.drawRightString(w - 15 * mm, 8 * mm, f"Page {doc.page}")
        canvas.restoreState()
    return _hf


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------
def _cover(data: ReportData, styles) -> list:
    story = [Spacer(1, 55 * mm)]
    story.append(Paragraph(PLATFORM_TITLE, styles["cover_title"]))
    story.append(Paragraph(PLATFORM_SUB, styles["cover_sub"]))
    story.append(Spacer(1, 6 * mm))
    story.append(HRFlowable(width="40%", thickness=1.2, color=BRAND_ACCENT,
                            spaceBefore=4, spaceAfter=12, hAlign="CENTER"))
    story.append(Paragraph(f"Analytics Report — {data.hospital_name}", styles["cover_sub"]))
    story.append(Paragraph(datetime.now().strftime("Generated %d %B %Y, %H:%M"),
                           styles["cover_sub"]))
    story.append(PageBreak())
    return story


def _hospital(sec: HospitalSection, styles) -> list:
    story = [Paragraph("1. Hospital Analytics", styles["h1"])]

    if sec.kpis:
        story.append(Paragraph("Key Performance Indicators", styles["h2"]))
        story.append(_kv_table(sec.kpis, styles, "KPI", "Value"))
        story.append(Spacer(1, 6))

    if sec.eda:
        story.append(Paragraph("Exploratory Data Overview", styles["h2"]))
        story.append(_kv_table(sec.eda, styles, "Property", "Value"))
        story.append(Spacer(1, 6))

    for i, fig in enumerate(sec.charts or []):
        img = _fig_to_image(fig)
        if img:
            story.append(KeepTogether([
                Paragraph(f"Chart {i + 1}", styles["h2"]), img, Spacer(1, 4)
            ]))

    if sec.model_metrics:
        story.append(Paragraph("Model Performance", styles["h2"]))
        # render R2 nicely as R<super>2</super> if key looks like it
        pretty = {}
        for k, v in sec.model_metrics.items():
            key = "R<super>2</super>" if k.lower() in ("r2", "r^2", "rsquared") else k
            pretty[key] = v
        story.append(_kv_table(pretty, styles, "Metric", "Score"))
        story.append(Spacer(1, 6))

    if sec.feature_importance:
        story.append(Paragraph("Feature Importance", styles["h2"]))
        fi = dict(sorted(sec.feature_importance.items(),
                         key=lambda kv: kv[1], reverse=True))
        story.append(_kv_table(fi, styles, "Feature", "Importance"))
        story.append(Spacer(1, 6))

    if sec.forecast is not None and len(sec.forecast):
        story.append(Paragraph("Admissions Forecast (Holt-Winters)", styles["h2"]))
        fdf = pd.DataFrame({"Period": [str(i) for i in sec.forecast.index],
                            "Forecast": sec.forecast.values})
        story.append(_df_table(fdf, styles))
        story.append(Spacer(1, 6))

    if sec.recommendation:
        story.append(Paragraph("Recommendation", styles["h2"]))
        story.append(Paragraph(sec.recommendation, styles["body"]))

    story.append(PageBreak())
    return story


def _doctor(sec: DoctorSection, styles) -> list:
    story = [Paragraph("2. Doctor Analytics", styles["h1"])]
    if sec.kpis:
        story.append(Paragraph("Summary KPIs", styles["h2"]))
        story.append(_kv_table(sec.kpis, styles, "KPI", "Value"))
        story.append(Spacer(1, 6))
    if sec.table is not None and len(sec.table):
        story.append(Paragraph("Per-Doctor Statistics", styles["h2"]))
        story.append(_df_table(sec.table, styles))
        story.append(Spacer(1, 6))
    for i, fig in enumerate(sec.charts or []):
        img = _fig_to_image(fig)
        if img:
            story.append(KeepTogether([
                Paragraph(f"Chart {i + 1}", styles["h2"]), img, Spacer(1, 4)
            ]))
    story.append(PageBreak())
    return story


def _agent(sec: AgentSection, styles) -> list:
    story = [Paragraph("3. Ask HAIP — Agent Q&A", styles["h1"])]
    for i, qa in enumerate(sec.qa, 1):
        block = [
            Paragraph(f"<b>Q{i}.</b> {qa.question}", styles["body"]),
            Spacer(1, 3),
            Paragraph(qa.answer, styles["agent_a"]),
        ]
        if qa.tools_used:
            block.append(Spacer(1, 3))
            block.append(Paragraph(
                "Tools used: " + ", ".join(qa.tools_used), styles["muted"]))
        block.append(Spacer(1, 10))
        story.append(KeepTogether(block))
    return story


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_report(data: ReportData, out_path: str = "haip_report.pdf") -> str:
    """Build the multi-section PDF. Returns the output path.
    Any section left as None is skipped. Safe to call with only one section."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    styles = _styles()

    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        topMargin=22 * mm, bottomMargin=18 * mm,
        leftMargin=15 * mm, rightMargin=15 * mm,
        title=f"HAIP Report — {data.hospital_name}",
        author="HAIP",
    )

    story: list = []
    story += _cover(data, styles)
    if data.hospital:
        story += _hospital(data.hospital, styles)
    if data.doctor:
        story += _doctor(data.doctor, styles)
    if data.agent and data.agent.qa:
        story += _agent(data.agent, styles)

    hf = _make_header_footer(data.hospital_name)
    doc.build(story, onFirstPage=hf, onLaterPages=hf)
    return out_path


# ---------------------------------------------------------------------------
# Convenience wrapper for the Dash download callback
# ---------------------------------------------------------------------------
def build_hospital_report(
    kpis: Optional[dict] = None,
    forecast_df: Optional[pd.DataFrame] = None,
    hospital_name: str = "Uploaded Dataset",
    model_metrics: Optional[dict] = None,
    feature_importance: Optional[dict] = None,
    recommendation: Optional[str] = None,
    charts: Optional[list] = None,
) -> bytes:
    """Hospital-only report, returned as PDF *bytes* for dcc.send_bytes().

    Thin adapter over build_report() so the Dash callback can pass plain
    kpis / forecast_df without constructing dataclasses itself.

        return dcc.send_bytes(
            build_hospital_report(kpis=kpis, forecast_df=forecast_df,
                                  hospital_name="Uploaded Dataset"),
            "haip_report.pdf",
        )
    """
    # forecast_df (column 'forecast') -> the Series the report expects
    forecast_series = None
    if forecast_df is not None and len(forecast_df):
        col = "forecast" if "forecast" in forecast_df.columns else forecast_df.columns[0]
        forecast_series = forecast_df[col]

    data = ReportData(
        hospital_name=hospital_name,
        hospital=HospitalSection(
            kpis=kpis,
            charts=charts or [],
            model_metrics=model_metrics,
            feature_importance=feature_importance,
            forecast=forecast_series,
            recommendation=recommendation,
        ),
    )

    buf = io.BytesIO()
    styles = _styles()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=22 * mm, bottomMargin=18 * mm,
        leftMargin=15 * mm, rightMargin=15 * mm,
        title=f"HAIP Report — {hospital_name}", author="HAIP",
    )
    story = _cover(data, styles) + _hospital(data.hospital, styles)
    hf = _make_header_footer(hospital_name)
    doc.build(story, onFirstPage=hf, onLaterPages=hf)
    return buf.getvalue()
