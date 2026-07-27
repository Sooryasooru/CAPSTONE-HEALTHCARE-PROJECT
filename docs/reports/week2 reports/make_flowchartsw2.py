"""Generate Week 2 module flowcharts (Analytics + Prediction) as PNGs.

Matches the Week 1 flowchart style: rounded matplotlib boxes, navy/blue/
gold/grey/green palette, blue decision diamonds with italic branch labels.
Reproducible source for the flowcharts/ deliverables.

Run from src/ with:  python -m <wherever placed>   (standalone: python make_flowcharts.py)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrow, Polygon

# Palette (matched to Week 1 flowcharts)
NAVY = "#1F3864"
BLUE = "#2E78B8"
GOLD = "#C9A005"
GREY = "#8C8C8C"
GREEN = "#2E7D32"
BROWN = "#B07A35"
ARROW_NAVY = "#1F3864"
ARROW_GREY = "#8C8C8C"


def _box(ax, x, y, w, h, text, color, rounded=True, text_color="white",
         fontsize=12, italic=False):
    """Draw a rounded/square process box with centred bold text."""
    style = "round,pad=0.02,rounding_size=0.12" if rounded else "square,pad=0.02"
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h, boxstyle=style,
        facecolor=color, edgecolor="none"))
    ax.text(x, y, text, ha="center", va="center", color=text_color,
            fontsize=fontsize, fontweight="bold",
            fontstyle="italic" if italic else "normal", wrap=True)


def _diamond(ax, x, y, w, h, text, color=BLUE):
    """Draw a decision diamond."""
    pts = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
    ax.add_patch(Polygon(pts, facecolor=color, edgecolor="none"))
    ax.text(x, y, text, ha="center", va="center", color="white",
            fontsize=11, fontweight="bold")


def _arrow(ax, x1, y1, x2, y2, color=ARROW_NAVY, label="", lx=0, ly=0):
    """Draw a flow arrow, optional italic label."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2,
                                mutation_scale=22))
    if label:
        ax.text(x2 + lx, (y1 + y2) / 2 + ly, label, ha="center", va="center",
                color=color, fontsize=10, fontstyle="italic")


def make_analytics(path: str) -> None:
    """Analytics module: Gold views -> engine -> kpis -> dashboard."""
    fig, ax = plt.subplots(figsize=(10, 7.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    _box(ax, 5, 9.2, 3.6, 0.9, "Gold analytics views\n(PostgreSQL)", GOLD)
    _arrow(ax, 5, 8.75, 5, 8.05)
    _box(ax, 5, 7.6, 6.2, 0.95, "engine.py\nSELECT * FROM gold.<view> → DataFrame", NAVY,
         fontsize=11)
    _arrow(ax, 5, 7.12, 5, 6.42)
    _box(ax, 5, 5.95, 6.2, 0.95, "kpis.py\n5 clinical KPIs → get_all_kpis()", BLUE,
         fontsize=11)
    _arrow(ax, 5, 5.47, 5, 4.77)
    _box(ax, 5, 4.3, 6.2, 0.95, "dashboard/app.py  (Plotly Dash)", BLUE, fontsize=12)

    # Three dashboard outputs (separated boxes)
    _arrow(ax, 4.2, 3.82, 2.05, 3.15)
    _arrow(ax, 5, 3.82, 5, 3.15)
    _arrow(ax, 5.8, 3.82, 7.95, 3.15)
    _box(ax, 2.05, 2.65, 2.4, 1.0, "KPI cards", GREY, fontsize=11)
    _box(ax, 5, 2.65, 2.4, 1.0, "Overview\ncharts", GREY, fontsize=11)
    _box(ax, 7.95, 2.65, 2.6, 1.0, "Drill-down\n(dropdown\ncallback)", GREY, fontsize=10)

    _arrow(ax, 2.05, 2.15, 4.4, 1.35)
    _arrow(ax, 5, 2.15, 5, 1.35)
    _arrow(ax, 7.95, 2.15, 5.6, 1.35)
    _box(ax, 5, 0.9, 3.8, 0.85, "Hospital administrator", GREEN)

    ax.set_title("Analytics Module — read path", fontsize=15,
                 fontweight="bold", color=NAVY, pad=12)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_prediction(path: str) -> None:
    """Prediction module: silver.patients -> forecasting + classification."""
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 12); ax.set_ylim(0, 11.5); ax.axis("off")

    _box(ax, 6, 10.8, 3.6, 0.85, "silver.patients\n(per-patient rows)", GREEN,
         fontsize=12)
    _diamond(ax, 6, 9.5, 2.0, 1.2, "Task?")
    _arrow(ax, 5.3, 9.5, 3.0, 8.7, label="forecast", lx=-0.2, ly=0.3)
    _arrow(ax, 6.7, 9.5, 9.0, 8.7, label="risk", lx=0.2, ly=0.3)

    # Left branch — forecasting
    _box(ax, 3.0, 8.2, 3.4, 0.85, "timeseries.py\nmonthly admissions", NAVY, fontsize=11)
    _arrow(ax, 3.0, 7.77, 3.0, 7.07)
    _box(ax, 3.0, 6.6, 3.4, 0.85, "forecast.py\nHolt-Winters (trend+season)", BLUE,
         fontsize=10)
    _arrow(ax, 3.0, 6.17, 3.0, 5.47)
    _box(ax, 3.0, 5.0, 3.4, 0.85, "evaluate.py\nMAPE (A trend / C seasonal)", BLUE,
         fontsize=10)
    _arrow(ax, 3.0, 4.57, 3.0, 3.87)
    _box(ax, 3.0, 3.4, 3.4, 0.85, "planning.py\nvolume, %, staffing", GOLD, fontsize=11)

    # Right branch — classification
    _box(ax, 9.0, 8.2, 3.4, 0.85, "features.py\nmissingness + leakage guard", NAVY,
         fontsize=10)
    _arrow(ax, 9.0, 7.77, 9.0, 7.07)
    _box(ax, 9.0, 6.6, 3.4, 0.85, "models.py\nLogReg / RF / XGBoost", BLUE, fontsize=11)
    _arrow(ax, 9.0, 6.17, 9.0, 5.47)
    _box(ax, 9.0, 5.0, 3.4, 0.85, "evaluate.py\nprecision/recall/F1/ROC-AUC", BLUE,
         fontsize=10)
    _arrow(ax, 9.0, 4.57, 9.0, 3.87)
    _box(ax, 9.0, 3.4, 3.4, 0.85, "profile.py\nmulti-risk panel", GOLD, fontsize=11)

    # Converge
    _arrow(ax, 3.0, 2.97, 6, 2.15)
    _arrow(ax, 9.0, 2.97, 6, 2.15)
    _box(ax, 6, 1.7, 4.4, 0.9,
         "Hospital planning + clinical triage", GREEN, fontsize=12)

    ax.set_title("Prediction Module — forecasting + multi-risk classification",
                 fontsize=15, fontweight="bold", color=NAVY, pad=12)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    make_analytics("/home/claude/analytics_module.png")
    make_prediction("/home/claude/prediction_module.png")
    print("done")
