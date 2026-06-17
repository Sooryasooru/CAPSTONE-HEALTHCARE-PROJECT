"""Dashboard layer: clinical analytics presentation (Plotly Dash).

Reads only from the analytics layer:
    - KPI cards   <- kpis.get_all_kpis()
    - charts      <- engine.get_* functions
Computes nothing itself; it is a pure presentation layer.

The dashboard spans independent data domains (cardiac admissions and a
general ICU population), so the header stays neutral and each section
names its own source rather than implying a single unified cohort.

Run from src/ with:  python -m dashboard.app
"""

import logging

import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from analytics import engine, kpis
from etl.utils import get_logger

logger: logging.Logger = get_logger(__name__)

# Clinical palette (red is reserved for mortality / critical only)
INK = "#0B1F33"
TEAL = "#1A7A8C"
TEAL_SOFT = "rgba(26, 122, 140, 0.28)"  # translucent teal for context bars
RED = "#C2453D"
AMBER = "#E8A33D"
GREEN = "#3C7A5A"
MUTED = "#6B7A88"

# KPI label -> clinical accent colour (the "vital sign" tag on each card)
KPI_ACCENTS = {
    "Mortality Rate": RED,
    "DAMA Rate": AMBER,
    "ICU Sepsis Rate": RED,
    "30-Day ICU Readmission Rate": AMBER,
    "Avg Comorbidity Burden": TEAL,
}

# Outcome category -> colour (kept consistent across all charts)
OUTCOME_COLOURS = {"Discharge": GREEN, "Expiry": RED, "Dama": AMBER}

app = Dash(__name__)
app.title = "HAIP — Clinical Analytics"


# --- KPI cards -------------------------------------------------------------
def kpi_card(metric: dict) -> html.Div:
    """Render one KPI dict {value, label, unit} as an accented card."""
    accent = KPI_ACCENTS.get(metric["label"], TEAL)
    return html.Div(
        className="kpi-card",
        style={"borderLeft": f"5px solid {accent}"},
        children=[
            html.Div(f"{metric['value']}{metric['unit']}",
                     className="kpi-value", style={"color": accent}),
            html.Div(metric["label"], className="kpi-label"),
        ],
    )


# --- Overview charts (always visible) --------------------------------------
def fig_outcome_donut() -> go.Figure:
    """Build the cardiac-admission outcome donut (discharge / expiry / DAMA).

    Labels sit outside the ring so the small slices (Expiry, DAMA) do not
    collide, with a horizontal legend beneath.
    """
    df = engine.get_outcome_distribution()
    colours = [OUTCOME_COLOURS.get(o, TEAL) for o in df["outcome"]]
    fig = go.Figure(go.Pie(
        labels=df["outcome"], values=df["patients"], hole=0.62,
        marker=dict(colors=colours), sort=False,
        textinfo="percent", textposition="outside",
        texttemplate="%{label}<br>%{percent}",
    ))
    fig.update_layout(
        title=dict(text="Patient Outcomes", x=0.02, y=0.97),
        showlegend=True,
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
        margin=dict(t=60, b=50, l=20, r=20), height=360,
        paper_bgcolor="white", font=dict(color=INK),
        annotations=[dict(text=f"{df['patients'].sum():,}<br>patients",
                          x=0.5, y=0.5, font=dict(size=14, color=MUTED),
                          showarrow=False)],
    )
    return fig


def fig_comorbidity_bar() -> go.Figure:
    """Build the comorbidity prevalence ranked horizontal bar."""
    df = engine.get_comorbidity_prevalence().sort_values("prevalence_pct")
    fig = go.Figure(go.Bar(
        x=df["prevalence_pct"], y=df["condition"], orientation="h",
        marker=dict(color=TEAL),
        text=[f"{v:.1f}%" for v in df["prevalence_pct"]], textposition="auto",
    ))
    fig.update_layout(
        title=dict(text="Comorbidity Prevalence", x=0.02, y=0.97),
        height=360, margin=dict(t=60, b=40, l=10, r=20),
        paper_bgcolor="white", plot_bgcolor="white", font=dict(color=INK),
        xaxis=dict(title="% of cohort", showgrid=True, gridcolor="#EEF2F5"),
        yaxis=dict(title=""),
    )
    return fig


# --- Drill-down builders (chart + clinical interpretation) -----------------
# Each returns (figure, interpretation_text) for the callback to render.
def drill_outcomes() -> tuple[go.Figure, str]:
    """Cardiac-admission outcome counts with a mortality interpretation."""
    df = engine.get_outcome_distribution()
    fig = go.Figure(go.Bar(
        x=df["outcome"], y=df["patients"],
        marker=dict(color=[OUTCOME_COLOURS.get(o, TEAL) for o in df["outcome"]]),
        text=[f"{p:,}<br>{pct}%" for p, pct in zip(df["patients"], df["pct"])],
        textposition="auto",
    ))
    fig.update_layout(
        title=dict(text="Outcome counts", x=0.02, y=0.97),
        height=380, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(color=INK), margin=dict(t=60, b=40, l=20, r=20),
        yaxis=dict(title="patients", showgrid=True, gridcolor="#EEF2F5"),
    )
    expiry = df.loc[df["outcome"] == "Expiry", "pct"].iloc[0]
    note = (f"Cardiac admissions: in-hospital mortality is {expiry}% of "
            f"admissions. The remainder discharge alive, with a smaller share "
            f"leaving against medical advice (DAMA) — a care-continuity signal "
            f"worth monitoring.")
    return fig, note


def drill_comorbidities() -> tuple[go.Figure, str]:
    """Ranked comorbidity prevalence with a cardiac-cohort interpretation."""
    df = engine.get_comorbidity_prevalence().sort_values(
        "prevalence_pct", ascending=False)
    fig = go.Figure(go.Bar(
        x=df["condition"], y=df["prevalence_pct"], marker=dict(color=TEAL),
        text=[f"{v:.1f}%" for v in df["prevalence_pct"]], textposition="auto",
    ))
    fig.update_layout(
        title=dict(text="Comorbidity prevalence (ranked)", x=0.02, y=0.97),
        height=380, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(color=INK), margin=dict(t=60, b=40, l=20, r=20),
        yaxis=dict(title="% of cohort", showgrid=True, gridcolor="#EEF2F5"),
    )
    top = df.iloc[0]
    note = (f"Cardiac admissions: {top['condition']} dominates at "
            f"{top['prevalence_pct']:.1f}%, consistent with a cardiac cohort. "
            f"Overlapping conditions (hypertension, diabetes) compound "
            f"cardiovascular risk.")
    return fig, note


def drill_icu() -> tuple[go.Figure, str]:
    """ICU sepsis rate vs patient volume by SOFA band.

    Patient counts are translucent teal context bars on the left axis; the
    sepsis RATE is the bold red line on the right axis. Plotting the rate
    (not raw counts) surfaces the clinical story — sepsis escalates with
    organ-failure severity even though the lower-severity bands hold most
    of the patients.
    """
    df = engine.get_icu_severity_summary()
    df["sepsis_pct"] = (100 * df["sepsis_cases"] / df["patients"]).round(1)

    fig = go.Figure()
    fig.add_bar(
        x=df["sofa_band"], y=df["patients"], name="Patients (volume)",
        marker=dict(color=TEAL_SOFT, line=dict(color=TEAL, width=1)),
        yaxis="y",
        text=[f"{p:,}" for p in df["patients"]], textposition="outside",
        textfont=dict(color=TEAL),
    )
    fig.add_scatter(
        x=df["sofa_band"], y=df["sepsis_pct"], name="Sepsis rate",
        mode="lines+markers+text", yaxis="y2",
        line=dict(color=RED, width=3), marker=dict(size=10, color=RED),
        text=[f"{v:.0f}%" for v in df["sepsis_pct"]], textposition="top center",
        textfont=dict(color=RED, size=13),
    )
    fig.update_layout(
        title=dict(text="ICU sepsis rate by SOFA band", x=0.02, y=0.97),
        height=380, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(color=INK), margin=dict(t=80, b=40, l=20, r=20),
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        yaxis=dict(title="patients", showgrid=True, gridcolor="#EEF2F5"),
        yaxis2=dict(title="sepsis rate (%)", overlaying="y", side="right",
                    range=[0, 105], showgrid=False),
    )
    note = ("Critical care (general ICU population): patient volume is "
            "concentrated in the lower-severity bands, but the sepsis rate "
            "(red line) escalates sharply with SOFA score — from near-zero to "
            "the entire critical band. Plotting the rate rather than raw counts "
            "reveals the escalation that volume alone would hide.")
    return fig, note


# Design note: the two Overview charts are a fixed summary that always shows
# the headline picture. The dropdown drives ONLY the drill-down panel below —
# a stable overview plus one interactive detail view is the standard analytics
# pattern (a constant reference point the user can always return to).
DRILL_VIEWS = {
    "outcomes": drill_outcomes,
    "comorbidities": drill_comorbidities,
    "icu": drill_icu,
}


# --- Layout ----------------------------------------------------------------
def build_layout() -> html.Div:
    """Build the full page layout (cards, overview charts, drill-down)."""
    logger.info("Building dashboard layout")
    return html.Div(className="page", children=[
        html.Div(className="header", children=[
            html.Div("HAIP", className="logo"),
            html.Div([
                html.H1("Clinical Analytics", className="title"),
                html.P("Population health and critical care analytics",
                       className="subtitle"),
            ]),
        ]),

        html.Div(className="section-label", children="Key indicators"),
        html.Div(className="kpi-row",
                 children=[kpi_card(m) for m in kpis.get_all_kpis()]),

        html.Div(className="section-label",
                 children="Overview · cardiac admissions"),
        html.Div(className="chart-row", children=[
            html.Div(dcc.Graph(figure=fig_outcome_donut()), className="chart-half"),
            html.Div(dcc.Graph(figure=fig_comorbidity_bar()), className="chart-half"),
        ]),

        html.Div(className="section-label", children="Drill-down"),
        html.Div(className="drill-controls", children=[
            html.Label("Clinical domain", className="control-label"),
            dcc.Dropdown(
                id="drill-select",
                options=[
                    {"label": "Outcomes (cardiac)", "value": "outcomes"},
                    {"label": "Comorbidities (cardiac)", "value": "comorbidities"},
                    {"label": "ICU severity (critical care)", "value": "icu"},
                ],
                value="icu", clearable=False, className="dropdown",
            ),
        ]),
        html.Div(className="drill-panel", children=[
            dcc.Graph(id="drill-chart"),
            html.Div(id="drill-note", className="drill-note"),
        ]),

        html.Div(className="footer",
                 children="HAIP · presentation layer · independent data domains "
                          "from PostgreSQL gold views"),
    ])


app.layout = build_layout()


# --- Callback: dropdown drives the drill-down chart + interpretation -------
@app.callback(
    Output("drill-chart", "figure"),
    Output("drill-note", "children"),
    Input("drill-select", "value"),
)
def update_drill(view: str) -> tuple[go.Figure, str]:
    """Rebuild the drill-down chart and clinical note for the selected domain."""
    logger.info("Drill-down view selected: %s", view)
    fig, note = DRILL_VIEWS[view]()
    return fig, note


if __name__ == "__main__":
    app.run(debug=True)