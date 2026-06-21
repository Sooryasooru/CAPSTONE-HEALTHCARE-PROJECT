"""Dashboard layer: clinical analytics + patient journey (Plotly Dash).

Reads only from the analytics layer; computes nothing itself.
    - KPI cards        <- derived from gold.patient_360 + revenue
    - analytics charts <- engine.get_* functions (connected Synthea views)
    - Patient 360      <- engine.get_patient_360()  (four tables joined)

Stage A: analytics + patient journey on the connected Synthea schema.
Forecasting and the multi-risk panel return in Stage B, after the
prediction layer is migrated.

Run from src/ with:  python -m dashboard.app
"""

import logging

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from analytics import engine
from etl.utils import get_logger

logger: logging.Logger = get_logger(__name__)

# Clinical palette (red reserved for mortality / deceased only)
INK = "#0B1F33"
TEAL = "#1A7A8C"
TEAL_SOFT = "rgba(26, 122, 140, 0.28)"
RED = "#C2453D"
AMBER = "#E8A33D"
GREEN = "#3C7A5A"
MUTED = "#6B7A88"

OUTCOME_COLOURS = {"Alive": GREEN, "Deceased": RED}

app = Dash(__name__)
app.title = "HAIP — Clinical Analytics"

# Patient 360 is read once at startup for KPIs and the journey panel.
_P360 = engine.get_patient_360()


# --- KPI cards -------------------------------------------------------------
def _kpis() -> list[dict]:
    """Derive headline KPIs from the connected patient_360 + revenue views."""
    p = _P360
    revenue = engine.get_revenue_by_month()
    total_patients = len(p)
    deceased_pct = round(100.0 * p["deceased"].sum() / total_patients, 1)
    avg_encounters = round(p["total_encounters"].mean(), 1)
    avg_cost = round(p["lifetime_cost"].mean(), 0)
    total_revenue = revenue["total_revenue"].sum()
    return [
        {"value": f"{total_patients:,}", "unit": "", "label": "Patients"},
        {"value": deceased_pct, "unit": "%", "label": "Mortality Rate"},
        {"value": avg_encounters, "unit": "", "label": "Avg Encounters / Patient"},
        {"value": f"{avg_cost:,.0f}", "unit": "", "label": "Avg Lifetime Cost ($)"},
        {"value": f"{total_revenue/1e6:.1f}", "unit": "M", "label": "Total Revenue ($)"},
    ]


def kpi_card(metric: dict) -> html.Div:
    """Render one KPI dict {value, label, unit} as an accented card."""
    accent = RED if metric["label"] == "Mortality Rate" else TEAL
    return html.Div(
        className="kpi-card",
        style={"borderLeft": f"5px solid {accent}"},
        children=[
            html.Div(f"{metric['value']}{metric['unit']}",
                     className="kpi-value", style={"color": accent}),
            html.Div(metric["label"], className="kpi-label"),
        ],
    )


# --- Overview charts -------------------------------------------------------
def fig_outcome_donut() -> go.Figure:
    """Patient outcome donut (alive / deceased)."""
    df = engine.get_outcome_distribution()
    colours = [OUTCOME_COLOURS.get(o, TEAL) for o in df["outcome"]]
    fig = go.Figure(go.Pie(
        labels=df["outcome"], values=df["patients"], hole=0.62,
        marker=dict(colors=colours), sort=False,
        textinfo="percent", textposition="outside",
        texttemplate="%{label}<br>%{percent}",
    ))
    fig.update_layout(
        title=dict(text="Patient Outcomes", x=0.02, y=0.97), showlegend=True,
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
        margin=dict(t=60, b=50, l=20, r=20), height=360,
        paper_bgcolor="white", font=dict(color=INK),
        annotations=[dict(text=f"{df['patients'].sum():,}<br>patients",
                          x=0.5, y=0.5, font=dict(size=14, color=MUTED),
                          showarrow=False)],
    )
    return fig


def fig_cost_by_condition() -> go.Figure:
    """Top conditions by total cost (conditions JOINed to encounters)."""
    df = engine.get_cost_by_condition().head(10).sort_values("total_cost")
    fig = go.Figure(go.Bar(
        x=df["total_cost"], y=df["condition"], orientation="h",
        marker=dict(color=TEAL),
        text=[f"${v/1e3:,.0f}K" for v in df["total_cost"]], textposition="auto",
    ))
    fig.update_layout(
        title=dict(text="Top conditions by total cost", x=0.02, y=0.97),
        height=360, margin=dict(t=60, b=40, l=10, r=20),
        paper_bgcolor="white", plot_bgcolor="white", font=dict(color=INK),
        xaxis=dict(title="total cost ($)", showgrid=True, gridcolor="#EEF2F5"),
        yaxis=dict(title=""),
    )
    return fig


# --- Revenue (financial) ---------------------------------------------------
def fig_revenue() -> go.Figure:
    """Monthly revenue trend from encounters."""
    df = engine.get_revenue_by_month().copy()
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month")
    fig = go.Figure()
    fig.add_scatter(x=df["month"], y=df["total_revenue"], name="Total revenue",
                    mode="lines", line=dict(color=TEAL, width=2))
    fig.add_scatter(x=df["month"], y=df["payer_paid"], name="Payer paid",
                    mode="lines", line=dict(color=GREEN, width=1.5, dash="dot"))
    fig.update_layout(
        title=dict(text="Monthly revenue trend", x=0.02, y=0.97),
        height=380, paper_bgcolor="white", plot_bgcolor="white", font=dict(color=INK),
        margin=dict(t=60, b=40, l=20, r=20),
        legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center"),
        yaxis=dict(title="revenue ($)", showgrid=True, gridcolor="#EEF2F5"),
    )
    return fig


# --- Patient 360 (flagship — connected journey) ----------------------------
def patient_journey_figure(patient_id: str) -> go.Figure:
    """One patient's journey summary — encounters, conditions, procedures, cost."""
    row = _P360.loc[_P360["patient_id"] == patient_id].iloc[0]
    metrics = ["Encounters", "Conditions", "Procedures"]
    values = [row["total_encounters"], row["distinct_conditions"],
              row["distinct_procedures"]]
    fig = go.Figure(go.Bar(
        x=metrics, y=values, marker=dict(color=[TEAL, AMBER, GREEN]),
        text=values, textposition="auto",
    ))
    fig.update_layout(
        title=dict(text=f"Journey — {row['gender']}, age {int(row['age'])}",
                   x=0.02, y=0.95),
        height=320, paper_bgcolor="white", plot_bgcolor="white", font=dict(color=INK),
        margin=dict(t=50, b=30, l=20, r=20),
        yaxis=dict(title="count", showgrid=True, gridcolor="#EEF2F5"),
    )
    return fig


def patient_summary(patient_id: str) -> html.Div:
    """Text summary card for a selected patient."""
    row = _P360.loc[_P360["patient_id"] == patient_id].iloc[0]
    status = "Deceased" if row["deceased"] else "Alive"
    return html.Div(className="drill-note", children=[
        html.Strong(f"Patient {patient_id[:8]}… · "),
        f"{row['gender']}, age {int(row['age'])}, {row['race']} · status: {status}. ",
        f"This patient has {int(row['total_encounters'])} encounters spanning "
        f"{int(row['distinct_conditions'])} distinct conditions and "
        f"{int(row['distinct_procedures'])} procedures, with a lifetime cost of "
        f"${row['lifetime_cost']:,.0f}. Every figure is joined from four tables "
        f"on this single patient ID.",
    ])


# --- Layout ----------------------------------------------------------------
def build_layout() -> html.Div:
    """Build the full page layout."""
    logger.info("Building dashboard layout")
    # Default to the patient with the most encounters (most interesting journey).
    top = _P360.sort_values("total_encounters", ascending=False)
    patient_options = [
        {"label": f"{r['patient_id'][:8]}… ({int(r['total_encounters'])} enc)",
         "value": r["patient_id"]}
        for _, r in top.head(200).iterrows()
    ]
    default_patient = top.iloc[0]["patient_id"]

    return html.Div(className="page", children=[
        html.Div(className="header", children=[
            html.Div("HAIP", className="logo"),
            html.Div([
                html.H1("Clinical Analytics", className="title"),
                html.P("Connected patient journeys, cost & population health",
                       className="subtitle"),
            ]),
        ]),

        html.Div(className="section-label", children="Key indicators"),
        html.Div(className="kpi-row",
                 children=[kpi_card(m) for m in _kpis()]),

        html.Div(className="section-label", children="Overview · population"),
        html.Div(className="chart-row", children=[
            html.Div(dcc.Graph(figure=fig_outcome_donut()), className="chart-half"),
            html.Div(dcc.Graph(figure=fig_cost_by_condition()), className="chart-half"),
        ]),

        html.Div(className="section-label", children="Financial · revenue"),
        html.Div(className="drill-panel", children=[
            dcc.Graph(figure=fig_revenue()),
            html.Div(className="drill-note", children=[
                "Monthly revenue from all encounters. Total claim cost vs the "
                "share covered by payers — the gap is patient responsibility.",
            ]),
        ]),

        html.Div(className="section-label",
                 children="Patient 360 · connected journey"),
        html.Div(className="drill-controls", children=[
            html.Label("Select patient", className="control-label"),
            dcc.Dropdown(id="patient-select", options=patient_options,
                         value=default_patient, clearable=False,
                         className="dropdown"),
        ]),
        html.Div(className="drill-panel", children=[
            dcc.Graph(id="journey-chart"),
            html.Div(id="patient-summary"),
        ]),

        html.Div(className="footer",
                 children="HAIP · presentation layer · connected analytics "
                          "from PostgreSQL (Synthea)"),
    ])


app.layout = build_layout()


# --- Callbacks -------------------------------------------------------------
@app.callback(
    Output("journey-chart", "figure"),
    Output("patient-summary", "children"),
    Input("patient-select", "value"),
)
def update_patient(patient_id: str):
    """Rebuild the patient journey chart and summary for the selection."""
    logger.info("Patient 360 for %s", patient_id)
    return patient_journey_figure(patient_id), patient_summary(patient_id)


if __name__ == "__main__":
    app.run(debug=True)
    