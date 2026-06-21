"""Dashboard layer: clinical analytics + prediction presentation (Plotly Dash).

Reads only from the analytics and prediction layers; computes nothing itself.
    - KPI cards        <- kpis.get_all_kpis()
    - analytics charts <- engine.get_* functions
    - forecast         <- prediction.forecast / prediction.planning
    - risk panel       <- prediction.classification.profile

The dashboard spans independent data domains (cardiac admissions and a general
ICU population), so the header stays neutral and each section names its source.

Run from src/ with:  python -m dashboard.app
"""

import logging

import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from analytics import engine, kpis
from etl.utils import get_logger
from prediction.classification import profile
from prediction.classification.features import get_features
from prediction.forecast import forecast_admissions
from prediction.planning import build_planning
from prediction.timeseries import build_monthly_admissions

logger: logging.Logger = get_logger(__name__)

# Clinical palette (red is reserved for mortality / critical only)
INK = "#0B1F33"
TEAL = "#1A7A8C"
TEAL_SOFT = "rgba(26, 122, 140, 0.28)"
RED = "#C2453D"
AMBER = "#E8A33D"
GREEN = "#3C7A5A"
MUTED = "#6B7A88"

KPI_ACCENTS = {
    "Mortality Rate": RED,
    "DAMA Rate": AMBER,
    "ICU Sepsis Rate": RED,
    "30-Day ICU Readmission Rate": AMBER,
    "Avg Comorbidity Burden": TEAL,
}
OUTCOME_COLOURS = {"Discharge": GREEN, "Expiry": RED, "Dama": AMBER}

# Risk label -> colour for the patient risk panel
RISK_COLOURS = {
    "mortality": RED,
    "aki": AMBER,
    "heart_failure": TEAL,
}

app = Dash(__name__)
app.title = "HAIP — Clinical Analytics"

# Patient count for the risk-panel dropdown (built once at startup).
_N_PATIENTS = len(get_features())


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
    """Build the cardiac-admission outcome donut (discharge / expiry / DAMA)."""
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


# --- Drill-down builders ---------------------------------------------------
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
            f"leaving against medical advice (DAMA).")
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
            f"{top['prevalence_pct']:.1f}%, consistent with a cardiac cohort.")
    return fig, note


def drill_icu() -> tuple[go.Figure, str]:
    """ICU sepsis rate vs patient volume by SOFA band."""
    df = engine.get_icu_severity_summary()
    df["sepsis_pct"] = (100 * df["sepsis_cases"] / df["patients"]).round(1)
    fig = go.Figure()
    fig.add_bar(x=df["sofa_band"], y=df["patients"], name="Patients (volume)",
                marker=dict(color=TEAL_SOFT, line=dict(color=TEAL, width=1)),
                yaxis="y", text=[f"{p:,}" for p in df["patients"]],
                textposition="outside", textfont=dict(color=TEAL))
    fig.add_scatter(x=df["sofa_band"], y=df["sepsis_pct"], name="Sepsis rate",
                    mode="lines+markers+text", yaxis="y2",
                    line=dict(color=RED, width=3), marker=dict(size=10, color=RED),
                    text=[f"{v:.0f}%" for v in df["sepsis_pct"]],
                    textposition="top center", textfont=dict(color=RED, size=13))
    fig.update_layout(
        title=dict(text="ICU sepsis rate by SOFA band", x=0.02, y=0.97),
        height=380, paper_bgcolor="white", plot_bgcolor="white", font=dict(color=INK),
        margin=dict(t=80, b=40, l=20, r=20),
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        yaxis=dict(title="patients", showgrid=True, gridcolor="#EEF2F5"),
        yaxis2=dict(title="sepsis rate (%)", overlaying="y", side="right",
                    range=[0, 105], showgrid=False),
    )
    note = ("Critical care (general ICU population): sepsis prevalence climbs "
            "sharply with SOFA severity. Plotting the rate rather than raw "
            "counts reveals the escalation that volume alone would hide.")
    return fig, note


DRILL_VIEWS = {
    "outcomes": drill_outcomes,
    "comorbidities": drill_comorbidities,
    "icu": drill_icu,
}


# --- Forecasting (prediction) ----------------------------------------------
def fig_forecast() -> go.Figure:
    """Historical admissions + the Holt-Winters forecast."""
    hist = build_monthly_admissions()
    fc = forecast_admissions(6)
    fig = go.Figure()
    fig.add_scatter(x=hist["month"], y=hist["admissions"], name="Actual",
                    mode="lines+markers", line=dict(color=TEAL, width=2),
                    marker=dict(size=5))
    # Join the forecast to the last actual point for a continuous line.
    bridge_x = [hist["month"].iloc[-1]] + list(fc["month"])
    bridge_y = [hist["admissions"].iloc[-1]] + list(fc["forecast"])
    fig.add_scatter(x=bridge_x, y=bridge_y, name="Forecast",
                    mode="lines+markers", line=dict(color=AMBER, width=2, dash="dash"),
                    marker=dict(size=5))
    fig.update_layout(
        title=dict(text="Monthly admissions — actual + 6-month forecast", x=0.02, y=0.97),
        height=380, paper_bgcolor="white", plot_bgcolor="white", font=dict(color=INK),
        margin=dict(t=60, b=40, l=20, r=20),
        legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center"),
        yaxis=dict(title="admissions / month", showgrid=True, gridcolor="#EEF2F5"),
    )
    return fig


def planning_table() -> html.Table:
    """Render the planning forecast as an HTML table."""
    df = build_planning(6)
    df["month"] = df["month"].dt.strftime("%b %Y")
    header = ["Month", "Forecast", "Change %", "Cardiac cases", "Nurse est."]
    cols = ["month", "forecast", "change_pct", "cardiac_cases", "nurse_shifts"]
    head = html.Tr([html.Th(h) for h in header])
    rows = [html.Tr([html.Td(str(r[c])) for c in cols])
            for _, r in df.iterrows()]
    return html.Table(className="plan-table", children=[head, *rows])


# --- Patient risk panel (prediction) ---------------------------------------
def risk_panel_figure(index: int) -> go.Figure:
    """Horizontal bar chart of one patient's three risk scores."""
    result = profile.score_patient_by_index(index)
    risks = result["risks"]
    labels = [r["label"] for r in risks]
    values = [r["probability_pct"] for r in risks]
    colours = [RISK_COLOURS.get(r["risk"], TEAL) for r in risks]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker=dict(color=colours),
        text=[f"{v:.1f}%" for v in values], textposition="auto",
    ))
    fig.update_layout(
        title=dict(text=f"Risk profile — patient #{index}", x=0.02, y=0.95),
        height=300, paper_bgcolor="white", plot_bgcolor="white", font=dict(color=INK),
        margin=dict(t=50, b=30, l=20, r=20),
        xaxis=dict(title="risk (%)", range=[0, 100], showgrid=True, gridcolor="#EEF2F5"),
    )
    return fig


# --- Layout ----------------------------------------------------------------
def build_layout() -> html.Div:
    """Build the full page layout."""
    logger.info("Building dashboard layout")
    patient_options = [{"label": f"Patient #{i}", "value": i}
                       for i in range(0, min(_N_PATIENTS, 200))]
    return html.Div(className="page", children=[
        html.Div(className="header", children=[
            html.Div("HAIP", className="logo"),
            html.Div([
                html.H1("Clinical Analytics", className="title"),
                html.P("Population health, forecasting & risk prediction",
                       className="subtitle"),
            ]),
        ]),

        html.Div(className="section-label", children="Key indicators"),
        html.Div(className="kpi-row",
                 children=[kpi_card(m) for m in kpis.get_all_kpis()]),

        html.Div(className="section-label", children="Overview · cardiac admissions"),
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

        # --- Forecasting section ---
        html.Div(className="section-label", children="Forecasting · admissions"),
        html.Div(className="drill-panel", children=[
            dcc.Graph(figure=fig_forecast()),
            html.Div(className="drill-note", children=[
                "Holt-Winters projection of monthly admissions. The planning "
                "table below translates the forecast into operational figures "
                "(cardiac cases use the cohort's 67% CAD prevalence; the nurse "
                "estimate uses a stated 1:4 ratio).",
            ]),
            planning_table(),
        ]),

        # --- Patient risk section ---
        html.Div(className="section-label", children="Patient risk · early-warning panel"),
        html.Div(className="drill-controls", children=[
            html.Label("Select patient", className="control-label"),
            dcc.Dropdown(id="patient-select", options=patient_options,
                         value=0, clearable=False, className="dropdown"),
        ]),
        html.Div(className="drill-panel", children=[
            dcc.Graph(id="risk-chart"),
            html.Div(className="drill-note", children=[
                "Multi-risk triage aid (not a diagnosis). Scores are produced "
                "by the best model per risk: XGBoost for mortality, Random "
                "Forest for AKI and heart failure. Scoped to a cardiac cohort.",
            ]),
        ]),

        html.Div(className="footer",
                 children="HAIP · presentation layer · analytics + prediction "
                          "from PostgreSQL and trained models"),
    ])


app.layout = build_layout()


# --- Callbacks -------------------------------------------------------------
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


@app.callback(
    Output("risk-chart", "figure"),
    Input("patient-select", "value"),
)
def update_risk(index: int) -> go.Figure:
    """Rebuild the patient risk panel for the selected patient index."""
    logger.info("Risk panel for patient #%s", index)
    return risk_panel_figure(index)


if __name__ == "__main__":
    app.run(debug=True)
    