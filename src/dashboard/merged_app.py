"""HAIP unified dashboard: connected analytics + hospital data intake (Dash).

Merges the two former apps into one tabbed application on a single port:

    Tab 1 · Analytics   <- the connected Synthea dashboard (was app.py:8050)
                           KPIs, population overview, revenue, forecasting,
                           Patient 360, and the 4-target risk panel.
    Tab 2 · Data Intake <- the hospital upload tool (was upload_app.py:8051)
                           drag-drop -> schema validation -> column mapping
                           -> EDA -> AutoML prediction.

Lazy loading: nothing heavy runs at import. Each tab's layout is built by a
callback when that tab is first selected, so startup stays light on a
RAM-constrained machine. The Synthea views (patient_360, features, figures)
load only when the Analytics tab is opened; the intake tab stays idle until
selected.

Run from src/ with:  python -m dashboard.merged_app   ->  http://127.0.0.1:8050
"""

import base64
import difflib
import io
import logging

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, dash_table, no_update

from analytics import engine
from etl.utils import get_logger
from prediction.classification import profile
from prediction.classification import evaluate as cls_evaluate
from prediction.classification.features import get_features
from prediction.forecast import forecast_admissions
from prediction.planning import build_planning
from dashboard.theme import (apply_theme, INK, TEAL, AMBER, RED, GREEN, MUTED)
from summary_creation.pdf_report import build_hospital_report

logger: logging.Logger = get_logger(__name__)

# --- Shared palette --------------------------------------------------------
# Colour tokens come from dashboard.theme (single source of truth). Only the
# dashboard-specific extra lives here.
TEAL_SOFT = "rgba(26, 122, 140, 0.28)"

OUTCOME_COLOURS = {"Alive": GREEN, "Deceased": RED}
# 4 connected targets — deterioration added alongside the original three.
RISK_COLOURS = {
    "readmission": AMBER,
    "mortality": RED,
    "high_cost": TEAL,
    "deterioration": GREEN,
}

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "HAIP — Data Intake & Prediction"
app.index_string = '''<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
  html,body{margin:0;background:#12140F;color:#EAEEE6;}
  .page{background:#12140F!important;color:#EAEEE6;} #upload-data{display:flex!important;align-items:center;justify-content:center;}
  ._dash-loading{color:#4ADE80;}
  .dash-table-container .dash-spreadsheet td{background:#181B15!important;color:#EAEEE6!important;}
  input,textarea{background:#181B15!important;color:#EAEEE6!important;border-color:#2B2F28!important;} .dash-uploader,#upload-data>div,#upload-data{background:#12140F!important;color:#EAEEE6!important;} #upload-status,#upload-status>*{background:transparent!important;} #upload-data:focus,#upload-data *:focus{outline:none!important;} #mapping-section:empty,#validation-section:empty,#analysis-section:empty,#prediction-section:empty{display:none!important;border:none!important;} .drill-note:empty{display:none!important;}
</style></head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>'''


# Allow this dashboard to be embedded in the React app via iframe.
@app.server.after_request
def _allow_iframe(response):
    response.headers.pop("X-Frame-Options", None)
    return response


# ===========================================================================
# LAZY SHARED STATE
# ---------------------------------------------------------------------------
# The analytics tab needs patient_360 and the encounter count. We load these
# once, on first use, instead of at import — so the upload tab (and app
# startup) never pays the cost.
# ===========================================================================
_CACHE: dict = {}


def _p360() -> pd.DataFrame:
    """patient_360, loaded once on first access (cached)."""
    if "p360" not in _CACHE:
        logger.info("Loading patient_360 (first access)")
        _CACHE["p360"] = engine.get_patient_360()
    return _CACHE["p360"]


def _n_encounters() -> int:
    """Encounter count for the risk dropdown, computed once on first access."""
    if "n_enc" not in _CACHE:
        _CACHE["n_enc"] = len(get_features())
    return _CACHE["n_enc"]


# ===========================================================================
# ANALYTICS TAB  (formerly app.py)
# ===========================================================================

# --- KPI cards -------------------------------------------------------------
# --- KPI icons (inline SVG — no external dependency) -----------------------
# Small, single-stroke clinical glyphs. Each returns an SVG string sized to
# inherit the card accent colour via currentColor.
_ICON_PATHS = {
    "patients": "M16 11a4 4 0 1 0-8 0M4 21v-1a6 6 0 0 1 12 0v1M20 8v6M23 11h-6",
    "heart": "M20.8 5.6a5 5 0 0 0-7.1 0L12 7.3l-1.7-1.7a5 5 0 1 0-7.1 7.1L12 21l8.8-8.3a5 5 0 0 0 0-7.1z",
    "activity": "M22 12h-4l-3 9L9 3l-3 9H2",
    "wallet": "M21 8V6a2 2 0 0 0-2-2H5a2 2 0 0 0 0 4h16a1 1 0 0 1 1 1v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6M18 12h.01",
    "trending": "M23 6l-9.5 9.5-5-5L1 18M17 6h6v6",
    "calendar": "M19 4H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2zM16 2v4M8 2v4M3 10h18",
}


def _icon(name: str, color: str) -> html.Img:
    """Return a small SVG icon as a data-URI image (Dash-safe, colourable).

    The stroke colour is baked into the SVG so each card's icon matches its
    accent. No external files or HTML injection.
    """
    import base64 as _b64
    path = _ICON_PATHS.get(name, _ICON_PATHS["activity"])
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        f'<path d="{path}"/></svg>'
    )
    data = _b64.b64encode(svg.encode()).decode()
    return html.Img(src=f"data:image/svg+xml;base64,{data}",
                    className="kpi-icon")


def _kpis() -> list[dict]:
    """Derive headline KPIs from the connected patient_360 + revenue views."""
    p = _p360()
    revenue = engine.get_revenue_by_month()
    total_patients = len(p)
    deceased_pct = round(100.0 * p["deceased"].sum() / total_patients, 1)
    avg_encounters = round(p["total_encounters"].mean(), 1)
    avg_cost = round(p["lifetime_cost"].mean(), 0)
    total_revenue = revenue["total_revenue"].sum()
    return [
        {"value": f"{total_patients:,}", "unit": "", "label": "Patients",
         "icon": "patients"},
        {"value": deceased_pct, "unit": "%", "label": "Mortality Rate",
         "icon": "heart"},
        {"value": avg_encounters, "unit": "", "label": "Avg Encounters / Patient",
         "icon": "activity"},
        {"value": f"{avg_cost:,.0f}", "unit": "", "label": "Avg Lifetime Cost ($)",
         "icon": "wallet"},
        {"value": f"{total_revenue/1e6:.1f}", "unit": "M", "label": "Total Revenue ($)",
         "icon": "trending"},
    ]


def kpi_card(metric: dict) -> html.Div:
    """Render one KPI dict {value, label, unit, icon} as an accented card."""
    accent = RED if metric["label"] == "Mortality Rate" else TEAL
    icon_name = metric.get("icon", "activity")
    return html.Div(
        className="kpi-card",
        style={"borderTop": f"3px solid {accent}"},
        children=[
            html.Div(className="kpi-head", children=[
                html.Div(metric["label"], className="kpi-label"),
                _icon(icon_name, accent),
            ]),
            html.Div(f"{metric['value']}{metric['unit']}",
                     className="kpi-value", style={"color": accent}),
        ],
    )


# --- Overview charts -------------------------------------------------------
def fig_outcome_donut() -> go.Figure:
    """Patient outcome donut (alive / deceased)."""
    df = engine.get_outcome_distribution()
    colours = [OUTCOME_COLOURS.get(o, TEAL) for o in df["outcome"]]
    fig = go.Figure(go.Pie(
        labels=df["outcome"], values=df["patients"], hole=0.62,
        marker=dict(colors=colours, line=dict(color="white", width=2)),
        sort=False, direction="clockwise",
        textinfo="percent", textposition="inside",
        insidetextorientation="horizontal",
        texttemplate="%{percent}",
        hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"),
        height=360,
        margin=dict(t=60, b=60, l=20, r=20),
        annotations=[dict(text=f"{df['patients'].sum():,}<br>patients",
                          x=0.5, y=0.5, font=dict(size=14, color=MUTED),
                          showarrow=False)],
    )
    return apply_theme(fig, title="Patient Outcomes")


def fig_cost_by_condition() -> go.Figure:
    """Top conditions by total cost (conditions JOINed to encounters)."""
    df = engine.get_cost_by_condition().head(10).sort_values("total_cost")
    fig = go.Figure(go.Bar(
        x=df["total_cost"], y=df["condition"], orientation="h",
        marker=dict(color=TEAL),
        text=[f"${v/1e3:,.0f}K" for v in df["total_cost"]], textposition="auto",
    ))
    fig.update_layout(height=360, yaxis=dict(title=""),
                      xaxis=dict(title="total cost ($)"))
    return apply_theme(fig, title="Top conditions by total cost")


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
        height=380,
        yaxis=dict(title="revenue ($)"),
    )
    return apply_theme(fig, title="Monthly revenue trend")


# --- Forecasting (prediction) ----------------------------------------------
def fig_forecast() -> go.Figure:
    """Historical encounter volume + the Holt-Winters forecast."""
    hist = build_monthly_admissions()
    fc = forecast_admissions(6)
    hist_recent = hist.tail(36)
    fig = go.Figure()
    fig.add_scatter(x=hist_recent["month"], y=hist_recent["admissions"],
                    name="Actual", mode="lines+markers",
                    line=dict(color=TEAL, width=2), marker=dict(size=4))
    bridge_x = [hist["month"].iloc[-1]] + list(fc["month"])
    bridge_y = [hist["admissions"].iloc[-1]] + list(fc["forecast"])
    fig.add_scatter(x=bridge_x, y=bridge_y, name="Forecast",
                    mode="lines+markers",
                    line=dict(color=AMBER, width=2, dash="dash"),
                    marker=dict(size=4))
    apply_theme(fig, title="Monthly encounters — actual + 6-month forecast", height=380)
    return fig


def planning_table() -> html.Table:
    """Render the planning forecast as an HTML table."""
    df = build_planning(6)
    df["month"] = df["month"].dt.strftime("%b %Y")
    header = ["Month", "Forecast", "Change %", "Inpatient cases", "Nurse est."]
    cols = ["month", "forecast", "change_pct", "inpatient_cases", "nurse_shifts"]
    head = html.Tr([html.Th(h) for h in header])
    rows = [html.Tr([html.Td(str(r[c])) for c in cols])
            for _, r in df.iterrows()]
    return html.Table(className="plan-table", children=[head, *rows])


# --- Patient risk panel (prediction — connected targets) -------------------
def risk_panel_figure(index: int) -> go.Figure:
    """Horizontal bar chart of one stay's connected risk scores (4 targets)."""
    result = profile.score_encounter_by_index(index)
    risks = result["risks"]
    labels = [r["label"] for r in risks]
    values = [r["probability_pct"] for r in risks]
    colours = [RISK_COLOURS.get(r["risk"], TEAL) for r in risks]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker=dict(color=colours),
        text=[f"{v:.1f}%" for v in values], textposition="auto",
    ))
    apply_theme(fig, title=f"Risk profile — inpatient stay #{index}", height=320)
    return fig


# --- Confusion matrix panel (model evaluation — best model per risk) --------
def _eval_table():
    """Evaluate all models once and cache the result (12 model loads).

    Cached in module memory so the dropdown re-renders instantly instead of
    re-evaluating on every change.
    """
    if "eval_table" not in _CACHE:
        logger.info("Evaluating all risk models (first access)")
        table = cls_evaluate.evaluate_all()
        _CACHE["eval_table"] = table
        _CACHE["eval_best"] = cls_evaluate.best_per_risk(table)
    return _CACHE["eval_best"]


def confusion_figure(risk: str) -> go.Figure:
    """Confusion-matrix heatmap for one risk's best model (cached eval)."""
    best = _eval_table()
    row = best[best["risk"] == risk].iloc[0]
    tn, fp, fn, tp = int(row["TN"]), int(row["FP"]), int(row["FN"]), int(row["TP"])
    # z laid out to match labels: rows = actual, cols = predicted
    z = [[tn, fp], [fn, tp]]
    labels_text = [[f"True Negative<br>{tn}", f"False Positive<br>{fp}"],
                   [f"False Negative<br>{fn}", f"True Positive<br>{tp}"]]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Predicted: No", "Predicted: Yes"],
        y=["Actual: No risk", "Actual: At risk"],
        text=labels_text, texttemplate="%{text}", textfont=dict(size=13),
        colorscale=[[0, "#12140F"], [1, TEAL]], showscale=False,
        hovertemplate="%{y} / %{x}: %{z}<extra></extra>"))
    apply_theme(fig, title=f"Confusion matrix — {risk} (best: {row['model']})",
                height=360)
    fig.update_yaxes(autorange="reversed")
    return fig


# --- Patient 360 (flagship — connected journey) ----------------------------
def patient_journey_figure(patient_id: str) -> go.Figure:
    """One patient's journey summary — encounters, conditions, procedures."""
    row = _p360().loc[_p360()["patient_id"] == patient_id].iloc[0]
    metrics = ["Encounters", "Conditions", "Procedures"]
    values = [row["total_encounters"], row["distinct_conditions"],
              row["distinct_procedures"]]
    fig = go.Figure(go.Bar(
        x=metrics, y=values, marker=dict(color=[TEAL, AMBER, GREEN]),
        text=values, textposition="auto",
    ))
    apply_theme(fig, title=f"Journey — {row['gender']}, age {int(row['age'])}", height=320)
    return fig


def patient_summary(patient_id: str) -> html.Div:
    """Text summary card for a selected patient."""
    row = _p360().loc[_p360()["patient_id"] == patient_id].iloc[0]
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


REQUIRED_COLUMNS = {
    "patient_id":     "Unique patient identifier (the join key)",
    "encounter_date": "Admission / visit date",
    "encounter_class": "Encounter type (inpatient, emergency, ...)",
    "total_cost":     "Encounter cost",
}
RECOMMENDED_COLUMNS = {
    "age":            "Patient age",
    "gender":         "Patient gender",
    "condition":      "Diagnosis / condition",
    "procedure":      "Procedure performed",
    "discharge_date": "Discharge date",
}
ALL_STANDARD = {**REQUIRED_COLUMNS, **RECOMMENDED_COLUMNS}



def _parse_upload(contents: str, filename: str) -> pd.DataFrame:
    """Decode an uploaded file's base64 contents into a DataFrame."""
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    name = filename.lower()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(decoded))
    if name.endswith(".json"):
        return pd.read_json(io.BytesIO(decoded))
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(decoded))
    raise ValueError(f"unsupported file type: {filename}")


def _auto_match(std_col: str, file_cols: list[str]) -> str | None:
    """Best-guess which uploaded column maps to a standard column."""
    norm = {c.lower().replace("_", "").replace(" ", ""): c for c in file_cols}
    target = std_col.lower().replace("_", "")
    for key, original in norm.items():
        if target in key or key in target:
            return original
    close = difflib.get_close_matches(target, list(norm.keys()), n=1, cutoff=0.8)
    return norm[close[0]] if close else None


def _quality(series: pd.Series) -> dict:
    """Per-column data-quality summary."""
    n = len(series)
    missing = int(series.isna().sum())
    return {
        "missing_pct": round(100 * missing / n, 1) if n else 0.0,
        "dtype": str(series.dtype),
        "unique": int(series.nunique(dropna=True)),
    }


def build_intake_layout() -> html.Div:
    """Build the data-intake tab layout (lazy — called on tab select)."""
    logger.info("Building data-intake tab layout")
    return html.Div(children=[
        html.Div(className="section-label", children="Step 1 · Upload a dataset"),
        dcc.Upload(
            id="upload-data",
            children=html.Div([
                html.Div("↑", style={"fontSize": "26px", "color": TEAL,
                                          "lineHeight": "1", "marginBottom": "10px"}),
                html.Div([
                    "Drag and drop a file here, or ",
                    html.A("browse", style={"color": TEAL, "fontWeight": "600",
                                            "textDecoration": "none"}),
                ], style={"color": INK, "fontSize": "15px", "fontWeight": "500"}),
                html.Span("CSV, JSON, or Excel",
                          style={"color": MUTED, "fontSize": "13px",
                                 "marginTop": "6px", "display": "block"}),
            ]),
            style={
                "width": "100%", "borderWidth": "1px",
                "borderStyle": "dashed", "borderColor": "#2B2F28",
                "borderRadius": "14px", "textAlign": "center",
                "padding": "36px 0", "background": "#181B15",
                "cursor": "pointer",
            },
            multiple=False,
        ),
        html.Div(id="upload-status", className="drill-note"),

        html.Div(id="mapping-section"),
        html.Div(id="validation-section"),
        html.Div(id="analysis-section"),
        html.Div(id="report-section", style={"marginTop": "16px", "display": "none"}, children=[
            html.Button("Download PDF report", id="dl-report-btn", n_clicks=0,
                        style={"padding": "10px 24px", "background": AMBER,
                               "color": "white", "border": "none",
                               "borderRadius": "8px", "cursor": "pointer",
                               "fontSize": "15px"}),
            dcc.Download(id="report-download"),
        ]),
        html.Div(id="prediction-section"),

        dcc.Store(id="data-store"),
        dcc.Store(id="mapping-store"),
    ])


@app.callback(
    Output("data-store", "data"),
    Output("upload-status", "children"),
    Output("mapping-section", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename):
    """Parse the file and build the column-mapping UI with auto-matches."""
    if contents is None:
        return None, "", None
    try:
        df = _parse_upload(contents, filename)
    except Exception as exc:
        return None, f"Could not read file: {exc}", None
    if df.empty or len(df.columns) < 2:
        return None, "File needs at least 2 columns and some rows.", None

    file_cols = list(df.columns)
    skip_option = "— skip / not present —"
    dropdown_opts = [{"label": skip_option, "value": "__none__"}] + \
                    [{"label": c, "value": c} for c in file_cols]

    rows = []
    for std_col, desc in ALL_STANDARD.items():
        guess = _auto_match(std_col, file_cols)
        required = std_col in REQUIRED_COLUMNS
        tag = "REQUIRED" if required else "recommended"
        tag_colour = RED if required else AMBER
        rows.append(html.Div(style={
            "display": "grid",
            "gridTemplateColumns": "200px 1fr 110px",
            "gap": "12px", "alignItems": "center", "marginBottom": "8px",
        }, children=[
            html.Div([
                html.Strong(std_col, style={"color": INK}),
                html.Div(desc, style={"fontSize": "11px", "color": MUTED}),
            ]),
            dcc.Dropdown(
                id={"type": "map", "std": std_col},
                options=dropdown_opts,
                value=guess if guess else "__none__",
                clearable=False, className="dropdown",
            ),
            html.Span(tag, style={
                "color": "white", "background": tag_colour,
                "padding": "3px 8px", "borderRadius": "6px",
                "fontSize": "11px", "textAlign": "center",
            }),
        ]))

    preview = dash_table.DataTable(
        data=df.head(5).to_dict("records"),
        columns=[{"name": c, "id": c} for c in file_cols],
        style_table={"overflowX": "auto"},
        style_cell={"fontSize": "12px", "padding": "6px",
                    "fontFamily": "inherit", "textAlign": "left"},
        style_header={"backgroundColor": "#181B15", "color": "#EAEEE6", "fontWeight": "600"},
    )

    mapping = html.Div([
        html.Div(className="section-label", children="Step 2 · Map your columns"),
        html.Div(className="drill-note", children=[
            f"Loaded {len(df):,} rows × {len(file_cols)} columns. ",
            "We auto-matched your columns to the HAIP standard below. ",
            "Fix any wrong matches, or set a column to “skip” if it isn’t "
            "present. Required columns are marked in red.",
        ]),
        html.Div(rows, style={"marginTop": "8px"}),
        html.Button("Validate dataset", id="validate-btn", n_clicks=0, style={
            "marginTop": "16px", "padding": "10px 24px", "background": TEAL,
            "color": "white", "border": "none", "borderRadius": "8px",
            "cursor": "pointer", "fontSize": "15px",
        }),
        html.Div(className="section-label", children="Preview",
                 style={"marginTop": "20px"}),
        preview,
    ])

    return df.to_json(date_format="iso", orient="split"), \
        f"Loaded: {filename}", mapping


@app.callback(
    Output("validation-section", "children", allow_duplicate=True),
    Output("mapping-store", "data"),
    Input("validate-btn", "n_clicks"),
    State("data-store", "data"),
    State({"type": "map", "std": dash.ALL}, "value"),
    State({"type": "map", "std": dash.ALL}, "id"),
    prevent_initial_call=True,
)
def validate(n_clicks, data_json, map_values, map_ids):
    """Check the mapped dataset against the standard and report issues."""
    if not n_clicks or data_json is None:
        return None, None
    df = pd.read_json(io.StringIO(data_json), orient="split")

    mapping = {}
    for val, idd in zip(map_values, map_ids):
        std = idd["std"]
        mapping[std] = None if val == "__none__" else val

    missing_required = [c for c in REQUIRED_COLUMNS if not mapping.get(c)]
    missing_recommended = [c for c in RECOMMENDED_COLUMNS if not mapping.get(c)]

    quality_rows = []
    for std, file_col in mapping.items():
        if not file_col:
            continue
        q = _quality(df[file_col])
        issue = q["missing_pct"] > 0
        quality_rows.append({
            "Standard": std, "Your column": file_col,
            "Type": q["dtype"], "Missing %": q["missing_pct"],
            "Unique": q["unique"],
            "Status": "⚠ has missing values" if issue else "✓ ok",
        })

    blocks = [html.Div(className="section-label", children="Step 3 · Validation report")]

    if missing_required:
        blocks.append(html.Div(className="drill-panel", style={
            "borderLeft": f"4px solid {RED}"}, children=[
            html.Strong("Required columns missing", style={"color": RED}),
            html.Div("The pipeline cannot fully run without these. You can "
                     "provide an alternative column or proceed with explicit "
                     "approval (some features will be unavailable).",
                     className="drill-note"),
            html.Ul([html.Li(f"{c} — {REQUIRED_COLUMNS[c]}")
                     for c in missing_required]),
        ]))
    else:
        blocks.append(html.Div(className="drill-note", style={"color": GREEN},
                      children="✓ All required columns are present."))

    if missing_recommended:
        blocks.append(html.Div(className="drill-panel", style={
            "borderLeft": f"4px solid {AMBER}"}, children=[
            html.Strong("Recommended columns missing", style={"color": AMBER}),
            html.Div("Not required — you can proceed, but these improve "
                     "analysis and prediction quality.", className="drill-note"),
            html.Ul([html.Li(f"{c} — {RECOMMENDED_COLUMNS[c]}")
                     for c in missing_recommended]),
        ]))

    if quality_rows:
        blocks.append(html.Div(className="section-label",
                      children="Mapped column quality"))
        blocks.append(dash_table.DataTable(
            data=quality_rows,
            columns=[{"name": c, "id": c} for c in
                     ["Standard", "Your column", "Type", "Missing %",
                      "Unique", "Status"]],
            style_table={"overflowX": "auto"},
            style_cell={"fontSize": "13px", "padding": "8px",
                        "fontFamily": "inherit", "textAlign": "left"},
            style_header={"backgroundColor": "#181B15", "color": "#EAEEE6", "fontWeight": "600"},
            style_data_conditional=[{
                "if": {"filter_query": "{Status} contains 'missing'"},
                "backgroundColor": "#2A2410"}],
        ))

    if missing_required:
        proceed = html.Div(className="drill-panel", children=[
            html.Strong("Approval required to proceed"),
            html.Div("Your dataset is missing required columns. Confirm you "
                     "want to proceed anyway — affected features will be "
                     "disabled.", className="drill-note"),
            html.Button("Proceed with approval", id="approve-btn", n_clicks=0,
                        style={"padding": "10px 24px", "background": AMBER,
                               "color": "white", "border": "none",
                               "borderRadius": "8px", "cursor": "pointer"}),
        ])
    else:
        proceed = html.Div(className="drill-panel", style={
            "borderLeft": f"4px solid {GREEN}"}, children=[
            html.Strong("Ready for the pipeline", style={"color": GREEN}),
            html.Div("All required columns are mapped. This standardized "
                     "dataset can now enter the ETL pipeline. You can also "
                     "build an analytics dashboard from this data below.",
                     className="drill-note"),
            html.Button("Build dashboard", id="build-dash-btn", n_clicks=0,
                        style={"marginTop": "12px", "padding": "10px 24px",
                               "background": TEAL, "color": "white",
                               "border": "none", "borderRadius": "8px",
                               "cursor": "pointer", "fontSize": "15px"}),
        ])
    blocks.append(proceed)
    # The dashboard renders here when "Build dashboard" is clicked.
    blocks.append(html.Div(id="uploaded-dashboard"))
    return html.Div(blocks), mapping

def _default_features(cols):
    """Auto-select feature columns, skipping id/name-like columns that would
    leak identity into the model (honest engineering: identifiers are not
    predictive features)."""
    skip_tokens = ("id", "name", "uuid", "identifier")
    feats = []
    for c in cols:
        lc = c.lower()
        if any(tok in lc for tok in skip_tokens):
            continue
        feats.append(c)
    # Leave the target out later in the callback; here just drop identifiers.
    return feats


def _eda_figures(df, mapping):
    """Build EDA figures: missingness, a distribution, correlation heatmap."""
    figs = []

    miss = (df.isna().mean() * 100).round(1).sort_values(ascending=False)
    miss = miss[miss > 0].head(15)
    if len(miss):
        f = go.Figure(go.Bar(
            x=list(miss.values[::-1]), y=list(miss.index)[::-1],
            orientation="h", marker=dict(color=AMBER),
            text=[f"{v}%" for v in miss.values[::-1]], textposition="auto"))
        apply_theme(f, height=max(260, 24 * len(miss) + 80))
        figs.append(("Missing values by column (%)", f))

    numeric_cols = [c for c in df.columns
                    if pd.api.types.is_numeric_dtype(df[c])][:12]

    if numeric_cols:
        col = numeric_cols[0]
        f = go.Figure(go.Histogram(
            x=df[col].dropna(), marker=dict(color=TEAL), nbinsx=30))
        apply_theme(f, height=300)
        figs.append((f"Distribution · {col}", f))

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().round(2)
        f = go.Figure(go.Heatmap(
            z=corr.values, x=list(corr.columns), y=list(corr.index),
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=corr.values, texttemplate="%{text}", textfont=dict(size=10)))
        apply_theme(f, height=max(320, 26 * len(numeric_cols) + 120))
        figs.append(("Correlation (numeric columns)", f))

    return figs



def _analysis_and_prediction_ui(df):
    """Build the EDA charts + the target/feature pickers + train button."""
    cols = list(df.columns)
    eda = _eda_figures(df, None)
    eda_children = [html.Div(className="section-label",
                             children="Step 5 · Analysis (EDA)")]
    for title, fig in eda:
        eda_children.append(html.Div(className="drill-panel", children=[
            html.Strong(title, style={"color": INK}),
            dcc.Graph(figure=fig),
        ]))

    predict_ui = html.Div([
        html.Div(className="section-label", children="Step 6 · Prediction"),
        html.Div(className="drill-note", children=[
            "Train a model on this dataset. Pick what to predict (target) and "
            "the input columns (features). The engine auto-detects "
            "classification vs regression and reports honest metrics.",
        ]),
        html.Label("Target column", className="control-label"),
        dcc.Dropdown(id="up-target",
                     options=[{"label": c, "value": c} for c in cols],
                     placeholder="Column to predict", className="dropdown"),
        html.Label("Feature columns", className="control-label",
                   style={"marginTop": "12px"}),
        dcc.Dropdown(id="up-features",
                     options=[{"label": c, "value": c} for c in cols],
                     value=_default_features(cols),
                     multi=True, placeholder="Input columns",
                     className="dropdown"),
        html.Button("Train model", id="up-train-btn", n_clicks=0, style={
            "marginTop": "16px", "padding": "10px 24px", "background": TEAL,
            "color": "white", "border": "none", "borderRadius": "8px",
            "cursor": "pointer", "fontSize": "15px"}),
        html.Div(id="up-results"),
    ])

    return html.Div(eda_children + [predict_ui])


@app.callback(
    Output("validation-section", "children", allow_duplicate=True),
    Output("analysis-section", "children"),
    Input("approve-btn", "n_clicks"),
    State("data-store", "data"),
    prevent_initial_call=True,
)
def confirm_proceed(n_clicks, data_json):
    """On approval: confirm acceptance AND reveal analysis + prediction."""
    if not n_clicks or data_json is None:
        return no_update, no_update
    confirmation = html.Div([
        html.Div(className="section-label", children="Step 4 · Accepted"),
        html.Div(className="drill-panel", style={
            "borderLeft": f"4px solid {GREEN}"}, children=[
            html.Strong("Dataset accepted for the pipeline",
                        style={"color": GREEN}),
            html.Div("Proceeding to analysis and prediction below.",
                     className="drill-note"),
        ]),
    ])
    df = pd.read_json(io.StringIO(data_json), orient="split")
    return confirmation, _analysis_and_prediction_ui(df)

def _forecast_from_upload(df, periods=6):
    """Monthly forecast from the uploaded df. Finds a date col + numeric
    volume col, aggregates monthly, fits Holt (trend). Returns fig or None."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    date_col = next((c for c in df.columns
                     if pd.to_datetime(df[c], errors="coerce").notna().mean() > 0.8), None)
    if not date_col:
        return None
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    monthly = d.set_index(date_col).resample("MS").size()
    monthly = monthly[monthly.index.notna()]
    if len(monthly) < 4:
        return None
    try:
        model = ExponentialSmoothing(monthly, trend="add", seasonal=None).fit()
        fc = model.forecast(periods).round().astype(int)
    except Exception:
        return None
    fig = go.Figure()
    fig.add_scatter(x=monthly.index, y=monthly.values, mode="lines+markers",
                    name="Actual", line=dict(color=INK, width=2))
    bridge_x = [monthly.index[-1]] + list(fc.index)
    bridge_y = [monthly.values[-1]] + list(fc.values)
    fig.add_scatter(x=bridge_x, y=bridge_y, mode="lines+markers",
                    name="Forecast", line=dict(color=TEAL, width=3, dash="dash"))
    apply_theme(fig, title=f"{periods}-month encounter forecast (Holt trend)",
                height=360)
    return fig

@app.callback(
    Output("up-results", "children"),
    Output("report-section", "style"),
    Input("up-train-btn", "n_clicks"),
    State("data-store", "data"),
    State("up-target", "value"),
    State("up-features", "value"),
    prevent_initial_call=True,
)
def train_uploaded(n_clicks, data_json, target, features):
    """Train the AutoML engine on the uploaded data and show results."""
    from dashboard.pipeline import train_on_upload
    if not n_clicks or data_json is None:
        return None, {"marginTop": "16px", "display": "none"}
    if not target:
        return html.Div("Pick a target column.", className="drill-note"), {"marginTop": "16px", "display": "none"}
    if not features:
        return html.Div("Pick at least one feature column.",
                        className="drill-note"), {"marginTop": "16px", "display": "none"}

    df = pd.read_json(io.StringIO(data_json), orient="split")
    result = train_on_upload(df, target, features)
    if result.error:
        return html.Div(f"Could not train: {result.error}",
                        className="drill-note", style={"color": RED}), {"marginTop": "16px", "display": "none"}

    cards = html.Div(className="kpi-row", children=[
        html.Div(className="kpi-card", style={"borderLeft": f"5px solid {TEAL}"},
                 children=[html.Div(str(v), className="kpi-value",
                                    style={"color": TEAL}),
                           html.Div(k.upper(), className="kpi-label")])
        for k, v in result.metrics.items()])

    children = [
        html.Div(className="drill-note", children=[
            html.Strong(f"{result.problem_type.title()} · "),
            f"target '{result.target}', trained on {result.n_train}, "
            f"tested on {result.n_test}. Random Forest."]),
        cards]

    if result.confusion:
        labels = result.class_labels
        cm_fig = go.Figure(go.Heatmap(
            z=result.confusion,
            x=[f"pred {l}" for l in labels],
            y=[f"actual {l}" for l in labels],
            colorscale="Blues", text=result.confusion,
            texttemplate="%{text}", showscale=False))
        apply_theme(cm_fig, title="Confusion matrix", height=320)
        children.append(html.Div(className="drill-panel",
                                 children=[dcc.Graph(figure=cm_fig)]))

    names = [n for n, _ in result.importances][::-1]
    vals = [v for _, v in result.importances][::-1]
    imp_fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h", marker=dict(color=TEAL),
        text=[f"{v:.3f}" for v in vals], textposition="auto"))
    apply_theme(imp_fig, title="Feature importance", height=max(300, 26 * len(names) + 80))
    children.append(html.Div(className="drill-panel",
                             children=[dcc.Graph(figure=imp_fig)]))

    # --- Forecast from uploaded data (trend-based, short-series safe) -------
    fc_fig = _forecast_from_upload(df)
    if fc_fig is not None:
        children.append(html.Div(className="drill-panel",
                                 children=[dcc.Graph(figure=fc_fig)]))
    else:
        children.append(html.Div(
            "Forecast needs a date column + a numeric volume/count column "
            "with enough history.", className="drill-note",
            style={"color": AMBER}))

    children.append(html.Div(className="drill-panel", style={
        "borderLeft": f"4px solid {TEAL}"}, children=[
        html.Strong("Recommendation", style={"color": TEAL}),
        html.Div(
            "Model trained on the uploaded dataset. Metrics and feature "
            "importance above are decision-support aids for human review, "
            "not diagnostic outputs. Use the forecast for capacity planning.",
            className="drill-note"),
    ]))
    return html.Div(children), {"marginTop": "16px", "display": "block"}


# ---------------------------------------------------------------------------
# Uploaded-data dashboard (built on demand from the validated upload)
# ---------------------------------------------------------------------------
# Every chart is GRACEFUL: it renders only if the columns it needs were
# mapped. A hospital file missing 'condition' simply skips that chart instead
# of erroring. All figures reuse the clinical palette so the panel looks
# identical to the Synthea analytics tab — but is honestly driven by the
# uploaded data, not Synthea.

def _up_kpis(df, m) -> html.Div:
    """KPI cards from the standard columns that mapped."""
    cards = []
    cards.append(("Encounters", f"{len(df):,}", "activity"))
    if m.get("patient_id"):
        cards.append(("Unique patients", f"{df[m['patient_id']].nunique():,}",
                      "patients"))
    if m.get("total_cost"):
        cost = pd.to_numeric(df[m["total_cost"]], errors="coerce")
        cards.append(("Total cost ($)", f"{cost.sum():,.0f}", "wallet"))
        cards.append(("Avg cost / enc ($)", f"{cost.mean():,.0f}", "trending"))
    return html.Div(className="kpi-row", children=[
        html.Div(className="kpi-card", style={"borderTop": f"3px solid {TEAL}"},
                 children=[
                     html.Div(className="kpi-head", children=[
                         html.Div(k, className="kpi-label"),
                         _icon(icon, TEAL),
                     ]),
                     html.Div(v, className="kpi-value", style={"color": TEAL}),
                 ])
        for k, v, icon in cards])


def _up_fig_class(df, m):
    """Encounters by class — needs encounter_class."""
    col = m.get("encounter_class")
    if not col:
        return None
    counts = df[col].value_counts()
    fig = go.Figure(go.Bar(
        x=list(counts.index), y=list(counts.values), marker=dict(color=TEAL),
        text=list(counts.values), textposition="auto"))
    apply_theme(fig, title="Encounters by class", height=340)
    return ("Encounters by class", fig)


def _up_fig_volume(df, m):
    """Monthly encounter volume — needs encounter_date."""
    col = m.get("encounter_date")
    if not col:
        return None
    dates = pd.to_datetime(df[col], errors="coerce").dropna()
    if dates.empty:
        return None
    monthly = dates.dt.to_period("M").value_counts().sort_index()
    x = [str(p) for p in monthly.index]
    fig = go.Figure(go.Scatter(
        x=x, y=list(monthly.values), mode="lines+markers",
        line=dict(color=TEAL, width=2), marker=dict(size=4)))
    apply_theme(fig, title="Monthly encounter volume", height=340)
    return ("Monthly encounter volume", fig)


def _up_fig_cost_hist(df, m):
    """Cost distribution — needs total_cost."""
    col = m.get("total_cost")
    if not col:
        return None
    cost = pd.to_numeric(df[col], errors="coerce").dropna()
    if cost.empty:
        return None
    fig = go.Figure(go.Histogram(x=cost, marker=dict(color=TEAL), nbinsx=30))
    apply_theme(fig, title="Cost distribution", height=340)
    return ("Cost distribution", fig)


def _up_fig_top_conditions(df, m):
    """Top conditions by total cost — needs condition + total_cost."""
    cond_col, cost_col = m.get("condition"), m.get("total_cost")
    if not cond_col or not cost_col:
        return None
    tmp = df[[cond_col, cost_col]].copy()
    tmp[cost_col] = pd.to_numeric(tmp[cost_col], errors="coerce")
    grouped = (tmp.groupby(cond_col)[cost_col].sum()
               .sort_values(ascending=False).head(10).sort_values())
    if grouped.empty:
        return None
    fig = go.Figure(go.Bar(
        x=list(grouped.values), y=list(grouped.index), orientation="h",
        marker=dict(color=TEAL),
        text=[f"${v/1e3:,.0f}K" for v in grouped.values], textposition="auto"))
    apply_theme(fig, title="Top conditions by total cost", height=max(340, 26 * len(grouped) + 80))
    return ("Top conditions by total cost", fig)


def _up_fig_demographics(df, m):
    """Age distribution + gender split — needs age and/or gender."""
    age_col, gen_col = m.get("age"), m.get("gender")
    figs = []
    if age_col:
        age = pd.to_numeric(df[age_col], errors="coerce").dropna()
        if not age.empty:
            f = go.Figure(go.Histogram(x=age, marker=dict(color=AMBER), nbinsx=20))
            apply_theme(f, title="Age distribution", height=320)
            figs.append(("Age distribution", f))
    if gen_col:
        counts = df[gen_col].value_counts()
        f = go.Figure(go.Pie(
            labels=list(counts.index), values=list(counts.values), hole=0.55,
            marker=dict(colors=[TEAL, AMBER, GREEN, MUTED][:len(counts)],
                        line=dict(color="white", width=2)),
            sort=False, textinfo="percent", textposition="inside",
            insidetextorientation="horizontal", texttemplate="%{percent}",
            hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>"))
        apply_theme(f, title="Gender split", height=320)
        f.update_layout(legend=dict(orientation="h", y=-0.05, x=0.5,
                                    xanchor="center"),
                        margin=dict(t=60, b=60, l=20, r=20))
        figs.append(("Gender split", f))
    return figs


@app.callback(
    Output("uploaded-dashboard", "children"),
    Input("build-dash-btn", "n_clicks"),
    State("data-store", "data"),
    State("mapping-store", "data"),
    prevent_initial_call=True,
)
def build_uploaded_dashboard(n_clicks, data_json, mapping):
    """Render an analytics dashboard from the validated uploaded data.

    Reads the stored dataframe + the standard->file column mapping, then
    builds each chart only if its required columns mapped. Same clinical
    styling as the Synthea tab; honestly driven by the upload.
    """
    if not n_clicks or data_json is None or not mapping:
        return None
    df = pd.read_json(io.StringIO(data_json), orient="split")

    children = [
        html.Div(className="section-label",
                 children="Uploaded data · analytics dashboard"),
        html.Div(className="drill-note", children=[
            "Built from your uploaded file using the columns you mapped. "
            "Charts that need a column you didn't map are skipped — nothing "
            "is invented. This is your data, standardized to the HAIP schema.",
        ]),
        _up_kpis(df, mapping),
    ]

    # Paired charts in rows; singles in panels. Collect, then lay out.
    paired = [_up_fig_class(df, mapping), _up_fig_volume(df, mapping)]
    paired = [p for p in paired if p]
    if paired:
        children.append(html.Div(className="chart-row", children=[
            html.Div(dcc.Graph(figure=f), className="chart-half")
            for _, f in paired]))

    cost_hist = _up_fig_cost_hist(df, mapping)
    if cost_hist:
        children.append(html.Div(className="drill-panel",
                                 children=[dcc.Graph(figure=cost_hist[1])]))

    top_cond = _up_fig_top_conditions(df, mapping)
    if top_cond:
        children.append(html.Div(className="drill-panel",
                                 children=[dcc.Graph(figure=top_cond[1])]))

    demo = _up_fig_demographics(df, mapping)
    if demo:
        children.append(html.Div(className="chart-row", children=[
            html.Div(dcc.Graph(figure=f), className="chart-half")
            for _, f in demo]))

    if len(children) <= 3:  # only label + note + KPIs, no charts built
        children.append(html.Div(className="drill-note", children=[
            "No optional columns were mapped, so only headline figures are "
            "shown. Map columns like encounter_class, encounter_date, "
            "condition, age or gender for richer charts."]))

    return html.Div(children)


# ===========================================================================
# SHELL + TAB ROUTING
# ===========================================================================
app.layout = html.Div(className="page", children=[
    html.Div(className="page-head", children=[
        html.Div(className="page-eyebrow", children=[
            html.Span(className="page-dot"),
            "Hospital data · analytics",
        ]),
        html.H1("Upload, analyze, and forecast admissions", className="title"),
        html.P("Map your columns, explore KPIs, then train a model and "
               "forecast the months ahead.", className="subtitle"),
    ]),

    dcc.Tabs(id="main-tabs", value="tab-intake", style={"display": "none"}, children=[
        dcc.Tab(label="Data Intake & Prediction", value="tab-intake"),
    ]),

    # Filled lazily by the routing callback on tab select.
    dcc.Loading(html.Div(id="tab-content", style={"marginTop": "16px"}),
                type="default"),

    html.Div(className="footer",
             children="HAIP · unified analytics and prediction platform."),
])


@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value"),
)
def render_tab(tab: str) -> html.Div:
    """Build the selected tab's layout on demand (lazy loading)."""
    return build_intake_layout()


@app.callback(
    Output("report-download", "data"),
    Input("dl-report-btn", "n_clicks"),
    State("data-store", "data"),
    State("mapping-store", "data"),
    prevent_initial_call=True,
)
def download_report(n_clicks, data_json, mapping):
    """Build a PDF summary of the uploaded dataset and stream it back.

    Reads the stored dataframe + column mapping, computes the same headline
    KPIs shown on the dashboard, and hands them to build_hospital_report,
    which returns PDF bytes for dcc.send_bytes. Nothing is invented — only
    KPIs whose columns were mapped are included.
    """
    if not n_clicks or data_json is None or not mapping:
        raise dash.exceptions.PreventUpdate

    df = pd.read_json(io.StringIO(data_json), orient="split")

    # Headline KPIs — same logic as _up_kpis, but as a plain dict.
    kpis = {"Encounters": f"{len(df):,}"}
    if mapping.get("patient_id"):
        kpis["Unique patients"] = f"{df[mapping['patient_id']].nunique():,}"
    if mapping.get("total_cost"):
        cost = pd.to_numeric(df[mapping["total_cost"]], errors="coerce")
        kpis["Total cost ($)"] = f"{cost.sum():,.0f}"
        kpis["Avg cost / enc ($)"] = f"{cost.mean():,.0f}"

    if mapping.get("encounter_class"):
        kpis["Encounter types"] = f"{df[mapping['encounter_class']].nunique():,}"

    chart_builders = [
        _up_fig_class, _up_fig_volume, _up_fig_cost_hist,
        _up_fig_top_conditions, _up_fig_demographics,
    ]
    charts = []
    for fn in chart_builders:
        try:
            res = fn(df, mapping)
        except Exception:
            res = None
        if res is None:
            continue
        if isinstance(res, tuple):
            charts.append(res[1])
        elif isinstance(res, list):
            charts.extend(r[1] if isinstance(r, tuple) else r for r in res)
        else:
            charts.append(res)

    recommendation = (
        "This report summarises the uploaded dataset only. Figures are "
        "descriptive analytics on the hospital's own data — no predictive "
        "model is trained here. HAIP is a proof-of-concept decision-support "
        "tool on synthetic / de-identified data; outputs are triage aids for "
        "human review, not diagnostic decisions."
    )

    pdf_bytes = build_hospital_report(
        kpis=kpis,
        forecast_df=None,
        hospital_name="Uploaded Dataset",
        charts=charts,
        recommendation=recommendation,
    )
    return dcc.send_bytes(pdf_bytes, "haip_report.pdf")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8060, debug=True)
