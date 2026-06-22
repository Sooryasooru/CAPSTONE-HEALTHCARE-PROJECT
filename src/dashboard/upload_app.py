"""Upload app: a standalone Dash app for hospital data upload + AutoML.

Lets any hospital drop a file (CSV / JSON / Excel), pick a target column and
feature columns, and train a fresh model on THEIR data — then see honest
metrics, a confusion matrix (classification) or error scores (regression),
and feature importances.

Runs independently of the main dashboard (which stays untouched):
    python -m dashboard.upload_app
Then open http://127.0.0.1:8051

Powered by the proven engine in pipeline.py — this file is only the UI.
"""

import base64
import io
import logging

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, dash_table

from dashboard.pipeline import train_on_upload

logger = logging.getLogger(__name__)

# Palette (matches the main dashboard)
INK = "#0B1F33"
TEAL = "#1A7A8C"
AMBER = "#E8A33D"
RED = "#C2453D"
GREEN = "#3C7A5A"
MUTED = "#6B7A88"

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "HAIP — Data Upload & AutoML"


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


def _confusion_figure(cm, labels) -> go.Figure:
    """Render a confusion-matrix heatmap."""
    fig = go.Figure(go.Heatmap(
        z=cm, x=[f"pred {l}" for l in labels], y=[f"actual {l}" for l in labels],
        colorscale="Blues", showscale=False,
        text=cm, texttemplate="%{text}", textfont=dict(size=14),
    ))
    fig.update_layout(
        title=dict(text="Confusion matrix", x=0.02),
        height=320, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(color=INK), margin=dict(t=50, b=40, l=60, r=20),
    )
    return fig


def _importance_figure(importances) -> go.Figure:
    """Render a feature-importance horizontal bar chart."""
    names = [n for n, _ in importances][::-1]
    vals = [v for _, v in importances][::-1]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h", marker=dict(color=TEAL),
        text=[f"{v:.3f}" for v in vals], textposition="auto",
    ))
    fig.update_layout(
        title=dict(text="Feature importance", x=0.02),
        height=max(320, 26 * len(names) + 80),
        paper_bgcolor="white", plot_bgcolor="white", font=dict(color=INK),
        margin=dict(t=50, b=40, l=20, r=20),
        xaxis=dict(title="importance", showgrid=True, gridcolor="#EEF2F5"),
    )
    return fig


app.layout = html.Div(className="page", children=[
    html.Div(className="header", children=[
        html.Div("HAIP", className="logo"),
        html.Div([
            html.H1("Data Upload & AutoML", className="title"),
            html.P("Upload any hospital dataset, pick a target, train a model",
                   className="subtitle"),
        ]),
    ]),

    # --- Step 1: upload ---
    html.Div(className="section-label", children="Step 1 · Upload a dataset"),
    dcc.Upload(
        id="upload-data",
        children=html.Div([
            "Drag and drop a file here, or ",
            html.A("browse", style={"color": TEAL, "textDecoration": "underline"}),
            html.Br(),
            html.Span("CSV, JSON, or Excel", style={"color": MUTED, "fontSize": "13px"}),
        ]),
        style={
            "width": "100%", "height": "120px", "lineHeight": "24px",
            "borderWidth": "2px", "borderStyle": "dashed",
            "borderColor": TEAL, "borderRadius": "12px",
            "textAlign": "center", "padding": "30px 0", "background": "#F7FAFB",
        },
        multiple=False,
    ),
    html.Div(id="upload-status", className="drill-note"),

    # --- Step 2: column mapping (hidden until a file is loaded) ---
    html.Div(id="mapping-section"),

    # --- Step 3: results ---
    html.Div(id="results-section"),

    # Stores the uploaded data as JSON between callbacks
    dcc.Store(id="data-store"),

    html.Div(className="footer",
             children="HAIP · upload pipeline · trains a fresh model on your data"),
])


@app.callback(
    Output("data-store", "data"),
    Output("upload-status", "children"),
    Output("mapping-section", "children"),
    Output("results-section", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename):
    """Parse the uploaded file and build the column-mapping UI."""
    if contents is None:
        return None, "", None, None
    try:
        df = _parse_upload(contents, filename)
    except Exception as exc:
        return None, f"Could not read file: {exc}", None, None

    if df.empty or len(df.columns) < 2:
        return None, "File needs at least 2 columns and some rows.", None, None

    preview = dash_table.DataTable(
        data=df.head(5).to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        style_table={"overflowX": "auto"},
        style_cell={"fontSize": "12px", "padding": "6px",
                    "fontFamily": "inherit", "textAlign": "left"},
        style_header={"backgroundColor": "#F1F5F6", "fontWeight": "600"},
    )

    cols = list(df.columns)
    mapping = html.Div([
        html.Div(className="section-label",
                 children="Step 2 · Map your columns"),
        html.Div(className="drill-note", children=[
            f"Loaded {len(df):,} rows × {len(cols)} columns. ",
            "Pick the column to predict (target) and the columns to use as "
            "inputs (features).",
        ]),
        html.Label("Target column (what to predict)",
                   className="control-label"),
        dcc.Dropdown(id="target-select",
                     options=[{"label": c, "value": c} for c in cols],
                     placeholder="Select the column to predict",
                     className="dropdown"),
        html.Label("Feature columns (inputs)", className="control-label",
                   style={"marginTop": "12px"}),
        dcc.Dropdown(id="feature-select",
                     options=[{"label": c, "value": c} for c in cols],
                     multi=True, placeholder="Select one or more input columns",
                     className="dropdown"),
        html.Button("Train model", id="train-btn", n_clicks=0,
                    style={
                        "marginTop": "16px", "padding": "10px 24px",
                        "background": TEAL, "color": "white", "border": "none",
                        "borderRadius": "8px", "cursor": "pointer",
                        "fontSize": "15px",
                    }),
        html.Div(className="section-label", children="Preview",
                 style={"marginTop": "20px"}),
        preview,
    ])

    return df.to_json(date_format="iso", orient="split"), \
        f"Loaded: {filename}", mapping, None


@app.callback(
    Output("results-section", "children", allow_duplicate=True),
    Input("train-btn", "n_clicks"),
    State("data-store", "data"),
    State("target-select", "value"),
    State("feature-select", "value"),
    prevent_initial_call=True,
)
def train_model(n_clicks, data_json, target, features):
    """Run the pipeline on the chosen target/features and show results."""
    if not n_clicks or data_json is None:
        return None
    if not target:
        return html.Div("Please select a target column.", className="drill-note")
    if not features:
        return html.Div("Please select at least one feature column.",
                        className="drill-note")

    df = pd.read_json(io.StringIO(data_json), orient="split")
    result = train_on_upload(df, target, features)

    if result.error:
        return html.Div(className="drill-panel", children=[
            html.Div(className="section-label", children="Result"),
            html.Div(f"Could not train: {result.error}", className="drill-note",
                     style={"color": RED}),
        ])

    # Metric cards
    metric_cards = html.Div(className="kpi-row", children=[
        html.Div(className="kpi-card", style={"borderLeft": f"5px solid {TEAL}"},
                 children=[
                     html.Div(str(v), className="kpi-value",
                              style={"color": TEAL}),
                     html.Div(k.upper(), className="kpi-label"),
                 ])
        for k, v in result.metrics.items()
    ])

    children = [
        html.Div(className="section-label", children="Step 3 · Results"),
        html.Div(className="drill-note", children=[
            html.Strong(f"{result.problem_type.title()} · "),
            f"target '{result.target}' · trained on {result.n_train} rows, "
            f"tested on {result.n_test}. ",
            "Random Forest. Metrics are on the held-out test set.",
        ]),
        metric_cards,
    ]

    if result.confusion:
        children.append(html.Div(className="drill-panel", children=[
            dcc.Graph(figure=_confusion_figure(result.confusion,
                                               result.class_labels)),
        ]))

    children.append(html.Div(className="drill-panel", children=[
        dcc.Graph(figure=_importance_figure(result.importances)),
    ]))

    return html.Div(children=children)


if __name__ == "__main__":
    app.run(debug=True, port=8051)