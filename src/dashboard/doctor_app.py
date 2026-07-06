"""HAIP provider performance dashboard (standalone Dash app, redesigned).

Self-contained inline styling — no dependency on external CSS. Flow:

    upload (friendly drop box)
      -> preview + build
      -> department card grid (click a card)
      -> department detail (table + per-doctor O/E charts)  -> back

Honest, risk-adjusted analytics carried from the rest of HAIP:
  * O/E ratios (observed / expected), not raw outcome rankings.
  * Low-volume providers suppressed (shown, not scored).
  * Null departments -> "Unassigned"; repeated typos folded to canonical.
  * Every exclusion disclosed. Synthetic data, illustrative of methodology.

Scales to any number of departments (card grid + search, not a fixed list).

Run from src/ with:  python -m dashboard.doctor_app  ->  http://127.0.0.1:8051
"""

import base64
import io
import logging

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, dash_table, ALL, ctx

logger = logging.getLogger(__name__)

# --- palette (inline, self-contained) --------------------------------------
INK = "#0B1F33"
TEAL = "#1A7A8C"
TEAL_DK = "#0F5563"
AMBER = "#E8A33D"
RED = "#C2453D"
GREEN = "#3C7A5A"
MUTED = "#6B7A88"
GRID = "#EDF1F4"
BG = "#F4F7F9"
CARD = "#FFFFFF"
FADE = "#C7D2DA"

MIN_VOLUME = 30
DOCTOR_COL = "doctor_name"
DEPT_COL = "department"
VOL_COL = "encounters"
OUTCOMES = {
    "mortality": ("mortality_count", "expected_mortality"),
    "readmission": ("readmission_count", "expected_readmission"),
}

# Icon per department (Tabler-ish unicode fallback via emoji-free glyphs).
# Kept simple: a coloured dot label, no external icon dependency.

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "HAIP — Provider Performance"


# ===========================================================================
# DATA LOGIC (verified)
# ===========================================================================
def _parse_upload(contents, filename):
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


def _clean_departments(df):
    notes = []
    out = df.copy()
    if DEPT_COL not in out.columns:
        return out, ["No department column found — grouping unavailable."]
    raw = out[DEPT_COL]
    n_null = int(raw.isna().sum())
    cleaned = raw.astype("object").where(raw.notna(), None)
    cleaned = [None if v is None else str(v).strip().title() for v in cleaned]
    import difflib
    from collections import Counter
    counts = Counter(c for c in cleaned if c)
    names = sorted(counts, key=lambda k: -counts[k])
    fold_map = {}
    for v in names:
        bigger = [c for c in names if c != v and counts[c] >= counts[v] * 5]
        match = difflib.get_close_matches(v, bigger, n=1, cutoff=0.85)
        if match:
            fold_map[v] = match[0]
    folded, fold_count = [], 0
    for v in cleaned:
        if v is None:
            folded.append("Unassigned")
        elif v in fold_map:
            folded.append(fold_map[v])
            fold_count += 1
        else:
            folded.append(v)
    out[DEPT_COL] = folded
    if n_null:
        notes.append(f"{n_null} provider(s) had no department — grouped as "
                     f"\u201cUnassigned\u201d.")
    if fold_count:
        notes.append(f"{fold_count} likely department typo(s) normalised.")
    return out, notes


def _compute_oe(df):
    out = df.copy()
    notes = []
    vol = pd.to_numeric(out.get(VOL_COL), errors="coerce")
    for name, (obs_col, exp_col) in OUTCOMES.items():
        oe_col = f"oe_{name}"
        if obs_col not in out.columns or exp_col not in out.columns:
            out[oe_col] = pd.NA
            notes.append(f"Missing columns for {name} O/E — skipped.")
            continue
        obs = pd.to_numeric(out[obs_col], errors="coerce")
        exp = pd.to_numeric(out[exp_col], errors="coerce")
        oe = obs / exp.where(exp > 0)
        suppressed = (vol < MIN_VOLUME) | obs.isna() | exp.isna() | (exp <= 0)
        out[oe_col] = oe.mask(suppressed).round(2)
        n_sup = int(suppressed.sum())
        if n_sup:
            notes.append(f"{name}: {n_sup} provider(s) suppressed "
                         f"(under {MIN_VOLUME} encounters or missing outcome).")
    return out, notes


def _dept_summary(df):
    """Per-department aggregate for the card grid."""
    rows = []
    for dept, g in df.groupby(DEPT_COL):
        enc = int(pd.to_numeric(g[VOL_COL], errors="coerce").sum())
        oe = pd.NA
        obs = pd.to_numeric(g["mortality_count"], errors="coerce").sum()
        exp = pd.to_numeric(g["expected_mortality"], errors="coerce").sum()
        if exp and exp > 0:
            oe = round(obs / exp, 2)
        rows.append({"department": dept, "doctors": len(g),
                     "encounters": enc, "oe": oe})
    return sorted(rows, key=lambda r: -r["encounters"])

# Threshold for flagging a whole DEPARTMENT as having a real cluster of
# concerning providers. Stricter than the individual-doctor review bucket
# (>1.1) so a department flag is a real signal, not normal variation.
DEPT_FLAG_OE = 1.25


def _hospital_insights(df):
    """Plain-language, rule-based takeaways across all departments."""
    out = []
    dept_oe = {}
    for d, g in df.groupby(DEPT_COL):
        if d == "Unassigned":
            continue
        obs = pd.to_numeric(g["mortality_count"], errors="coerce").sum()
        exp = pd.to_numeric(g["expected_mortality"], errors="coerce").sum()
        if exp and exp > 0:
            dept_oe[d] = round(obs / exp, 2)
    if dept_oe:
        ranked = sorted(dept_oe.items(), key=lambda x: x[1])
        best, best_v = ranked[0]
        worst, worst_v = ranked[-1]
        out.append(("good", f"{best} is the strongest-performing department "
                            f"(O/E {best_v:.2f} \u2014 fewer adverse outcomes "
                            f"than expected)."))
        if worst_v > 1.0:
            out.append(("watch", f"{worst} runs above expected "
                                 f"(O/E {worst_v:.2f}) \u2014 worth attention."))
            
    # Exclude the strongest department so it can't be both "best" and
    # "flagged" — a great average with one or two outliers isn't a concern.
    best_dept = ranked[0][0] if dept_oe else None
    clustered = []
    for d, g in df.groupby(DEPT_COL):
        if d == best_dept:
            continue
        s = g["oe_mortality"].dropna()
        if len(s) >= 5 and (s > DEPT_FLAG_OE).mean() > 0.20:
            clustered.append(d)

    if clustered:
        names = ", ".join(clustered[:4])
        out.append(("watch", f"{len(clustered)} department(s) have a cluster "
                             f"of providers well above expected: {names}."))
    else:
        out.append(("good", "No department shows a concerning cluster of "
                           "high-O/E providers \u2014 variation looks normal."))
    obs = pd.to_numeric(df["mortality_count"], errors="coerce").sum()
    exp = pd.to_numeric(df["expected_mortality"], errors="coerce").sum()
    if exp and exp > 0:
        oe = obs / exp
        word = "better than" if oe < 1.0 else "about as" if oe <= 1.02 \
            else "worse than"
        out.append(("info", f"Hospital-wide mortality O/E is {oe:.2f} "
                           f"\u2014 {word} expected overall."))
    n_ns = int(df["oe_mortality"].isna().sum())
    if n_ns:
        out.append(("info", f"{n_ns} provider(s) couldn't be scored (too few "
                           f"cases or missing outcome) \u2014 shown, not scored."))
    return out


def _dept_insights(sub, hospital_oe):
    """Plain-language takeaways for a single department."""
    out = []
    s = sub["oe_mortality"].dropna()
    if len(s):
        flagged = int((s > 1.1).sum())
        pct = round(100 * flagged / len(s))
        out.append((("watch" if pct > 20 else "info"),
                    f"{flagged} of {len(s)} scored providers are above "
                    f"expected ({pct}%)."))
        obs = pd.to_numeric(sub["mortality_count"], errors="coerce").sum()
        exp = pd.to_numeric(sub["expected_mortality"], errors="coerce").sum()
        if exp and exp > 0:
            d_oe = obs / exp
            cmp = "better than" if d_oe < hospital_oe else "worse than" \
                if d_oe > hospital_oe else "in line with"
            out.append(("info", f"Department O/E {d_oe:.2f} \u2014 {cmp} the "
                               f"hospital average ({hospital_oe:.2f})."))
        best = sub.loc[s.idxmin()]
        worst = sub.loc[s.idxmax()]
        out.append(("good", f"Top performer: {best[DOCTOR_COL]} "
                           f"(O/E {best['oe_mortality']:.2f})."))
        if worst["oe_mortality"] > 1.1:
            out.append(("watch", f"Most worth reviewing: {worst[DOCTOR_COL]} "
                                f"(O/E {worst['oe_mortality']:.2f})."))
    else:
        out.append(("info", "No providers in this department could be scored "
                           "(too few cases or missing outcomes)."))
    return out

def _dept_oe_rows(df):
    """(department, O/E) pairs, best-first, excluding Unassigned."""
    rows = []
    for d, g in df.groupby(DEPT_COL):
        if d == "Unassigned":
            continue
        obs = pd.to_numeric(g["mortality_count"], errors="coerce").sum()
        exp = pd.to_numeric(g["expected_mortality"], errors="coerce").sum()
        if exp and exp > 0:
            rows.append((d, round(obs / exp, 2)))
    rows.sort(key=lambda x: x[1])
    return rows


def _fig_dept_dotplot(df):
    """Lollipop/dot plot of department O/E — compact, spread is the story."""
    rows = _dept_oe_rows(df)
    if not rows:
        return _style(go.Figure(), "Department O/E", 300)
    rows = rows[::-1]  # best at top
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colours = [GREEN if v < 0.98 else (AMBER if v > 1.02 else MUTED)
               for v in vals]
    fig = go.Figure()
    for n, v, c in zip(names, vals, colours):
        fig.add_trace(go.Scatter(
            x=[1.0, v], y=[n, n], mode="lines",
            line=dict(color=GRID, width=1), showlegend=False,
            hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=vals, y=names, mode="markers",
        marker=dict(color=colours, size=11, line=dict(color="white", width=1)),
        showlegend=False,
        hovertemplate="%{y}: O/E %{x:.2f}<extra></extra>"))
    fig.add_vline(x=1.0, line_dash="dash", line_color=MUTED)
    fig = _style(fig, "Department O/E — mortality (1.0 = as expected)",
                 max(340, 22 * len(names) + 90))
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    return fig


def _insight_tiles(df):
    """Four visual summary tiles: strongest, watch, hospital O/E, not scored."""
    rows = _dept_oe_rows(df)
    tiles = []

    def _tile(label, value, sub, bg, fg):
        return html.Div(style={"background": bg, "borderRadius": "12px",
                               "padding": "16px"}, children=[
            html.Div(label, style={"fontSize": "11px", "color": fg,
                                    "fontWeight": "600",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "0.04em"}),
            html.Div(value, style={"fontSize": "16px", "fontWeight": "700",
                                   "color": INK, "marginTop": "4px"}),
            html.Div(sub, style={"fontSize": "12px", "color": fg,
                                 "marginTop": "2px"})])

    if rows:
        best, best_v = rows[0]
        worst, worst_v = rows[-1]
        tiles.append(_tile("Strongest", best,
                           f"O/E {best_v:.2f} · better than expected",
                           "#E6F4EE", "#3C7A5A"))
        if worst_v > 1.0:
            tiles.append(_tile("Worth attention", worst,
                               f"O/E {worst_v:.2f} · above expected",
                               "#FCF0DD", "#B5791F"))
    obs = pd.to_numeric(df["mortality_count"], errors="coerce").sum()
    exp = pd.to_numeric(df["expected_mortality"], errors="coerce").sum()
    if exp and exp > 0:
        oe = obs / exp
        word = "better than" if oe < 1.0 else "about as" if oe <= 1.02 \
            else "worse than"
        tiles.append(_tile("Hospital-wide", f"O/E {oe:.2f}",
                           f"{word} expected overall", "#EEF3F4", "#4A6572"))
    n_ns = int(df["oe_mortality"].isna().sum())
    tiles.append(_tile("Not scored", f"{n_ns} providers",
                       "too few cases · shown, not scored",
                       "#EEF3F4", "#4A6572"))
    return html.Div(style={"display": "grid",
                           "gridTemplateColumns":
                               "repeat(auto-fit, minmax(180px, 1fr))",
                           "gap": "12px", "marginBottom": "16px"},
                    children=tiles)

def _insight_panel(insights, title="Key insights"):
    """Render a list of (kind, text) insights as a styled panel."""
    tone = {"good": ("#E6F4EE", "#3C7A5A", "\u2713"),
            "watch": ("#FCF0DD", "#B5791F", "!"),
            "info": ("#EEF3F4", "#4A6572", "i")}
    items = []
    for kind, text in insights:
        bg, fg, mark = tone.get(kind, tone["info"])
        items.append(html.Div(style={
            "display": "flex", "gap": "10px", "alignItems": "flex-start",
            "padding": "8px 0"}, children=[
            html.Span(mark, style={"flexShrink": "0", "width": "20px",
                                   "height": "20px", "borderRadius": "50%",
                                   "background": bg, "color": fg,
                                   "fontSize": "12px", "fontWeight": "700",
                                   "display": "flex", "alignItems": "center",
                                   "justifyContent": "center"}),
            html.Span(text, style={"fontSize": "13px", "color": INK,
                                   "lineHeight": "1.5"})]))
    return html.Div(style={"background": CARD, "border": f"1px solid {GRID}",
                           "borderRadius": "12px", "padding": "16px 20px",
                           "marginBottom": "16px"}, children=[
        html.Div(title, style={"fontSize": "14px", "fontWeight": "600",
                               "color": INK, "marginBottom": "6px"}),
        html.Div(children=items)])

# Threshold for flagging a whole DEPARTMENT as having a real cluster of
# concerning providers. Deliberately stricter than the individual-doctor
# review bucket (>1.1): it takes a cluster above 1.25 to flag a department,
# so the insight is a real signal rather than normal statistical variation.
def _style(fig, title, height):
    fig.update_layout(
        title=dict(text=title, x=0.02, y=0.97, font=dict(size=15, color=INK)),
        font=dict(color=INK, size=13), paper_bgcolor=CARD, plot_bgcolor=CARD,
        margin=dict(t=54, b=40, l=20, r=20), height=height, showlegend=False)
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False,
                     tickfont=dict(color=MUTED, size=11))
    fig.update_yaxes(showgrid=False, zeroline=False,
                     tickfont=dict(color=MUTED, size=11))
    return fig


def _fig_doctor_oe(df, outcome, dept):
    oe_col = f"oe_{outcome}"
    sub = df[df[DEPT_COL] == dept].dropna(subset=[oe_col]).sort_values(oe_col)
    if sub.empty:
        return _style(go.Figure(), f"No scorable providers — {outcome}", 240)
    colours = [GREEN if v <= 1.0 else RED for v in sub[oe_col]]
    fig = go.Figure(go.Bar(
        x=list(sub[oe_col]), y=list(sub[DOCTOR_COL]), orientation="h",
        marker=dict(color=colours),
        text=[f"{v:.2f}" for v in sub[oe_col]], textposition="auto"))
    fig.add_vline(x=1.0, line_dash="dash", line_color=MUTED)
    return _style(fig, f"Provider O/E — {outcome}", max(260, 26 * len(sub) + 80))


def _fig_volume_vs_oe(df, dept):
    """Scatter: each scored provider by volume (x) vs O/E mortality (y).

    The honest picture — reliable scores cluster near 1.0 at higher volume;
    extreme O/E values sit at low volume (which is why we suppress them).
    """
    sub = df[df[DEPT_COL] == dept].dropna(subset=["oe_mortality"]).copy()
    vol = pd.to_numeric(sub[VOL_COL], errors="coerce")
    oe = sub["oe_mortality"]
    colours = [GREEN if v < 0.9 else (AMBER if v > 1.1 else MUTED) for v in oe]
    fig = go.Figure(go.Scatter(
        x=list(vol), y=list(oe), mode="markers",
        marker=dict(color=colours, size=10, opacity=0.8,
                    line=dict(color="white", width=1)),
        text=list(sub[DOCTOR_COL]),
        hovertemplate="%{text}<br>%{x} encounters<br>O/E %{y:.2f}<extra></extra>"))
    fig.add_hline(y=1.0, line_dash="dash", line_color=MUTED)
    fig.update_xaxes(title="encounters (volume)")
    fig.update_yaxes(title="O/E mortality")
    return _style(fig, "Volume vs performance", 340)


def _fig_bucket_bar(better, expected, review, not_scored):
    """Simple bar: provider counts per plain-language bucket."""
    labels = ["Better", "As expected", "Needs review", "Not scored"]
    vals = [better, expected, review, not_scored]
    colours = [GREEN, MUTED, AMBER, FADE]
    fig = go.Figure(go.Bar(
        x=labels, y=vals, marker=dict(color=colours),
        text=vals, textposition="auto"))
    return _style(fig, "Providers by performance group", 340)


def _fig_doctor_vs_dept(doc_oe, dept_oe, outcome):
    """Gauge for one doctor's O/E — green/grey/amber zones, needle at value.

    Intuitive for non-technical viewers: the needle lands in a zone that reads
    at a glance. (Name kept for call-site compatibility; now a gauge.)
    """
    val = 1.0 if pd.isna(doc_oe) else round(float(doc_oe), 2)
    if val < 0.9:
        col = GREEN
    elif val > 1.1:
        col = "#B5791F"
    else:
        col = MUTED
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val,
        number={"font": {"size": 30, "color": col}, "valueformat": ".2f"},
        gauge={
            "axis": {"range": [0.5, 1.5], "tickvals": [0.9, 1.1],
                     "tickfont": {"size": 10, "color": MUTED}},
            "bar": {"color": "rgba(0,0,0,0)"},
            "borderwidth": 0,
            "steps": [
                {"range": [0.5, 0.9], "color": "#3C7A5A"},
                {"range": [0.9, 1.1], "color": "#C7D2DA"},
                {"range": [1.1, 1.5], "color": "#E8A33D"}],
            "threshold": {"line": {"color": INK, "width": 4},
                          "thickness": 0.85, "value": val}}))
    fig.update_layout(
        title=dict(text=f"O/E {outcome}", x=0.5, y=0.96, xanchor="center",
                   font=dict(size=14, color=INK)),
        paper_bgcolor=CARD, height=260, margin=dict(t=50, b=10, l=30, r=30))
    return fig


# ===========================================================================
# STYLE HELPERS (inline)
# ===========================================================================
def _page(*children):
    return html.Div(style={"maxWidth": "1100px", "margin": "0 auto",
                           "padding": "0 20px"}, children=list(children))


def _header():
    return html.Div(style={
        "display": "flex", "alignItems": "center", "gap": "14px",
        "padding": "20px 0 16px", "borderBottom": f"1px solid {GRID}",
        "marginBottom": "24px"}, children=[
        html.Div("HAIP", style={
            "background": TEAL, "color": "white", "fontWeight": "700",
            "fontSize": "18px", "padding": "8px 14px", "borderRadius": "10px",
            "letterSpacing": "0.05em"}),
        html.Div([
            html.Div("Provider Performance", style={
                "fontSize": "20px", "fontWeight": "600", "color": INK}),
            html.Div("Risk-adjusted, honest, scales to any hospital",
                     style={"fontSize": "13px", "color": MUTED}),
        ]),
    ])


def _stat(label, value):
    return html.Div(style={
        "flex": "1", "background": CARD, "border": f"1px solid {GRID}",
        "borderRadius": "12px", "padding": "16px"}, children=[
        html.Div(label, style={"fontSize": "12px", "color": MUTED}),
        html.Div(value, style={"fontSize": "22px", "fontWeight": "600",
                               "color": INK, "marginTop": "2px"}),
    ])


def _oe_badge(oe):
    if oe is None or pd.isna(oe):
        return html.Span("n/a", style={
            "fontSize": "11px", "background": GRID, "color": MUTED,
            "padding": "3px 9px", "borderRadius": "10px"})
    good = oe <= 1.0
    return html.Span(f"O/E {oe:.2f}", style={
        "fontSize": "11px",
        "background": "#E6F4EE" if good else "#FBEAEA",
        "color": GREEN if good else RED,
        "padding": "3px 9px", "borderRadius": "10px", "fontWeight": "500"})


# ===========================================================================
# LAYOUT — single page, sections swap via callbacks
# ===========================================================================
app.layout = html.Div(style={
    "background": BG, "minHeight": "100vh", "paddingBottom": "48px",
    "fontFamily": '-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif'},
    children=[
    _page(
        _header(),
        # Step 1: upload box
        html.Div(id="upload-wrap", children=[
            html.Div(style={"textAlign": "center", "marginBottom": "10px"},
                     children=[
                html.Div("Upload your hospital's doctor dataset", style={
                    "fontSize": "16px", "fontWeight": "600", "color": INK}),
                html.Div("CSV, JSON, or Excel — with doctor, department, "
                         "encounters and outcome columns",
                         style={"fontSize": "13px", "color": MUTED,
                                "marginTop": "2px"}),
            ]),
            dcc.Upload(id="doc-upload", multiple=False, children=html.Div(
                style={"border": f"2px dashed {TEAL}", "borderRadius": "14px",
                       "padding": "40px 20px", "textAlign": "center",
                       "background": "#EAF3F5", "cursor": "pointer"},
                children=[
                    html.Div("\u2601", style={"fontSize": "34px",
                                              "color": TEAL}),
                    html.Div("Drop file here or click to browse", style={
                        "fontSize": "15px", "color": TEAL_DK,
                        "fontWeight": "500", "marginTop": "6px"}),
                ])),
            html.Div(id="doc-upload-status", style={
                "fontSize": "13px", "color": MUTED, "marginTop": "10px",
                "textAlign": "center"}),
        ]),
        html.Div(id="doc-preview-section"),
        html.Div(id="doc-grid-section"),
        html.Div(id="doc-detail-section"),
        html.Div(id="doc-person-section"),
        dcc.Store(id="doc-store"),
        dcc.Store(id="doc-scored-store"),
        dcc.Store(id="doc-current-dept"),
    ),
])


# ===========================================================================
# CALLBACKS
# ===========================================================================
@app.callback(
    Output("doc-store", "data"),
    Output("doc-upload-status", "children"),
    Output("doc-preview-section", "children"),
    Input("doc-upload", "contents"),
    State("doc-upload", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename):
    if contents is None:
        return None, "", None
    try:
        df = _parse_upload(contents, filename)
    except Exception as exc:
        return None, f"Could not read file: {exc}", None
    if df.empty or DOCTOR_COL not in df.columns or DEPT_COL not in df.columns:
        return (None, f"File needs '{DOCTOR_COL}' and '{DEPT_COL}' columns.",
                None)
    preview = dash_table.DataTable(
        data=df.head(6).to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        style_table={"overflowX": "auto"},
        style_cell={"fontSize": "12px", "padding": "6px",
                    "fontFamily": "inherit", "textAlign": "left"},
        style_header={"backgroundColor": "#EEF3F4", "fontWeight": "600"},
        page_size=6)
    section = html.Div(style={
        "background": CARD, "border": f"1px solid {GRID}",
        "borderRadius": "12px", "padding": "20px", "marginTop": "16px"},
        children=[
        html.Div(f"Preview — {len(df):,} providers, {len(df.columns)} columns",
                 style={"fontSize": "14px", "fontWeight": "600", "color": INK,
                        "marginBottom": "10px"}),
        preview,
        html.Button("Build dashboard", id="doc-build-btn", n_clicks=0, style={
            "marginTop": "16px", "padding": "11px 26px", "background": TEAL,
            "color": "white", "border": "none", "borderRadius": "9px",
            "cursor": "pointer", "fontSize": "15px", "fontWeight": "500"}),
    ])
    return (df.to_json(date_format="iso", orient="split"),
            f"Loaded {filename}", section)


@app.callback(
    Output("doc-grid-section", "children"),
    Output("doc-scored-store", "data"),
    Output("doc-preview-section", "style"),
    Input("doc-build-btn", "n_clicks"),
    State("doc-store", "data"),
    prevent_initial_call=True,
)
def build_grid(n_clicks, data_json):
    if not n_clicks or not data_json:
        return None, None, dash.no_update
    df = pd.read_json(io.StringIO(data_json), orient="split")
    df, dept_notes = _clean_departments(df)
    df, oe_notes = _compute_oe(df)
    notes = dept_notes + oe_notes
    summ = _dept_summary(df)
    total_docs = len(df)
    total_enc = int(pd.to_numeric(df[VOL_COL], errors="coerce").sum())

    stats = html.Div(style={"display": "flex", "gap": "12px",
                            "marginBottom": "16px"}, children=[
        _stat("Departments", f"{len(summ)}"),
        _stat("Doctors", f"{total_docs:,}"),
        _stat("Encounters", f"{total_enc:,}"),
    ])

    disclosure = html.Div(style={
        "background": "#FEF7EC", "border": f"1px solid {AMBER}",
        "borderLeft": f"4px solid {AMBER}", "borderRadius": "10px",
        "padding": "12px 16px", "marginBottom": "16px",
        "fontSize": "13px", "color": INK}, children=[
        html.Strong("Data handling (disclosed): "),
        "; ".join(notes) if notes else "no missing data or suppression.",
    ])

    cards = []
    for r in summ:
        cards.append(html.Div(
            id={"type": "dept-card", "dept": r["department"]},
            n_clicks=0,
            style={"background": CARD, "border": f"1px solid {GRID}",
                   "borderRadius": "12px", "padding": "16px",
                   "cursor": "pointer"},
            children=[
                html.Div(style={"display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center"}, children=[
                    html.Span("\u25CF", style={"color": TEAL,
                                               "fontSize": "14px"}),
                    _oe_badge(r["oe"]),
                ]),
                html.Div(r["department"], style={
                    "fontSize": "15px", "fontWeight": "600", "color": INK,
                    "marginTop": "8px"}),
                html.Div(f"{r['doctors']} doctors · {r['encounters']:,} enc",
                         style={"fontSize": "12px", "color": MUTED,
                                "marginTop": "2px"}),
            ]))

    grid = html.Div(style={
        "display": "grid",
        "gridTemplateColumns": "repeat(auto-fill, minmax(200px, 1fr))",
        "gap": "12px"}, children=cards)

    section = html.Div([
        stats,
        _insight_tiles(df),
        html.Div(style={"background": CARD, "border": f"1px solid {GRID}",
                        "borderRadius": "12px", "padding": "12px",
                        "marginBottom": "16px"},
                 children=[dcc.Graph(figure=_fig_dept_dotplot(df))]),
        disclosure,
        html.Div("Departments — click a card to open", style={
            "fontSize": "13px", "color": MUTED, "marginBottom": "10px"}),
        grid,
    ])
    return section, df.to_json(orient="split"), {"display": "none"}


@app.callback(
    Output("doc-detail-section", "children"),
    Output("doc-grid-section", "style"),
    Output("doc-current-dept", "data"),
    Input({"type": "dept-card", "dept": ALL}, "n_clicks"),
    State("doc-scored-store", "data"),
    prevent_initial_call=True,
)
def open_dept(clicks, scored_json):
    if not scored_json or not any(clicks):
        return dash.no_update, dash.no_update, dash.no_update
    trig = ctx.triggered_id
    if not trig or "dept" not in trig:
        return dash.no_update, dash.no_update, dash.no_update
    dept = trig["dept"]
    df = pd.read_json(io.StringIO(scored_json), orient="split")
    sub = df[df[DEPT_COL] == dept].copy()
    total_enc = int(pd.to_numeric(sub[VOL_COL], errors="coerce").sum())
    # hospital-wide mortality O/E, for the department comparison insight
    _ho = pd.to_numeric(df["mortality_count"], errors="coerce").sum()
    _he = pd.to_numeric(df["expected_mortality"], errors="coerce").sum()
    hospital_oe = round(_ho / _he, 2) if _he and _he > 0 else 1.0

    oe = sub["oe_mortality"]
    better = sub[oe < 0.9]
    expected = sub[(oe >= 0.9) & (oe <= 1.1)]
    review = sub[oe > 1.1]
    not_scored = sub[oe.isna()]

    def _bucket(count, label, bg, fg):
        return html.Div(style={"background": bg, "borderRadius": "12px",
                               "padding": "16px"}, children=[
            html.Div(str(count), style={"fontSize": "26px",
                                        "fontWeight": "700", "color": fg}),
            html.Div(label, style={"fontSize": "13px", "color": fg,
                                   "fontWeight": "500"})])

    buckets = html.Div(style={"display": "grid",
                              "gridTemplateColumns": "1fr 1fr 1fr 1fr",
                              "gap": "12px", "marginBottom": "16px"}, children=[
        _bucket(len(better), "Better than expected", "#E6F4EE", "#3C7A5A"),
        _bucket(len(expected), "As expected", "#F0F2F4", "#6B7A88"),
        _bucket(len(review), "Needs review", "#FCF0DD", "#B5791F"),
        _bucket(len(not_scored), "Not scored", "#F0F2F4", "#6B7A88")])

    # Legend — plain-language explanation, collapsible via <details>.
    legend = html.Details(style={
        "background": "#F7FAFB", "border": f"1px solid {GRID}",
        "borderRadius": "10px", "padding": "10px 14px",
        "marginBottom": "16px", "fontSize": "13px", "color": INK}, children=[
        html.Summary("What do these mean?", style={
            "cursor": "pointer", "color": TEAL, "fontWeight": "500"}),
        html.Div(style={"marginTop": "8px", "lineHeight": "1.7",
                        "color": MUTED}, children=[
            html.Div([html.Strong("Encounters: "),
                      "how many patient cases the doctor handled (their "
                      "volume). Under 30 cases isn't scored — too few to "
                      "judge fairly."]),
            html.Div([html.Strong("O/E: "),
                      "Observed \u00f7 Expected. Compares actual outcomes to "
                      "what you'd expect given how sick the patients were."]),
            html.Div([html.Strong("Below 1.0: "),
                      "better than expected \u00b7 ",
                      html.Strong("Around 1.0: "), "as expected \u00b7 ",
                      html.Strong("Above 1.1: "), "flagged for review."]),
            html.Div("Because it's risk-adjusted, a doctor with very sick "
                     "patients can still score below 1.0 — the fair measure."),
        ])])

    def _chips_expandable(frame, bg, fg, kind, limit=12):
        names = frame[DOCTOR_COL].tolist()
        shown = names[:limit]

        def _chip(n):
            return html.Span(n, id={"type": "doc-chip", "name": n},
                             n_clicks=0, style={
                                 "fontSize": "12px", "background": bg,
                                 "color": fg, "padding": "4px 10px",
                                 "borderRadius": "12px", "cursor": "pointer"})

        chips = [_chip(n) for n in shown]
        extra = len(names) - len(shown)
        children = list(chips)
        if extra > 0:
            hidden = [_chip(n) for n in names[limit:]]
            children.append(html.Details(style={"display": "inline"},
                children=[
                    html.Summary(f"+{extra} more", style={
                        "cursor": "pointer", "color": MUTED,
                        "fontSize": "12px", "display": "inline"}),
                    html.Div(style={"display": "flex", "flexWrap": "wrap",
                                    "gap": "6px", "marginTop": "6px"},
                             children=hidden)]))
        return html.Div(style={"display": "flex", "flexWrap": "wrap",
                               "gap": "6px", "marginBottom": "14px"},
                        children=children)

    name_lists = html.Div([
        html.Div("Needs review — worth a closer look", style={
            "fontSize": "13px", "fontWeight": "500", "color": INK,
            "marginBottom": "6px"}),
        _chips_expandable(review, "#FCF0DD", "#B5791F", "review") if len(review)
        else html.Div("None flagged.", style={"fontSize": "12px",
                                               "color": MUTED,
                                               "marginBottom": "14px"}),
        html.Div("Top performers", style={"fontSize": "13px",
                                          "fontWeight": "500", "color": INK,
                                          "marginBottom": "6px"}),
        _chips_expandable(better, "#E6F4EE", "#3C7A5A", "better") if len(better)
        else html.Div("None.", style={"fontSize": "12px", "color": MUTED})])

    charts = html.Div(style={"display": "grid",
                             "gridTemplateColumns": "1fr 1fr", "gap": "12px",
                             "marginBottom": "16px"}, children=[
        html.Div(style={"background": CARD, "border": f"1px solid {GRID}",
                        "borderRadius": "12px", "padding": "12px"},
                 children=[dcc.Graph(figure=_fig_volume_vs_oe(df, dept))]),
        html.Div(style={"background": CARD, "border": f"1px solid {GRID}",
                        "borderRadius": "12px", "padding": "12px"},
                 children=[dcc.Graph(figure=_fig_bucket_bar(
                     len(better), len(expected), len(review),
                     len(not_scored)))])])

    show = [DOCTOR_COL, VOL_COL, "oe_mortality", "oe_readmission"]
    show = [c for c in show if c in sub.columns]
    tbl = sub[show].rename(columns={
        DOCTOR_COL: "Doctor", VOL_COL: "Encounters",
        "oe_mortality": "O/E mortality", "oe_readmission": "O/E readmission"})
    for c in ["O/E mortality", "O/E readmission"]:
        if c in tbl.columns:
            tbl[c] = tbl[c].apply(lambda v: "\u2014" if pd.isna(v)
                                  else f"{v:.2f}")
    table = dash_table.DataTable(
        data=tbl.to_dict("records"),
        columns=[{"name": c, "id": c} for c in tbl.columns],
        sort_action="native", filter_action="native", page_size=15,
        style_table={"overflowX": "auto"},
        style_cell={"fontSize": "13px", "padding": "9px",
                    "fontFamily": "inherit", "textAlign": "left"},
        style_header={"backgroundColor": TEAL_DK, "color": "#E1F5EE",
                      "fontWeight": "600"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#F7FAFB"}])

    detail = html.Div([
        html.Div(id="doc-back-btn", n_clicks=0, style={
            "display": "inline-block", "cursor": "pointer", "color": TEAL,
            "fontSize": "13px", "marginBottom": "14px"},
            children="\u2190 back to departments"),
        html.Div(f"{dept} · {len(sub)} providers · {total_enc:,} encounters",
                 style={"fontSize": "17px", "fontWeight": "600", "color": INK,
                        "marginBottom": "4px"}),
        html.Div("Grouped by performance vs expected for their patient mix. "
                 "Green = better, amber = worth a review.",
                 style={"fontSize": "12px", "color": MUTED,
                        "marginBottom": "16px"}),
        buckets,
        _insight_panel(_dept_insights(sub, hospital_oe),
                       f"{dept} — key insights"),
        legend,
        charts,
        html.Div(style={"background": CARD, "border": f"1px solid {GRID}",
                        "borderRadius": "12px", "padding": "20px",
                        "marginBottom": "16px"}, children=[name_lists]),
        html.Div(style={"background": CARD, "border": f"1px solid {GRID}",
                        "borderRadius": "12px", "padding": "20px"}, children=[
            html.Div("All providers — exact figures", style={
                "fontSize": "14px", "fontWeight": "600", "color": INK,
                "marginBottom": "4px"}),
            html.Div("Sortable, filterable. \u201c\u2014\u201d = not scored "
                     "(too few cases or missing outcome).",
                     style={"fontSize": "12px", "color": MUTED,
                            "margin": "0 0 12px"}),
            table]),
    ])
    return detail, {"display": "none"}, dept


@app.callback(
    Output("doc-detail-section", "children", allow_duplicate=True),
    Output("doc-grid-section", "style", allow_duplicate=True),
    Input("doc-back-btn", "n_clicks"),
    prevent_initial_call=True,
)
def back_to_grid(n):
    if not n:
        return dash.no_update, dash.no_update
    return None, {"display": "block"}


@app.callback(
    Output("doc-person-section", "children"),
    Output("doc-detail-section", "style"),
    Input({"type": "doc-chip", "name": ALL}, "n_clicks"),
    State("doc-scored-store", "data"),
    State("doc-current-dept", "data"),
    prevent_initial_call=True,
)
def open_doctor(clicks, scored_json, dept):
    """Chip click -> render that doctor's individual page."""
    if not scored_json or not any(clicks):
        return dash.no_update, dash.no_update
    trig = ctx.triggered_id
    if not trig or "name" not in trig:
        return dash.no_update, dash.no_update
    name = trig["name"]
    df = pd.read_json(io.StringIO(scored_json), orient="split")
    row = df[df[DOCTOR_COL] == name]
    if row.empty:
        return dash.no_update, dash.no_update
    r = row.iloc[0]

    # Department averages for comparison (volume-weighted O/E).
    dsub = df[df[DEPT_COL] == r[DEPT_COL]]
    dept_avg = {}
    for oc, (obs_c, exp_c) in OUTCOMES.items():
        obs = pd.to_numeric(dsub[obs_c], errors="coerce").sum()
        exp = pd.to_numeric(dsub[exp_c], errors="coerce").sum()
        dept_avg[oc] = (obs / exp) if exp and exp > 0 else float("nan")
    dept_vol_avg = int(pd.to_numeric(dsub[VOL_COL], errors="coerce").mean())

    oe_m = r.get("oe_mortality")
    oe_r = r.get("oe_readmission")

    def _status(v):
        if pd.isna(v):
            return ("Not scored", "#F0F2F4", "#6B7A88")
        if v < 0.9:
            return ("Better than expected", "#E6F4EE", "#3C7A5A")
        if v > 1.1:
            return ("Needs review", "#FCF0DD", "#B5791F")
        return ("As expected", "#F0F2F4", "#6B7A88")

    s_label, s_bg, s_fg = _status(oe_m)
    initials = "".join([w[0] for w in name.replace("Dr. ", "").split()[:2]])
    enc = int(pd.to_numeric(pd.Series([r[VOL_COL]]), errors="coerce").iloc[0])

    def _mini(label, value, colour=INK, accent=TEAL, context=""):
        return html.Div(style={"background": CARD, "border": f"1px solid {GRID}",
                               "borderLeft": f"3px solid {accent}",
                               "borderRadius": "10px", "padding": "14px"},
                        children=[
            html.Div(label, style={"fontSize": "11px", "color": MUTED}),
            html.Div(value, style={"fontSize": "22px", "fontWeight": "700",
                                   "color": colour, "marginTop": "2px"}),
            html.Div(context, style={"fontSize": "11px", "color": colour
                                     if colour != INK else MUTED,
                                     "marginTop": "2px"}) if context
            else html.Div()])

    def _fmt(v):
        return "\u2014" if pd.isna(v) else f"{v:.2f}"

    m_col = s_fg if not pd.isna(oe_m) else MUTED
    r_col = (GREEN if (not pd.isna(oe_r) and oe_r < 0.9)
             else AMBER if (not pd.isna(oe_r) and oe_r > 1.1) else MUTED)

    # plain-language summary
    bits = []
    if enc >= dept_vol_avg:
        bits.append(f"handles above-average volume ({enc} vs "
                    f"{dept_vol_avg} dept avg)")
    else:
        bits.append(f"handles below-average volume ({enc} vs "
                    f"{dept_vol_avg} dept avg)")
    if not pd.isna(oe_m):
        if oe_m > 1.1:
            bits.append("mortality O/E is above expected — flagged for review")
        elif oe_m < 0.9:
            bits.append("mortality O/E is better than expected")
        else:
            bits.append("mortality O/E is about as expected")
    summary = ("This provider " + "; ".join(bits) +
               ". Risk-adjusted; a flag means look closer, not a verdict.")

    page = html.Div([
        html.Div(id="doc-person-back", n_clicks=0, style={
            "display": "inline-block", "cursor": "pointer", "color": TEAL,
            "fontSize": "13px", "marginBottom": "14px"},
            children=f"\u2190 back to {r[DEPT_COL]}"),
        html.Div(style={"background": CARD, "border": f"1px solid {GRID}",
                        "borderRadius": "12px", "padding": "1.25rem"},
                 children=[
            html.Div(style={"display": "flex", "alignItems": "center",
                            "gap": "14px", "marginBottom": "18px"}, children=[
                html.Div(initials, style={
                    "width": "52px", "height": "52px", "borderRadius": "50%",
                    "background": "#EAF3F5", "color": TEAL,
                    "display": "flex", "alignItems": "center",
                    "justifyContent": "center", "fontSize": "20px",
                    "fontWeight": "600"}),
                html.Div(style={"flex": "1"}, children=[
                    html.Div(name, style={"fontSize": "18px",
                                          "fontWeight": "600", "color": INK}),
                    html.Div(f"{r[DEPT_COL]} · {r.get('specialty','')} · "
                             f"{r.get('tenure_years','?')} yrs tenure",
                             style={"fontSize": "13px", "color": MUTED})]),
                html.Span(s_label, style={"fontSize": "13px", "background": s_bg,
                                          "color": s_fg, "padding": "6px 14px",
                                          "borderRadius": "14px",
                                          "fontWeight": "500"})]),
            html.Div(style={"display": "grid",
                            "gridTemplateColumns": "repeat(4,1fr)",
                            "gap": "10px", "marginBottom": "18px"}, children=[
                _mini("Encounters", f"{enc:,}", INK, TEAL,
                      ("above dept avg" if enc >= dept_vol_avg
                       else "below dept avg") + f" ({dept_vol_avg})"),
                _mini("Patients", f"{int(r.get('patients_seen', 0)):,}",
                      INK, TEAL, "unique seen"),
                _mini("O/E mortality", _fmt(oe_m), m_col,
                      (m_col if not pd.isna(oe_m) else FADE),
                      ("above expected · review" if (not pd.isna(oe_m)
                       and oe_m > 1.1) else "better than expected"
                       if (not pd.isna(oe_m) and oe_m < 0.9)
                       else "as expected" if not pd.isna(oe_m)
                       else "not scored")),
                _mini("O/E readmission", _fmt(oe_r), r_col,
                      (r_col if not pd.isna(oe_r) else FADE),
                      ("above expected · review" if (not pd.isna(oe_r)
                       and oe_r > 1.1) else "better than expected"
                       if (not pd.isna(oe_r) and oe_r < 0.9)
                       else "as expected" if not pd.isna(oe_r)
                       else "not scored"))]),
            html.Div("How this doctor compares to the "
                     f"{r[DEPT_COL]} average", style={
                         "fontSize": "13px", "fontWeight": "500",
                         "color": INK, "marginBottom": "8px"}),
            html.Div(style={"display": "grid",
                            "gridTemplateColumns": "1fr 1fr", "gap": "12px",
                            "marginBottom": "12px"}, children=[
                dcc.Graph(figure=_fig_doctor_vs_dept(
                    oe_m, dept_avg["mortality"], "mortality")),
                dcc.Graph(figure=_fig_doctor_vs_dept(
                    oe_r, dept_avg["readmission"], "readmission"))]),
            html.Div(summary, style={"fontSize": "12px",
                                     "color": MUTED, "lineHeight": "1.6",
                                     "paddingTop": "10px",
                                     "borderTop": f"0.5px solid {GRID}"}),
        ]),
    ])
    return page, {"display": "none"}


@app.callback(
    Output("doc-person-section", "children", allow_duplicate=True),
    Output("doc-detail-section", "style", allow_duplicate=True),
    Input("doc-person-back", "n_clicks"),
    prevent_initial_call=True,
)
def back_to_dept(n):
    if not n:
        return dash.no_update, dash.no_update
    return None, {"display": "block"}


if __name__ == "__main__":
    app.run(debug=True, port=8051)