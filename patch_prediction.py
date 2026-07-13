#!/usr/bin/env python3
"""
HAIP — two targeted fixes for the upload/prediction flow:
  Fix 2: auto-select feature columns (excluding target + id/name columns to
         avoid data leakage) so the user doesn't have to pick manually.
  Fix 3: move the "Download PDF report" button out of validation-section
         (which confirm_proceed overwrites) into the static intake layout,
         so it never disappears after training.

Run from project root:  python3 patch_prediction.py
"""
import sys, ast

P = "src/dashboard/merged_app.py"
try:
    s = open(P).read()
except FileNotFoundError:
    print(f"ERROR: {P} not found. Run from project root."); sys.exit(1)

orig = s
changes = []

# --- Fix 3a: remove the download-button block from _finalize (validation-section)
dl_block = '''    blocks.append(html.Div(id="uploaded-dashboard"))
    # PDF report: download a summary of the uploaded dataset.
    blocks.append(html.Div(style={"marginTop": "16px"}, children=[
        html.Button("Download PDF report", id="dl-report-btn", n_clicks=0,
                    style={"padding": "10px 24px", "background": AMBER,
                           "color": "white", "border": "none",
                           "borderRadius": "8px", "cursor": "pointer",
                           "fontSize": "15px"}),
        dcc.Download(id="report-download"),
    ]))
    return html.Div(blocks), mapping'''

dl_block_new = '''    blocks.append(html.Div(id="uploaded-dashboard"))
    return html.Div(blocks), mapping'''

if dl_block in s:
    s = s.replace(dl_block, dl_block_new)
    changes.append("Fix 3a: removed download button from validation-section")
else:
    print("WARN: Fix 3a anchor not found (download block in _finalize)")

# --- Fix 3b: add a static download button after analysis-section in build_intake_layout
intake_anchor = '''        html.Div(id="validation-section"),
        html.Div(id="analysis-section"),'''

intake_new = '''        html.Div(id="validation-section"),
        html.Div(id="analysis-section"),
        html.Div(id="report-section", style={"marginTop": "16px"}, children=[
            html.Button("Download PDF report", id="dl-report-btn", n_clicks=0,
                        style={"padding": "10px 24px", "background": AMBER,
                               "color": "white", "border": "none",
                               "borderRadius": "8px", "cursor": "pointer",
                               "fontSize": "15px"}),
            dcc.Download(id="report-download"),
        ]),'''

if intake_anchor in s:
    s = s.replace(intake_anchor, intake_new)
    changes.append("Fix 3b: added static download button after analysis-section")
else:
    print("WARN: Fix 3b anchor not found (validation/analysis section in build_intake_layout)")

# --- Fix 2: auto-select feature columns (exclude target + id/name-like cols)
feat_anchor = '''        dcc.Dropdown(id="up-features",
                     options=[{"label": c, "value": c} for c in cols],
                     multi=True, placeholder="Input columns",
                     className="dropdown"),'''

feat_new = '''        dcc.Dropdown(id="up-features",
                     options=[{"label": c, "value": c} for c in cols],
                     value=_default_features(cols),
                     multi=True, placeholder="Input columns",
                     className="dropdown"),'''

if feat_anchor in s:
    s = s.replace(feat_anchor, feat_new)
    changes.append("Fix 2: feature dropdown now auto-selects sensible defaults")
else:
    print("WARN: Fix 2 anchor not found (up-features dropdown)")

# --- Add the _default_features helper just before _analysis_and_prediction_ui
helper = '''def _default_features(cols):
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


def _analysis_and_prediction_ui(df):'''

if 'def _analysis_and_prediction_ui(df):' in s and '_default_features' not in orig:
    s = s.replace('def _analysis_and_prediction_ui(df):', helper, 1)
    changes.append("Added _default_features helper")
else:
    print("WARN: could not add _default_features helper (already present?)")

# --- Validate + write
try:
    ast.parse(s)
except SyntaxError as e:
    print(f"NOT written — syntax error: {e}"); sys.exit(3)

if s == orig:
    print("No changes applied (all anchors missing)."); sys.exit(2)

open(P, "w").write(s)
print("PATCHED OK:")
for c in changes:
    print("  -", c)
