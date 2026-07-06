"""
HAIP - Automatic feature suggestion for uploaded-data analytics.
=================================================================
Given an arbitrary uploaded dataframe and a chosen target column,
rank the remaining columns by predictive signal and auto-select a
sensible feature set -- WITH a leakage guard.

Unlike the Synthea pipeline (fixed schema, hand-curated EXTRA_LEAKAGE),
uploaded data has an unknown schema, so leakage must be caught by
heuristic rules rather than a fixed column list. Same discipline,
different mechanism.

Returns (suggested_features, dropped) where `dropped` is a list of
(column, reason) pairs so every exclusion is DISCLOSED, not hidden.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    mutual_info_classif, mutual_info_regression,
)
from sklearn.preprocessing import LabelEncoder

# Column-name patterns that commonly leak the outcome or are post-outcome /
# diagnostic markers. Matched case-insensitively as substrings. This is a
# heuristic net, deliberately conservative -- flagged items are disclosed,
# and the clinician can override.
LEAKY_PATTERNS = [
    "expected_", "_expected", "predicted", "prediction", "score",
    "outcome", "died", "death", "mortality", "expired",
    "discharge", "length_of_stay", "los", "readmit", "readmission",
    "probability", "risk_flag", "label", "target",
]

# Diagnostic markers: if the target looks like a specific condition, these
# lab/marker fragments are likely post-hoc confirmation of it, not honest
# pre-outcome predictors.
DIAGNOSTIC_MARKERS = ["creatinine", "urea", "egfr", "troponin", "lactate",
                      "bilirubin", "inr", "d_dimer", "bnp"]


def _is_id_like(series: pd.Series) -> bool:
    """A column that is (near-)unique per row carries no general signal
    and is almost always an identifier -- BUT continuous float columns are
    naturally near-unique and are legitimate features, so exempt them.
    Only integer / string / categorical near-unique columns are treated
    as identifiers."""
    n = len(series)
    if n == 0:
        return False
    # Continuous floats (e.g. bmi, expected_mortality) are not identifiers.
    if pd.api.types.is_float_dtype(series):
        return False
    return series.nunique(dropna=True) >= 0.95 * n

def _name_is_leaky(col: str, target: str) -> str | None:
    """Return awer()
    t = targ reason string if the column name looks leaky, else None."""
    c = col.loet.lower()

    # Any column whose name embeds the target name (e.g. target 'mortality',
    # column 'expected_mortality') is suspect.
    t_core = re.sub(r"[^a-z0-9]+", "", t)
    c_core = re.sub(r"[^a-z0-9]+", "", c)
    if t_core and t_core in c_core and c_core != t_core:
        return f"name embeds target '{target}'"

    for pat in LEAKY_PATTERNS:
        if pat in c:
            return f"matches leakage pattern '{pat}'"

    for marker in DIAGNOSTIC_MARKERS:
        if marker in c:
            return f"diagnostic marker '{marker}' (post-outcome)"

    return None


def _encode_for_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric-encode a frame so mutual_info can consume it. Categoricals
    are label-encoded; this is only for ranking, not for training."""
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            out[col] = pd.to_numeric(s, errors="coerce").fillna(s.median()
                                                                if s.notna().any() else 0)
        else:
            enc = LabelEncoder()
            out[col] = enc.fit_transform(s.astype(str).fillna("Missing"))
    return out


def suggest_features(df: pd.DataFrame, target: str, top_k: int = 12):
    """
    Rank candidate columns for predicting `target` and return a suggested
    subset plus a disclosed list of dropped columns.

    Returns:
        suggested: list[str]         -- recommended feature columns
        dropped:   list[(col, why)]  -- excluded columns with reasons
        ranked:    list[(col, score)]-- surviving candidates by signal
    """
    if target not in df.columns:
        return [], [], []

    dropped: list[tuple[str, str]] = []
    candidates: list[str] = []

    for col in df.columns:
        if col == target:
            continue
        # 1. Structural: id-like columns.
        if _is_id_like(df[col]):
            dropped.append((col, "id-like (near-unique per row)"))
            continue
        # 2. Name-based leakage / diagnostic markers.
        reason = _name_is_leaky(col, target)
        if reason:
            dropped.append((col, reason))
            continue
        # 3. Empty / constant columns carry no signal.
        if df[col].nunique(dropna=True) <= 1:
            dropped.append((col, "constant or empty"))
            continue
        candidates.append(col)

    if not candidates:
        return [], dropped, []

    # Determine problem type from the target.
    y_raw = df[target]
    is_classification = (not pd.api.types.is_numeric_dtype(y_raw)) or \
        (y_raw.nunique(dropna=True) <= 10)

    X = _encode_for_scoring(df[candidates])
    if pd.api.types.is_numeric_dtype(y_raw):
        y = pd.to_numeric(y_raw, errors="coerce")
    else:
        y = pd.Series(LabelEncoder().fit_transform(y_raw.astype(str).fillna("Missing")),
                      index=y_raw.index)

    mask = y.notna()
    X, y = X[mask], y[mask]
    if len(y) < 10:
        # Too little data to score reliably -- return name-clean candidates as-is.
        return candidates[:top_k], dropped, [(c, 0.0) for c in candidates[:top_k]]

    try:
        if is_classification:
            scores = mutual_info_classif(X, y, random_state=0)
        else:
            scores = mutual_info_regression(X, y, random_state=0)
    except Exception:
        return candidates[:top_k], dropped, [(c, 0.0) for c in candidates[:top_k]]

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    # Keep only columns with non-trivial signal, cap at top_k.
    suggested = [c for c, s in ranked if s > 0][:top_k]
    if not suggested:  # fall back to top_k by rank even if signal is weak
        suggested = [c for c, _ in ranked[:top_k]]

    return suggested, dropped, ranked


if __name__ == "__main__":
    # Smoke test with a tiny synthetic frame.
    demo = pd.DataFrame({
        "patient_id": range(100),
        "age": np.random.randint(20, 90, 100),
        "bmi": np.random.normal(27, 5, 100),
        "expected_mortality": np.random.rand(100),
        "mortality": np.random.randint(0, 2, 100),
        "constant_col": [1] * 100,
    })
    sug, drp, rk = suggest_features(demo, "mortality")
    print("suggested:", sug)
    print("dropped:", drp)
    print("ranked:", [(c, round(s, 3)) for c, s in rk])
