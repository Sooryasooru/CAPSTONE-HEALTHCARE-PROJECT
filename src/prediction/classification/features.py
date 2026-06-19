"""Classification layer: build the patient feature matrix and risk targets.

One shared feature set feeds several risk models (mortality, AKI, heart
failure). Features are read from silver.patients (per-patient rows).

Design choices, following the Week 1 EDA principles:
    - Missingness is informative: for each lab, add a `<lab>_missing` flag
      BEFORE filling, so the model can use "test not ordered" as signal.
    - Missing numeric values are then filled with the column median.
    - Outliers are kept (they are real critical patients).

Leakage guard (feature_columns_for):
    - A target's own column is never a feature for that target.
    - 'duration_of_stay' is dropped for mortality (length of stay is partly
      an effect of the outcome).
    - 'creatinine' / 'urea' are dropped for AKI, because they are the lab
      markers used to DIAGNOSE AKI — keeping them lets the model read the
      answer (confirmed: AKI patients average ~3x higher creatinine/urea).

Run from src/ with:  python -m prediction.classification.features
"""

import logging

import pandas as pd

from etl.utils import get_engine, get_logger

logger: logging.Logger = get_logger(__name__)

# Numeric predictors (labs + age). Missingness flagged then median-filled.
NUMERIC_FEATURES = [
    "age", "hb", "tlc", "platelets", "glucose", "urea",
    "creatinine", "bnp", "ef", "duration_of_stay",
]

# Binary comorbidity / history predictors (already 0/1 in silver).
BINARY_FEATURES = [
    "smoking", "alcohol", "dm", "htn", "cad", "prior_cmp", "ckd",
    "raised_cardiac_enzymes", "severe_anaemia", "anaemia",
    "stable_angina", "acs", "stemi", "atypical_chestpain",
]

# Targets we model. Each is excluded from its own feature set to avoid leakage.
TARGETS = {
    "mortality": "outcome",        # derived: outcome == 'Expiry'
    "aki": "aki",
    "heart_failure": "heart_failure",
}

# Extra features to drop per target beyond the same-name column (leakage).
# These are columns that effectively reveal the target.
EXTRA_LEAKAGE = {
    "mortality": ["duration_of_stay", "duration_of_stay_missing"],
    "aki": ["creatinine", "creatinine_missing", "urea", "urea_missing"],
}


def _load_raw() -> pd.DataFrame:
    """Read the patient columns needed for features and targets."""
    cols = set(NUMERIC_FEATURES + BINARY_FEATURES + ["outcome", "aki",
                                                     "heart_failure"])
    engine = get_engine()
    logger.info("Reading %d columns from silver.patients", len(cols))
    return pd.read_sql(f"SELECT {', '.join(cols)} FROM silver.patients", engine)


def get_features() -> pd.DataFrame:
    """Build the predictor matrix with missingness flags and median fill.

    Returns:
        DataFrame of numeric + binary features, plus `<lab>_missing` flags.
        No missing values remain.
    """
    df = _load_raw()

    numeric = df[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    # Missingness is signal: flag before filling.
    for col in NUMERIC_FEATURES:
        numeric[f"{col}_missing"] = numeric[col].isna().astype(int)
    numeric = numeric.fillna(numeric.median())

    binary = df[BINARY_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    binary = binary.astype(int)

    features = pd.concat([numeric, binary], axis=1)
    logger.info("Built feature matrix: %d rows x %d features", *features.shape)
    return features


def get_targets() -> pd.DataFrame:
    """Build the three binary risk targets.

    Returns:
        DataFrame with columns: mortality, aki, heart_failure (all 0/1).
    """
    df = _load_raw()
    targets = pd.DataFrame({
        "mortality": (df["outcome"] == "Expiry").astype(int),
        "aki": pd.to_numeric(df["aki"], errors="coerce").fillna(0).astype(int),
        "heart_failure": pd.to_numeric(df["heart_failure"],
                                       errors="coerce").fillna(0).astype(int),
    })
    logger.info("Built targets: %s",
                {c: int(targets[c].sum()) for c in targets.columns})
    return targets


def feature_columns_for(target: str) -> list[str]:
    """Return feature names valid for a target, excluding leakage columns.

    Drops the target's own same-name column (if present) plus any extra
    leakage-prone columns declared in EXTRA_LEAKAGE.

    Args:
        target: One of TARGETS keys.

    Returns:
        List of feature column names safe to use for this target.
    """
    cols = list(get_features().columns)
    drop = set(EXTRA_LEAKAGE.get(target, []))
    same_name = {"aki": "aki", "heart_failure": "heart_failure"}.get(target)
    if same_name:
        drop.add(same_name)
    removed = [c for c in cols if c in drop]
    cols = [c for c in cols if c not in drop]
    if removed:
        logger.info("Leakage guard for '%s': dropped %s", target, removed)
    return cols


if __name__ == "__main__":
    X = get_features()
    y = get_targets()
    print("Feature matrix:", X.shape)
    for t in TARGETS:
        print(f"  {t:14s} uses {len(feature_columns_for(t))} features")
    print("\nTargets (positive counts):")
    print(y.sum())
    print("\nMissing values left in X:", int(X.isna().sum().sum()))