"""Classification layer: produce a multi-risk patient risk profile.

Loads the best model per risk (from best_models.json, written by evaluate.py)
and scores a patient across all risks at once — mortality, AKI, heart failure.

This is a triage / early-warning aid, NOT a diagnosis. Because the patient
data is a cardiac cohort, the risks are scoped to cardiac / kidney / mortality.

Two entry points:
    score_patient_by_index(i) - score an existing silver.patients row (demo)
    score_patient(features)    - score a manual feature dict ("what-if")

Run from src/ with:  python -m prediction.classification.profile
"""

import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from etl.utils import get_logger
from prediction.classification.features import (
    TARGETS, feature_columns_for, get_features,
)

logger: logging.Logger = get_logger(__name__)

MODELS_DIR = Path(__file__).resolve().parent / "models"
BEST_MODELS_FILE = MODELS_DIR / "best_models.json"

# Human-readable labels for the risk panel.
RISK_LABELS = {
    "mortality": "Mortality risk",
    "aki": "Acute kidney injury risk",
    "heart_failure": "Heart failure risk",
}


def _load_best_models() -> dict:
    """Load the best model artefact per risk, as chosen by evaluate.py.

    Returns:
        Dict risk -> {"model": estimator, "features": [...]}.

    Raises:
        FileNotFoundError: if best_models.json is missing (run evaluate first).
    """
    if not BEST_MODELS_FILE.exists():
        raise FileNotFoundError(
            "best_models.json not found — run prediction.classification."
            "evaluate first to choose the best model per risk.")
    choices = json.loads(BEST_MODELS_FILE.read_text())
    loaded = {}
    for risk, model_name in choices.items():
        artefact = joblib.load(MODELS_DIR / f"{risk}__{model_name}.joblib")
        loaded[risk] = artefact
        logger.info("Loaded best model for '%s': %s", risk, model_name)
    return loaded


def _score(row: pd.DataFrame) -> dict:
    """Score one single-row feature frame across all risks.

    Args:
        row: One-row DataFrame holding every feature column in get_features().

    Returns:
        Dict with a "risks" list of {risk, label, probability_pct}.
    """
    models = _load_best_models()
    risks = []
    for risk in TARGETS:
        artefact = models[risk]
        cols = artefact["features"]
        proba = artefact["model"].predict_proba(row[cols])[0, 1]
        risks.append({
            "risk": risk,
            "label": RISK_LABELS[risk],
            "probability_pct": round(float(proba) * 100, 1),
        })
    return {"risks": risks}


def score_patient_by_index(index: int) -> dict:
    """Score an existing patient row from silver.patients.

    Args:
        index: Row position in the feature matrix (0-based).

    Returns:
        Risk profile dict (see _score).
    """
    features = get_features()
    if not 0 <= index < len(features):
        raise IndexError(f"index {index} out of range (0..{len(features) - 1})")
    row = features.iloc[[index]]
    logger.info("Scoring patient at index %d", index)
    return _score(row)


def score_patient(feature_values: dict) -> dict:
    """Score a manually-supplied patient ("what-if").

    Missing features default to the cohort median/zero via the feature matrix,
    so only the values you care about need to be supplied.

    Args:
        feature_values: Partial mapping of feature_name -> value.

    Returns:
        Risk profile dict (see _score).
    """
    template = get_features().median(numeric_only=True).to_frame().T
    for key, value in feature_values.items():
        if key not in template.columns:
            raise KeyError(f"unknown feature '{key}'")
        template[key] = value
    logger.info("Scoring manual patient with %d supplied features",
                len(feature_values))
    return _score(template)


if __name__ == "__main__":
    print("Risk profile — existing patient (index 0):\n")
    profile = score_patient_by_index(0)
    for r in profile["risks"]:
        print(f"  {r['label']:30s} {r['probability_pct']:5.1f}%")

    print("\nRisk profile — manual high-risk 'what-if' "
          "(age 78, low EF, high glucose):\n")
    whatif = score_patient({"age": 78, "ef": 25, "glucose": 280,
                            "ckd": 1, "cad": 1})
    for r in whatif["risks"]:
        print(f"  {r['label']:30s} {r['probability_pct']:5.1f}%")