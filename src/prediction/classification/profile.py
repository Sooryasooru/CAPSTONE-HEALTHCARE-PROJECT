"""Classification layer: produce a multi-risk profile for one inpatient stay.

Loads the best model per risk (from best_models.json, written by evaluate.py)
and scores a single inpatient encounter across all risks at once —
readmission, mortality, high cost.

This is a triage / planning aid, NOT a diagnosis. Scores reflect what is
known at or before discharge (the same leakage-safe features used in
training).

Two entry points:
    score_encounter_by_index(i) - score an existing feature-matrix row (demo)
    score_encounter(features)   - score a manual feature dict ("what-if")

Run from src/ with:  python -m prediction.classification.profile
"""

import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from etl.utils import get_logger
from prediction.classification.features import (
    TARGETS, get_features,
)

logger: logging.Logger = get_logger(__name__)

MODELS_DIR = Path(__file__).resolve().parent / "models"
BEST_MODELS_FILE = MODELS_DIR / "best_models.json"

# Human-readable labels for the risk panel.
RISK_LABELS = {
    "readmission": "30-day readmission risk",
    "mortality": "30-day mortality risk",
    "high_cost": "High-cost stay risk",
}


def _load_best_models() -> dict:
    """Load the best model artefact per risk, as chosen by evaluate.py."""
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

    Each model uses only its own leakage-safe feature columns (stored in
    the artefact), so high_cost correctly excludes total_claim_cost.
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


def score_encounter_by_index(index: int) -> dict:
    """Score an existing inpatient-encounter row from the feature matrix."""
    features = get_features()
    if not 0 <= index < len(features):
        raise IndexError(f"index {index} out of range (0..{len(features) - 1})")
    row = features.iloc[[index]]
    logger.info("Scoring encounter at index %d", index)
    return _score(row)


def score_encounter(feature_values: dict) -> dict:
    """Score a manually-supplied stay ("what-if").

    Missing features default to the cohort median/zero via the feature
    matrix, so only the values you care about need to be supplied.
    """
    template = get_features().median(numeric_only=True).to_frame().T
    for key, value in feature_values.items():
        if key not in template.columns:
            raise KeyError(f"unknown feature '{key}'")
        template[key] = value
    logger.info("Scoring manual stay with %d supplied features",
                len(feature_values))
    return _score(template)


if __name__ == "__main__":
    print("Risk profile — existing inpatient stay (index 0):\n")
    prof = score_encounter_by_index(0)
    for r in prof["risks"]:
        print(f"  {r['label']:28s} {r['probability_pct']:5.1f}%")

    print("\nRisk profile — manual 'what-if' "
          "(older patient, long stay, hypertension):\n")
    whatif = score_encounter({"age": 78, "stay_length_days": 12,
                              "cmb_hypertension": 1, "cmb_ischemic_heart": 1})
    for r in whatif["risks"]:
        print(f"  {r['label']:28s} {r['probability_pct']:5.1f}%")