"""Classification layer: evaluate the trained risk models.

Loads each saved (risk x model) artefact, rebuilds the same stratified
holdout split, and reports the spec's classification metrics:
precision, recall, F1, ROC-AUC (plus accuracy for context).

For imbalanced clinical risks, recall and ROC-AUC matter most: missing a
true high-risk patient (false negative) is costlier than a false alarm,
so accuracy alone is misleading.

The best model per risk (by ROC-AUC) is written to best_models.json so the
profile layer always loads the current winner without hardcoding or
re-evaluating.

Run from src/ with:  python -m prediction.classification.evaluate
"""

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

from etl.utils import get_logger
from prediction.classification.features import (
    TARGETS, feature_columns_for, get_features, get_targets,
)

logger: logging.Logger = get_logger(__name__)

MODELS_DIR = Path(__file__).resolve().parent / "models"
BEST_MODELS_FILE = MODELS_DIR / "best_models.json"
MODEL_NAMES = ["logreg", "random_forest", "xgboost"]
RANDOM_STATE = 42
TEST_SIZE = 0.2


def _holdout(risk: str) -> tuple:
    """Rebuild the same stratified test split used during training.

    Args:
        risk: Target name.

    Returns:
        (X_test, y_test) for the given risk.
    """
    cols = feature_columns_for(risk)
    X = get_features()[cols]
    y = get_targets()[risk]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    return X_test, y_test


def evaluate_all() -> pd.DataFrame:
    """Evaluate every saved model and return a metrics comparison table.

    Returns:
        DataFrame with one row per (risk, model) and metric columns.
    """
    rows = []
    for risk in TARGETS:
        X_test, y_test = _holdout(risk)
        for name in MODEL_NAMES:
            artefact = joblib.load(MODELS_DIR / f"{risk}__{name}.joblib")
            model = artefact["model"]
            pred = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]
            rows.append({
                "risk": risk,
                "model": name,
                "accuracy": round(accuracy_score(y_test, pred), 3),
                "precision": round(precision_score(y_test, pred, zero_division=0), 3),
                "recall": round(recall_score(y_test, pred, zero_division=0), 3),
                "f1": round(f1_score(y_test, pred, zero_division=0), 3),
                "roc_auc": round(roc_auc_score(y_test, proba), 3),
            })
            logger.info("Evaluated %s / %s", risk, name)
    return pd.DataFrame(rows)


def best_per_risk(table: pd.DataFrame) -> pd.DataFrame:
    """Pick the best model per risk by ROC-AUC (robust to imbalance).

    Args:
        table: Output of evaluate_all().

    Returns:
        One row per risk: the highest-ROC-AUC model.
    """
    idx = table.groupby("risk")["roc_auc"].idxmax()
    return table.loc[idx].reset_index(drop=True)


def save_best_models(best: pd.DataFrame) -> None:
    """Write the best model name per risk to best_models.json.

    Args:
        best: Output of best_per_risk().
    """
    mapping = dict(zip(best["risk"], best["model"]))
    BEST_MODELS_FILE.write_text(json.dumps(mapping, indent=2))
    logger.info("Saved best models: %s", mapping)


if __name__ == "__main__":
    results = evaluate_all()
    print("Model comparison (all risks x all models):\n")
    print(results.to_string(index=False))

    best = best_per_risk(results)
    print("\nBest model per risk (by ROC-AUC):\n")
    print(best.to_string(index=False))

    save_best_models(best)
    print(f"\nBest-model choices saved to: {BEST_MODELS_FILE.name}")