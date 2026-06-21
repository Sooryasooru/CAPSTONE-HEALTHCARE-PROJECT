"""Classification layer: evaluate the trained risk models — honestly.

Loads each saved (risk x model) artefact, rebuilds the same stratified
holdout split, and reports a full, honest picture:

    - confusion matrix (TN / FP / FN / TP) — where the model is WRONG
    - precision, recall, F1
    - ROC-AUC and PR-AUC (PR-AUC is the honest headline for imbalanced
      targets: it ignores the easy true-negatives and focuses on the
      positive class we actually care about)

Why PR-AUC over ROC-AUC for selection: with 19% readmission / 3% mortality,
ROC-AUC looks flatteringly high because true-negatives dominate. PR-AUC
exposes how well the model really finds the rare positive cases, so the
best model per risk is chosen by PR-AUC.

The best model per risk is written to best_models.json for the profile layer.

Run from src/ with:  python -m prediction.classification.evaluate
"""

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
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
    """Rebuild the same stratified test split used during training."""
    cols = feature_columns_for(risk)
    X = get_features()[cols]
    y = get_targets()[risk]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    return X_test, y_test


def evaluate_all() -> pd.DataFrame:
    """Evaluate every saved model and return a full metrics table.

    Returns:
        DataFrame with one row per (risk, model): confusion-matrix cells
        plus precision, recall, F1, ROC-AUC, PR-AUC.
    """
    rows = []
    for risk in TARGETS:
        X_test, y_test = _holdout(risk)
        for name in MODEL_NAMES:
            artefact = joblib.load(MODELS_DIR / f"{risk}__{name}.joblib")
            model = artefact["model"]
            pred = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]

            tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
            rows.append({
                "risk": risk,
                "model": name,
                "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
                "precision": round(precision_score(y_test, pred, zero_division=0), 3),
                "recall": round(recall_score(y_test, pred, zero_division=0), 3),
                "f1": round(f1_score(y_test, pred, zero_division=0), 3),
                "roc_auc": round(roc_auc_score(y_test, proba), 3),
                "pr_auc": round(average_precision_score(y_test, proba), 3),
            })
            logger.info("Evaluated %s / %s", risk, name)
    return pd.DataFrame(rows)


def best_per_risk(table: pd.DataFrame) -> pd.DataFrame:
    """Pick the best model per risk by PR-AUC (honest for imbalance)."""
    idx = table.groupby("risk")["pr_auc"].idxmax()
    return table.loc[idx].reset_index(drop=True)


def save_best_models(best: pd.DataFrame) -> None:
    """Write the best model name per risk to best_models.json."""
    mapping = dict(zip(best["risk"], best["model"]))
    BEST_MODELS_FILE.write_text(json.dumps(mapping, indent=2))
    logger.info("Saved best models: %s", mapping)


def _print_confusion(row: pd.Series) -> None:
    """Print a single model's confusion matrix in a readable 2x2 block."""
    print(f"\n  {row['risk']} / {row['model']} — confusion matrix")
    print(f"                 pred 0    pred 1")
    print(f"    actual 0   {row['TN']:7d}  {row['FP']:7d}   (FP = false alarms)")
    print(f"    actual 1   {row['FN']:7d}  {row['TP']:7d}   (FN = missed cases)")


if __name__ == "__main__":
    results = evaluate_all()

    metric_cols = ["risk", "model", "precision", "recall", "f1",
                   "roc_auc", "pr_auc"]
    print("Model comparison — metrics (all risks x all models):\n")
    print(results[metric_cols].to_string(index=False))

    print("\n" + "=" * 60)
    print("CONFUSION MATRICES — where each best model gets it wrong")
    print("=" * 60)
    best = best_per_risk(results)
    for _, row in best.iterrows():
        _print_confusion(row)

    print("\n" + "=" * 60)
    print("Best model per risk (by PR-AUC — honest for imbalance):\n")
    print(best[metric_cols].to_string(index=False))

    save_best_models(best)
    print(f"\nBest-model choices saved to: {BEST_MODELS_FILE.name}")