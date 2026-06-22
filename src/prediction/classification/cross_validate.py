"""Classification layer: k-fold cross-validation with mean +/- std.

A single train/test split can be lucky or unlucky — especially on a small
dataset. This module runs stratified k-fold cross-validation so each metric
is reported as mean +/- standard deviation across folds. That turns a single
number ("PR-AUC 0.58") into a defensible, stable estimate ("0.58 +/- 0.04"),
directly answering "is this score just a lucky split?".

Uses the same leakage-safe features and the best model per risk as the rest
of the pipeline. PR-AUC remains the honest headline for imbalanced targets.

Run from src/ with:  python -m prediction.classification.cross_validate
"""

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate as sk_cv
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from etl.utils import get_logger
from prediction.classification.features import (
    TARGETS, feature_columns_for, get_features, get_targets,
)

logger: logging.Logger = get_logger(__name__)

MODELS_DIR = Path(__file__).resolve().parent / "models"
BEST_MODELS_FILE = MODELS_DIR / "best_models.json"
RANDOM_STATE = 42
N_SPLITS = 5

# Scoring metrics (PR-AUC = average_precision is the honest headline)
SCORING = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


def _estimator(model_name: str, spw: float):
    """Rebuild an estimator by name (matches models.py configuration)."""
    if model_name == "logreg":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       random_state=RANDOM_STATE)),
        ])
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1)
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1,
        scale_pos_weight=spw, eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=-1)


def _best_model_names() -> dict:
    """Load the best model per risk chosen by evaluate.py."""
    if not BEST_MODELS_FILE.exists():
        # fall back to random_forest if evaluate hasn't been run
        return {t: "random_forest" for t in TARGETS}
    return json.loads(BEST_MODELS_FILE.read_text())


def cross_validate_all() -> dict:
    """Run stratified k-fold CV for the best model of each risk.

    Returns:
        Dict risk -> {metric: (mean, std)} across folds.
    """
    features = get_features()
    targets = get_targets()
    best = _best_model_names()
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                         random_state=RANDOM_STATE)

    results = {}
    for risk in TARGETS:
        cols = feature_columns_for(risk)
        X = features[cols]
        y = targets[risk]

        neg, pos = int((y == 0).sum()), int((y == 1).sum())
        spw = neg / pos if pos else 1.0
        est = _estimator(best.get(risk, "random_forest"), spw)

        logger.info("Cross-validating '%s' (%s) over %d folds, %d positives",
                    risk, best.get(risk, "random_forest"), N_SPLITS, pos)
        scores = sk_cv(est, X, y, cv=cv, scoring=SCORING, n_jobs=-1)

        results[risk] = {
            metric: (round(float(np.mean(scores[f"test_{metric}"])), 3),
                     round(float(np.std(scores[f"test_{metric}"])), 3))
            for metric in SCORING
        }
    return results


if __name__ == "__main__":
    res = cross_validate_all()
    print(f"\nStratified {N_SPLITS}-fold cross-validation "
          f"(mean +/- std across folds)\n")
    header = f"{'risk':12s} " + " ".join(f"{m:>16s}" for m in SCORING)
    print(header)
    print("-" * len(header))
    for risk, metrics in res.items():
        cells = []
        for m in SCORING:
            mean, std = metrics[m]
            cells.append(f"{mean:.3f}+/-{std:.3f}".rjust(16))
        print(f"{risk:12s} " + " ".join(cells))
    print("\nPR-AUC is the honest headline for imbalanced targets. The +/- "
          "shows stability: a small std means the score is not a lucky split.")