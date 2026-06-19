"""Classification layer: train risk models (LogReg, RandomForest, XGBoost).

For each risk target, trains three classifiers on a stratified 80/20 split
with class-imbalance handling, and saves them for evaluation and scoring.

    - Logistic Regression : interpretable baseline (scaled features).
    - Random Forest       : robust non-linear ensemble.
    - XGBoost             : gradient-boosted trees, strong on tabular data.

Imbalance handling:
    - LogReg / RandomForest: class_weight="balanced".
    - XGBoost: scale_pos_weight = negatives / positives.

Saved artefacts (per risk x model) go to classification/models/.

Run from src/ with:  python -m prediction.classification.models
"""

import logging
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from etl.utils import get_logger
from prediction.classification.features import (
    TARGETS, feature_columns_for, get_features, get_targets,
)

logger: logging.Logger = get_logger(__name__)

MODELS_DIR = Path(__file__).resolve().parent / "models"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def _build_estimators(scale_pos_weight: float) -> dict:
    """Return the three estimators, configured for imbalance.

    Args:
        scale_pos_weight: negatives/positives ratio for XGBoost.

    Returns:
        Dict of model_name -> estimator (LogReg wrapped in a scaling pipeline).
    """
    return {
        "logreg": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000,
                                       class_weight="balanced",
                                       random_state=RANDOM_STATE)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1),
    }


def train_all() -> None:
    """Train all (risk x model) pairs and save them to MODELS_DIR."""
    MODELS_DIR.mkdir(exist_ok=True)
    features = get_features()
    targets = get_targets()

    for risk in TARGETS:
        cols = feature_columns_for(risk)
        X = features[cols]
        y = targets[risk]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

        neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
        spw = neg / pos if pos else 1.0
        logger.info("Risk '%s': train=%d test=%d pos_weight=%.2f",
                    risk, len(X_train), len(X_test), spw)

        for name, est in _build_estimators(spw).items():
            est.fit(X_train, y_train)
            path = MODELS_DIR / f"{risk}__{name}.joblib"
            joblib.dump({"model": est, "features": cols}, path)
            logger.info("Saved %s", path.name)

    # Save the test split indices so evaluate.py uses the same holdout.
    joblib.dump(
        {"test_size": TEST_SIZE, "random_state": RANDOM_STATE},
        MODELS_DIR / "_split_config.joblib")
    logger.info("Training complete: %d models saved", len(TARGETS) * 3)


if __name__ == "__main__":
    train_all()
    print("Done. Models saved to:", MODELS_DIR)