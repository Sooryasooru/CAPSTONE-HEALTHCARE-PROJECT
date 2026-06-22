"""Upload pipeline: train a model on ANY uploaded tabular dataset.

This is a generic, dataset-agnostic engine — separate from the Synthea-specific
models. A hospital uploads a file, picks a target column and feature columns,
and this trains a fresh model on THEIR data and returns honest metrics.

Auto-detection:
    The target column decides the problem type.
      - few unique values OR non-numeric  -> classification
      - many unique numeric values        -> regression
    The right model and metrics are chosen automatically.

Preprocessing (generic, works on any columns):
    - numeric features:    median-fill missing
    - categorical features: most-frequent fill, then one-hot encode
    - target: label-encoded for classification, numeric for regression

Honest evaluation (same discipline as the Synthea models):
    - stratified train/test split (classification) or plain split (regression)
    - classification: accuracy, precision, recall, F1, confusion matrix
    - regression: R2, MAE, RMSE
    - feature importances from the trained Random Forest

Run standalone:  python pipeline.py <csv_path> <target_col>
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    precision_score, r2_score, recall_score, root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.2
# A numeric target with at most this many unique values is treated as
# classification (e.g. 0/1 labels stored as numbers).
MAX_CLASSES_FOR_NUMERIC = 12
MAX_TEXT_CLASSES = 50  # text targets with more distinct values are IDs/dates/names

@dataclass
class PipelineResult:
    """Everything the UI needs to display a trained-model summary."""
    problem_type: str                       # "classification" | "regression"
    target: str
    features: list[str]
    n_train: int
    n_test: int
    metrics: dict                           # name -> value
    importances: list[tuple[str, float]]    # (feature, importance), sorted
    confusion: list[list[int]] | None = None  # classification only
    class_labels: list[str] = field(default_factory=list)
    error: str | None = None


def detect_problem_type(y: pd.Series) -> str:
    """Decide classification vs regression from the target column.

    Raises ValueError for targets that aren't usable for prediction:
    non-numeric columns with too many unique values (dates, IDs, names,
    free text) can't form sensible classes and aren't numeric either.
    """
    non_null = y.dropna()
    if non_null.empty:
        raise ValueError("target column is entirely empty")

    if pd.api.types.is_numeric_dtype(non_null):
        # numeric: few values -> classes, many -> regression
        if non_null.nunique() <= MAX_CLASSES_FOR_NUMERIC:
            return "classification"
        return "regression"

    # non-numeric: only valid as classification if the number of distinct
    # categories is small relative to the data (otherwise it's an ID/date/name)
    n_unique = non_null.nunique()
    if n_unique > MAX_TEXT_CLASSES or n_unique > 0.5 * len(non_null):
        raise ValueError(
            f"target has {n_unique} distinct text values — not suitable for "
            f"prediction. Pick a column with a few categories (e.g. yes/no, "
            f"a status) or a numeric column.")
    return "classification"


def _split_feature_types(df: pd.DataFrame, features: list[str]):
    """Return (numeric_cols, categorical_cols) among the chosen features."""
    numeric, categorical = [], []
    for col in features:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def _build_preprocessor(numeric: list[str], categorical: list[str]):
    """Generic preprocessing: impute + (one-hot for categoricals)."""
    transformers = []
    if numeric:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric))
    if categorical:
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", cat_pipe, categorical))
    return ColumnTransformer(transformers)


def _feature_names(preprocessor, numeric, categorical) -> list[str]:
    """Recover feature names after one-hot expansion, for importances."""
    names = list(numeric)
    if categorical:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        names += list(ohe.get_feature_names_out(categorical))
    return names


def train_on_upload(df: pd.DataFrame, target: str,
                    features: list[str]) -> PipelineResult:
    """Train a model on an uploaded dataset and return honest results.

    Args:
        df: The uploaded data (already a DataFrame).
        target: Column name to predict.
        features: Column names to use as predictors.

    Returns:
        PipelineResult with metrics, confusion matrix (if classification),
        and feature importances. On failure, result.error is set.
    """
    try:
        if target in features:
            features = [f for f in features if f != target]
        if not features:
            return PipelineResult("unknown", target, [], 0, 0, {}, [],
                                  error="no feature columns selected")

        data = df[features + [target]].dropna(subset=[target])
        if len(data) < 20:
            return PipelineResult("unknown", target, features, 0, 0, {}, [],
                                  error=f"too few rows after cleaning: {len(data)}")

        X = data[features]
        y_raw = data[target]
        problem = detect_problem_type(y_raw)

        numeric, categorical = _split_feature_types(X, features)
        pre = _build_preprocessor(numeric, categorical)

        if problem == "classification":
            le = LabelEncoder()
            y = le.fit_transform(y_raw.astype(str))
            class_labels = list(le.classes_)
            model = RandomForestClassifier(
                n_estimators=200, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1)
            stratify = y if pd.Series(y).value_counts().min() >= 2 else None
        else:
            y = pd.to_numeric(y_raw, errors="coerce").to_numpy()
            class_labels = []
            model = RandomForestRegressor(
                n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
            stratify = None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
            stratify=stratify)

        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        # Feature importances (map back through the preprocessor)
        fitted_pre = pipe.named_steps["pre"]
        feat_names = _feature_names(fitted_pre, numeric, categorical)
        importances = sorted(
            zip(feat_names, pipe.named_steps["model"].feature_importances_),
            key=lambda t: t[1], reverse=True)[:15]
        importances = [(n, round(float(v), 4)) for n, v in importances]

        if problem == "classification":
            metrics = {
                "accuracy": round(accuracy_score(y_test, pred), 3),
                "precision": round(precision_score(
                    y_test, pred, average="weighted", zero_division=0), 3),
                "recall": round(recall_score(
                    y_test, pred, average="weighted", zero_division=0), 3),
                "f1": round(f1_score(
                    y_test, pred, average="weighted", zero_division=0), 3),
            }
            cm = confusion_matrix(y_test, pred).tolist()
            return PipelineResult(
                problem, target, features, len(X_train), len(X_test),
                metrics, importances, confusion=cm, class_labels=class_labels)
        else:
            metrics = {
                "r2": round(r2_score(y_test, pred), 3),
                "mae": round(mean_absolute_error(y_test, pred), 3),
                "rmse": round(root_mean_squared_error(y_test, pred), 3),
            }
            return PipelineResult(
                problem, target, features, len(X_train), len(X_test),
                metrics, importances)

    except Exception as exc:  # surface any failure to the UI cleanly
        logger.exception("Pipeline failed")
        return PipelineResult("unknown", target, features, 0, 0, {}, [],
                              error=str(exc))


def read_any(path: str) -> pd.DataFrame:
    """Read CSV, JSON, or Excel into a DataFrame by file extension."""
    p = path.lower()
    if p.endswith(".csv"):
        return pd.read_csv(path)
    if p.endswith(".json"):
        return pd.read_json(path)
    if p.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    raise ValueError(f"unsupported file type: {path}")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 3:
        print("usage: python pipeline.py <csv_path> <target_col> "
              "[feature_col ...]")
        sys.exit(1)

    path, target = sys.argv[1], sys.argv[2]
    frame = read_any(path)
    feats = sys.argv[3:] or [c for c in frame.columns if c != target]

    result = train_on_upload(frame, target, feats)
    if result.error:
        print("ERROR:", result.error)
        sys.exit(1)

    print(f"\nProblem type : {result.problem_type}")
    print(f"Target       : {result.target}")
    print(f"Train / Test : {result.n_train} / {result.n_test}")
    print(f"Metrics      : {result.metrics}")
    if result.confusion:
        print(f"Classes      : {result.class_labels}")
        print(f"Confusion    : {result.confusion}")
    print("\nTop features :")
    for name, imp in result.importances[:8]:
        print(f"  {name:30s} {imp}")