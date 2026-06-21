"""Classification layer: render evaluation plots as report-ready images.

Reuses the trained models and the same stratified holdout split, then saves
presentation-ready figures to reports/evaluation/:

    - confusion_<risk>.png   : confusion-matrix heatmap for the best model
    - metrics_comparison.png : grouped bar chart of precision/recall/F1/PR-AUC
    - roc_pr_<risk>.png      : ROC curve + Precision-Recall curve, side by side

These turn the terminal-only metrics into deliverables for slides and the
report. Honest by design: the PR curves and confusion cells make failures
(false negatives, false positives) visible, not just headline scores.

Run from src/ with:  python -m prediction.classification.plots
"""

import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # no display needed; save to file
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, roc_curve,
    average_precision_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

from etl.utils import get_logger
from prediction.classification.features import (
    TARGETS, feature_columns_for, get_features, get_targets,
)
from prediction.classification.evaluate import evaluate_all, best_per_risk

logger: logging.Logger = get_logger(__name__)

MODELS_DIR = Path(__file__).resolve().parent / "models"
OUT_DIR = Path(__file__).resolve().parents[3] / "reports" / "evaluation"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Clinical palette (matches the dashboard)
TEAL = "#1A7A8C"
AMBER = "#E8A33D"
RED = "#C2453D"
INK = "#0B1F33"


def _holdout(risk: str):
    """Rebuild the same stratified test split used during training."""
    cols = feature_columns_for(risk)
    X = get_features()[cols]
    y = get_targets()[risk]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    return X_test, y_test


def _load_best(risk: str, model_name: str):
    """Load the best model artefact for a risk."""
    return joblib.load(MODELS_DIR / f"{risk}__{model_name}.joblib")


def plot_confusion(risk: str, model_name: str) -> Path:
    """Save a confusion-matrix heatmap for the best model of a risk."""
    X_test, y_test = _holdout(risk)
    model = _load_best(risk, model_name)["model"]
    cm = confusion_matrix(y_test, model.predict(X_test))

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["Negative", "Positive"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Predicted", color=INK)
    ax.set_ylabel("Actual", color=INK)
    ax.set_title(f"Confusion matrix — {risk} ({model_name})", color=INK)
    cell_notes = [["TN", "FP"], ["FN", "TP"]]
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            colour = "white" if cm[i, j] > thresh else INK
            ax.text(j, i, f"{cm[i, j]}\n{cell_notes[i][j]}",
                    ha="center", va="center", color=colour, fontsize=12)
    fig.tight_layout()
    path = OUT_DIR / f"confusion_{risk}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path.name)
    return path


def plot_metrics_comparison(table) -> Path:
    """Save a grouped bar chart comparing key metrics across best models."""
    best = best_per_risk(table)
    metrics = ["precision", "recall", "f1", "pr_auc"]
    risks = list(best["risk"])
    x = np.arange(len(risks))
    width = 0.2
    colours = [TEAL, AMBER, RED, INK]

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, best[m], width, label=m.upper(),
               color=colours[i])
    ax.set_xticks(x, labels=[f"{r}\n({m})" for r, m in
                             zip(best["risk"], best["model"])])
    ax.set_ylim(0, 1)
    ax.set_ylabel("score", color=INK)
    ax.set_title("Best model per risk — key metrics (PR-AUC is the honest headline)",
                 color=INK, fontsize=11)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "metrics_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path.name)
    return path


def plot_roc_pr(risk: str, model_name: str) -> Path:
    """Save ROC and Precision-Recall curves side by side for a risk."""
    X_test, y_test = _holdout(risk)
    model = _load_best(risk, model_name)["model"]
    proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)
    prec, rec, _ = precision_recall_curve(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    baseline = y_test.mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    ax1.plot(fpr, tpr, color=TEAL, lw=2, label=f"ROC (AUC={roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="chance")
    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.set_title(f"ROC — {risk}")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    ax2.plot(rec, prec, color=AMBER, lw=2, label=f"PR (AP={pr_auc:.3f})")
    ax2.axhline(baseline, color="gray", ls="--", lw=1,
                label=f"baseline ({baseline:.2f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"Precision-Recall — {risk}")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)

    fig.suptitle(f"{risk} ({model_name}) — ROC looks high, PR tells the truth",
                 color=INK, fontsize=11)
    fig.tight_layout()
    path = OUT_DIR / f"roc_pr_{risk}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path.name)
    return path


def render_all() -> None:
    """Render every evaluation figure to reports/evaluation/."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = evaluate_all()
    best = best_per_risk(table)

    plot_metrics_comparison(table)
    for _, row in best.iterrows():
        plot_confusion(row["risk"], row["model"])
        plot_roc_pr(row["risk"], row["model"])

    logger.info("All evaluation plots saved to %s", OUT_DIR)


if __name__ == "__main__":
    render_all()
    print(f"\nDone. Evaluation images saved to: {OUT_DIR}")