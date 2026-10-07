from __future__ import annotations

import json
import logging

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import (
    FIGURES_DIR,
    METRICS_DIR,
    MODEL_DIR,
)


MODEL_FILE = MODEL_DIR / "churn_model.joblib"
TEST_DATA_FILE = MODEL_DIR / "test_data.joblib"

EVALUATION_FILE = (
    METRICS_DIR
    / "model_evaluation.json"
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Model not found. Run python -m src.train first."
        )

    if not TEST_DATA_FILE.exists():
        raise FileNotFoundError(
            "Test data not found. Run python -m src.train first."
        )

    model_package = joblib.load(
        MODEL_FILE
    )

    test_package = joblib.load(
        TEST_DATA_FILE
    )

    pipeline = model_package["pipeline"]
    threshold = float(
        model_package["threshold"]
    )

    X_test = test_package["X_test"]
    y_test = test_package["y_test"]

    probabilities = (
        pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    evaluation = {
        "model_name":
            model_package["model_name"],
        "threshold":
            threshold,
        "roc_auc":
            round(float(roc_auc), 4),
        "confusion_matrix":
            matrix.tolist(),
        "classification_report":
            report,
    }

    with open(
        EVALUATION_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            evaluation,
            file,
            indent=4,
        )

    # ROC curve
    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities,
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {roc_auc:.3f}",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    ax.set_title(
        "Churn Model ROC Curve"
    )
    ax.set_xlabel(
        "False Positive Rate"
    )
    ax.set_ylabel(
        "True Positive Rate"
    )
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "07_model_roc_curve.png",
        dpi=150,
    )

    plt.close(fig)

    # Precision-recall curve
    precision, recall, _ = (
        precision_recall_curve(
            y_test,
            probabilities,
        )
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.plot(
        recall,
        precision,
    )

    ax.set_title(
        "Churn Model Precision-Recall Curve"
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "08_precision_recall_curve.png",
        dpi=150,
    )

    plt.close(fig)

    # Confusion matrix
    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Retained",
            "Churn",
        ],
    )

    display.plot(
        ax=ax,
        values_format="d",
    )

    ax.set_title(
        "Churn Model Confusion Matrix"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "09_confusion_matrix.png",
        dpi=150,
    )

    plt.close(fig)

    logger.info(
        "Model: %s",
        model_package["model_name"],
    )

    logger.info(
        "Decision threshold: %.2f",
        threshold,
    )

    logger.info(
        "Test ROC-AUC: %.4f",
        roc_auc,
    )

    logger.info(
        "Confusion matrix: %s",
        matrix.tolist(),
    )

    logger.info(
        "Evaluation completed successfully."
    )


if __name__ == "__main__":
    main()