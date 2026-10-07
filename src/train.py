from __future__ import annotations

import json
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from xgboost import XGBClassifier

from src.config import (
    METRICS_DIR,
    MODEL_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.features import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    build_feature_target,
    load_modeling_data,
)


MODEL_FILE = (
    MODEL_DIR
    / "churn_model.joblib"
)

TRAINING_REPORT_FILE = (
    METRICS_DIR
    / "model_training_report.json"
)

TEST_DATA_FILE = (
    MODEL_DIR
    / "test_data.joblib"
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing pipeline."""

    numerical_pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_models() -> dict[str, Any]:
    """Create candidate classification models."""

    return {
        "logistic_regression":
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),

        "random_forest":
            RandomForestClassifier(
                n_estimators=400,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),

        "xgboost":
            XGBClassifier(
                n_estimators=400,
                learning_rate=0.03,
                max_depth=4,
                min_child_weight=3,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
    }


def build_pipeline(
    model: Any,
) -> Pipeline:
    """Combine preprocessing and estimator."""

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                model,
            ),
        ]
    )


def evaluate_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "threshold": round(
            float(threshold),
            2,
        ),
        "precision": round(
            float(
                precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),
            4,
        ),
        "recall": round(
            float(
                recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),
            4,
        ),
        "f1": round(
            float(
                f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),
            4,
        ),
    }


def optimize_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> tuple[float, list[dict[str, float]]]:
    """
    Select threshold using F1 score.

    Threshold tuning is performed on the validation split,
    never on the final test set.
    """

    thresholds = np.arange(
        0.20,
        0.81,
        0.02,
    )

    results = [
        evaluate_threshold(
            y_true,
            probabilities,
            float(threshold),
        )
        for threshold in thresholds
    ]

    best = max(
        results,
        key=lambda item: item["f1"],
    )

    return (
        float(best["threshold"]),
        results,
    )


def main() -> None:

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_modeling_data()

    X, y, customer_ids = (
        build_feature_target(df)
    )

    # ---------------------------------------------------------
    # Hold out the final test set.
    # ---------------------------------------------------------

    (
        X_train_validation,
        X_test,
        y_train_validation,
        y_test,
        id_train_validation,
        id_test,
    ) = train_test_split(
        X,
        y,
        customer_ids,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # ---------------------------------------------------------
    # Create a validation split for threshold optimization.
    # Final test data remains untouched.
    # ---------------------------------------------------------

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
    ) = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=0.20,
        stratify=y_train_validation,
        random_state=RANDOM_STATE,
    )

    logger.info(
        "Training rows: %s",
        len(X_train),
    )

    logger.info(
        "Validation rows: %s",
        len(X_validation),
    )

    logger.info(
        "Final test rows: %s",
        len(X_test),
    )

    models = build_models()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    comparison: dict[
        str,
        dict[str, float],
    ] = {}

    pipelines: dict[
        str,
        Pipeline,
    ] = {}

    # ---------------------------------------------------------
    # Model comparison
    # ---------------------------------------------------------

    for model_name, model in models.items():

        logger.info(
            "Training candidate model: %s",
            model_name,
        )

        pipeline = build_pipeline(
            model
        )

        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        validation_probabilities = (
            pipeline.predict_proba(
                X_validation
            )[:, 1]
        )

        validation_auc = roc_auc_score(
            y_validation,
            validation_probabilities,
        )

        comparison[model_name] = {
            "cv_roc_auc_mean": round(
                float(cv_scores.mean()),
                4,
            ),
            "cv_roc_auc_std": round(
                float(cv_scores.std()),
                4,
            ),
            "validation_roc_auc": round(
                float(validation_auc),
                4,
            ),
        }

        pipelines[model_name] = pipeline

        logger.info(
            "%s | CV ROC-AUC = %.4f | "
            "Validation ROC-AUC = %.4f",
            model_name,
            cv_scores.mean(),
            validation_auc,
        )

    # ---------------------------------------------------------
    # Choose best candidate using validation ROC-AUC.
    # ---------------------------------------------------------

    best_model_name = max(
        comparison,
        key=lambda name:
            comparison[name][
                "validation_roc_auc"
            ],
    )

    logger.info(
        "Selected model: %s",
        best_model_name,
    )

    selected_pipeline = (
        pipelines[best_model_name]
    )

    # ---------------------------------------------------------
    # Optimize business classification threshold using
    # validation data only.
    # ---------------------------------------------------------

    validation_probabilities = (
        selected_pipeline.predict_proba(
            X_validation
        )[:, 1]
    )

    (
        best_threshold,
        threshold_results,
    ) = optimize_threshold(
        y_validation,
        validation_probabilities,
    )

    logger.info(
        "Selected probability threshold: %.2f",
        best_threshold,
    )

    # ---------------------------------------------------------
    # Refit selected model using all non-test observations.
    # ---------------------------------------------------------

    final_pipeline = build_pipeline(
        models[best_model_name]
    )

    final_pipeline.fit(
        X_train_validation,
        y_train_validation,
    )

    # ---------------------------------------------------------
    # Final test evaluation.
    # ---------------------------------------------------------

    test_probabilities = (
        final_pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    test_predictions = (
        test_probabilities
        >= best_threshold
    ).astype(int)

    test_metrics = {
        "roc_auc": round(
            float(
                roc_auc_score(
                    y_test,
                    test_probabilities,
                )
            ),
            4,
        ),
        "precision": round(
            float(
                precision_score(
                    y_test,
                    test_predictions,
                    zero_division=0,
                )
            ),
            4,
        ),
        "recall": round(
            float(
                recall_score(
                    y_test,
                    test_predictions,
                    zero_division=0,
                )
            ),
            4,
        ),
        "f1": round(
            float(
                f1_score(
                    y_test,
                    test_predictions,
                    zero_division=0,
                )
            ),
            4,
        ),
    }

    logger.info(
        "Final test ROC-AUC: %.4f",
        test_metrics["roc_auc"],
    )

    logger.info(
        "Final test Precision: %.4f",
        test_metrics["precision"],
    )

    logger.info(
        "Final test Recall: %.4f",
        test_metrics["recall"],
    )

    logger.info(
        "Final test F1: %.4f",
        test_metrics["f1"],
    )

    # ---------------------------------------------------------
    # Persist model package.
    # ---------------------------------------------------------

    model_package = {
        "pipeline": final_pipeline,
        "threshold": best_threshold,
        "model_name": best_model_name,
        "features": list(X.columns),
    }

    joblib.dump(
        model_package,
        MODEL_FILE,
    )

    # Save untouched test observations for evaluate.py.
    test_package = {
        "X_test": X_test,
        "y_test": y_test,
        "customer_ids": id_test,
    }

    joblib.dump(
        test_package,
        TEST_DATA_FILE,
    )

    report = {
        "dataset": {
            "total_rows": int(len(df)),
            "training_validation_rows":
                int(len(X_train_validation)),
            "test_rows": int(len(X_test)),
            "positive_rate_pct": round(
                float(y.mean() * 100),
                2,
            ),
        },
        "model_comparison": comparison,
        "selected_model":
            best_model_name,
        "selected_threshold":
            best_threshold,
        "threshold_analysis":
            threshold_results,
        "final_test_metrics":
            test_metrics,
    }

    with open(
        TRAINING_REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    logger.info(
        "Model saved to %s",
        MODEL_FILE,
    )

    logger.info(
        "Training report saved to %s",
        TRAINING_REPORT_FILE,
    )

    logger.info(
        "ML training pipeline completed successfully."
    )


if __name__ == "__main__":
    main()