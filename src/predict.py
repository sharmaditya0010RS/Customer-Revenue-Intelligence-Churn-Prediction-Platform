from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text

from src.config import (
    MODEL_DIR,
    PREDICTIONS_DIR,
)
from src.database import get_engine
from src.features import (
    MODEL_FEATURES,
    load_modeling_data,
)


MODEL_FILE = MODEL_DIR / "churn_model.joblib"

PREDICTIONS_FILE = (
    PREDICTIONS_DIR
    / "customer_churn_predictions.csv"
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_model_package() -> dict:
    """Load trained model package."""

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Trained model not found. "
            "Run python -m src.train first."
        )

    package = joblib.load(
        MODEL_FILE
    )

    required_keys = {
        "pipeline",
        "threshold",
        "model_name",
        "features",
    }

    missing = (
        required_keys
        - set(package.keys())
    )

    if missing:
        raise ValueError(
            f"Invalid model package. Missing: {missing}"
        )

    return package


def assign_risk_segment(
    probability: float,
) -> str:
    """Convert churn probability into business risk band."""

    if probability >= 0.70:
        return "Critical"

    if probability >= 0.50:
        return "High"

    if probability >= 0.30:
        return "Medium"

    return "Low"


def assign_retention_priority(
    risk_segment: str,
    monthly_charges: float,
) -> str:
    """
    Prioritize customers using both churn risk
    and current monthly revenue exposure.
    """

    if (
        risk_segment == "Critical"
        and monthly_charges >= 70
    ):
        return "P1 - Immediate Action"

    if risk_segment == "Critical":
        return "P2 - High Priority"

    if (
        risk_segment == "High"
        and monthly_charges >= 70
    ):
        return "P2 - High Priority"

    if risk_segment == "High":
        return "P3 - Targeted Retention"

    if risk_segment == "Medium":
        return "P4 - Monitor"

    return "P5 - Low Priority"


def score_customers(
    df: pd.DataFrame,
    model_package: dict,
) -> pd.DataFrame:
    """Score all customers using the production model."""

    pipeline = model_package[
        "pipeline"
    ]

    threshold = float(
        model_package["threshold"]
    )

    expected_features = list(
        model_package["features"]
    )

    if expected_features != MODEL_FEATURES:
        raise ValueError(
            "Model feature definition does not match "
            "the current production feature schema."
        )

    X = df[expected_features].copy()

    probabilities = (
        pipeline.predict_proba(X)[:, 1]
    )

    result = pd.DataFrame(
        {
            "customer_id":
                df["customer_id"].astype(str),

            "churn_probability":
                probabilities,

            "predicted_churn":
                (
                    probabilities >= threshold
                ).astype(int),

            "actual_churn":
                df["churn_flag"].astype(int),

            "monthly_charges":
                df["monthly_charges"].astype(float),

            "contract":
                df["contract"].astype(str),

            "tenure":
                df["tenure"].astype(int),

            "internet_service":
                df["internet_service"].astype(str),

            "service_count":
                df["service_count"].astype(int),
        }
    )

    result["risk_segment"] = (
        result["churn_probability"]
        .apply(assign_risk_segment)
    )

    result["retention_priority"] = [
        assign_retention_priority(
            risk_segment,
            monthly_charge,
        )
        for risk_segment, monthly_charge
        in zip(
            result["risk_segment"],
            result["monthly_charges"],
        )
    ]

    # Probability-weighted expected monthly revenue exposure.
    result["expected_monthly_revenue_at_risk"] = (
        result["monthly_charges"]
        * result["churn_probability"]
    )

    result["churn_probability"] = (
        result["churn_probability"]
        .round(6)
    )

    result[
        "expected_monthly_revenue_at_risk"
    ] = (
        result[
            "expected_monthly_revenue_at_risk"
        ]
        .round(2)
    )

    return result


def validate_predictions(
    predictions: pd.DataFrame,
    expected_rows: int,
) -> None:
    """Validate scoring output before persistence."""

    if len(predictions) != expected_rows:
        raise ValueError(
            "Prediction row count does not match "
            "source customer count."
        )

    if predictions[
        "customer_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate customer predictions detected."
        )

    probabilities = predictions[
        "churn_probability"
    ]

    if not probabilities.between(
        0,
        1,
        inclusive="both",
    ).all():
        raise ValueError(
            "Invalid churn probabilities detected."
        )

    valid_risk_segments = {
        "Low",
        "Medium",
        "High",
        "Critical",
    }

    if not set(
        predictions["risk_segment"].unique()
    ).issubset(valid_risk_segments):
        raise ValueError(
            "Invalid risk segments detected."
        )

    logger.info(
        "Prediction validation passed."
    )


def save_predictions(
    predictions: pd.DataFrame,
) -> None:
    """Save scored customer dataset."""

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    logger.info(
        "Predictions saved to %s",
        PREDICTIONS_FILE,
    )


def load_predictions_to_postgres(
    predictions: pd.DataFrame,
) -> None:
    """Persist current scoring snapshot to PostgreSQL."""

    engine = get_engine()

    logger.info(
        "Loading prediction snapshot into PostgreSQL."
    )

    predictions.to_sql(
        name="customer_predictions",
        con=engine,
        schema="analytics",
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=1000,
    )

    with engine.begin() as connection:

        connection.execute(
            text(
                """
                ALTER TABLE analytics.customer_predictions
                ADD PRIMARY KEY (customer_id);
                """
            )
        )

        connection.execute(
            text(
                """
                CREATE INDEX
                idx_customer_predictions_risk
                ON analytics.customer_predictions(
                    risk_segment
                );
                """
            )
        )

        connection.execute(
            text(
                """
                CREATE INDEX
                idx_customer_predictions_priority
                ON analytics.customer_predictions(
                    retention_priority
                );
                """
            )
        )

    logger.info(
        "Prediction snapshot loaded into PostgreSQL."
    )


def print_scoring_summary(
    predictions: pd.DataFrame,
) -> None:
    """Print management-level scoring summary."""

    logger.info(
        "========== SCORING SUMMARY =========="
    )

    logger.info(
        "Customers scored: %s",
        len(predictions),
    )

    logger.info(
        "Predicted churn customers: %s",
        int(
            predictions[
                "predicted_churn"
            ].sum()
        ),
    )

    logger.info(
        "Average churn probability: %.2f%%",
        predictions[
            "churn_probability"
        ].mean() * 100,
    )

    logger.info(
        "Expected monthly revenue at risk: %.2f",
        predictions[
            "expected_monthly_revenue_at_risk"
        ].sum(),
    )

    risk_counts = (
        predictions[
            "risk_segment"
        ]
        .value_counts()
    )

    for segment, count in risk_counts.items():
        logger.info(
            "%s risk customers: %s",
            segment,
            count,
        )

    logger.info(
        "====================================="
    )


def main() -> None:

    df = load_modeling_data()

    model_package = (
        load_model_package()
    )

    logger.info(
        "Production model: %s",
        model_package["model_name"],
    )

    logger.info(
        "Production threshold: %.2f",
        model_package["threshold"],
    )

    predictions = score_customers(
        df,
        model_package,
    )

    validate_predictions(
        predictions,
        expected_rows=len(df),
    )

    save_predictions(
        predictions
    )

    load_predictions_to_postgres(
        predictions
    )

    print_scoring_summary(
        predictions
    )

    logger.info(
        "Production scoring completed successfully."
    )


if __name__ == "__main__":
    main()