from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from src.config import MODEL_DIR
from src.features import MODEL_FEATURES


MODEL_FILE = (
    MODEL_DIR
    / "churn_model.joblib"
)


@lru_cache(maxsize=1)
def load_production_model() -> dict:
    """Load and cache the production model package."""

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Production model not found. "
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
            "Invalid model package. "
            f"Missing keys: {sorted(missing)}"
        )

    if list(
        package["features"]
    ) != MODEL_FEATURES:
        raise ValueError(
            "Production model feature schema "
            "does not match MODEL_FEATURES."
        )

    return package


def probability_to_risk(
    probability: float,
) -> str:
    """Convert churn probability to risk band."""

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
    """Assign operational retention priority."""

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


def score_customer(
    customer: dict,
) -> dict:
    """
    Score one customer using the persisted production model.
    """

    package = load_production_model()

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in customer
    ]

    if missing_features:
        raise ValueError(
            "Customer is missing model features: "
            f"{missing_features}"
        )

    frame = pd.DataFrame(
        [
            {
                feature: customer[feature]
                for feature in MODEL_FEATURES
            }
        ]
    )

    probability = float(
        package[
            "pipeline"
        ].predict_proba(
            frame
        )[0, 1]
    )

    threshold = float(
        package["threshold"]
    )

    predicted_churn = int(
        probability >= threshold
    )

    risk_segment = (
        probability_to_risk(
            probability
        )
    )

    monthly_charges = float(
        customer["monthly_charges"]
    )

    retention_priority = (
        assign_retention_priority(
            risk_segment,
            monthly_charges,
        )
    )

    revenue_risk = (
        monthly_charges
        * probability
    )

    return {
        "churn_probability":
            round(probability, 6),

        "predicted_churn":
            predicted_churn,

        "risk_segment":
            risk_segment,

        "retention_priority":
            retention_priority,

        "expected_monthly_revenue_at_risk":
            round(revenue_risk, 2),

        "model_name":
            str(package["model_name"]),

        "decision_threshold":
            threshold,
    }