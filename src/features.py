from __future__ import annotations

import logging

import pandas as pd

from src.config import PROCESSED_DATA_DIR


DATA_FILE = (
    PROCESSED_DATA_DIR
    / "customers_clean.csv"
)

TARGET_COLUMN = "churn_flag"

LEAKAGE_COLUMNS = [
    "customer_id",
    "churn",
    "churn_flag",
    "monthly_revenue_at_risk",
]

NUMERICAL_FEATURES = [
    "senior_citizen",
    "tenure",
    "monthly_charges",
    "total_charges",
    "avg_revenue_per_tenure_month",
    "service_count",
]

CATEGORICAL_FEATURES = [
    "gender",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
    "tenure_group",
]

MODEL_FEATURES = (
    NUMERICAL_FEATURES
    + CATEGORICAL_FEATURES
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_modeling_data() -> pd.DataFrame:
    """Load validated customer data for modeling."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Processed dataset not found. "
            "Run python -m src.data_cleaning first."
        )

    df = pd.read_csv(DATA_FILE)

    logger.info(
        "Modeling dataset loaded: %s rows x %s columns",
        df.shape[0],
        df.shape[1],
    )

    return df


def validate_modeling_data(
    df: pd.DataFrame,
) -> None:
    """Validate required ML features and target."""

    required_columns = (
        MODEL_FEATURES
        + [TARGET_COLUMN]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing modeling columns: "
            f"{missing_columns}"
        )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    target_values = set(
        df[TARGET_COLUMN].unique()
    )

    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"Unexpected target values: {target_values}"
        )

    duplicate_features = (
        len(MODEL_FEATURES)
        != len(set(MODEL_FEATURES))
    )

    if duplicate_features:
        raise ValueError(
            "Duplicate model features detected."
        )

    leaked_features = (
        set(MODEL_FEATURES)
        & set(LEAKAGE_COLUMNS)
    )

    if leaked_features:
        raise ValueError(
            "Target leakage detected: "
            f"{sorted(leaked_features)}"
        )

    logger.info(
        "Modeling-data validation passed."
    )


def build_feature_target(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Return features, target and customer IDs.

    Customer IDs are preserved separately for prediction output
    but are never used as model features.
    """

    validate_modeling_data(df)

    X = df[MODEL_FEATURES].copy()

    y = df[TARGET_COLUMN].astype(int).copy()

    customer_ids = (
        df["customer_id"]
        .astype(str)
        .copy()
    )

    logger.info(
        "Feature matrix created: %s rows x %s features",
        X.shape[0],
        X.shape[1],
    )

    return X, y, customer_ids