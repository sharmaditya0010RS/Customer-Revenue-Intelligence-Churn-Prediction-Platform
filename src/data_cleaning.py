from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    METRICS_DIR,
)


RAW_FILE = RAW_DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
CLEAN_FILE = PROCESSED_DATA_DIR / "customers_clean.csv"
QUALITY_REPORT_FILE = METRICS_DIR / "data_quality_report.json"


EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


COLUMN_RENAME_MAP = {
    "customerID": "customer_id",
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn",
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_raw_data(path: Path = RAW_FILE) -> pd.DataFrame:
    """Load the immutable raw customer dataset."""

    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {path}"
        )

    logger.info("Loading raw dataset from %s", path)

    df = pd.read_csv(path)

    logger.info(
        "Raw dataset loaded successfully: %s rows x %s columns",
        df.shape[0],
        df.shape[1],
    )

    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Validate expected columns and basic dataset structure."""

    missing_columns = sorted(
        set(EXPECTED_COLUMNS) - set(df.columns)
    )

    unexpected_columns = sorted(
        set(df.columns) - set(EXPECTED_COLUMNS)
    )

    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}"
        )

    if unexpected_columns:
        raise ValueError(
            f"Unexpected columns found: {unexpected_columns}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df["customerID"].isna().any():
        raise ValueError("customerID contains missing values.")

    logger.info("Schema validation passed.")


def generate_raw_quality_report(
    df: pd.DataFrame,
) -> dict:
    """Generate data-quality metrics before cleaning."""

    total_charges_numeric = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    report = {
        "raw_rows": int(df.shape[0]),
        "raw_columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_customer_ids": int(
            df["customerID"].duplicated().sum()
        ),
        "missing_values": {
            column: int(value)
            for column, value in df.isna().sum().items()
        },
        "blank_total_charges": int(
            total_charges_numeric.isna().sum()
        ),
        "churn_distribution": {
            str(key): int(value)
            for key, value in df["Churn"]
            .value_counts(dropna=False)
            .items()
        },
    }

    return report


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize customer data."""

    logger.info("Starting cleaning pipeline.")

    clean_df = df.copy()

    # Remove exact duplicate records.
    clean_df = clean_df.drop_duplicates().copy()

    # Standardize column names.
    clean_df = clean_df.rename(
        columns=COLUMN_RENAME_MAP
    )

    # Strip whitespace from text columns.
    object_columns = clean_df.select_dtypes(
        include="object"
    ).columns

    for column in object_columns:
        clean_df[column] = clean_df[column].str.strip()

    # Convert numeric fields explicitly.
    clean_df["tenure"] = pd.to_numeric(
        clean_df["tenure"],
        errors="raise",
    )

    clean_df["monthly_charges"] = pd.to_numeric(
        clean_df["monthly_charges"],
        errors="raise",
    )

    clean_df["total_charges"] = pd.to_numeric(
        clean_df["total_charges"],
        errors="coerce",
    )

    # Blank TotalCharges values correspond to customers with
    # zero tenure. They have not accumulated historical charges.
    zero_tenure_mask = (
        clean_df["total_charges"].isna()
        & clean_df["tenure"].eq(0)
    )

    clean_df.loc[
        zero_tenure_mask,
        "total_charges",
    ] = 0.0

    # Any remaining missing TotalCharges would indicate
    # an unexpected data-quality issue.
    if clean_df["total_charges"].isna().any():
        raise ValueError(
            "Unexpected missing values remain in total_charges."
        )

    # Convert binary indicator.
    clean_df["senior_citizen"] = (
        clean_df["senior_citizen"]
        .astype(int)
    )

    # Create ML/BI target.
    churn_mapping = {
        "No": 0,
        "Yes": 1,
    }

    clean_df["churn_flag"] = (
        clean_df["churn"]
        .map(churn_mapping)
    )

    if clean_df["churn_flag"].isna().any():
        invalid_values = (
            clean_df.loc[
                clean_df["churn_flag"].isna(),
                "churn",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Unexpected churn values: {invalid_values}"
        )

    clean_df["churn_flag"] = (
        clean_df["churn_flag"].astype(int)
    )

    # Business validation.
    if clean_df["customer_id"].duplicated().any():
        raise ValueError(
            "Duplicate customer IDs remain after cleaning."
        )

    if (clean_df["tenure"] < 0).any():
        raise ValueError("Negative tenure detected.")

    if (clean_df["monthly_charges"] < 0).any():
        raise ValueError(
            "Negative monthly charges detected."
        )

    if (clean_df["total_charges"] < 0).any():
        raise ValueError(
            "Negative total charges detected."
        )

    logger.info("Cleaning pipeline completed.")

    return clean_df


def create_business_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create reusable BI-oriented customer features."""

    logger.info("Creating business features.")

    featured_df = df.copy()

    # Tenure segmentation.
    tenure_bins = [
        -1,
        12,
        24,
        48,
        60,
        np.inf,
    ]

    tenure_labels = [
        "0-12 Months",
        "13-24 Months",
        "25-48 Months",
        "49-60 Months",
        "61+ Months",
    ]

    featured_df["tenure_group"] = pd.cut(
        featured_df["tenure"],
        bins=tenure_bins,
        labels=tenure_labels,
    ).astype(str)

    # Average historical revenue per month of tenure.
    featured_df["avg_revenue_per_tenure_month"] = np.where(
        featured_df["tenure"] > 0,
        featured_df["total_charges"]
        / featured_df["tenure"],
        featured_df["monthly_charges"],
    )

    service_columns = [
        "phone_service",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
    ]

    featured_df["service_count"] = sum(
        featured_df[column].eq("Yes").astype(int)
        for column in service_columns
    )

    # Useful BI indicator for current monthly revenue exposure.
    featured_df["monthly_revenue_at_risk"] = np.where(
        featured_df["churn_flag"].eq(1),
        featured_df["monthly_charges"],
        0.0,
    )

    return featured_df


def validate_clean_data(df: pd.DataFrame) -> None:
    """Final validation before persistence."""

    required_columns = [
        "customer_id",
        "tenure",
        "monthly_charges",
        "total_charges",
        "churn",
        "churn_flag",
        "tenure_group",
        "service_count",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Clean dataset missing columns: {missing}"
        )

    if df["customer_id"].isna().any():
        raise ValueError(
            "Missing customer IDs detected."
        )

    if df["customer_id"].duplicated().any():
        raise ValueError(
            "Duplicate customer IDs detected."
        )

    if not set(df["churn_flag"].unique()).issubset({0, 1}):
        raise ValueError(
            "churn_flag contains invalid values."
        )

    logger.info("Final clean-data validation passed.")


def save_outputs(
    df: pd.DataFrame,
    quality_report: dict,
) -> None:
    """Persist processed dataset and quality report."""

    CLEAN_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    QUALITY_REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        CLEAN_FILE,
        index=False,
    )

    quality_report["clean_rows"] = int(df.shape[0])
    quality_report["clean_columns"] = int(df.shape[1])

    quality_report["clean_missing_values"] = {
        column: int(value)
        for column, value in df.isna().sum().items()
    }

    with open(
        QUALITY_REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            quality_report,
            file,
            indent=4,
        )

    logger.info(
        "Clean dataset saved to %s",
        CLEAN_FILE,
    )

    logger.info(
        "Quality report saved to %s",
        QUALITY_REPORT_FILE,
    )


def main() -> None:
    """Execute complete data-cleaning pipeline."""

    df = load_raw_data()

    validate_schema(df)

    quality_report = generate_raw_quality_report(df)

    clean_df = clean_data(df)

    clean_df = create_business_features(clean_df)

    validate_clean_data(clean_df)

    save_outputs(
        clean_df,
        quality_report,
    )

    logger.info(
        "Pipeline successful: %s customers processed.",
        len(clean_df),
    )


if __name__ == "__main__":
    main()