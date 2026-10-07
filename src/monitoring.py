from __future__ import annotations

from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text

from src.config import MODEL_DIR
from src.database import get_engine


MODEL_PATH = MODEL_DIR / "churn_model.joblib"


def _scalar(query: str) -> Any:
    engine = get_engine()

    with engine.connect() as connection:
        return connection.execute(
            text(query)
        ).scalar()


def get_data_quality_report() -> dict[str, Any]:
    """
    Validate core production data integrity.

    Historical customers remain the fixed labeled population.
    Live customers remain a separate unlabeled production layer.
    """

    historical_rows = int(
        _scalar(
            """
            SELECT COUNT(*)
            FROM analytics.customers
            """
        )
        or 0
    )

    historical_unique = int(
        _scalar(
            """
            SELECT COUNT(DISTINCT customer_id)
            FROM analytics.customers
            """
        )
        or 0
    )

    historical_predictions = int(
        _scalar(
            """
            SELECT COUNT(*)
            FROM analytics.customer_predictions
            """
        )
        or 0
    )

    live_rows = int(
        _scalar(
            """
            SELECT COUNT(*)
            FROM analytics.live_customers
            """
        )
        or 0
    )

    live_unique = int(
        _scalar(
            """
            SELECT COUNT(DISTINCT customer_id)
            FROM analytics.live_customers
            """
        )
        or 0
    )

    live_predictions = int(
        _scalar(
            """
            SELECT COUNT(*)
            FROM analytics.live_predictions
            """
        )
        or 0
    )

    unified_rows = int(
        _scalar(
            """
            SELECT COUNT(*)
            FROM analytics.vw_all_customers
            """
        )
        or 0
    )

    unified_risk_rows = int(
        _scalar(
            """
            SELECT COUNT(*)
            FROM analytics.vw_all_customer_risk
            """
        )
        or 0
    )

    historical_null_ids = int(
        _scalar(
            """
            SELECT COUNT(*)
            FROM analytics.customers
            WHERE customer_id IS NULL
            """
        )
        or 0
    )

    live_null_ids = int(
        _scalar(
            """
            SELECT COUNT(*)
            FROM analytics.live_customers
            WHERE customer_id IS NULL
            """
        )
        or 0
    )

    historical_live_overlap = int(
        _scalar(
            """
            SELECT COUNT(*)
            FROM analytics.customers h
            INNER JOIN analytics.live_customers l
                ON h.customer_id = l.customer_id
            """
        )
        or 0
    )

    expected_unified = (
        historical_rows
        + live_rows
    )

    expected_prediction_rows = (
        historical_predictions
        + live_predictions
    )

    checks = {
        "Historical IDs Unique":
            historical_rows
            == historical_unique,

        "Live IDs Unique":
            live_rows
            == live_unique,

        "Historical IDs Complete":
            historical_null_ids == 0,

        "Live IDs Complete":
            live_null_ids == 0,

        "Historical Population Preserved":
            historical_rows == 7043,

        "Historical Prediction Coverage":
            historical_predictions
            == historical_rows,

        "Live Prediction Coverage":
            live_predictions
            == live_rows,

        "Unified Population Reconciled":
            unified_rows
            == expected_unified,

        "Unified Risk Coverage":
            unified_risk_rows
            == expected_prediction_rows,

        "Historical / Live Separation":
            historical_live_overlap == 0,
    }

    passed_checks = sum(
        bool(value)
        for value in checks.values()
    )

    total_checks = len(checks)

    return {
        "historical_rows":
            historical_rows,

        "historical_unique":
            historical_unique,

        "historical_predictions":
            historical_predictions,

        "live_rows":
            live_rows,

        "live_unique":
            live_unique,

        "live_predictions":
            live_predictions,

        "unified_rows":
            unified_rows,

        "unified_risk_rows":
            unified_risk_rows,

        "historical_live_overlap":
            historical_live_overlap,

        "checks":
            checks,

        "passed_checks":
            passed_checks,

        "total_checks":
            total_checks,

        "health_pct":
            (
                passed_checks
                / total_checks
                * 100
                if total_checks
                else 0.0
            ),
    }


def load_monitoring_population() -> pd.DataFrame:
    """
    Load the unified scored population used for
    prediction and drift monitoring.
    """

    query = """
        SELECT
            customer_id,
            customer_source,
            tenure,
            monthly_charges,
            total_charges,
            service_count,
            contract,
            internet_service,
            payment_method,
            churn_probability,
            predicted_churn,
            risk_segment,
            expected_monthly_revenue_at_risk
        FROM analytics.vw_all_customer_risk
    """

    return pd.read_sql(
        query,
        get_engine(),
    )


def calculate_psi(
    baseline: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float | None:
    """
    Calculate Population Stability Index for a
    continuous numeric feature.

    Baseline:
        Historical customer population.

    Current:
        Generated/live customer population.

    PSI interpretation used by the dashboard:
        < 0.10   Stable
        0.10-0.25 Moderate shift
        > 0.25   Significant shift
    """

    baseline_values = pd.to_numeric(
        baseline,
        errors="coerce",
    ).dropna()

    current_values = pd.to_numeric(
        current,
        errors="coerce",
    ).dropna()

    if (
        baseline_values.empty
        or current_values.empty
    ):
        return None

    if baseline_values.nunique() <= 1:
        return None

    quantiles = np.linspace(
        0,
        1,
        bins + 1,
    )

    boundaries = np.unique(
        baseline_values.quantile(
            quantiles
        ).to_numpy()
    ).tolist()

    if len(boundaries) < 3:
        return None

    boundaries[0] = -np.inf
    boundaries[-1] = np.inf

    baseline_bins = pd.cut(
        baseline_values,
        bins=boundaries,
        include_lowest=True,
    )

    current_bins = pd.cut(
        current_values,
        bins=boundaries,
        include_lowest=True,
    )

    categories = (
        baseline_bins.cat.categories
    )

    baseline_distribution = (
        baseline_bins
        .value_counts(
            sort=False,
            normalize=True,
        )
        .reindex(
            categories,
            fill_value=0.0,
        )
    )

    current_distribution = (
        current_bins
        .value_counts(
            sort=False,
            normalize=True,
        )
        .reindex(
            categories,
            fill_value=0.0,
        )
    )

    epsilon = 0.0001

    expected = np.clip(
        baseline_distribution.to_numpy(
            dtype=float
        ),
        epsilon,
        None,
    )

    actual = np.clip(
        current_distribution.to_numpy(
            dtype=float
        ),
        epsilon,
        None,
    )

    psi = np.sum(
        (actual - expected)
        * np.log(
            actual / expected
        )
    )

    return float(psi)


def classify_psi(
    psi: float | None,
) -> str:
    if psi is None:
        return "Insufficient Data"

    if psi < 0.10:
        return "Stable"

    if psi < 0.25:
        return "Moderate Shift"

    return "Significant Shift"


def get_drift_report() -> pd.DataFrame:
    """
    Compare live/generated customers against the
    fixed historical baseline.

    This is input/prediction drift monitoring.
    It is not live model-performance monitoring.
    """

    df = load_monitoring_population()

    historical = df[
        df["customer_source"]
        == "Historical"
    ].copy()

    live = df[
        df["customer_source"]
        == "Generated"
    ].copy()

    if live.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "baseline_mean",
                "live_mean",
                "psi",
                "status",
            ]
        )

    features = [
        "tenure",
        "monthly_charges",
        "total_charges",
        "service_count",
        "churn_probability",
    ]

    rows = []

    for feature in features:
        psi = calculate_psi(
            historical[feature],
            live[feature],
        )

        rows.append(
            {
                "feature":
                    feature,

                "baseline_mean":
                    float(
                        historical[
                            feature
                        ].mean()
                    ),

                "live_mean":
                    float(
                        live[
                            feature
                        ].mean()
                    ),

                "psi":
                    psi,

                "status":
                    classify_psi(
                        psi
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def get_prediction_monitoring() -> dict[str, Any]:
    df = load_monitoring_population()

    total = len(df)

    historical = df[
        df["customer_source"]
        == "Historical"
    ]

    live = df[
        df["customer_source"]
        == "Generated"
    ]

    high_critical = int(
        df[
            "risk_segment"
        ]
        .isin(
            [
                "High",
                "Critical",
            ]
        )
        .sum()
    )

    predicted_churn = int(
        df[
            "predicted_churn"
        ].sum()
    )

    return {
        "total_scored":
            total,

        "historical_scored":
            len(historical),

        "live_scored":
            len(live),

        "predicted_churn":
            predicted_churn,

        "high_critical":
            high_critical,

        "avg_probability":
            float(
                df[
                    "churn_probability"
                ].mean()
            )
            if total
            else 0.0,

        "high_critical_pct":
            (
                high_critical
                / total
                if total
                else 0.0
            ),

        "expected_exposure":
            float(
                df[
                    "expected_monthly_revenue_at_risk"
                ].sum()
            )
            if total
            else 0.0,
    }


def get_model_governance() -> dict[str, Any]:
    """
    Report persisted production-model governance
    metadata without retraining or modifying it.
    """

    artifact_exists = (
        MODEL_PATH.exists()
    )

    if not artifact_exists:
        return {
            "artifact_available":
                False,

            "model_name":
                None,

            "threshold":
                None,

            "feature_count":
                0,

            "artifact_path":
                str(MODEL_PATH),

            "monitored_at":
                datetime.now(),
        }

    artifact = joblib.load(
        MODEL_PATH
    )

    return {
        "artifact_available":
            True,

        "model_name":
            artifact.get(
                "model_name"
            ),

        "threshold":
            float(
                artifact.get(
                    "threshold",
                    0.0,
                )
            ),

        "feature_count":
            len(
                artifact.get(
                    "features",
                    [],
                )
            ),

        "artifact_path":
            str(MODEL_PATH),

        "monitored_at":
            datetime.now(),
    }


def get_monitoring_snapshot() -> dict[str, Any]:
    return {
        "data_quality":
            get_data_quality_report(),

        "prediction_monitoring":
            get_prediction_monitoring(),

        "model_governance":
            get_model_governance(),
    }


if __name__ == "__main__":
    quality = (
        get_data_quality_report()
    )

    prediction = (
        get_prediction_monitoring()
    )

    governance = (
        get_model_governance()
    )

    drift = (
        get_drift_report()
    )

    print(
        "Production Monitoring Snapshot"
    )
    print(
        "------------------------------"
    )

    print(
        "Data quality: "
        f"{quality['passed_checks']}/"
        f"{quality['total_checks']} checks passed"
    )

    print(
        "Historical customers: "
        f"{quality['historical_rows']:,}"
    )

    print(
        "Live customers: "
        f"{quality['live_rows']:,}"
    )

    print(
        "Unified customers: "
        f"{quality['unified_rows']:,}"
    )

    print(
        "Scored customers: "
        f"{prediction['total_scored']:,}"
    )

    print(
        "Average churn probability: "
        f"{prediction['avg_probability']:.2%}"
    )

    print(
        "Model: "
        f"{governance['model_name']}"
    )

    print(
        "Threshold: "
        f"{governance['threshold']}"
    )

    print(
        "Features: "
        f"{governance['feature_count']}"
    )

    print(
        "\nDrift Monitoring"
    )
    print(
        "----------------"
    )

    if drift.empty:
        print(
            "No generated customers available "
            "for drift monitoring."
        )
    else:
        print(
            drift.to_string(
                index=False
            )
        )