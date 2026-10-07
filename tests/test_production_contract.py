from pathlib import Path

import joblib
import pandas as pd

from src.config import MODEL_DIR
from src.monitoring import (
    get_data_quality_report,
    get_model_governance,
)
from src.scoring import (
    load_production_model,
)


MODEL_PATH = (
    MODEL_DIR
    / "churn_model.joblib"
)


EXPECTED_MODEL_FEATURES = [
    "senior_citizen",
    "tenure",
    "monthly_charges",
    "total_charges",
    "avg_revenue_per_tenure_month",
    "service_count",
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


def test_production_model_artifact_exists():
    assert MODEL_PATH.exists()
    assert MODEL_PATH.is_file()


def test_production_model_contract():
    package = (
        load_production_model()
    )

    assert (
        package[
            "model_name"
        ]
        == "xgboost"
    )

    assert (
        float(
            package[
                "threshold"
            ]
        )
        == 0.34
    )

    assert (
        package[
            "features"
        ]
        == EXPECTED_MODEL_FEATURES
    )


def test_persisted_artifact_contract():
    artifact = joblib.load(
        MODEL_PATH
    )

    assert isinstance(
        artifact,
        dict,
    )

    assert {
        "pipeline",
        "threshold",
        "model_name",
        "features",
    }.issubset(
        artifact.keys()
    )

    assert (
        artifact[
            "features"
        ]
        == EXPECTED_MODEL_FEATURES
    )


def test_production_pipeline_contract():
    package = (
        load_production_model()
    )

    pipeline = (
        package[
            "pipeline"
        ]
    )

    assert hasattr(
        pipeline,
        "predict_proba",
    )

    assert hasattr(
        pipeline,
        "named_steps",
    )

    assert (
        "preprocessor"
        in pipeline.named_steps
    )

    assert (
        "model"
        in pipeline.named_steps
    )


def test_model_governance_matches_artifact():
    package = (
        load_production_model()
    )

    governance = (
        get_model_governance()
    )

    assert (
        governance[
            "artifact_available"
        ]
        is True
    )

    assert (
        governance[
            "model_name"
        ]
        == package[
            "model_name"
        ]
    )

    assert (
        governance[
            "threshold"
        ]
        == float(
            package[
                "threshold"
            ]
        )
    )

    assert (
        governance[
            "feature_count"
        ]
        == len(
            package[
                "features"
            ]
        )
    )


def test_historical_population_contract():
    report = (
        get_data_quality_report()
    )

    assert (
        report[
            "historical_rows"
        ]
        == 7043
    )

    assert (
        report[
            "historical_unique"
        ]
        == 7043
    )

    assert (
        report[
            "historical_predictions"
        ]
        == 7043
    )


def test_live_population_isolation_contract():
    report = (
        get_data_quality_report()
    )

    assert (
        report[
            "historical_live_overlap"
        ]
        == 0
    )

    assert (
        report[
            "unified_rows"
        ]
        == (
            report[
                "historical_rows"
            ]
            + report[
                "live_rows"
            ]
        )
    )


def test_prediction_coverage_contract():
    report = (
        get_data_quality_report()
    )

    assert (
        report[
            "live_predictions"
        ]
        == report[
            "live_rows"
        ]
    )

    assert (
        report[
            "unified_risk_rows"
        ]
        == (
            report[
                "historical_predictions"
            ]
            + report[
                "live_predictions"
            ]
        )
    )