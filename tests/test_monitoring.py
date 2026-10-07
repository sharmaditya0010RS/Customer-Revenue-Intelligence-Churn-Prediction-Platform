from src.monitoring import (
    calculate_psi,
    classify_psi,
    get_data_quality_report,
    get_drift_report,
    get_model_governance,
    get_prediction_monitoring,
)

import pandas as pd


def test_data_quality_report():
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
            "historical_rows"
        ]
        == report[
            "historical_unique"
        ]
    )

    assert (
        report[
            "historical_predictions"
        ]
        == 7043
    )

    assert (
        report[
            "live_rows"
        ]
        == report[
            "live_unique"
        ]
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

    assert (
        report[
            "historical_live_overlap"
        ]
        == 0
    )

    assert (
        report[
            "passed_checks"
        ]
        == report[
            "total_checks"
        ]
    )


def test_prediction_monitoring():
    report = (
        get_prediction_monitoring()
    )

    assert (
        report[
            "total_scored"
        ]
        >= 7043
    )

    assert (
        report[
            "historical_scored"
        ]
        == 7043
    )

    assert (
        report[
            "live_scored"
        ]
        >= 0
    )

    assert (
        0.0
        <= report[
            "avg_probability"
        ]
        <= 1.0
    )

    assert (
        0.0
        <= report[
            "high_critical_pct"
        ]
        <= 1.0
    )

    assert (
        report[
            "expected_exposure"
        ]
        >= 0.0
    )


def test_model_governance():
    report = (
        get_model_governance()
    )

    assert (
        report[
            "artifact_available"
        ]
        is True
    )

    assert (
        report[
            "model_name"
        ]
        == "xgboost"
    )

    assert (
        report[
            "threshold"
        ]
        == 0.34
    )

    assert (
        report[
            "feature_count"
        ]
        == 22
    )


def test_psi_identical_distribution():
    baseline = pd.Series(
        list(
            range(
                1,
                101,
            )
        )
    )

    current = baseline.copy()

    psi = calculate_psi(
        baseline,
        current,
    )

    assert psi is not None
    assert psi < 0.01

    assert (
        classify_psi(
            psi
        )
        == "Stable"
    )


def test_psi_shifted_distribution():
    baseline = pd.Series(
        list(
            range(
                1,
                101,
            )
        )
    )

    current = pd.Series(
        list(
            range(
                101,
                201,
            )
        )
    )

    psi = calculate_psi(
        baseline,
        current,
    )

    assert psi is not None

    assert (
        classify_psi(
            psi
        )
        == "Significant Shift"
    )


def test_drift_report_structure():
    report = (
        get_drift_report()
    )

    expected_columns = {
        "feature",
        "baseline_mean",
        "live_mean",
        "psi",
        "status",
    }

    assert (
        expected_columns
        .issubset(
            report.columns
        )
    )