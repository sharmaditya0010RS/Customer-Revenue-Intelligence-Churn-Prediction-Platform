from __future__ import annotations

import json
from typing import Any

import streamlit as st

from src.config import METRICS_DIR


EVALUATION_FILE = (
    METRICS_DIR
    / "model_evaluation.json"
)


def load_model_evaluation() -> dict[str, Any]:
    """Load persisted held-out model evaluation."""

    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            (
                "Model evaluation report not found: "
                f"{EVALUATION_FILE}"
            )
        )

    with EVALUATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        report = json.load(file)

    if not isinstance(
        report,
        dict,
    ):
        raise ValueError(
            "Model evaluation report must contain a JSON object."
        )

    return report


def _search_metric(
    data: Any,
    possible_keys: tuple[str, ...],
) -> float | None:
    """
    Recursively locate a numeric metric.

    This keeps the UI independent from minor report
    nesting differences without changing the ML pipeline.
    """

    if isinstance(
        data,
        dict,
    ):
        for key in possible_keys:
            if key in data:
                value = data[key]

                if isinstance(
                    value,
                    (int, float),
                ):
                    return float(value)

        for value in data.values():
            found = _search_metric(
                value,
                possible_keys,
            )

            if found is not None:
                return found

    return None


def render_model_performance() -> None:
    """Render held-out test metrics from evaluation JSON."""

    try:
        report = (
            load_model_evaluation()
        )

    except (
        FileNotFoundError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        st.warning(
            "Model evaluation metrics are currently unavailable."
        )

        st.caption(
            str(exc)
        )

        return

    roc_auc = _search_metric(
        report,
        (
            "roc_auc",
            "test_roc_auc",
            "roc_auc_score",
        ),
    )

    precision = _search_metric(
        report,
        (
            "precision",
            "test_precision",
        ),
    )

    recall = _search_metric(
        report,
        (
            "recall",
            "test_recall",
        ),
    )

    f1 = _search_metric(
        report,
        (
            "f1",
            "f1_score",
            "test_f1",
        ),
    )

    columns = st.columns(
        4,
        gap="medium",
    )

    columns[0].metric(
        "Test ROC-AUC",
        (
            f"{roc_auc:.3f}"
            if roc_auc is not None
            else "—"
        ),
    )

    columns[1].metric(
        "Precision",
        (
            f"{precision:.1%}"
            if precision is not None
            else "—"
        ),
    )

    columns[2].metric(
        "Recall",
        (
            f"{recall:.1%}"
            if recall is not None
            else "—"
        ),
    )

    columns[3].metric(
        "F1 Score",
        (
            f"{f1:.3f}"
            if f1 is not None
            else "—"
        ),
    )

    if any(
        metric is None
        for metric in (
            roc_auc,
            precision,
            recall,
            f1,
        )
    ):
        st.caption(
            (
                "Some metrics could not be located in "
                "the persisted evaluation report."
            )
        )