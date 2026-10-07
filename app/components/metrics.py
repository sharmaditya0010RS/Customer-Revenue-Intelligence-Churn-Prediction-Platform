from __future__ import annotations

from typing import Any

import streamlit as st

from app.components.layout import (
    format_currency,
    format_number,
)


def platform_kpi_row(
    kpis: dict[str, Any],
) -> None:
    """Standard six-card platform KPI row."""

    columns = st.columns(
        6,
        gap="medium",
    )

    columns[0].metric(
        "Total Customers",
        format_number(
            kpis[
                "total_customers"
            ]
        ),
    )

    columns[1].metric(
        "Generated",
        format_number(
            kpis[
                "generated_customers"
            ]
        ),
    )

    columns[2].metric(
        "Predicted Churn",
        format_number(
            kpis[
                "predicted_churn_customers"
            ]
        ),
    )

    columns[3].metric(
        "High + Critical",
        format_number(
            kpis[
                "high_critical_risk_customers"
            ]
        ),
    )

    columns[4].metric(
        "Monthly Revenue",
        format_currency(
            kpis[
                "total_monthly_revenue"
            ]
        ),
    )

    columns[5].metric(
        "Revenue at Risk",
        format_currency(
            kpis[
                "expected_monthly_revenue_at_risk"
            ]
        ),
    )


def metric_row(
    metrics: list[
        tuple[
            str,
            str,
            str | None,
        ]
    ],
) -> None:
    """
    Render arbitrary native Streamlit metric cards.

    Each tuple:
        label,
        value,
        optional delta
    """

    if not metrics:
        return

    columns = st.columns(
        len(metrics),
        gap="medium",
    )

    for column, metric in zip(
        columns,
        metrics,
        strict=True,
    ):

        label, value, delta = (
            metric
        )

        column.metric(
            label,
            value,
            delta=delta,
        )