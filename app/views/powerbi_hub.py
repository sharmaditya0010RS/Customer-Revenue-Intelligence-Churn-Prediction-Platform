from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.components.data import (
    load_platform_kpis,
)
from app.components.layout import (
    page_header,
    section_header,
)
from app.components.metrics import (
    platform_kpi_row,
)


BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
)

PBIX_FILE = (
    BASE_DIR
    / "powerbi"
    / "dashboard.pbix"
)


def render() -> None:

    page_header(
        "Power BI Hub",
        (
            "Bridge the operational PostgreSQL "
            "platform with management-grade "
            "Power BI reporting."
        ),
        eyebrow="Business Intelligence",
    )

    kpis = load_platform_kpis()

    platform_kpi_row(
        kpis
    )

    st.write("")

    left, right = st.columns(
        [1.35, 1],
        gap="large",
    )

    with left:

        section_header(
            "Power BI Integration",
            (
                "The Power BI semantic layer uses "
                "PostgreSQL in Import mode."
            ),
        )

        st.info(
            (
                "Generated customers are immediately "
                "available in PostgreSQL and Streamlit. "
                "Power BI receives them after a dataset "
                "refresh."
            )
        )

        st.subheader(
            "Recommended Power BI Views"
        )

        st.code(
            """
analytics.vw_all_customer_risk
analytics.vw_all_customers
analytics.vw_platform_kpis
            """.strip(),
            language="sql",
        )

    with right:

        section_header(
            "Report Status"
        )

        if PBIX_FILE.exists():

            st.success(
                "● Power BI report file detected"
            )

            st.caption(
                str(
                    PBIX_FILE
                )
            )

        else:

            st.warning(
                "Power BI report file not detected."
            )

        st.info(
            "● Connection Mode · Import"
        )

        st.success(
            "● PostgreSQL Source · Available"
        )

        st.warning(
            "● Refresh · Manual / Scheduled"
        )

    st.divider()

    section_header(
        "Refresh Workflow",
        (
            "How permanent generated customers "
            "flow into the BI reporting layer."
        ),
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "01",
        "Generate",
    )

    col1.caption(
        "Streamlit creates a permanent customer."
    )

    col2.metric(
        "02",
        "Score",
    )

    col2.caption(
        "Production XGBoost calculates churn risk."
    )

    col3.metric(
        "03",
        "Persist",
    )

    col3.caption(
        "Customer and prediction enter PostgreSQL."
    )

    col4.metric(
        "04",
        "Refresh",
    )

    col4.caption(
        "Power BI imports the latest platform state."
    )

    st.divider()

    section_header(
        "Current Database Snapshot"
    )

    st.caption(
        (
            "These values represent PostgreSQL now. "
            "After Power BI refresh, the report should "
            "reconcile with this snapshot."
        )
    )

    snapshot = {
        "Total Customers":
            int(
                kpis[
                    "total_customers"
                ]
            ),

        "Historical Customers":
            int(
                kpis[
                    "historical_customers"
                ]
            ),

        "Generated Customers":
            int(
                kpis[
                    "generated_customers"
                ]
            ),

        "Predicted Churn":
            int(
                kpis[
                    "predicted_churn_customers"
                ]
            ),

        "Critical Risk":
            int(
                kpis[
                    "critical_risk_customers"
                ]
            ),

        "High + Critical":
            int(
                kpis[
                    "high_critical_risk_customers"
                ]
            ),
    }

    st.json(
        snapshot
    )