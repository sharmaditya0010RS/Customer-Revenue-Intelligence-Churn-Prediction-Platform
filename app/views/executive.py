from __future__ import annotations

import plotly.express as px
import streamlit as st

from app.components.charts import (
    apply_chart_theme,
    contract_risk_chart,
    revenue_risk_chart,
    risk_distribution_chart,
)
from app.components.data import (
    load_customer_risk_data,
    load_platform_kpis,
)
from app.components.insights import (
    build_executive_insights,
)
from app.components.layout import (
    page_header,
    section_header,
)
from app.components.metrics import (
    platform_kpi_row,
)


def render() -> None:
    page_header(
        "Executive Overview",
        (
            "Management view of customer scale, "
            "revenue performance, churn exposure "
            "and portfolio risk."
        ),
        eyebrow="Executive Intelligence",
    )

    kpis = (
        load_platform_kpis()
    )

    df = (
        load_customer_risk_data()
    )

    platform_kpi_row(
        kpis
    )

    st.write("")

    section_header(
        "Executive Briefing",
        (
            "Automatically calculated findings from "
            "the current customer portfolio."
        ),
    )

    insights = (
        build_executive_insights(
            df
        )
    )

    if not insights:
        st.info(
            "No executive insights are currently available."
        )

    else:
        insight_columns = (
            st.columns(2)
        )

        for index, insight in enumerate(
            insights
        ):
            with insight_columns[
                index % 2
            ]:
                st.info(
                    insight
                )

    st.divider()

    left, right = (
        st.columns(
            2,
            gap="large",
        )
    )

    with left:
        section_header(
            "Customer Risk Mix",
            (
                "Current ML segmentation across "
                "the unified customer portfolio."
            ),
        )

        st.plotly_chart(
            risk_distribution_chart(
                df
            ),
            use_container_width=True,
        )

    with right:
        section_header(
            "Contract Risk",
            (
                "Average predicted churn propensity "
                "across contract categories."
            ),
        )

        st.plotly_chart(
            contract_risk_chart(
                df
            ),
            use_container_width=True,
        )

    left, right = (
        st.columns(
            2,
            gap="large",
        )
    )

    with left:
        section_header(
            "Revenue Exposure",
            (
                "Probability-weighted monthly "
                "revenue exposure by risk segment."
            ),
        )

        st.plotly_chart(
            revenue_risk_chart(
                df
            ),
            use_container_width=True,
        )

    with right:
        section_header(
            "Portfolio Composition",
            (
                "Historical and permanently generated "
                "customer population."
            ),
        )

        source = (
            df.groupby(
                "customer_source",
                observed=True,
            )
            .agg(
                customers=(
                    "customer_id",
                    "count",
                ),
                monthly_revenue=(
                    "monthly_charges",
                    "sum",
                ),
            )
            .reset_index()
        )

        figure = px.bar(
            source,
            x="customer_source",
            y="customers",
            text_auto=True,
            title=(
                "Customers by Platform Source"
            ),
            labels={
                "customer_source":
                    "Customer Source",
                "customers":
                    "Customers",
            },
        )

        st.plotly_chart(
            apply_chart_theme(
                figure
            ),
            use_container_width=True,
        )

    st.divider()

    section_header(
        "Commercial Risk Matrix",
        (
            "Identify customers combining elevated "
            "churn propensity with significant "
            "monthly commercial value."
        ),
    )

    figure = px.scatter(
        df,
        x="monthly_charges",
        y="churn_probability",
        color="risk_segment",
        size="expected_monthly_revenue_at_risk",
        hover_name="customer_id",
        hover_data=[
            "customer_source",
            "contract",
            "tenure",
            "retention_priority",
        ],
        category_orders={
            "risk_segment": [
                "Low",
                "Medium",
                "High",
                "Critical",
            ],
        },
        title=(
            "Customer Value vs Churn Propensity"
        ),
        labels={
            "monthly_charges":
                "Monthly Customer Value",
            "churn_probability":
                "Churn Probability",
            "risk_segment":
                "Risk Segment",
        },
    )

    figure.update_xaxes(
        tickprefix="$"
    )

    figure.update_yaxes(
        tickformat=".0%"
    )

    st.plotly_chart(
        apply_chart_theme(
            figure,
            height=520,
        ),
        use_container_width=True,
    )