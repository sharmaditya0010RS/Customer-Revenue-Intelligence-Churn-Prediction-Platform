from __future__ import annotations

import plotly.express as px
import streamlit as st

from app.components.charts import (
    apply_chart_theme,
    revenue_risk_chart,
)
from app.components.data import (
    load_customer_risk_data,
)
from app.components.layout import (
    format_currency,
    page_header,
    section_header,
)
from app.components.metrics import metric_row


def render() -> None:

    page_header(
        "Revenue Intelligence",
        (
            "Analyze customer revenue, "
            "probability-weighted exposure and "
            "commercial concentration."
        ),
        eyebrow="Financial Intelligence",
    )

    df = load_customer_risk_data()

    monthly_revenue = float(
        df["monthly_charges"].sum()
    )

    expected_risk = float(
        df[
            "expected_monthly_revenue_at_risk"
        ].sum()
    )

    avg_charge = float(
        df["monthly_charges"].mean()
    )

    exposure_rate = (
        expected_risk
        / monthly_revenue
        if monthly_revenue
        else 0
    )

    high_value_threshold = (
        df["monthly_charges"]
        .quantile(0.75)
    )

    high_value_customers = int(
        (
            df["monthly_charges"]
            >= high_value_threshold
        ).sum()
    )

    metric_row(
        [
            (
                "Monthly Revenue",
                format_currency(
                    monthly_revenue,
                    0,
                ),
                None,
            ),
            (
                "Expected Revenue at Risk",
                format_currency(
                    expected_risk,
                    0,
                ),
                None,
            ),
            (
                "Revenue Exposure Rate",
                f"{exposure_rate:.1%}",
                None,
            ),
            (
                "Average Monthly Charge",
                format_currency(
                    avg_charge,
                    2,
                ),
                None,
            ),
            (
                "High Value Customers",
                f"{high_value_customers:,}",
                None,
            ),
        ]
    )

    st.write("")

    left, right = st.columns(
        2,
        gap="large",
    )

    with left:

        section_header(
            "Revenue by Contract",
            (
                "Monthly portfolio revenue "
                "across contract types."
            ),
        )

        contract = (
            df.groupby(
                "contract",
                observed=True,
            )
            .agg(
                monthly_revenue=(
                    "monthly_charges",
                    "sum",
                ),
            )
            .reset_index()
            .sort_values(
                "monthly_revenue",
                ascending=False,
            )
        )

        figure = px.bar(
            contract,
            x="contract",
            y="monthly_revenue",
            text_auto=True,
            title=(
                "Monthly Revenue by Contract"
            ),
            labels={
                "contract":
                    "Contract",
                "monthly_revenue":
                    "Monthly Revenue",
            },
        )

        figure.update_yaxes(
            tickprefix="$"
        )

        st.plotly_chart(
            apply_chart_theme(
                figure
            ),
            use_container_width=True,
        )

    with right:

        section_header(
            "Revenue Exposure",
            (
                "Probability-weighted revenue "
                "exposure across ML risk bands."
            ),
        )

        st.plotly_chart(
            revenue_risk_chart(
                df
            ),
            use_container_width=True,
        )

    left, right = st.columns(
        2,
        gap="large",
    )

    with left:

        section_header(
            "Revenue by Customer Source"
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
                revenue=(
                    "monthly_charges",
                    "sum",
                ),
            )
            .reset_index()
        )

        figure = px.bar(
            source,
            x="customer_source",
            y="revenue",
            title=(
                "Historical vs Generated Revenue"
            ),
            labels={
                "customer_source":
                    "Customer Source",
                "revenue":
                    "Monthly Revenue",
            },
        )

        figure.update_yaxes(
            tickprefix="$"
        )

        st.plotly_chart(
            apply_chart_theme(
                figure
            ),
            use_container_width=True,
        )

    with right:

        section_header(
            "Customer Value Distribution"
        )

        figure = px.histogram(
            df,
            x="monthly_charges",
            nbins=35,
            title=(
                "Monthly Charge Distribution"
            ),
            labels={
                "monthly_charges":
                    "Monthly Charge",
            },
        )

        figure.update_xaxes(
            tickprefix="$"
        )

        st.plotly_chart(
            apply_chart_theme(
                figure
            ),
            use_container_width=True,
        )

    st.divider()

    section_header(
        "Revenue Risk Portfolio",
        (
            "Highest probability-weighted revenue "
            "exposures requiring commercial attention."
        ),
    )

    top_risk = (
        df.sort_values(
            "expected_monthly_revenue_at_risk",
            ascending=False,
        )
        .head(25)
        .copy()
    )

    st.dataframe(
        top_risk[
            [
                "customer_id",
                "customer_source",
                "monthly_charges",
                "churn_probability",
                "risk_segment",
                "retention_priority",
                "expected_monthly_revenue_at_risk",
                "contract",
                "tenure",
            ]
        ],
        hide_index=True,
        use_container_width=True,
        column_config={
            "monthly_charges":
                st.column_config.NumberColumn(
                    "Monthly Charge",
                    format="$%.2f",
                ),

            "churn_probability":
                st.column_config.ProgressColumn(
                    "Churn Probability",
                    min_value=0.0,
                    max_value=1.0,
                    format="percent",
                ),

            "expected_monthly_revenue_at_risk":
                st.column_config.NumberColumn(
                    "Expected Exposure",
                    format="$%.2f",
                ),
        },
    )