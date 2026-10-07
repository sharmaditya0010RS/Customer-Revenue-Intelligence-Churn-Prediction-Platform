from __future__ import annotations

import plotly.express as px
import streamlit as st

from app.components.charts import apply_chart_theme
from app.components.data import load_historical_customers
from app.components.layout import (
    page_header,
    section_header,
)
from app.components.metrics import metric_row


def render() -> None:

    page_header(
        "Churn Analytics",
        (
            "Understand observed customer attrition, "
            "behavioral patterns and the strongest "
            "historical churn signals."
        ),
        eyebrow="Customer Analytics",
    )

    df = load_historical_customers()

    total_customers = len(df)
    churned_customers = int(
        df["churn_flag"].sum()
    )

    retained_customers = (
        total_customers
        - churned_customers
    )

    churn_rate = (
        churned_customers
        / total_customers
        if total_customers
        else 0
    )

    avg_churn_tenure = (
        df.loc[
            df["churn_flag"] == 1,
            "tenure",
        ].mean()
    )

    metric_row(
        [
            (
                "Historical Customers",
                f"{total_customers:,}",
                None,
            ),
            (
                "Churned Customers",
                f"{churned_customers:,}",
                None,
            ),
            (
                "Retained Customers",
                f"{retained_customers:,}",
                None,
            ),
            (
                "Observed Churn Rate",
                f"{churn_rate:.2%}",
                None,
            ),
            (
                "Avg Churn Tenure",
                f"{avg_churn_tenure:.1f} mo",
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
            "Contract Risk",
            (
                "Observed churn rate by "
                "customer contract type."
            ),
        )

        contract = (
            df.groupby(
                "contract",
                observed=True,
            )
            .agg(
                customers=(
                    "customer_id",
                    "count",
                ),
                churn_rate=(
                    "churn_flag",
                    "mean",
                ),
            )
            .reset_index()
        )

        figure = px.bar(
            contract,
            x="contract",
            y="churn_rate",
            text_auto=True,
            title=(
                "Churn Rate by Contract"
            ),
            labels={
                "contract":
                    "Contract",
                "churn_rate":
                    "Churn Rate",
            },
        )

        figure.update_yaxes(
            tickformat=".0%"
        )

        st.plotly_chart(
            apply_chart_theme(
                figure
            ),
            use_container_width=True,
        )

    with right:

        section_header(
            "Internet Service Risk",
            (
                "Attrition patterns across "
                "internet service categories."
            ),
        )

        internet = (
            df.groupby(
                "internet_service",
                observed=True,
            )
            .agg(
                churn_rate=(
                    "churn_flag",
                    "mean",
                ),
            )
            .reset_index()
        )

        figure = px.bar(
            internet,
            x="internet_service",
            y="churn_rate",
            text_auto=True,
            title=(
                "Churn Rate by Internet Service"
            ),
            labels={
                "internet_service":
                    "Internet Service",
                "churn_rate":
                    "Churn Rate",
            },
        )

        figure.update_yaxes(
            tickformat=".0%"
        )

        st.plotly_chart(
            apply_chart_theme(
                figure
            ),
            use_container_width=True,
        )

    left, right = st.columns(
        2,
        gap="large",
    )

    with left:

        section_header(
            "Customer Lifecycle",
            (
                "Tenure distribution for retained "
                "and churned customers."
            ),
        )

        figure = px.box(
            df,
            x="churn",
            y="tenure",
            points=False,
            title=(
                "Tenure Distribution by Outcome"
            ),
            labels={
                "churn":
                    "Customer Outcome",
                "tenure":
                    "Tenure (Months)",
            },
        )

        st.plotly_chart(
            apply_chart_theme(
                figure
            ),
            use_container_width=True,
        )

    with right:

        section_header(
            "Monthly Charge Pressure",
            (
                "Compare monthly pricing across "
                "retained and churned customers."
            ),
        )

        figure = px.box(
            df,
            x="churn",
            y="monthly_charges",
            points=False,
            title=(
                "Monthly Charges by Outcome"
            ),
            labels={
                "churn":
                    "Customer Outcome",
                "monthly_charges":
                    "Monthly Charges",
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

    st.divider()

    section_header(
        "Payment Method Exposure",
        (
            "Observed churn behavior by "
            "customer payment method."
        ),
    )

    payment = (
        df.groupby(
            "payment_method",
            observed=True,
        )
        .agg(
            customers=(
                "customer_id",
                "count",
            ),
            churn_rate=(
                "churn_flag",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "churn_rate",
            ascending=False,
        )
    )

    figure = px.bar(
        payment,
        x="payment_method",
        y="churn_rate",
        text_auto=True,
        title=(
            "Churn Rate by Payment Method"
        ),
        labels={
            "payment_method":
                "Payment Method",
            "churn_rate":
                "Churn Rate",
        },
    )

    figure.update_yaxes(
        tickformat=".0%"
    )

    st.plotly_chart(
        apply_chart_theme(
            figure,
            height=430,
        ),
        use_container_width=True,
    )