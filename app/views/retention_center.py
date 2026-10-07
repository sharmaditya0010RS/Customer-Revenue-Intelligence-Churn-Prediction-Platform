from __future__ import annotations

import plotly.express as px
import streamlit as st

from app.components.charts import (
    apply_chart_theme,
)
from app.components.data import (
    load_customer_risk_data,
)
from app.components.layout import (
    format_currency,
    page_header,
    section_header,
)
from app.components.metrics import (
    metric_row,
)


RISK_ORDER = [
    "Low",
    "Medium",
    "High",
    "Critical",
]


def render() -> None:

    page_header(
        "Retention Center",
        (
            "Convert ML risk into a prioritized "
            "customer retention action queue."
        ),
        eyebrow="Retention Operations",
    )

    df = load_customer_risk_data()

    st.caption(
        "FILTER CUSTOMER PORTFOLIO"
    )

    filter_1, filter_2, filter_3 = (
        st.columns(3)
    )

    with filter_1:

        selected_risk = st.multiselect(
            "Risk Segment",
            RISK_ORDER,
            default=[
                "High",
                "Critical",
            ],
        )

    sources = sorted(
        df[
            "customer_source"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    with filter_2:

        selected_sources = (
            st.multiselect(
                "Customer Source",
                sources,
                default=sources,
            )
        )

    contracts = sorted(
        df[
            "contract"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    with filter_3:

        selected_contracts = (
            st.multiselect(
                "Contract",
                contracts,
                default=contracts,
            )
        )

    filtered = df.loc[
        df[
            "risk_segment"
        ].isin(
            selected_risk
        )
        & df[
            "customer_source"
        ].isin(
            selected_sources
        )
        & df[
            "contract"
        ].isin(
            selected_contracts
        )
    ].copy()

    filtered = (
        filtered.sort_values(
            [
                "expected_monthly_revenue_at_risk",
                "churn_probability",
            ],
            ascending=False,
        )
    )

    selected_customers = len(
        filtered
    )

    monthly_revenue = float(
        filtered[
            "monthly_charges"
        ].sum()
    )

    revenue_exposure = float(
        filtered[
            "expected_monthly_revenue_at_risk"
        ].sum()
    )

    p1_customers = int(
        filtered[
            "retention_priority"
        ]
        .astype(str)
        .str.startswith(
            "P1"
        )
        .sum()
    )

    metric_row(
        [
            (
                "Selected Customers",
                f"{selected_customers:,}",
                None,
            ),
            (
                "P1 Immediate Action",
                f"{p1_customers:,}",
                None,
            ),
            (
                "Monthly Revenue",
                format_currency(
                    monthly_revenue,
                    0,
                ),
                None,
            ),
            (
                "Revenue Exposure",
                format_currency(
                    revenue_exposure,
                    0,
                ),
                None,
            ),
        ]
    )

    st.write("")

    if filtered.empty:

        st.info(
            "No customers match the selected filters."
        )

        return

    left, right = st.columns(
        2,
        gap="large",
    )

    with left:

        section_header(
            "Retention Priority Queue"
        )

        priority_summary = (
            filtered.groupby(
                "retention_priority",
                observed=True,
            )
            .agg(
                customers=(
                    "customer_id",
                    "count",
                ),
            )
            .reset_index()
            .sort_values(
                "customers",
                ascending=False,
            )
        )

        figure = px.bar(
            priority_summary,
            x="retention_priority",
            y="customers",
            text_auto=True,
            title=(
                "Customers by Retention Priority"
            ),
            labels={
                "retention_priority":
                    "Retention Priority",
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

    with right:

        section_header(
            "Revenue Exposure by Priority"
        )

        revenue_summary = (
            filtered.groupby(
                "retention_priority",
                observed=True,
            )
            .agg(
                exposure=(
                    "expected_monthly_revenue_at_risk",
                    "sum",
                ),
            )
            .reset_index()
            .sort_values(
                "exposure",
                ascending=False,
            )
        )

        figure = px.bar(
            revenue_summary,
            x="retention_priority",
            y="exposure",
            title=(
                "Revenue Exposure by Priority"
            ),
            labels={
                "retention_priority":
                    "Retention Priority",
                "exposure":
                    "Revenue Exposure",
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
        "Retention Action Queue",
        (
            "Operational customer list ranked by "
            "expected monthly revenue exposure."
        ),
    )

    action_queue = filtered[
        [
            "customer_id",
            "customer_source",
            "retention_priority",
            "risk_segment",
            "churn_probability",
            "monthly_charges",
            "expected_monthly_revenue_at_risk",
            "contract",
            "tenure",
            "internet_service",
            "payment_method",
        ]
    ].copy()

    st.dataframe(
        action_queue,
        hide_index=True,
        use_container_width=True,
        height=520,
        column_config={
            "churn_probability":
                st.column_config.ProgressColumn(
                    "Churn Probability",
                    min_value=0,
                    max_value=1,
                    format="percent",
                ),

            "monthly_charges":
                st.column_config.NumberColumn(
                    "Monthly Charge",
                    format="$%.2f",
                ),

            "expected_monthly_revenue_at_risk":
                st.column_config.NumberColumn(
                    "Revenue Exposure",
                    format="$%.2f",
                ),
        },
    )

    csv = (
        action_queue
        .to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )

    st.download_button(
        "Download Retention Action Queue",
        data=csv,
        file_name=(
            "retention_action_queue.csv"
        ),
        mime="text/csv",
        use_container_width=False,
    )