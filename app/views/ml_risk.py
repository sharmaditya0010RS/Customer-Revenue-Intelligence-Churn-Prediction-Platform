from __future__ import annotations

import plotly.express as px
import streamlit as st

from app.components.charts import (
    apply_chart_theme,
    contract_risk_chart,
    probability_distribution_chart,
    risk_distribution_chart,
)
from app.components.data import (
    load_customer_risk_data,
)
from app.components.layout import (
    page_header,
    section_header,
)
from app.components.metrics import (
    metric_row,
)
from app.components.model_metrics import (
    render_model_performance,
)
from src.explainability import (
    get_global_feature_importance,
    get_model_metadata,
)
from src.scoring import (
    load_production_model,
)


def render() -> None:
    page_header(
        "ML Risk Intelligence",
        (
            "Production churn propensity, model "
            "performance, explainability, customer "
            "segmentation and portfolio-level "
            "predictive intelligence."
        ),
        eyebrow="Machine Learning",
    )

    df = load_customer_risk_data()
    package = load_production_model()
    metadata = get_model_metadata()

    predicted_churn = int(
        df["predicted_churn"].sum()
    )

    critical = int(
        (
            df["risk_segment"]
            == "Critical"
        ).sum()
    )

    high_critical = int(
        df["risk_segment"]
        .isin(
            [
                "High",
                "Critical",
            ]
        )
        .sum()
    )

    avg_probability = float(
        df["churn_probability"].mean()
    )

    metric_row(
        [
            (
                "Production Model",
                str(
                    package["model_name"]
                ).upper(),
                None,
            ),
            (
                "Decision Threshold",
                f"{float(package['threshold']):.2f}",
                None,
            ),
            (
                "Predicted Churn",
                f"{predicted_churn:,}",
                None,
            ),
            (
                "High + Critical",
                f"{high_critical:,}",
                None,
            ),
            (
                "Critical Risk",
                f"{critical:,}",
                None,
            ),
            (
                "Avg Probability",
                f"{avg_probability:.1%}",
                None,
            ),
        ]
    )

    st.write("")

    section_header(
        "Validated Model Performance",
        (
            "Held-out test performance from the "
            "persisted production model evaluation."
        ),
    )

    render_model_performance()

    st.caption(
        (
            "The production model was selected using "
            "the predefined validation ROC-AUC criterion. "
            "The held-out test set remained untouched "
            "until model and threshold selection were complete."
        )
    )

    st.divider()

    section_header(
        "Global Model Drivers",
        (
            "Features with the strongest influence "
            "inside the production XGBoost model."
        ),
    )

    try:
        importance = (
            get_global_feature_importance(
                top_n=15
            )
            .copy()
        )

        left, right = st.columns(
            [1.7, 1],
            gap="large",
        )

        with left:
            chart_data = (
                importance
                .sort_values(
                    "importance",
                    ascending=True,
                )
                .copy()
            )

            figure = px.bar(
                chart_data,
                x="importance",
                y="feature_label",
                orientation="h",
                title=(
                    "Top Production Model Drivers"
                ),
                labels={
                    "importance":
                        "XGBoost Feature Importance",
                    "feature_label":
                        "Model Feature",
                },
            )

            figure.update_layout(
                yaxis_title=None,
            )

            st.plotly_chart(
                apply_chart_theme(
                    figure,
                    height=560,
                ),
                use_container_width=True,
            )

        with right:
            st.markdown(
                "#### Model interpretation"
            )

            st.write(
                (
                    "These drivers describe which "
                    "transformed features the production "
                    "XGBoost model relies on most when "
                    "separating churn risk."
                )
            )

            top_driver = (
                importance.iloc[0]
            )

            st.metric(
                "Strongest Model Driver",
                str(
                    top_driver[
                        "feature_label"
                    ]
                ),
            )

            st.metric(
                "Raw Model Features",
                int(
                    metadata[
                        "raw_feature_count"
                    ]
                ),
            )

            st.metric(
                "Production Threshold",
                f"{float(metadata['threshold']):.2f}",
            )

            st.warning(
                (
                    "Feature importance represents "
                    "predictive model influence, not "
                    "causal impact. A high-importance "
                    "feature should not be interpreted "
                    "as proof that changing that feature "
                    "will cause churn to increase or decrease."
                )
            )

        with st.expander(
            "View model-driver details",
            expanded=False,
        ):
            display_importance = (
                importance[
                    [
                        "feature_label",
                        "importance",
                        "relative_importance_pct",
                    ]
                ]
                .copy()
            )

            display_importance.columns = [
                "Feature",
                "Importance",
                "Relative Importance",
            ]

            st.dataframe(
                display_importance,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Importance":
                        st.column_config.NumberColumn(
                            "Importance",
                            format="%.4f",
                        ),
                    "Relative Importance":
                        st.column_config.ProgressColumn(
                            "Relative Importance",
                            min_value=0.0,
                            max_value=100.0,
                            format="%.1f%%",
                        ),
                },
            )

        st.caption(
            (
                "Global importance is calculated from "
                "the persisted production XGBoost model. "
                "Categorical variables are represented "
                "through their one-hot encoded model features."
            )
        )

    except Exception as exc:
        st.warning(
            (
                "Global model explainability could not "
                "be rendered. The production scoring "
                "pipeline remains available."
            )
        )

        with st.expander(
            "Explainability diagnostic"
        ):
            st.code(
                str(exc)
            )

    st.divider()

    left, right = st.columns(
        2,
        gap="large",
    )

    with left:
        section_header(
            "Portfolio Segmentation",
            (
                "Customer distribution across "
                "production risk bands."
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
            "Probability Landscape",
            (
                "Distribution of churn propensity "
                "across the scored population."
            ),
        )

        st.plotly_chart(
            probability_distribution_chart(
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
            "Contract Risk Profile",
            (
                "Average churn propensity across "
                "customer contract categories."
            ),
        )

        st.plotly_chart(
            contract_risk_chart(
                df
            ),
            use_container_width=True,
        )

    with right:
        section_header(
            "Historical vs Generated Risk",
            (
                "Compare model risk across the "
                "fixed historical population and "
                "permanent generated customers."
            ),
        )

        source_risk = (
            df.groupby(
                "customer_source",
                observed=True,
            )
            .agg(
                customers=(
                    "customer_id",
                    "count",
                ),
                avg_probability=(
                    "churn_probability",
                    "mean",
                ),
            )
            .reset_index()
        )

        figure = px.bar(
            source_risk,
            x="customer_source",
            y="avg_probability",
            text_auto=True,
            title=(
                "Average Risk by Customer Source"
            ),
            labels={
                "customer_source":
                    "Customer Source",
                "avg_probability":
                    "Average Churn Probability",
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

    st.divider()

    section_header(
        "Customer Risk Landscape",
        (
            "Analyze customer lifecycle, monthly "
            "commercial value and churn propensity "
            "in one model-driven view."
        ),
    )

    figure = px.scatter(
        df,
        x="tenure",
        y="churn_probability",
        size="monthly_charges",
        color="risk_segment",
        hover_name="customer_id",
        hover_data=[
            "customer_source",
            "contract",
            "monthly_charges",
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
            "Tenure vs Churn Probability"
        ),
        labels={
            "tenure":
                "Tenure (Months)",
            "churn_probability":
                "Churn Probability",
            "risk_segment":
                "Risk Segment",
        },
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

    st.divider()

    section_header(
        "Highest-Risk Customers",
        (
            "Customers ranked by production churn "
            "probability and commercial value."
        ),
    )

    highest_risk = (
        df.sort_values(
            [
                "churn_probability",
                "monthly_charges",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(30)
        .copy()
    )

    st.dataframe(
        highest_risk[
            [
                "customer_id",
                "customer_source",
                "churn_probability",
                "risk_segment",
                "retention_priority",
                "monthly_charges",
                "expected_monthly_revenue_at_risk",
                "contract",
                "tenure",
                "internet_service",
            ]
        ],
        hide_index=True,
        use_container_width=True,
        column_config={
            "churn_probability":
                st.column_config.ProgressColumn(
                    "Churn Probability",
                    min_value=0.0,
                    max_value=1.0,
                    format="percent",
                ),
            "monthly_charges":
                st.column_config.NumberColumn(
                    "Monthly Charge",
                    format="$%.2f",
                ),
            "expected_monthly_revenue_at_risk":
                st.column_config.NumberColumn(
                    "Expected Exposure",
                    format="$%.2f",
                ),
        },
    )