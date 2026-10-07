from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.components.charts import (
    apply_chart_theme,
)
from app.components.layout import (
    page_header,
    section_header,
)
from app.components.metrics import (
    metric_row,
)
from src.monitoring import (
    get_data_quality_report,
    get_drift_report,
    get_model_governance,
    get_prediction_monitoring,
    load_monitoring_population,
)


def _format_feature(
    value: str,
) -> str:
    return (
        value
        .replace(
            "_",
            " ",
        )
        .title()
    )


def render() -> None:
    page_header(
        "Model & Data Monitoring",
        (
            "Production data quality, prediction "
            "health, population drift and model "
            "governance for the customer intelligence "
            "platform."
        ),
        eyebrow="Production Monitoring",
    )

    quality = (
        get_data_quality_report()
    )

    prediction = (
        get_prediction_monitoring()
    )

    governance = (
        get_model_governance()
    )

    metric_row(
        [
            (
                "Data Quality",
                (
                    f"{quality['health_pct']:.0f}%"
                ),
                (
                    f"{quality['passed_checks']}/"
                    f"{quality['total_checks']} checks"
                ),
            ),
            (
                "Unified Customers",
                f"{quality['unified_rows']:,}",
                None,
            ),
            (
                "Live Customers",
                f"{quality['live_rows']:,}",
                None,
            ),
            (
                "Scored Customers",
                f"{prediction['total_scored']:,}",
                None,
            ),
            (
                "Avg Risk",
                (
                    f"{prediction['avg_probability']:.1%}"
                ),
                None,
            ),
            (
                "High + Critical",
                f"{prediction['high_critical']:,}",
                (
                    f"{prediction['high_critical_pct']:.1%}"
                ),
            ),
        ]
    )

    st.write("")

    if (
        quality[
            "passed_checks"
        ]
        == quality[
            "total_checks"
        ]
    ):
        st.success(
            (
                "All production data-integrity checks "
                "are currently passing."
            )
        )
    else:
        st.error(
            (
                "One or more production data-integrity "
                "checks require attention."
            )
        )

    st.divider()

    section_header(
        "Data Quality Health",
        (
            "Automated integrity checks across "
            "historical, live and unified serving layers."
        ),
    )

    check_rows = []

    for name, passed in (
        quality[
            "checks"
        ].items()
    ):
        check_rows.append(
            {
                "Check":
                    name,

                "Status":
                    (
                        "PASS"
                        if passed
                        else "FAIL"
                    ),
            }
        )

    check_df = pd.DataFrame(
        check_rows
    )

    left, right = st.columns(
        [1.4, 1],
        gap="large",
    )

    with left:
        st.dataframe(
            check_df,
            hide_index=True,
            use_container_width=True,
        )

    with right:
        st.metric(
            "Historical Population",
            (
                f"{quality['historical_rows']:,}"
            ),
        )

        st.metric(
            "Historical Predictions",
            (
                f"{quality['historical_predictions']:,}"
            ),
        )

        st.metric(
            "Live Predictions",
            (
                f"{quality['live_predictions']:,}"
            ),
        )

        st.metric(
            "Cross-Layer ID Overlap",
            (
                f"{quality['historical_live_overlap']:,}"
            ),
        )

    st.caption(
        (
            "The historical labeled population remains "
            "fixed at 7,043 customers. Generated customers "
            "are stored separately and must never overlap "
            "with historical customer IDs."
        )
    )

    st.divider()

    section_header(
        "Prediction Monitoring",
        (
            "Monitor production scoring behavior "
            "without claiming unavailable live ground-truth "
            "performance."
        ),
    )

    population = (
        load_monitoring_population()
    )

    left, right = st.columns(
        2,
        gap="large",
    )

    with left:
        probability_figure = (
            px.histogram(
                population,
                x="churn_probability",
                color="customer_source",
                nbins=30,
                barmode="overlay",
                opacity=0.70,
                title=(
                    "Prediction Probability Distribution"
                ),
                labels={
                    "churn_probability":
                        "Churn Probability",
                    "customer_source":
                        "Customer Source",
                },
            )
        )

        probability_figure.update_xaxes(
            tickformat=".0%"
        )

        st.plotly_chart(
            apply_chart_theme(
                probability_figure,
                height=460,
            ),
            use_container_width=True,
        )

    with right:
        risk_distribution = (
            population.groupby(
                [
                    "customer_source",
                    "risk_segment",
                ],
                observed=True,
            )
            .size()
            .reset_index(
                name="customers"
            )
        )

        risk_figure = px.bar(
            risk_distribution,
            x="risk_segment",
            y="customers",
            color="customer_source",
            barmode="group",
            category_orders={
                "risk_segment": [
                    "Low",
                    "Medium",
                    "High",
                    "Critical",
                ]
            },
            title=(
                "Risk Segment Distribution"
            ),
            labels={
                "risk_segment":
                    "Risk Segment",
                "customers":
                    "Customers",
                "customer_source":
                    "Customer Source",
            },
        )

        st.plotly_chart(
            apply_chart_theme(
                risk_figure,
                height=460,
            ),
            use_container_width=True,
        )

    st.info(
        (
            "Generated customers do not yet have future "
            "ground-truth churn outcomes. Therefore this "
            "section monitors prediction behavior and "
            "population stability—not live accuracy, "
            "precision, recall or F1."
        )
    )

    st.divider()

    section_header(
        "Population Drift",
        (
            "Compare generated customers with the "
            "fixed historical baseline using Population "
            "Stability Index (PSI)."
        ),
    )

    drift = (
        get_drift_report()
    )

    if drift.empty:
        st.info(
            (
                "Generate live customers to activate "
                "population-drift monitoring."
            )
        )

    else:
        drift_display = (
            drift.copy()
        )

        drift_display[
            "feature"
        ] = (
            drift_display[
                "feature"
            ]
            .map(
                _format_feature
            )
        )

        drift_display.columns = [
            "Feature",
            "Historical Mean",
            "Live Mean",
            "PSI",
            "Status",
        ]

        left, right = st.columns(
            [1.4, 1],
            gap="large",
        )

        with left:
            st.dataframe(
                drift_display,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Historical Mean":
                        st.column_config.NumberColumn(
                            "Historical Mean",
                            format="%.3f",
                        ),
                    "Live Mean":
                        st.column_config.NumberColumn(
                            "Live Mean",
                            format="%.3f",
                        ),
                    "PSI":
                        st.column_config.NumberColumn(
                            "PSI",
                            format="%.4f",
                        ),
                },
            )

        with right:
            st.markdown(
                "#### PSI Interpretation"
            )

            st.write(
                "**< 0.10** — Stable"
            )

            st.write(
                "**0.10 – 0.25** — Moderate shift"
            )

            st.write(
                "**> 0.25** — Significant shift"
            )

            significant = int(
                (
                    drift[
                        "status"
                    ]
                    == "Significant Shift"
                ).sum()
            )

            moderate = int(
                (
                    drift[
                        "status"
                    ]
                    == "Moderate Shift"
                ).sum()
            )

            stable = int(
                (
                    drift[
                        "status"
                    ]
                    == "Stable"
                ).sum()
            )

            st.metric(
                "Stable Features",
                stable,
            )

            st.metric(
                "Moderate Shift",
                moderate,
            )

            st.metric(
                "Significant Shift",
                significant,
            )

        psi_chart = (
            drift
            .dropna(
                subset=[
                    "psi"
                ]
            )
            .copy()
        )

        if not psi_chart.empty:
            psi_chart[
                "feature"
            ] = (
                psi_chart[
                    "feature"
                ]
                .map(
                    _format_feature
                )
            )

            figure = px.bar(
                psi_chart,
                x="feature",
                y="psi",
                color="status",
                title=(
                    "Population Stability Index"
                ),
                labels={
                    "feature":
                        "Feature",
                    "psi":
                        "PSI",
                    "status":
                        "Drift Status",
                },
            )

            figure.add_hline(
                y=0.10,
                line_dash="dash",
                annotation_text=(
                    "Moderate threshold"
                ),
            )

            figure.add_hline(
                y=0.25,
                line_dash="dash",
                annotation_text=(
                    "Significant threshold"
                ),
            )

            st.plotly_chart(
                apply_chart_theme(
                    figure,
                    height=460,
                ),
                use_container_width=True,
            )

    st.caption(
        (
            "PSI is a distribution-shift indicator. "
            "A drift alert does not automatically mean "
            "that model performance has degraded."
        )
    )

    st.divider()

    section_header(
        "Model Governance",
        (
            "Persisted production-model identity "
            "and scoring contract."
        ),
    )

    governance_columns = (
        st.columns(
            4,
            gap="medium",
        )
    )

    governance_columns[0].metric(
        "Artifact",
        (
            "AVAILABLE"
            if governance[
                "artifact_available"
            ]
            else "MISSING"
        ),
    )

    governance_columns[1].metric(
        "Model",
        str(
            governance[
                "model_name"
            ]
        ).upper(),
    )

    governance_columns[2].metric(
        "Threshold",
        (
            f"{float(governance['threshold']):.2f}"
        ),
    )

    governance_columns[3].metric(
        "Raw Features",
        governance[
            "feature_count"
        ],
    )

    st.caption(
        (
            "Monitoring timestamp · "
            f"{governance['monitored_at']:%Y-%m-%d %H:%M:%S}"
        )
    )

    with st.expander(
        "Model governance notes"
    ):
        st.markdown(
            """
- Production estimator: **XGBoost**
- Decision threshold: **0.34**
- Raw production feature contract: **22 features**
- Historical labels remain isolated from generated customers.
- Manual Prediction Lab scenarios are not persisted.
- Live generated customers are scored immediately but do not have future ground-truth churn labels.
- Global and local model explanations describe predictive behavior, not causal effects.
- Population drift is monitored independently from model performance.
"""
        )