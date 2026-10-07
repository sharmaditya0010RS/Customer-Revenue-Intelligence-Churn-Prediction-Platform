from __future__ import annotations

import plotly.express as px
import streamlit as st

from app.components.charts import (
    apply_chart_theme,
)
from app.components.data import (
    load_engine_state,
    load_generated_customers,
    load_recent_generation_log,
)
from app.components.layout import (
    page_header,
    section_header,
)
from app.components.live_monitor import (
    render_live_monitor,
)
from app.components.metrics import (
    metric_row,
)
from app.components.status import (
    format_timestamp,
)
from src.engine_controller import (
    generate_one_customer,
    start_engine,
    stop_engine,
)


def render() -> None:
    page_header(
        "Live Data Engine",
        (
            "Control the permanent synthetic customer "
            "stream and monitor production ML scoring "
            "in near real time."
        ),
        eyebrow="AI & Operations",
    )

    render_live_monitor()

    st.divider()

    state = (
        load_engine_state()
    )

    metric_row(
        [
            (
                "Engine Status",
                (
                    "RUNNING"
                    if state["is_running"]
                    else "STOPPED"
                ),
                None,
            ),
            (
                "Generated This Run",
                f"{int(state['generated_this_run']):,}",
                None,
            ),
            (
                "Total Generated",
                f"{int(state['total_generated']):,}",
                None,
            ),
            (
                "Generation Interval",
                (
                    f"{float(state['generation_interval_seconds']):.1f}s"
                ),
                None,
            ),
        ]
    )

    st.write("")

    controls, health = (
        st.columns(
            [1.15, 1],
            gap="large",
        )
    )

    with controls:
        section_header(
            "Engine Controls",
            (
                "Start or stop the continuous "
                "production customer generator."
            ),
        )

        interval = st.slider(
            "Customer generation interval",
            min_value=0.5,
            max_value=10.0,
            value=float(
                state[
                    "generation_interval_seconds"
                ]
            ),
            step=0.5,
            disabled=bool(
                state["is_running"]
            ),
            help=(
                "Delay between permanently generated "
                "customer records."
            ),
        )

        start_column, stop_column = (
            st.columns(2)
        )

        with start_column:
            if st.button(
                "Start Engine",
                type="primary",
                disabled=bool(
                    state["is_running"]
                ),
                use_container_width=True,
            ):
                try:
                    started = (
                        start_engine(
                            interval
                        )
                    )

                    if started:
                        st.success(
                            "Generation engine started."
                        )
                    else:
                        st.warning(
                            (
                                "Generation engine is "
                                "already running."
                            )
                        )

                    st.rerun()

                except Exception as exc:
                    st.error(
                        "Unable to start the generation engine."
                    )

                    st.caption(
                        str(exc)
                    )

        with stop_column:
            if st.button(
                "Stop Engine",
                disabled=not bool(
                    state["is_running"]
                ),
                use_container_width=True,
            ):
                try:
                    stop_engine()

                    st.success(
                        "Stop request submitted."
                    )

                    st.rerun()

                except Exception as exc:
                    st.error(
                        "Unable to stop the generation engine."
                    )

                    st.caption(
                        str(exc)
                    )

        if st.button(
            "Generate One Customer",
            disabled=bool(
                state["is_running"]
            ),
            use_container_width=True,
        ):
            try:
                with st.spinner(
                    (
                        "Generating, scoring and "
                        "persisting customer..."
                    )
                ):
                    result = (
                        generate_one_customer()
                    )

                st.success(
                    (
                        f"{result['customer_id']} generated "
                        "and permanently stored."
                    )
                )

                st.rerun()

            except Exception as exc:
                st.error(
                    "Unable to generate customer."
                )

                st.caption(
                    str(exc)
                )

        if state["is_running"]:
            st.info(
                (
                    "The background worker is active. "
                    "The live monitor above refreshes "
                    "automatically every two seconds."
                )
            )

    with health:
        section_header(
            "Worker Health",
            (
                "Persistent operational state stored "
                "inside PostgreSQL."
            ),
        )

        st.metric(
            "Last Customer",
            (
                state[
                    "last_customer_id"
                ]
                or "—"
            ),
        )

        st.metric(
            "Heartbeat",
            format_timestamp(
                state[
                    "heartbeat_at"
                ]
            ),
        )

        st.metric(
            "Started",
            format_timestamp(
                state[
                    "started_at"
                ]
            ),
        )

        st.metric(
            "Stopped",
            format_timestamp(
                state[
                    "stopped_at"
                ]
            ),
        )

    st.divider()

    section_header(
        "Generated Customer Stream",
        (
            "Permanent customers created by the "
            "production simulation layer."
        ),
    )

    generated = (
        load_generated_customers()
    )

    if generated.empty:
        st.info(
            "No generated customers are available yet."
        )

    else:
        recent = (
            generated
            .sort_values(
                "generated_at"
            )
            .tail(100)
            .copy()
        )

        left, right = (
            st.columns(
                2,
                gap="large",
            )
        )

        with left:
            figure = px.scatter(
                recent,
                x="generated_at",
                y="churn_probability",
                color="risk_segment",
                size="monthly_charges",
                hover_name="customer_id",
                hover_data=[
                    "contract",
                    "internet_service",
                    "monthly_charges",
                    "retention_priority",
                ],
                title=(
                    "Incoming Customer Risk Stream"
                ),
                labels={
                    "generated_at":
                        "Generated At",
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
                    figure
                ),
                use_container_width=True,
            )

        with right:
            priority = (
                generated.groupby(
                    "retention_priority",
                    observed=True,
                )
                .agg(
                    customers=(
                        "customer_id",
                        "count",
                    ),
                    revenue_exposure=(
                        "expected_monthly_revenue_at_risk",
                        "sum",
                    ),
                )
                .reset_index()
                .sort_values(
                    "revenue_exposure",
                    ascending=False,
                )
            )

            figure = px.bar(
                priority,
                x="retention_priority",
                y="revenue_exposure",
                text_auto=True,
                title=(
                    "Generated Revenue Exposure"
                ),
                labels={
                    "retention_priority":
                        "Retention Priority",
                    "revenue_exposure":
                        "Expected Revenue Exposure",
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

        st.dataframe(
            recent[
                [
                    "customer_id",
                    "generated_at",
                    "contract",
                    "internet_service",
                    "monthly_charges",
                    "churn_probability",
                    "risk_segment",
                    "retention_priority",
                ]
            ]
            .sort_values(
                "generated_at",
                ascending=False,
            ),
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
            },
        )

    st.divider()

    section_header(
        "Engine Audit Trail",
        (
            "Latest operational events recorded "
            "by the generation platform."
        ),
    )

    logs = (
        load_recent_generation_log(
            limit=30
        )
    )

    if logs.empty:
        st.info(
            "No engine audit events are available."
        )

    else:
        st.dataframe(
            logs,
            hide_index=True,
            use_container_width=True,
        )