from __future__ import annotations

import streamlit as st

from app.components.charts import (
    revenue_risk_chart,
    risk_distribution_chart,
)
from app.components.data import (
    load_customer_risk_data,
    load_engine_state,
    load_generated_customers,
    load_platform_kpis,
    load_recent_generation_log,
)
from app.components.layout import (
    page_header,
    section_header,
)
from app.components.metrics import (
    platform_kpi_row,
)
from app.components.status import (
    engine_status,
    format_timestamp,
)
from src.engine_controller import (
    generate_one_customer,
)


def render() -> None:

    page_header(
        "Command Center",
        (
            "Enterprise overview of customer growth, "
            "revenue exposure, churn intelligence and "
            "live platform operations."
        ),
        eyebrow="Executive Control Plane",
    )

    kpis = load_platform_kpis()
    state = load_engine_state()
    risk = load_customer_risk_data()

    platform_kpi_row(
        kpis
    )

    st.write("")

    left, right = st.columns(
        [1.65, 1],
        gap="large",
    )

    with left:

        section_header(
            "Portfolio Risk",
            (
                "ML risk segmentation across historical "
                "and generated customers."
            ),
        )

        st.plotly_chart(
            risk_distribution_chart(
                risk
            ),
            use_container_width=True,
        )

    with right:

        section_header(
            "Operational Status",
            (
                "Current health of the live "
                "customer intelligence platform."
            ),
        )

        engine_status(
            state
        )

        st.success(
            "● PostgreSQL analytics database connected"
        )

        st.success(
            "● Production churn model available"
        )

        st.info(
            "● Power BI operating in Import mode"
        )

        st.metric(
            "Generated This Run",
            int(
                state[
                    "generated_this_run"
                ]
            ),
        )

        st.caption(
            (
                "Last generated customer · "
                f"{state['last_customer_id'] or '—'}"
            )
        )

        st.caption(
            (
                "Worker heartbeat · "
                f"{format_timestamp(state['heartbeat_at'])}"
            )
        )

    left, right = st.columns(
        2,
        gap="large",
    )

    with left:

        section_header(
            "Revenue Exposure",
            (
                "Probability-weighted monthly "
                "revenue exposure by ML risk band."
            ),
        )

        st.plotly_chart(
            revenue_risk_chart(
                risk
            ),
            use_container_width=True,
        )

    with right:

        section_header(
            "Quick Action",
            (
                "Generate and score one permanent "
                "customer without starting the stream."
            ),
        )

        generated = (
            load_generated_customers()
        )

        st.metric(
            "Permanent Generated Customers",
            len(generated),
        )

        if st.button(
            "Generate One Customer",
            type="primary",
            disabled=bool(
                state["is_running"]
            ),
            use_container_width=True,
        ):

            with st.spinner(
                "Generating, scoring and persisting customer..."
            ):

                result = (
                    generate_one_customer()
                )

            st.success(
                (
                    f"{result['customer_id']} created · "
                    f"{result['risk_segment']} risk · "
                    f"{result['churn_probability']:.1%} "
                    "churn probability"
                )
            )

            st.rerun()

        if state["is_running"]:

            st.caption(
                "Generate One is disabled while "
                "the continuous engine is running."
            )

    st.divider()

    section_header(
        "Recent Platform Activity",
        (
            "Latest persistent events from the "
            "customer generation engine."
        ),
    )

    logs = (
        load_recent_generation_log(
            limit=12
        )
    )

    if logs.empty:

        st.info(
            "No generation activity available."
        )

    else:

        st.dataframe(
            logs[
                [
                    "created_at",
                    "customer_id",
                    "event_type",
                    "message",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )