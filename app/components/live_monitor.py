from __future__ import annotations

import streamlit as st

from app.components.data import (
    load_live_engine_snapshot,
)
from app.components.layout import (
    format_currency,
)
from app.components.status import (
    format_timestamp,
)


@st.fragment(run_every="2s")
def render_live_monitor() -> None:
    """
    Auto-refresh only the operational monitor.

    This avoids rerunning the complete Streamlit page
    every two seconds.
    """

    try:
        snapshot = (
            load_live_engine_snapshot()
        )

    except Exception as exc:
        st.error(
            "Unable to retrieve the live platform snapshot."
        )

        st.caption(
            f"Database error: {exc}"
        )

        return

    state = snapshot["state"]
    kpis = snapshot["kpis"]

    if state["is_running"]:
        st.success(
            "● LIVE CUSTOMER STREAM ACTIVE"
        )
    else:
        st.warning(
            "● CUSTOMER STREAM STOPPED"
        )

    columns = st.columns(
        5,
        gap="medium",
    )

    columns[0].metric(
        "Total Customers",
        f"{int(kpis['total_customers']):,}",
    )

    columns[1].metric(
        "Generated",
        f"{int(kpis['generated_customers']):,}",
    )

    columns[2].metric(
        "Generated This Run",
        f"{int(state['generated_this_run']):,}",
    )

    columns[3].metric(
        "Monthly Revenue",
        format_currency(
            float(
                kpis[
                    "total_monthly_revenue"
                ]
            )
        ),
    )

    columns[4].metric(
        "Revenue at Risk",
        format_currency(
            float(
                kpis[
                    "expected_monthly_revenue_at_risk"
                ]
            )
        ),
    )

    heartbeat = format_timestamp(
        state["heartbeat_at"]
    )

    last_customer = (
        state["last_customer_id"]
        or "—"
    )

    interval = float(
        state[
            "generation_interval_seconds"
        ]
    )

    st.caption(
        (
            f"Latest customer · {last_customer}"
            f"   |   Worker heartbeat · {heartbeat}"
            f"   |   Generation interval · {interval:.1f}s"
            "   |   Monitor refresh · 2s"
        )
    )