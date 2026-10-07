from __future__ import annotations

from typing import Any

import streamlit as st

from app.components.status import (
    engine_status,
)


NAVIGATION = {
    "Overview": [
        "Command Center",
        "Executive Overview",
    ],

    "Analytics": [
        "Churn Analytics",
        "Revenue Intelligence",
        "ML Risk Intelligence",
        "Retention Center",
    ],

    "AI & Operations": [
        "Prediction Lab",
        "Live Data Engine",
        "Model & Data Monitoring",
    ],

    "Business Intelligence": [
        "Power BI Hub",
    ],
}


def render_sidebar(
    engine_state: dict[str, Any],
    model_name: str,
    threshold: float,
) -> str:

    with st.sidebar:

        st.title(
            "Customer IQ"
        )

        st.caption(
            "Revenue Intelligence Platform"
        )

        st.divider()

        all_pages = [
            page
            for pages in NAVIGATION.values()
            for page in pages
        ]

        page = st.radio(
            "Workspace",
            all_pages,
            label_visibility="collapsed",
        )

        st.divider()

        st.caption(
            "PLATFORM STATUS"
        )

        engine_status(
            engine_state
        )

        st.caption(
            f"Model · {model_name.upper()}"
        )

        st.caption(
            f"Decision threshold · {threshold:.2f}"
        )

        st.caption(
            "PostgreSQL · Connected"
        )

        st.caption(
            "Power BI · Import Mode"
        )

        st.divider()

        if st.button(
            "Refresh Data",
            use_container_width=True,
        ):
            st.rerun()

        st.caption(
            "Customer Revenue Intelligence"
        )

    return page