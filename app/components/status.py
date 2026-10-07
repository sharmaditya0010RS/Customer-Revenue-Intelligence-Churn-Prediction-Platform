from __future__ import annotations

from datetime import datetime
from typing import Any

import streamlit as st


def engine_status(
    state: dict[str, Any],
) -> None:

    if state["is_running"]:

        st.success(
            "● Data generation engine running"
        )

    else:

        st.warning(
            "● Data generation engine stopped"
        )


def database_status(
    connected: bool = True,
) -> None:

    if connected:
        st.success(
            "● PostgreSQL connected"
        )
    else:
        st.error(
            "● PostgreSQL unavailable"
        )


def model_status(
    model_name: str,
    threshold: float,
) -> None:

    st.success(
        (
            "● Production ML model ready\n\n"
            f"{model_name.upper()} · "
            f"Threshold {threshold:.2f}"
        )
    )


def powerbi_status() -> None:

    st.info(
        (
            "● Power BI Import Mode\n\n"
            "Refresh required for new PostgreSQL data."
        )
    )


def format_timestamp(
    value: datetime | None,
) -> str:

    if value is None:
        return "—"

    return value.strftime(
        "%d %b %Y · %H:%M:%S"
    )