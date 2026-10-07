from __future__ import annotations

from pathlib import Path

import streamlit as st


APP_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)

CSS_FILE = (
    APP_DIR
    / "assets"
    / "styles.css"
)


def load_css() -> None:
    """Load the centralized Streamlit design system."""

    if not CSS_FILE.exists():
        raise FileNotFoundError(
            f"CSS file not found: {CSS_FILE}"
        )

    css = CSS_FILE.read_text(
        encoding="utf-8"
    )

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True,
    )


def page_header(
    title: str,
    description: str,
    eyebrow: str | None = None,
) -> None:
    """
    Render page heading using native Streamlit elements.

    No HTML content rendering is used.
    """

    if eyebrow:
        st.caption(
            eyebrow.upper()
        )

    st.title(
        title
    )

    st.caption(
        description
    )


def section_header(
    title: str,
    description: str | None = None,
) -> None:

    st.subheader(
        title
    )

    if description:
        st.caption(
            description
        )


def section_divider() -> None:
    st.divider()


def empty_state(
    title: str,
    message: str,
) -> None:

    st.info(
        f"**{title}**\n\n{message}"
    )


def format_currency(
    value: float,
    decimals: int = 0,
) -> str:

    return (
        f"${float(value):,.{decimals}f}"
    )


def format_percent(
    value: float,
    decimals: int = 1,
) -> str:

    return (
        f"{float(value):.{decimals}%}"
    )


def format_number(
    value: int | float,
) -> str:

    return (
        f"{int(value):,}"
    )