from __future__ import annotations

import streamlit as st

from app.components.layout import (
    page_header,
)
from app.components.metrics import (
    metric_row,
)
from app.components.prediction_form import (
    render_prediction_form,
)
from src.scoring import (
    load_production_model,
)


def render() -> None:
    page_header(
        "Prediction Lab",
        (
            "Build a customer scenario and score it "
            "using the same persisted production churn "
            "pipeline used by the live platform."
        ),
        eyebrow="AI Decision Support",
    )

    package = (
        load_production_model()
    )

    metric_row(
        [
            (
                "Production Model",
                str(
                    package[
                        "model_name"
                    ]
                ).upper(),
                None,
            ),
            (
                "Decision Threshold",
                f"{float(package['threshold']):.2f}",
                None,
            ),
            (
                "Model Features",
                f"{len(package['features'])}",
                None,
            ),
        ]
    )

    st.write("")

    st.info(
        (
            "Prediction Lab scenarios are what-if "
            "analyses only. They are not automatically "
            "persisted to PostgreSQL and do not increase "
            "the production customer count."
        )
    )

    st.divider()

    render_prediction_form()