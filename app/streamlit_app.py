from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(BASE_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(BASE_DIR),
    )


from app.components.data import (  # noqa: E402
    load_engine_state,
)
from app.components.layout import (  # noqa: E402
    load_css,
)
from app.components.navigation import (  # noqa: E402
    render_sidebar,
)

from app.views import (  # noqa: E402
    churn_analytics,
    command_center,
    data_engine,
    executive,
    ml_risk,
    monitoring,
    powerbi_hub,
    prediction_lab,
    retention_center,
    revenue_intelligence,
)

from src.scoring import (  # noqa: E402
    load_production_model,
)


st.set_page_config(
    page_title=(
        "Customer Revenue Intelligence"
    ),
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


PAGE_RENDERERS = {
    "Command Center":
        command_center.render,

    "Executive Overview":
        executive.render,

    "Churn Analytics":
        churn_analytics.render,

    "Revenue Intelligence":
        revenue_intelligence.render,

    "ML Risk Intelligence":
        ml_risk.render,

    "Retention Center":
        retention_center.render,

    "Prediction Lab":
        prediction_lab.render,

    "Live Data Engine":
        data_engine.render,

    "Model & Data Monitoring":
        monitoring.render,

    "Power BI Hub":
        powerbi_hub.render,
}


def main() -> None:

    load_css()

    package = (
        load_production_model()
    )

    engine_state = (
        load_engine_state()
    )

    page = render_sidebar(
        engine_state=engine_state,
        model_name=str(
            package[
                "model_name"
            ]
        ),
        threshold=float(
            package[
                "threshold"
            ]
        ),
    )

    renderer = (
        PAGE_RENDERERS[
            page
        ]
    )

    renderer()


if __name__ == "__main__":
    main()