from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


RISK_ORDER = [
    "Low",
    "Medium",
    "High",
    "Critical",
]


def apply_chart_theme(
    figure: go.Figure,
    *,
    height: int = 390,
) -> go.Figure:
    """Apply shared enterprise Plotly styling."""

    figure.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(
            l=25,
            r=25,
            t=65,
            b=30,
        ),
        paper_bgcolor=(
            "rgba(0,0,0,0)"
        ),
        plot_bgcolor=(
            "rgba(0,0,0,0)"
        ),
        font=dict(
            family=(
                "Inter, Segoe UI, "
                "Arial, sans-serif"
            ),
            size=13,
        ),
        title=dict(
            font=dict(
                size=17,
            ),
            x=0.02,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hoverlabel=dict(
            font_size=13,
        ),
    )

    figure.update_xaxes(
        showgrid=False,
        zeroline=False,
    )

    figure.update_yaxes(
        gridcolor=(
            "rgba(148,163,184,0.10)"
        ),
        zeroline=False,
    )

    return figure


def risk_distribution_chart(
    df: pd.DataFrame,
) -> go.Figure:

    summary = (
        df[
            "risk_segment"
        ]
        .value_counts()
        .reindex(
            RISK_ORDER,
            fill_value=0,
        )
        .rename_axis(
            "risk_segment"
        )
        .reset_index(
            name="customers"
        )
    )

    figure = px.bar(
        summary,
        x="risk_segment",
        y="customers",
        category_orders={
            "risk_segment":
                RISK_ORDER,
        },
        title="Customer Risk Distribution",
        labels={
            "risk_segment":
                "Risk Segment",
            "customers":
                "Customers",
        },
    )

    figure.update_traces(
        marker_line_width=0,
    )

    return apply_chart_theme(
        figure
    )


def probability_distribution_chart(
    df: pd.DataFrame,
) -> go.Figure:

    figure = px.histogram(
        df,
        x="churn_probability",
        nbins=35,
        title=(
            "Churn Probability Distribution"
        ),
        labels={
            "churn_probability":
                "Churn Probability",
        },
    )

    figure.update_layout(
        bargap=0.04
    )

    return apply_chart_theme(
        figure
    )


def contract_risk_chart(
    df: pd.DataFrame,
) -> go.Figure:

    summary = (
        df.groupby(
            "contract",
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
        summary,
        x="contract",
        y="avg_probability",
        text_auto=True,
        title=(
            "Average Churn Risk by Contract"
        ),
        labels={
            "contract":
                "Contract",
            "avg_probability":
                "Average Churn Probability",
        },
    )

    figure.update_yaxes(
        tickformat=".0%"
    )

    return apply_chart_theme(
        figure
    )


def revenue_risk_chart(
    df: pd.DataFrame,
) -> go.Figure:

    summary = (
        df.groupby(
            "risk_segment",
            observed=True,
        )
        .agg(
            revenue_at_risk=(
                "expected_monthly_revenue_at_risk",
                "sum",
            ),
        )
        .reset_index()
    )

    figure = px.bar(
        summary,
        x="risk_segment",
        y="revenue_at_risk",
        category_orders={
            "risk_segment":
                RISK_ORDER,
        },
        title=(
            "Expected Revenue Exposure by Risk"
        ),
        labels={
            "risk_segment":
                "Risk Segment",
            "revenue_at_risk":
                "Revenue Exposure",
        },
    )

    figure.update_yaxes(
        tickprefix="$",
        separatethousands=True,
    )

    return apply_chart_theme(
        figure
    )