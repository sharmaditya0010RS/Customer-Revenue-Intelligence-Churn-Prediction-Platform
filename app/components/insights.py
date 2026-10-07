from __future__ import annotations

import pandas as pd


def build_executive_insights(
    df: pd.DataFrame,
) -> list[str]:
    """
    Generate deterministic executive findings from
    the current scored customer portfolio.

    These are calculated insights, not LLM-generated
    claims.
    """

    insights: list[str] = []

    if df.empty:
        return insights

    # --------------------------------------------------
    # Highest average-risk contract
    # --------------------------------------------------

    contract_risk = (
        df.groupby(
            "contract",
            observed=True,
        )["churn_probability"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    if not contract_risk.empty:
        contract = str(
            contract_risk.index[0]
        )

        probability = float(
            contract_risk.iloc[0]
        )

        insights.append(
            (
                f"{contract} customers currently carry "
                "the highest average predicted churn "
                f"risk at {probability:.1%}."
            )
        )

    # --------------------------------------------------
    # Largest probability-weighted revenue exposure
    # --------------------------------------------------

    risk_revenue = (
        df.groupby(
            "risk_segment",
            observed=True,
        )[
            "expected_monthly_revenue_at_risk"
        ]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    if not risk_revenue.empty:
        segment = str(
            risk_revenue.index[0]
        )

        exposure = float(
            risk_revenue.iloc[0]
        )

        insights.append(
            (
                f"The {segment} risk segment represents "
                "the largest probability-weighted "
                "monthly revenue exposure at "
                f"${exposure:,.0f}."
            )
        )

    # --------------------------------------------------
    # High + Critical portfolio concentration
    # --------------------------------------------------

    high_risk_mask = (
        df["risk_segment"]
        .isin(
            [
                "High",
                "Critical",
            ]
        )
    )

    high_risk_count = int(
        high_risk_mask.sum()
    )

    high_risk_rate = (
        high_risk_count
        / len(df)
    )

    insights.append(
        (
            f"{high_risk_count:,} customers "
            f"({high_risk_rate:.1%} of the portfolio) "
            "are currently classified as High or "
            "Critical churn risk."
        )
    )

    # --------------------------------------------------
    # High-value + high-risk opportunity
    # --------------------------------------------------

    value_threshold = float(
        df[
            "monthly_charges"
        ].quantile(0.75)
    )

    high_value_risk = df.loc[
        (
            df[
                "monthly_charges"
            ]
            >= value_threshold
        )
        &
        high_risk_mask
    ]

    high_value_exposure = float(
        high_value_risk[
            "expected_monthly_revenue_at_risk"
        ].sum()
    )

    insights.append(
        (
            f"{len(high_value_risk):,} customers combine "
            "top-quartile monthly value with High or "
            "Critical churn risk, representing "
            f"${high_value_exposure:,.0f} in expected "
            "monthly revenue exposure."
        )
    )

    # --------------------------------------------------
    # Highest-risk internet service
    # --------------------------------------------------

    internet_risk = (
        df.groupby(
            "internet_service",
            observed=True,
        )["churn_probability"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    if not internet_risk.empty:
        internet_service = str(
            internet_risk.index[0]
        )

        internet_probability = float(
            internet_risk.iloc[0]
        )

        insights.append(
            (
                f"{internet_service} currently has the "
                "highest average predicted churn "
                "probability among internet service "
                f"segments at {internet_probability:.1%}."
            )
        )

    return insights