from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.components.charts import apply_chart_theme
from src.explainability import explain_customer
from src.scoring import score_customer
from src.synthetic_data import (
    calculate_service_count,
    calculate_tenure_group,
)


def build_manual_customer(
    *,
    gender: str,
    senior_citizen: int,
    partner: str,
    dependents: str,
    tenure: int,
    phone_service: str,
    multiple_lines: str,
    internet_service: str,
    online_security: str,
    online_backup: str,
    device_protection: str,
    tech_support: str,
    streaming_tv: str,
    streaming_movies: str,
    contract: str,
    paperless_billing: str,
    payment_method: str,
    monthly_charges: float,
    total_charges: float,
) -> dict:
    """Build production model input from raw form attributes."""

    tenure_group = calculate_tenure_group(
        tenure
    )

    service_input = {
        "phone_service": phone_service,
        "multiple_lines": multiple_lines,
        "internet_service": internet_service,
        "online_security": online_security,
        "online_backup": online_backup,
        "device_protection": device_protection,
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,
    }

    service_count = calculate_service_count(
        service_input
    )

    if tenure > 0:
        avg_revenue_per_tenure_month = (
            total_charges / tenure
        )
    else:
        avg_revenue_per_tenure_month = (
            monthly_charges
        )

    return {
        "gender": gender,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "tenure": tenure,
        "phone_service": phone_service,
        "multiple_lines": multiple_lines,
        "internet_service": internet_service,
        "online_security": online_security,
        "online_backup": online_backup,
        "device_protection": device_protection,
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,
        "contract": contract,
        "paperless_billing": paperless_billing,
        "payment_method": payment_method,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "tenure_group": tenure_group,
        "avg_revenue_per_tenure_month":
            avg_revenue_per_tenure_month,
        "service_count": service_count,
    }


def _render_local_explanation(
    customer: dict,
) -> None:
    """
    Render customer-level XGBoost contribution analysis.

    Positive contribution values push the model toward churn.
    Negative contribution values push the model toward retention.
    """

    st.divider()

    st.subheader(
        "Why This Prediction?"
    )

    st.caption(
        (
            "Customer-level explanation from the same "
            "persisted XGBoost model used for production "
            "scoring."
        )
    )

    try:
        explanation = explain_customer(
            customer,
            top_n=10,
        )

    except Exception as exc:
        st.warning(
            (
                "Customer-level explanation could not "
                "be generated. The churn prediction "
                "above remains valid."
            )
        )

        with st.expander(
            "Explainability diagnostic"
        ):
            st.code(
                str(exc)
            )

        return

    contributions = (
        explanation[
            "top_contributions"
        ]
        .copy()
    )

    if contributions.empty:
        st.info(
            (
                "No non-zero model contributions were "
                "available for this customer."
            )
        )
        return

    chart_data = (
        contributions
        .sort_values(
            "contribution",
            ascending=True,
        )
        .copy()
    )

    figure = px.bar(
        chart_data,
        x="contribution",
        y="feature_label",
        orientation="h",
        color="direction",
        title=(
            "Top Customer-Level Model Drivers"
        ),
        labels={
            "contribution":
                "Contribution to Model Score",
            "feature_label":
                "Model Feature",
            "direction":
                "Direction",
        },
        category_orders={
            "direction": [
                "Increases churn risk",
                "Reduces churn risk",
                "Neutral",
            ]
        },
    )

    figure.add_vline(
        x=0,
        line_width=1,
        line_dash="dash",
    )

    figure.update_layout(
        yaxis_title=None,
        legend_title_text=None,
    )

    st.plotly_chart(
        apply_chart_theme(
            figure,
            height=500,
        ),
        use_container_width=True,
    )

    increasing = (
        explanation[
            "risk_increasing_drivers"
        ]
        .copy()
    )

    reducing = (
        explanation[
            "risk_reducing_drivers"
        ]
        .copy()
    )

    left, right = st.columns(
        2,
        gap="large",
    )

    with left:
        st.markdown(
            "#### Risk-Increasing Drivers"
        )

        if increasing.empty:
            st.success(
                (
                    "No material positive churn "
                    "contributions appear among the "
                    "top model drivers."
                )
            )
        else:
            for _, row in (
                increasing.head(5).iterrows()
            ):
                st.write(
                    (
                        f"**{row['feature_label']}**  \n"
                        f"Contribution: "
                        f"`+{float(row['contribution']):.4f}`"
                    )
                )

    with right:
        st.markdown(
            "#### Risk-Reducing Drivers"
        )

        if reducing.empty:
            st.warning(
                (
                    "No material negative churn "
                    "contributions appear among the "
                    "top model drivers."
                )
            )
        else:
            for _, row in (
                reducing.head(5).iterrows()
            ):
                st.write(
                    (
                        f"**{row['feature_label']}**  \n"
                        f"Contribution: "
                        f"`{float(row['contribution']):.4f}`"
                    )
                )

    st.info(
        (
            "Interpretation: positive contribution values "
            "push the model toward churn, while negative "
            "values push it toward retention. These values "
            "operate in the XGBoost model's raw margin "
            "(log-odds) space; they are not percentage-point "
            "changes in churn probability."
        )
    )

    st.caption(
        (
            "Model explanations describe predictive behavior, "
            "not causality. A feature contributing to a high "
            "risk score does not prove that changing that "
            "feature would prevent churn."
        )
    )

    with st.expander(
        "View explanation details",
        expanded=False,
    ):
        explanation_table = (
            contributions[
                [
                    "feature_label",
                    "contribution",
                    "direction",
                    "absolute_contribution",
                ]
            ]
            .copy()
        )

        explanation_table.columns = [
            "Feature",
            "Contribution",
            "Direction",
            "Absolute Contribution",
        ]

        st.dataframe(
            explanation_table,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Contribution":
                    st.column_config.NumberColumn(
                        "Contribution",
                        format="%.4f",
                    ),
                "Absolute Contribution":
                    st.column_config.NumberColumn(
                        "Absolute Contribution",
                        format="%.4f",
                    ),
            },
        )


def render_prediction_form() -> None:
    """
    Render manual prediction form.

    Manual scenarios are scored only and are not
    persisted to PostgreSQL.
    """

    st.subheader(
        "Customer Profile"
    )

    st.caption(
        (
            "Enter raw customer attributes. "
            "Engineered model features are calculated "
            "automatically before scoring."
        )
    )

    with st.form(
        "manual_prediction_form"
    ):
        identity_1, identity_2 = st.columns(
            2,
            gap="large",
        )

        with identity_1:
            gender = st.selectbox(
                "Gender",
                [
                    "Female",
                    "Male",
                ],
            )

            senior_citizen_label = st.selectbox(
                "Senior Citizen",
                [
                    "No",
                    "Yes",
                ],
            )

            partner = st.selectbox(
                "Partner",
                [
                    "No",
                    "Yes",
                ],
            )

            dependents = st.selectbox(
                "Dependents",
                [
                    "No",
                    "Yes",
                ],
            )

        with identity_2:
            tenure = st.slider(
                "Tenure (Months)",
                min_value=0,
                max_value=72,
                value=24,
                step=1,
            )

            contract = st.selectbox(
                "Contract",
                [
                    "Month-to-month",
                    "One year",
                    "Two year",
                ],
            )

            paperless_billing = st.selectbox(
                "Paperless Billing",
                [
                    "Yes",
                    "No",
                ],
            )

            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        st.divider()

        st.subheader(
            "Services"
        )

        service_1, service_2 = st.columns(
            2,
            gap="large",
        )

        with service_1:
            phone_service = st.selectbox(
                "Phone Service",
                [
                    "Yes",
                    "No",
                ],
            )

            if phone_service == "Yes":
                multiple_lines = st.selectbox(
                    "Multiple Lines",
                    [
                        "No",
                        "Yes",
                    ],
                )
            else:
                multiple_lines = (
                    "No phone service"
                )

            internet_service = st.selectbox(
                "Internet Service",
                [
                    "Fiber optic",
                    "DSL",
                    "No",
                ],
            )

            online_security = "No internet service"
            online_backup = "No internet service"
            device_protection = "No internet service"
            tech_support = "No internet service"
            streaming_tv = "No internet service"
            streaming_movies = "No internet service"

            if internet_service == "No":
                st.info(
                    (
                        "Internet-dependent services "
                        "are automatically set to "
                        "'No internet service'."
                    )
                )

            else:
                online_security = st.selectbox(
                    "Online Security",
                    [
                        "No",
                        "Yes",
                    ],
                )

                online_backup = st.selectbox(
                    "Online Backup",
                    [
                        "No",
                        "Yes",
                    ],
                )

                device_protection = st.selectbox(
                    "Device Protection",
                    [
                        "No",
                        "Yes",
                    ],
                )

        with service_2:
            if internet_service != "No":
                tech_support = st.selectbox(
                    "Tech Support",
                    [
                        "No",
                        "Yes",
                    ],
                )

                streaming_tv = st.selectbox(
                    "Streaming TV",
                    [
                        "No",
                        "Yes",
                    ],
                )

                streaming_movies = st.selectbox(
                    "Streaming Movies",
                    [
                        "No",
                        "Yes",
                    ],
                )

        st.divider()

        st.subheader(
            "Commercial Profile"
        )

        commercial_1, commercial_2 = st.columns(
            2,
            gap="large",
        )

        with commercial_1:
            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                max_value=250.0,
                value=70.0,
                step=1.0,
                format="%.2f",
            )

        with commercial_2:
            default_total = (
                0.0
                if tenure == 0
                else float(
                    monthly_charges
                    * tenure
                )
            )

            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                max_value=20000.0,
                value=default_total,
                step=10.0,
                format="%.2f",
            )

        submitted = st.form_submit_button(
            "Run Churn Prediction",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return

    if (
        tenure == 0
        and total_charges != 0
    ):
        st.error(
            (
                "A zero-tenure customer must have "
                "Total Charges equal to 0."
            )
        )
        return

    senior_citizen = (
        1
        if senior_citizen_label == "Yes"
        else 0
    )

    customer = build_manual_customer(
        gender=gender,
        senior_citizen=senior_citizen,
        partner=partner,
        dependents=dependents,
        tenure=tenure,
        phone_service=phone_service,
        multiple_lines=multiple_lines,
        internet_service=internet_service,
        online_security=online_security,
        online_backup=online_backup,
        device_protection=device_protection,
        tech_support=tech_support,
        streaming_tv=streaming_tv,
        streaming_movies=streaming_movies,
        contract=contract,
        paperless_billing=paperless_billing,
        payment_method=payment_method,
        monthly_charges=float(
            monthly_charges
        ),
        total_charges=float(
            total_charges
        ),
    )

    try:
        prediction = score_customer(
            customer
        )

    except Exception as exc:
        st.error(
            "Unable to score this customer."
        )
        st.caption(
            str(exc)
        )
        return

    probability = float(
        prediction[
            "churn_probability"
        ]
    )

    predicted_churn = bool(
        prediction[
            "predicted_churn"
        ]
    )

    risk_segment = str(
        prediction[
            "risk_segment"
        ]
    )

    priority = str(
        prediction[
            "retention_priority"
        ]
    )

    exposure = float(
        prediction[
            "expected_monthly_revenue_at_risk"
        ]
    )

    threshold = float(
        prediction[
            "decision_threshold"
        ]
    )

    st.divider()

    st.subheader(
        "Customer Risk Assessment"
    )

    st.caption(
        (
            "Production churn model decision "
            "for the submitted customer profile."
        )
    )

    result_columns = st.columns(
        5,
        gap="medium",
    )

    result_columns[0].metric(
        "Churn Probability",
        f"{probability:.1%}",
    )

    result_columns[1].metric(
        "Model Decision",
        (
            "AT RISK"
            if predicted_churn
            else "RETAIN"
        ),
    )

    result_columns[2].metric(
        "Risk Segment",
        risk_segment,
    )

    result_columns[3].metric(
        "Retention Priority",
        priority,
    )

    result_columns[4].metric(
        "Monthly Exposure",
        f"${exposure:,.2f}",
    )

    st.write("")

    st.progress(
        probability,
        text=(
            "Predicted churn probability · "
            f"{probability:.1%}"
        ),
    )

    st.write("")

    if risk_segment == "Critical":
        st.error(
            (
                "**Immediate retention attention recommended.** "
                "This customer falls within the Critical "
                "production risk segment."
            )
        )

    elif risk_segment == "High":
        st.warning(
            (
                "**Targeted retention intervention recommended.** "
                "The production model identifies elevated "
                "churn propensity."
            )
        )

    elif risk_segment == "Medium":
        st.info(
            (
                "**Monitor customer behavior.** "
                "Risk is elevated but remains below the "
                "platform's High-risk boundary."
            )
        )

    else:
        st.success(
            (
                "**Low predicted churn risk.** "
                "No immediate retention intervention is "
                "indicated by the production model."
            )
        )

    commercial, model = st.columns(
        2,
        gap="large",
    )

    with commercial:
        st.subheader(
            "Commercial Impact"
        )

        st.metric(
            "Current Monthly Value",
            f"${float(monthly_charges):,.2f}",
        )

        st.metric(
            "Probability-Weighted Exposure",
            f"${exposure:,.2f}",
        )

    with model:
        st.subheader(
            "Model Decision"
        )

        st.metric(
            "Production Threshold",
            f"{threshold:.2f}",
        )

        st.metric(
            "Customer Probability",
            f"{probability:.2f}",
        )

    st.caption(
        (
            "Churn probability is a predictive risk score. "
            "It does not prove that a customer will churn "
            "and should not be interpreted as a causal "
            "effect of any individual attribute."
        )
    )

    _render_local_explanation(
        customer
    )

    with st.expander(
        "View engineered model input"
    ):
        st.dataframe(
            pd.DataFrame(
                [customer]
            ),
            hide_index=True,
            use_container_width=True,
        )