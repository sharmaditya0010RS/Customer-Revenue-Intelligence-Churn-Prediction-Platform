from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text

from src.database import get_engine
from src.scoring import score_customer
from src.synthetic_data import (
    generate_customer_profile,
)


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    __name__
)


def get_next_customer_id(
    connection,
) -> str:
    """
    Generate the next permanent synthetic customer ID.

    PostgreSQL advisory locking prevents two generator
    processes from allocating the same ID concurrently.
    """

    connection.execute(
        text(
            """
            SELECT pg_advisory_xact_lock(
                70430001
            );
            """
        )
    )

    maximum_id = connection.execute(
        text(
            """
            SELECT COALESCE(
                MAX(
                    CAST(
                        SUBSTRING(
                            customer_id
                            FROM 5
                        )
                        AS BIGINT
                    )
                ),
                0
            )
            FROM analytics.live_customers
            WHERE customer_id
                ~ '^SYN-[0-9]+$';
            """
        )
    ).scalar_one()

    next_number = (
        int(maximum_id) + 1
    )

    return (
        f"SYN-{next_number:08d}"
    )


def insert_customer(
    connection,
    customer_id: str,
    customer: dict[str, Any],
) -> None:
    """Insert permanent generated customer."""

    connection.execute(
        text(
            """
            INSERT INTO analytics.live_customers (
                customer_id,
                gender,
                senior_citizen,
                partner,
                dependents,
                tenure,
                phone_service,
                multiple_lines,
                internet_service,
                online_security,
                online_backup,
                device_protection,
                tech_support,
                streaming_tv,
                streaming_movies,
                contract,
                paperless_billing,
                payment_method,
                monthly_charges,
                total_charges,
                tenure_group,
                avg_revenue_per_tenure_month,
                service_count,
                source
            )
            VALUES (
                :customer_id,
                :gender,
                :senior_citizen,
                :partner,
                :dependents,
                :tenure,
                :phone_service,
                :multiple_lines,
                :internet_service,
                :online_security,
                :online_backup,
                :device_protection,
                :tech_support,
                :streaming_tv,
                :streaming_movies,
                :contract,
                :paperless_billing,
                :payment_method,
                :monthly_charges,
                :total_charges,
                :tenure_group,
                :avg_revenue_per_tenure_month,
                :service_count,
                'synthetic_engine'
            );
            """
        ),
        {
            "customer_id":
                customer_id,

            **customer,
        },
    )


def insert_prediction(
    connection,
    customer_id: str,
    prediction: dict[str, Any],
) -> None:
    """Persist prediction for generated customer."""

    connection.execute(
        text(
            """
            INSERT INTO analytics.live_predictions (
                customer_id,
                churn_probability,
                predicted_churn,
                risk_segment,
                retention_priority,
                expected_monthly_revenue_at_risk,
                model_name,
                decision_threshold
            )
            VALUES (
                :customer_id,
                :churn_probability,
                :predicted_churn,
                :risk_segment,
                :retention_priority,
                :expected_monthly_revenue_at_risk,
                :model_name,
                :decision_threshold
            );
            """
        ),
        {
            "customer_id":
                customer_id,

            **prediction,
        },
    )


def insert_generation_log(
    connection,
    customer_id: str,
    prediction: dict[str, Any],
) -> None:
    """Create permanent generation audit entry."""

    message = (
        f"Generated {customer_id} | "
        f"Risk={prediction['risk_segment']} | "
        f"Probability="
        f"{prediction['churn_probability']:.2%}"
    )

    connection.execute(
        text(
            """
            INSERT INTO analytics.generation_log (
                customer_id,
                event_type,
                message
            )
            VALUES (
                :customer_id,
                'CUSTOMER_GENERATED',
                :message
            );
            """
        ),
        {
            "customer_id":
                customer_id,

            "message":
                message,
        },
    )


def update_generator_counters(
    connection,
    customer_id: str,
) -> None:
    """Update permanent engine counters."""

    connection.execute(
        text(
            """
            UPDATE analytics.generator_state
            SET
                generated_this_run =
                    generated_this_run + 1,

                total_generated =
                    total_generated + 1,

                last_customer_id =
                    :customer_id,

                heartbeat_at =
                    CURRENT_TIMESTAMP,

                updated_at =
                    CURRENT_TIMESTAMP

            WHERE engine_id = 1;
            """
        ),
        {
            "customer_id":
                customer_id,
        },
    )


def generate_and_persist_customer() -> dict:
    """
    Generate, score and permanently persist one customer.
    """

    customer = (
        generate_customer_profile()
    )

    prediction = (
        score_customer(
            customer
        )
    )

    engine = get_engine()

    with engine.begin() as connection:

        customer_id = (
            get_next_customer_id(
                connection
            )
        )

        insert_customer(
            connection,
            customer_id,
            customer,
        )

        insert_prediction(
            connection,
            customer_id,
            prediction,
        )

        insert_generation_log(
            connection,
            customer_id,
            prediction,
        )

        update_generator_counters(
            connection,
            customer_id,
        )

    result = {
        "customer_id":
            customer_id,

        **customer,

        **prediction,
    }

    logger.info(
        "Generated %s | risk=%s | probability=%.2f%%",
        customer_id,
        prediction["risk_segment"],
        prediction[
            "churn_probability"
        ] * 100,
    )

    return result


def main() -> None:
    """Generate one permanent customer for testing."""

    result = (
        generate_and_persist_customer()
    )

    print(
        "\n========== GENERATED CUSTOMER =========="
    )

    print(
        f"Customer ID: "
        f"{result['customer_id']}"
    )

    print(
        f"Contract: "
        f"{result['contract']}"
    )

    print(
        f"Internet Service: "
        f"{result['internet_service']}"
    )

    print(
        f"Monthly Charges: "
        f"{result['monthly_charges']:.2f}"
    )

    print(
        f"Churn Probability: "
        f"{result['churn_probability']:.2%}"
    )

    print(
        f"Risk Segment: "
        f"{result['risk_segment']}"
    )

    print(
        f"Retention Priority: "
        f"{result['retention_priority']}"
    )

    print(
        "========================================\n"
    )


if __name__ == "__main__":
    main()