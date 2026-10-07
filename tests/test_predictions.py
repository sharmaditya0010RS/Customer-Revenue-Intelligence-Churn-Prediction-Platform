from sqlalchemy import text

from src.database import get_engine


def test_prediction_count():

    engine = get_engine()

    with engine.connect() as connection:

        count = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM analytics.customer_predictions;
                """
            )
        ).scalar()

    assert count == 7043


def test_unique_prediction_customers():

    engine = get_engine()

    with engine.connect() as connection:

        result = connection.execute(
            text(
                """
                SELECT
                    COUNT(*),
                    COUNT(DISTINCT customer_id)
                FROM analytics.customer_predictions;
                """
            )
        ).one()

    assert result[0] == result[1]


def test_probability_range():

    engine = get_engine()

    with engine.connect() as connection:

        invalid_count = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM analytics.customer_predictions
                WHERE churn_probability < 0
                   OR churn_probability > 1;
                """
            )
        ).scalar()

    assert invalid_count == 0


def test_customer_risk_view():

    engine = get_engine()

    with engine.connect() as connection:

        count = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM analytics.vw_customer_risk_360;
                """
            )
        ).scalar()

    assert count == 7043