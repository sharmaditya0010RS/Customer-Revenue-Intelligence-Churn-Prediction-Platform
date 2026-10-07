from sqlalchemy import text

from src.database import get_engine


def test_customer_count():

    engine = get_engine()

    with engine.connect() as connection:

        count = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM analytics.customers;
                """
            )
        ).scalar()

    assert count == 7043


def test_unique_customers():

    engine = get_engine()

    with engine.connect() as connection:

        result = connection.execute(
            text(
                """
                SELECT
                    COUNT(*),
                    COUNT(DISTINCT customer_id)
                FROM analytics.customers;
                """
            )
        ).one()

    assert result[0] == result[1]


def test_churn_flag_domain():

    engine = get_engine()

    with engine.connect() as connection:

        invalid_count = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM analytics.customers
                WHERE churn_flag NOT IN (0, 1);
                """
            )
        ).scalar()

    assert invalid_count == 0


def test_customer_360_view():

    engine = get_engine()

    with engine.connect() as connection:

        count = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM analytics.vw_customer_360;
                """
            )
        ).scalar()

    assert count == 7043