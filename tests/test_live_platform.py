from sqlalchemy import text

from src.database import get_engine


def test_live_customer_table_exists():
    engine = get_engine()

    with engine.connect() as connection:

        exists = connection.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'analytics'
                      AND table_name = 'live_customers'
                );
                """
            )
        ).scalar()

    assert exists is True


def test_live_prediction_table_exists():
    engine = get_engine()

    with engine.connect() as connection:

        exists = connection.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'analytics'
                      AND table_name = 'live_predictions'
                );
                """
            )
        ).scalar()

    assert exists is True


def test_historical_population_preserved():
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


def test_unified_population_consistency():
    engine = get_engine()

    with engine.connect() as connection:

        historical = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM analytics.customers;
                """
            )
        ).scalar()

        generated = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM analytics.live_customers;
                """
            )
        ).scalar()

        unified = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM analytics.vw_all_customers;
                """
            )
        ).scalar()

    assert unified == (
        int(historical or 0) + int(generated or 0)
    )


def test_platform_risk_population_consistency():
    engine = get_engine()

    with engine.connect() as connection:

        risk_count = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM analytics.vw_all_customer_risk;
                """
            )
        ).scalar()

        expected = connection.execute(
            text(
                """
                SELECT
                    (
                        SELECT COUNT(*)
                        FROM analytics.customer_predictions
                    )
                    +
                    (
                        SELECT COUNT(*)
                        FROM analytics.live_predictions
                    );
                """
            )
        ).scalar()

    assert risk_count == expected