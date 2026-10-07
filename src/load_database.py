from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import text

from src.config import (
    BASE_DIR,
    PROCESSED_DATA_DIR,
)
from src.database import get_engine


CLEAN_FILE = (
    PROCESSED_DATA_DIR
    / "customers_clean.csv"
)

SCHEMA_FILE = (
    BASE_DIR
    / "database"
    / "01_schema.sql"
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def execute_schema() -> None:
    """Create PostgreSQL schema and customer table."""

    if not SCHEMA_FILE.exists():
        raise FileNotFoundError(
            f"Schema file not found: {SCHEMA_FILE}"
        )

    sql = SCHEMA_FILE.read_text(
        encoding="utf-8"
    )

    engine = get_engine()

    logger.info("Executing PostgreSQL schema.")

    with engine.begin() as connection:
        connection.exec_driver_sql(sql)

    logger.info(
        "PostgreSQL schema created successfully."
    )


def load_customers() -> None:
    """Load processed customer data into PostgreSQL."""

    if not CLEAN_FILE.exists():
        raise FileNotFoundError(
            "Processed dataset not found. "
            "Run python -m src.data_cleaning first."
        )

    df = pd.read_csv(CLEAN_FILE)

    logger.info(
        "Loading %s customers into PostgreSQL.",
        len(df),
    )

    engine = get_engine()

    df.to_sql(
        name="customers",
        con=engine,
        schema="analytics",
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )

    logger.info(
        "Customer data loaded successfully."
    )


def validate_database_load() -> None:
    """Validate PostgreSQL ingestion."""

    engine = get_engine()

    validation_query = text(
        """
        SELECT
            COUNT(*) AS total_customers,
            COUNT(DISTINCT customer_id)
                AS unique_customers,
            SUM(churn_flag)
                AS churned_customers,
            ROUND(
                AVG(churn_flag::numeric) * 100,
                2
            ) AS churn_rate_pct,
            ROUND(
                SUM(monthly_charges),
                2
            ) AS monthly_revenue
        FROM analytics.customers;
        """
    )

    with engine.connect() as connection:
        result = (
            connection.execute(validation_query)
            .mappings()
            .one()
        )

    logger.info("Database validation results:")

    for key, value in result.items():
        logger.info("%s = %s", key, value)

    if result["total_customers"] != 7043:
        raise ValueError(
            "Unexpected number of customers "
            "loaded into PostgreSQL."
        )

    if (
        result["total_customers"]
        != result["unique_customers"]
    ):
        raise ValueError(
            "Duplicate customers detected "
            "in PostgreSQL."
        )

    logger.info(
        "Database ingestion validation passed."
    )


def main() -> None:

    execute_schema()

    load_customers()

    validate_database_load()


if __name__ == "__main__":
    main()