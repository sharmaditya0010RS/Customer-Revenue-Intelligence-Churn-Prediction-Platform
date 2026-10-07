from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import text

from src.config import BASE_DIR
from src.database import get_engine


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


VIEWS_FILE = (
    BASE_DIR
    / "database"
    / "03_views.sql"
)


def execute_sql_file(path: Path) -> None:

    if not path.exists():
        raise FileNotFoundError(
            f"SQL file not found: {path}"
        )

    sql = path.read_text(
        encoding="utf-8"
    )

    engine = get_engine()

    logger.info(
        "Executing SQL file: %s",
        path.name,
    )

    with engine.begin() as connection:
        connection.exec_driver_sql(sql)

    logger.info(
        "%s executed successfully.",
        path.name,
    )


def validate_views() -> None:

    query = text(
        """
        SELECT
            table_name
        FROM information_schema.views
        WHERE table_schema = 'analytics'
        ORDER BY table_name;
        """
    )

    engine = get_engine()

    with engine.connect() as connection:

        views = connection.execute(
            query
        ).scalars().all()

    logger.info(
        "Analytics views available:"
    )

    for view in views:
        logger.info(" - %s", view)

    required_views = {
        "vw_executive_kpis",
        "vw_churn_analysis",
        "vw_revenue_analysis",
        "vw_customer_segments",
        "vw_customer_360",
        "vw_customer_risk_360",
        "vw_retention_command_center",
    }

    missing = required_views - set(views)

    if missing:
        raise ValueError(
            f"Missing analytics views: {missing}"
        )

    logger.info(
        "All required views validated."
    )


def show_executive_kpis() -> None:

    query = text(
        """
        SELECT *
        FROM analytics.vw_executive_kpis;
        """
    )

    engine = get_engine()

    with engine.connect() as connection:

        result = (
            connection
            .execute(query)
            .mappings()
            .one()
        )

    logger.info(
        "Executive KPI snapshot:"
    )

    for key, value in result.items():
        logger.info(
            "%s = %s",
            key,
            value,
        )


def main() -> None:

    execute_sql_file(
        VIEWS_FILE
    )

    validate_views()

    show_executive_kpis()


if __name__ == "__main__":
    main()