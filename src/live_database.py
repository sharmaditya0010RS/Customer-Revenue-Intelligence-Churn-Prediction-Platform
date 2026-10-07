from __future__ import annotations

from pathlib import Path

from sqlalchemy import text

from src.config import BASE_DIR
from src.database import get_engine


LIVE_SCHEMA_FILE = (
    BASE_DIR
    / "database"
    / "04_live_platform.sql"
)


def execute_live_schema() -> None:
    """Create permanent live-platform database objects."""

    if not LIVE_SCHEMA_FILE.exists():
        raise FileNotFoundError(
            f"SQL file not found: {LIVE_SCHEMA_FILE}"
        )

    sql = LIVE_SCHEMA_FILE.read_text(
        encoding="utf-8"
    )

    engine = get_engine()

    with engine.begin() as connection:
        connection.execute(
            text(sql)
        )

    print(
        "Live platform database objects created."
    )


def validate_live_platform() -> None:
    """Validate tables and views required by the platform."""

    engine = get_engine()

    required_tables = {
        "live_customers",
        "live_predictions",
        "generator_state",
        "generation_log",
    }

    required_views = {
        "vw_all_customers",
        "vw_all_customer_risk",
        "vw_platform_kpis",
    }

    with engine.connect() as connection:

        tables = connection.execute(
            text(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'analytics';
                """
            )
        ).scalars().all()

        views = connection.execute(
            text(
                """
                SELECT table_name
                FROM information_schema.views
                WHERE table_schema = 'analytics';
                """
            )
        ).scalars().all()

    missing_tables = (
        required_tables - set(tables)
    )

    missing_views = (
        required_views - set(views)
    )

    if missing_tables:
        raise RuntimeError(
            "Missing live tables: "
            f"{sorted(missing_tables)}"
        )

    if missing_views:
        raise RuntimeError(
            "Missing live views: "
            f"{sorted(missing_views)}"
        )

    print(
        "Live platform validation passed."
    )


def show_platform_snapshot() -> None:
    """Display current permanent customer counts."""

    engine = get_engine()

    with engine.connect() as connection:
        row = connection.execute(
            text(
                """
                SELECT *
                FROM analytics.vw_platform_kpis;
                """
            )
        ).mappings().one()

    print("\n========== PLATFORM SNAPSHOT ==========")

    for key, value in row.items():
        print(
            f"{key}: {value}"
        )

    print(
        "=======================================\n"
    )


def main() -> None:

    execute_live_schema()
    validate_live_platform()
    show_platform_snapshot()


if __name__ == "__main__":
    main()