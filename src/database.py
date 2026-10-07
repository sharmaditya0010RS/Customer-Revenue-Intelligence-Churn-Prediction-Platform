from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import DATABASE_URL


def get_engine() -> Engine:
    """
    Create and return a SQLAlchemy PostgreSQL engine.
    """
    return create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
    )


def test_connection() -> bool:
    """
    Verify connectivity to PostgreSQL.
    """
    engine = get_engine()

    try:
        with engine.connect() as connection:
            result = connection.execute(
                text("SELECT version();")
            )

            version = result.scalar()

            print("Database connection successful.")
            print(f"PostgreSQL version: {version}")

        return True

    except Exception as exc:
        print("Database connection failed.")
        print(f"Error: {exc}")

        return False


if __name__ == "__main__":
    test_connection()