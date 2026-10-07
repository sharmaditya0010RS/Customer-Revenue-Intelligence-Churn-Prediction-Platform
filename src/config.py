from pathlib import Path
import os

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR / ".env")


# --------------------------------------------------
# Project directories
# --------------------------------------------------

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"

MODEL_DIR = BASE_DIR / "models"

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"


# --------------------------------------------------
# Runtime environment
# --------------------------------------------------

APP_ENV = os.getenv(
    "APP_ENV",
    "local",
).strip().lower()

IS_CLOUD = (
    APP_ENV == "cloud"
)


# --------------------------------------------------
# Database configuration
# --------------------------------------------------

# Managed cloud platforms normally provide a complete
# DATABASE_URL. Local development continues to use the
# individual DB_* variables from .env.

ENV_DATABASE_URL = os.getenv(
    "DATABASE_URL"
)


if ENV_DATABASE_URL:

    DATABASE_URL = (
        ENV_DATABASE_URL
    )

else:

    DB_HOST = os.getenv(
        "DB_HOST",
        "localhost",
    )

    DB_PORT = os.getenv(
        "DB_PORT",
        "5432",
    )

    DB_NAME = os.getenv(
        "DB_NAME",
        "customer_intelligence",
    )

    DB_USER = os.getenv(
        "DB_USER",
        "postgres",
    )

    DB_PASSWORD = os.getenv(
        "DB_PASSWORD"
    )

    if not DB_PASSWORD:
        raise ValueError(
            "Database credentials are not configured. "
            "Set DATABASE_URL for cloud deployment or "
            "create a local .env file using .env.example."
        )

    DATABASE_URL = (
        f"postgresql+psycopg2://"
        f"{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20


# --------------------------------------------------
# Create generated directories if required
# --------------------------------------------------

for directory in [
    PROCESSED_DATA_DIR,
    PREDICTIONS_DIR,
    MODEL_DIR,
    FIGURES_DIR,
    METRICS_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )