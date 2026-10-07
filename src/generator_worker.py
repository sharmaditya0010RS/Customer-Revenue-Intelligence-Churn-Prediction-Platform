from __future__ import annotations

import logging
import time

from sqlalchemy import text

from src.customer_generator import (
    generate_and_persist_customer,
)
from src.database import get_engine


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


def get_worker_state() -> dict:
    """Read the current persisted engine state."""

    engine = get_engine()

    with engine.connect() as connection:

        row = connection.execute(
            text(
                """
                SELECT
                    is_running,
                    generation_interval_seconds
                FROM analytics.generator_state
                WHERE engine_id = 1;
                """
            )
        ).mappings().one()

    return dict(row)


def update_heartbeat() -> None:
    """Update worker heartbeat."""

    engine = get_engine()

    with engine.begin() as connection:

        connection.execute(
            text(
                """
                UPDATE analytics.generator_state
                SET
                    heartbeat_at =
                        CURRENT_TIMESTAMP,

                    updated_at =
                        CURRENT_TIMESTAMP

                WHERE engine_id = 1
                  AND is_running = TRUE;
                """
            )
        )


def record_worker_error(
    error: Exception,
) -> None:
    """Persist worker failure information."""

    engine = get_engine()

    message = (
        f"{type(error).__name__}: "
        f"{error}"
    )

    with engine.begin() as connection:

        connection.execute(
            text(
                """
                INSERT INTO analytics.generation_log (
                    event_type,
                    message
                )
                VALUES (
                    'ENGINE_ERROR',
                    :message
                );
                """
            ),
            {
                "message":
                    message[:2000],
            },
        )


def mark_worker_stopped() -> None:
    """Ensure state is stopped when worker exits."""

    engine = get_engine()

    with engine.begin() as connection:

        connection.execute(
            text(
                """
                UPDATE analytics.generator_state
                SET
                    is_running = FALSE,

                    stopped_at =
                        COALESCE(
                            stopped_at,
                            CURRENT_TIMESTAMP
                        ),

                    updated_at =
                        CURRENT_TIMESTAMP

                WHERE engine_id = 1;
                """
            )
        )

        connection.execute(
            text(
                """
                INSERT INTO analytics.generation_log (
                    event_type,
                    message
                )
                VALUES (
                    'ENGINE_STOPPED',
                    'Generation worker stopped.'
                );
                """
            )
        )


def run_worker() -> None:
    """Run generation loop until stop is requested."""

    logger.info(
        "Generation worker started."
    )

    try:

        while True:

            state = (
                get_worker_state()
            )

            if not state[
                "is_running"
            ]:
                break

            update_heartbeat()

            generate_and_persist_customer()

            state = (
                get_worker_state()
            )

            if not state[
                "is_running"
            ]:
                break

            interval_seconds = float(
                state[
                    "generation_interval_seconds"
                ]
            )

            # Sleep in small increments so STOP remains
            # responsive even with a large generation interval.

            remaining = (
                interval_seconds
            )

            while remaining > 0:

                sleep_time = min(
                    0.25,
                    remaining,
                )

                time.sleep(
                    sleep_time
                )

                remaining -= (
                    sleep_time
                )

                state = (
                    get_worker_state()
                )

                if not state[
                    "is_running"
                ]:
                    remaining = 0
                    break

    except Exception as exc:

        logger.exception(
            "Generation worker failed."
        )

        record_worker_error(
            exc
        )

    finally:

        mark_worker_stopped()

        logger.info(
            "Generation worker exited."
        )


def main() -> None:
    run_worker()


if __name__ == "__main__":
    main()