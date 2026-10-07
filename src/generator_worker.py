from __future__ import annotations

import logging
import os
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


WORKER_MODE = os.getenv(
    "GENERATOR_WORKER_MODE",
    "local",
).strip().lower()

CLOUD_POLL_SECONDS = 1.0


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

    try:

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

    except Exception:

        logger.exception(
            "Unable to persist worker error."
        )


def record_worker_event(
    event_type: str,
    message: str,
) -> None:
    """Persist a worker lifecycle event."""

    engine = get_engine()

    with engine.begin() as connection:

        connection.execute(
            text(
                """
                INSERT INTO analytics.generation_log (
                    event_type,
                    message
                )
                VALUES (
                    :event_type,
                    :message
                );
                """
            ),
            {
                "event_type":
                    event_type,

                "message":
                    message[:2000],
            },
        )


def mark_local_worker_stopped() -> None:
    """
    Mark the local detached worker as stopped.

    Cloud workers remain alive while generation is disabled,
    so they must not force the persisted engine state to stop
    when simply waiting for another START request.
    """

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
                    'Local generation worker stopped.'
                );
                """
            )
        )


def responsive_sleep(
    seconds: float,
) -> bool:
    """
    Sleep while checking for a STOP request.

    Returns True if generation should continue.
    """

    remaining = float(
        seconds
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

        state = get_worker_state()

        if not state[
            "is_running"
        ]:
            return False

    return True


def run_local_worker() -> None:
    """
    Run the original local detached-worker lifecycle.

    The process exits after STOP.
    """

    logger.info(
        "Local generation worker started."
    )

    try:

        while True:

            state = get_worker_state()

            if not state[
                "is_running"
            ]:
                break

            update_heartbeat()

            generate_and_persist_customer()

            state = get_worker_state()

            if not state[
                "is_running"
            ]:
                break

            interval_seconds = float(
                state[
                    "generation_interval_seconds"
                ]
            )

            if not responsive_sleep(
                interval_seconds
            ):
                break

    except Exception as exc:

        logger.exception(
            "Local generation worker failed."
        )

        record_worker_error(
            exc
        )

    finally:

        mark_local_worker_stopped()

        logger.info(
            "Local generation worker exited."
        )


def run_cloud_worker() -> None:
    """
    Run a persistent cloud worker.

    The worker remains alive when generation is disabled.
    START and STOP are controlled through PostgreSQL state.
    """

    logger.info(
        "Persistent cloud generation worker started."
    )

    try:

        record_worker_event(
            "WORKER_ONLINE",
            (
                "Persistent cloud generation "
                "worker is online."
            ),
        )

    except Exception:

        logger.exception(
            "Unable to record cloud worker startup."
        )

    while True:

        try:

            state = get_worker_state()

            if not state[
                "is_running"
            ]:

                time.sleep(
                    CLOUD_POLL_SECONDS
                )

                continue

            update_heartbeat()

            result = (
                generate_and_persist_customer()
            )

            customer_id = (
                result.get(
                    "customer_id"
                )
                if isinstance(
                    result,
                    dict,
                )
                else None
            )

            logger.info(
                "Generated customer: %s",
                customer_id,
            )

            state = get_worker_state()

            if not state[
                "is_running"
            ]:
                continue

            interval_seconds = float(
                state[
                    "generation_interval_seconds"
                ]
            )

            responsive_sleep(
                interval_seconds
            )

        except KeyboardInterrupt:

            logger.info(
                "Cloud worker interrupted."
            )

            break

        except Exception as exc:

            logger.exception(
                "Cloud generation iteration failed."
            )

            record_worker_error(
                exc
            )

            time.sleep(
                2.0
            )

    logger.info(
        "Persistent cloud worker exited."
    )


def run_worker() -> None:
    """Select the appropriate worker lifecycle."""

    if WORKER_MODE == "cloud":

        run_cloud_worker()

    else:

        run_local_worker()


def main() -> None:
    run_worker()


if __name__ == "__main__":
    main()