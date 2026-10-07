from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import text

from src.config import BASE_DIR
from src.customer_generator import (
    generate_and_persist_customer,
)
from src.database import get_engine


WORKER_MODULE = "src.generator_worker"

MINIMUM_INTERVAL_SECONDS = 0.5


def get_engine_state() -> dict[str, Any]:
    """Return the persisted generation-engine state."""

    engine = get_engine()

    with engine.connect() as connection:

        row = connection.execute(
            text(
                """
                SELECT
                    engine_id,
                    is_running,
                    generation_interval_seconds,
                    generated_this_run,
                    total_generated,
                    last_customer_id,
                    started_at,
                    stopped_at,
                    heartbeat_at,
                    updated_at
                FROM analytics.generator_state
                WHERE engine_id = 1;
                """
            )
        ).mappings().one()

    return dict(row)


def set_generation_interval(
    seconds: float,
) -> None:
    """Persist the requested generation interval."""

    seconds = float(seconds)

    if seconds < MINIMUM_INTERVAL_SECONDS:
        raise ValueError(
            "Generation interval must be at least "
            f"{MINIMUM_INTERVAL_SECONDS} seconds."
        )

    engine = get_engine()

    with engine.begin() as connection:

        connection.execute(
            text(
                """
                UPDATE analytics.generator_state
                SET
                    generation_interval_seconds =
                        :seconds,

                    updated_at =
                        CURRENT_TIMESTAMP
                WHERE engine_id = 1;
                """
            ),
            {
                "seconds":
                    seconds,
            },
        )


def mark_engine_started(
    interval_seconds: float,
) -> bool:
    """
    Atomically transition the engine from stopped to running.

    Returns False when another worker is already marked running.
    """

    interval_seconds = float(
        interval_seconds
    )

    if (
        interval_seconds
        < MINIMUM_INTERVAL_SECONDS
    ):
        raise ValueError(
            "Generation interval must be at least "
            f"{MINIMUM_INTERVAL_SECONDS} seconds."
        )

    engine = get_engine()

    with engine.begin() as connection:

        updated = connection.execute(
            text(
                """
                UPDATE analytics.generator_state
                SET
                    is_running = TRUE,

                    generation_interval_seconds =
                        :interval_seconds,

                    generated_this_run = 0,

                    started_at =
                        CURRENT_TIMESTAMP,

                    stopped_at = NULL,

                    heartbeat_at =
                        CURRENT_TIMESTAMP,

                    updated_at =
                        CURRENT_TIMESTAMP

                WHERE engine_id = 1
                  AND is_running = FALSE;
                """
            ),
            {
                "interval_seconds":
                    interval_seconds,
            },
        )

        started = (
            updated.rowcount == 1
        )

        if started:

            connection.execute(
                text(
                    """
                    INSERT INTO analytics.generation_log (
                        event_type,
                        message
                    )
                    VALUES (
                        'ENGINE_START',
                        :message
                    );
                    """
                ),
                {
                    "message":
                        (
                            "Generation engine started "
                            f"with interval "
                            f"{interval_seconds:.2f} seconds."
                        ),
                },
            )

    return started


def request_engine_stop() -> bool:
    """
    Request a graceful stop.

    The worker will observe is_running=False and exit.
    """

    engine = get_engine()

    with engine.begin() as connection:

        updated = connection.execute(
            text(
                """
                UPDATE analytics.generator_state
                SET
                    is_running = FALSE,

                    stopped_at =
                        CURRENT_TIMESTAMP,

                    updated_at =
                        CURRENT_TIMESTAMP

                WHERE engine_id = 1
                  AND is_running = TRUE;
                """
            )
        )

        stopped = (
            updated.rowcount == 1
        )

        if stopped:

            connection.execute(
                text(
                    """
                    INSERT INTO analytics.generation_log (
                        event_type,
                        message
                    )
                    VALUES (
                        'ENGINE_STOP_REQUEST',
                        'Generation engine stop requested.'
                    );
                    """
                )
            )

    return stopped


def reset_stale_engine_state() -> None:
    """
    Recover from an interrupted worker.

    This is useful if the computer or worker process was
    terminated while is_running remained TRUE.
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
                        CURRENT_TIMESTAMP,

                    updated_at =
                        CURRENT_TIMESTAMP

                WHERE engine_id = 1
                  AND is_running = TRUE
                  AND (
                        heartbeat_at IS NULL
                        OR heartbeat_at
                           < CURRENT_TIMESTAMP
                             - INTERVAL '30 seconds'
                  );
                """
            )
        )


def launch_worker_process() -> None:
    """
    Launch the generator worker as a detached process.

    stdout/stderr are detached so the Streamlit process
    does not block waiting for the worker.
    """

    creation_flags = 0

    if sys.platform == "win32":

        creation_flags = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.DETACHED_PROCESS
        )

    subprocess.Popen(
        [
            sys.executable,
            "-m",
            WORKER_MODULE,
        ],
        cwd=str(
            Path(BASE_DIR)
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation_flags,
        close_fds=True,
    )


def start_engine(
    interval_seconds: float = 2.0,
) -> bool:
    """Start the continuous generation engine."""

    reset_stale_engine_state()

    started = mark_engine_started(
        interval_seconds
    )

    if not started:
        return False

    try:

        launch_worker_process()

    except Exception:

        request_engine_stop()
        raise

    return True


def stop_engine() -> bool:
    """Request graceful worker shutdown."""

    return request_engine_stop()


def generate_one_customer() -> dict:
    """Generate one permanent customer manually."""

    return (
        generate_and_persist_customer()
    )


def get_recent_logs(
    limit: int = 25,
) -> list[dict[str, Any]]:
    """Return recent engine audit events."""

    limit = max(
        1,
        min(
            int(limit),
            200,
        ),
    )

    engine = get_engine()

    with engine.connect() as connection:

        rows = connection.execute(
            text(
                """
                SELECT
                    log_id,
                    customer_id,
                    event_type,
                    message,
                    created_at
                FROM analytics.generation_log
                ORDER BY log_id DESC
                LIMIT :limit;
                """
            ),
            {
                "limit":
                    limit,
            },
        ).mappings().all()

    return [
        dict(row)
        for row in rows
    ]


def get_platform_kpis() -> dict[str, Any]:
    """Return current unified platform KPIs."""

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

    return dict(row)