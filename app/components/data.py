from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import text

from src.database import get_engine


def load_platform_kpis() -> dict[str, Any]:
    """Load the current unified platform KPI snapshot."""

    engine = get_engine()

    query = text(
        """
        SELECT *
        FROM analytics.vw_platform_kpis;
        """
    )

    with engine.connect() as connection:
        row = (
            connection.execute(query)
            .mappings()
            .one()
        )

    return dict(row)


def load_customer_risk_data() -> pd.DataFrame:
    """Load historical + generated customer risk data."""

    engine = get_engine()

    query = """
        SELECT *
        FROM analytics.vw_all_customer_risk;
    """

    return pd.read_sql(
        query,
        engine,
    )


def load_all_customers() -> pd.DataFrame:
    """Load the unified historical + generated population."""

    engine = get_engine()

    query = """
        SELECT *
        FROM analytics.vw_all_customers;
    """

    return pd.read_sql(
        query,
        engine,
    )


def load_historical_customers() -> pd.DataFrame:
    """Load the fixed labeled historical customer population."""

    engine = get_engine()

    query = """
        SELECT *
        FROM analytics.customers;
    """

    return pd.read_sql(
        query,
        engine,
    )


def load_generated_customers() -> pd.DataFrame:
    """Load permanent generated customers with ML predictions."""

    engine = get_engine()

    query = """
        SELECT
            c.*,
            p.churn_probability,
            p.predicted_churn,
            p.risk_segment,
            p.retention_priority,
            p.expected_monthly_revenue_at_risk,
            p.model_name,
            p.decision_threshold,
            p.scored_at
        FROM analytics.live_customers AS c
        INNER JOIN analytics.live_predictions AS p
            ON c.customer_id = p.customer_id
        ORDER BY c.generated_at DESC;
    """

    return pd.read_sql(
        query,
        engine,
    )


def load_recent_generation_log(
    limit: int = 25,
) -> pd.DataFrame:
    """Load recent live-engine audit events."""

    engine = get_engine()

    safe_limit = max(
        1,
        min(int(limit), 500),
    )

    query = text(
        """
        SELECT
            log_id,
            event_type,
            customer_id,
            message,
            created_at
        FROM analytics.generation_log
        ORDER BY created_at DESC
        LIMIT :limit;
        """
    )

    with engine.connect() as connection:
        result = connection.execute(
            query,
            {
                "limit": safe_limit,
            },
        )

        rows = result.fetchall()
        columns = list(result.keys())

    return pd.DataFrame(
        rows,
        columns=columns,
    )


def load_engine_state() -> dict[str, Any]:
    """Load persistent generator control state."""

    engine = get_engine()

    query = text(
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

    with engine.connect() as connection:
        row = (
            connection.execute(query)
            .mappings()
            .one()
        )

    return dict(row)


def load_live_engine_snapshot() -> dict[str, dict[str, Any]]:
    """
    Lightweight operational snapshot for the
    auto-refreshing live monitor.
    """

    engine = get_engine()

    state_query = text(
        """
        SELECT
            is_running,
            generation_interval_seconds,
            generated_this_run,
            total_generated,
            last_customer_id,
            heartbeat_at
        FROM analytics.generator_state
        WHERE engine_id = 1;
        """
    )

    kpi_query = text(
        """
        SELECT
            total_customers,
            historical_customers,
            generated_customers,
            predicted_churn_customers,
            critical_risk_customers,
            high_critical_risk_customers,
            total_monthly_revenue,
            expected_monthly_revenue_at_risk,
            avg_churn_probability
        FROM analytics.vw_platform_kpis;
        """
    )

    with engine.connect() as connection:
        state = (
            connection.execute(state_query)
            .mappings()
            .one()
        )

        kpis = (
            connection.execute(kpi_query)
            .mappings()
            .one()
        )

    return {
        "state": dict(state),
        "kpis": dict(kpis),
    }