from src.engine_controller import (
    get_engine_state,
    get_platform_kpis,
    get_recent_logs,
)


def test_engine_state_available():

    state = get_engine_state()

    assert (
        state["engine_id"]
        == 1
    )

    assert isinstance(
        state["is_running"],
        bool,
    )

    assert (
        float(
            state[
                "generation_interval_seconds"
            ]
        )
        >= 0.5
    )


def test_platform_kpis_available():

    kpis = get_platform_kpis()

    assert (
        kpis["historical_customers"]
        == 7043
    )

    assert (
        kpis["total_customers"]
        >= 7043
    )

    assert (
        kpis["generated_customers"]
        >= 0
    )


def test_recent_logs_available():

    logs = get_recent_logs(
        limit=10
    )

    assert isinstance(
        logs,
        list,
    )

    assert len(logs) <= 10