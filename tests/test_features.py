from src.features import (
    LEAKAGE_COLUMNS,
    MODEL_FEATURES,
    build_feature_target,
    load_modeling_data,
)


def test_no_target_leakage():

    leaked = (
        set(MODEL_FEATURES)
        & set(LEAKAGE_COLUMNS)
    )

    assert not leaked


def test_feature_target_row_count():

    df = load_modeling_data()

    X, y, customer_ids = (
        build_feature_target(df)
    )

    assert len(X) == len(y)
    assert len(X) == len(customer_ids)
    assert len(X) == 7043


def test_binary_target():

    df = load_modeling_data()

    _, y, _ = build_feature_target(df)

    assert set(
        y.unique()
    ).issubset({0, 1})


def test_customer_id_not_model_feature():

    assert (
        "customer_id"
        not in MODEL_FEATURES
    )