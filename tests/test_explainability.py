import pandas as pd

from src.explainability import (
    get_global_feature_importance,
    get_model_metadata,
    get_transformed_feature_names,
    load_explainability_artifact,
)
from src.features import build_feature_target
from src.config import PROCESSED_DATA_DIR


CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "customers_clean.csv"


def test_explainability_artifact_structure():
    artifact = load_explainability_artifact()

    assert isinstance(artifact, dict)

    required_keys = {
        "pipeline",
        "threshold",
        "model_name",
        "features",
    }

    assert required_keys.issubset(artifact.keys())


def test_model_metadata():
    metadata = get_model_metadata()

    assert metadata["model_name"] == "xgboost"
    assert metadata["threshold"] == 0.34
    assert metadata["raw_feature_count"] == 22
    assert len(metadata["raw_features"]) == 22


def test_transformed_feature_names_match_model():
    artifact = load_explainability_artifact()

    model = artifact["pipeline"].named_steps["model"]

    feature_names = get_transformed_feature_names()

    assert len(feature_names) == len(model.feature_importances_)
    assert len(feature_names) > len(artifact["features"])


def test_global_feature_importance():
    importance = get_global_feature_importance(top_n=10)

    assert isinstance(importance, pd.DataFrame)
    assert len(importance) == 10

    required_columns = {
        "feature",
        "feature_label",
        "importance",
        "relative_importance_pct",
    }

    assert required_columns.issubset(importance.columns)

    assert importance["importance"].notna().all()
    assert (importance["importance"] >= 0).all()

    assert importance["importance"].is_monotonic_decreasing


def test_global_feature_importance_top_n_validation():
    try:
        get_global_feature_importance(top_n=0)
    except ValueError:
        assert True
    else:
        assert False, "Expected ValueError when top_n < 1"


def test_production_pipeline_can_score_clean_customer():
    data = pd.read_csv(CLEAN_DATA_PATH)

    X, _, _ = build_feature_target(data)

    artifact = load_explainability_artifact()

    customer = X.iloc[[0]]

    probability = artifact["pipeline"].predict_proba(customer)[:, 1][0]

    assert 0.0 <= probability <= 1.0