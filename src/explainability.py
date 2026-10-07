from __future__ import annotations

from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import MODEL_DIR


MODEL_PATH = MODEL_DIR / "churn_model.joblib"


@lru_cache(maxsize=1)
def load_explainability_artifact() -> dict[str, Any]:
    """
    Load and validate the persisted production churn-model artifact.

    Expected artifact structure:
        {
            "pipeline": sklearn Pipeline,
            "threshold": float,
            "model_name": str,
            "features": list[str],
        }
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Production model artifact was not found at: {MODEL_PATH}"
        )

    artifact = joblib.load(MODEL_PATH)

    if not isinstance(artifact, dict):
        raise TypeError(
            "Expected churn_model.joblib to contain a dictionary artifact."
        )

    required_keys = {
        "pipeline",
        "threshold",
        "model_name",
        "features",
    }

    missing_keys = required_keys.difference(artifact.keys())

    if missing_keys:
        raise KeyError(
            "Production model artifact is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    pipeline = artifact["pipeline"]

    if not hasattr(pipeline, "named_steps"):
        raise TypeError(
            "The persisted pipeline does not expose named_steps."
        )

    if "preprocessor" not in pipeline.named_steps:
        raise KeyError(
            "The persisted pipeline does not contain a 'preprocessor' step."
        )

    if "model" not in pipeline.named_steps:
        raise KeyError(
            "The persisted pipeline does not contain a 'model' step."
        )

    return artifact


def get_model_metadata() -> dict[str, Any]:
    """
    Return production-model metadata useful for dashboards and audit views.
    """
    artifact = load_explainability_artifact()

    return {
        "model_name": artifact["model_name"],
        "threshold": float(artifact["threshold"]),
        "raw_feature_count": len(artifact["features"]),
        "raw_features": list(artifact["features"]),
    }


def _get_pipeline_components():
    artifact = load_explainability_artifact()

    pipeline = artifact["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    return artifact, pipeline, preprocessor, model


def get_transformed_feature_names() -> list[str]:
    """
    Recover the feature names produced by the fitted ColumnTransformer.

    Names are cleaned from forms such as:
        numeric__tenure
        categorical__contract_Month-to-month

    into:
        tenure
        contract_Month-to-month
    """
    _, _, preprocessor, _ = _get_pipeline_components()

    if not hasattr(preprocessor, "get_feature_names_out"):
        raise AttributeError(
            "The fitted preprocessor does not support get_feature_names_out()."
        )

    names = preprocessor.get_feature_names_out()

    cleaned_names: list[str] = []

    for name in names:
        name = str(name)

        if "__" in name:
            name = name.split("__", 1)[1]

        cleaned_names.append(name)

    return cleaned_names


def _humanize_feature_name(feature_name: str) -> str:
    """
    Convert transformed ML feature names into presentation-friendly labels.
    """
    text = feature_name.replace("_", " ").strip()

    replacements = {
        "senior citizen": "Senior Citizen",
        "monthly charges": "Monthly Charges",
        "total charges": "Total Charges",
        "avg revenue per tenure month": "Average Revenue / Tenure Month",
        "service count": "Service Count",
        "phone service": "Phone Service",
        "multiple lines": "Multiple Lines",
        "internet service": "Internet Service",
        "online security": "Online Security",
        "online backup": "Online Backup",
        "device protection": "Device Protection",
        "tech support": "Tech Support",
        "streaming tv": "Streaming TV",
        "streaming movies": "Streaming Movies",
        "paperless billing": "Paperless Billing",
        "payment method": "Payment Method",
        "tenure group": "Tenure Group",
    }

    lowered = text.lower()

    for source, target in replacements.items():
        if lowered == source:
            return target

        prefix = f"{source} "

        if lowered.startswith(prefix):
            suffix = text[len(prefix):]
            return f"{target}: {suffix}"

    return text.title()


def get_global_feature_importance(
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Return global XGBoost feature importance.

    This represents model-level importance across the fitted model and should
    not be interpreted as causal impact.
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    _, _, _, model = _get_pipeline_components()

    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            "The production model does not expose feature_importances_."
        )

    feature_names = get_transformed_feature_names()
    importances = np.asarray(model.feature_importances_, dtype=float)

    if len(feature_names) != len(importances):
        raise ValueError(
            "Transformed feature-name count does not match model importance "
            f"count: {len(feature_names)} names vs {len(importances)} values."
        )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    importance_df["feature_label"] = importance_df["feature"].map(
        _humanize_feature_name
    )

    importance_df = (
        importance_df
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    total = importance_df["importance"].sum()

    if total > 0:
        importance_df["relative_importance_pct"] = (
            importance_df["importance"] / total * 100
        )
    else:
        importance_df["relative_importance_pct"] = 0.0

    return importance_df[
        [
            "feature",
            "feature_label",
            "importance",
            "relative_importance_pct",
        ]
    ]


def _prepare_customer_frame(
    customer: dict[str, Any] | pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert a customer record into the exact raw feature frame expected by
    the persisted production pipeline.
    """
    artifact = load_explainability_artifact()
    required_features = list(artifact["features"])

    if isinstance(customer, pd.DataFrame):
        frame = customer.copy()

    elif isinstance(customer, pd.Series):
        frame = customer.to_frame().T

    elif isinstance(customer, dict):
        frame = pd.DataFrame([customer])

    else:
        raise TypeError(
            "customer must be a dict, pandas Series, or pandas DataFrame."
        )

    if len(frame) != 1:
        raise ValueError(
            "Local explanation requires exactly one customer record."
        )

    missing_features = [
        feature
        for feature in required_features
        if feature not in frame.columns
    ]

    if missing_features:
        raise ValueError(
            "Customer record is missing production-model features: "
            f"{missing_features}"
        )

    return frame[required_features].copy()


def explain_customer(
    customer: dict[str, Any] | pd.Series | pd.DataFrame,
    top_n: int = 10,
) -> dict[str, Any]:
    """
    Explain one customer's churn prediction using native XGBoost
    contribution values.

    Positive contribution:
        pushes prediction toward churn.

    Negative contribution:
        pushes prediction toward retention.

    The returned contribution values operate in the model's raw margin
    (log-odds) space. They are not percentage-point changes in churn
    probability.
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    artifact, pipeline, preprocessor, model = _get_pipeline_components()

    frame = _prepare_customer_frame(customer)

    probability = float(
        pipeline.predict_proba(frame)[:, 1][0]
    )

    threshold = float(artifact["threshold"])
    predicted_churn = int(probability >= threshold)

    transformed = preprocessor.transform(frame)

    feature_names = get_transformed_feature_names()

    booster = model.get_booster()

    dmatrix = xgb.DMatrix(transformed)

    contributions = booster.predict(
        dmatrix,
        pred_contribs=True,
    )

    if contributions.shape[0] != 1:
        raise ValueError(
            "Expected exactly one row of local model contributions."
        )

    row = contributions[0]

    expected_length = len(feature_names) + 1

    if len(row) != expected_length:
        raise ValueError(
            "Unexpected XGBoost contribution shape. "
            f"Expected {expected_length} values, received {len(row)}."
        )

    feature_contributions = row[:-1]
    bias = float(row[-1])

    explanation_df = pd.DataFrame(
        {
            "feature": feature_names,
            "contribution": feature_contributions,
        }
    )

    explanation_df["feature_label"] = explanation_df["feature"].map(
        _humanize_feature_name
    )

    explanation_df["direction"] = np.where(
        explanation_df["contribution"] > 0,
        "Increases churn risk",
        np.where(
            explanation_df["contribution"] < 0,
            "Reduces churn risk",
            "Neutral",
        ),
    )

    explanation_df["absolute_contribution"] = (
        explanation_df["contribution"].abs()
    )

    explanation_df = (
        explanation_df
        .sort_values(
            "absolute_contribution",
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    positive_drivers = (
        explanation_df[
            explanation_df["contribution"] > 0
        ]
        .sort_values("contribution", ascending=False)
        .reset_index(drop=True)
    )

    negative_drivers = (
        explanation_df[
            explanation_df["contribution"] < 0
        ]
        .sort_values("contribution", ascending=True)
        .reset_index(drop=True)
    )

    return {
        "model_name": artifact["model_name"],
        "threshold": threshold,
        "churn_probability": probability,
        "predicted_churn": predicted_churn,
        "bias": bias,
        "top_contributions": explanation_df,
        "risk_increasing_drivers": positive_drivers,
        "risk_reducing_drivers": negative_drivers,
    }


def explain_customers(
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return production probabilities for a batch of customers.

    This helper intentionally does not calculate local contribution matrices
    for every row. It is intended for lightweight validation and monitoring.
    """
    artifact = load_explainability_artifact()

    required_features = list(artifact["features"])

    missing_features = [
        feature
        for feature in required_features
        if feature not in customers.columns
    ]

    if missing_features:
        raise ValueError(
            "Input data is missing production-model features: "
            f"{missing_features}"
        )

    pipeline = artifact["pipeline"]
    threshold = float(artifact["threshold"])

    frame = customers[required_features].copy()

    probabilities = pipeline.predict_proba(frame)[:, 1]

    result = pd.DataFrame(
        {
            "churn_probability": probabilities,
            "predicted_churn": (
                probabilities >= threshold
            ).astype(int),
        },
        index=customers.index,
    )

    return result


if __name__ == "__main__":
    metadata = get_model_metadata()
    global_importance = get_global_feature_importance(top_n=15)

    print("Production model metadata")
    print("-------------------------")
    print(f"Model: {metadata['model_name']}")
    print(f"Decision threshold: {metadata['threshold']:.2f}")
    print(f"Raw feature count: {metadata['raw_feature_count']}")

    print("\nTop global model drivers")
    print("------------------------")
    print(
        global_importance[
            [
                "feature_label",
                "importance",
            ]
        ].to_string(index=False)
    )