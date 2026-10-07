from __future__ import annotations

import json
import logging

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

from src.config import (
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    METRICS_DIR,
)


DATA_FILE = PROCESSED_DATA_DIR / "customers_clean.csv"
EDA_REPORT_FILE = METRICS_DIR / "eda_report.json"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_data() -> pd.DataFrame:
    """Load validated processed customer data."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Processed dataset not found. "
            "Run python -m src.data_cleaning first."
        )

    df = pd.read_csv(DATA_FILE)

    logger.info(
        "EDA dataset loaded: %s rows x %s columns",
        df.shape[0],
        df.shape[1],
    )

    return df


def churn_summary(df: pd.DataFrame) -> dict:
    """Calculate top-level churn metrics."""

    total = len(df)
    churned = int(df["churn_flag"].sum())
    retained = total - churned

    return {
        "total_customers": total,
        "retained_customers": retained,
        "churned_customers": churned,
        "churn_rate_pct": round(
            churned / total * 100,
            2,
        ),
    }


def categorical_churn_analysis(
    df: pd.DataFrame,
    column: str,
) -> list[dict]:
    """Calculate churn metrics for a categorical feature."""

    result = (
        df.groupby(
            column,
            dropna=False,
            observed=True,
        )
        .agg(
            customers=("customer_id", "count"),
            churned_customers=("churn_flag", "sum"),
            churn_rate=("churn_flag", "mean"),
            avg_monthly_charge=("monthly_charges", "mean"),
        )
        .reset_index()
    )

    result["churn_rate_pct"] = (
        result["churn_rate"] * 100
    ).round(2)

    result["avg_monthly_charge"] = (
        result["avg_monthly_charge"].round(2)
    )

    result = result.drop(
        columns=["churn_rate"]
    )

    return result.to_dict(
        orient="records"
    )


def numerical_summary(df: pd.DataFrame) -> dict:
    """Generate descriptive statistics for numerical variables."""

    numerical_columns = [
        "tenure",
        "monthly_charges",
        "total_charges",
        "service_count",
    ]

    summary = (
        df[numerical_columns]
        .describe()
        .round(2)
    )

    return {
        column: {
            str(stat): float(value)
            for stat, value
            in summary[column].items()
        }
        for column in summary.columns
    }


def monthly_charge_statistical_test(
    df: pd.DataFrame,
) -> dict:
    """
    Compare monthly charges for churned vs retained customers.

    Mann-Whitney U is used because we do not need to assume
    normally distributed customer charges.
    """

    churned = df.loc[
        df["churn_flag"] == 1,
        "monthly_charges",
    ]

    retained = df.loc[
        df["churn_flag"] == 0,
        "monthly_charges",
    ]

    statistic, p_value = mannwhitneyu(
        churned,
        retained,
        alternative="two-sided",
    )

    return {
        "test": "Mann-Whitney U",
        "feature": "monthly_charges",
        "churned_mean": round(
            float(churned.mean()),
            2,
        ),
        "retained_mean": round(
            float(retained.mean()),
            2,
        ),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant_at_0_05": bool(
            p_value < 0.05
        ),
    }


def tenure_statistical_test(
    df: pd.DataFrame,
) -> dict:
    """Compare tenure between churned and retained customers."""

    churned = df.loc[
        df["churn_flag"] == 1,
        "tenure",
    ]

    retained = df.loc[
        df["churn_flag"] == 0,
        "tenure",
    ]

    statistic, p_value = mannwhitneyu(
        churned,
        retained,
        alternative="two-sided",
    )

    return {
        "test": "Mann-Whitney U",
        "feature": "tenure",
        "churned_mean": round(
            float(churned.mean()),
            2,
        ),
        "retained_mean": round(
            float(retained.mean()),
            2,
        ),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant_at_0_05": bool(
            p_value < 0.05
        ),
    }


def chi_square_test(
    df: pd.DataFrame,
    feature: str,
) -> dict:
    """Test association between a categorical feature and churn."""

    contingency_table = pd.crosstab(
        df[feature],
        df["churn"],
    )

    contingency_array = contingency_table.to_numpy()

    result = tuple(
        chi2_contingency(contingency_array)
    )

    chi2 = float(result[0])
    p_value = float(result[1])
    dof = int(result[2])

    n = int(contingency_array.sum())

    minimum_dimension = (
        min(contingency_array.shape) - 1
    )

    if n == 0 or minimum_dimension <= 0:
        cramers_v = 0.0
    else:
        cramers_v = (
            (chi2 / n)
            / minimum_dimension
        ) ** 0.5

    return {
        "feature": feature,
        "test": "Chi-square",
        "chi_square": chi2,
        "degrees_of_freedom": dof,
        "p_value": p_value,
        "significant_at_0_05": p_value < 0.05,
        "cramers_v": round(
            cramers_v,
            4,
        ),
    }


def correlation_analysis(
    df: pd.DataFrame,
) -> dict:
    """Calculate numerical correlation with churn."""

    columns = [
        "senior_citizen",
        "tenure",
        "monthly_charges",
        "total_charges",
        "service_count",
        "churn_flag",
    ]

    correlation = (
        df[columns]
        .corr(numeric_only=True)
        ["churn_flag"]
        .drop("churn_flag")
        .sort_values(
            key=abs,
            ascending=False,
        )
    )

    return {
        key: round(float(value), 4)
        for key, value
        in correlation.items()
    }


def create_churn_distribution(
    df: pd.DataFrame,
) -> None:

    counts = (
        df["churn"]
        .value_counts()
        .reindex(["No", "Yes"])
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    counts.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(
        "Customer Churn Distribution"
    )
    ax.set_xlabel("Churn")
    ax.set_ylabel("Customers")

    ax.tick_params(
        axis="x",
        rotation=0,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "01_churn_distribution.png",
        dpi=150,
    )

    plt.close(fig)


def create_contract_churn_chart(
    df: pd.DataFrame,
) -> None:

    analysis = (
        df.groupby(
            "contract",
            observed=True,
        )["churn_flag"]
        .mean()
        .mul(100)
        .sort_values(
            ascending=False
        )
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    analysis.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(
        "Churn Rate by Contract Type"
    )
    ax.set_xlabel("Contract")
    ax.set_ylabel("Churn Rate (%)")

    ax.tick_params(
        axis="x",
        rotation=20,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "02_contract_churn.png",
        dpi=150,
    )

    plt.close(fig)


def create_tenure_churn_chart(
    df: pd.DataFrame,
) -> None:

    analysis = (
        df.groupby(
            "tenure_group",
            observed=True,
        )["churn_flag"]
        .mean()
        .mul(100)
    )

    desired_order = [
        "0-12 Months",
        "13-24 Months",
        "25-48 Months",
        "49-60 Months",
        "61+ Months",
    ]

    analysis = analysis.reindex(
        desired_order
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    analysis.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(
        "Churn Rate by Customer Tenure"
    )
    ax.set_xlabel("Tenure Group")
    ax.set_ylabel("Churn Rate (%)")

    ax.tick_params(
        axis="x",
        rotation=25,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "03_tenure_churn.png",
        dpi=150,
    )

    plt.close(fig)


def create_monthly_charge_chart(
    df: pd.DataFrame,
) -> None:

    retained = df.loc[
        df["churn_flag"] == 0,
        "monthly_charges",
    ]

    churned = df.loc[
        df["churn_flag"] == 1,
        "monthly_charges",
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.hist(
        retained,
        bins=30,
        alpha=0.6,
        label="Retained",
    )

    ax.hist(
        churned,
        bins=30,
        alpha=0.6,
        label="Churned",
    )

    ax.set_title(
        "Monthly Charges: Churned vs Retained"
    )

    ax.set_xlabel(
        "Monthly Charges"
    )

    ax.set_ylabel(
        "Customers"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "04_monthly_charges.png",
        dpi=150,
    )

    plt.close(fig)


def create_service_churn_chart(
    df: pd.DataFrame,
) -> None:

    analysis = (
        df.groupby(
            "service_count",
            observed=True,
        )["churn_flag"]
        .mean()
        .mul(100)
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    analysis.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(
        "Churn Rate by Number of Services"
    )

    ax.set_xlabel(
        "Number of Services"
    )

    ax.set_ylabel(
        "Churn Rate (%)"
    )

    ax.tick_params(
        axis="x",
        rotation=0,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "05_service_count_churn.png",
        dpi=150,
    )

    plt.close(fig)


def create_correlation_chart(
    df: pd.DataFrame,
) -> None:

    columns = [
        "senior_citizen",
        "tenure",
        "monthly_charges",
        "total_charges",
        "service_count",
        "churn_flag",
    ]

    correlation = (
        df[columns]
        .corr(numeric_only=True)
        ["churn_flag"]
        .drop("churn_flag")
        .sort_values()
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    correlation.plot(
        kind="barh",
        ax=ax,
    )

    ax.set_title(
        "Numerical Correlation with Churn"
    )

    ax.set_xlabel(
        "Correlation"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "06_churn_correlation.png",
        dpi=150,
    )

    plt.close(fig)


def create_figures(
    df: pd.DataFrame,
) -> None:

    logger.info(
        "Generating EDA figures."
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    create_churn_distribution(df)
    create_contract_churn_chart(df)
    create_tenure_churn_chart(df)
    create_monthly_charge_chart(df)
    create_service_churn_chart(df)
    create_correlation_chart(df)

    logger.info(
        "EDA figures generated successfully."
    )


def generate_report(
    df: pd.DataFrame,
) -> dict:

    categorical_features = [
        "contract",
        "internet_service",
        "payment_method",
        "tenure_group",
        "gender",
        "partner",
        "dependents",
        "tech_support",
        "online_security",
    ]

    categorical_analysis = {
        feature: categorical_churn_analysis(
            df,
            feature,
        )
        for feature in categorical_features
    }

    chi_square_features = [
        "contract",
        "internet_service",
        "payment_method",
        "tech_support",
        "online_security",
    ]

    statistical_tests = {
        "monthly_charges":
            monthly_charge_statistical_test(df),

        "tenure":
            tenure_statistical_test(df),

        "categorical_associations": {
            feature: chi_square_test(
                df,
                feature,
            )
            for feature in chi_square_features
        },
    }

    report = {
        "churn_summary":
            churn_summary(df),

        "numerical_summary":
            numerical_summary(df),

        "correlation_with_churn":
            correlation_analysis(df),

        "categorical_churn_analysis":
            categorical_analysis,

        "statistical_tests":
            statistical_tests,
    }

    return report


def save_report(
    report: dict,
) -> None:

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        EDA_REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    logger.info(
        "EDA report saved to %s",
        EDA_REPORT_FILE,
    )


def print_key_findings(
    report: dict,
) -> None:

    logger.info(
        "========== KEY EDA FINDINGS =========="
    )

    churn = report[
        "churn_summary"
    ]

    logger.info(
        "Overall churn rate: %.2f%%",
        churn["churn_rate_pct"],
    )

    correlations = report[
        "correlation_with_churn"
    ]

    for feature, value in correlations.items():

        logger.info(
            "Correlation with churn | %s = %.4f",
            feature,
            value,
        )

    logger.info(
        "======================================"
    )


def main() -> None:

    df = load_data()

    report = generate_report(df)

    create_figures(df)

    save_report(report)

    print_key_findings(report)

    logger.info(
        "EDA pipeline completed successfully."
    )


if __name__ == "__main__":
    main()