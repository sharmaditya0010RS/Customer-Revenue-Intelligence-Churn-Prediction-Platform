from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR


CUSTOMER_FILE = (
    PROCESSED_DATA_DIR
    / "customers_clean.csv"
)

RNG = np.random.default_rng()


@lru_cache(maxsize=1)
def load_reference_data() -> pd.DataFrame:
    """Load historical data used as the generation reference."""

    if not CUSTOMER_FILE.exists():
        raise FileNotFoundError(
            "Processed customer dataset not found."
        )

    df = pd.read_csv(
        CUSTOMER_FILE
    )

    if df.empty:
        raise ValueError(
            "Reference customer dataset is empty."
        )

    return df


def sample_category(
    series: pd.Series,
) -> Any:
    """Sample a categorical value using historical frequency."""

    distribution = (
        series
        .dropna()
        .value_counts(
            normalize=True
        )
    )

    if distribution.empty:
        raise ValueError(
            f"No values available for {series.name}."
        )

    return RNG.choice(
        distribution.index.to_numpy(),
        p=distribution.to_numpy(),
    )


def sample_conditional_category(
    df: pd.DataFrame,
    target: str,
    conditions: dict[str, Any],
) -> Any:
    """Sample a category from a historical conditional subset."""

    subset = df

    for column, value in conditions.items():
        subset = subset.loc[
            subset[column] == value
        ]

    if subset.empty:
        return sample_category(
            df[target]
        )

    return sample_category(
        subset[target]
    )


def calculate_tenure_group(
    tenure: int,
) -> str:
    """Create tenure group using production feature logic."""

    if tenure <= 12:
        return "0-12 Months"

    if tenure <= 24:
        return "13-24 Months"

    if tenure <= 48:
        return "25-48 Months"

    if tenure <= 60:
        return "49-60 Months"

    return "61+ Months"


def calculate_service_count(
    customer: dict[str, Any],
) -> int:
    """Count subscribed telecom services."""

    service_columns = [
        "phone_service",
        "multiple_lines",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
    ]

    return sum(
        customer[column] == "Yes"
        for column in service_columns
    )


def generate_customer_profile() -> dict[str, Any]:
    """
    Generate one realistic synthetic customer.

    Historical distributions provide the statistical reference.
    Telecom dependencies are enforced explicitly.
    """

    df = load_reference_data()

    gender = str(
        sample_category(
            df["gender"]
        )
    )

    senior_citizen = int(
        sample_category(
            df["senior_citizen"]
        )
    )

    partner = str(
        sample_conditional_category(
            df,
            "partner",
            {
                "senior_citizen":
                    senior_citizen,
            },
        )
    )

    dependents = str(
        sample_conditional_category(
            df,
            "dependents",
            {
                "partner":
                    partner,
            },
        )
    )

    tenure = int(
        sample_category(
            df["tenure"]
        )
    )

    phone_service = str(
        sample_category(
            df["phone_service"]
        )
    )

    if phone_service == "No":
        multiple_lines = (
            "No phone service"
        )
    else:
        multiple_lines = str(
            sample_conditional_category(
                df,
                "multiple_lines",
                {
                    "phone_service":
                        "Yes",
                },
            )
        )

    internet_service = str(
        sample_category(
            df["internet_service"]
        )
    )

    internet_features = [
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
    ]

    internet_values: dict[
        str,
        str,
    ] = {}

    if internet_service == "No":

        for feature in internet_features:
            internet_values[
                feature
            ] = "No internet service"

    else:

        for feature in internet_features:

            internet_values[
                feature
            ] = str(
                sample_conditional_category(
                    df,
                    feature,
                    {
                        "internet_service":
                            internet_service,
                    },
                )
            )

    contract = str(
        sample_conditional_category(
            df,
            "contract",
            {
                "tenure_group":
                    calculate_tenure_group(
                        tenure
                    ),
            },
        )
    )

    paperless_billing = str(
        sample_category(
            df["paperless_billing"]
        )
    )

    payment_method = str(
        sample_conditional_category(
            df,
            "payment_method",
            {
                "paperless_billing":
                    paperless_billing,
            },
        )
    )

    customer: dict[str, Any] = {
        "gender":
            gender,

        "senior_citizen":
            senior_citizen,

        "partner":
            partner,

        "dependents":
            dependents,

        "tenure":
            tenure,

        "phone_service":
            phone_service,

        "multiple_lines":
            multiple_lines,

        "internet_service":
            internet_service,

        "online_security":
            internet_values[
                "online_security"
            ],

        "online_backup":
            internet_values[
                "online_backup"
            ],

        "device_protection":
            internet_values[
                "device_protection"
            ],

        "tech_support":
            internet_values[
                "tech_support"
            ],

        "streaming_tv":
            internet_values[
                "streaming_tv"
            ],

        "streaming_movies":
            internet_values[
                "streaming_movies"
            ],

        "contract":
            contract,

        "paperless_billing":
            paperless_billing,

        "payment_method":
            payment_method,
    }

    service_count = (
        calculate_service_count(
            customer
        )
    )

    # Select historically comparable customers
    # to create realistic charge distributions.

    comparable = df.loc[
        (
            df["internet_service"]
            == internet_service
        )
        & (
            df["phone_service"]
            == phone_service
        )
    ]

    if not comparable.empty:

        service_distance = (
            comparable[
                "service_count"
            ]
            .sub(
                service_count
            )
            .abs()
        )

        minimum_distance = (
            service_distance.min()
        )

        comparable = comparable.loc[
            service_distance
            == minimum_distance
        ]

    if comparable.empty:
        comparable = df

    base_monthly_charge = float(
        RNG.choice(
            comparable[
                "monthly_charges"
            ].to_numpy()
        )
    )

    charge_noise = float(
        RNG.normal(
            loc=0.0,
            scale=2.0,
        )
    )

    monthly_charges = round(
        max(
            0.0,
            base_monthly_charge
            + charge_noise,
        ),
        2,
    )

    if tenure == 0:

        total_charges = 0.0

    else:

        lifetime_multiplier = float(
            RNG.normal(
                loc=1.0,
                scale=0.05,
            )
        )

        total_charges = round(
            max(
                0.0,
                monthly_charges
                * tenure
                * lifetime_multiplier,
            ),
            2,
        )

    tenure_group = (
        calculate_tenure_group(
            tenure
        )
    )

    if tenure > 0:

        avg_revenue_per_tenure_month = (
            total_charges
            / tenure
        )

    else:

        avg_revenue_per_tenure_month = (
            monthly_charges
        )

    customer.update(
        {
            "monthly_charges":
                monthly_charges,

            "total_charges":
                total_charges,

            "tenure_group":
                tenure_group,

            "avg_revenue_per_tenure_month":
                round(
                    avg_revenue_per_tenure_month,
                    4,
                ),

            "service_count":
                service_count,
        }
    )

    validate_generated_customer(
        customer
    )

    return customer


def validate_generated_customer(
    customer: dict[str, Any],
) -> None:
    """Validate generated business rules."""

    if customer["tenure"] < 0:
        raise ValueError(
            "Tenure cannot be negative."
        )

    if customer["monthly_charges"] < 0:
        raise ValueError(
            "Monthly charges cannot be negative."
        )

    if customer["total_charges"] < 0:
        raise ValueError(
            "Total charges cannot be negative."
        )

    if (
        customer["phone_service"] == "No"
        and customer["multiple_lines"]
        != "No phone service"
    ):
        raise ValueError(
            "Invalid phone service combination."
        )

    internet_features = [
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
    ]

    if customer[
        "internet_service"
    ] == "No":

        for feature in internet_features:

            if (
                customer[feature]
                != "No internet service"
            ):
                raise ValueError(
                    "Invalid internet-service "
                    f"combination for {feature}."
                )

    if (
        customer["tenure"] == 0
        and customer["total_charges"] != 0
    ):
        raise ValueError(
            "Zero-tenure customer must have "
            "zero total charges."
        )