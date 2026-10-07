from src.features import MODEL_FEATURES
from src.scoring import score_customer
from src.synthetic_data import (
    generate_customer_profile,
)


def test_generated_customer_has_model_features():

    customer = (
        generate_customer_profile()
    )

    missing = [
        feature
        for feature in MODEL_FEATURES
        if feature not in customer
    ]

    assert missing == []


def test_phone_service_business_rule():

    for _ in range(25):

        customer = (
            generate_customer_profile()
        )

        if (
            customer[
                "phone_service"
            ]
            == "No"
        ):
            assert (
                customer[
                    "multiple_lines"
                ]
                == "No phone service"
            )


def test_internet_service_business_rule():

    internet_features = [
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
    ]

    for _ in range(25):

        customer = (
            generate_customer_profile()
        )

        if (
            customer[
                "internet_service"
            ]
            == "No"
        ):

            for feature in (
                internet_features
            ):
                assert (
                    customer[feature]
                    == "No internet service"
                )


def test_generated_customer_can_be_scored():

    customer = (
        generate_customer_profile()
    )

    prediction = (
        score_customer(
            customer
        )
    )

    assert (
        0
        <= prediction[
            "churn_probability"
        ]
        <= 1
    )

    assert prediction[
        "risk_segment"
    ] in {
        "Low",
        "Medium",
        "High",
        "Critical",
    }


def test_zero_tenure_total_charges():

    for _ in range(50):

        customer = (
            generate_customer_profile()
        )

        if customer["tenure"] == 0:
            assert (
                customer[
                    "total_charges"
                ]
                == 0.0
            )