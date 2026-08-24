import pandas as pd

from data_profile import create_profile
from analyst import run_analysis


def test_total_revenue():
    df = pd.DataFrame(
        {
            "Product": [
                "Laptop",
                "Phone",
                "Tablet",
                "Laptop",
            ],
            "Revenue": [
                50000,
                30000,
                20000,
                40000,
            ],
            "Units": [
                10,
                20,
                15,
                8,
            ],
            "City": [
                "Delhi",
                "Mumbai",
                "Delhi",
                "Mumbai",
            ],
        }
    )

    profile = create_profile(df)

    question = "What is the total revenue?"

    response = run_analysis(
        df,
        profile,
        question
    )

    assert response["result"] == 140000.0


def test_filtered_revenue():
    df = pd.DataFrame(
        {
            "Product": [
                "Laptop",
                "Phone",
                "Tablet",
                "Laptop",
            ],
            "Revenue": [
                50000,
                30000,
                20000,
                40000,
            ],
            "Units": [
                10,
                20,
                15,
                8,
            ],
            "City": [
                "Delhi",
                "Mumbai",
                "Delhi",
                "Mumbai",
            ],
        }
    )

    profile = create_profile(df)

    question = "What is the total revenue for Delhi?"

    response = run_analysis(
        df,
        profile,
        question
    )

    assert response["result"] == 70000.0