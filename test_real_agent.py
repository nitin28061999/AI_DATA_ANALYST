import pandas as pd

from data_profile import create_profile
from analyst import run_analysis


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


print("\nPLAN:")
print(response["plan"])

print("\nRESULT:")
print(response["result"])

print("\nAI EXPLANATION:")
print(response["explanation"])