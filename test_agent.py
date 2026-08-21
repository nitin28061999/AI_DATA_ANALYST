import pandas as pd

from data_profile import create_profile
from ai_agent import ask_data_question # pyright: ignore[reportAttributeAccessIssue]


# ---------------------------------------------------------
# Create test dataset
# ---------------------------------------------------------

data = {
    "Product": [
        "Laptop",
        "Phone",
        "Tablet",
        "Laptop",
        "Phone",
    ],
    "Revenue": [
        50000,
        30000,
        20000,
        45000,
        35000,
    ],
    "Units": [
        10,
        20,
        15,
        8,
        25,
    ],
    "City": [
        "Delhi",
        "Mumbai",
        "Delhi",
        "Mumbai",
        "Delhi",
    ],
}


df = pd.DataFrame(data)


# ---------------------------------------------------------
# Create dataset profile
# ---------------------------------------------------------

profile = create_profile(df)


# ---------------------------------------------------------
# Ask AI
# ---------------------------------------------------------

question = "Which columns could I use to analyze revenue?"


answer = ask_data_question(
    profile,
    question
)


print("\nAI ANALYST RESPONSE:\n")
print(answer)