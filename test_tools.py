import pandas as pd

from analysis_tools import (
    calculate_sum,
    calculate_average,
    calculate_count,
    calculate_min,
    calculate_max,
    group_and_sum,
)


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
    }
)


print("SUM:")
print(calculate_sum(df, "Revenue"))

print("\nAVERAGE:")
print(calculate_average(df, "Revenue"))

print("\nCOUNT:")
print(calculate_count(df))

print("\nMIN:")
print(calculate_min(df, "Revenue"))

print("\nMAX:")
print(calculate_max(df, "Revenue"))

print("\nGROUP BY PRODUCT:")
print(
    group_and_sum(
        df,
        "Product",
        "Revenue"
    )
)