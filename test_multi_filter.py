import pandas as pd

from analysis_tools import (
    filtered_sum_multi,
    filtered_average_multi,
    filtered_count_multi,
    filtered_unique_count_multi,
)


df = pd.DataFrame({
    "City": [
        "Delhi",
        "Delhi",
        "Mumbai",
        "Delhi",
        "Mumbai",
    ],

    "Product": [
        "Laptop",
        "Phone",
        "Laptop",
        "Laptop",
        "Phone",
    ],

    "Revenue": [
        1000,
        500,
        2000,
        1500,
        800,
    ],

    "Customer_ID": [
        "C001",
        "C002",
        "C003",
        "C001",
        "C004",
    ],

    "Invoice_ID": [
        "INV001",
        "INV002",
        "INV003",
        "INV004",
        "INV005",
    ],
})


filters = [
    {
        "column": "City",
        "value": "Delhi"
    },
    {
        "column": "Product",
        "value": "Laptop"
    }
]


print(
    "SUM:",
    filtered_sum_multi(
        df,
        filters,
        "Revenue"
    )
)


print(
    "AVERAGE:",
    filtered_average_multi(
        df,
        filters,
        "Revenue"
    )
)


print(
    "COUNT:",
    filtered_count_multi(
        df,
        filters,
        "Invoice_ID"
    )
)


print(
    "UNIQUE CUSTOMERS:",
    filtered_unique_count_multi(
        df,
        filters,
        "Customer_ID"
    )
)