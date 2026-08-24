import pandas as pd
import pytest
from data_profile import create_profile
from analyst import validate_plan


def test_validate_total_revenue_plan():
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet", "Laptop"],
        "Revenue": [50000, 30000, 20000, 40000],
        "Units": [10, 20, 15, 8],
        "City": ["Delhi", "Mumbai", "Delhi", "Mumbai"],
    })
    profile = create_profile(df)

    plan = {
        "operation": "calculate_sum",
        "column": "Revenue",
        "group_column": None,
        "value_column": None,
        "count_column": None,
        "date_column": None,
        "n": None,
        "filters": None,
    }

    validated_plan = validate_plan(plan, profile)

    assert validated_plan["operation"] == "calculate_sum"
    assert validated_plan["column"] == "Revenue"


def test_validate_filtered_revenue_plan():
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet", "Laptop"],
        "Revenue": [50000, 30000, 20000, 40000],
        "Units": [10, 20, 15, 8],
        "City": ["Delhi", "Mumbai", "Delhi", "Mumbai"],
    })
    profile = create_profile(df)

    plan = {
        "operation": "filtered_sum",
        "column": None,
        "group_column": None,
        "value_column": "Revenue",
        "count_column": None,
        "date_column": None,
        "n": None,
        "filters": [
            {
                "column": "City",
                "operator": "=",
                "value": "Delhi",
            }
        ],
    }

    validated_plan = validate_plan(plan, profile)

    assert validated_plan["operation"] == "filtered_sum"
    assert validated_plan["value_column"] == "Revenue"
    assert validated_plan["filters"] == [
        {
            "column": "City",
            "operator": "=",
            "value": "Delhi",
        }
    ]


def test_validate_rejects_invalid_column():
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet", "Laptop"],
        "Revenue": [50000, 30000, 20000, 40000],
        "Units": [10, 20, 15, 8],
        "City": ["Delhi", "Mumbai", "Delhi", "Mumbai"],
    })
    profile = create_profile(df)

    plan = {
        "operation": "calculate_sum",
        "column": "NotARealColumn",
        "group_column": None,
        "value_column": None,
        "count_column": None,
        "date_column": None,
        "n": None,
        "filters": None,
    }

    with pytest.raises((ValueError, TypeError)):
        validate_plan(plan, profile)


def test_validate_rejects_invalid_filter_column():
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet", "Laptop"],
        "Revenue": [50000, 30000, 20000, 40000],
        "Units": [10, 20, 15, 8],
        "City": ["Delhi", "Mumbai", "Delhi", "Mumbai"],
    })
    profile = create_profile(df)

    plan = {
        "operation": "filtered_sum",
        "column": None,
        "group_column": None,
        "value_column": "Revenue",
        "count_column": None,
        "date_column": None,
        "n": None,
        "filters": [
            {
                "column": "NotARealColumn",
                "operator": "=",
                "value": "Delhi",
            }
        ],
    }

    with pytest.raises((ValueError, TypeError)):
        validate_plan(plan, profile)