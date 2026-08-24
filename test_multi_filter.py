import pandas as pd
import pytest

from analysis_tools import apply_filters, filtered_sum


@pytest.fixture
def multi_filter_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Region": ["North", "North", "South", "East"],
            "Category": ["A", "B", "A", "B"],
            "Sales": [100.0, 200.0, 150.0, 300.0],
        }
    )


def test_multiple_filters_and_logic(multi_filter_df):
    filters = [
        {"column": "Region", "operator": "=", "value": "North"},
        {"column": "Category", "operator": "=", "value": "A"},
    ]
    result = apply_filters(multi_filter_df, filters)
    assert len(result) == 1
    assert result.iloc[0]["Sales"] == 100.0


def test_filtered_sum_with_multiple_filters(multi_filter_df):
    filters = [
        {"column": "Region", "operator": "=", "value": "North"},
        {"column": "Sales", "operator": ">", "value": 150.0},
    ]
    total = filtered_sum(multi_filter_df, filters, "Sales")
    assert total == 200.0