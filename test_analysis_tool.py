import pandas as pd
import pytest

from analysis_tools import (
    apply_filters,
    dataframe_to_records,
    filtered_sum,
    group_and_sum,
    monthly_sum,
    validate_column,
    validate_dataframe,
    validate_numeric_column,
)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Fixture providing a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "Region": ["North", "South", "North", "East", "North", "South"],
            "Category": ["A", "B", "A", "B", "B", "A"],
            "Sales": [100.0, 150.0, 200.0, 50.0, 300.0, 120.0],
            "Date": [
                "2026-01-15",
                "2026-01-20",
                "2026-02-10",
                "2026-02-14",
                "2026-03-01",
                "2026-03-05",
            ],
        }
    )


class TestValidation:
    def test_validate_dataframe_empty(self):
        with pytest.raises(ValueError, match="The dataset is empty."):
            validate_dataframe(pd.DataFrame())

    def test_validate_column_missing(self, sample_dataframe):
        with pytest.raises(ValueError, match="does not exist"):
            validate_column(sample_dataframe, "NonExistentColumn")

    def test_validate_numeric_column_type(self, sample_dataframe):
        with pytest.raises(ValueError, match="must be numeric"):
            validate_numeric_column(sample_dataframe, "Region")


class TestFilterEngine:
    def test_apply_filters_equality(self, sample_dataframe):
        self._assert_filter_matches_count("=", "North", sample_dataframe, expected_count=3)

    def test_apply_filters_between(self, sample_dataframe):
        filters = [
            {"column": "Sales", "operator": "between", "value": [100, 200]}
        ]

        result = apply_filters(sample_dataframe, filters)

        assert len(result) == 4
        assert result["Sales"].tolist() == [100.0, 150.0, 200.0, 120.0]

    def test_apply_filters_contains(self, sample_dataframe):
        self._assert_filter_matches_count("contains", "ort", sample_dataframe, expected_count=3)

    def _assert_filter_matches_count(self, operator, value, sample_dataframe, expected_count):
        filters = [{"column": "Region", "operator": operator, "value": value}]
        result = apply_filters(sample_dataframe, filters)
        assert len(result) == expected_count


class TestAggregations:
    def test_filtered_sum(self, sample_dataframe):
        filters = [{"column": "Region", "operator": "=", "value": "North"}]
        result = filtered_sum(sample_dataframe, filters, "Sales")
        assert result == 600.0

    def test_filtered_sum_no_matches(self, sample_dataframe):
        filters = [{"column": "Region", "operator": "=", "value": "West"}]
        with pytest.raises(ValueError, match="No rows matched"):
            filtered_sum(sample_dataframe, filters, "Sales")

    def test_group_and_sum_identical_columns(self, sample_dataframe):
        result = group_and_sum(sample_dataframe, "Sales", "Sales")
        assert "Sum" in result.columns

    def test_monthly_sum(self, sample_dataframe):
        result = monthly_sum(sample_dataframe, "Date", "Sales")
        assert len(result) == 3
        assert result.loc[result["Month"] == "2026-01", "Sales"].iloc[0] == 250.0


class TestSerialization:
    def test_dataframe_to_records(self, sample_dataframe):
        result = dataframe_to_records(sample_dataframe.head(1))
        assert isinstance(result, list)
        assert result[0]["Region"] == "North"